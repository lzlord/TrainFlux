import numpy as np
import pytest
import torch
import torch.nn.functional as F

from types import SimpleNamespace
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from config import GlobalConfig, ModelConfig, TokenSuperpositionConfig
from engine.dataloaders.pretraining_dataloader import PretrainingDataLoader
from engine.token_superposition import SuperpositionSchedule
from engine.trainer import Trainer
from models.implementations.tendril import TendrilTransformer
from tasks.pretraining import next_bag_cross_entropy


def test_default_and_fixed_schedule():
    assert SuperpositionSchedule(TokenSuperpositionConfig()).bag_size == 1
    schedule = SuperpositionSchedule(TokenSuperpositionConfig(mode='fixed', switch_step=2))
    assert schedule.prepare_step(1) is None
    assert schedule.bag_size == 2
    assert 'configured switch step' in schedule.prepare_step(2)
    assert schedule.bag_size == 1
    assert schedule.prepare_step(3) is None


def test_adaptive_validation_plateau_resume_and_fallback():
    config = TokenSuperpositionConfig(
        mode='adaptive', min_superposition_steps=4, max_superposition_steps=10,
        patience=2, min_delta=0.1,
    )
    schedule = SuperpositionSchedule(config)
    schedule.observe_validation(1, 5.0)
    schedule.observe_validation(2, 5.0)
    assert schedule.state.stale_validations == 0
    schedule.observe_validation(4, 4.96)
    schedule.observe_validation(5, 4.85)
    assert schedule.state.stale_validations == 0
    schedule.observe_validation(6, 4.84)
    resumed = SuperpositionSchedule(config)
    resumed.load_state_dict(schedule.state_dict())
    assert 'validation loss plateau' in resumed.observe_validation(7, 4.82)
    assert resumed.state.recovery_step == 7
    schedule.load_state_dict(resumed.state_dict(), reset_validation_history=True)
    assert schedule.bag_size == 1 and schedule.state.best_loss is None
    fallback = SuperpositionSchedule(config)
    assert 'maximum superposition steps' in fallback.prepare_step(10)
    with pytest.raises(ValueError, match='finite validation loss'):
        SuperpositionSchedule(config).observe_validation(1, float('nan'))


def test_mce_matches_mean_ce_and_gradient_with_duplicate_targets():
    logits = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(11, (2, 3, 2))
    targets[0, 0] = 3
    expected = torch.stack([
        F.cross_entropy(logits.flatten(0, 1), ids.flatten())
        for ids in targets.unbind(-1)
    ]).mean()
    actual = next_bag_cross_entropy(logits, targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def make_loader(tmp_path, rank):
    return PretrainingDataLoader(2, 2, rank == 0, rank, 2, tmp_path, 'train')


def test_causal_bags_rank_windows_and_resume_into_recovery(tmp_path):
    train = tmp_path / 'train'
    train.mkdir()
    np.save(train / 'data_0.npy', np.arange(40))
    for rank in range(2):
        loader = make_loader(tmp_path, rank)
        x, y, mask = loader.next_batch(bag_size=2)
        start = 8 * rank
        torch.testing.assert_close(x.flatten(), torch.arange(start, start + 8))
        torch.testing.assert_close(y.flatten(), torch.arange(start + 2, start + 10))
        assert mask is None
        resumed = make_loader(tmp_path, rank)
        resumed.load_state_dict(loader.state_dict())
        x, y, _ = resumed.next_batch()
        start = 16 + 4 * rank
        torch.testing.assert_close(x.flatten(), torch.arange(start, start + 4))
        torch.testing.assert_close(y.flatten(), torch.arange(start + 1, start + 5))


def test_tendril_bag_embedding_is_mean_and_does_not_leak_future():
    config = ModelConfig(dim=32, n_layers=1, n_heads=4, n_kv_heads=2, multiple_of=32, max_seq_len=4)
    model = TendrilTransformer(config=config, pad_token_id=0, vocab_size=16, ignore_index=-100)
    ids = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
    captured = []
    handle = model.layers[0].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    original = model(ids)['logits']
    handle.remove()
    torch.testing.assert_close(captured[0], model.tok_embeddings(ids).mean(dim=2))
    modified = ids.clone()
    modified[:, -1] = torch.tensor([8, 9])
    torch.testing.assert_close(model(modified)['logits'][:, :-1], original[:, :-1])
    torch.testing.assert_close(model(ids[:, :, :1])['logits'], model(ids[:, :, 0])['logits'])


def test_adaptive_trainer_checkpoint_resumes_after_plateau(tmp_path):
    for split in ('train', 'val'):
        directory = tmp_path / split
        directory.mkdir()
        np.save(directory / 'data_0.npy', np.tile([1, 2, 3], 100))
    tokenizer_path = tmp_path / 'tokenizer'
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({'<pad>': 0, 'a': 1, 'b': 2, '<eos>': 3})),
        pad_token='<pad>',
        eos_token='<eos>',
    )
    tokenizer.save_pretrained(tokenizer_path)
    config = GlobalConfig.model_validate({
        'run': {'name': 'resume'},
        'runtime': {'device_type': 'cpu', 'training_precision': 'fp32'},
        'model': {'dim': 16, 'n_layers': 1, 'n_heads': 2, 'n_kv_heads': 1,
                  'multiple_of': 16, 'max_seq_len': 4},
        'training': {
            'total_batch_size': 4, 'micro_batch_size': 1, 'max_steps': 5, 'max_steps_per_run': 2,
            'token_superposition': {
                'mode': 'adaptive', 'min_superposition_steps': 1, 'max_superposition_steps': 4,
                'min_delta': 10, 'patience': 1,
            },
        },
        'validation': {'run_on_first_step': True, 'every_x_steps': 1, 'validation_steps': 1},
        'checkpointing': {'save_checkpoints': True, 'every_x_steps': 1},
        'tokenizer': {'checkpoint_path': str(tokenizer_path)},
        'paths': {
            'datasets': {'training_path': str(tmp_path)},
            'runs': {'output_dir_path': str(tmp_path / 'runs')},
        },
        'optimizers': {'adamw': {'schedulers': {'cosine': {'warmup_steps': 0}}}},
        'logging': {'write_to_file': False},
    })
    args = SimpleNamespace(
        checkpoint=None, start_step=None, micro_batch_size=None,
        reset_optimizers=False, reset_dataloaders=False, reset_validation_history=False,
    )
    initial = Trainer(config, args)
    initial.train()
    assert initial.superposition.state.recovery_step == 2
    checkpoint = initial.get_checkpoints_dir_path() / 'model_1.pt'

    args.checkpoint = str(checkpoint)
    resumed = Trainer(config, args)
    resumed.setup()
    try:
        assert resumed.trainer_state.current_step == 2
        assert resumed.superposition.bag_size == 1
        assert resumed.train_loader.current_position == 16
        resumed.run_train(pbar=None)
        assert resumed.train_loader.current_position == 20
    finally:
        resumed.cleanup()
