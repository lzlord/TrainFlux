import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from config import GlobalConfig
from engine.trainer import Trainer
from logger import logger
from metrics.step import StepType


MODES = ('standard', 'fixed', 'adaptive')
TOKEN_BUDGET = 3_145_728
TIME_BUDGET = 60.0
GPU_RATE = 0.000222


def trial_config(data, mode):
    return GlobalConfig.model_validate({
        'runtime': {
            'device_type': 'cuda',
            'training_precision': 'bf16',
        },
        'model': {
            'dim': 512,
            'n_layers': 6,
            'n_heads': 8,
            'n_kv_heads': 4,
            'multiple_of': 128,
            'max_seq_len': 256,
            'tied_embeddings': True,
        },
        'training': {
            'max_steps': 20_000,
            'total_batch_size': 2048,
            'micro_batch_size': 8,
            'token_superposition': {
                'mode': mode,
                'bag_size': 2,
                'switch_step': 250,
                'min_superposition_steps': 100,
                'max_superposition_steps': 360,
                'patience': 3,
                'min_delta': 0.05,
            },
        },
        'tokenizer': {'checkpoint_path': str(data / 'tokenizer')},
        'paths': {'datasets': {'training_path': str(data)}},
        'validation': {'every_x_steps': 40, 'validation_steps': 32},
        'optimizers': {'adamw': {
            'min_lr': 0.0006,
            'max_lr': 0.0006,
            'schedulers': {'active': 'wsd', 'wsd': {'warmup_steps': 20, 'stable_steps': 19_979}},
        }},
        'logging': {'write_to_file': False},
    })


class BenchmarkTrainer(Trainer):
    def __init__(self, config, token_limit, time_limit):
        args = SimpleNamespace(
            checkpoint=None, start_step=None, micro_batch_size=None,
            reset_optimizers=False, reset_dataloaders=False, reset_validation_history=False,
        )
        super().__init__(config, args)
        self.token_limit = token_limit
        self.time_limit = time_limit
        self.tokens = 0
        self.wall_start = 0.0
        self.train_loop_seconds = 0.0
        self.train_step_seconds = 0.0
        self.final_validation_seconds = 0.0
        self.peak_bytes = 0
        self.peak_reserved_bytes = 0
        self.points = []
        self.phase_metrics = {
            'standard': {'tokens': 0, 'step_seconds': 0.0, 'steps': 0},
            'superposition': {'tokens': 0, 'step_seconds': 0.0, 'steps': 0},
        }

    def start_training_loop(self):
        self.wall_start = time.perf_counter()
        super().start_training_loop()

    def process_step(self, pbar):
        super().process_step(pbar)
        elapsed = time.perf_counter() - self.wall_start
        if (
            (self.token_limit is not None and self.tokens >= self.token_limit)
            or (self.time_limit is not None and elapsed >= self.time_limit)
        ):
            self.train_loop_seconds = elapsed
            if not self.points or self.points[-1]['tokens'] != self.tokens:
                self.run_validation(pbar)
                self.final_validation_seconds = time.perf_counter() - self.wall_start - elapsed
            self.trainer_state.should_stop = True

    def run_validation(self, pbar):
        if self.superposition.config.mode != 'adaptive':
            self.val_loader.reset()
        return super().run_validation(pbar)

    def log_step_metrics(self, *, step_metrics, aggregated_metrics=None, **kwargs):
        if self.device_ctx.device_type == 'cuda':
            self.peak_bytes = max(self.peak_bytes, torch.cuda.max_memory_allocated())
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, torch.cuda.max_memory_reserved())
        step = self.trainer_state.current_step + 1
        if step_metrics.step_type == StepType.TRAIN:
            bag_size = self.superposition.bag_size
            tokens = self.config.training.total_batch_size * bag_size
            self.tokens += tokens
            self.train_step_seconds += step_metrics.dt
            phase = self.phase_metrics['superposition' if bag_size > 1 else 'standard']
            phase['tokens'] += tokens
            phase['step_seconds'] += step_metrics.dt
            phase['steps'] += 1
            if step != 1 and step % 40 != 0:
                return
        else:
            loss = aggregated_metrics['Val CE Loss']
            self.points.append({
                'step': step,
                'tokens': self.tokens,
                'seconds': self.train_loop_seconds or time.perf_counter() - self.wall_start,
                'loss': loss,
                'perplexity': math.exp(loss),
            })
        super().log_step_metrics(step_metrics=step_metrics, aggregated_metrics=aggregated_metrics, **kwargs)


def run_trial(data, mode, budget):
    token_limit = TOKEN_BUDGET if budget == 'tokens' else None
    time_limit = TIME_BUDGET if budget == 'time' else None
    config = trial_config(data, mode)
    trainer = BenchmarkTrainer(config, token_limit, time_limit)
    started = time.perf_counter()
    trainer.train()
    worker_seconds = time.perf_counter() - started
    if not trainer.train_loop_seconds or (token_limit is not None and trainer.tokens != token_limit):
        raise RuntimeError('Training did not finish at the requested benchmark budget')
    if mode != 'standard' and not trainer.phase_metrics['standard']['steps']:
        raise RuntimeError('Benchmark budget expired before token recovery training')

    val = trainer.points[-1]
    return {
        'mode': mode,
        'budget': budget,
        'target': token_limit if budget == 'tokens' else time_limit,
        'gpu': torch.cuda.get_device_name(),
        'torch_version': torch.__version__,
        'seed': config.training.seed,
        'config': config.model_dump(mode='json'),
        'parameters': trainer.model.get_total_parameters_count(),
        'steps': val['step'],
        'tokens': trainer.tokens,
        'recovery_step': trainer.superposition.state.recovery_step,
        'validation_loss': val['loss'],
        'perplexity': val['perplexity'],
        'train_loop_seconds': trainer.train_loop_seconds,
        'train_step_seconds': trainer.train_step_seconds,
        'final_validation_seconds': trainer.final_validation_seconds,
        'tokens_per_second': trainer.tokens / trainer.train_loop_seconds,
        'train_step_tokens_per_second': trainer.tokens / trainer.train_step_seconds,
        'peak_gpu_allocated_mib': trainer.peak_bytes / 2**20,
        'peak_gpu_reserved_mib': trainer.peak_reserved_bytes / 2**20,
        'worker_seconds': worker_seconds,
        'estimated_gpu_usd': worker_seconds * GPU_RATE,
        'phases': trainer.phase_metrics,
        'validation': trainer.points,
        'dataset': json.loads((data / 'manifest.json').read_text(encoding='utf-8')),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('mode', choices=MODES)
    parser.add_argument('budget', choices=('tokens', 'time'))
    args = parser.parse_args()
    result = run_trial(args.data, args.mode, args.budget)
    logger.info('Benchmark endpoint recorded.')
    print('BENCHMARK_RESULT ' + json.dumps(result, allow_nan=False), flush=True)
