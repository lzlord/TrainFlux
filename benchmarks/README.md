# TinyStories benchmark protocol

The benchmark compares next-token training, fixed TST, and Adaptive TST on one NVIDIA L4 per trial. The model uses six Tendril decoder layers, 512 hidden dimensions, a 256-position context, and tied embeddings. A 4,096-token BPE tokenizer is trained on the first 35,000 stories from the pinned TinyStories training split. Validation uses a fixed 65,536-token slice from the held-out validation split.

| Setting | Value |
| --- | --- |
| Token budget | 3,145,728 input tokens |
| Time budget | 60 seconds of training-loop wall time |
| Per-step baseline batch | 2,048 input tokens |
| Per-step superposition batch | 4,096 input tokens over 2,048 transformer positions |
| Optimizer | AdamW, 20-step warmup, peak learning rate of 0.0006 |
| Precision and seed | BF16, 42 |
| Validation | Every 40 optimizer steps and at the endpoint |

Time includes data loading, optimizer updates, and periodic validation. The run stops after a completed optimizer step. When an additional endpoint validation is needed, its time is recorded separately. Throughput is also available for optimizer steps alone. GPU memory figures report PyTorch's peak allocated and reserved memory.

## Configuration

Add these settings to a dense Tendril pretraining YAML file and run `python train.py --config config.yaml`. Use the existing tokenizer and training shard paths.

```yaml
training:
  stage: pretraining
  max_steps: 1500
  token_superposition:
    mode: adaptive
    bag_size: 2
    switch_step: 250
    min_superposition_steps: 100
    max_superposition_steps: 360
    patience: 3
    min_delta: 0.05
validation:
  every_x_steps: 40
  validation_steps: 32
```

- `standard` is the default. `fixed` starts recovery at the zero-based `switch_step`.
- `adaptive` compares next-token validation loss with its best reference value. An improvement of at least `min_delta` updates the reference and resets patience. After the minimum superposition duration, `patience` consecutive checks without sufficient improvement trigger recovery. The step cap can trigger recovery independently.
- The trainer logs validation decisions on lines prefixed with `TST |`. Checkpoints store the reference loss, patience counter, and recovery step. Resume with the same TST configuration. `--reset-validation-history` resets the reference and counter while preserving a completed transition.
- Training shards must fit one distributed superposition batch plus the target offset. The loader carries its cursor into recovery. Validation always uses standard next-token cross-entropy.

## Modal execution

```bash
python -m benchmarks.modal_tst runs/tst-modal
python -m benchmarks.modal_tst runs/tst-modal --budget time --mode adaptive
python -m benchmarks.report_tst runs/tst-modal runs/tst-report
```

The [measurement files](results/tinystories-l4) contain six run summaries and the validation and throughput plots.
