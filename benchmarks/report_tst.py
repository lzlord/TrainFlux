import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


MODES = ('standard', 'fixed', 'adaptive')
LABELS = {'standard': 'Standard', 'fixed': 'Fixed TST', 'adaptive': 'Adaptive TST'}
COLORS = {'standard': '#253659', 'fixed': '#c25a21', 'adaptive': '#007f77'}


def read_trials(source):
    trials = {}
    for budget in ('tokens', 'time'):
        for mode in MODES:
            path = source / f'{budget}-{mode}.json'
            if path.exists():
                row = json.loads(path.read_text(encoding='utf-8'))
                if (row['budget'], row['mode']) != (budget, mode) or not row['gpu']:
                    raise ValueError(f'Unexpected benchmark result in {path}')
                trials[budget, mode] = row
    if not trials:
        raise ValueError(f'No GPU benchmark results in {source}')
    return trials


def plot_validation(trials, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained', sharey=True)
    for ax, budget, x_key, x_label in zip(
        axes, ('tokens', 'time'), ('tokens', 'seconds'), ('Input tokens (millions)', 'Training time (s)')
    ):
        ax.set(title='Equal-token budget' if budget == 'tokens' else 'Equal-time budget', xlabel=x_label)
        ax.grid(alpha=0.25)
        for mode in MODES:
            row = trials.get((budget, mode))
            if row is None:
                continue
            points = row['validation']
            scale = 1_000_000 if budget == 'tokens' else 1
            x = [point[x_key] / scale for point in points]
            y = [point['loss'] for point in points]
            ax.plot(x, y, color=COLORS[mode], label=LABELS[mode], linewidth=2)
            ax.scatter(x[-1], y[-1], color=COLORS[mode], s=25, zorder=3)
        if ax.lines:
            ax.legend(frameon=False)
    axes[0].set_ylabel('Next-token validation loss')
    fig.savefig(output / 'validation.png', dpi=180)
    plt.close(fig)


def plot_resources(trials, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].set_ylabel('Tokens/s including periodic validation')
    axes[1].set_ylabel('Peak allocated GPU memory (MiB)')
    for ax, field in zip(axes, ('tokens_per_second', 'peak_gpu_allocated_mib')):
        for i, budget in enumerate(('tokens', 'time')):
            for j, mode in enumerate(MODES):
                row = trials.get((budget, mode))
                if row is not None:
                    ax.bar(i + (j - 1) * 0.22, row[field], width=0.2, color=COLORS[mode])
        ax.set_xticks([0, 1], ['Equal tokens', 'Equal time'])
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[mode], label=LABELS[mode]) for mode in MODES]
    fig.legend(handles=handles, loc='outside upper center', ncol=3, frameon=False)
    fig.savefig(output / 'resources.png', dpi=180)
    plt.close(fig)


def write_report(trials, output):
    reference = next(iter(trials.values()))
    dataset = reference['dataset']['splits']
    lines = [
        '# TrainFlux Adaptive TST benchmark',
        '',
        f"Tendril has {reference['parameters']:,} parameters. "
        f"The runs used {reference['gpu']}, PyTorch {reference['torch_version']}, and seed {reference['seed']}.",
        '',
        f"The TinyStories subset contains {dataset['train']['stories']:,} training stories and {dataset['val']['stories']:,} "
        f"validation stories. The 4,096-token BPE tokenizer was trained on the training stories. "
        f"The training shard contains {dataset['train']['tokens']:,} tokens. Validation uses next-token cross-entropy "
        "on a fixed slice of 65,536 target tokens. Perplexity is the exponential of this loss.",
        '',
        '## Results',
    ]
    for budget, title in (('tokens', 'Equal-token budget'), ('time', 'Equal-time budget')):
        lines.extend([
            '',
            f'### {title}',
            '',
            '| Mode | Tokens | Training (s) | Val loss | Perplexity | Tokens/s | Step tokens/s | Peak MiB (allocated / reserved) | Estimated GPU cost (USD) | Recovery step |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for mode in MODES:
            row = trials.get((budget, mode))
            if row is None:
                lines.append(f'| {LABELS[mode]} | — | — | — | — | — | — | — | — | — |')
                continue
            recovery = row['recovery_step']
            lines.append(
                f"| {LABELS[mode]} | {row['tokens']:,} | {row['train_loop_seconds']:.2f} | "
                f"{row['validation_loss']:.4f} | {row['perplexity']:.2f} | "
                f"{row['tokens_per_second']:,.0f} | {row['train_step_tokens_per_second']:,.0f} | "
                f"{row['peak_gpu_allocated_mib']:.0f} / {row['peak_gpu_reserved_mib']:.0f} | "
                f"${row['estimated_gpu_usd']:.4f} | {recovery if recovery is not None else '—'} |"
            )
        baseline = trials.get((budget, 'standard'))
        if baseline is not None:
            lines.append('')
            for mode in ('fixed', 'adaptive'):
                row = trials.get((budget, mode))
                if row is None:
                    continue
                loss = row['validation_loss'] - baseline['validation_loss']
                tps = (row['tokens_per_second'] / baseline['tokens_per_second'] - 1) * 100
                lines.append(
                    f'- {LABELS[mode]} changed validation loss by {loss:+.4f} nats '
                    f'and throughput by {tps:+.1f}% relative to standard training.'
                )
                phases = row['phases']
                bag = phases['superposition']
                recovery = phases['standard']
                if bag['step_seconds'] and recovery['step_seconds']:
                    lines.append(
                        f"  Within the same sandbox, optimizer-step throughput was "
                        f"{bag['tokens'] / bag['step_seconds']:,.0f} tokens/s during superposition and "
                        f"{recovery['tokens'] / recovery['step_seconds']:,.0f} tokens/s during recovery."
                    )
        adaptive = trials.get((budget, 'adaptive'))
        if adaptive is not None and adaptive['recovery_step'] is not None:
            cap = adaptive['config']['training']['token_superposition']['max_superposition_steps']
            reason = 'after a validation-loss plateau' if adaptive['recovery_step'] < cap else 'at the configured step cap'
            lines.append(f"- Adaptive TST began recovery at optimizer step {adaptive['recovery_step']} ({reason}).")

    gpu_cost = sum(row['estimated_gpu_usd'] for row in trials.values())
    standard_tokens = trials.get(('tokens', 'standard'))
    standard_time = trials.get(('time', 'standard'))
    lines.extend([
        '',
        '![Validation loss](validation.png)',
        '',
        '![Throughput and GPU memory](resources.png)',
        '',
        '## Experimental protocol',
        '',
        '- The equal-token target is 3,145,728 input tokens. The equal-time target is 60 seconds. Timed trials '
        'stop at a completed optimizer step and may slightly exceed the target.',
        '- Training time covers the training loop, including data loading and periodic validation. The run data '
        'records an additional endpoint validation separately when needed. Step throughput uses optimizer '
        'step time.',
        '- All modes share the model, initialization seed, tokenizer, input order, AdamW settings, and validation '
        'sample. A TST step consumes twice as many input tokens. Equal-token trials can therefore use '
        'different numbers of optimizer steps.',
        '- Fixed TST enters recovery at step 250. Adaptive TST can switch after 100 steps and three consecutive '
        'validation checks without a loss improvement of at least 0.05. The step cap is 360. Steps are zero-indexed.',
        f'- Estimated GPU cost for completed trials is ${gpu_cost:.4f}, calculated from process duration and the '
        'L4 list rate.',
        '- Each setting uses a single seed. The results describe this model and training budget.',
        '- The adaptive policy and reported comparison share a validation slice. No independent test split was evaluated.',
    ])
    if standard_tokens is not None and standard_time is not None:
        lines.append(
            f"- Standard training achieved "
            f"{standard_tokens['tokens_per_second']:,.0f} and {standard_time['tokens_per_second']:,.0f} tokens/s "
            "across two L4 sandboxes. Run duration and sandbox variability limit interpretation of throughput differences."
        )
    lines.extend([
        '',
        '[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) · '
        '[Modal pricing](https://modal.com/pricing)',
    ])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trials = read_trials(args.source)
    plot_validation(trials, args.output)
    plot_resources(trials, args.output)
    write_report(trials, args.output)


if __name__ == '__main__':
    main()
