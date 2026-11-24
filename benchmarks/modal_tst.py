import argparse
import json
from pathlib import Path
import uuid

import modal


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/tst-tinystories'
LEDGER = ROOT / 'runs/tst-modal-ledger'
MODES = ('standard', 'fixed', 'adaptive')
RUN_TIMEOUT = 210
SANDBOX_TIMEOUT = 260
CPU_LIMIT = 2.0
MEMORY_LIMIT_MIB = 4096
GPU = 'L4'
# Hourly workspace rates from `modal billing rates`, with the sandbox multiplier.
GPU_USD_PER_HOUR = 0.800
CPU_USD_PER_CORE_HOUR = 0.141900
MEMORY_USD_PER_GIB_HOUR = 0.024000
MAX_RUN_COST = SANDBOX_TIMEOUT / 3600 * (
    GPU_USD_PER_HOUR + CPU_LIMIT * CPU_USD_PER_CORE_HOUR
    + MEMORY_LIMIT_MIB / 1024 * MEMORY_USD_PER_GIB_HOUR
)
# Reserve up to $1 of bounded sandbox compute across invocations.
BUDGET = 1.0
TORCH_IMAGE = 'pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime'


def reserve_trial(name):
    LEDGER.mkdir(parents=True, exist_ok=True)
    for slot in range(int(BUDGET / MAX_RUN_COST)):
        try:
            with (LEDGER / f'{slot}.json').open('x', encoding='utf-8') as handle:
                json.dump({'trial': name, 'max_usd': MAX_RUN_COST, 'timeout': SANDBOX_TIMEOUT}, handle)
            return
        except FileExistsError:
            continue
    raise RuntimeError('Modal GPU trial budget exhausted')


def prepare_image(app):
    base = modal.Image.from_registry(TORCH_IMAGE).pip_install(
        'transformers==4.57.6', 'datasets==4.5.0', 'wandb==0.24.2',
        'pydantic-settings==2.12.0', 'numpy==2.2.6', 'pyyaml==6.0.3', 'tqdm==4.67.3',
        extra_options='--break-system-packages',
    )
    image = base.build(app)
    for folder in ('benchmarks', 'engine', 'evals', 'inference', 'metrics', 'models', 'tasks', 'tokenization'):
        image = image.add_local_dir(ROOT / folder, f'/trainflux/{folder}', ignore=['**/__pycache__/**'])
    for filename in ('config.py', 'logger.py', 'utils.py', 'requirements.txt'):
        image = image.add_local_file(ROOT / filename, f'/trainflux/{filename}')
    return image.add_local_dir(DATA, '/data')


def run_trial(app, image, output, mode, budget):
    name = f'{budget}-{mode}'
    result_path = output / f'{name}.json'
    if result_path.exists():
        print(f'{name}: already complete', flush=True)
        return
    lock = output / f'.{name}.lock'
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f'{name} already has a launcher reservation: {lock}') from exc
    try:
        if not result_path.exists():
            run_reserved_trial(app, image, output, mode, budget, result_path)
    finally:
        lock.unlink()


def run_reserved_trial(app, image, output, mode, budget, result_path):
    name = f'{budget}-{mode}'
    reserve_trial(name)
    print(f'{name}: reserved up to ${MAX_RUN_COST:.4f}', flush=True)
    sandbox = modal.Sandbox.create(
        'bash', '-c', 'exec timeout --kill-after=5s "$@" 2>&1',
        'trainflux', f'{RUN_TIMEOUT}s', 'python', '-u', '-m', 'benchmarks.tst_trial',
        '/data', mode, budget,
        app=app,
        name=f'{name}-{uuid.uuid4().hex[:8]}',
        tags={'budget': budget, 'mode': mode},
        image=image,
        gpu=GPU,
        cpu=(1.0, CPU_LIMIT),
        memory=(2048, MEMORY_LIMIT_MIB),
        timeout=SANDBOX_TIMEOUT,
        workdir='/trainflux',
        block_network=True,
        env={'TQDM_DISABLE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
    )
    print(f'{name}: {sandbox.object_id}', flush=True)
    result = None
    log_path = output / f'{name}-{sandbox.object_id}.log'
    try:
        with log_path.open('w', encoding='utf-8') as log:
            for line in sandbox.stdout:
                log.write(line)
                log.flush()
                if line.startswith('BENCHMARK_RESULT '):
                    result = json.loads(line.removeprefix('BENCHMARK_RESULT '))
                elif 'TST |' in line:
                    print(line.strip(), flush=True)
        sandbox.wait()
        if sandbox.returncode != 0 or result is None:
            raise RuntimeError(f'{name} failed (exit {sandbox.returncode}). See {log_path}')
        result['modal_sandbox_id'] = sandbox.object_id
        result['modal_timeout_seconds'] = SANDBOX_TIMEOUT
        result['modal_max_compute_usd'] = MAX_RUN_COST
        temporary = result_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(result_path)
        print(f'{name}: loss={result["validation_loss"]:.4f} tokens={result["tokens"]:,}', flush=True)
    finally:
        sandbox.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--budget', choices=['tokens', 'time'])
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if not (DATA / 'manifest.json').is_file():
        parser.error('Prepare TinyStories with benchmarks.prepare_tinystories first')
    args.output.mkdir(parents=True, exist_ok=True)
    app = modal.App.lookup('trainflux-tst-benchmark', create_if_missing=True)
    print('Preparing the Modal image', flush=True)
    image = prepare_image(app)
    for budget in [args.budget] if args.budget else ['tokens', 'time']:
        for mode in [args.mode] if args.mode else MODES:
            run_trial(app, image, args.output, mode, budget)


if __name__ == '__main__':
    main()
