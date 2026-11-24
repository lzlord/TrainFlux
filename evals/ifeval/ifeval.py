import json

from evals.ifeval.checkers import CHECKERS
from engine.distributed import load_jsonl_file_and_scatter
from evals.instruction_list import score_instruction_list_example


def load_ifeval_eval_file(
    *,
    filepath,
    ddp,
    is_master_process,
    size=None
):
    def prepare_line_fn(line):
        example = json.loads(line)

        instruction_id_list = example['instruction_id_list']
        kwargs = example['kwargs']

        assert isinstance(example['prompt'], str)
        assert isinstance(instruction_id_list, list)
        assert isinstance(kwargs, list)
        assert len(instruction_id_list) == len(kwargs)

        return {
            'key': example['key'],
            'prompt': example['prompt'],
            'instruction_id_list': instruction_id_list,
            'kwargs': kwargs,
            'valid': True
        }

    def prepare_dummy_line_fn():
        return {
            'key': '__dummy__',
            'prompt': 'Say hello.',
            'instruction_id_list': [],
            'kwargs': [],
            'valid': False
        }

    return load_jsonl_file_and_scatter(
        filepath=filepath,
        ddp=ddp,
        is_master_process=is_master_process,
        prepare_line_fn=prepare_line_fn,
        prepare_dummy_line_fn=prepare_dummy_line_fn,
        size=size
    )

def score_ifeval_example(*, example, response):
    return score_instruction_list_example(
        example=example,
        response=response,
        checkers=CHECKERS,
        unsupported_error_prefix='Unsupported IFEval checker'
    )
