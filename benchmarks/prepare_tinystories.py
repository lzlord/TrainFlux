import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
SOURCE = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}'
BOUNDARY = b'<|endoftext|>'


def stories(filename):
    with requests.get(f'{SOURCE}/{filename}', stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        paragraphs = []
        for line in response.iter_lines():
            if line.strip() == BOUNDARY:
                if paragraphs:
                    yield b'\n'.join(paragraphs).decode('utf-8')
                    paragraphs.clear()
            elif line.strip():
                paragraphs.append(line)


def prepare(destination, train_count, val_count):
    train = list(islice(stories('TinyStories-train.txt'), train_count))
    val = list(islice(stories('TinyStories-valid.txt'), val_count))
    if len(train) != train_count or len(val) != val_count:
        raise RuntimeError('TinyStories download ended before the requested sample')

    special = ['<pad>', '<unk>', '<eos>']
    backend = Tokenizer(models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        train,
        trainers.BpeTrainer(
            vocab_size=4096,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special,
            show_progress=False,
        ),
        length=len(train),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'
    )
    tokenizer.save_pretrained(destination / 'tokenizer')

    manifest = {'source': SOURCE, 'vocab_size': len(tokenizer), 'splits': {}}
    for split, texts in [('train', train), ('val', val)]:
        path = destination / split / 'data_0.npy'
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = backend.encode_batch(texts, add_special_tokens=False)
        stream = (token for story in encoded for token in [*story.ids, tokenizer.eos_token_id])
        tokens = np.fromiter(stream, dtype=np.uint16)
        np.save(path, tokens, allow_pickle=False)
        manifest['splits'][split] = {
            'stories': len(texts),
            'tokens': len(tokens),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest['tokenizer_sha256'] = hashlib.sha256(
        (destination / 'tokenizer' / 'tokenizer.json').read_bytes()
    ).hexdigest()
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--train-stories', type=int, default=35000)
    parser.add_argument('--val-stories', type=int, default=1000)
    args = parser.parse_args()
    prepare(args.destination, args.train_stories, args.val_stories)
