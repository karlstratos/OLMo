# python inspect_npy_file.py data --max_sequence_length 2048 --seed 42

import argparse
import numpy as np
import os
import random

from logger import Logger, add_colors
from olmo.data.memmap_dataset import MemMapDataset
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


END = '<|endoftext|>'


parser = argparse.ArgumentParser()
parser.add_argument('top_dir', type=str)
parser.add_argument('--max_sequence_length', type=int, default=16)
parser.add_argument('--N', type=int, default=1)
parser.add_argument('--show_input_ids', action='store_true')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
logger = Logger(stamp=False)

data_dir = Path(args.top_dir) / 'preprocessed/olmo-mix/v1_5/gpt-neox-20b-pii-special/'
npy_paths = [path for path in data_dir.iterdir() if path.suffix == '.npy']

logger(f'Constructing a MemMapDataset using {len(npy_paths)} npy files in directory {data_dir}', ['yellow'])
for i, path in enumerate(npy_paths):
    logger(f'  File {i + 1}: {path.name}, just a np seq of {os.stat(path).st_size} bytes representing np.uint16, sent loaded thru np.frombuffer(sent_bytes, dtype=np.uint16)', ['yellow'])


dataset = MemMapDataset(
    *npy_paths,
    chunk_size=args.max_sequence_length,
    memmap_dtype=np.uint16,
    metadata={'path': str(path) for path in npy_paths},
    include_instance_metadata=True,
    generate_attention_mask=False,
    pad_token_id=1,
    label_mask_paths=None
)

logger(f'\n# instances (i.e., input sequences): {len(dataset)}', ['yellow'])
logger(f'  This built offsets for the first time: ', ['yellow'], newline=False)
for (s, t) in dataset.offsets:
    logger(f' ({s}, {t})', ['yellow'], newline=False)
logger('. Num instances is just taking the last value.', ['yellow'])
logger(f'  Since each sequence has length 2048, this means the total number of tokens is {2048 * len(dataset) / 1000000000:5.2f} billions.', ['yellow'])





tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-1B', trust_remote_code=True)
indices = random.sample(range(len(dataset)), args.N)  # random global indices
for i in indices:
    seq = dataset[i]
    input_ids = seq['input_ids']
    source_file = Path(seq['metadata']['path'])
    logger('-' * 80 + f'dataset[{i}] ({source_file.name})', ['blue'])
    if args.show_input_ids:
        logger(str(input_ids.tolist()))
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    docs = text.split(END)
    text_colored = (add_colors(END, ['red'])).join([add_colors(doc, ['cyan']) for doc in docs])
    logger(text_colored)
