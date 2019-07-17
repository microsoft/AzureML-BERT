from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import os
import logging
from multiprocessing import Pool
import multiprocessing
import os
import logging
import shutil
import tempfile
import argparse
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO, Callable, Set
from hashlib import sha256
from functools import wraps
import random
from tqdm import tqdm
from random import shuffle
import pickle

import sys
sys.path.append("..")
from pytorch_pretrained_bert.tokenization import BertTokenizer
from dataset import TokenInstance, PretrainingDataCreator, WikiNBookCorpusPretrainingDataCreator


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_data(input_file, output_file):
    if not os.path.exists(output_file):
        print(input_file)
        dataset = WikiNBookCorpusPretrainingDataCreator(
            input_file, tokenizer, dupe_factor=6, max_seq_length=512)
        dataset.save(output_file)
        print(f"Completed Pickling: {output_file}")
    else:
        print(f'Already parsed: {output_file}')


parser = argparse.ArgumentParser(
    description="Give initial arguments for parsing")

parser.add_argument("--input_dir", type=str, help="This folder contains .txt files of Wikipedia or BookCorpus Data. Each .txt file contains the text from the documents. \
                                              It makes an assumption that each line in the file represents a single line in the document too.\
                                              A blank line represents completion of a document.")
parser.add_argument("--output_dir", type=str, help="Path to Output Directory.")
parser.add_argument("--token_file", default="bert-large-uncased", type=str)
parser.add_argument("--do_lower_case", default=False, action="store_true",
                    help="This flag indicates the wheter the text should be lowercase or not")
parser.add_argument("--processes", "-p", default=0, type=int,
                    help="This is to do parallel processing of the txt files. It should be >=0. Default: 0 represents that it will use all the available cores in the CPU.")

args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(
    args.token_file, do_lower_case=args.do_lower_case)

input_files = []
output_files = []
num_processes = 1

if args.processes < 0 or args.processes > multiprocessing.cpu_count():
    raise ValueError(
        "The value of --processes should be >=0 and less than the max cores in the CPU.")
elif args.processes == 0:  # Use all cores
    num_processes = multiprocessing.cpu_count()
else:
    num_processes = args.processes

for filename in os.listdir(args.input_dir):
    input_file = os.path.join(args.input_dir, filename)
    #outfilename = "_".join(filename.split('.')[:-1]) + ".bin"
    outfilename = filename + ".bin"
    output_file = os.path.join(args.output_dir, outfilename)
    input_files.append(input_file)
    output_files.append(output_file)
    parse_data(input_file, output_file)

with Pool(processes=num_processes) as pool:
    pool.starmap(parse_data, zip(input_files, output_files))
