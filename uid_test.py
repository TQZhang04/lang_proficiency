import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from utils import calc_surprisal

import time
import os
from pathlib import Path


if __name__ == "__main__":
    dataset = load_dataset("UniversalCEFR/readme_en", split="train")
    dataset = dataset.shuffle(seed=42).select(range(300))

    dataset['text']

    OUTPUT_ROOTDIR = "../Surprisal_outputs"
    MODEL_NAME = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, padding=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    calc_surprisal(model, tokenizer, dataset['text'], OUTPUT_ROOTDIR, 'gpt2', verbose=True)

