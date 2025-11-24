import numpy as np
import re 
import time
import random
import requests
import csv
import pandas as pd
import json
import torch
from datasets import load_dataset
from urllib.parse import urljoin, urlparse
import nltk
from tqdm import tqdm
from pathlib import Path
import os
import glob
import sys

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# %%
def split_batches(lst, batch_size):
    """splits a given list into batches of a given size

    Args:
        lst (list): list to batch
        batch_size (int): size of batches

    Returns:
        list: list of batches of size batch_size
    """
    num_batches = int(np.ceil(len(lst) / batch_size))
    batched = []
    for i in range(num_batches):
        start_idx = i * batch_size
        batch = lst[start_idx:start_idx + batch_size]
        batched.append(batch)
    return batched

def to_tokens_and_logprobs(model, tokenizer, input_texts, disable_progress=False, quiet=False):
    """
    Calculate token-level surprisals for input texts
    
    Args:
        model: The language model to use
        tokenizer: The tokenizer for the model
        input_texts: List of texts to process
        disable_progress: Whether to disable progress bars
        quiet: Whether to suppress all print statements
    
    Returns:
        List of dataframes containing token and surprisal information
    """
    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize inputs
    input_ids = tokenizer(input_texts, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    
    # Run model inference
    if not quiet:
        print("Running model inference...", end="", flush=True)
    t0 = time.time()
    outputs = model(input_ids)
    t1 = time.time()
    model.to("cpu")  # Free up GPU memory
    if not quiet:
        print(f" done in {t1-t0:.2f}s", flush=True)

    # Calculate surprisals
    if not quiet:
        print("Computing surprisals...", end="", flush=True)
    t0 = time.time()  
    logits = outputs.logits.cpu().detach()
    probs = torch.softmax(logits, dim=-1)
    surprisals = -1 * torch.log2(probs)

    # Align tokens and surprisals
    input_ids = input_ids.cpu().detach()
    surprisals = surprisals[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)
    t1 = time.time()
    if not quiet:
        print(f" done in {t1-t0:.2f}s", flush=True)
    
    # Create dataframes
    batch = []
    for i, id_surp in enumerate(zip(input_ids, gen_surprisals)):
        sentence = []
        input_sentence, input_surprisals = id_surp
        for token, p in zip(input_sentence, input_surprisals):
            if token not in tokenizer.all_special_ids:
                sentence.append({
                    "token": tokenizer.decode(token),
                    "surprisal": p.item()
                })
        batch.append(pd.DataFrame(sentence))
    
    return batch

def calc_surprisal(model, tokenizer, input_texts, output_dir, model_name, num_files=-1, batch_size=20, verbose=False):
    """Calculate surprisal for each token of a corpus of texts

    Args:
        model (GPT2LMHeadModel): pretrained language model to use for probability calculation
        tokenizer (GPT2Tokenizer): tokenizer associated with the model
        input_dir (str/path): path to directory containing corpus
        output_dir (str/path): desired output directory for csv files
        model_name (str): name of the model to filter files for
        num_files (int, optional): number of files to analyze. Passing -1 will analyze all files in the directory.
        batch_size (int, optional): batch size for surprisal calculation input. Defaults to 20.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_texts = input_texts
    
    if verbose:
        print(f"Successfully read {len(all_texts)} files")
    
    batched_texts = split_batches(all_texts, batch_size=batch_size)
    
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    batch_pbar = tqdm(
        total=len(batched_texts),
        desc=f"Processing {model_name} batches",
        unit="batch",
        disable=not verbose,
        position=0,
        leave=True
    )
    
    for batch_idx, texts in enumerate(batched_texts):
        try:
            # if verbose:
                # print(f"\nProcessing batch {batch_idx+1}/{len(batched_texts)} with {len(texts)} files")
            
            outputs = to_tokens_and_logprobs(model, tokenizer, texts, disable_progress=True, quiet=True)
            
            texts_pbar = tqdm(
                total=len(batched_texts),
                desc=f"Processing batch {batch_idx}",
                unit="text",
                disable=not verbose,
                position=0,
                leave=True
            )
            
            for idx, df in enumerate(outputs):
                text_idx = batch_idx * batch_size + idx
                try:
                    output_fp = output_root / f"{text_idx}.csv"
                    df.to_csv(output_fp, index=False)
                except Exception as e:
                    print(f"Error writing file {text_idx}: {e}")
                finally:
                    texts_pbar.update(1)
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
        finally:
            batch_pbar.update(1)
    
    batch_pbar.close()
    
            
    if verbose:
        print(f"Surprisal calculation complete for {model_name}. Results saved to {output_dir}")

def UID_variance(text):
    """Calculate UID variance metric from surprisal values"""
    N = text.shape[0]
    if N == 0:
        return np.nan
    mu = text['surprisal'].mean()
    surprisals = text['surprisal']
    return ((surprisals - mu) ** 2).sum() / N

def UID_pairwise(text):
    """Calculate UID pairwise metric from surprisal values"""
    N = text.shape[0]
    if N <= 1:
        return np.nan
    surprisals = text['surprisal']
    return (surprisals.diff() ** 2).dropna().sum() / (N - 1)