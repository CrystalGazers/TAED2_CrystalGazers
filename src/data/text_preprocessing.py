from types import SimpleNamespace
from collections import Counter
import os
import re
import pathlib
import subprocess
import array
import pickle
import numpy as np
import pandas as pd

class Vocabulary(object):
    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.token2idx = {}
        self.idx2token = []
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        if pad_token is not None:
            self.pad_index = self.add_token(pad_token)
        if unk_token is not None:
            self.unk_index = self.add_token(unk_token)
        if eos_token is not None:
            self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def get_index(self, token):
        if isinstance(token, str):
            return self.token2idx.get(token, self.unk_index)
        else:
            return [self.token2idx.get(t, self.unk_index) for t in token]

    def get_token(self, index):
        return self.idx2token[index]

    def __len__(self):
        return len(self.idx2token)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

class Punctuation:
    html = re.compile(r'&apos;|&quot;')
    punctuation = re.compile(r'[^\w\s·]|_')
    spaces = re.compile(r'\s+')
    ela_geminada = re.compile(r'l · l')

    def strip(self, s):
        '''
        Remove all punctuation characters.
        '''
        s = self.html.sub(' ', s)
        s = self.punctuation.sub(' ', s)
        s = self.spaces.sub(' ', s).strip()
        s = self.ela_geminada.sub('l·l', s)
        return s

def remove_punctuation(input_path, output_path):
    punc = Punctuation()
    with open(input_path, 'r', encoding='utf-8') as inpf, open(output_path, 'w', encoding='utf-8') as outf:
        for line in inpf:
            line = punc.strip(line)
            print(line, file=outf)

def get_token_counter(file_path):
    counter = Counter()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = line.split()
                counter.update(tokens)
    return counter

def get_token_vocabulary(token_counter, cutoff=3, maxtokens=None, verbose=1, eos_token=None):
    vocab = Vocabulary(eos_token=eos_token)
    total_count = sum(token_counter.values())
    in_vocab_count = 0

    for token, count in token_counter.most_common(maxtokens):
        if count >= cutoff:
            vocab.add_token(token)
            in_vocab_count += count

    if verbose:
        OOV_count = total_count - in_vocab_count
        print('OOV ratio: %.2f%%.' % (100*OOV_count / total_count))
    return vocab

def get_token_index(file_path, vocab, eos_token=None):
    index_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if eos_token is not None:
                    line += ' ' + eos_token
                tokens = line.strip().split()
                index_list.append([vocab.get_index(token) for token in tokens])
    return index_list

def get_number_of_samples(idx_list, window_size):
    nsamples = 0
    for line in idx_list:
        if len(line) <= window_size // 2:
            continue
        nsamples += len(line)
    return nsamples

def get_data(idx_list, window_size, pad_index=0):
    nsamples = get_number_of_samples(idx_list, window_size)
    winput = np.empty((nsamples, window_size - 1), dtype=np.int32)
    target = np.empty(nsamples, dtype=np.int32)
    left_window = window_size // 2
    right_window = window_size - left_window - 1
    sample = 0
    for line in idx_list:
        if len(line) <= window_size // 2:
            continue
        ext_line = [pad_index] * left_window + line + [pad_index] * right_window
        for i, token_id in enumerate(line):
            winput[sample] = ext_line[i:i + left_window] + ext_line[i + left_window + 1:i + window_size]
            target[sample] = token_id
            sample += 1
    assert nsamples == sample
    return winput, target

def prepare_dataset(params):
    dataset_prefix = params.dataset
    working_prefix = params.working
    cutoff = params.cutoff
    maxtokens = params.maxtokens
    window_size = params.window_size

    data = []
    for part in ['train', 'valid', 'test']:
        data_filename = f'{dataset_prefix}.{part}.tokens'
        data_filename_nopunct = f'{working_prefix}.{part}.tokens.nopunct'
        remove_punctuation(data_filename, data_filename_nopunct)

        if part == 'train':
            # Basic token statistics
            token_counter = get_token_counter(data_filename_nopunct)
            print(f'Number of Tokens: {sum(token_counter.values())}')
            print(f'Number of different Tokens: {len(token_counter)}')
            pickle.dump(token_counter, open(f'{data_filename_nopunct}.dic', 'wb'))

            # Token vocabulary
            token_vocab = get_token_vocabulary(token_counter, cutoff=cutoff, maxtokens=maxtokens)
            token_vocab.save(f'{working_prefix}.vocab')
            print(f'Vocabulary size: {len(token_vocab)}')

        # Token indexes
        train_idx = get_token_index(data_filename_nopunct, token_vocab)
        print(f'Number of lines ({part}): {len(train_idx)}')

        # Get input and target arrays
        idata, target = get_data(train_idx, window_size)
        data.append((idata, target))
        print(f'Number of samples ({part}): {len(target)}')

        # Save numpy arrays
        np.savez(f'{working_prefix}.{part}.npz', idata=idata, target=target)
    return token_vocab, data

DATASET_ROOT = f'./raw'
WORKING_ROOT = f'./preprocessed'
DATASET_PREFIX = 'ca.wiki'

for DATASET_VERSION in ["ca-all"]:#os.listdir(DATASET_ROOT):
    # Create working dir
    pathlib.Path(f"{WORKING_ROOT}/{DATASET_VERSION}").mkdir(parents=True, exist_ok=True)

    params = SimpleNamespace(
        window_size = 7,
        cutoff = 3,
        maxtokens = 100000,
        dataset = f'{DATASET_ROOT}/{DATASET_VERSION}/{DATASET_PREFIX}',
        working = f'{WORKING_ROOT}/{DATASET_VERSION}/{DATASET_PREFIX}',
    )

    print(f"Processing {DATASET_VERSION}")
    vocab, data = prepare_dataset(params)

    print('token to index:')
    for word in ['raïm', 'intel·ligent']:
        index = vocab.get_index(word)
        print(f'{word} -> {index}')

    print('\nindex to token:')
    for index in [8428, 7466]:
        word = vocab.get_token(index)
        print(f'{index} -> {word}')
