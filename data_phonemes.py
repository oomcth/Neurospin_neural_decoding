import pandas as pd
from utils_load_phonemes import get_phonemes
from utils_bis import read_raw
import torch
import numpy as np
import operator


sigma = 0.02


def load(subject, run_id):
    df_phonemes = get_phonemes(run_id)
    raw, meta = read_raw(subject, run_id, False, "auditory")

    gap = meta['onset'][0] - df_phonemes['start'][1]

    def rectifier(x):
        return x + gap

    df_phonemes['start'] = df_phonemes['start'].apply(rectifier)
    df_phonemes['end'] = df_phonemes['end'].apply(rectifier)
    df_phonemes['delta'] = df_phonemes['end'] - df_phonemes['start']

    tensor_meg = torch.tensor(raw.get_data())

    return tensor_meg, df_phonemes


def time_to_index(array):
    return (array * 100).astype(int)


def generate_samples(n_samples, tensor_meg, df_phonemes):
    noise = np.random.normal(0, 1, 2*n_samples)
    pick = np.random.choice(range(len(df_phonemes['phoneme'])), n_samples)

    starts = np.array(df_phonemes.loc[pick, 'start'])
    starts += sigma * noise[:n_samples] * starts
    starts = time_to_index(starts)
    ends = np.array(df_phonemes.loc[pick, 'end'])
    ends += sigma * noise[:n_samples] * ends
    ends = time_to_index(ends)

    tensors = [torch.index_select(tensor_meg, 1, torch.arange(d, f))
               for d, f in zip(starts, ends)]

    phonemes = list(df_phonemes.loc[pick, "phoneme"])

    return tensors, phonemes


def generate_samples_All(subject, samples_per_run, noise=True):

    train_ids = ["01", "02", "03", "04", "05", "06", "07"]
    test_ids = ["09"]
    valid_ids = ["08"]

    train_tensors, train_phonemes = [], []
    for id in train_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors, temp_phonemes = generate_samples(samples_per_run,
                                                       tensor_meg,
                                                       df_phonemes)
        train_tensors += temp_tensors
        train_phonemes += temp_phonemes

    valid_tensors, valid_phonemes = [], []
    for id in valid_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors, temp_phonemes = generate_samples(samples_per_run,
                                                       tensor_meg,
                                                       df_phonemes)
        valid_tensors += temp_tensors
        valid_phonemes += temp_phonemes

    test_tensors, test_phonemes = [], []
    for id in test_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors, temp_phonemes = generate_samples(samples_per_run,
                                                       tensor_meg,
                                                       df_phonemes)
        test_tensors += temp_tensors
        test_phonemes += temp_phonemes

    train_tensors = torch.stack(train_tensors)
    valid_tensors = torch.stack(valid_tensors)
    test_tensors = torch.stack(test_tensors)

    if noise:
        train_tensors = add_noise(train_tensors)
        indices = torch.randperm(len(train_phonemes))
        temp_tensor = train_tensors[indices]
        temp_phonemes = list(operator.itemgetter(*indices.tolist())(train_phonemes))
        train_tensors = temp_tensor
        train_phonemes = temp_phonemes

        valid_tensors = add_noise(valid_tensors)
        indices = torch.randperm(len(valid_phonemes))
        temp_tensor = valid_tensors[indices]
        temp_phonemes = list(operator.itemgetter(*indices.tolist())(valid_phonemes))
        valid_tensors = temp_tensor
        valid_phonemes = temp_phonemes

        test_tensors = add_noise(test_tensors)
        indices = torch.randperm(len(test_phonemes))
        temp_tensor = test_tensors[indices]
        temp_phonemes = list(operator.itemgetter(*indices.tolist())(test_phonemes))
        test_tensors = temp_tensors
        test_phonemes = temp_phonemes

    return ((train_tensors, train_phonemes),
            (valid_tensors, valid_phonemes),
            (test_tensors, test_phonemes))


def add_noise(tensor):

    n_percent = 0.01
    n = int(tensor.size(0) * n_percent)

    min_change = -0.02
    max_change = 0.02

    noise = torch.empty(tensor[:n].shape).uniform_(min_change, max_change)

    temp = tensor.clone()
    temp[:n] *= (1 + noise)
    return temp


if __name__ == "__main__":
    tensors, phonemes = generate_samples_All("1", 2)
    print(phonemes)
