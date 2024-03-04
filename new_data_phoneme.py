from utils_load_phonemes import get_phonemes
from utils_bis import read_raw
import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter as getter
import pandas as pd
from copy import copy
import torchvision.transforms as transforms
from tqdm import tqdm


pixel_simga = 0.05
choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']


def load(subject, run_id, equilibrate=False, freq=75):
    df_phonemes = get_phonemes(run_id)

    raw, meta = read_raw(subject, run_id, False, "auditory")

    gap = meta['onset'][0] - df_phonemes['start'][1]

    def rectifier(x):
        return x + gap

    df_phonemes['start'] = df_phonemes['start'].apply(rectifier)
    df_phonemes['end'] = df_phonemes['end'].apply(rectifier)
    df_phonemes['delta'] = df_phonemes['end'] - df_phonemes['start']

    tensor_meg = torch.tensor(raw.get_data())

    df_phonemes = df_phonemes.drop(
        df_phonemes[df_phonemes['delta'] > 30 / 100].index
        )

    del raw
    del meta

    if equilibrate:
        return tensor_meg, equilibrate_phonemes(df_phonemes, freq)
    else:
        return tensor_meg, df_phonemes.reset_index(drop=True)


def equilibrate_phonemes(df, freq=75):
    phoneme_counts = df['phoneme'].value_counts()

    frequent_phonemes = phoneme_counts[phoneme_counts > freq]

    max_count = frequent_phonemes.max()

    duplicated_rows = []

    for phoneme, count in frequent_phonemes.items():
        duplications_needed = max_count - count

        phoneme_rows = df[df['phoneme'] == phoneme]

        duplicated = phoneme_rows.sample(n=duplications_needed, replace=True)
        duplicated_rows.append(duplicated)

    df_balanced = pd.concat([df] + duplicated_rows, ignore_index=True)

    df_balanced = df_balanced.reset_index(drop=True)

    return df_balanced


def generate_samples_All(subject, noise=False, clamp=False, equilibrate=True,
                         freq=75):

    train_ids = ["01", "02", "03", "04", "05", "06", "07"]
    valid_ids = ["08"]
    test_ids = ["09"]

    train_tensors, train_phonemes = [], []
    for id in train_ids:
        tensor_meg, df_phonemes = load(subject, id, equilibrate, freq)
        temp_tensors = [tensor_meg[:, int((df_phonemes.loc[i, 'start'] + 0.7)
                                          * 100):
                                   int((df_phonemes.loc[i, 'start'] + 0.7)
                                       * 100) + 100]
                        for i in range(len(df_phonemes))]
        train_tensors += temp_tensors
        train_phonemes += df_phonemes['phoneme'].tolist()

    valid_tensors, valid_phonemes = [], []
    for id in valid_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors = [tensor_meg[:, int((df_phonemes.loc[i, 'start'] + 0.7)
                                          * 100):
                                   int((df_phonemes.loc[i, 'start'] + 0.7)
                                       * 100) + 100]
                        for i in range(len(df_phonemes))]
        valid_tensors += temp_tensors
        valid_phonemes += df_phonemes['phoneme'].tolist()

    test_tensors, test_phonemes = [], []
    for id in test_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors = [tensor_meg[:, int((df_phonemes.loc[i, 'start'] + 0.7)
                                          * 100):
                                   int((df_phonemes.loc[i, 'start'] + 0.7)
                                       * 100) + 100]
                        for i in range(len(df_phonemes))]
        test_tensors += temp_tensors
        test_phonemes += df_phonemes['phoneme'].tolist()

    train_tensors = torch.stack(train_tensors, dim=0)
    valid_tensors = torch.stack(valid_tensors, dim=0)
    test_tensors = torch.stack(test_tensors, dim=0)

    train_tensors[:, :-19, :] *= 10**12
    valid_tensors[:, :-19, :] *= 10**12
    test_tensors[:, :-19, :] *= 10**12

    if clamp:
        print("clamping")
        train_tensors.clamp(-100, 100)
        valid_tensors.clamp(-100, 100)
        test_tensors.clamp(-100, 100)
        print("done")

    if noise:
        print("adding noise")
        train_tensors = add_noise(train_tensors)
        indices = torch.randperm(len(train_phonemes))
        temp_tensor = train_tensors[indices]
        temp_phonemes = list(getter(*indices.tolist())(train_phonemes))
        train_tensors = temp_tensor
        train_phonemes = temp_phonemes
        print("done")

    print("one hot encoding...")
    indices = [choices.index(value) for value in train_phonemes]
    train_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in valid_phonemes]
    valid_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in test_phonemes]
    test_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))
    print("done")

    print('deletion of certain rows')
    train_tensors = train_tensors[:, :-19, :]
    valid_tensors = valid_tensors[:, :-19, :]
    test_tensors = test_tensors[:, :-19, :]
    print('done')

    norm = False
    if norm:
        print('normalizing data')
        train_tensors = (
            (train_tensors - train_tensors.mean([1, 2], keepdim=True)) /
            train_tensors.std([1, 2], keepdim=True)
        )

        valid_tensors = (
            (valid_tensors - valid_tensors.mean([1, 2], keepdim=True)) /
            valid_tensors.std([1, 2], keepdim=True)
        )

        test_tensors = (
            (test_tensors - test_tensors.mean([1, 2], keepdim=True)) /
            test_tensors.std([1, 2], keepdim=True)
        )
        print("done")

    return ((train_tensors, train_phonemes),
            (valid_tensors, valid_phonemes),
            (test_tensors, test_phonemes))


def uneven_stack(tensors, resize=False):
    if not resize:
        n = len(tensors)
        output_tensor = torch.zeros(n, 325, 50)

        for i, tensor in enumerate(tensors):
            n_cols = min(tensor.size(1), 50)
            output_tensor[i, :, :n_cols] = tensor[:, :n_cols]

        return output_tensor
    else:
        transformation = transforms.Resize((325, 306), antialias=True)

        tensors_redimensionnes = []
        print("resizing")
        for tensor in tqdm(tensors):
            tensor = tensor.unsqueeze(0)
            tensor_redimensionne = transformation(tensor)

            tensors_redimensionnes.append(tensor_redimensionne)
            del tensor
        return torch.stack(tensors_redimensionnes).squeeze(1)


def add_noise(tensor):

    n_percent = 0.25
    n = int(tensor.size(0) * n_percent)

    min_change = -pixel_simga
    max_change = pixel_simga

    noise = torch.zeros(tensor[:n].shape).uniform_(min_change, max_change)

    temp = tensor.clone()
    temp[:n] *= (1 + noise)
    return temp


if __name__ == "__main__":
    data = generate_samples_All("1", False, False, True, 75)
    torch.save(data, 'new_data_highf.pth')
    del data
