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

sigma = 0.05
pixel_simga = 0.05
offset = 15
choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']


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

    del raw
    del meta

    return tensor_meg, equilibrate_phonemes(df_phonemes)


def equilibrate_phonemes(df):
    phoneme_counts = df['phoneme'].value_counts()

    frequent_phonemes = phoneme_counts[phoneme_counts > 100]

    max_count = frequent_phonemes.max()

    duplicated_rows = []

    for phoneme, count in frequent_phonemes.items():
        duplications_needed = max_count - count

        phoneme_rows = df[df['phoneme'] == phoneme]

        duplicated = phoneme_rows.sample(n=duplications_needed, replace=True)
        duplicated_rows.append(duplicated)

    df_balanced = pd.concat([df] + duplicated_rows, ignore_index=True)

    return df_balanced


def time_to_index(array):
    return (offset + array * 100).astype(int)


def generate_samples(n_samples, tensor_meg, df_phonemes):
    noise = np.random.normal(0, 1, 2*n_samples)
    pick = df_phonemes.sample(n=n_samples, replace=True).index
    df_phonemes = df_phonemes.reset_index()
    df_phonemes = df_phonemes.drop_duplicates(subset='index', keep='first')
    df_phonemes = df_phonemes.set_index('index')

    starts = np.array(df_phonemes.loc[pick, 'start'])
    starts += sigma * noise[:n_samples] * starts
    starts = time_to_index(starts)
    ends = np.array(df_phonemes.loc[pick, 'end'])
    ends += sigma * noise[:n_samples] * ends
    ends = time_to_index(ends)

    tensors = [torch.index_select(tensor_meg, 1, torch.arange(d, f))
               for d, f in zip(starts, ends)]

    phonemes = list(df_phonemes.loc[pick, "phoneme"])

    del starts
    del ends

    return tensors, phonemes


def generate_bag(tensor_meg, df_phonemes, freq_per_bag=0.4, n_bag=4):

    taille_tas = int(len(df_phonemes) * freq_per_bag)

    tas = []

    for _ in range(n_bag):
        bag = (tensor_meg, df_phonemes.sample(n=taille_tas, replace=True))
        tas.append(bag)
    return tas


def generate_bag_All(subject, n_samples=2_000, freq_per_bag=0.4,
                     n_bag=4, noise=True):

    train_ids = ["01", "02", "03", "04", "05", "06", "07"]
    test_ids = ["09"]
    valid_ids = ["08"]

    bags = [(torch.Tensor(), torch.Tensor()) for _ in range(n_bag)]
    for id in train_ids:
        tensor_meg, df_phonemes = load(subject, id)
        generated_bags = generate_bag(tensor_meg, df_phonemes,
                                      freq_per_bag, n_bag)
        temp = []
        for (tensor, df) in generated_bags:
            temp.append(generate_samples(n_samples, tensor, df))
            new_first_element = uneven_stack(copy(temp[-1][0]))
            temp[-1] = (new_first_element,) + temp[-1][1:]
            temp[-1][0][:, :-19, :] *= 10**12
            temp[-1][0].clamp(-100, 100)
            if noise:
                new_first_element = add_noise(temp[-1][0].clone())
                temp[-1] = (new_first_element,) + temp[-1][1:]
                indices = torch.randperm(len(temp[-1][1]))
                temp_tensor = temp[-1][0][indices]
                temp_phonemes = list(getter(*indices.tolist())(temp[-1][1]))
                temp[-1] = (temp_tensor, temp_phonemes)
            indices = [choices.index(value) for value in temp[-1][1]]
            one_hot_tensor = F.one_hot(torch.tensor(indices),
                                       num_classes=len(choices))
            temp[-1] = (temp[-1][0][:, :-19, :], one_hot_tensor)

        bags = list(map(lambda x: (torch.cat([x[0][0], x[1][0]], dim=0),
                                   torch.cat([x[0][1], x[1][1]], dim=0)),
                    zip(bags, temp)))
        del temp, tensor_meg, df_phonemes

    valid_tensors, valid_phonemes = [], []
    for id in valid_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors, temp_phonemes = generate_samples(n_samples,
                                                       tensor_meg,
                                                       df_phonemes)
        valid_tensors += temp_tensors
        valid_phonemes += temp_phonemes

    test_tensors, test_phonemes = [], []
    for id in test_ids:
        tensor_meg, df_phonemes = load(subject, id)
        temp_tensors, temp_phonemes = generate_samples(n_samples,
                                                       tensor_meg,
                                                       df_phonemes)
        test_tensors += temp_tensors
        test_phonemes += temp_phonemes

    valid_tensors = uneven_stack(valid_tensors)
    test_tensors = uneven_stack(test_tensors)

    valid_tensors[:, :-19, :] *= 10**12
    test_tensors[:, :-19, :] *= 10**12

    valid_tensors.clamp(-100, 100)
    test_tensors.clamp(-100, 100)

    indices = [choices.index(value) for value in valid_phonemes]
    valid_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in test_phonemes]
    test_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    valid_tensors = valid_tensors[:, :-19, :]
    test_tensors = test_tensors[:, :-19, :]

    return (bags, (valid_tensors, valid_phonemes),
            (test_tensors, test_phonemes))


def generate_samples_All(subject, samples_per_run, noise=True, resize=False):

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

    train_tensors = uneven_stack(train_tensors, resize)
    valid_tensors = uneven_stack(valid_tensors, resize)
    test_tensors = uneven_stack(test_tensors, resize)

    if resize:
        print("resizing done")
    print("clamping + deletion of certain rows")
    train_tensors[:, :-19, :] *= 10**12
    valid_tensors[:, :-19, :] *= 10**12
    test_tensors[:, :-19, :] *= 10**12

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

    # assert all(value in choices for value in train_phonemes)
    # assert all(value in choices for value in valid_phonemes)
    # assert all(value in choices for value in test_phonemes)

    print("one hot encoding...")
    indices = [choices.index(value) for value in train_phonemes]
    train_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in valid_phonemes]
    valid_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in test_phonemes]
    test_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))
    print("done")
    train_tensors = train_tensors[:, :-19, :]
    valid_tensors = valid_tensors[:, :-19, :]
    test_tensors = test_tensors[:, :-19, :]

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


def add_noise(tensor):

    n_percent = 0.25
    n = int(tensor.size(0) * n_percent)

    min_change = -pixel_simga
    max_change = pixel_simga

    noise = torch.zeros(tensor[:n].shape).uniform_(min_change, max_change)

    temp = tensor.clone()
    temp[:n] *= (1 + noise)
    return temp


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


def generate_train_test_from_single(subject, run_id, n_samples,
                                    train_prop, noise=True):
    df_phonemes = get_phonemes(run_id)

    raw, meta = read_raw(subject, run_id, False, "auditory")

    gap = meta['onset'][0] - df_phonemes['start'][1]

    def rectifier(x):
        return x + gap

    df_phonemes['start'] = df_phonemes['start'].apply(rectifier)
    df_phonemes['end'] = df_phonemes['end'].apply(rectifier)
    df_phonemes['delta'] = df_phonemes['end'] - df_phonemes['start']

    tensor_meg = torch.tensor(raw.get_data())

    del raw
    del meta

    lim_index = int(len(df_phonemes) * train_prop)

    ((train_tensors, train_phonemes), (test_tensors, test_phonemes)) = (
        (generate_samples(n_samples, tensor_meg,
                          df_phonemes.iloc[0:lim_index])),
        (generate_samples(int(n_samples/5), tensor_meg,
                          df_phonemes[lim_index:len(df_phonemes)]))
    )

    train_tensors = uneven_stack(train_tensors)
    test_tensors = uneven_stack(test_tensors)

    train_tensors[:, :-19, :] *= 10**12
    test_tensors[:, :-19, :] *= 10**12

    train_tensors.clamp(-100, 100)
    test_tensors.clamp(-100, 100)

    if noise:
        train_tensors = add_noise(train_tensors)
        indices = torch.randperm(len(train_phonemes))
        temp_tensor = train_tensors[indices]
        temp_phonemes = list(getter(*indices.tolist())(train_phonemes))
        train_tensors = temp_tensor
        train_phonemes = temp_phonemes

        indices = [choices.index(value) for value in train_phonemes]
    train_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    indices = [choices.index(value) for value in test_phonemes]
    test_phonemes = F.one_hot(torch.tensor(indices), num_classes=len(choices))

    train_tensors = train_tensors[:, :-19, :]
    test_tensors = test_tensors[:, :-19, :]

    return ((train_tensors, train_phonemes),
            (test_tensors, test_phonemes))


if __name__ == "__main__":
    data = generate_samples_All("1",  2_500, True, True)
    torch.save(data, 'resized0.pth')
    del data

    data = generate_samples_All("1",  2_500, True, True)
    torch.save(data, 'resized1.pth')
    del data

    data = generate_samples_All("1",  2_500, True, True)
    torch.save(data, 'resized2.pth')
    del data

    data = generate_samples_All("1",  2_500, True, True)
    torch.save(data, 'resized3.pth')
    del data
