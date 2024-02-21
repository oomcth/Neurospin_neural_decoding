import os
import sys

sys.path.append('neurospin-petit-prince-main/decoding/ntbk')

import utils
import datasets
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

figure = plt.figure(figsize=(32, 20), dpi=80)
fig, axes = plt.subplots(1, 1)
from dataset import analysis_subject, get_subjects, get_path, load_scores

from utils import (
    match_list,
    add_syntax,
    mne_events,
    decoding_from_criterion,
    get_code_path,
    get_path,
)

print("import ok")

# First, you have to run an analysis for set parameters:
level = 'sentence'
starts = np.atleast_1d('onset')
subject = '26'
# All the criterions used for later plotting:
# - total BOW (Bag of words)
# - embeddings (if at sentence level: LASER, if at word: spacy)
# - onlyX embeddings

criterions = ('bow', 'embeddings', 'only1', 'only2', 'only3', 'only4', 'only5')
modalities = ('visual', 'auditory')

for modality in modalities:
    for start in starts:
        for decoding_criterion in criterions:
            analysis_subject(subject, modality, start, level, decoding_criterion)


# Then you can plot the output of it
