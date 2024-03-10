
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import mne_bids
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from mne.preprocessing import find_bad_channels_maxwell

sys.path.append('neurospin-petit-prince-main/decoding/ntbk')
from utils import (
    match_list,
    add_syntax,
    get_code_path,
)


def read_raw(subject, run_id, events_return=False, modality="auditory"):
    """
    Epoching function that for a subject, modality and run returns the epochs
    object which triggers have been aligned from the
    MEG metadata and STIM events file.

    Args:
        - subject: str
        - run_id: str
        - events_return: bool
        - modality: str (visual or auditory)
    Returns:
        - mne.Epochs
    """

    # path = get_path(modality)
    path = "LPP_MEG_auditory"
    task_map = {"auditory": "listen", "visual": "read", "fmri": "listen"}
    task = task_map[modality]
    print(f"\n Epoching for run {run_id}, subject: {subject}\n")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task,
        datatype="meg",
        root=path,
        run=run_id
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.del_proj()  # To fix proj issues
    raw.pick_types(meg=True, stim=True)

    # Generate event_file path
    event_file = path + "/" + f"sub-{bids_path.subject}"
    event_file = event_file + "/" + f"ses-{bids_path.session}"
    event_file = event_file + "/" + "meg"
    event_file = str(event_file + "/" + f"sub-{bids_path.subject}")
    event_file += f"_ses-{bids_path.session}"
    event_file += f"_task-{bids_path.task}"
    event_file += f"_run-{bids_path.run}_events.tsv"

    assert Path(event_file).exists()

    # read events
    meta = pd.read_csv(event_file, sep="\t")
    events = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

    if modality == "auditory":
        meta["word"] = meta["trial_type"].apply(
            lambda x: eval(x)["word"] if type(eval(x)) == dict else np.nan
        )
    # Initial wlength, as presented in the stimuli / triggers to match list
    meta["wlength"] = meta.word.apply(len)
    meta["run"] = run_id
    # Enriching the metadata with outside files:
    # path_syntax = get_code_path() / "data/syntax"

    # testing new syntax
    path_syntax = get_code_path() + "/" + "data" + "/" + "syntax_new_no_punct"

    # Send raw metadata
    # meta = add_new_syntax(meta, path_syntax, int(run_id))
    meta = add_syntax(meta, path_syntax, int(run_id))

    # add sentence and word positions
    meta["sequence_id"] = np.cumsum(meta.is_last_word.
                                    shift(1, fill_value=False))
    for s, d in meta.groupby("sequence_id"):
        meta.loc[d.index, "word_id"] = range(len(d))

    # Making sure that there is no problem with words that contain ""
    meta.word = meta.word.str.replace('"', "")

    # Two cases for match list: is it auditory or visual ?
    if modality == "auditory":
        word_events = events[events[:, 2] > 1]
        meg_delta = np.round(np.diff(word_events[:, 0] / raw.info["sfreq"]))
        meta_delta = np.round(np.diff(meta.onset.values))
        i, j = match_list(meg_delta, meta_delta)
        assert len(i) > 1
        # events = events[i]  # events = words_events[i]

    meta["has_trigger"] = False
    meta.loc[j, "has_trigger"] = True

    # integrate events to meta for simplicity
    meta.loc[j, "start"] = events[i, 0] / raw.info["sfreq"]

    # preproc raw
    raw.load_data()
    raw = raw.filter(5, 200)

    if events_return:
        return raw, meta, events[i]
    else:
        return raw, meta


if __name__ == "__main__":
    raw, _, _ = read_raw('7', '05', True, "auditory")

    # raw_sss = mne.preprocessing.maxwell_filter(raw)
    # auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    #     raw,
        # cross_talk=crosstalk_file,
        # calibration=fine_cal_file,
    #     return_scores=True,
    #     verbose=True,
    # )
    # print(auto_noisy_chs)  # we should find them!
    # print(auto_flat_chs)  # none for this dataset
    # input()
    raw.compute_psd(fmax=200).plot(picks="data", exclude="bads")
    raw.plot(duration=0.25, n_channels=5)
    input()
