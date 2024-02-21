import pandas as pd
import textgrid


number_to_range = {
        1: "1-3",
        2: "4-6",
        3: "7-9",
        4: "10-12",
        5: "13-15",
        6: "15-19",
        7: "20-22",
        8: "23-25",
        9: "26-27"
    }


def map_number_to_range(number):
    return number_to_range.get(number, "error")


def get_syntax(file):
    """
    Add the syntactic information to the existing metadata
    Feed the existing: word information with
    n_closing, number of closing nodes
    is_last_word, boolean value
    pos, the position in the sentence

    Args:
        - file: str
    Returns:
        - pd.DataFrame

    """

    with open(file, "r") as f:
        txt = f.readlines()

    # parse syntactic trees
    out = []
    for sequence_id, sent in enumerate(txt):
        splits = sent.split("=")

        for prev, token in zip(splits, splits[1:]):
            out.append(
                dict(
                    pos=prev.split("(")[-1].split()[0],
                    word_id=int(prev.split()[-1]),
                    word=token.split(")")[0],
                    n_closing=token.count(")"),
                    sequence_id=sequence_id,
                    is_last_word=False,
                )
            )
        out[-1]["is_last_word"] = True

    synt = pd.DataFrame(out)

    # add deal with apostrophe
    out = []
    for sent, d in synt.groupby("sequence_id"):
        for token in d.itertuples():
            for tok in token.word.split("'"):
                out.append(dict(word=tok, n_closing=1, is_last_word=False, pos="XXX"))
            out[-1]["n_closing"] = token.n_closing
            out[-1]["is_last_word"] = token.is_last_word
            out[-1]["pos"] = token.pos
    return pd.DataFrame(out)


def extract_interval_data(interval):
    return interval.minTime, interval.maxTime, interval.mark


def get_file_phonemes(run_id):
    path = "phonemes/ch" + map_number_to_range(run_id) + ".TextGrid"
    return path


def get_phonemes(file):
    raw_data = textgrid.TextGrid.fromFile(file)
    extracted_data = map(extract_interval_data, raw_data[-1])
    df = pd.DataFrame(extracted_data, columns=['start', 'end', 'phoneme'])
    print(df.head(10))


get_phonemes(get_file_phonemes(2))
