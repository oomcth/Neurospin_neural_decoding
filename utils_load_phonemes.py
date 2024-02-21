import pandas as pd
import textgrid
import os
import matplotlib.pyplot as plt


number_to_range = {
        "01": "1-3",
        "02": "4-6",
        "03": "7-9",
        "04": "10-12",
        "05": "13-14",
        "06": "15-19",
        "07": "20-22",
        "08": "23-25",
        "09": "26-27"
    }


def map_number_to_range(number):
    return number_to_range.get(number, "error")


def extract_interval_data(interval):
    return interval.minTime, interval.maxTime, interval.mark


def get_file_phonemes(run_id):
    path = "phonemes/ch" + map_number_to_range(run_id) + ".TextGrid"
    return path


def get_phonemes(run_id):
    file = get_file_phonemes(run_id)
    assert os.path.exists(file), file
    raw_data = textgrid.TextGrid.fromFile(file)
    extracted_data = map(extract_interval_data, raw_data[-1])
    df = pd.DataFrame(extracted_data, columns=['start', 'end', 'phoneme'])
    return df


def print_histo(df_phonemes):
    mask_counts = df_phonemes['phoneme'].value_counts()

    # Créer un histogramme
    plt.figure(figsize=(8, 6))
    mask_counts.plot(kind='bar', color='skyblue')

    # Titres et labels
    plt.title('Occurrences des objets dans la colonne "mask"')
    plt.xlabel('Objets')
    plt.ylabel('Occurrences')

    # Afficher l'histogramme
    plt.show()


if __name__ == "__main__":
    print_histo(get_phonemes("01"))
