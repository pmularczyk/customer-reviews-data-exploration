# standard library imports
import json
import itertools
from pathlib import Path

# third party imports
import pandas as pd

# local services imports
from services.utilities import get_input_file
from services.cleaning import replace_umlaute


def load_data(filename: Path) -> pd.DataFrame:
    df = pd.read_csv(filename, encoding="latin-1", sep=";")
    df.columns = map(str.lower, df.columns)
    df.styleid = df.styleid.apply(str)
    return df


def load_stopwords(filename: Path) -> list:
    with open(filename, "r", encoding="utf-8") as file:
        stopwords = [replace_umlaute(line.strip()) for line in file]
    return stopwords


def load_sentiment_words(filename: Path) -> dict:
    with open(filename, "r") as json_file:
        sentiment_words = json.load(json_file)
    return sentiment_words


def get_list_of_nouns(filename: Path) -> list:
    list_of_words = []
    with open(filename, "r") as file:
        for line in file.readlines():
            list_of_words.append(line.split("\t")[0])

    list_of_nouns = []
    for element in list_of_words:
        if element.split("|")[1] == "NN":
            noun = element.split("|")[0]
            noun = replace_umlaute(noun.lower())
            list_of_nouns.append(noun)

    return list_of_nouns


def load_filter_nouns(filenames: list, additional_words_path: Path = None) -> list:
    if additional_words_path:
        with open(additional_words_path, "r", encoding="utf-8") as file:
            additional = [replace_umlaute(line.strip()) for line in file]
    filter_nouns = []
    for filename in filenames:
        list_of_nouns = get_list_of_nouns(filename)
        filter_nouns.append(list_of_nouns)
    all_filter_nouns = list(itertools.chain.from_iterable(filter_nouns))
    all_filter_nouns += additional
    return all_filter_nouns
