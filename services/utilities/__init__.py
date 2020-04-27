# standard library imports
import os
import itertools
import functools
from pathlib import Path
from typing import Callable, Sequence, TypeVar

# third party imports
import pandas as pd


def get_input_file(folder: str, file: str) -> Path:
    project_dir = Path(os.path.dirname(__package__)).absolute()
    resources_folder = project_dir.joinpath(folder)
    input_file = resources_folder.joinpath(file)
    return input_file


def get_output_path(folder: str, file: str, create_file: bool = False) -> Path:
    project_dir = Path(os.path.dirname(__package__)).absolute()
    resources_folder = project_dir.joinpath("out").joinpath(folder)
    if not os.path.exists(resources_folder):
        os.mkdir(resources_folder)
        print(f"Creating folder with name {resources_folder}")
    output_path = resources_folder.joinpath(file)
    return output_path


def clean_text_column(df: pd.DataFrame, col: str) -> list:
    text_list = df[col].to_list()
    filtered_text = list(filter(None, text_list))
    filtered_text_list = list(itertools.chain.from_iterable(filtered_text))
    return filtered_text_list


def get_sentiment(words: list, sentiment_words: dict) -> list:
    scores = []
    for word in words:
        score = sentiment_words.get(word, {}).get("score", 0.0)
        scores.append(score)
    sentiment = sum(scores)
    return sentiment


def filter_words(words: list, stopwords: list, nouns: list) -> list:
    filtered = [word for word in words if word not in stopwords]
    filtered = [word for word in filtered if word in nouns]
    return filtered


T = TypeVar("T")


def pipeline(value: T, function_pipeline: Sequence[Callable[[T], T]],) -> T:
    """A generic Unix-like pipeline
    :param value: the value you want to pass through a pipeline
    :param function_pipeline: an ordered list of functions that
        comprise your pipeline
    """
    return functools.reduce(lambda val, func: func(val), function_pipeline, value)
