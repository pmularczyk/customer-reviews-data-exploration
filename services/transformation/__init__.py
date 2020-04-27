# standard library imports
import re
from collections import Counter

# third party imports
import pandas as pd

# local services imports
from services.encoding import encode_sentiment_scores
from services.utilities import get_sentiment, filter_words

# NOTE: not really needed
# or rather only for nicer annotations while plotting
def reformat_product_ids(df: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(list(set(df.styleid.values)))
    mapping = {ids[i]: "article No.{}".format(i + 1) for i in range(len(ids))}
    df.styleid = df.styleid.map(mapping)
    return df


def get_average_rating_per_id(df: pd.DataFrame) -> pd.DataFrame:
    avg_rating_per_id = round(df.groupby("styleid").rating.mean(), 2)
    avg_rating = avg_rating_per_id.values.tolist()
    ids = sorted(list(set(df.styleid.values)))
    avg_rating_df = pd.DataFrame({"article": ids, "rating": avg_rating})
    return avg_rating_df


def get_ratings_in_total(df: pd.DataFrame) -> pd.DataFrame:
    total_ratings_df = df.drop("text", axis=1)
    total_ratings_df = total_ratings_df.groupby("rating").count()
    total_ratings_df.columns = ["count"]
    total_ratings_df = total_ratings_df.reset_index()
    return total_ratings_df


def get_sentiment_from_text_df(df: pd.DataFrame, sentiment_words: dict) -> pd.DataFrame:
    sentiment_df = df.copy()
    sentiment_df.text = sentiment_df.text.apply(lambda val: re.sub(r"\.", "", val))
    sentiment_df["tokenized"] = sentiment_df.text.str.split()
    sentiment_df["score"] = sentiment_df.tokenized.apply(
        get_sentiment, args=(sentiment_words,)
    )
    sentiment_df["encoded_score"] = sentiment_df.score.apply(encode_sentiment_scores)
    sentiment_df = sentiment_df[["styleid", "encoded_score"]]
    return sentiment_df


def get_overall_sentiment_from_text_df(sentiment_df: pd.DataFrame):
    overall_sentiment_df = sentiment_df.encoded_score.value_counts().reset_index()
    overall_sentiment_df.columns = ["sentiment", "count"]
    return overall_sentiment_df


def get_filtered_text_df(
    df: pd.DataFrame, stopwords: list, nouns: list
) -> pd.DataFrame:
    filtered_text_df = df.copy()
    filtered_text_df.text = filtered_text_df.text.apply(
        lambda val: re.sub(r"\.", "", val)
    )
    filtered_text_df["tokenized"] = filtered_text_df.text.str.split()
    filtered_text_df.text = filtered_text_df.tokenized.apply(
        filter_words, args=(stopwords, nouns)
    )
    filtered_text_df = filtered_text_df[["styleid", "text"]]
    return filtered_text_df


def get_top_ten_words_df_from_list(text_list: list) -> pd.DataFrame:
    top_ten_words = Counter(text_list).most_common(10)
    top_ten_words_df = pd.DataFrame(top_ten_words, columns=["word", "count"])
    return top_ten_words_df
