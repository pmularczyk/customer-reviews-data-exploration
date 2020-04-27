# third party imports
import pandas as pd


def encode_rating(df: pd.DataFrame) -> pd.DataFrame:
    replacement_map = {
        "5": "very positive",
        "4": "positive",
        "3": "neutral",
        "2": "negative",
        "1": "very negative",
    }
    rating_encoded_df = df.copy()
    rating_encoded_df.rating = rating_encoded_df.rating.apply(str)
    rating_encoded_df.rating = rating_encoded_df.rating.map(replacement_map)
    return rating_encoded_df


def encode_sentiment_scores(score: float) -> str:
    if score >= 0.33:
        return "positive"
    if score < 0.33 and score > -0.33:
        return "neutral"
    if score <= 0.33:
        return "negative"
