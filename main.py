# standard library imports
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor as Executor

# third party imports
import pandas as pd

# local services imports
from services.cleaning import clean_text_values
from services.dataloaders import (
    load_data,
    load_filter_nouns,
    load_sentiment_words,
    load_stopwords,
)
from services.encoding import encode_rating
from services.plotting import plot_histogram, plot_pie_chart, plot_wordcloud
from services.transformation import (
    get_average_rating_per_id,
    get_filtered_text_df,
    get_overall_sentiment_from_text_df,
    get_ratings_in_total,
    get_sentiment_from_text_df,
    get_top_ten_words_df_from_list,
)
from services.utilities import clean_text_column, get_input_file, get_output_path


def instantiate_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("./log/logfile.log")],
    )


try:
    instantiate_logger()
except FileNotFoundError:
    os.mkdir("./log")
    instantiate_logger()


def save_plot_sentiment_from_text_per_product(sentiment_df: pd.DataFrame) -> None:
    product_ids = sentiment_df.product_id.unique()
    for product_id in product_ids:
        title = f"Sentiment from text for product: {product_id}"
        output_path = get_output_path(
            "plots", f"product_{product_id}_sentiment_from_text.png"
        )
        subset_df = sentiment_df.loc[sentiment_df.product_id == product_id, :]
        plot_pie_chart(subset_df.encoded_score, output_path, title)


def save_plot_rating_distribution_per_product(df: pd.DataFrame) -> None:
    product_ids = df.product_id.unique()
    for product_id in product_ids:
        subset_df = df.loc[df.product_id == product_id, :]
        subset_rating_distribution_df = get_ratings_in_total(subset_df)
        output_path = get_output_path(
            "plots", f"product_{product_id}_rating_distribution.png"
        )
        config = {
            "x_value": "rating",
            "y_value": "count",
            "x_label": "Ratings",
            "y_label": "Count",
            "title": f"Total ratings for product: {product_id}",
        }
        plot_histogram(subset_rating_distribution_df, config, output_path)


def save_plot_overall_rating_distribution(total_rating_df: pd.DataFrame) -> None:
    output_path = get_output_path("plots", "total_rating_distribution.png")
    config = {
        "x_value": "rating",
        "y_value": "count",
        "x_label": "Ratings",
        "y_label": "Count",
        "title": "Total Ratings",
    }
    plot_histogram(total_rating_df, config, output_path)


def get_stopwords_and_nouns(nlp_resources_path: str) -> (list, list):
    stopwords_path = get_input_file(nlp_resources_path, "german_stopwords_full.txt")
    stopwords = load_stopwords(stopwords_path)
    words_resources = [
        "SentiWS_v2.0_Negative.txt",
        "SentiWS_v2.0_Positive.txt",
    ]
    noun_paths = []
    for words_resource in words_resources:
        resource_path = get_input_file(nlp_resources_path, words_resource)
        noun_paths.append(resource_path)
    additional_nouns_path = get_input_file("config", "additional_nouns.txt")
    nouns = load_filter_nouns(noun_paths, additional_nouns_path)
    return stopwords, nouns


def main():
    nlp_resources_path = "resources/nlp_resources"
    input_filename = get_input_file("resources/dataset", "bonprix.csv")
    df = load_data(input_filename)
    df = clean_text_values(df)
    logging.info(f"Initial customer data:\n {df.head()} \n")

    # TODO: USES INITIAL DF
    path_sentiment_words = get_input_file(
        nlp_resources_path, "complete_sentiment_words.json"
    )
    sentiment_words = load_sentiment_words(path_sentiment_words)
    sentiment_df = get_sentiment_from_text_df(df, sentiment_words)
    logging.info(
        f"Sentiment scores of each customers review:\n {sentiment_df.head()} \n"
    )

    # NOTE: PLOT: sentiment from text per product as pie chart
    save_plot_sentiment_from_text_per_product(sentiment_df)

    # NOTE: STATS: overall sentiment
    overall_sentiment_df = get_overall_sentiment_from_text_df(sentiment_df)
    logging.info(f"Overall sentiment from text:\n {overall_sentiment_df} \n")

    # NOTE: PLOT: overall sentiment from text as pie chart
    title = f"Overall sentiment from text"
    output_path = get_output_path("plots", "overall_sentiment_from_text.png")
    plot_pie_chart(sentiment_df.encoded_score, output_path, title)

    # NOTE: STATS: sentiment per product
    grouped_sentiment = (
        sentiment_df.groupby(["product_id", "encoded_score"])
        .size()
        .reset_index(name="counts")
    )
    logging.info(f"Sentiment from text per product:\n {grouped_sentiment} \n")

    # TODO: USES INITIAL DF
    # NOTE: STATS: avg rating per product
    avg_rating_df = get_average_rating_per_id(df)
    logging.info(f"Average rating per product:\n {avg_rating_df} \n")

    # TODO: USES INITIAL DF
    # NOTE: STATS: overall rating
    total_rating_df = get_ratings_in_total(df)
    logging.info(f"Overall rating distribution:\n {total_rating_df} \n")

    # NOTE: PLOT: overall rating distribution and rating distribution per product
    save_plot_overall_rating_distribution(total_rating_df)
    save_plot_rating_distribution_per_product(df)

    # TODO: USES INITIAL DF
    # NOTE: PLOT: rating from positive to negative as pie chart
    rating_encoded_df = encode_rating(df)
    title = f"Ratings from very positive to very negative"
    output_path = get_output_path("plots", f"total_ratings_encoded.png")
    plot_pie_chart(rating_encoded_df.rating, output_path, title)

    # TODO: USES INITIAL DF
    # NOTE: PLOT: most common words histogram and wordcloud
    stopwords, nouns = get_stopwords_and_nouns(nlp_resources_path)
    filtered_text_df = get_filtered_text_df(df, stopwords, nouns)
    words_list = clean_text_column(filtered_text_df, "text")
    top_ten_df = get_top_ten_words_df_from_list(words_list)
    logging.info(f"10 Most frequent words:\n {top_ten_df} \n")
    config = {
        "x_value": "word",
        "y_value": "count",
        "x_label": "Word",
        "y_label": "Count",
        "title": "Most frequent words",
    }
    output_path = get_output_path("plots", "most_frequent_words.png")
    plot_histogram(top_ten_df, config, output_path)
    output_path = get_output_path("plots", "wordcloud_frequent_words.png")
    plot_wordcloud(words_list, output_path)


def multiprocessed_main():
    input_filename = get_input_file("resources/dataset", "bonprix.csv")
    df = load_data(input_filename)
    df = clean_text_values(df)
    logging.info(f"Initial customer data:\n {df.head()} \n")
    nlp_resources_path = "resources/nlp_resources"
    path_sentiment_words = get_input_file(
        nlp_resources_path, "complete_sentiment_words.json"
    )
    sentiment_words = load_sentiment_words(path_sentiment_words)
    stopwords, nouns = get_stopwords_and_nouns(nlp_resources_path)

    workers = multiprocessing.cpu_count()
    with Executor(max_workers=workers) as exe:
        jobs = [
            # 0 returns sentiment df
            exe.submit(get_sentiment_from_text_df, df, sentiment_words),
            # 1 returns avg_rating_df
            exe.submit(get_average_rating_per_id, df),
            # 2 returns total rating df
            exe.submit(get_ratings_in_total, df),
            # 3 returns rating encoded df
            exe.submit(encode_rating, df),
            # 4 returns filtered text df
            exe.submit(get_filtered_text_df, df, stopwords, nouns),
        ]
        results = [job.result() for job in jobs]

    sentiment_df = results[0]
    logging.info(
        f"Sentiment scores of each customers review:\n {sentiment_df.head()} \n"
    )

    # NOTE: PLOT: sentiment from text per product as pie chart
    save_plot_sentiment_from_text_per_product(sentiment_df)

    # NOTE: STATS: overall sentiment
    overall_sentiment_df = get_overall_sentiment_from_text_df(sentiment_df)
    logging.info(f"Overall sentiment from text:\n {overall_sentiment_df} \n")

    # NOTE: PLOT: overall sentiment from text as pie chart
    title = f"Overall sentiment from text"
    output_path = get_output_path("plots", "overall_sentiment_from_text.png")
    plot_pie_chart(sentiment_df.encoded_score, output_path, title)

    # NOTE: STATS: sentiment per product
    grouped_sentiment = (
        sentiment_df.groupby(["product_id", "encoded_score"])
        .size()
        .reset_index(name="counts")
    )
    logging.info(f"Sentiment from text per product:\n {grouped_sentiment} \n")

    avg_rating_df = results[1]
    logging.info(f"Average rating per product:\n {avg_rating_df} \n")

    total_rating_df = results[2]
    logging.info(f"Overall rating distribution:\n {total_rating_df} \n")
    save_plot_overall_rating_distribution(total_rating_df)
    save_plot_rating_distribution_per_product(df)

    rating_encoded_df = results[3]
    title = f"Ratings from very positive to very negative"
    output_path = get_output_path("plots", f"total_ratings_encoded.png")
    plot_pie_chart(rating_encoded_df.rating, output_path, title)

    filtered_text_df = results[4]
    words_list = clean_text_column(filtered_text_df, "text")
    top_ten_df = get_top_ten_words_df_from_list(words_list)
    logging.info(f"10 Most frequent words:\n {top_ten_df} \n")
    config = {
        "x_value": "word",
        "y_value": "count",
        "x_label": "Word",
        "y_label": "Count",
        "title": "Most frequent words",
    }
    output_path = get_output_path("plots", "most_frequent_words.png")
    plot_histogram(top_ten_df, config, output_path)
    output_path = get_output_path("plots", "wordcloud_frequent_words.png")
    plot_wordcloud(words_list, output_path)


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    duration = time.time() - start
    logging.info(f"execution time of sequential main {duration} in seconds")

    start = time.time()
    multiprocessed_main()
    duration = time.time() - start
    logging.info(f"execution time of multiprocessed main {duration} in seconds")
