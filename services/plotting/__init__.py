# standard library imports
from pathlib import Path

# third party imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud


def plot_pie_chart(series: pd.Series, output_path: Path, title: str) -> None:
    labels = series.value_counts().index
    sizes = series.value_counts().values
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, orientation="landscape")
    plt.close()


def plot_histogram(data: pd.DataFrame, config: dict, output_path: Path) -> None:
    if config:
        x_value = config.get("x_value", "")
        y_value = config.get("y_value", "")
        x_label = config.get("x_label", "")
        y_label = config.get("y_label", "")
        title = config.get("title", "")
    else:
        raise ValueError("No plotting config provided")
    sns.set()
    _ = plt.figure(figsize=(10, 6))
    _ = plt.xticks(rotation=45, fontsize=12)
    _ = plt.xlabel(x_label, fontsize=20)
    _ = plt.ylabel(y_label, fontsize=20)
    _ = plt.title(title, fontsize=30)
    _ = sns.barplot(x_value, y=y_value, data=data)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, orientation="landscape")
    plt.close()


def plot_wordcloud(data: list, output_path: Path) -> None:
    wordcloud = WordCloud(
        width=1600, height=800, background_color="white", collocations=False
    ).generate_from_text(" ".join(data))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    wordcloud.to_file(output_path)
    plt.close()
