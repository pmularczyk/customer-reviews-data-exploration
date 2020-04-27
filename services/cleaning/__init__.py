# third party imports
import pandas as pd

# local application imports
from services.formatting import (
    remove_quotes,
    replace_umlaute,
    remove_accents,
    transform_text,
)


def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    clean_funcs = [
        remove_quotes,
        replace_umlaute,
        remove_accents,
        transform_text,
        str.strip,
        str.lower,
    ]
    for func in clean_funcs:
        df.text = df.text.apply(func)
    return df
