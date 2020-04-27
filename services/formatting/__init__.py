# standard library imports
import re
from string import printable
from unicodedata import normalize

# local application imports
from services.utilities import pipeline


def remove_accents(text: str) -> str:
    list_of_chars = list()
    for char in normalize("NFKD", text):
        if char in printable:
            list_of_chars.append(char)
    result = "".join(list_of_chars)
    return result


def remove_quotes(text: str) -> str:
    text = text.replace('"', "")
    text = text.replace("'", "")
    return text


def generate_replacement_map():
    replacement_map = {
        u"ä": "ae",
        u"Ä": "Ae",
        u"ö": "oe",
        u"Ö": "Oe",
        u"ü": "ue",
        u"Ü": "Ue",
        u"ß": "ss",
    }
    return replacement_map


def replace_umlaute(text: str) -> str:
    for key, value in generate_replacement_map().items():
        text = text.replace(key, value)
    return text


def get_first_string_element(text: str) -> str:
    removed_non_word_chars = re.sub(r"\W", " ", text)
    first_string_element = removed_non_word_chars.split()[0].lower()
    return first_string_element


def remove_newline_chars(text: str) -> str:
    return re.sub(r"\n", "", text)


def remove_special_chars(text: str) -> str:
    return re.sub(r"[,|():]", "", text)


def replace_dashes(text: str) -> str:
    return re.sub(r"-", " ", text)


def normalize_thousands_range_numbers(text: str) -> str:
    return re.sub(r"(\d)(\.)(\d)", r"\1\3", text)


def replace_dot_from_date(text: str) -> str:
    return re.sub(r"(\s\d{1,2})(\.)", r"\1 ", text)


def replace_forward_slash(text: str) -> str:
    return re.sub(r"/", " ", text)


def replace_multiple_dots(text: str) -> str:
    return re.sub(r"\.{3}", ". ", text)


def remove_trailing_dot(text: str) -> str:
    return re.sub(r"\s\.\s", " ", text)


def remove_headline_within_article(text: str) -> str:
    return re.sub(r"\+{2,3}.*?\+{2,3}", "", text)


def replace_sentence_ending_with_dot(text: str) -> str:
    return re.sub(r"[\!\?\;]", ".", text)


def remove_dot_from_abbreviated_names(text: str) -> str:
    return re.sub(r"(\s\w)(\.)", r"\1", text)


def replace_multiple_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def transform_text(text: str) -> str:
    functions = [
        remove_newline_chars,
        remove_special_chars,
        replace_dashes,
        normalize_thousands_range_numbers,
        replace_dot_from_date,
        replace_forward_slash,
        replace_multiple_dots,
        remove_trailing_dot,
        replace_sentence_ending_with_dot,
        remove_dot_from_abbreviated_names,
        replace_multiple_spaces,
    ]
    transformed_text = pipeline(value=text, function_pipeline=functions)
    return transformed_text
