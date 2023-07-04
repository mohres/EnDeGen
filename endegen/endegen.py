import os
import time

import tensorflow as tf


def download_data(url, filename):
    """
    Downloads data from the given URL and saves it with the specified filename.

    Args:
        url (str): The URL of the data to download.
        filename (str): The name to use for saving the downloaded data.

    Returns:
        str: The path to the downloaded file.
    """
    path_to_file = tf.keras.utils.get_file(filename, url)
    return path_to_file


def read_text_from_file(path_to_file):
    """Reads text data from a file.

    Args:
        path_to_file (str): The path to the file containing the text data.

    Returns:
        str: The text content of the file.
    """
    with open(path_to_file, "rb") as file:
        text = file.read().decode(encoding="utf-8")
    return text


def display_text_length(text):
    """Displays the length of the given text.

    Args:
        text (str): The text to compute the length of.
    """
    print(f"Length of text: {len(text)} characters")


def display_n_characters(text, n):
    """Displays the first n characters of the given text.

    Args:
        text (str): The text to display the characters from.
        n (int): The number of characters to display.
    """
    print(text[:n])


def get_unique_characters(text):
    """Retrieves the unique characters present in the given text.

    Args:
        text (str): The text to compute the unique characters from.

    Returns:
        list: A list of unique characters.
    """
    return sorted(set(text))


# Download the data
url = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
filename = "shakespeare.txt"
path_to_file = download_data(url, filename)

# Read the data
text = read_text_from_file(path_to_file)

# Display the length of the text
# display_text_length(text)

# Display the first 250 characters in the text
# display_n_characters(text, 250)

# Get the number of unique characters
vocab = get_unique_characters(text)
# print(f"{len(vocab)} unique characters")


"""## Process the text

### Vectorize the text

Before training, we need to convert the strings to a numerical representation.

Using `tf.keras.layers.StringLookup` layer can convert each character into a numeric ID. 
It just needs the text to be split into tokens first.
"""


def tokenize_text(texts):
    """Tokenizes the given texts into individual characters.

    Args:
        texts (list): A list of strings to be tokenized.

    Returns:
        tf.RaggedTensor: A ragged tensor containing the tokens.
    """
    return tf.strings.unicode_split(texts, input_encoding="UTF-8")


def create_string_lookup_layer(vocab):
    """
    Creates a StringLookup layer for converting tokens to character IDs.

    Args:
        vocab (list): A list of unique characters in the corpus.

    Returns:
        tf.keras.layers.StringLookup: The StringLookup layer.
    """
    return tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)


def create_inverse_string_lookup_layer(ids_from_chars):
    """
    Creates an inverse StringLookup layer for converting character IDs to tokens.

    Args:
        ids_from_chars (tf.keras.layers.StringLookup): The StringLookup layer used for token to ID conversion.

    Returns:
        tf.keras.layers.StringLookup: The inverse StringLookup layer.
    """
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None
    )
    return chars_from_ids


def convert_text_to_ids(ids_from_chars, chars):
    """Converts tokens to character IDs using the StringLookup layer.

    Args:
        ids_from_chars (tf.keras.layers.StringLookup): The StringLookup layer for token to ID conversion.
        chars (tf.RaggedTensor): The tokens to be converted to character IDs.

    Returns:
        tf.RaggedTensor: A ragged tensor containing the character IDs.
    """
    return ids_from_chars(chars)


def convert_ids_to_text(chars_from_ids_, ids_):
    """
    Converts character IDs to tokens using the inverse StringLookup layer.

    Args:
        chars_from_ids_ (tf.keras.layers.StringLookup): The inverse StringLookup layer for ID to token conversion.
        ids_ (tf.RaggedTensor): The character IDs to be converted to tokens.

    Returns:
        tf.Tensor: A tensor containing the recovered human-readable strings.
    """
    chars = chars_from_ids_(ids_)
    text_ = tf.strings.reduce_join(chars, axis=-1)
    return text_.numpy()


ids_from_chars = create_string_lookup_layer(vocab)
chars_from_ids = create_inverse_string_lookup_layer(ids_from_chars)


def text_from_ids(ids_):
    """
    Recovers the human-readable strings from character IDs.

    Args:
        ids_ (tf.RaggedTensor): The character IDs to be converted to human-readable strings.

    Returns:
        str: The recovered human-readable strings.
    """
    return tf.strings.reduce_join(chars_from_ids(ids_), axis=-1).numpy()

