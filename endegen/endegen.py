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
