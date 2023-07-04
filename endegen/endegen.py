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
    """Creates a StringLookup layer for converting tokens to character IDs.

    Args:
        vocab (list): A list of unique characters in the corpus.

    Returns:
        tf.keras.layers.StringLookup: The StringLookup layer.
    """
    return tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)


def create_inverse_string_lookup_layer(ids_from_chars):
    """Creates an inverse StringLookup layer for converting character IDs to tokens.

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
    """Converts character IDs to tokens using the inverse StringLookup layer.

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
    """Recovers the human-readable strings from character IDs.

    Args:
        ids_ (tf.RaggedTensor): The character IDs to be converted to human-readable strings.

    Returns:
        str: The recovered human-readable strings.
    """
    return tf.strings.reduce_join(chars_from_ids(ids_), axis=-1).numpy()


"""
Given a character, or a sequence of characters, what is the most probable next character? This is the task you're training the model to perform. The input to the model will be a sequence of characters, and you train the model to predict the output—the following character at each time step.

Since RNNs maintain an internal state that depends on the previously seen elements, given all the characters computed until this moment, what is the next character?

### Create training examples and targets

Next divide the text into example sequences. Each input sequence will contain `seq_length` characters from the text.

For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.

So break the text into chunks of `seq_length+1`. For example, say `seq_length` is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".

First use the `tf.data.Dataset.from_tensor_slices` function to convert the text vector into a stream of character indices.
"""
# Parameters
SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000


def create_dataset(text, seq_length, batch_size, buffer_size):
    """Creates a dataset of input-output pairs for training the model.

    Args:
        text (str): The input text.
        seq_length (int): The sequence length.
        batch_size (int): The batch size.

    Returns:
        tf.data.Dataset: The dataset containing input-output pairs.
    """
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))
    sequences = tf.data.Dataset.from_tensor_slices(all_ids).batch(
        seq_length + 1, drop_remainder=True
    )
    dataset = sequences.map(split_input_target)
    # Create training batches
    # Buffer size to shuffle the dataset. (TF data is designed to work with possibly infinite sequences, so
    # it doesn't attempt to shuffle the entire sequence in memory. Instead, it maintains a buffer in which
    # it shuffles elements).
    dataset = (
        dataset.shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def split_input_target(sequence):
    """Splits a sequence into input and target sequences.

    Args:
        sequence (tf.Tensor): The input sequence.

    Returns:
        tuple: A tuple containing the input and target sequences.
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# Create dataset
dataset = create_dataset(text, SEQ_LENGTH, BATCH_SIZE, BUFFER_SIZE)


# Build The Model
class CharacterLanguageModel(tf.keras.Model):
    """character-level language modeling"""
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x


def train_model(model, dataset, epochs, checkpoint_dir):
    """
    Train the model.

    Args:
        model (tf.keras.Model): The model to train.
        dataset (tf.data.Dataset): The preprocessed training dataset.
        epochs (int): Number of training epochs.
        checkpoint_dir (str): Directory to save checkpoints.

    Returns:
        The training history.
    """
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    return history


def save_model(model, output_path):
    """
    Save the trained model to a file.

    Args:
        model (tf.keras.Model): The trained model.
        output_path (str): Path to save the model file.
    """
    model.save(output_path)


"""
The simplest way to generate text with this model is to run it in a loop, and keep track of the model's internal state as you execute it.

Each time you call the model you pass in some text and an internal state. The model returns a prediction for the next character and its new state. Pass the prediction and state back in to continue generating text.

The following makes a single step prediction:
"""


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(["[UNK]"])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float("inf")] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())],
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, "UTF-8")
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


def generate_text(one_step_model, start_string, num_generate, silent=True):
    """Generate text using the trained model.

    Args:
        one_step_model (OneStep): The OneStep model for text generation.
        start_string (str): The starting string for text generation.
        num_generate (int): Number of characters to generate.

    Returns:
        The generated text as a string.
    """
    start_time = time.time()
    states = None
    next_char = tf.constant([start_string])
    result = [next_char]

    for _ in range(num_generate):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end_time = time.time()

    generated_text = result[0].numpy().decode("utf-8")

    if not silent:
        print(f"Run time: {end_time - start_time}")
        print(f"Generated Text: {generated_text}")

    return generated_text


