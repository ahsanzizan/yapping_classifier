import re
import pickle

import tensorflow as tf
from nltk.corpus import stopwords

# Hyperparameters
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 20
TRUNCATING_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"

# Preprocessing
stop_words = stopwords.words("english")


def clean_yapp(stemmer, text: str):
    # Casefolding and remove extra spaces
    output = text.lower().strip()

    # Remove extra spaces in between
    output = re.sub(' +', ' ', output)

    # Remove @username
    output = re.sub('@\w+', ' ', output)

    # Remove punctuations
    output = re.sub('[^a-zA-Z]', ' ', output)

    # Remove all words with only one char in it
    output = re.sub('\b\w\b', '', output)

    # Remove stopwords and stemming
    output = ' '.join(stemmer.stem(word)
                      for word in output.split() if word not in stop_words)

    return output


def preprocess_yapps(tokenizer, stemmer, texts: list[str]):
    # Use the saved tokenizer to tokenize the given text
    cleaned_texts = [clean_yapp(stemmer, text) for text in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    return padded_sequences


def classify_yapp(model, tokenizer, stemmer, text: str):
    # Pre-processed texts must be a list for the model to consume
    preprocessed_text = preprocess_yapps(tokenizer, stemmer, [text])
    # Use the saved model to make a prediction on the Pre-processed text
    prediction = model.predict(preprocessed_text)

    prediction = prediction.flatten()
    negative_confidence = 1 - prediction[0]
    confidence = {
        "negative": negative_confidence,
        "positive": 1 - negative_confidence,
    }

    return confidence
