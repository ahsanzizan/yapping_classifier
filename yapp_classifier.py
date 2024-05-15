import pickle
import re

import tensorflow as tf
from nltk.corpus import stopwords


class YappClassifier:
    def __init__(self, model_path: str, tokenizer_path: str, stemmer_path: str, vocab_size: int = 1000,
                 embedding_dim: int = 16, max_length: int = 20, truncating_type: str = 'post',
                 padding_type: str = 'post', oov_token: str = "<OOV>"):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.truncating_type = truncating_type
        self.padding_type = padding_type
        self.oov_token = oov_token

        try:
            # Load the stopwords
            self.stop_words = stopwords.words("english")
        except Exception as e:
            raise RuntimeError("Error loading stopwords: {}".format(e))

        try:
            # Load the tokenizer
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tokenizer file not found at path: {tokenizer_path}")
        except pickle.UnpicklingError:
            raise ValueError(
                f"Error unpickling tokenizer file at path: {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading tokenizer: {e}")

        try:
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at path: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

        try:
            # Load the tokenizer
            with open(stemmer_path, 'rb') as handle:
                self.stemmer = pickle.load(handle)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Stemmer file not found at path: {tokenizer_path}")
        except pickle.UnpicklingError:
            raise ValueError(
                f"Error unpickling stemmer file at path: {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading stemmer: {e}")

    def clean_yapp(self, yapp: str) -> str:
        try:
            # Casefolding and remove extra spaces
            output = yapp.lower().strip()

            # Remove extra spaces in between
            output = re.sub(' +', ' ', output)

            # Remove @username
            output = re.sub('@\w+', ' ', output)

            # Remove punctuations
            output = re.sub('[^a-zA-Z]', ' ', output)

            # Remove all words with only one char in it
            output = re.sub(r'\b\w\b', '', output)

            # Remove stopwords and apply stemming
            output = ' '.join(self.stemmer.stem(word)
                              for word in output.split() if word not in self.stop_words)

            return output
        except Exception as e:
            raise ValueError(f"Error cleaning text: {e}")

    def preprocess_yapps(self, yapps: list[str]):
        try:
            cleaned_yapps = [self.clean_yapp(yapp) for yapp in yapps]
            sequences = self.tokenizer.texts_to_sequences(cleaned_yapps)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, maxlen=self.max_length, padding=self.padding_type, truncating=self.truncating_type)
            return padded_sequences
        except Exception as e:
            raise ValueError(f"Error preprocessing texts: {e}")

    def classify_yapp(self, yapp: str) -> dict:
        try:
            preprocessed_yapp = self.preprocess_yapps([yapp])

            prediction = self.model.predict(preprocessed_yapp)

            # Calculate confidence
            prediction = prediction.flatten()
            negative_confidence = 1 - prediction[0]
            confidence = {
                "negative": negative_confidence,
                "positive": 1 - negative_confidence,
            }

            return confidence
        except Exception as e:
            raise RuntimeError(f"Error classifying yapp: {e}")
