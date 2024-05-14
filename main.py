import tensorflow as tf
import pickle

tf.get_logger().setLevel(level='ERROR')  # Avoid printing warnings

# EXAMPLE USAGE OF THE YAPPING CLASSIFIER MODEL

# Load the model
model = tf.keras.models.load_model('./models/yapping_classifier_model')

# Load the tokenizer
tokenizer = None
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Hyperparameters
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 20
TRUNCATING_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"


def preprocess_texts(texts: list[str]):
    # Use the saved tokenizer to tokenize the given text
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    return padded_sequences


def classify_text(model, text: str):
    # Pre-processed texts must be a list for the model to consume
    preprocessed_text = preprocess_texts([text])
    # Use the saved model to make a prediction on the Pre-processed text
    prediction = model.predict(preprocessed_text)

    prediction = prediction.flatten()
    negative_confidence = 1 - prediction[0]
    print(
        f"Negative Confidence: {negative_confidence * 100:.3f}%\nPositive Confidence: {(1 - negative_confidence) * 100:.3f}")

    # Classify the text as either Negative or Positive
    return "Positive" if prediction >= .5 else "Negative"


if __name__ == '__main__':
    input_text = input("Input text > ")
    print(
        f"The text is classified as a {classify_text(model, input_text)} text")
