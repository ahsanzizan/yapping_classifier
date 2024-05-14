import logging
import pickle
import tensorflow as tf
import utils

logging.getLogger('tensorflow').setLevel(logging.ERROR)

# EXAMPLE USAGE OF THE YAPPING CLASSIFIER MODEL

# Load the model
model = tf.keras.models.load_model('./models/yapping_classifier_model')

# Load the tokenizer
tokenizer = None
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load stemmer
stemmer = None
with open('./models/stemmer.pickle', 'rb') as handle:
    stemmer = pickle.load(handle)

if __name__ == '__main__':
    input_text = input("Your yapp > ")
    confidence = utils.classify_yapp(model, tokenizer, stemmer, input_text)
    print(
        f"Negative Confidence: {confidence['negative'] * 100:.3f}%\nPositive Confidence: {(confidence['positive']) * 100:.3f}%")
    classification = max(confidence, key=confidence.get)
    print(
        f"The yapp is classified as a {classification} yapp")
