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

if __name__ == '__main__':
    input_text = input("Your yapp > ")
    print(
        f"The yapp is classified as a {utils.classify_text(model, tokenizer, input_text)} yapp")
