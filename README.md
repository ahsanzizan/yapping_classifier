# Yapping Classifier

The Yapping Classifier will classify your yapp either as a negative yapp or a positive yapp.

## Example Usage:

To use this model on your project, you can follow this guide

### 1. Clone / download this repository

You can download this repostiory from GitHub, or clone it:

```bash
git clone https://github.com/ahsanzizan/yapping_classifier.git
```

### 2. Install the required packages (optional)

#### If you want to contribute / run the code

After installing the repository, you have to install the required packages from the `requirements.txt` file. Or just run

```bash
pip install
```

#### If you just want to use the model

You have to install Tensorflow to load the model because its based on the Tensorflow keras model.

```bash
pip install tensorflow
```

### 3. Load the model

You can now use the model in your desired project by loading it with Tensorflow:

```py
import tensorflow as tf
import pickle
```

```py
# Load the model
model = tf.keras.models.load_model('./models/yapping_classifier_model')

# Load the tokenizer
tokenizer = None
with open('./models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

### 4. Make predictions

Now that you've loaded the model, you can now use it to make predictions as such:

```py
def preprocess_texts(texts: list[str]):
    # Use the saved tokenizer to tokenize the given text
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    return padded_sequences
```

```py
def classify_text(text: str):
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
```

```py
classify_text("<your yapping>")
```

# Author

Ahsan Awadullah Azizan - [@ahsanzizan](https://www.ahsanzizan.xyz)
