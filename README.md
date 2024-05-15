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

#### - Import the `yapp_classifier.py`

The `yapp_classifier.py` file comes with several useful functions:

1. `clean_yapp` for cleaning the input yapp
2. `preprocess_yapps` for pre-processing the yapps so that the model understands what you're yapping about
3. `classify_yapp` for classifying the yapp as either a negative or positive yapp, will return a confidence dictionary

#### - Initialize the model

```py
from yapp_classifier import YappClassifier

yapp_classifier = YappClassifier(model_path='./models/yapping_classifier_model',
                                 tokenizer_path='./models/tokenizer.pickle',
                                 stemmer_path='./models/stemmer.pickle')

input_text = "I love you so much that even the moon knows"

# Output: '{'positive': 0.99788, 'negative': 0.00212}'
yapp_classifier.classify_yapp(input_text)
```

# Author

Ahsan Awadullah Azizan - [@ahsanzizan](https://www.ahsanzizan.xyz)
