import pandas as pd
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the cleaned dataset
data = pd.read_csv('./datasets/cleaned_twitter_training.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], stratify=data['sentiment'])

# BERT preprocessor
bert_preprocess = tf_hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = tf_hub.KerasLayer(
    'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

# Layerings
# BERT layer
text_input_layer = tf.keras.layers.Input(
    shape=(), dtype=tf.string, name='Text')
preprocessed_text = bert_preprocess(text_input_layer)
pooled_output_layer = bert_encoder(preprocessed_text)['pooled_output']
# Neural network layers
layer = tf.keras.layers.Dropout(0.1, name='dropout')(pooled_output_layer)
layer = tf.keras.layers.Dense(1, name='dense')(layer)

# Constructing the actual model
model = tf.keras.Model(inputs=[text_input_layer], outputs=[layer])
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]
model.compile(optimized='adam', loss='binary_crossentropy', metrics=METRICS)

EPOCHS = 15
model.fit(X_train, y_train, epochs=EPOCHS)

# Evaluating the model
model.evaluate(X_test, y_test)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > .5, 1, 0)

cm = confusion_matrix(y_test, y_predicted)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print(f"Classification Report: {classification_report(y_test, y_predicted)}")

# Save the model
model.save('./models/yapping_classifier.keras')
print("Model saved successfully")
