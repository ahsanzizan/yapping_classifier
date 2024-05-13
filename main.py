import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./datasets/cleaned_twitter_training.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['sentiment'], stratify=data['sentiment'])
