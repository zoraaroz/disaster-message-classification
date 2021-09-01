# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import sys
import time

# nltk
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)

    # create X and y dataframes for pipeline
    X = df['message'] # predictors
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1).astype('int64') # labels
    return X, y

def tokenize(text):
    text_norm = re.sub(r'[^a-zA-Z0-9]', ' ', text) # normalize: remove punctuation
    word_list = word_tokenize(text_norm) # tokenize
    word_list_clean = [w for w in word_list if w not in stopwords.words('english')] # remove stopwords
    word_list_stemmed = [PorterStemmer().stem(w) for w in word_list_clean] # stemm words
    return word_list_stemmed


def build_model():
    # build pipeline with count vecotrizer, tfidf and support vector machine
    pipeline_SVC = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('multi-clf', MultiOutputClassifier(LinearSVC()))
    ])

    # define parameters for gridsearch
    parameters_SVC = {
        'vect__max_df': (.6, 1),
        'tfidf__norm': ('l1', 'l2'),
        'multi-clf__estimator__C': (.1, 1, 100)
    }

    # build parameter grid and fit data
    model = GridSearchCV(pipeline_SVC, parameters_SVC)

    return model


def evaluate_model(model, X_test, y_test):
    # run prediction with test data
    y_pred = model.predict(X_test)

    # print precision, recall and f1-score
    i = 0
    for col in y_test:
        print('Evaluation for "{}": \n {} \n\n'.format(col, classification_report(y_test[col], y_pred[:,i])))
        i += 1

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)

        # split dataset into train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        start = time.time()
        model = build_model()
        print('Elapsed time: ', time.time() - start, ' seconds')

        print('Training model...')
        start = time.time()
        model.fit(X_train, y_train.drop('child_alone', axis = 1))
        print('Elapsed time: ', time.time() - start, ' seconds')

        print('Evaluating model...')
        start = time.time()
        evaluate_model(model, X_test, y_test)
        print('Elapsed time: ', time.time() - start, ' seconds')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
