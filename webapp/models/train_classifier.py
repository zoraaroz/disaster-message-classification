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
    """Establishes connection to data base stored in database_filepath, loads the
    table containing the message data and returns X and y dataframes containing
    predictors and labels for the ML pipeline.

    Args:
        database_filepath (string): name .db file with database

    Returns:
        X: dataframe with predictors (in this case the message column)
        y: dataframe with labels (in this case all category columns)

    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)

    # create X and y dataframes for pipeline
    X = df['message'] # predictors
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1).astype('int64') # labels
    return X, y

def tokenize(text):
    """Tokenizes input text data by applying normalization, tokenization, removal
    of stopwords and stemmatization. Returns a list of words.

    Args:
        text (string): Text to be tokenized

    Returns:
        word_list_stemmed: A list of words (strings) that are normalized, stemmed
        words that contain no stopwords

    """
    text_norm = re.sub(r'[^a-zA-Z0-9]', ' ', text) # normalize: remove punctuation
    word_list = word_tokenize(text_norm) # tokenize
    word_list_clean = [w for w in word_list if w not in stopwords.words('english')] # remove stopwords
    word_list_stemmed = [PorterStemmer().stem(w) for w in word_list_clean] # stemm words
    return word_list_stemmed


def build_model():
    """Builds a ML pipeline and model for the classifcation of the disaster messages.
    The pipeline uses a count vectorizer, a TFIDF transformer and a linear SVC
    model. A gridsearch for different parameter combinations is employed. The best
    solution is stored in the variable model, which is the output of this function.

    Args:
        None

    Returns:
        model: ML pipeline with SVC classification model with optimal parameters

    """
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
    """Evaluates the model using the dataframes X_test and y_test. The function
    prints the precision, recall and f1-score for each label.

    Args:
        model: model to be evaluated
        X_test: dataframe containing messages to be used for the test classification
        y_test: dataframe containing labels to be used for the test classification

    Returns:
        Prints precision, recall and f1-score for each label

    """
    # run prediction with test data
    y_pred = model.predict(X_test)

    # print precision, recall and f1-score
    i = 0
    for col in y_test:
        print('Evaluation for "{}": \n {} \n\n'.format(col, classification_report(y_test[col], y_pred[:,i])))
        i += 1

def save_model(model, model_filepath):
    """Saves the model as a pickle file

    Args:
        model: model to be saved
        model_filepath (string): path of the pickle file in which the model is
        stored

    Returns:
         None
         
    """
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
        # drop child_alone column because it contains no information
        model.fit(X_train, y_train.drop('child_alone', axis = 1))
        print('Elapsed time: ', time.time() - start, ' seconds')

        print('Evaluating model...')
        start = time.time()
        # drop child_alone column because it contains no information
        evaluate_model(model, X_test, y_test.drop('child_alone', axis = 1))
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
