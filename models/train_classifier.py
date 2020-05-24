import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessageTable', engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    ''' '''
    stop_words = stopwords.words("english")
    
    text = re.sub(r"[a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in stop_words]

    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'clf__estimator__n_neighbors': [2, 4, 6]
        }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    for i in range(len(category_names)):
        accuracy_list.append(accuracy_score(Y_test[:,i], y_pred[:,i]))
        precision_list.append(precision_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        recall_list.append(recall_score(Y_test[:,i], y_pred[:,i], average='weighted'))
        
    print('Model performance:\naccuracy: {}\nprecisicion: {}\nrecall: {}'
          .format(model_performance(y_true, y_predicted)))
    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list)


def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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