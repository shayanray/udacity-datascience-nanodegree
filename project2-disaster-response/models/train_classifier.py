import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])



def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    df = df.fillna(0)
    
    X = df['message']#.get_values()
    y = df.iloc[:, 4:]#.get_values()
    column_names = list(df.columns[4:])
    return X, y, column_names


def tokenize(text):
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = list()
    for word in words:
        clean_tokens.append(lemmatizer.lemmatize(word).lower().strip())
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    
    
    parameters = {
    'model__estimator__n_estimators': [50,  100]
    }
    '''
    parameters = {
    'tfidf__use_idf': (True, False),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'model__estimator__n_estimators': [50, 100, 200],
    'model__estimator__min_samples_split': [2, 3, 4]
    
    }
    '''

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5, cv=2, n_jobs=2)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for col_index, column_name in enumerate(Y_test):
        print(f"column_name={column_name} col_index={col_index}")
        print(classification_report(Y_test[column_name], y_pred[:, col_index]))
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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