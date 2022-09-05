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
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



def load_data(database_filepath):
    '''
    load_data
    Load data from sqllite DB into a single pandas dataframe
    
    Input:
    database_filename       file path to custom sqllite db file
    
    Returns:
    X                       features (messages)
    y                       36 columns for multi-class classification
    column_names            36 column names for classification report purposes
    
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', engine)
    df = df.fillna(0)
    
    X = df['message']#.get_values()
    y = df.iloc[:, 4:]#.get_values()
    
    # convert 2 to 1 for 'related' category (has values 0,1,2)
    # convert non-binary classes to binary
    y['related'] = y['related'].replace(2, 1)

    column_names = list(df.columns[4:])
    return X, y, column_names


def tokenize(text):
    '''
    tokenize
    tokenize into words and remove stop words
    
    Input:
    text                   raw text
    
    Returns:
    clean_tokens           no stop words and lemmatized words
    
    '''
    words = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = list()
    for word in words:
        if word not in stop_words:
            clean_tokens.append(lemmatizer.lemmatize(word).lower().strip())
    
    return clean_tokens


def build_model():
    '''
    build_model
    pipeline for count vectorization, tfidf and ML model for multi-class classification
    
    Input:
    None                    raw text
    
    Returns:
    cv                      pipeline with GridSearchCV
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    
    
    parameters = {
    'model__estimator__n_estimators': [50,  100]
    }
    
    '''
    # more params for GridSearch - takes few hours
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
    '''
    evaluate_model
    predict on test data and print classification report
    
    Input:
    model                   pipeline with grid search
    X_test                  test features (messages)
    Y_test                  test labels (36 classes/categories)
    category_names          36 category names for classification report
    
    Returns:
    None                    
    
    '''
    y_pred = model.predict(X_test)
    for col_index, column_name in enumerate(Y_test):
        print(f"column_name={column_name} col_index={col_index}")
        print(classification_report(Y_test[column_name], y_pred[:, col_index]))



def save_model(model, model_filepath):
    '''
    save_model
    save model binary file as pkl for online inferencing
    
    Input:
    model                   pipeline with GridSearchCV
    model_filepath          local filepath to save model
    
    
    Returns:
    None                    raw text
    
    '''
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