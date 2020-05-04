# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load Data from the Database    
    Input:
        database_filepath -> Path to SQLite destination database 
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> A list conatining categories name
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('clean_df', con=engine)
    X=df['message']
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    return X,Y,category_names


def tokenize(text):
    """
    tokenize() : a function to tokenize text strings
    Input:
    text: string. text to tokenize.
    output::
    clean_tokens --- list of clean text after tokenization, lemmatization etc.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    build_model() : to build ML model 
    Input:
        None
    output:
        cv --- final model after doing grid search
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'clf__estimator__n_estimators': [10,50],
                  'clf__estimator__min_samples_split': [2, 4]
                  
                 }
    cv = GridSearchCV(pipeline, parameters,n_jobs=1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model() : to test the effecincy of our model and print the f1_score, precision and recall results
    Input:
        model: ML model that was built in build_model()
        X_test: the features dataframe
        Y_test: the target dataframe
        category_names: list of categories names
    Output:
        print the f1_score, precision and recall results
    
    """
    y_pred = model.predict(X_test)
    for n, category in enumerate(category_names):
        print(category, classification_report(Y_test.iloc[:,n],y_pred[:,n]))


def save_model(model, model_filepath):
    """
    save_model(): to save trained model as a pickle file
    Input:
        model: ML model that was trained
        model_filepath: the filepath where the trained model will be saved
    Output:
        None
    """
    pickle.dump(model, open('model_filepath', 'wb'))


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