import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data() will take message and categories files filepath and return a Dataframe 
    Input:
        messages_filepath - filepath for the message csv file
        categories_filepath - filepath for the categories csv file
    output:
        df - merged dataframe by 'id' column
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories dataset
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    clean_data() will take dataframe and return it after cleaning 
    Input:
        df - Merged dataset 
    output:
        df - Returns the cleaned dataframe.
    
    '''
    # clean categories column in df
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # extract columns from first row and assign them to categories df
    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x[:-2], row)) 
    categories.columns = category_colnames
    
    # convert category values to 0s or 1s from the data
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    
        
    # drop duplicates
    df.drop_duplicates(inplace=True)
  
    return df


def save_data(df, database_filename):
    '''
    save_data() will take the cleaned dataframe and save it into a database
    Input:
        df - cleaned dataframe
        database_filename - filename of the database that the dataframe will be saved into 
    output:
        None
    '''
    # save the dataframe in SQLite
    # create the SQLite engine
    engine = create_engine('sqlite:///'+database_filename)
    # Write the Dataframe into Disaster_Msg table
    df.to_sql('df_clean', engine, index=False, if_exists='replace')
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()