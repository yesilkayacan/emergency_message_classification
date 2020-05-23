# Required libraries for ETL
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Load the messages and categories datasets and merge them. Return the resulting dataframe.
    
    Args
    ----
    messages_filepath: String
        File path for the messages dataset
    categories_filepath: String
        File path for the categories dataset

    Returns
    -------
    df: pandas.DataFrame
        Resulting dataframe from the merge of messages and categories datasets
    '''
    
    messages = pd.read_csv(messages_filepath) # load messages dataset
    categories = pd.read_csv(categories_filepath) # load categories dataset

    df = messages.merge(categories, how='inner', on=['id']) # merge the datasets
    return df
    


def clean_data(df):
    '''Cleans the data category information by assigning individual categories with
    corresponding values as columns. Filters out the duplicated messages and removes 
    them from the data. Returns the cleaned dataframe.
    
    Args
    ----
    df: pandas.DataFrame
        dataframe to be cleaned
        
    Returns
    -------
    df_clean: pandas.DataFrame
        Resulting cleaned dataframe
    '''

    ## Cleaning the category labels
    # get the propper category names
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:] # use the first row to get the categories
    category_colnames = row.apply(lambda x: x[:-2]) # clean the category labels
    categories.columns = category_colnames # assign the cleand category names to the columns

    # assign the propper value of the columns with right format
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1]) # set each value to be the last character of the string
    
        categories[column] = pd.to_numeric(categories[column]) # convert column from string to numeric

   # insert the cleaned categories into the dataframe
    df.drop('categories', axis=1, inplace=True)  # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe

    ## Filtering out duplicates
    dupes = df[['message','original']].duplicated() # find the duplicate messages
    df_clean = df.drop(df[dupes].index, axis=0) # drop duplicates
    return df_clean


def save_data(df, database_filename):
    '''Saves the passed dataframe into a SQL database (DisasterMessageTable table) in the specified path.
    
    Args
    ----
    df: pandas.DataFrame
        Dataframe which whill be saved to the database
    database_filename: String
        File path for the SQL database
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessageTable', engine, index=False)  


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