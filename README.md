# Disaster Response Pipeline Project

## Getting Started

### Required Libraries:
- pandas
- sqlalchemy
- pickle
- numpy
- Scikit-learn
- re
- nltk
- json
- plotly
- flask
- joblib

## Use Commands

Use below commands in the terminal to use the functions.

- To run ETL pipeline that cleans data and stores in database
      `$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

- To run ML pipeline that trains classifier and saves
      `$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run web app.
      `$ python run.py`

The web app can be access from the browser at http://0.0.0.0:3001/

## Data Cleaning and Preperation
### ETL

Data categories are extracted. The duplicated data are filtered from the dataset messages column. This is done in order to not process the same message twice.

### NLP Tokenization

Characters other then letters or numbers are removed. The texts are tokenized. Stopwords are removed from the token list and all the data is converted into lowercase.

### ML Pipeline

The data is split into a test and training set with a ratio of 0.2.

## Issues

It has been observed that there is high imbalance in the data. Each category only has a few real emergency cases while most other cases are non emergency for that category.

This imbalance affects the classification model training where it fits to the data in a biased way. Therefore the model quality is not at a desired level.

The best performance so far was 52% which is very poor.
