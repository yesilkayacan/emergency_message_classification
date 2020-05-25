import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as gobj
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''Function normalizes the text input, removes non letter or number characters, tokenizes the text.
    Removes the stopword tokens and lemmatizes the result.
    
    Args
    ----
    text: String
        Text data which will be processed
        
    Returns
    -------
    clean_tokens: List (String)
        Normalized, cleaned, tokenized and lemmatized list of words
    '''

    stop_words = stopwords.words("english")
    
    text = re.sub(r"[a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens if tok not in stop_words]
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessageTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visual1
    graphs = [
        {
            'data': [
                gobj.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    ## second plot work
    # extract data for visuals
    categrories = list(df.columns[4:])
    n_data_in_categories = df[categrories].sum(axis=0)

    # create visual2
    graphs.append(
        {
            'data': [
                gobj.Bar(
                    x=categrories,
                    y=n_data_in_categories
                )
            ],

            'layout': {
                'title': 'Distribution of Emergency Categories in Training Data',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Emergency Categories",
                    'tickangle': "-45"
                }
            }
        }
    )

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
