import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

# load in pickled model
model = pickle.load(open('wiki_movies_model.p', 'rb'))

# generate analogies
def analogy(x1, x2, y1):
# Find the vector y2 that is closest to $-x1 + x2 + y1$.
    y2 = model.most_similar(positive = [x2, y1], negative = [x1])
# Return the result.
    return [i[0] for i in y2[0:10]]

app = Flask(__name__)

# route 1: hello world
@app.route('/')
def home():
    return 'Hello, world!'

# route 2: show a form to the user
@app.route('/form')
def form():
    return render_template('basic_form.html')

# route 3: accept the form submission and do something fancy with it
@app.route('/submit')

def make_predictions():
    # load in form data
    user_input = request.args
    print(user_input)

    data = [
        str(user_input['Word 1a']),
        str(user_input['Word 1b']),
        str(user_input['Word 2a'])
    ]

    analogies = analogy(data[0], data[1], data[2])

    return render_template('basic_results.html', analogies=analogies)

    # return render_template('results.html',
    #                         pred_class=pred_class,
    #                         proba=proba)


if __name__ == '__main__':
    app.run(debug=True)
