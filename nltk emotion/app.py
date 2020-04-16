import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    data = pd.read_csv("ISEAR2.csv")

    def simple_split(data, y, length, split_mark=0.95):
        if split_mark > 0. and split_mark < 1.0:
            n = int(split_mark * length)
        else:
            n = int(split_mark)
        xtrain = data[:n].copy()
        xtest = data[n:].copy()
        ytrain = y[:n].copy()
        ytest = y[n:].copy()
        return xtrain, xtest, ytrain, ytest

    vectorizer = CountVectorizer()
    xtrain, xtest, ytrain, ytest = simple_split(data.text, data.emotion, len(data))
    xtrain = vectorizer.fit_transform(xtrain)
    xtest = vectorizer.transform(xtest)
    mnb = MultinomialNB()
    mnb.fit(xtrain, ytrain)
    if request.method == 'POST':
        emotion = request.form['emotion']
        data1 = [emotion]
        vect = vectorizer.transform(data1)
        my_prediction = mnb.predict(vect)
    if my_prediction=='sadness':
        return render_template('result.html', prediction=my_prediction)
    if my_prediction=='joy':
        return render_template('joy.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)




