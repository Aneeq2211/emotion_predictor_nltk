import numpy as np                                          #importing neccessary lib
from flask import Flask, request, jsonify, render_template  #used flask for server
import pickle
from sklearn.externals import joblib
app = Flask(__name__)
@app.route('/')     #redirects to the main page when app is started
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])     #this function predicts the emotion based on input
def predict():
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    data = pd.read_csv("ISEAR2.csv")            #training dataset is loaded
    def simple_split(data, y, length, split_mark=0.95): #95% data is used to train the model
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
        my_prediction = mnb.predict(vect)       #here the predicted emotion is stored
    if my_prediction == 'sadness':              #these if statements redirect user to web pages according to their predicted emotions.
        return render_template('sadness.html', prediction=my_prediction)
    if my_prediction == 'joy':
        return render_template('joy.html', prediction=my_prediction)
    if my_prediction == 'anger':
        return render_template('anger.html', prediction=my_prediction)
    if my_prediction == 'disgust':
        return render_template('anger.html', prediction=my_prediction)
    if my_prediction == 'fear':
        return render_template('fear.html', prediction=my_prediction)
    if my_prediction == 'guilt':
        return render_template('shame.html', prediction=my_prediction)
    if my_prediction == 'shame':
        return render_template('shame.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)




