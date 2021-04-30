import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import flask
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)
tfidfVector=TfidfVectorizer(stop_words='english',max_df=0.7)
countVector = CountVectorizer(stop_words='english')
load_model = pickle.load(open('detectionModel.pkl','rb'))
df = pd.read_csv('NewsDataset.csv',header=0)
x = df['text']
y = df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.35, random_state=7)

def fakeNewsDetection(var):
    # tfidfVector_train=tfidfVector.fit_transform(x_train.values.astype('U')) 
    # tfidfVector_test=tfidfVector.transform(x_test.values.astype('U'))
    # countTrain = countVector.fit_transform(x_train)
    prediction = load_model.predict([var])
    return prediction

@app.route('/')
def home():
    return render_template('DetectionView.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fakeNewsDetection(message)
        print(pred)
        return render_template('DetectionView.html', prediction=pred)
    else:
        return render_template('index.html', prediction="ERROR")


if __name__ == '__main__':
    app.run(debug=True)

