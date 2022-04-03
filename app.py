from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle

app = Flask(__name__)

# load machine learning models
mnb_model = pickle.load(open(r"mnb_model.sav", "rb"))
lr_model = pickle.load(open(r"lr_model.sav", "rb"))
rfc_model = pickle.load(open(r"rfc_model.sav", "rb"))


# create a function to clean input text
def textCleaning(text):
    
    # transform the text into lower cases
    text = text.lower()
    
    # remove special characters and punctuations in the text
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # remove digits in the text 
    text = re.sub(r"[\d]", " ", text)
    
    # tokenize the text
    tokenizedText = word_tokenize(text)
    
    # remove stopwords from the text
    filteredText = list()
    for word in tokenizedText:
        if word not in stopwords.words('english'):
            filteredText.append(word)
    
    # lemmatize the word 
    lemmatizeWords = list()
    lemmatizer = WordNetLemmatizer()
    for word in filteredText:
        lemmatizeWords.append(lemmatizer.lemmatize(word))
    
    # reform the tokenize words into text
    newText = ' '.join(lemmatizeWords)
    
    return newText


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        ml_model = request.form['ml']
        textNews = request.form['news']
        
        # call the function to clean input text
        cleanedText = textCleaning(textNews)
        
        if ml_model == "mnb":
            predict = mnb_model.predict([cleanedText])
        elif ml_model == "lr":
            predict = lr_model.predict([cleanedText])
        else:
            predict = rfc_model.predict([cleanedText])
        
        return render_template('outcome.html', rawText = textNews, normalizedText = cleanedText, prediction = predict)

if __name__ == '__main__':
    app.run()
    
    
    
    
