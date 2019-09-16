# Managing dependecies
from tensorflow.keras.models import load_model
from keras import backend as K

import pickle
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Loading curse word data from file
with open('./web/curse-words-Google.txt') as curseWordFile:
    curseWordList = curseWordFile.read().split('\n')
curseWordDict = {}
for curseWordIndex in range(len(curseWordList)):
    curseWordDict[curseWordList[curseWordIndex]] = curseWordIndex

# Custom function to check the existence of any curse words from curseWordDict in the input sentence    
def IsSentenceCursed(sentence):
    for item in sentence.split():
        if(curseWordDict.get(item)):
            return True
    return False

# Custom function to filter stop words from the input sentence
def filterStopWords(sentence):
    stop_words = set(stopwords.words('english')) 
    filteredSentence = " ".join([w for w in sentence.split(' ') if not w in stop_words])
    return filteredSentence

# Loading vectorizer set to training data for tokenizing input sentence words in the format of input data before passing to model
with open('./backup/wordVectorizer/vectorizer.pkl', 'rb') as vectorizerFile:
    vectorizer = pickle.load(vectorizerFile)

# Custom function to caculate F1 score of model prediction, defined here because of model's architecture dependency
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Loading the model
model = load_model('./backup/model/baselineMLP/zs_mlp_model.h5', custom_objects= {'f1': f1}) # We serialized the weights of the feed-forward network



@app.route("/DatacampSentimentAnalyser", methods=["OPTIONS", "POST"])
def classifyComment():
    inputSentence = request.form['comment']
    response = {}

    # Checking for curse words contain in sentence
    if IsSentenceCursed(inputSentence) == True:
        response['success'] = False
        response['message'] = 'Categorized as spam comment due to existence of curse words'
        
    else:

        # Filtering stop words from sentence
        filteredSentence = filterStopWords(inputSentence)

        if(len(filteredSentence) == 0):
            response['success'] = False
            response['message'] = 'Categorized as spam comments, due unavailability of useful content after removing stop words'
        
        response['success'] = True
        # Vectorizing sentence for model input
        inputData = vectorizer.transform(np.array([filteredSentence]))
        
        print('BEFORE RESHAPING, shape of input data: {}'.format(inputData.shape))
        # inputData = inputData.reshape(inputData.shape[1], inputData.shape[0])
        print('AFTER RESHAPING, shape of input data: {}'.format(inputData.shape))
        if model.predict_classes(inputData)[0][0] == 1:
            response['message'] = 'Categorized as spam comment by the model'
        else:
            response['message'] = 'Categorized as non-spam comment by the model'
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)