import re
import contractions
import jieba
import pandas as pd
import nltk
import json
import tensorflow as tf
import tensorflow_text
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from zhconv import convert
from keras.models import load_model

nltk.data.path.append('./nltk_data')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
jieba.load_userdict('./jieba/dict_big.txt')

def scToTc(text):
    text = convert(text, 'zh-tw')

    return text

def expandContraction(text):
    # specific
    text = re.sub(r'i[\'?]m', 'i am', text)
    text = re.sub(r'let[\'?]s', 'let us', text)
    text = re.sub(r'don[\'?]t', 'do not', text)
    text = re.sub(r'can[\'?]t', 'can not', text)
    text = re.sub(r'won[\'?]t', 'will not', text)

    # general
    text = re.sub(r'[\'?]s', ' is', text)
    text = re.sub(r'[\'?]re', ' are', text)
    text = re.sub(r'[\'?]ll', ' will', text)
    text = re.sub(r'[\'?]d', ' would', text)
    text = re.sub(r'[\'?]ve', ' have', text)
    text = re.sub(r'n[\'?]t', ' not', text)

    # library
    text = contractions.fix(text)

    return text

def cleanData(text):
    # expand contraction
    text = expandContraction(text)

    # replace hyperlink
    text = re.sub(r'http[s]?:\/\/[\w\/.?=-]+', ' link ', text)

    # replace email address
    text = re.sub(r'[\w\.+]+@[\w\.]+\.[a-z]{2,}', ' email ', text)

    # replace currency sign
    text = re.sub(r'[\$€£¥]', ' money ', text)

    # replace number
    text = re.sub(r'[\d]+', ' number ', text)

    # replace special char, other than a-z, A-Z, 0-9 and chinese
    text = re.sub(r'[^a-zA-Z0-9\u4E00-\u9FFF]+', ' ', text)

    # replace new line (carriage return and line feed)
    text = re.sub(r'[\r\n]', ' ', text)

    # replace white space
    text = re.sub(r'[\s]{2,}', ' ', text)
    text = re.sub(r'^[\s]+|[\s]+$', '', text)

    return text

def stopWords(text, words):
    text = ' '.join([word for word in text.split() if word not in (words)])

    return text

def stemming(text, stemmer):
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text

def lemmatization(text, lemmatizer):
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text

def segmentation(text):
    text = ' '.join(jieba.cut(text))

    return text

def preprocess(input, remove_stop, stem, lemmatize):
    words = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    output = []

    for text in input:
        text = text.lower()
        text = scToTc(text)
        text = cleanData(text)

        if remove_stop:
            text = stopWords(text, words)

        if stem:
            text = stemming(text, stemmer)

        if lemmatize:
            text = lemmatization(text, lemmatizer)

        text = segmentation(text)
        text = re.sub(r'[\s]{2,}', ' ', text)

        output.append(text)
    return output

model = load_model('saved_model/bertSmsClassifier')

def predict(features):
    return model.predict(features).tolist()

def lambda_handler(event, context):
    sms = event['sms']
    result = preprocess(sms, True, True, True)
    result = str(result).lstrip('[').rstrip(']')
    result = str(result).lstrip('\'').rstrip('\'')
    #json_string = '{"sms": ["'+result+'"]}'
    #result = json.loads(json_string)
    result = predict([result])
    result = str(result).lstrip('[').rstrip(']')
    result = float(result)
    result = [float(tf.sigmoid((float(result))/10))]
    return result
# predict([[1,2,3,4]])