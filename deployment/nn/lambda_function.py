import json
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

loaded_model = load_model('./best_model.h5')

def featureExtraction(input):
    tokenizer = pickle.load(open('./best_tokenizer.pkl', 'rb'))

    print(len(tokenizer.word_index))

    output = input.copy()
    output = tokenizer.texts_to_sequences(output)
    output = pad_sequences(output, maxlen=100, padding='post', truncating='post')

    return output

def predict(input):
    output = featureExtraction(input)
    output = loaded_model.predict(output)
    return output

def lambda_handler(event, context):
    sms = event['preprocessed_sms']
    result = predict(sms)
    result = str(result).lstrip('[').rstrip(']')
    result = str(result).lstrip('\'').rstrip('\'')
    json_string = '{"score": ['+result+']}'
    result = json.loads(json_string)
    return result
# predict([[1,2,3,4]])