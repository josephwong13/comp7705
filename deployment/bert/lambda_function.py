import tensorflow as tf
import tensorflow_text as text
from keras.models import load_model
import json

model = load_model('saved_model/final_dataset_bert_en_uncased_L-8_H-512_A-8',compile=False)

def predict(features):
    return model.predict(features).tolist()

def lambda_handler(event, context):
    values = event['sms']
    result = predict(values)
    result = str(result).lstrip('[').rstrip(']')
    result = str(result).lstrip('\'').rstrip('\'')
    result = float(result)
    result = [float(tf.sigmoid((float(result))/10))]
    json_string = '{"score": ['+result+']}'
    result = json.loads(json_string)
    return result
# predict([[1,2,3,4]])