import tensorflow as tf
import tensorflow_text as text
from keras.models import load_model

model = load_model('saved_model/bertSmsClassifier',compile=False)

def predict(features):
    return model.predict(features).tolist()

def lambda_handler(event, context):
    values = event['sms']
    result = predict(values)
    result = str(result).lstrip('[').rstrip(']')
    result = float(result)
    result = [float(tf.sigmoid((float(result))/10))]
    return result
# predict([[1,2,3,4]])