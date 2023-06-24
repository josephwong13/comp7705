FROM public.ecr.aws/lambda/python:3.10

RUN pip install tensorflow
RUN pip install tensorflow-text
RUN pip install tensorflow-hub
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install scipy

COPY saved_model/bertSmsClassifier saved_model/bertSmsClassifier
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]