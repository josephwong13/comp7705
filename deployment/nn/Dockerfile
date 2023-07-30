FROM public.ecr.aws/lambda/python:3.10

RUN pip install tensorflow
RUN pip install tensorflow-text
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install scipy

COPY best_model.h5 .
COPY best_tokenizer.pkl .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]