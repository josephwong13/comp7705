FROM public.ecr.aws/lambda/python:3.10

RUN pip install tensorflow
RUN pip install tensorflow-text
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install scipy

COPY saved_model/final_dataset_bert_en_uncased_L-8_H-512_A-8 saved_model/final_dataset_bert_en_uncased_L-8_H-512_A-8
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]