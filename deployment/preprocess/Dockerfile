FROM public.ecr.aws/lambda/python:3.10

RUN pip install scikit-learn
RUN pip install pandas
RUN pip install scipy
RUN pip install jieba
RUN pip install zhconv
RUN pip install nltk
RUN pip install contractions

COPY nltk_data nltk_data
COPY jieba jieba
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]