FROM tensorflow/tensorflow

WORKDIR /app

RUN apt-get update && apt-get install -y python3
RUN pip install flask
RUN  pip3 install Flask Werkzeug --upgrade
COPY data   data

COPY predict.py .

COPY cnn_model.py .

COPY checkpoints checkpoints

EXPOSE 5000

CMD ["python3","predict.py"]