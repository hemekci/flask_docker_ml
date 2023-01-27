FROM python:3.6-slim
RUN mkdir /app
COPY . /app/
WORKDIR /app
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT FLASK_APP=deploy_flask.py flask run --host=0.0.0.0 --port=8080