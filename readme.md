

## Step By Step Dockerized ML Flask project

### Create a project on Local 

- Created an environment for this project. 

```
 $ conda create -n flask_docker_ML python=3.6 anaconda
 $ conda activate flask_docker_ML
```

- Create a workspace put inside

    1- ML.py - Preprocess data, create models and saved them

    2- deploy_flask - Simple flask project uses index.html in templates

    3- directory 'saved_models' - is used for saving trained models

    4- directory 'templates'

        4a- index.html - a form with submit 

    5- requirements.txt 
- 
```
$ pip install -r requirements.txt
``` 

### Build an image (Dockerize) and run on Docker container

- Create a 'dockerfile' to dockerize this app

dockerfile
```
FROM python:3.6-slim
RUN mkdir /app
COPY . /app/
WORKDIR /app
RUN pip install --upgrade pip && RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT FLASK_APP=app.py flask run --host=0.0.0.0 --port8080
```

 - create docker image and container.
```
$docker image build -t hemekci/flask_ml:1.0 .
$docker run --rm --name my_flask_ml -p 8080:8080 -d hemekci/flask_ml:1.0
```

- check all on the localhost
    
    http://localhost:8080/