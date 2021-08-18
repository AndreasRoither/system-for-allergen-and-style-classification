<h1 align="center">
  <!--<a name="logo" href=""><img src="" alt="Logo" width="200"></a>-->
  <br>  
  Model REST API Documentation

  ![Python](https://img.shields.io/badge/python-v3.7-blue.svg)
</h1>

## Overview

The Model REST API is a RESTful API that provides access to various machine learning models. The REST API is written in Python and uses FastAPI, a high-performance web framework to build APIs. The API has three main components: the model loader, preprocessor and the main app, which serves the endpoints.

## Endpoints

The REST API has three main endpoints that can be used for prediction:

- /predict_style
- /predict_allergens
- /predict_proba

## Startup

The main app can be started in the source folder with the command:

```shell
uvicorn main:app --reload
```

Or alternatively with docker:

```docker
docker-compose up -d
```

## Technology used

Idea:  
[IntelliJ IDEA](https://www.jetbrains.com/idea/)  

[Python 3.7](https://www.python.org/)  
[Docker](https://www.docker.com/)  
[scikit-learn](https://scikit-learn.org/)
