import uvicorn
from fastapi import FastAPI, HTTPException

from api_models import PredictRequest, PredictProbaResponse, ModelType, PredictRequestType, PredictionItem
from modelresolver import ModelResolver
from preprocessor import Preprocessor

print("- Model Rest API -")
print("[*] Loading models...")
modelResolver = ModelResolver()
preprocessor = Preprocessor()

app = FastAPI()


def check_model_exists(request: PredictRequest):
    """
    Check if model resolver has model available; raises HttpException if model has not been found/loaded
    :param request: incoming api request
    :return: -
    """
    if not modelResolver.check_model_exists(model_name=request.model, model_type=ModelType.allergens.name):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")


@app.get("/")
def read_root():
    return {"msg": "Hello World"}


@app.post("/predict_style", response_model=PredictionItem)
def predict(request: PredictRequest):
    """
    Predict style from request
    :param request: incoming api request
    :return:
    """
    check_model_exists(request)

    response = modelResolver.predict(model_name=request.model,
                                     data=preprocessor.process(request.data),
                                     model_type=ModelType.style.name)

    if not response:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' could not predict")

    return response


@app.post("/predict_allergens", response_model=PredictProbaResponse)
def predict(request: PredictRequest):
    """
    Predict allergens from request
    :param request: incoming api request
    :return:
    """
    check_model_exists(request)

    response = modelResolver.predict(model_name=request.model,
                                     data=preprocessor.process(request.data),
                                     model_type=ModelType.allergens_single.name)

    if isinstance(response, bool):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' could not predict")

    return PredictProbaResponse(predictions=response)


@app.post("/predict_proba", response_model=PredictProbaResponse)
def predict_proba(request: PredictRequestType):
    """
    Predict probability for each class / label depending on the model type
    :param request: incoming api request
    :return:
    """

    if not modelResolver.check_model_exists(model_name=request.model, model_type=request.type):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    response = modelResolver.predict_proba(model_name=request.model,
                                           data=preprocessor.process(request.data),
                                           model_type=request.type)

    if isinstance(response, bool):
        if not response:
            raise HTTPException(status_code=404, detail=f"Model '{request.model}' could not predict")

    return PredictProbaResponse(predictions=response)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
