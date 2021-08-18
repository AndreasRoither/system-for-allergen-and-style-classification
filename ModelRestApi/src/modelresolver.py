import numpy as np

from api_models import PredictionItem, ModelType
from modelloader import ModelLoader


class ModelResolver:
    """
    Class that holds models
    """
    ROUND_AMOUNT = 4

    encoders = []
    style_models = {}
    allergen_models = {}
    allergen_single_models = []

    allergens_classifier = [
        "gluten", "crustaceans", "eggs", "fish", "peanuts", "soybeans", "milk", "nuts", "celery",
        "mustard", "sesame-seeds", "sulphur-dioxide-and-sulphites", "lupin", "molluscs"
    ]

    # the single classifier has its class list according to the dateset, which is different from the order which the others are trained
    allergens_single_classifier = [
        "celery", "crustaceans", "eggs", "fish", "gluten", "lupin", "milk", "molluscs", "mustard",
        "nuts", "peanuts", "sesame-seeds", "soybeans", "sulphur-dioxide-and-sulphites"
    ]

    def __init__(self):
        """
        Load models / encoders
        """
        loader = ModelLoader()
        self.style_models, self.allergen_models, self.allergen_single_models = loader.load_models()
        self.encoders = loader.load_model_classes()

    def check_model_exists(self, model_name: str, model_type: str):
        """
        Check if specific model exists
        :param model_name: name of the model
        :param model_type: category of the model
        :return: true if model exists
        """

        if not ModelType.has_key_value(model_type):
            return False

        if ModelType.style.name == model_type:
            return model_name in self.style_models

        if ModelType.allergens.name == model_type:
            return model_name in self.allergen_models

        if ModelType.allergens_single.name == model_type:
            return model_name in self.allergen_single_models

    def get_model(self, model_name: str, model_type: str):
        """
        Get latest version of model with a specific name
        :param model_type: which model category should be used; see ModelCategory
        :param model_name: name of the model category
        :return: model (dictionary) or model (for single) or False if empty list
        """

        if not ModelType.has_key_value(model_type):
            return False

        if ModelType.style.name == model_type:
            return self.style_models[model_name][-1]

        if ModelType.allergens.name == model_type:
            return self.allergen_models[model_name]

        if ModelType.allergens_single.name == model_type:
            return self.allergen_single_models[model_name]

    def predict(self, model_name: str, data: str, model_type: str):
        """
        Predict with confidence of style / allergen for certain data
        :param model_name: name of the model: lr/svc etc..
        :param data: string of ingredients
        :param model_type: style or allergen
        :return: False if no model found otherwise prediction with probability
        """
        if not ModelType.has_key_value(model_type):
            return False

        if ModelType.style.name == model_type:
            return self.__predict_style(model_name, data)

        if ModelType.allergens.name == model_type:
            return self.__predict_allergens(model_name, data)

        if ModelType.allergens_single.name == model_type:
            return self.__predict_allergens_single(model_name, data)

    def predict_proba(self, model_name: str, data: str, model_type: str):
        """
        Predict probability of style / allergen for certain data
        :param model_name: name of the model: lr/svc etc..
        :param data: string of ingredients
        :param model_type: style or allergen
        :return: False if no model found otherwise prediction with probability
        """

        if not ModelType.has_key_value(model_type):
            return False

        if ModelType.style.name == model_type:
            return self.__predict_proba_style(model_name, data)

        if ModelType.allergens.name == model_type:
            return self.__predict_proba_allergens(model_name, data)

        if ModelType.allergens_single.name == model_type:
            return self.__predict_proba_allergens_single(model_name, data)

    def __predict_style(self, model_name: str, data: str):
        """
        Predicts with model and label encoder
        :param model_name: name of the model category
        :param data: data which is used to predict
        :return: class and confidence of the prediction OR False, False if no prediction or encoder has been loaded
        """
        if not self.encoders:
            return False

        model = self.get_model(model_name, ModelType.style.name)
        y_pred = model.predict([data])

        if len(y_pred) <= 0:
            return False

        confidence = model.predict_proba([data])[0][y_pred]
        class_predictions = self.encoders[-1].inverse_transform(y_pred)

        if len(class_predictions) <= 0:
            return False

        return PredictionItem(name=str(class_predictions[0]), probability=round(confidence[0], self.ROUND_AMOUNT))

    def __predict_proba_style(self, model_name: str, data: str):
        """
        Predicts probabilities with model and label encoder
        :param model_name: name of the model category
        :param data: data which is used to predict
        :return: [], [] of classes and predictions for each class rounded OR False, False if encoder has not been loaded or no predictions have been created
        """
        if not self.encoders:
            return False

        model = self.get_model(model_name, ModelType.style.name)
        y_pred = model.predict_proba([data])

        if len(y_pred) <= 0:
            return False

        response = []
        classes = self.encoders[-1].classes_.tolist()
        prediction = np.round(y_pred[0], self.ROUND_AMOUNT).tolist()

        if len(classes) != len(prediction):
            raise Exception(f"Model '{model_name}' amount of classes does not match with the corresponding encoder! Encoder: {len(classes)} Model: {len(prediction)}")

        for c, p in zip(classes, prediction):
            response.append(PredictionItem(name=c, probability=p))

        return response

    # TODO: set a good threshold value
    def __predict_allergens(self, model_name, data, score_threshold: float = 0.5):
        """
        Predict which allergens are in a text
        :param model_name: name of the model category
        :param data: data which is used to predict
        :return:
        """

        # get_model returns a dictionary of a models
        model_dict = self.get_model(model_name, ModelType.allergens.name)
        response = []

        for allergen in self.allergens_classifier:
            if allergen in model_dict:
                model = model_dict[allergen]
                probability_positive = model.predict_proba([data])[0][1]

                if probability_positive >= score_threshold:
                    response.append(PredictionItem(name=allergen, probability=round(probability_positive, self.ROUND_AMOUNT)))
            else:
                # found missing allergen model
                print(f"[Error] Missing allergen model for: {allergen}")
                return False
        return response

    def __predict_allergens_single(self, model_name, data, score_threshold: float = 0.5):
        """
        Predict which allergens are in a text
        :param model_name: name of the model category
        :param data: data which is used to predict
        :return:
        """

        # get_model returns a dictionary of a models
        model = self.get_model(model_name, ModelType.allergens_single.name)
        probability_positive = model.predict_proba([data])[0]

        response = []

        for index, prob in enumerate(probability_positive):
            allergen = self.allergens_single_classifier[index]
            if prob >= score_threshold:
                response.append(PredictionItem(name=allergen, probability=round(prob, self.ROUND_AMOUNT)))

        return response

    def __predict_proba_allergens(self, model_name, data):
        return self.__predict_allergens(model_name, data, score_threshold=0)

    def __predict_proba_allergens_single(self, model_name, data):
        return self.__predict_allergens_single(model_name, data, score_threshold=0)
