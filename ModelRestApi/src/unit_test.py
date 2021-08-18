import unittest

from fastapi.testclient import TestClient

from api_models import PredictionItem
from main import app, PredictProbaResponse
from modelresolver import ModelResolver


class TestModelResolver(unittest.TestCase):
    """
    Test class for the model resolver which tests loading and predictions
    """
    resolver = ModelResolver()

    def test_get_model(self):
        self.assertTrue(self.resolver.get_model("lr", "style"))

    def test_le_classes(self):
        self.assertTrue(self.resolver.encoders)
        self.assertGreater(len(self.resolver.encoders[-1].classes_), 0)

    def test_model_prediction(self):
        response = self.resolver.predict(model_name="lr",
                                         data="romaine lettuce,black olive,grape tomato,garlic,pepper,purple onion,garbanzo bean,feta cheese crumbles",
                                         model_type="style")
        self.assertTrue(response.name == "greek")
        self.assertTrue(response.probability >= 0.90)

    def test_model_prediction_proba(self):
        response = self.resolver.predict_proba(model_name="lr",
                                               data="romaine lettuce,black olive,grape tomato,garlic,pepper,purple onion,garbanzo bean,feta cheese crumbles",
                                               model_type="style")

        greek_filter_iter = filter(lambda x: x.name == "greek", response)
        assert next(greek_filter_iter, ).probability > 0.90

    def test_model_prediction_proba_allergen(self):
        response = self.resolver.predict_proba(model_name="lr",
                                               data="niacin, contains or less of wheat gluten, yeast, cul, folic acid, reduced iron, water, thiamin mononitrate, unbleached enriched flour wheat flour, sugar, wheat flour, salt, barley malt flour, cultured corn syrup solids, distilled vinegar, riboflavin",
                                               model_type="allergens")

        gluten_filter_iter = filter(lambda x: x.name == "gluten", response)
        assert next(gluten_filter_iter, ).probability > 0.90


class TestRestApi(unittest.TestCase):
    """
    Test class for RestApi
    """
    client = TestClient(app)

    def test_read_main(self):
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json() == {"msg": "Hello World"}

    def test_read_predict(self):
        response = self.client.post("/predict_style",
                                    json={"data": "romaine lettuce,black olive,grape tomato,garlic,pepper,purple onion,garbanzo bean,feta cheese crumbles",
                                          "model": "lr"})
        print(response.json())
        assert response.status_code == 200

        predict_response = PredictionItem(**response.json())

        assert predict_response.name == "greek"
        assert predict_response.probability > 0.90

    def test_read_predict_proba(self):
        response = self.client.post("/predict_proba",
                                    json={"data": "romaine lettuce,black olive,grape tomato,garlic,pepper,purple onion,garbanzo bean,feta cheese crumbles",
                                          "model": "lr",
                                          "type": "style"})
        print(response.json())
        assert response.status_code == 200

        predict_response = PredictProbaResponse(**response.json())
        greek_filter_iter = filter(lambda x: x.name == "greek", predict_response.predictions)

        assert len(predict_response.predictions) > 1
        assert next(greek_filter_iter, ).probability > 0.90

    def test_read_predict_proba_allergens(self):
        response = self.client.post("/predict_proba",
                                    json={"data": "niacin, contains or less of wheat gluten, yeast, cul, folic acid, reduced iron, water, thiamin mononitrate, unbleached enriched flour wheat flour, sugar, wheat flour, salt, barley malt flour, cultured corn syrup solids, distilled vinegar, riboflavins",
                                          "model": "lr",
                                          "type": "allergens"})
        print(response.json())
        assert response.status_code == 200

        predict_response = PredictProbaResponse(**response.json())
        gluten_filter_iter = filter(lambda x: x.name == "gluten", predict_response.predictions)

        assert len(predict_response.predictions) > 1
        assert next(gluten_filter_iter, ).probability > 0.90


if __name__ == '__main__':
    unittest.main()
