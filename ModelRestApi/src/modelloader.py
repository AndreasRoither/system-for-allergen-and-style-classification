import fnmatch
import os
from os import listdir

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ModelLoader:
    """
    Class to load ml models for style and allergen detection
    """

    encoders = []
    style_models = {}
    allergen_models = {}
    allergen_single_models = {}

    style_le_encoder_path = '../models/style/en/le_classes/'
    style_model_path = '../models/style/en'
    allergen_model_path = '../models/allergen/en'
    allergen_single_model_path = '../models/allergen_single/en'

    names = [
        "knn",
        "svc",
        "linearsvc",
        "cart",
        "rf",
        "mlp",
        "lda",
        "lr",
        "nb"
    ]

    style_model_paths = {
        'knn': f"{style_model_path}/knn/",
        'svc': f"{style_model_path}/svc/",
        'linearsvc': f"{style_model_path}/linearsvc/",
        'cart': f"{style_model_path}/cart/",
        'rf': f"{style_model_path}/rf/",
        'xgboost': f"{style_model_path}/xgboost",
        'mlp': f"{style_model_path}/mlp/",
        'lda': f"{style_model_path}/lda/",
        'lr': f"{style_model_path}/lr/",
        'nb': f"{style_model_path}/nb/"
    }

    allergen_model_paths = {
        'knn': f"{allergen_model_path}/knn/",
        'svc': f"{allergen_model_path}/svc/",
        'linearsvc': f"{allergen_model_path}/linearsvc/",
        'cart': f"{allergen_model_path}/cart/",
        'rf': f"{allergen_model_path}/rf/",
        'xgboost': f"{allergen_model_path}/xgboost",
        'mlp': f"{allergen_model_path}/mlp/",
        'lda': f"{allergen_model_path}/lda/",
        'lr': f"{allergen_model_path}/lr/",
        'nb': f"{allergen_model_path}/nb/"
    }

    allergen_single_model_paths = {
        'knn': f"{allergen_single_model_path}/knn/",
        'svc': f"{allergen_single_model_path}/svc/",
        'linearsvc': f"{allergen_single_model_path}/linearsvc/",
        'cart': f"{allergen_single_model_path}/cart/",
        'rf': f"{allergen_single_model_path}/rf/",
        'xgboost': f"{allergen_single_model_path}/xgboost",
        'mlp': f"{allergen_single_model_path}/mlp/",
        'lda': f"{allergen_single_model_path}/lda/",
        'lr': f"{allergen_single_model_path}/lr/",
        'nb': f"{allergen_single_model_path}/nb/"
    }

    def load_models(self):
        """
        Load all models in the current file system for each type

        Parameters
        ----------
        :return: style models (dictionary of arrays), allergen models (dictionary of dictionaries with arrays)
        """

        for name in self.names:
            self.style_models[name] = []
            self.allergen_models[name] = {}

            style_path = self.style_model_paths[name]
            allergen_path = self.allergen_model_paths[name]
            allergen_single_path = self.allergen_single_model_paths[name]

            if os.path.exists(style_path):
                style_model_file_names = [f for f in sorted(listdir(style_path)) if fnmatch.fnmatch(f, '*.sav')]
                for model_name in style_model_file_names:
                    loaded_model = joblib.load(f'{style_path}{model_name}')
                    self.style_models[name].append(loaded_model)

            if os.path.exists(allergen_path):
                allergen_model_file_names = [f for f in sorted(listdir(allergen_path)) if fnmatch.fnmatch(f, '*.sav')]
                for model_name in allergen_model_file_names:
                    loaded_model = joblib.load(f'{allergen_path}{model_name}')
                    category = model_name.lower().split('_')[2]

                    self.allergen_models[name][category] = loaded_model

            if os.path.exists(allergen_single_path):
                allergen_model_file_names = [f for f in sorted(listdir(allergen_single_path)) if fnmatch.fnmatch(f, '*.sav')]
                for model_name in allergen_model_file_names:
                    loaded_model = joblib.load(f'{allergen_single_path}{model_name}')

                    self.allergen_single_models[name] = loaded_model

        return self.style_models, self.allergen_models, self.allergen_single_models

    def load_model_classes(self):
        """
        Loads label encoder classes
        :return: encoder classes
        """

        if os.path.exists(self.style_le_encoder_path):
            encoder_files = [f for f in sorted(listdir(self.style_le_encoder_path)) if fnmatch.fnmatch(f, '*.npy')]

            for encoder_path in encoder_files:
                encoder = LabelEncoder()
                encoder.classes_ = np.load(f'{self.style_le_encoder_path}{encoder_path}', allow_pickle=True)
                self.encoders.append(encoder)

            return self.encoders
        else:
            raise Exception(f"Could not find label encoder path: {self.style_le_encoder_path}")
