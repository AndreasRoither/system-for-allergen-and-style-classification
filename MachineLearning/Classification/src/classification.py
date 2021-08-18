import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import colors
from src.analytics import Analytics


class Classification:
    job_number = 3

    analytics = Analytics()
    le = LabelEncoder()

    modelPath = "../models/"
    modelParamPath = "../models/param/"

    vectorizer = CountVectorizer(ngram_range=(1, 3))
    Path(modelPath).mkdir(parents=True, exist_ok=True)
    Path(modelParamPath).mkdir(parents=True, exist_ok=True)

    def fit_label_encoder(self, data_frame, classifier_col):
        """
        Encode target labels with value between 0 and n_classes-1.

        Parameters
        ----------
        :param data_frame: data frame with classifiers</br>
        :param classifier_col: column number of the classifier</br>
        :return: -
        """
        self.le.fit(pd.unique(data_frame.iloc[:, classifier_col]))

    def train_test_split(self, data_frame, ingredient_col, classifier_col):
        """
        Split list of data frames into train and test sets using sci kit learn</br>
        Creates one big data frame.</br>
        Return order is important!!</br>

        Parameters
        ----------
        :param data_frame:  </br>
        :param ingredient_col: </br>
        :param classifier_col: </br>
        :return: data_train, data_test, target_train, target_test
        """

        # fit for the label encoder first
        self.fit_label_encoder(data_frame, classifier_col)

        # don't forget stratify as we might have class imbalance problems, using stratify means that the proportion
        # of values in the sample produced in our test group will be the same as the proportion of
        # values provided to parameter stratify
        data_train, data_test, target_train, target_test = train_test_split(data_frame.iloc[:, ingredient_col],
                                                                            self.le.transform(
                                                                                data_frame.iloc[:, classifier_col]),
                                                                            stratify=data_frame.iloc[:, classifier_col],
                                                                            train_size=0.75,
                                                                            random_state=123456,
                                                                            shuffle=True)

        return data_train, data_test, target_train, target_test

    def vectorize_data(self, data_train, data_test):
        """
        Vectorize data_train, data_test with in class vectorizer (currently CountVectorizer)

        Parameters
        ----------
        :param data_train: train list</br>
        :param data_test: test list</br>
        :return: Vectorized: train, test
        """
        # since we can't use text directly we have to vectorize the text

        train = self.vectorizer.fit_transform(data_train)
        test = self.vectorizer.transform(data_test)
        return train, test

    def load_test(self, model_path, model_name, data_test, target_test):
        """
        Load saved test from joblib dump

        Parameters
        ----------
        :param model_path: Path to the model, including file name</br>
        :param model_name: name of the current loaded model</br>
        :param data_test: data to test against</br>
        :param target_test: target data</br>
        :return: loaded model
        """
        print(f'{colors.neutral}Checking model {model_name}')

        loaded_model = joblib.load(model_path)
        # result = loaded_model.score(data_test, target_test)
        # print(colors.positive + "Test score: {0:.2f} %".format(100 * result))

        return loaded_model

    def export_lf_classes(self):
        """
        Exports the currently used label encoder classes
        :return:
        """
        current = str(datetime.now().replace(second=0, microsecond=0)).replace(":", "_").replace(" ", "_").replace("-", "_")
        np.save(f'{self.modelPath}le_classes_{current}.npy', self.le.classes_)

    def train_test(self, data_train, data_test, target_train, target_test, class_num, advanced_analytics=False):
        """
        Train and test data with various models

        Parameters
        ----------
        :param data_train: train set</br>
        :param data_test: test set</br>
        :param target_train: train set</br>
        :param target_test: test set</br>
        :param class_num: number of existing classes</br>
        :param advanced_analytics: more in depth test and analysis of gridSearchCV models</br>
        :return: -
        """

        ###############
        # Train Model #
        ###############
        print(f'{colors.neutral}Training models..')

        names = [
            # "KNN",
            "SVC",
            "LinearSVC",
            # "CART",
            # "RF",
            # "XGBoost",
            # "NN",
            # "LDA",
            "LR",
            # "NB"
        ]

        classifiers = {
            'KNN': KNeighborsClassifier(),
            'SVC': SVC(),
            'LinearSVC': LinearSVC(),
            'CART': DecisionTreeClassifier(),
            'RF': RandomForestClassifier(),
            'XGBoost': xgb.XGBRegressor(),
            'NN': MLPClassifier(),
            'LDA': LinearDiscriminantAnalysis(),
            'LR': LogisticRegression(),
            'NB': GaussianNB()
        }

        params = {
            'KNN': {'n_neighbors': [50, 75, 100]},  # 1, 3, 5, 7, 10, 15, 25, 35,
            'SVC': [
                # {'kernel': ['sigmoid'], 'C': [1, 10, 100]},  # 'linear', 'poly' # 0.001, 0.01, 0.1,
                {'kernel': ['rbf'], 'C': [10], 'gamma': [0.001]}  # 0.001, 0.01, 0.1, 1,   |  0.1, 1
            ],
            'LinearSVC': {
                'penalty': ['l1'],  # 'l2'
                'C': [0.2],  # 0.1, 0.2, 0.3, 0.4
                'dual': [False],
                'max_iter': [1100]
            },
            'CART': {
                'max_features': ['auto'],  # 'max_features': ['auto', 'log2'],
                'min_samples_leaf': [1],  # 'min_samples_leaf': [1, 5, 10],
                'max_depth': [120],  # 'max_depth': [1, 5, 10, 20, 25, 50],
            },
            'RF': {
                'max_features': ['auto'],  # 'auto', 'log2'
                'max_depth': [75],  # 5, 10,
                'n_estimators': [100],  # 1, 5,
                'class_weight': ['balanced']
            },
            'XGBoost': {
                'n_estimators': [100, 200],  # [1, 5, 10, 100]
                'max_depth': [5, 7, 10],  # [5, 10, 20, 50]
                'objective': ['multi:softmax'],
                'learning_rate': [0.1, 0.2, 0.3],  # [0.1, 0.5, 1]
                'alpha': [0, 1, 5],  # [1, 5, 10]
                'num_class': class_num
            },
            'NN': {
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'hidden_layer_sizes': [(5,), (5, 5), (10,), (20, 14, 8)],
                # , (10,), (1024, 512, 16), (13, 13, 13), (20, 14, 8)
                'shuffle': [True],
                'random_state': [True],
                'early_stopping': [True],
                'activation': ['relu', 'logistic'],  # 'identity', 'logistic', 'tanh',
                'batch_size': ['auto'],
                'max_iter': [500, 1000]  # , 500, 200, 100
            },
            'LDA': {'solver': ['lsqr']},  # 'svd'
            'LR': {'solver': ['lbfgs'], 'multi_class': ['auto'], 'max_iter': [1000], 'C': [0.5]},
            # [0.01, 0.1, 1, 2, 5, 10] 'liblinear''C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            'NB': {}
        }

        k_fold = KFold(n_splits=10, random_state=123456, shuffle=True)
        grid_searches = {}

        print(f'{colors.neutral}Testing various models: {" ".join(names)}')

        # fit for all defined models and parameter configurations
        for name in names:
            print("\n" + "-------------------")
            print(f'{colors.neutral}Running GridSearchCV for {name}.')

            start = time.time()
            model = classifiers[name]
            parameter_set = params[name]
            gs = GridSearchCV(model, parameter_set, cv=k_fold, scoring='accuracy', verbose=1, n_jobs=self.job_number)

            if name == "LDA" or name == "NB":
                gs.fit(data_train.todense(), target_train)
            else:
                gs.fit(data_train, target_train)

            grid_searches[name] = gs

            print(f'{colors.positive}Done in: {str(round(time.time() - start, 4))}s')

            print(f'{colors.neutral}Best parameters for {name} fit are:')
            print(gs.best_params_)

            # save model
            current = str(datetime.now().replace(second=0, microsecond=0)).replace(":", "_").replace(" ", "_").replace("-", "_")
            pipe = Pipeline([('vect', self.vectorizer), ('model', gs.best_estimator_)])
            joblib.dump(pipe, self.modelPath + 'best_model_' + name + '_' + current + '.sav', compress=1)

            json_txt = json.dumps(gs.best_params_, indent=4)

            # save parameters as json
            with open(self.modelParamPath + 'best_model_param_' + name + '_' + current + '.txt', 'w') as file:
                file.write(json_txt)

        accuracies_test = []
        accuracies_train = []
        predicted_classes_all = []
        kappas_test = {}
        mae = {}
        mse = {}
        labels = self.le.classes_

        for name in names:
            print("\n" + "-------------------")
            print(f'{colors.neutral}Testing: {name}')

            start = time.time()
            gs_test = grid_searches[name]

            if name == "LDA" or name == "NB":
                predicted_classes = gs_test.best_estimator_.predict(data_test.todense())
            else:
                predicted_classes = gs_test.best_estimator_.predict(data_test)

            predicted_accuracy = round(accuracy_score(target_test, predicted_classes), 4)
            predicted_classes_all.append(predicted_classes)

            if advanced_analytics:
                if name == "LDA" or name == "NB":
                    print(f'{colors.neutral}Running CV on train data for: {name}')
                    accuracy_train = model_selection.cross_val_score(gs_test, data_train.todense(), target_train,
                                                                     cv=k_fold,
                                                                     scoring='accuracy',
                                                                     n_jobs=self.job_number,
                                                                     verbose=0)
                    print(f'{colors.neutral}Running CV on test data for: {name}')
                    accuracy_test = model_selection.cross_val_score(gs_test, data_test.todense(), target_test,
                                                                    cv=k_fold,
                                                                    scoring='accuracy',
                                                                    n_jobs=self.job_number,
                                                                    verbose=0)
                else:
                    print(f'{colors.neutral}Running CV on train data for: {name}')
                    accuracy_train = model_selection.cross_val_score(gs_test, data_train, target_train,
                                                                     cv=k_fold,
                                                                     scoring='accuracy',
                                                                     n_jobs=self.job_number,
                                                                     verbose=0)
                    print(f'{colors.neutral}Running CV on test data for: {name}')
                    accuracy_test = model_selection.cross_val_score(gs_test, data_test, target_test,
                                                                    cv=k_fold,
                                                                    scoring='accuracy',
                                                                    n_jobs=self.job_number,
                                                                    verbose=0)

                print(colors.positive + "Done in: " + str(round(time.time() - start, 4)) + "s")

                print(colors.positive + "Accuracy for train data on " + name + " is: " + str(accuracy_train))
                accuracies_train.append(accuracy_train)
                print(colors.positive + "Accuracy for test data on " + name + " is: " + str(accuracy_test))
                accuracies_test.append(accuracy_test)

            #   best_parameters, score, _ = max(gs_test.cv_results_, key=lambda x: x[1])
            #   print('Raw AUC score:', score)
            #   for param_name in sorted(best_parameters.keys()):
            #    print("%s: %r" % (param_name, best_parameters[param_name]))

            kappas_test[name] = round(cohen_kappa_score(target_test, predicted_classes), 5)
            mae[name] = round(metrics.mean_absolute_error(target_test, predicted_classes), 5)
            mse[name] = round(metrics.mean_squared_error(target_test, predicted_classes), 5)

            print('Classification Report ' + name + ':')
            print('f1 score weighted %s' % f1_score(target_test, predicted_classes, average='weighted'))
            print(classification_report(self.le.inverse_transform(target_test),
                                        self.le.inverse_transform(predicted_classes)))
            print(colors.positive + 'Mean Absolute Error ' + name + ': ' + str(mae[name]))
            print(colors.positive + 'Mean Squared Error ' + name + ': ' + str(mse[name]))
            print(colors.positive + "Predicted accuracy for " + name + ": %.4f" % float(predicted_accuracy))
            print(colors.positive + "Cohen Kappa Score of " + name + " on test: " + str(kappas_test[name]))

        if advanced_analytics:
            self.analytics.plot_boxplot(names, accuracies_test, "Test data accuracy")
            self.analytics.plot_boxplot(names, accuracies_train, "Train data accuracy")

            count = 0
            for p_classes in predicted_classes_all:
                cm = confusion_matrix(self.le.inverse_transform(target_test),
                                      self.le.inverse_transform(p_classes),
                                      labels=labels)
                self.analytics.show_confusion_matrix(cm, labels, names[count])
                self.analytics.plot_confusion_matrix(cm, labels, title="Confusion matrix " + names[count] + "_2")
                count += 1

        ########################################
        # CLEANUP
        ########################################

        del accuracies_test, accuracies_train, mae, mse, k_fold, grid_searches
        del classifiers, names, params

        print(colors.positive + "Finished training / testing")
