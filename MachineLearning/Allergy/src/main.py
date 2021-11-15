import json
import time
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# Tweet tokenizer does not split at apostrophes which is what we want
from nltk.tokenize import TweetTokenizer
# FeatureEngineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import colors
# settings
from analytics import Analytics

# set background color
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.dpi'] = 360

# stats
start_time = time.time()
color = sns.color_palette()
sns.set_style("dark")
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

analytics = Analytics()

openf_datasets_processed_path = "../data/03_processed/openfoodfacts/"
modelPath = "../models/"
modelParamPath = "../models/param/"

job_number = 4


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case without weight and normalization
    http://stackoverflow.com/q/32239577/395857
    :param y_true: y labels, 2d Array
    :param y_pred: y predicted labels, 2d Array
    :param normalize:
    :param sample_weight:
    :return:
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


# noinspection DuplicatedCode
def main():
    print("> Allergen classification <")

    df = pd.read_csv(openf_datasets_processed_path + "openfoodfacts.csv", low_memory=False, sep="\t", dtype={'product_name': str, 'ingredients_text': str})
    df = df[df['ingredients_text'].notna()]

    allergens_classifier = [
        "en:celery", "en:crustaceans", "en:eggs", "en:fish", "en:gluten", "en:lupin", "en:milk", "en:molluscs", "en:mustard",
        "en:nuts", "en:peanuts", "en:sesame-seeds", "en:soybeans", "en:sulphur-dioxide-and-sulphites"
    ]

    # Applying Label Powerset Tranformation
    # https://medium.com/the-owl/imbalanced-multilabel-image-classification-using-keras-fbd8c60d7a4b
    df['powerlabel'] = df.apply(lambda x:
                                8192 * x["en:celery"] + 4096 * x["en:crustaceans"] + 2048 * x["en:eggs"] + 1024 * x["en:fish"]
                                + 512 * x["en:gluten"] + 256 * x["en:lupin"] + 128 * x["en:milk"] + 64 * x["en:molluscs"]
                                + 32 * x["en:mustard"] + 16 * x["en:nuts"] + 8 * x["en:peanuts"] + 4 * x["en:sesame-seeds"]
                                + 2 * x["en:soybeans"] + 1 * x["en:sulphur-dioxide-and-sulphites"], axis=1)

    powercount = {}
    powerlabels = np.unique(df['powerlabel'])
    for p in powerlabels:
        powercount[p] = np.count_nonzero(df['powerlabel'] == p)

    max_label_count = np.max(list(powercount.values()))
    for p in powerlabels:
        gap = int((max_label_count - powercount[p]) / 50)
        temp_df = df.iloc[np.random.choice(np.where(df['powerlabel'] == p)[0], size=gap)]
        df = df.append(temp_df, ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)
    del df['powerlabel']

    x_features = df['ingredients_text']
    y_labels = df[[i for i in df.columns if i not in ["product_name", "countries_en", "ingredients_text"]]]

    analytics.class_distribution(y_labels, "class_distribution_post_upsampling")

    # ---------------------
    # Train test split
    # ---------------------

    print(f'{colors.neutral}Train test split')

    data_train, data_test, target_train, target_test = train_test_split(x_features, y_labels,
                                                                        train_size=0.75, random_state=123456, shuffle=True)

    # ---------------------
    # TF-IDF
    # ---------------------

    # some detailed description of the parameters
    # min_df=10 --- ignore terms that appear lesser than 10 times
    # max_features=None  --- Create as many words as present in the text corpus
    # changing max_features to 10k for memory issues
    # analyzer='word'  --- Create features from words (alternatively char can also be used)
    # ngram_range=(1,1)  --- Use only one word at a time (uni grams)
    # strip_accents='unicode' -- removes accents
    # use_idf=1,smooth_idf=1 --- enable IDF
    # sublinear_tf=1   --- Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
    # temp settings to min=200 to facilitate top features section to run in kernels
    # change back to min=10 to get better results

    tfv = TfidfVectorizer(max_features=500,
                          min_df=10,
                          strip_accents='unicode',
                          analyzer='word',
                          ngram_range=(1, 2),
                          stop_words='english')

    # only test data should be fitted, as the vectorizer should only know about the training data ->
    # this ensures the true unseen nature of the test set.
    print(f'{colors.neutral}Tf-idf fit')
    tfv.fit(data_train)

    print(f'{colors.neutral}Tf-idf transform')
    data_train = tfv.transform(data_train)
    data_test = tfv.transform(data_test)

    names = [
        # "CART",
        "LR",
        # "MLP",
        # "RF",
    ]

    classifiers = {
        'CART': OneVsRestClassifier(DecisionTreeClassifier()),
        'RF': OneVsRestClassifier(RandomForestClassifier()),
        'MLP': MLPClassifier(),
        'LR': OneVsRestClassifier(LogisticRegression()),
    }

    params = {
        'CART': {
            'estimator__max_features': ['auto', 'sqrt'],
            'estimator__min_samples_leaf': [5, 10],
            'estimator__max_depth': [100, 200]
        },
        'RF': {
            'estimator__max_features': ['auto'],
            'estimator__max_depth': [10, 50, 70],  # 400
            'estimator__n_estimators': [100, 200, 300],  # 2000
            'estimator__class_weight': ['balanced']
        },
        'MLP': {
            'learning_rate': ['constant'],
            'hidden_layer_sizes': [(130,), (130, 100, 50), (30, 30, 30)],
            'shuffle': [True],
            'random_state': [True],
            'early_stopping': [True],
            'activation': ['relu'],
            'batch_size': ['auto'],
            'max_iter': [300]  # 550, 650
        },
        'LR': {
            'estimator__solver': ['sag', 'saga'],  # ['lbfgs', 'sag', 'saga', 'liblinear'],
            'estimator__multi_class': ['auto'],
            'estimator__max_iter': [2000, 3000, 5000],  # [1000, 1500, 2000, 2500, 3000],
            'estimator__C': [20, 30, 50],  # [0.001, 0.01, 0.1, 1, 5, 8, 10, 12, 20, 25 100, 1000],
            'estimator__class_weight': ['balanced']
        },
    }

    k_fold = KFold(n_splits=5, random_state=123456, shuffle=True)
    # k_fold = IterativeStratification(n_splits=5, order=1)
    grid_searches = {}
    cv_score_roc = {}
    cv_score_acc = {}
    cfs_matrix_test = {}
    cfs_matrix_train = {}

    print(f'{colors.neutral}Testing various models: {" ".join(names)}')

    # fit for all defined models and parameter configurations
    for name in names:
        print("\n-------------------")
        print(f'{colors.neutral}Running GridSearchCV for {name}.')

        start = time.time()

        cfs_matrix_test[name] = []
        cfs_matrix_train[name] = []
        cv_score_roc[name] = []
        cv_score_acc[name] = []
        classifier = classifiers[name]
        parameter_set = params[name]

        # try different scoring
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        gs = GridSearchCV(classifier, parameter_set, scoring='f1_weighted', verbose=1, n_jobs=job_number, cv=k_fold, return_train_score=True)
        test_loss = []

        # --------------------
        # Fit
        # --------------------
        gs.fit(data_train, target_train)
        grid_searches[name] = gs

        print(colors.positive + f'Best parameters for {name} fit are:')
        print(gs.best_params_)

        pred_train = gs.best_estimator_.predict(data_train)
        pred_test = gs.best_estimator_.predict(data_test)
        pred_test_proba = gs.best_estimator_.predict_proba(data_test)

        # --------------------
        # save model and parameter
        # --------------------
        current = str(datetime.now().replace(second=0, microsecond=0)).replace(":", "_").replace(" ", "_").replace("-", "_")
        pipe = Pipeline([('vect', tfv), ('model', gs.best_estimator_)])
        joblib.dump(pipe, f'{modelPath}{name}_{current}.sav', compress=1)

        # save parameters as json
        json_txt = json.dumps(gs.best_params_, indent=4)
        with open(f'{modelParamPath}model_param_{name}_{current}.json', 'w') as file:
            file.write(json_txt)

        predicted_classes = grid_searches[name].best_estimator_.predict_proba(data_test)

        print(f'{colors.positive}Done in: {str(round((time.time() - start) / 60, 4))} min')

        # ---------------------
        # Test for each class
        # ---------------------
        for index in range(pred_test.shape[1]):
            print("\n-------------------")
            allergen_pred_train = pred_train[:, index]
            allergen_pred_test = pred_test[:, index]

            test_target = target_test.iloc[:, index].to_numpy()
            train_target = target_train.iloc[:, index].to_numpy()

            # --------------------
            # confusion matrix
            # --------------------

            cfs_matrix_train[name].append(confusion_matrix(train_target, allergen_pred_train))
            cfs_matrix_test[name].append(confusion_matrix(test_target, allergen_pred_test))

            # --------------------
            # accuracy & classification report
            # --------------------
            print(f'Analytics for class {allergens_classifier[index]}:')
            print(colors.positive + f'Training accuracy for {allergens_classifier[index]} is {accuracy_score(train_target, allergen_pred_train)}')
            print(colors.positive + f'Test accuracy for {allergens_classifier[index]} is {accuracy_score(test_target, allergen_pred_test)}')

            # --------------------
            # ROC
            # --------------------
            # roc_auc_train = roc_auc_score(train_target, pred_train)
            roc_auc_test = roc_auc_score(test_target, allergen_pred_test)
            try:
                analytics.plot_roc_auc(target_test.iloc[:, index], pred_test_proba[:, index], name, allergens_classifier[index].replace("en:", ""))
            except TypeError:
                print("Could not print ROC")

            # print(colors.positive + f'Train ROC AUC for {allergens_classifier[index]} is {roc_auc_train}')
            print(colors.positive + f'Test ROC AUC for {allergens_classifier[index]} is {roc_auc_test}')

            # --------------------
            # log loss
            # --------------------

            # keep probabilities for the positive outcome only
            try:
                test = gs.best_estimator_.predict_proba(data_train)
                pred_train_positive = gs.best_estimator_.predict_proba(data_train)[:, 1]
                pred_test_positive = gs.best_estimator_.predict_proba(data_test)[:, 1]
                train_loss_class = log_loss(train_target, pred_train_positive)
                test_loss_class = log_loss(test_target, pred_test_positive)
                print(colors.positive + 'Train loss = log loss:', train_loss_class)
                print(colors.positive + 'Test loss = log loss:', test_loss_class)

                # train_loss.append(train_loss_class)
                test_loss.append(test_loss_class)
            except TypeError:
                print("Could not test proba")

        # --------------------
        # Analytics
        # --------------------

        print("\n-------------------")
        print(f'Classification Report {name}:')

        # ---------------------
        # Hamming loss and score
        # ---------------------
        print(colors.positive + f'Hamming loss train: {hamming_loss(target_train, pred_train)}')
        print(colors.positive + f'Hamming loss test: {hamming_loss(target_test, pred_test)}')
        print("")

        print(colors.positive + f'Hamming score train: {hamming_score(target_train.to_numpy(), pred_train)}')
        print(colors.positive + f'Hamming score test: {hamming_score(target_test.to_numpy(), pred_test)}')
        print("")

        print(colors.positive + f'log loss train: {log_loss(target_train, pred_train)}')
        print(colors.positive + f'log loss test: {log_loss(target_test, pred_test)}')
        print("")

        print(colors.positive + f'roc auc score train weighted: {roc_auc_score(target_train, pred_train, average="weighted")}')
        print(colors.positive + f'roc auc score test weighted: {roc_auc_score(target_test, pred_test, average="weighted")}')
        print(colors.positive + f'roc auc score train micro: {roc_auc_score(target_train, pred_train, average="micro")}')
        print(colors.positive + f'roc auc score test micro: {roc_auc_score(target_test, pred_test, average="micro")}')
        print(colors.positive + f'roc auc score train micro: {roc_auc_score(target_train, pred_train, average="macro")}')
        print(colors.positive + f'roc auc score test micro: {roc_auc_score(target_test, pred_test, average="macro")}')
        print("")

        print(colors.positive + f'F1-score weighted train: {f1_score(target_train, pred_train, average="weighted")}')
        print(colors.positive + f'F1-score weighted test: {f1_score(target_test, pred_test, average="weighted")}')
        print(colors.positive + f'F1-score macro train: {f1_score(target_train, pred_train, average="macro")}')
        print(colors.positive + f'F1-score macro test: {f1_score(target_test, pred_test, average="macro")}')
        print(colors.positive + f'F1-score micro train: {f1_score(target_train, pred_train, average="micro")}')
        print(colors.positive + f'F1-score micro test: {f1_score(target_test, pred_test, average="micro")}')
        print("")

        # Return the mean accuracy on the given test data and labels.
        # In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
        print(colors.positive + f'{name}: score on train data {gs.best_estimator_.score(data_train, target_train)}')
        print(colors.positive + f'{name}: score on test data {gs.best_estimator_.score(data_test, target_test)}')

        # --------------------
        # Confusion matrix print for every matrix
        # --------------------
        print(colors.positive + f'{name}: classification report train')
        print(classification_report(target_train, pred_train))
        print(colors.positive + f'{name}: classification report test')
        print(classification_report(target_test, pred_test))

        # print(multilabel_confusion_matrix(test_target_overall, pred_test))
        # analytics.plot_multi_label(multilabel_confusion_matrix(test_target_overall, pred_test), allergens_classifier)
        analytics.plot_multi_label(cfs_matrix_test[name], allergens_classifier, f'{name}')

        # ---------------------
        # TF-IDF top n features
        # ---------------------

        if name == "LR":
            print(f'TF-IDF top n features')
            analytics.print_topN_feature(tfv, gs.best_estimator_, allergens_classifier, 30)
            print("")

        try:
            analytics.plot_validation_curves(gs, params[name], name)
        except Exception as e:
            print(e)

        if name == "LR":
            analytics.plot_grid_search_validation_curve(gs, "estimator__max_iter")
            analytics.plot_grid_search_validation_curve(gs, "estimator__C")

        print(f'{colors.neutral}{name} done')

    print("Done")


if __name__ == "__main__":
    main()
