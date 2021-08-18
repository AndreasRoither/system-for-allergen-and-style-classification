from pathlib import Path

import pandas as pd

from src import colors
from src.analytics import Analytics
from src.classification import Classification
from src.cleaner import Cleaner
from src.recipe_style import RecipeStyle

recipe_dfs = []
recipe_df_cleaned = []
top_n = 40

# allrecipes sets from webscraper
allrecipes_datasets_in_path = "../data/01_raw/allrecipes/"
allrecipes_datasets_out_path = "../data/03_processed/allrecipes/"
allrecipes_save_path_pre = "./analytics/pre_clean/allrecipes/top_" + str(top_n) + "_ingredients/"
allrecipes_save_path_post = "./analytics/post_clean/allrecipes/top_" + str(top_n) + "_ingredients/"

# kaggle dataset
# https://www.kaggle.com/c/whats-cooking/data
kaggle_datasets_in_path = "../data/01_raw/kaggle/"
kaggle_datasets_out_path = "../data/03_processed/kaggle/"
kaggle_save_path_pre = "./analytics/pre_clean/kaggle/top_" + str(top_n) + "_ingredients/"
kaggle_save_path_post = "./analytics/post_clean/kaggle/top_" + str(top_n) + "_ingredients/"

recipe_ingredients_column = "recipe_ingredients"
recipe_ingredients_column_kaggle = "ingredients"
recipe_df_labels = [e.value for e in RecipeStyle]


def allrecipes_load_in_csv(path, data_frame_list):
    """
    Load allrecipes csvs</br>

    Parameters
    ----------
    :param path: Path to csvs</br>
    :param data_frame_list: which data_frame to append to</br>
    :return: -
    """
    df_diabetic = pd.read_csv(path + "healthy-recipes_diabetic_.csv")
    df_dairy_free = pd.read_csv(path + "healthy-recipes_dairy-free_.csv")
    df_sugar_free = pd.read_csv(path + "healthy-recipes_sugar-free_.csv")
    df_gluten_free = pd.read_csv(path + "healthy-recipes_gluten-free_.csv")
    df_low_cholesterol = pd.read_csv(path + "healthy-recipes_low-cholesterol_.csv")
    df_mediterranean = pd.read_csv(path + "healthy-recipes_mediterranean-diet_.csv")
    df_chinese = pd.read_csv(path + "world-cuisine_asian_chinese_.csv")
    df_indian = pd.read_csv(path + "world-cuisine_asian_indian_.csv")
    df_japanese = pd.read_csv(path + "world-cuisine_asian_japanese_.csv")
    df_korean = pd.read_csv(path + "world-cuisine_asian_korean_.csv")
    df_thai = pd.read_csv(path + "world-cuisine_asian_thai_.csv")
    df_european = pd.read_csv(path + "world-cuisine_european_.csv")
    df_italian = pd.read_csv(path + "world-cuisine_european_italian_.csv")
    df_american = pd.read_csv(path + "world-cuisine_latin-american_.csv")
    df_mexican = pd.read_csv(path + "world-cuisine_latin-american_mexican_.csv")
    df_eastern = pd.read_csv(path + "world-cuisine_middle-eastern_.csv")

    data_frame_list.extend(
        (df_diabetic, df_dairy_free, df_sugar_free, df_gluten_free, df_low_cholesterol, df_mediterranean,
         df_chinese, df_indian, df_japanese, df_korean, df_thai, df_european, df_italian, df_american,
         df_mexican, df_eastern))


def export_allrecipes_csv(data_frame):
    data_frame.to_csv(r"" + allrecipes_datasets_out_path + "full.csv", index=False)


def export_kaggle_csv(df_train):
    df_train.to_csv(r"" + kaggle_datasets_out_path + "train.csv", index=False)
    # df_test.to_csv(r"" + allrecipes_datasets_out_path + "test.csv", index=False)


def load_tests(classification, data_test, target_test):
    """
    Load test models and test against data to give accuracy

    Parameters
    ----------
    :param classification: Classification class</br>
    :param data_test: vectorized test data -> data to test against</br>
    :param target_test: vectorized target data -> data to verify test</br>
    :return: -
    """

    names = [
        # "KNN",
        # "SVC",
        'LinearSVC',
        # "CART",
        # "RF",
        # "XGBoost",
        # "NN",
        # "LDA",
        "LR",
        # "NB"
    ]

    paths = {
        'KNN': "../models/best_model_KNN_2020_05_19_08_13_03.161924.sav",
        'SVC': "../models/best_model_SVC_2020_05_19_08_20_23.423507.sav",
        'LinearSVC': "../models/best_model_LinearSVC_2021_01_25_11_39_00.sav",
        'CART': "../models/best_model_CART_2020_05_19_08_20_26.169106.sav",
        'RF': "../models/best_model_RF_2020_05_19_10_33_42.123227.sav",
        'XGBoost': "../models/best_model_XGBoost_2020_06_01_17_54_37.634103.sav",
        'NN': "../models/best_model_NN_2020_05_19_10_38_23.252567.sav",
        'LDA': "../models/",
        'LR': "../models/best_model_LR_2021_01_25_11_58_00.sav",
        'NB': "../models/best_model_NB_2020_05_19_12_04_17.694805.sav"
    }

    for name in names:
        path = paths[name]
        loaded_model = classification.load_test(path, name, data_test, target_test)

        print("\n" + colors.neutral + "Most important features for each class")
        class_labels = classification.le.inverse_transform(loaded_model.classes_)
        classification.analytics.print_topN_feature(classification.vectorizer, loaded_model[1], class_labels, 40)


def main():
    """
    Main entry point for classification project</br>
    :return: -
    """
    print("> Recipe Analytics & training <")
    analytics = Analytics()
    cleaner = Cleaner()
    classification = Classification()

    global recipe_df_cleaned

    # Make sure paths exist to save analytics
    Path(kaggle_save_path_pre).mkdir(parents=True, exist_ok=True)
    Path(kaggle_save_path_post).mkdir(parents=True, exist_ok=True)
    Path(allrecipes_save_path_pre).mkdir(parents=True, exist_ok=True)
    Path(allrecipes_save_path_post).mkdir(parents=True, exist_ok=True)
    Path(kaggle_datasets_out_path).mkdir(parents=True, exist_ok=True)
    Path(allrecipes_datasets_out_path).mkdir(parents=True, exist_ok=True)

    # -----------------
    # Data Cleaning Allrecipes
    # -----------------

    # analytics pre clean
    # print(colors.neutral + "Loading data..")
    # allrecipes_load_in_csv(allrecipes_datasets_in_path, recipe_dfs)
    # analytics.word_frequency_for_data_frames(recipe_df_cleaned, recipe_df_labels, recipe_ingredients_column, 40, "allrecipes")

    # print(colors.neutral + "Cleaning data..")
    # recipe_df_cleaned = cleaner.clean_data_frames(recipe_dfs, recipe_ingredients_column)
    # analytics.word_frequency_for_data_frames(recipe_df_cleaned, recipe_df_labels, recipe_ingredients_column, 40, "allrecipes")

    # append extra column category and
    # print(colors.neutral + "Exporting data..")
    # cleaner.append_style_column(recipe_df_cleaned, "category", recipe_df_labels)
    # export_allrecipes_csv(pd.concat(recipe_df_cleaned))

    # -----------------
    # After clean allrecipes
    # -----------------

    # recipe_df_cleaned = pd.read_csv(allrecipes_datasets_out_path + "full.csv")
    # check class distribution
    # analytics.recipe_style_class_distribution(recipe_df_cleaned, 5)

    # prepare for train by splitting and vectorizing data
    # data_train, data_test, target_train, target_test = classification.train_test_split(recipe_df_cleaned)
    # data_train_vec, data_test_vec = classification.vectorize_data(data_train, data_test)

    # train and test with data
    # classification.train_test(data_train_vec, data_test_vec, target_train, target_test)

    # load_tests(classification, data_test_vec, target_test)

    # -----------------
    # Pre clean kaggle
    # -----------------

    # analytics pre clean
    # print(colors.neutral + "Loading data..")
    # df_kaggle = pd.read_json(kaggle_datasets_in_path + "train.json")
    # kaggle_labels = df_kaggle['cuisine'].unique()
    # data_test = pd.read_json(kaggle_datasets_in_path + "test.json")

    # analytics.word_frequency_for_data_frame(df_kaggle, kaggle_labels, "cuisine", recipe_ingredients_column_kaggle, top_n, kaggle_save_path_pre)

    # print(colors.neutral + "Cleaning data..")
    # recipe_kaggle_df_cleaned = cleaner.clean_data_frame(df_kaggle, recipe_ingredients_column_kaggle)
    # analytics.word_frequency_for_data_frame(recipe_kaggle_df_cleaned, kaggle_labels, "cuisine",
    #                                        recipe_ingredients_column_kaggle, top_n, kaggle_save_path_post)

    # export_kaggle_csv(recipe_kaggle_df_cleaned)

    # -----------------
    # After clean kaggle
    # -----------------

    recipe_kaggle_df_cleaned = pd.read_csv(kaggle_datasets_out_path + "train.csv")

    # analytics.recipe_style_class_distribution(recipe_kaggle_df_cleaned, "cuisine",
    #                                          "./analytics/class_distribution/class_distribution_kaggle_train.png")

    # prepare for train by splitting and vectorizing data
    data_train, data_test, target_train, target_test = classification.train_test_split(recipe_kaggle_df_cleaned, 2, 1)
    data_train_vec, data_test_vec = classification.vectorize_data(data_train, data_test)

    # classification.export_lf_classes()

    # train and test with data
    # classification.train_test(data_train_vec, data_test_vec, target_train, target_test,
    #                          [recipe_kaggle_df_cleaned['cuisine'].unique().size], advanced_analytics=False)

    # -----------------
    # Load models and test accuracy
    # -----------------
    classification.fit_label_encoder(recipe_kaggle_df_cleaned, 1)
    load_tests(classification, data_test_vec, target_test)

    print(colors.positive + "Done")


if __name__ == "__main__":
    main()
