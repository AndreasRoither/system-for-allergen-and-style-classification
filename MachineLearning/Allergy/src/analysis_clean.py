import fnmatch
import locale
import re
import time
from os import listdir

from collections import OrderedDict
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import colors
# set background color
plt.rcParams['axes.facecolor'] = 'white'
nltk.download('stopwords')

kaggle_datasets_processed_path = "../data/03_processed/kaggle/"
datasets_intermediate_path = "../data/02_intermediate/"
openf_datasets_processed_path = "../data/03_processed/openfoodfacts/"
openf_datasets_in_path = "../data/01_raw/"

allergens_classifier = [
    "en:gluten", "en:crustaceans", "en:eggs", "en:fish", "en:peanuts", "en:soybeans", "en:milk", "en:nuts", "en:celery",
    "en:mustard", "en:sesame-seeds", "en:sulphur-dioxide-and-sulphites", "en:lupin", "en:molluscs"
]

# Set locale for better number printing from 1000 -> 1.000
locale.setlocale(locale.LC_ALL, "de")
lemmatizer = WordNetLemmatizer()
enc = OneHotEncoder(handle_unknown='ignore')
label_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()
integer_encoded = label_encoder.fit_transform(allergens_classifier)

allergens = {}
found_allergens = {}
undesirables = set()
stops = set(stopwords.words('english'))


def load_undesirables():
    """
    load .txt files
    """
    # from:
    # https://www.kidspot.com.au/health/early-life-nutrition/features/40-common-cooking-terms-to-make-you-sound-like-a-pro-in-the-kitchen/news-story/2a1e7d892e68183bfc434e6c89e643bf
    # https://pos.toasttab.com/blog/culinary-terms
    with open('../data/03_processed/lists/common_cooking_terms.txt', encoding="utf-8") as f:
        undesirable_lemmatize_strip(f)

    # from:
    # https://en.wikipedia.org/wiki/Cooking_weights_and_measures
    # https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement
    with open('../data/03_processed/lists/measures.txt', encoding="utf-8") as f:
        undesirable_lemmatize_strip(f)

    # data leaks
    with open('../data/03_processed/lists/data_leaks.txt', encoding="utf-8") as f:
        undesirable_lemmatize_strip(f)


def undesirable_lemmatize_strip(f):
    data = re.sub(r"[\n\r\t]", "", f.read().lower()).split(",")
    data = [item.strip() for item in data]
    for item in data:
        lemmatized_terms = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in item.split(" ")]
        undesirables.add(" ".join(lemmatized_terms))


def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts

    Parameters
    ----------
    :return: tag from dictionary, N if nothing was found
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def splitDataFrameList(df, target_column, separator):
    """
    :param df: dataframe to split
    :param target_column: the column which contains the values
    :param separator: sep. symbol
    :return: dataframe with entries for the target column seperated -> each entry now has a new row instead "Test, Test" -> "Test", "Test" where "" is a row
    """

    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column, separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def exchange_characters(text):
    """
    Remove anything that is not a letter except commas and lowers the result </br>
    'and' has to be taken care of since it's mostly used to concatenate two ingredients

    Parameters
    ----------
    :param text: string </br>
    :return: lowered text with replaced characters
    """

    # remove all whitespace characters (space, tab, newline, return, form feed)
    result = " ".join(str(text).lower().split())
    result.replace(" and ", ",")
    result.replace(",,", ",")

    # remove B-23..
    result = re.sub(r"[a-z]-?\d+", " ", result)
    # remove non alphanumeric characters
    result = re.sub(r"[^a-z\s,]", " ", result)
    result = " ".join(result.split())
    return result


def regex_allergen_strip(text):
    """
    Match text that starts with 'en:', will be used to filter out english allergen tags: </br>
    en:gluten, en:crustaceans, en:eggs, en:fish, en:peanuts, en:soybeans, en:milk, en:nuts, en:celery, </br>
    en:mustard, en:sesame-seeds, en:sulphur-dioxide-and-sulphites, en:lupin, en:molluscs

    Parameters
    ----------
    :param text: </br>
    :return: list of matched strings
    """
    result = re.findall(r"en:[A-Za-z\-]+", text.lower())
    result = [x for x in result if x in allergens_classifier]
    final_result = ", ".join(map(str, result))

    return final_result


def replace_strip(text):
    """
    Remove newline, carriage return, tabs from string and split by ','

    Parameters
    ----------
    :param text: input text </br>
    :return: list of words
    """
    data = re.sub(r"[\n\r\t]", "", text.lower()).split(",")
    data = [item.strip() for item in data]
    data = list(filter(None, data))
    return data


def lemmatize_strip(text):
    """
    Lemmatize string, remove words if they are contained in the unwanted list and remove stopwords

    Parameters
    ----------
    :param text: input text </br>
    :return: list of lemmatized unique words
    """
    data = [item.strip() for item in text.split(",")]
    data = list(filter(None, data))
    filtered_words = []

    for item in data:

        filtered_terms = [word_filter(word) for word in item.split(" ")]
        filtered_terms = list(filter(None, dict.fromkeys(filtered_terms)))

        if filtered_terms:
            filtered_words.append(" ".join(filtered_terms))

    result = ", ".join(dict.fromkeys(filtered_words))
    return result


def word_filter(word: str):
    if len(word.strip()) == 0:
        return ""

    new_word = lemmatizer.lemmatize(word, get_wordnet_pos(word)).strip()

    if new_word in undesirables or new_word in stops:
        return ""

    return new_word


def plot_country_distribution(df: pd.DataFrame, save_path: str):
    food_countries = df[df['countries_en'].notnull()]
    food_countries = splitDataFrameList(food_countries, "countries_en", ",")
    countries = food_countries["countries_en"].value_counts()
    countries[:20][::-1].plot.barh()
    plt.show()
    plt.savefig(f'../data/04_analytics/{save_path}/country_distribution.png')


def initial_processing():
    """
    Load dataframe and drop columns which are not needed, then exports dataframe to .csv</br>

    Parameters
    ----------
    :return: preprocessed dataframe
    """
    print(colors.neutral + "Loading dataset...")

    df_open_food_facts = pd.read_csv(openf_datasets_in_path + "en.openfoodfacts.org.products.csv", sep="\t",
                                     low_memory=False)
    print(colors.positive + "Finished loading dataset...")
    print(colors.positive + "Loaded dataset shape " + str(df_open_food_facts.shape))  # > (1382804, 181)

    cols_to_keep = ["url", "product_name", "categories", "categories_en", "origins", "countries_en", "ingredients_text",
                    "allergens", "allergens_en", "traces", "traces_tags", "traces_en", "additives", "additives_en",
                    "main_category_en"]

    df_open_food_facts = df_open_food_facts[cols_to_keep]
    df_open_food_facts.to_csv(r"" + datasets_intermediate_path + "openfoodfacts_cols.csv", index=False, sep=";")
    print(colors.positive + "Exported dataset with fewer cols")

    return df_open_food_facts.reset_index(drop=True)


def dataset_analysis_pre_clean(df):
    """
    Show various statistics of the dataframe including: </br>
    shape, amount of duplicates, null/uniques of possible relevant columns, top n entry counts</br>

    Parameters
    ----------
    :param df: dataframe </br>
    :return: -  
    """
    print(colors.neutral + "Analysing dataset")

    print(colors.positive + "Loaded dataset shape " + str(df.shape))

    print(colors.neutral + "Lowercase dataset")
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)

    # head = "name\tn\n"
    fmt = "{name:<20s}\t{n:n}"

    print(colors.neutral + "Duplicates ")
    dup = pd.Series(df.duplicated()).where(lambda x: x).dropna()
    print(len(dup))
    print("\n")

    # drop duplicates as they are not valuable
    df = df.drop_duplicates(keep=False)

    print(colors.neutral + "Null values")
    print(fmt.format(name="product_name", n=len(df[df.product_name.isnull()])))
    print(fmt.format(name="categories ", n=len(df[df.categories.isnull()])))
    print(fmt.format(name="categories_en ", n=len(df[df.categories_en.isnull()])))
    print(fmt.format(name="origins ", n=len(df[df.origins.isnull()])))
    print(fmt.format(name="countries_en ", n=len(df[df.countries_en.isnull()])))
    print(fmt.format(name="ingredients_text ", n=len(df[df.ingredients_text.isnull()])))
    print(fmt.format(name="allergens ", n=len(df[df.allergens.isnull()])))
    print(fmt.format(name="allergens_en ", n=len(df[df.allergens_en.isnull()])))
    print(fmt.format(name="traces ", n=len(df[df.traces.isnull()])))
    print(fmt.format(name="traces_tags ", n=len(df[df.traces_tags.isnull()])))
    print(fmt.format(name="traces_en ", n=len(df[df.traces_en.isnull()])))
    print(fmt.format(name="additives ", n=len(df[df.additives.isnull()])))
    print(fmt.format(name="additives_en ", n=len(df[df.additives_en.isnull()])))
    print(fmt.format(name="main_category_en ", n=len(df[df.main_category_en.isnull()])))

    print("\n")
    print(colors.neutral + "Uniques")
    print(colors.positive + "product_name\t " + str(df.product_name.unique()))
    print(colors.positive + "categories\t " + str(df.categories.unique()))
    print(colors.positive + "categories_en\t " + str(df.categories_en.unique()))
    print(colors.positive + "origins\t " + str(df.origins.unique()))
    print(colors.positive + "origins\t " + str(df.countries_en.unique()))
    print(colors.positive + "ingredients_text\t " + str(df.ingredients_text.unique()))
    print(colors.positive + "allergens\t " + str(df.allergens.unique()))
    print(colors.positive + "traces\t " + str(df.traces.unique()))
    print(colors.positive + "traces_tags\t " + str(df.traces_tags.unique()))
    print(colors.positive + "traces_en\t " + str(df.traces_en.unique()))
    print(colors.positive + "additives\t " + str(df.additives.unique()))
    print(colors.positive + "additives_en\t " + str(df.additives_en.unique()))
    print(colors.positive + "main_category_en\t " + str(df.main_category_en.unique()))

    print("\n")
    print(colors.neutral + "Top N Origin counts")
    country_counts = df.drop_duplicates().origins.value_counts(dropna=False)
    print(country_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N Country_en counts")
    countries_en_counts = df.drop_duplicates().countries_en.value_counts(dropna=False)
    print(countries_en_counts.nlargest(10))

    plot_country_distribution(df, "preclean")

    print("\n")
    print(colors.neutral + "Top N allergen counts")
    allergen_counts = df["allergens"].groupby(df["allergens"]).count().sort_values(ascending=False)
    print(allergen_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N traces counts")
    traces_counts = df["traces"].groupby(df["traces"]).count().sort_values(ascending=False)
    print(traces_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N traces_en counts")
    traces_en_counts = df["traces_en"].groupby(df["traces_en"]).count().sort_values(ascending=False)
    print(traces_en_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N traces_tags counts")
    traces_tags_counts = df["traces_tags"].groupby(df["traces_tags"]).count().sort_values(ascending=False)
    print(traces_tags_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N ingredients_text counts")
    ingredients_text_counts = df["ingredients_text"].groupby(df["ingredients_text"]).count().sort_values(
        ascending=False)
    print(ingredients_text_counts.nlargest(10))

    print("\n")
    print(colors.neutral + "Top N additives counts")
    additives_counts = df["additives"].groupby(df["additives"]).count().sort_values(
        ascending=False)
    print(additives_counts.nlargest(10))

    print(colors.neutral + "Analysis end")


def dataset_analysis_post_clean(df):
    """
    Show various statistics of the dataframe including: </br>
    top allergen counts, shape of dataframe</br>

    Parameters
    ----------
    :param df:
    :return:
    """

    print("\n")
    print(colors.neutral + "Top N allergen counts")
    allergen_counts = df['allergens'].explode().value_counts()
    print(allergen_counts)

    distinct_values = df['allergens'].value_counts()
    print(distinct_values)

    plot_country_distribution(df, "afterclean")

def clean_allergen_set(df, save_title:str):
    df["ingredients_text"] = df["ingredients_text"].apply(exchange_characters)
    df["ingredients_text"] = df["ingredients_text"].apply(lemmatize_strip)

    # drop empty allergens again
    df = df[~df.ingredients_text.str.len().eq(0)]

    print(colors.neutral + "Finished cleaning dataframe")

    df = df.reset_index(drop=True)
    df.to_csv(r"" + datasets_intermediate_path + f"{save_title}.csv", index=False, sep="\t")

def clean_dataset(df, save_title:str):
    """
    Remove rows and further trim down columns which are not needed. </br>
    Rows that contain no information about allergens are dropped </br>

    Parameters
    ----------
    :param save_title: name of the csv file
    :param df: dataframe to clean </br>
    :return: cleaned dataframe
    """
    print(colors.neutral + "Cleaning dataframe..")

    cols_to_keep = ["product_name", "countries_en", "ingredients_text",
                    "allergens"]
    df = df[cols_to_keep]

    df['countries_en'] = df['countries_en'].str.lower()

    df = df[df['countries_en'].isin(['united states', 'united kingdom', 'australia', '+'])]

    # remove null rows
    df = df[df['product_name'].notnull() & df['allergens'].notnull()]

    df["allergens"] = df["allergens"].apply(regex_allergen_strip)
    df["product_name"] = df["product_name"].apply(exchange_characters)
    df["ingredients_text"] = df["ingredients_text"].apply(exchange_characters)
    df["ingredients_text"] = df["ingredients_text"].apply(lemmatize_strip)

    # drop empty allergens again
    df = df[~df.allergens.str.len().eq(0)]

    print(colors.neutral + "Finished cleaning dataframe")

    df = df.reset_index(drop=True)
    df.to_csv(r"" + datasets_intermediate_path + f"{save_title}.csv", index=False, sep="\t")

    return df


def multi_label_binarize_dataframe(df):
    """
    One hot encode a given dataframe

    Parameters
    ----------
    :param df: dataframe to one hot encode </br>
    :return: dataframe with one hot encoding
    """

    # strip whitespaces otherwise the mlb will have wrong classes with whitespaces
    df['allergens'] = df['allergens'].apply(lambda x: x.replace(" ", ""))
    df['allergens'] = df['allergens'].apply(lambda x: x.split(','))
    final_df = df.join(pd.DataFrame(mlb.fit_transform(df['allergens'].tolist()), columns=mlb.classes_, index=df.index))

    return final_df


def load_allergen_list():
    """
    Load .txt with allergens

    Parameters
    ----------
    :return: -
    """

    allergen_files = [f for f in sorted(listdir('../data/03_processed/allergens/')) if fnmatch.fnmatch(f, '*.txt')]

    for allergen_file in allergen_files:
        with open(f'../data/03_processed/allergens/{allergen_file}', encoding="utf8") as f:
            category = allergen_file.replace(".txt", "")
            allergens[category] = set()
            allergens[category].update(replace_strip(f.read()))


def check_value_exist(test_dict, value):
    """
    Checks if value exists in dictionary, if so returns key and value
    :param test_dict:
    :param value:
    :return:
    """
    for key, val in test_dict.items():
        if value in val:
            return key, val

    return False, False


def check_allergens(ingredients):
    """
    Checks for allergens and adds it to a dictionary

    Parameters
    ----------
    :param ingredients: list of ingredients </br>
    :return:
    """

    for ingredient_item in ingredients:
        ingredient_parts = list(filter(None, ingredient_item.split(" ")))

        for ingredient in ingredient_parts:
            key, value = check_value_exist(allergens, ingredient)

            if key:
                if key not in found_allergens.keys():
                    found_allergens[key] = set()
                found_allergens[key].add(ingredient_item)


def main():
    """
    Main entry function.

    Parameters
    ----------
    :return: -
    """
    print("> Allergens analysis <")

    # #################
    # initial processing
    # #################

    load_undesirables()

    # initial_processing()
    df_open_food_facts = pd.read_csv(datasets_intermediate_path + "openfoodfacts_cols.csv", low_memory=False, sep=";")
    df_allergens = pd.read_csv("../data/01_raw/custom_allergens.csv", low_memory=False, sep="\t")

    # #################
    # dataset analysis before clean
    # #################

    dataset_analysis_pre_clean(df_open_food_facts)

    # #################
    # dataset cleaning
    # #################

    start = time.time()
    clean_dataset(df_open_food_facts, save_title="openfoodfacts_cleaned")
    clean_allergen_set(df_allergens, save_title="custom_allergens")
    end = time.time()
    print(f'Cleaned in: {round((end - start) / 60, 2)}min')

    # #################
    # allergy test
    # #################

    # df_open_food_facts = pd.read_csv(openf_datasets_intermediate_path + "openfoodfacts_cleaned.csv", low_memory=False, sep=";")
    # recipe_kaggle_df_cleaned = pd.read_csv(kaggle_datasets_processed_path + "train.csv")

    # Safely evaluate an expression node or a Unicode or Latin-1 encoded string containing a Python expression. The string or node provided may only consist of the following
    # Python literal structures: strings, numbers, tuples, lists, dicts, booleans, and None.
    # in this case it is used to evaluate the lists for allergens and ingredients
    # df_open_food_facts.allergens = df_open_food_facts.allergens.apply(ast.literal_eval)
    # df_open_food_facts.ingredients_text = df_open_food_facts.ingredients_text.apply(ast.literal_eval)

    # load_allergen_list()

    # #################
    # dataset analysis after clean
    # #################

    dataset_analysis_post_clean(df_open_food_facts)
    #
    # for index, row in recipe_kaggle_df_cleaned.head(n=50).iterrows():
    #     ingredient_list = row.ingredients.split(",")
    #     check_allergens(ingredient_list)
    #
    # for key in found_allergens.keys():
    #     print(f'{colors.positive}{key}: {", ".join(found_allergens[key])}')

    # #################
    # one hot encoding
    # #################

    df_open_food_facts = pd.read_csv(datasets_intermediate_path + "openfoodfacts_cleaned.csv", low_memory=False, sep="\t")
    custom_allergens = pd.read_csv(datasets_intermediate_path + "custom_allergens.csv", low_memory=False, sep="\t")
    custom_allergens.drop('allergens', axis=1, inplace=True)

    df_hot_encoded = multi_label_binarize_dataframe(df_open_food_facts)
    df_hot_encoded.drop('allergens', axis=1, inplace=True)

    df_hot_encoded = df_hot_encoded.append(custom_allergens)
    df_hot_encoded.to_csv(r"" + openf_datasets_processed_path + "openfoodfacts.csv", index=False, sep='\t', encoding='utf-8')

    print("> Done <")


if __name__ == "__main__":
    main()
