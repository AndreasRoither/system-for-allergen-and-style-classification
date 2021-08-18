import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Uncomment if packages have not been downloaded
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')


class Cleaner:
    """
    Cleaner class to remove undesired words, characters etc from data frames
    """
    undesirables = []
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    def __init__(self):
        """
        Initialize cleaner class, load .txt files
        """
        # from:
        # https://www.kidspot.com.au/health/early-life-nutrition/features/40-common-cooking-terms-to-make-you-sound-like-a-pro-in-the-kitchen/news-story/2a1e7d892e68183bfc434e6c89e643bf
        # https://pos.toasttab.com/blog/culinary-terms
        with open('../data/03_processed/lists/common_cooking_terms.txt') as f:
            self.undesirables.extend(self.lemmatize_strip(f.read()))

        # from:
        # https://en.wikipedia.org/wiki/Cooking_weights_and_measures
        # https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement
        with open('../data/03_processed/lists/measures.txt') as f:
            self.undesirables.extend(self.lemmatize_strip(f.read()))

        # data leaks
        with open('../data/03_processed/lists/data_leaks.txt') as f:
            self.undesirables.extend(self.lemmatize_strip(f.read()))

    def append_style_column(self, data_frame, new_column_name, style_labels):
        """
        Appends the style column with the correct label to a dataframe

        Parameters
        ----------
        :param data_frame: data frame to which a column is added </br>
        :param new_column_name: the name of the new column</br>
        :param style_labels: labels for each column row</br>
        :return: dataframe
        """
        for df, label in zip(data_frame, style_labels):
            df[new_column_name] = label

    def clean_data_frame(self, data_frame, column_name):
        """
        Cleans a data frame by applying functions to a column of a data frame

        Parameters
        ----------
        :param data_frame: data frame</br>
        :param column_name: name of the column which should be cleaned</br>
        :return: dataframe
        """

        data_frame[column_name] = data_frame[column_name].apply(self.exchange_characters)
        data_frame[column_name] = data_frame[column_name].apply(self.remove_unwanted_words)

        return data_frame

    def lemmatize_strip(self, text):
        """
        Lemmatize and remove newline, carriage return, tabs, whitespaces

        Parameters
        ----------
        :param text: input text</br>
        :return: list of lemmatized unique words
        """
        data = re.sub(r"[\n\r\t\s]", "", text.lower()).split(",")
        data = list(filter(None, data))
        lemmatized_terms = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in data]
        return list(set(lemmatized_terms))

    def get_wordnet_pos(self, word):
        """
        Map POS tag to first character lemmatize() accepts

        Parameters
        ----------
        :return: wordnet tag, N if nothing is found
        """
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def exchange_characters(self, text):
        """
        Remove anything that is not a letter except commas and lowers the result</br>
        'and' has to be taken care of since it's mostly used to concatenate two ingredients</br>
        remove ```[^a-zA-Z\s,]```

        Parameters
        ----------
        :param text: string</br>
        :return: lowered text with exchanged characters
        """
        result = re.sub(r"[^a-zA-Z\s,]", "", str(text)).strip()
        result = re.sub(r" and ", ",", result)

        # remove all whitespace characters (space, tab, newline, return, formfeed)
        result = " ".join(result.split())
        return result.lower()

    def remove_unwanted_words(self, text):
        """
        Procedure:
        text is split according to ingredients
            - words are tokenized
            - stopwords are removed
            - word is lemmatized
            - undesired words are removed
            - words are joined again to preserve connected words like "soy sauce"

        Parameters
        ----------
        :param text: string to clean</br>
        :return: cleaned ingredients list
        """
        tokens = list(filter(None, text.lower().split(",")))
        filtered_tokens = []

        for token in tokens:
            word_tokens = word_tokenize(token)

            # remove stopwords first to reduce the word count
            filtered_words = [word for word in word_tokens if word not in self.stop_words]

            lemmatized_terms = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in filtered_words]

            # prevent double words for each ingredient, using dict since it keeps the insertion order unlike set
            lemmatized_terms = list(dict.fromkeys(lemmatized_terms))

            filtered_words = [word for word in lemmatized_terms if word not in self.undesirables]

            joined_token = ' '.join(filtered_words)
            if joined_token:
                filtered_tokens.append(joined_token)

        return ','.join(filtered_tokens)
