import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Preprocessor:
    """
    Processes data sent to the API
    """
    lemmatizer = WordNetLemmatizer()
    undesirables = set()
    stops = set(stopwords.words('english'))

    def __init__(self):
        self.__load_undesirables()

    def process(self, text: str) -> str:
        result = self.__exchange_characters(text)
        result = self.__lemmatize_strip(result)
        print(f"[+] Results: {result}")
        return result

    def __load_undesirables(self):
        """
        load .txt files
        """
        # from:
        # https://www.kidspot.com.au/health/early-life-nutrition/features/40-common-cooking-terms-to-make-you-sound-like-a-pro-in-the-kitchen/news-story/2a1e7d892e68183bfc434e6c89e643bf
        # https://pos.toasttab.com/blog/culinary-terms
        with open('../data/lists/common_cooking_terms.txt', encoding="utf-8") as f:
            self.__undesirable_lemmatize_strip(f)

        # from:
        # https://en.wikipedia.org/wiki/Cooking_weights_and_measures
        # https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement
        with open('../data/lists/measures.txt', encoding="utf-8") as f:
            self.__undesirable_lemmatize_strip(f)

        # data leaks
        with open('../data/lists/data_leaks.txt', encoding="utf-8") as f:
            self.__undesirable_lemmatize_strip(f)

    def __undesirable_lemmatize_strip(self, f):
        data = re.sub(r"[\n\r\t]", "", f.read().lower()).split(",")
        data = [item.strip() for item in data]
        for item in data:
            lemmatized_terms = [self.lemmatizer.lemmatize(word, self.__get_wordnet_pos(word)) for word in item.split(" ")]
            self.undesirables.add(" ".join(lemmatized_terms))

    def __get_wordnet_pos(self, word):
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

    def __word_filter(self, word: str):
        if len(word.strip()) == 0:
            return ""

        new_word = self.lemmatizer.lemmatize(word, self.__get_wordnet_pos(word)).strip()

        if new_word in self.undesirables:
            return ""

        if new_word in self.stops:
            return ""

        return new_word

    def __exchange_characters(self, text: str) -> str:
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

    def __lemmatize_strip(self, text):
        """
        Lemmatize string, remove words if they are contained in the unwanted list and remove stopwords

        Parameters
        ----------
        :param text: input text </br>
        :return: list of processed unique words
        """
        data = [item.strip() for item in text.split(",")]
        data = list(filter(None, data))
        filtered_words = []

        for item in data:
            filtered_terms = [self.__word_filter(word) for word in item.split(" ")]
            filtered_terms = list(filter(None, dict.fromkeys(filtered_terms)))

            if filtered_terms:
                filtered_words.append(" ".join(filtered_terms))

        result = ", ".join(dict.fromkeys(filtered_words))
        return result
