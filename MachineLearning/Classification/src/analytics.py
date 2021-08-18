import itertools
from heapq import nlargest

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import colors


class Analytics:
    """
    Class for data analysis of recipes
    """

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    def word_frequency_for_data_frame(self, data_frame, recipe_df_labels, class_column_name, analyzed_column_name,
                                      top_n, save_path):
        """
        Get word frequency for data frame per label</br>

        Parameters
        ----------
        :param data_frame: pandas data frame</br>
        :param recipe_df_labels: all labels for classes</br>
        :param class_column_name: column name for which class the word frequency will be generated</br>
        :param analyzed_column_name column name which should be analyzed</br>
        :param top_n: how many top_n results should be shown</br>
        :param save_path: path where results should be saved to</br>
        :return:-
        """

        for label in recipe_df_labels:
            data_frame_filtered = data_frame.loc[data_frame[class_column_name] == label]
            word_freq = self.word_frequency(data_frame_filtered, analyzed_column_name, ',')
            self.top_n_word_occurrence(top_n, word_freq, label, save_path)

    def word_frequency(self, data_frame, column_name, split_separator):
        """
        Returns word count for a column in a data frame

        Parameters
        ----------
        :param split_separator: separator for strings</br>
        :param data_frame: dataframe to use</br>
        :param column_name: name of the analyzed column</br>
        :return: word frequencies
        """
        word_freq = {}
        for sentence in data_frame[column_name]:

            # check if ingredient list is a string or already a list
            if isinstance(sentence, str):
                splitted_sentence = sentence.split(split_separator)
            else:
                splitted_sentence = sentence

            for word in splitted_sentence:
                word_freq[word] = word_freq.get(word, 0) + 1

        return word_freq

    def top_n_word_occurrence(self, top_n, word_frequency_dict, label, save_path):
        """
        Plots top_n word occurrences of a word dictionary

        Parameters
        ----------
        :param label: label which should be used to calculate frequencies</br>
        :param top_n: how many top words should be displayed</br>
        :param word_frequency_dict: frequencies dictionary</br>
        :return: -
        """
        largest = nlargest(top_n, word_frequency_dict, key=word_frequency_dict.get)

        largest_dict = {}
        for large in largest:
            largest_dict[large] = word_frequency_dict[large]

        fig, ax = plt.subplots()

        plt.bar(range(len(largest_dict)), list(largest_dict.values()), align='center')
        plt.xticks(range(len(largest_dict)), list(largest_dict.keys()))
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.title("recipe style: " + label + " top " + str(top_n) + " words")
        plt.tight_layout()

        plt.savefig(save_path + label + "_top_" + str(top_n) + ".png")
        # plt.show()

    def recipe_style_class_distribution(self, df, class_column_name, save_path):
        """
        Display the distribution of different recipe styles.

        Parameters
        ----------
        :param df: dataframe</br>
        :param class_column_name: name of the class column</br>
        :param save_path: path of the saved plot</br>
        :return: -
        """
        x = df.groupby(class_column_name).size()

        # plot
        # plt.figure(figsize=(8, 4))
        ax = sns.barplot(x.index, x.values, alpha=0.8)
        plt.title("Class distribution", fontsize=18)
        plt.ylabel('# of Occurrences', fontsize=18)
        plt.xlabel('Type ', fontsize=18)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.tight_layout()

        plt.savefig(save_path)
        plt.show()

    def show_correlation_plot(self, correlations, data):
        """
        Plots a correlation plot and displays it.

        Parameters
        ----------
        :param correlations: Pairwise correlation of columns, excluding NA/null values</br>
        :param data: Original data frame</br>
        :return: -
        """

        # plot correlation matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)

        # ticks = np.arange(0, len(data.columns), 1)
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        ax.set_title("Correlation plot #" + str(len(data.columns)) + " of features", fontsize=14, pad=90)
        ax.set_xticklabels(data.columns, fontsize=14)
        ax.set_yticklabels(data.columns, fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="left")
        plt.tight_layout()
        # plt.gcf().subplots_adjust(top=0.15, bottom=0.1)

        plt.savefig('correlation_plot.png')
        plt.show()

    # function to use the matplotlib imgshow to create a heatmap confusion matrix
    # taken from here https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html and modified
    def plot_confusion_matrix(self, cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function is modified to show the color range as normalized to f1 score</br>
        both f1 score and class count are printed in the squares

        Parameters
        ----------
        :param cm: Confusion matrix
        :param classes: class labels
        :param normalize: if values should be normalized
        :param title: title of the plot
        :param cmap: color of the matrix
        :return:
        """

        if normalize:
            cm_normal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = plt.imshow(cm_normal, interpolation='nearest', cmap=cmap)
        # plt.grid(None)
        plt.rc('font', size=self.SMALL_SIZE)

        # plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(title)

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, horizontalalignment="right")
        plt.yticks(tick_marks, classes)

        # using the raw cm so the counts are printed on the heat map
        thresh = cm_normal.max() / 2.

        # set color based on threshold / white or black
        for i, j in itertools.product(range(cm_normal.shape[0]), range(cm_normal.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_normal[i, j] > thresh else "black")
            plt.text(j, i + 0.27, format(cm_normal[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm_normal[i, j] > thresh else "black")

        plt.ylabel('true label')
        plt.xlabel('predicted label')
        plt.tight_layout()
        plt.savefig("./analytics/confusion_matrix/confusion_matrix_" + title + ".png")
        plt.show()

    def show_confusion_matrix(self, cm, labels, cm_title=""):
        """
        Plots a confusion matrix with a given title, predicted and true label.

        Parameters
        ----------
        :param cm: array, shape = [n_classes, n_classes]</br>
        :param labels:</br>
        :param cm_title:  Title of the plot
        """

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               title='Confusion Matrix ' + cm_title,
               ylabel='True label',
               xlabel='Predicted label')

        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)

        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.tick_params(axis='both', which='minor', labelsize=9)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right",
                 rotation_mode="anchor")
        plt.tight_layout()

        plt.savefig("./analytics/confusion_matrix/confusion_matrix_" + cm_title + ".png")
        # plt.show()

    def plot_boxplot(self, labels, accuracies, title):
        """
        Plots a box plot for predicted accuracies of each model. Plot is saved under "./analytics/"

        Parameters
        ----------
        :param labels: labels for each model</br>
        :param accuracies: accuracies array</br>
        :param title: title of the plot</br>
        :return: -
        """
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.boxplot(accuracies)
        ax.set_xticklabels(labels)
        plt.tight_layout()
        plt.show()
        plt.savefig("./analytics/" + title + ".png")

    def print_topN_feature(self, vectorizer, clf, class_labels, n=20):
        """
        Prints features with the highest coefficient values, per class

        Parameters
        ----------
        :param vectorizer: vectorizer used to vectorize data</br>
        :param clf: loaded model</br>
        :param class_labels: labels for each class </br>
        :param n: number of features to be printed</br>
        :return: -
        """

        feature_names = vectorizer.get_feature_names()
        for i, class_label in enumerate(class_labels):
            top_n = np.argsort(clf.coef_[i])[-n:]
            print(colors.positive + "%s: %s" % (class_label, ", ".join(feature_names[j] for j in top_n)))
