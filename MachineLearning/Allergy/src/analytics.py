import itertools
import numbers
from heapq import nlargest

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc
from wordcloud import WordCloud
import pandas as pd

from src import colors

# set background color
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.dpi'] = 360
sns.set(rc={'axes.facecolor': '#FFFFFF', 'figure.facecolor': '#FFFFFF'})


class Analytics:
    """
    Class for data analysis of recipes
    """

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    def pooled_var(self, stds):
        # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
        n = 5  # size of each group
        return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))

    def word_frequency_for_data_frame(self, data_frame, df_labels, analyzed_column_name,
                                      top_n, save_path):
        """
        Get word frequency for data frame per label</br>

        Parameters
        ----------
        :param data_frame: pandas data frame</br>
        :param df_labels: all labels for classes</br>
        :param analyzed_column_name column name which should be analyzed</br>
        :param top_n: how many top_n results should be shown</br>
        :param save_path: path where results should be saved to</br>
        :return:-
        """

        for label in df_labels:
            data_frame_filtered = data_frame.loc[data_frame[df_labels] == label]
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
        plt.title(f'recipe style: {label} top {str(top_n)} words')
        plt.tight_layout()

        plt.savefig(f'../data/04_analytics/{label}_top_{str(top_n)}.png')
        # plt.show()

    def wordcloud(self, df, column_name, word_column):
        subset = df[df[column_name] == 1]
        text = map(str, subset[word_column].values)
        wc = WordCloud(background_color="black", max_words=2000)
        wc.generate(" ".join(text))
        plt.figure(figsize=(6, 3), dpi=200)
        plt.axis("off")
        plt.title(f"Words frequented in {column_name}", fontsize=20)
        plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)
        plt.savefig(f'../data/04_analytics/{column_name.replace(":", "_")}_wordcloud.png')
        plt.show()

    def class_distribution(self, df, save_title="class_distribution"):
        """
        Display the distribution of different recipe styles.

        Parameters
        ----------
        :param df: dataframe</br>
        :param save_title: title of the png saved
        :return: -
        """
        x = df.iloc[:, :].sum()
        # plot
        plt.figure(figsize=(8, 5))
        sns.set_theme(style="whitegrid")
        sns.set_style("white")
        ax = sns.barplot(x.index, x.values, alpha=0.8)
        sns.despine()
        # plt.title("# per class")
        plt.ylabel('# of Occurrences', fontsize=12)
        plt.xlabel('Type ', fontsize=12)

        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right",
                 rotation_mode="anchor")

        # adding the text labels
        rects = ax.patches
        labels = x.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

        plt.tight_layout()
        plt.autoscale()
        plt.savefig(f"../data/04_analytics/{save_title}.png")
        plt.show()

    def plot_roc_auc(self, y_true, predict_proba, model_name, allergen):
        fpr1, tpr1, thresh1 = roc_curve(y_true, predict_proba, pos_label=1)
        random_probs = [0 for i in range(len(y_true))]
        p_fpr, p_tpr, _ = roc_curve(y_true, random_probs, pos_label=1)

        plt.style.use('seaborn')

        # plot roc curves
        plt.plot(fpr1, tpr1, linestyle='--', color='orange', label=model_name)
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title(f'ROC curve - {allergen}')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.savefig(f"../data/04_analytics/roc/roc_{model_name}_{allergen}.png")
        plt.show()

    # https://matthewbilyeu.com/blog/2019-02-05/validation-curve-plot-from-gridsearchcv-results
    def plot_grid_search_validation_curve(self, grid, param_to_vary, title='Validation Curve', ylim=None, xlim=None, log=None):
        """
        Plots train and cross-validation scores from a GridSearchCV instance's
        best params while varying one of those params.
        :param grid:
        :param param_to_vary:
        :param title:
        :param ylim:
        :param xlim:
        :param log:
        :return:
        """

        df_cv_results = pd.DataFrame(grid.cv_results_)
        train_scores_mean = df_cv_results['mean_train_score']
        valid_scores_mean = df_cv_results['mean_test_score']
        train_scores_std = df_cv_results['std_train_score']
        valid_scores_std = df_cv_results['std_test_score']

        param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
        param_ranges = [grid.param_grid[p[6:]] for p in param_cols]
        param_ranges_lengths = [len(pr) for pr in param_ranges]

        train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_lengths)
        valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_lengths)
        train_scores_std = np.array(train_scores_std).reshape(*param_ranges_lengths)
        valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_lengths)

        param_to_vary_idx = param_cols.index('param_{}'.format(param_to_vary))

        slices = []
        for idx, param in enumerate(grid.best_params_):
            if (idx == param_to_vary_idx):
                slices.append(slice(None))
                continue
            best_param_val = grid.best_params_[param]
            idx_of_best_param = 0
            if isinstance(param_ranges[idx], np.ndarray):
                idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
            else:
                idx_of_best_param = param_ranges[idx].index(best_param_val)
            slices.append(idx_of_best_param)

        train_scores_mean = train_scores_mean[tuple(slices)]
        valid_scores_mean = valid_scores_mean[tuple(slices)]
        train_scores_std = train_scores_std[tuple(slices)]
        valid_scores_std = valid_scores_std[tuple(slices)]

        plt.clf()

        plt.title(title)
        plt.xlabel(param_to_vary)
        plt.ylabel('Score')

        if (ylim is None):
            plt.ylim(0.0, 1.1)
        else:
            plt.ylim(*ylim)

        if (not (xlim is None)):
            plt.xlim(*xlim)

        lw = 2

        plot_fn = plt.plot
        if log:
            plot_fn = plt.semilogx

        param_range = param_ranges[param_to_vary_idx]

        if not isinstance(param_range[0], numbers.Number):
            param_range = [str(x) for x in param_range]

        plot_fn(param_range, train_scores_mean, label='Training score', color='r', lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r', lw=lw)

        plot_fn(param_range, valid_scores_mean, label='Cross-validation score', color='b', lw=lw)
        plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color='b', lw=lw)

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def plot_validation_curves(self, gs, grid_params, classifier_name=""):
        """
        Plot validation curves for each param in the param grid
        :param gs: gridsearchcv object
        :param grid_params:param grid
        :return:
        """
        df = pd.DataFrame(gs.cv_results_)
        results = ['mean_test_score',
                   'mean_train_score',
                   'std_test_score',
                   'std_train_score']

        fig, axes = plt.subplots(1, len(grid_params),
                                 figsize=(6 * len(grid_params), 12),
                                 sharey='row')
        axes[0].set_ylabel("Score", fontsize=25)

        for idx, (param_name, param_range) in enumerate(grid_params.items()):
            grouped_df = df.groupby(f'param_{param_name}')[results] \
                .agg({'mean_train_score': 'mean',
                      'mean_test_score': 'mean',
                      'std_train_score': self.pooled_var,
                      'std_test_score': self.pooled_var})

            previous_group = df.groupby(f'param_{param_name}')[results]
            # rotation=20, horizontalalignment="right"
            axes[idx].set_xlabel(param_name.replace("estimator__", ""), fontsize=30)
            axes[idx].set_ylim(0.0, 1.1)
            lw = 2

            if param_name == "hidden_layer_size":
                continue
                # param_range = [''.join(i) for i in param_range]

            axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                           color="darkorange", lw=lw)
            axes[idx].fill_between(param_range, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                                   grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                                   color="darkorange", lw=lw)
            axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                           color="navy", lw=lw)
            axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                                   grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                                   color="navy", lw=lw)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(f'Validation curves {classifier_name}', fontsize=40)
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.05), loc="lower center", fontsize=20,
                   bbox_transform=plt.gcf().transFigure)

        fig.subplots_adjust(bottom=0.35, top=0.85)
        # plt.tight_layout()
        plt.savefig(f"../data/04_analytics/validation_curves_{classifier_name}.png")
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

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(correlations,
                         xticklabels=correlations.columns.values,
                         yticklabels=correlations.columns.values, annot=True)

        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right",
                 rotation_mode="anchor")

        plt.tight_layout()
        plt.autoscale()
        plt.savefig("../data/04_analytics/correlation_plot.png")
        plt.show()

    # taken from https://github.com/MDamiano/mlcm
    def confusion_matrix(self, y_test, y_pred):
        if len(y_test.shape) != 2:
            raise IOError('y_test must be a 2D array (Matrix)')
        elif len(y_pred.shape) != 2:
            raise IOError('y_pred must be a 2D array (Matrix)')

        cm = np.zeros((y_test.shape[1], y_test.shape[1]))

        for obs in range(0, len(y_pred[:, 0])):
            j = y_pred[obs, :].argmax()
            i = y_test[obs, :].argmax()
            cm[i, j] += 1

        accuracy = 0.0
        for i in range(0, cm.shape[1]):
            accuracy += cm[i, i]
        accuracy /= len(y_test.argmax(axis=1))
        print("Accuracy on the test-set: " + str(accuracy))

        return cm

    # taken from https://github.com/MDamiano/mlcm
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Reds):
        plt.ion()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            if np.isnan(cm).any():
                np.nan_to_num(cm, copy=False)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'../data/04_analytics/confusion_matrix_{title}.png')
        plt.ioff()

    # taken from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    def print_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=14):
        """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

        Arguments
        ---------
        confusion_matrix: numpy.ndarray
            The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
            Similarly constructed ndarrays can also be used.
        class_names: list
            An ordered list of class names, in the order they index the given confusion matrix.
        figsize: tuple
            A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
            the second determining the vertical size. Defaults to (10,7).
        fontsize: int
            Font size for axes labels. Defaults to 14.

        Returns
        -------
        matplotlib.figure.Figure
            The resulting confusion matrix figure
        """
        df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
        )

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        axes.set_xlabel('Predicted label')
        axes.set_ylabel('True label')
        axes.set_title(f'{class_label}')

    def plot_multi_label(self, confusion_matrix, class_labels, title="Confusion matrix"):
        fig, ax = plt.subplots(4, 4, figsize=(12, 7))

        for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrix, class_labels):
            self.print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

        fig.tight_layout()
        plt.savefig(f'../data/04_analytics/multi_label_confusion_matrix_{title}.png')
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
        plt.savefig(f'../data/04_analytics/confusion_matrix_{title}.png')
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

        plt.savefig(f'../data/04_analytics/confusion_matrix_{cm_title}.png')
        plt.show()

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
        plt.savefig(f'../data/04_analytics/{title}.png')

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
