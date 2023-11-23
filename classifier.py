import numpy as np
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import glob
import os
from sklearn import metrics


class Classifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tokenizer = WordPunctTokenizer()
        self.stopwords_list = stopwords.words('english')

        # build bag of words and vocabulary
        self.medical_bag_of_words = self.bag_of_words('Corpora/Medical/*.txt')
        self.non_medical_bag_of_words = self.bag_of_words('Corpora/NonMedical/*.txt')
        self.vocabulary = self.build_vocabulary()

        medical_count = len(os.listdir('Corpora/Medical'))
        non_medical_count = len(os.listdir('Corpora/NonMedical'))

        # compute the priors relatively to the available training set
        self.medical_prior = medical_count/(medical_count+non_medical_count)
        self.non_medical_prior = non_medical_count/(medical_count+non_medical_count)

        print(f"The priors of the two classes are:\n\tmedical: {self.medical_prior}\n\tnon medical: {self.non_medical_prior}")

    def bag_of_words(self, path):
        print(f"Building the bag of words for path: {path} ...")

        BoW = {}    # category related bag of words

        files = sorted(glob.glob(path))

        # for each file
        for file in files:
            with open(file, "r") as f:
                data = f.read()

                current_bag_of_words = self.normalize(data)     # compute the bag of words representation of the file

                # merge this bag of word into the category bag_of_word
                for word, count in current_bag_of_words.items():
                    if word not in BoW.keys():
                        BoW.update({word: count})
                    else:
                        BoW.update({word: (BoW.get(word) + count)})

        print(f"\tDone: \tThe corresponding bag of words contains {len(BoW)} records\n")
        return BoW

    def normalize(self, data):
        """
        Normalizes a file by means of tokenization, stemming, stopwords elimination, returning its representation as a Bag of Words

        :param data: the file we want to normalize represented as a string
        :return: the bag of words representation of the input file
        """

        file_bag_of_words = {}  # file representation as a bag of words

        tokens = self.tokenizer.tokenize(data)  # tokenization

        for token in tokens:  # for each token check that it is significant (not a stopword and longer than 3)
            if token not in self.stopwords_list and len(token) > 3:
                stem = self.stemmer.stem(word=token, to_lowercase=True)  # stemming

                # the string '== Section Name ==' is used to divide sections, don't want to include this stems
                if '=' not in stem:
                    # if we find a new stem, simply add it to the BoW with count 1
                    if stem not in file_bag_of_words.keys():
                        file_bag_of_words.update({stem: 1})
                    else:   # otherwise just update the count
                        value = file_bag_of_words.get(stem)
                        file_bag_of_words.update({stem: value + 1})

        return file_bag_of_words

    def build_vocabulary(self):
        vocab = {}
        print("Merging the bag of words into a single vocabulary ...\n")
        # put all the elements from the medical bag of words in the vocabulary
        for word, count in self.medical_bag_of_words.items():
            vocab.update({word: count})

        # put all the elements from the non-medical bag of words in the vocabulary
        for word, count in self.non_medical_bag_of_words.items():
            if word not in vocab.keys():
                vocab.update({word: count})
            else:
                vocab.update({word: vocab.get(word) + count})

        return vocab

    def classify(self, path):
        files = glob.glob(f"{path}/*.txt")

        # the output will be a dictionary, associating for each filename a particular label
        labels = {}

        for file in files:
            likelihoods = [self.non_medical_prior, self.medical_prior]      # will contain the probabilities of attaining to a particular class for each test documents
            with open(file, "r") as f:
                data = f.read()

                file_BoW = self.normalize(data)     # represent the input document by its bag of words

                # classification by Naive Bayes
                for word in file_BoW:
                    if word in self.medical_bag_of_words:
                        likelihoods[1] += np.log(self.medical_bag_of_words.get(word) / self.vocabulary.get(word))
                    if word in self.non_medical_bag_of_words:
                        likelihoods[0] += np.log(self.non_medical_bag_of_words.get(word) / self.vocabulary.get(word))

            # the document's class is the one that maximizes its likelihood to attain to a particular class
            labels.update({f.name: np.argmax(likelihoods)})

        return labels


if __name__ == "__main__":
    classifier = Classifier()

    print("\nClassifying the documents ...")
    predicted_labels = classifier.classify(path='Test/TestSet')
    print(len(predicted_labels))

    # sort the dictionary of predicted labels by key
    predicted_labels = dict(sorted(predicted_labels.items(), key=lambda item: item[1]))

    with open('Test/test_labels.txt', 'r') as f:
        data = f.read()

        true_labels = eval(data)
        # sort the true labels by key
        true_labels = dict(sorted(true_labels.items(), key=lambda item: item[1]))

    # only get the labels to show results
    predicted_labels = list(predicted_labels.values())
    true_labels = list(true_labels.values())

    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    print(f"Following will be displayed the confusion matrix: \n{confusion_matrix}")

    print(f"\nTrue medicals: {confusion_matrix[1][1]}\nTrue non medicals: {confusion_matrix[0][0]}")
    print(f"False negatives: {confusion_matrix[1][0]}\nFalse positives: {confusion_matrix[0][1]}")

    print(f"\nThe average precision is: {metrics.average_precision_score(true_labels, predicted_labels)}")
    print(f"The average recall is: {metrics.recall_score(true_labels, predicted_labels)}")
    print(f"The accuracy is: {metrics.accuracy_score(true_labels, predicted_labels)}")



