import numpy as np
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import glob
import os


class Classifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.tokenizer = WordPunctTokenizer()
        self.stopwords_list = stopwords.words('english')

        self.medical_bag_of_words = self.bag_of_words('Corpora/Medical/*.txt')
        self.non_medical_bag_of_words = self.bag_of_words('Corpora/NonMedical/*.txt')
        self.vocabulary = self.build_vocabulary()

        medical_count = len(os.listdir('Corpora/Medical'))
        non_medical_count = len(os.listdir('Corpora/NonMedical'))

        self.medical_prior = medical_count/(medical_count+non_medical_count)
        self.non_medical_prior = non_medical_count/(medical_count+non_medical_count)

        print(f"The priors of the two classes are:\n\tmedical: {self.medical_prior}\n\tnon medical: {self.non_medical_prior}")

    def bag_of_words(self, path):
        print(f"Building the bag of words for path: {path} ...")
        BoW = {}

        files = sorted(glob.glob(path))

        for file in files:
            with open(file, "r") as f:
                data = f.read()

                current_bag_of_words = self.normalize(data)

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

        file_bag_of_words = {}

        tokens = self.tokenizer.tokenize(data)  # tokenization

        for token in tokens:  # for each token check that it is significant (not a stopword and longer than 3)
            if token not in self.stopwords_list and len(token) > 3:
                stem = self.stemmer.stem(word=token, to_lowercase=True)  # stemming

                # the string '== Section Name ==' is used to divide sections, don't want to include this tokens
                if '=' not in stem:
                    if stem not in file_bag_of_words:
                        file_bag_of_words.update({stem: 1})
                    else:
                        value = file_bag_of_words.get(stem)
                        file_bag_of_words.update({stem: value + 1})

        return file_bag_of_words

    def build_vocabulary(self):
        vocab = {}
        print("Merging the bag of words into a single vocabulary ...\n")
        for word, count in self.medical_bag_of_words.items():
            vocab.update({word: count})

        for word, count in self.non_medical_bag_of_words.items():
            if word not in vocab.keys():
                vocab.update({word: count})
            else:
                vocab.update({word: vocab.get(word) + count})

        return vocab

    def classify(self, path):
        files = glob.glob(f"{path}/*.txt")

        labels = []

        for file in files:
            likelihoods = [self.non_medical_prior, self.medical_prior]
            with open(file, "r") as f:
                data = f.read()

                file_BoW = self.normalize(data)

                # actual classification by Naive Bayes technique
                for word in file_BoW:
                    if word in self.medical_bag_of_words:
                        likelihoods[1] += np.log(self.medical_bag_of_words.get(word) / self.vocabulary.get(word))
                    if word in self.non_medical_bag_of_words:
                        likelihoods[0] += np.log(self.non_medical_bag_of_words.get(word) / self.vocabulary.get(word))

            labels.append(np.argmax(likelihoods))

        return labels


if __name__ == "__main__":
    classifier = Classifier()

    print("\nClassifying the documents ...")
    predicted_labels = classifier.classify(path='Test/TestSet')

    true_labels = []
    with open('Test/test_labels.txt', 'r') as f:
        lines = f.readlines()

        for line in lines:
            true_labels.append(eval(line))

    correct = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i]:
            correct += 1

    print(f"The total number of correct labels is: {correct} \n\tAccuracy: {correct / len(predicted_labels)}")