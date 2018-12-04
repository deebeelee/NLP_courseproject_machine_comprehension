import os
import json
import pickle
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression


class Andrew:

    def __init__(self, data_filepath):
        path_to_data = os.path.join(os.getcwd(), "development.json")
        current_loc = os.getcwd()
        data_path, data_file_name = path_to_data.rsplit(os.sep, 1)
        os.chdir(data_path)
        with open(data_file_name, "r") as fileHandle:
            self.data = json.load(fileHandle)
        os.chdir(current_loc)

    # Count the number of things (nouns, verbs, words) in a question. 
    def count_things(self, question):
        things = [0, 0]
        noun_list = []
        verb_list = []
        word_list = []
        special_list = []
        tokens = nltk.word_tokenize(question)
        pos_tags = nltk.pos_tag(tokens)
        for tag in pos_tags:
            word_list.append(tag[0])
            if (tag[1][0:2] == "NN"):
                noun_list.append(tag[0])
                special_list.append(tag[0])
            elif (tag[1][0:2] == "VB"):
                verb_list.append(tag[0])
                special_list.append(tag[0])
        return noun_list, verb_list, word_list, special_list

    # Given a list of nouns/verbs (things) that appear in a question, identify the
    # number of matches (percent of matches) of the exact appearance in the context.
    # Also returns the sentence with the highest incidence of matches. 
    def count_matches(self, thing_list, context):
        noun_list = thing_list[0]
        verb_list = thing_list[1]
        word_list = thing_list[2]
        max_score_sentence = ""
        max_match_count = -1.0
        sentences = context.split(".")
        for sentence in sentences:
            noun_count = 0.0
            verb_count = 0.0
            word_count = 0.0
            for noun in noun_list:
                if (sentence.find(noun) > 0):
                    noun_count += 1.0
            for verb in verb_list:
                if (sentence.find(verb) > 0):
                    verb_count += 1.0
            for word in word_list:
                if (sentence.find(word) > 0):
                    word_count += 1.0
            if (noun_count + verb_count + word_count > max_match_count):
                matches = [noun_count / (len(noun_list) + 1.0), verb_count / (len(verb_list) + 1.0), word_count / (len(word_list) + 1.0)]
                max_match_count = noun_count + verb_count + word_count
                max_score_sentence = sentence
        return matches, max_score_sentence

    # This detects if the order in which nounes and verbs (special keywords) appear in the same order
    # in the question as it does in the context (in the sentence with the most keyword matches).
    # For every word that doesn't appear in an expected order, it adds 1 to the "inversion" count. 
    def detect_inversion(self, special_list, max_score_sentence):
        index_array = []
        for word in special_list:
            if (max_score_sentence.find(word) > 0):
                index_array.append(max_score_sentence.find(word))
        inversion_count = 0
        previous_index = -1
        for index in index_array:
            if (index < previous_index):
                inversion_count += 1
            previous_index = index
        return inversion_count

    def featurize(self, question, context):
        features = []
        noun_list, verb_list, word_list, special_list = self.count_things(question)
        features.append(len(noun_list)) #0
        features.append(len(verb_list)) #1
        features.append(len(word_list)) #2
        matches, max_score_sentence = self.count_matches([noun_list, verb_list, word_list], context)
        features.append(matches[0]) #3
        features.append(matches[1]) #4
        features.append(matches[2]) #5
        inversions = self.detect_inversion(special_list, max_score_sentence)
        features.append(inversions) #6
        return features

    # Extract X and Y vectors from the data to use for logistic regression
    def vectorize(self, x_filename, y_filename):
        X = []
        Y = []
        for title in self.data["data"]:
            for paragraph in title["paragraphs"]:
                context = paragraph["context"]
                for question in paragraph["qas"]:
                    if question["is_impossible"]:
                        Y.append(0)
                    else:
                        Y.append(1)
                    X.append(self.featurize(question["question"], context))
        with open(x_filename, 'wb') as fp:
            pickle.dump(X, fp)
        with open(y_filename, 'wb') as fp:
            pickle.dump(Y, fp)
        return X, Y

    def open_vectors(self, x_filename, y_filename):
        X = pickle.load(open(x_filename, 'rb'))
        Y = pickle.load(open(y_filename, 'rb'))
        return X, Y

    def train_LR(self, X, Y, model_filename):
        model = LogisticRegression()
        model.fit(X, Y)
        with open(filename, 'wb') as fp:
            pickle.dump(model, fp)
        return model

    def open_model(self, model_filename):
        return pickle.load(open(model_filename, 'rb'))

    def generate_predictions(self, model, test_data_filename, output_filename):
        # First load the test data into memory
        # Then, featurize the questions from the test data
        # Keep track of the question IDs in order
        # Feed the features into the model
        # Generate classifications from the model in vector form
        # Use a helper function to take a vector plus IDs and convert it to output form
        # Write to output. 

if __name__ == "__main__":

    a = Andrew("development.json")
    X, Y = a.vectorize("x_trial_1.sav", "y_trial_1.sav")
    print(X)
    print(Y)
    print(len(X))
    print(len(X[0]))
    print(len(Y))
    """
    question = "What led to the corporate America of start?"
    context = "Nobody saw it coming. The rise of populism and the death of manufacturing led to the start of corporate America. There was plenty of warning and yet no signs."
    thing_list = a.count_things(question)
    matches, sentence = a.count_matches(thing_list, context)
    inversion_count = a.detect_inversion(thing_list[3], sentence)
    print(thing_list)
    print(matches)
    print(sentence)
    print(inversion_count)
    """

