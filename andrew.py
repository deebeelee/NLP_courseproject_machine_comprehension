import os
import json
import pickle
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression


class Andrew:

    def __init__(self, data_filepath):
        path_to_data = os.path.join(os.getcwd(), data_filepath)
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

    def when_special(self, word_list, max_score_sentence_tags):
        if (word_list[0] == "When"):
            for tag in max_score_sentence_tags:
                if (tag[1] == "CD"):
                    return 1
                else: 
                    return 0
        return 0

    def where_special(self, word_list, max_score_sentence_tags):
        if (word_list[0] == "Where"):
            for i in range(len(max_score_sentence_tags)):
                if (max_score_sentence_tags[i][1] == "NNP"):
                    try:
                        if (max_score_sentence_tags[i-1][1] == "IN"):
                            return 1
                    except:
                        return 0
        return 0

    def who_special(self, word_list, max_score_sentence_tags):
        if (word_list[0] == "Who"):
            for tag in max_score_sentence_tags:
                if (tag[1] == "NNP" or tag[1] == "NNPS"):
                    return 1
        return 0

    def other_special(self, word_list, max_score_sentence_tags):
        if (word_list[0] == "In" or word_list[0] == "From"):
            for tag in max_score_sentence_tags:
                if (tag[0] == word_list[0]):
                    return 1
        return 0

    def featurize(self, question, context, ablation):
        features = []
        noun_list, verb_list, word_list, special_list = self.count_things(question)
        features.append(len(noun_list)) #0
        features.append(len(verb_list)) #1
        features.append(len(word_list)) #2
        matches, max_score_sentence = self.count_matches([noun_list, verb_list, word_list], context)
        features.append(matches[0]) #3
        #features.append(matches[1]) #4
        features.append(matches[2]) #5
        #inversions = self.detect_inversion(special_list, max_score_sentence)
        #features.append(inversions) #6
        max_score_sentence_tags = nltk.pos_tag(nltk.word_tokenize(max_score_sentence))
        #special_question_rule = self.when_special(word_list, max_score_sentence_tags) + self.where_special(word_list, max_score_sentence_tags) + self.who_special(word_list, max_score_sentence_tags)
        #features.append(special_question_rule)
        features.append(self.when_special(word_list, max_score_sentence_tags)) #7
        features.append(self.where_special(word_list, max_score_sentence_tags)) #8
        features.append(self.who_special(word_list, max_score_sentence_tags)) #9
        features.append(self.other_special(word_list, max_score_sentence_tags)) #10
        if (ablation >= 0):
            features.pop(ablation)
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
                    X.append(self.featurize(question["question"], context, -1))
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
        with open(model_filename, 'wb') as fp:
            pickle.dump(model, fp)
        return model

    def open_model(self, model_filename):
        return pickle.load(open(model_filename, 'rb'))

    def write_output(self, ids, labels, filename):
        output_dict = {}
        for i in range(len(ids)):
            output_dict[ids[i]] = labels[i]
        with open(filename, 'w') as outfile:
            json.dump(output_dict, outfile)

    def generate_predictions(self, model, test_data_filename, output_filename, ablation):
        # First load the test data into memory
        path_to_data = os.path.join(os.getcwd(), test_data_filename)
        current_loc = os.getcwd()
        data_path, data_file_name = path_to_data.rsplit(os.sep, 1)
        os.chdir(data_path)
        with open(data_file_name, "r") as fileHandle:
            test_data = json.load(fileHandle)
        os.chdir(current_loc)

        # Then, featurize the questions from the test data
        Xtest=[]
        # Keep track of the question IDs in order
        Xids =[]
        for title in test_data["data"]:
            for paragraph in title["paragraphs"]:
                context = paragraph["context"]
                for question in paragraph["qas"]:
                    Xtest.append(self.featurize(question["question"], context, ablation))
                    Xids.append(question["id"])

        # Feed the features into the model
        C = model.predict(Xtest)
        C = C.tolist()
        # Use a helper function to take a vector plus IDs and convert it to output form
        self.write_output(Xids, C, output_filename)

    def json_to_kaggle(self, json_filename, csv_filename):
        with open(json_filename, "r") as fileHandle:
            output_dict = json.load(fileHandle)
        with open(csv_filename, "w") as g:
            g.write("Id,Category\n")
            for k, v in output_dict.items():
                g.write(k)
                g.write(",")
                g.write(str(v))
                g.write("\n")

if __name__ == "__main__":

    a = Andrew("training.json")
    # Advanced Feature Set and Ablation Testing
    X, Y = a.vectorize("x_trial_reduced.sav", "y_trial_reduced.sav")
    #X, Y = a.open_vectors("x_trial_ablation.sav", "y_trial_3.sav")
    """
    new_X = []
    ablation = 10
    for features in X:
        new_features = []
        for i in range(len(features)):
            if not (i == ablation):
                new_features.append(features[i])
        new_X.append(new_features)
    print(len(new_X[0])) # Size of features
    """
    model = a.train_LR(X, Y, "LR_model_reduced.sav")
    #model = a.open_model("LR_model_reduced.sav")
    a.generate_predictions(model, "testing.json", "baseline.json", -1)
    a.json_to_kaggle("baseline.json", "baseline.csv")
    
    """
    #Training and Results Below:
    #X, Y = a.vectorize("x_trial_2.sav", "y_trial_2.sav")
    X, Y = a.open_vectors("x_trial_2.sav", "y_trial_2.sav")
    model = a.open_model("LR_model_training.sav")
    #model = a.train_LR(X, Y, "LR_model_training.sav")
    a.generate_predictions(model, "testing.json", "testing_output.json")
    
    a.json_to_kaggle("testing_output.json", "kaggle_submit_a2.csv")

    #Testing Below:
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