import json
from nltk.tokenize import RegexpTokenizer
from rnnmrc import model
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm

class RNNModel():
    def __init__(self):
        self.model = None
        self.tokenizer = RegexpTokenizer('\w+|[^\w\s]+')

    def train(self,fpath,validationpath=None):
        """
        train_set["data"] -> list of titles
            title["title"] -> title name
            title["paragraphs"] -> list of paragraphs
                paragraph["context"] -> data which might answer questions in this paragraph
                paragraph["qas"] -> list of questions
                    question["question"] -> question content
                    question["id"] -> question ID
        """
        with open(fpath,'r') as fp:
            train_set = json.load(fp)
        forward_dict = {'UNK': 0} # vocabs
        forward_ind = 1
        context_seqs = [] # List of context sequences. [[],[],...,[]]
        question_for_context_seqs = [] # List of list of questions for each context. [[[],...,[]],...,[[],...,[]]]
        gold_dict = {} # id->actual label
        id_dict = {} # id->(i,j) ith context, jth question. q_for_c_seqs[i][j]
        # read relevant information and save them. Possible memory error?
        for title in train_set["data"]:
            # ?: include titles in dictionary?
            for p_ind, paragraph in enumerate(title["paragraphs"]):
                wrds = self.tokenizer.tokenize(paragraph["context"])
                # save context and add to worddict
                context_seqs.append(wrds)
                for wrd in wrds:
                    if wrd not in forward_dict:
                        forward_dict[wrd] = forward_ind
                        forward_ind+=1
                # save lists of qs per context and add to worddict
                questions_ls = []
                for q_ind, question in enumerate(paragraph["qas"]):
                    wrds = self.tokenizer.tokenize(question['question'])
                    questions_ls.append(wrds)
                    for wrd in wrds:
                        if wrd not in forward_dict:
                            forward_dict[wrd] = forward_ind
                            forward_ind+=1
                    id_dict[question['id']]=(p_ind,q_ind)
                    gold_dict[question['id']] = 0 if question['is_impossible'] else 1
                question_for_context_seqs.append(questions_ls)
        
        # context, questions, and gold outputs are saved
        # input sequences for context and qs
        train_ctxt = [list(map(lambda t: forward_dict.get(t,0), line)) for line in context_seqs]
        train_qs = [[list(map(lambda t: forward_dict.get(t,0), line)) for line in qs]
                    for qs in question_for_context_seqs]
        id_ls = list(id_dict.keys())

        print("Data loaded!")
        m = model(vocab_size = len(forward_dict), hidden_dim = 512, out_dim = 2)
        self.model = m
        optimizer = optim.SGD(m.parameters(), lr=1.0)
        minibatch_size = 100
        num_minibatches = len(id_ls) // minibatch_size

        for epoch in (range(5)):
            # Training
            print("Training")
            # Put the model in training mode
            m.train()
            start_train = time.time()

            for group in tqdm(range(num_minibatches)):
                predictions = None
                gold_outputs = None
                loss = 0
                optimizer.zero_grad()
                for i in range(group * minibatch_size, (group + 1) * minibatch_size):
                    q_id = id_ls[i]
                    c_ind,q_ind = id_dict[q_id]
                    input_c = train_ctxt[c_ind]
                    input_q = train_qs[c_ind][q_ind]
                    gold_output = gold_dict[q_id]
                    prediction_vec, prediction = m(input_c,input_q)

                    if predictions is None:
                        predictions = [prediction_vec]
                        gold_outputs = [gold_output] 
                    else:
                        predictions.append(prediction_vec)
                        gold_outputs.append(gold_output)
                loss = m.compute_Loss(torch.stack(predictions), torch.tensor(gold_outputs).squeeze())
                print(loss)
                loss.backward()
                optimizer.step()
            print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

            if validationpath:
                # Evaluation
                print("Evaluation enabled")
                with open(validationpath,'r') as fp:
                    val_set = json.load(fp)
                # Put the model in evaluation mode
                m.eval()
                start_eval = time.time()

                predictions = 0 # number of predictions
                correct = 0 # number of outputs predicted correctly
                for title in val_set["data"]:
                    # ?: include titles in dictionary?
                    for paragraph in enumerate(title["paragraphs"]):
                        wrds = self.tokenizer.tokenize(paragraph["context"])
                        # save context
                        val_cseq = list(map(lambda t: forward_dict.get(t,0), wrds))
                        # save lists of qs per context and add to worddict
                        for question in enumerate(paragraph["qas"]):
                            wrds = self.tokenizer.tokenize(question['question'])
                            val_qseq = list(map(lambda t: forward_dict.get(t,0), wrds))
                            _, predicted_output = m(val_cseq,val_qseq)
                            gold_output = 0 if question['is_impossible'] else 1
                            correct += int((gold_output == predicted_output))
                            predictions += 1
                accuracy = correct / predictions
                assert 0 <= accuracy <= 1
                print("Evaluation time: {} for epoch {}, Accuracy: {}".format(time.time() - start_eval, epoch, accuracy))
    
    def predict(self,fpath,outpath):
        if self.model == None:
            self.train('./training.json','./development.json')
        else:
            with open(fpath, 'r') as fp:
                test_set = json.load(fp)
            pred_dict = {} # id->prediction to fill in
            # Put the model in evaluation mode
            m.eval()
            for title in test_set["data"]:
                for paragraph in enumerate(title["paragraphs"]):
                    wrds = self.tokenizer.tokenize(paragraph["context"])
                    # save context
                    val_cseq = list(map(lambda t: forward_dict.get(t,0), wrds))
                    # save lists of qs per context and add to worddict
                    for question in enumerate(paragraph["qas"]):
                        wrds = self.tokenizer.tokenize(question['question'])
                        val_qseq = list(map(lambda t: forward_dict.get(t,0), wrds))
                        _, predicted_output = m(val_cseq,val_qseq)
                        pred_dict[question['id']] = predicted_output
            with open(outpath,'w') as outp:
                json.dump(pred_dict,outp)
            return