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
import csv
import time
from tqdm import tqdm

class RNNModel():
    """An instance of an RNN model for this MRC task"""
    def __init__(self, model=None, forward_dict=None):
        if model: self.model = model
        self.tokenizer = RegexpTokenizer('\w+|[^\w\s]+')
        self.forward_dict = forward_dict
        if forward_dict:
            self.model = model(vocab_size = len(forward_dict),
                            hidden_dim = 512, out_dim = 2)
        else:
            self.model = None

    """train the model given a json file of training information"""
    def train(self,fpath):
        """
        train_set["data"] -> list of titles
            title["title"] -> title name
            title["paragraphs"] -> list of paragraphs
                paragraph["context"] -> data which might answer questions in this paragraph
                paragraph["qas"] -> list of questions
                    question["question"] -> question content
                    question["id"] -> question ID
        """
        use_cuda = True
        with open(fpath,'r') as fp:
            train_set = json.load(fp)
        forward_dict = {'UNK': 0} # vocabs
        forward_ind = 1
        context_seqs = [] # List of context sequences. [[],[],...,[]]
        question_for_context_seqs = [] # List of list of questions for each context. [[[],...,[]],...,[[],...,[]]]
        gold_dict = {} # id->actual label
        id_dict = {} # id->(i,j) ith context, jth question. q_for_c_seqs[i][j]
        # read relevant information and save them. Possible memory error?
        prev_cont = 0
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
                    id_dict[question['id']]=(p_ind+prev_cont,q_ind,question)
                    gold_dict[question['id']] = 0 if question['is_impossible'] else 1
                question_for_context_seqs.append(questions_ls)
            prev_cont+=len(title["paragraphs"])
        # context, questions, and gold outputs are saved
        # input sequences for context and qs
        train_ctxt = [list(map(lambda t: forward_dict.get(t,0), line)) for line in context_seqs]
        train_qs = [[list(map(lambda t: forward_dict.get(t,0), line)) for line in qs]
                    for qs in question_for_context_seqs]
        id_ls = list(id_dict.keys())
        self.forward_dict = forward_dict

        print("Data loaded!")
        if self.model==None:
            self.model = model(vocab_size = len(forward_dict),
                                hidden_dim = 512, out_dim = 2)
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()
        optimizer = optim.SGD(self.model.parameters(), lr=1.0)
        minibatch_size = 1024
        num_minibatches = len(id_ls) // minibatch_size

        for epoch in (range(1)):
            # Training
            print("Training")
            # Put the model in training mode
            self.model.train()
            start_train = time.time()

            for group in tqdm(range(num_minibatches)):
                predictions = None
                gold_outputs = None
                loss = 0
                optimizer.zero_grad()
                for i in range(group * minibatch_size, (group + 1) * minibatch_size):
                    q_id = id_ls[i]
                    c_ind,q_ind,q = id_dict[q_id]
                    input_c = []
                    input_c = train_ctxt[c_ind]
                    input_q = train_qs[c_ind][q_ind]
                    gold_output = gold_dict[q_id]
                    prediction_vec, prediction = self.model(input_c,input_q)

                    if predictions is None:
                        predictions = [prediction_vec]
                        gold_outputs = [gold_output] 
                    else:
                        predictions.append(prediction_vec)
                        gold_outputs.append(gold_output)
                pred_vecs = torch.stack(predictions)
                pred_vals = torch.tensor(gold_outputs).squeeze()
                if use_cuda and torch.cuda.is_available():
                    pred_vecs = pred_vecs.cuda()
                    pred_vals = pred_vals.cuda()
                loss = self.model.compute_Loss(pred_vecs, pred_vals)
                loss.backward()
                optimizer.step()
            print("Training time: {} for epoch {}".format(time.time() - start_train, epoch))

    def save_model(self, model_name):
        torch.save(self.forward_dict, model_name+'_vocab')
        torch.save(self.model.state_dict(), model_name+'_model')

    def load_model(self, model_name):
        self.forward_dict = torch.load(model_name+'_vocab')
        self.model = model(vocab_size = len(self.forward_dict), hidden_dim = 512, out_dim = 2)
        self.model.load_state_dict(torch.load(model_name+'_model'))

    def predict(self,fpath,outpath):
        if self.model == None:
            print('Call obj.train(fpath) first!')
        else:
            with open(fpath, 'r') as fp:
                test_set = json.load(fp)
            pred_dict = {} # id->prediction to fill in
            # Put the model in evaluation mode
            self.model.eval()
            for title in test_set["data"]:
                for paragraph in title["paragraphs"]:
                    wrds = self.tokenizer.tokenize(paragraph["context"])
                    # save context
                    val_cseq = list(map(lambda t: self.forward_dict.get(t,0), wrds))
                    #save lists of qs per context and add to worddict
                    for question in paragraph["qas"]:
                        wrds = self.tokenizer.tokenize(question['question'])
                        val_qseq = list(map(lambda t: self.forward_dict.get(t,0), wrds))
                        _, predicted_output = self.model(val_cseq,val_qseq)
                        pred_dict[question['id']] = predicted_output
            if len(outpath)>5 and outpath[-5:] == '.json':
                with open(outpath,'w') as outp:
                    json.dump(pred_dict, outp)
                files.download(outpath)
            else:
                with open(outpath,'w',newline='') as outp:
                    csvwriter = csv.writer(outp, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow(['Id', 'Category'])
                    for qid in pred_dict:
                        csvwriter.writerow([qid, str(pred_dict[qid])])
                files.download(outpath)