from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
from collections import Counter
import string
from nltk.tokenize.treebank import TreebankWordTokenizer

import time


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    tokenizer = TreebankWordTokenizer()
    with open(path , encoding = 'utf8') as f:
        for l in f:
            table = str.maketrans(dict.fromkeys(string.punctuation + '0123456789')) #to remove numbers & punctuation
            sentences.append( tokenizer.tokenize(l.translate(table).lower()) )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self,sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):

        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount



    def train(self,stepsize, epochs):

        # Functions
        def gradients(y , ypred , W , C):
            # Function that computes the gradients.
            root = -y*ypred*np.exp( -W.dot(C) ) + (1 - y)*(1-ypred)*np.exp( W.dot(C) )
            
            first = root.reshape(-1 , 1).dot(C.reshape(1 , -1)) 
            second = W.T.dot(root)
            
            return first , second


        # First Step: We build the vocabulary

        # We count the frequency of each word in the vocabulary.
        word_occurences = dict(Counter([e for sub in self.sentences for e in sub]))
        
        # We remove unfrequent words:
        unfreqs = [] # list that will contain deleted words

        voc_keys = list(word_occurences.keys())
        for k in voc_keys:
            if word_occurences[k] < self.minCount:
                unfreqs.append(k)
                word_occurences.pop(k, None) # we delete a word if it's unfrequent.

        # the vocabulary construction:
        vocabulary  = list(word_occurences.keys()) # Unique words in the vocabulary.

        len_vocab = len(vocabulary) #length of the vocabulary.
        vocabulary = dict( zip(vocabulary + unfreqs , list(range(len_vocab)) + [-1 for _ in unfreqs] ) ) # Word to index dictionnary.
        self.vocabulary = vocabulary

        word_occurences = np.array(list(word_occurences.values())) # We keep only the values of the frequencies.



        # Second step : We build the set of pairs
        D = [] # set of (context, words)

        for l in self.sentences:
    
            lid = [vocabulary[w] for w in l]
            for i,c in enumerate(lid):
                # We add all the words in the window
                if c != -1:
                    appD = (c , [w for w in lid[i - self.winSize//2:i]+ lid[i + 1 : i + self.winSize//2] if c != -1 ] ) # pair words.
                    D.append(appD)


        # The negative sampling probabilities:
        all_ids = list(range(len_vocab))
        neg_probs = word_occurences**0.75
        neg_probs = neg_probs / sum(neg_probs)

        # Third step: Training:

        # We intialize the weights:
        Embs , cont_Embs = np.random.rand(len_vocab , self.nEmbed), np.random.rand(len_vocab , self.nEmbed)

        for epoch in range(epochs):
            
            neg_ids_epoch = np.random.choice( all_ids , size = (len(D), self.negativeRate * self.winSize) ,  p = neg_probs)

            for pos,pair in enumerate(D):

                # Negative Ids
                cont_id = pair[0]
                pos_ids = pair[1]
                neg_ids = list(neg_ids_epoch[pos])

                # We train the weights on the pairs.
                word_emb = Embs[ pos_ids + neg_ids ] # Word embedding
                cont_emb = cont_Embs[cont_id] # Context Embedding

                output = expit(word_emb.dot(cont_emb)) # Probabilities

                # We construct a 'target' variable to simplify the computations. ( = 1 if in the window 0 else)
                target = np.zeros(word_emb.shape[0])
                target[: len(pos_ids)] = 1

                # Loss Computation & Gradient Descent
                gradient_loss = gradients(target , output , word_emb , cont_emb)

                # Update:
                Embs[ pos_ids + neg_ids ] += - stepsize * gradient_loss[0]
                cont_Embs[cont_id] += -stepsize * gradient_loss[1]


        self.Embeddings = Embs





    def save(self,path):
        #We concatain the vocabulary and the embeddings into one array.
        to_save =np.array([ np.array(list(self.vocabulary.keys())) , np.array(list(self.vocabulary.values())) , self.Embeddings])
        np.save(path , to_save)


    def similarity(self,word1,word2):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """

        i1 , i2 = self.vocabulary[word1] , self.vocabulary[word2]
        vec1 , vec2 = self.Embeddings[i1] , self.Embeddings[i2]

        # We compute the cosine similarity
        return np.dot(vec1 , vec2)/ ( np.linalg.norm(vec1) * np.linalg.norm(vec2) )


    @staticmethod
    def load(path):
        
        loaded = np.load(path)
        obj = SkipGram(sentences = [])
        obj.vocabulary = dict(zip(loaded[0] , loaded[1]))
        obj.Embeddings = loaded[2]

        return obj



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    start = time.time()

    if not opts.test:
        epochs , stepsize = 1 , 0.01
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(stepsize , epochs)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = mSkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))

    end = time.time()

    print(end - start)