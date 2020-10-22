# NaiveBayes.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata
from scipy import sparse
from tqdm import tqdm

class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA=ALPHA
        #TODO: Initalize parameters 
        probWordGivenPositive, probWordGivenNegative, Prob_Positive_Train, Prob_Negative_Train = self.Train(X,Y)
        self.probWordGivenPositive = probWordGivenPositive
        self.probWordGivenNegative = probWordGivenNegative
        self.Prob_Positive_Train = Prob_Positive_Train
        self.Prob_Negative_Train = Prob_Negative_Train
        
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        #which file is positive or negative
        Idx_Positive_Train = np.where(Y > 0)[0]
        Idx_Negative_Train = np.where(Y < 0)[0]
        #how many words in every file
        Every_Fileword_Num = X.sum(axis = 1)
        #how many words in pos or neg files
        Num_Positive_Train = np.sum(Every_Fileword_Num[i][0] for i in Idx_Positive_Train)
        Num_Negative_Train = np.sum(Every_Fileword_Num[i][0] for i in Idx_Negative_Train)
        #probabilty of a word in positive or negative file,i.e. P(Positive) and P(Negative)
        Prob_Positive_Train = Num_Positive_Train/(Num_Positive_Train + Num_Negative_Train)
        Prob_Negative_Train = 1 - Prob_Positive_Train
        #P(This word exists | Positive) and P(This word exists | Negative) for every word
        probWordGivenPositive = np.zeros((X.shape[1]),dtype='float')
        probWordGivenNegative = np.zeros((X.shape[1]),dtype='float')    
        
        Col = np.zeros((X.shape[0]),dtype='float')
        for i in tqdm(range(X.shape[1])):
            Col_Tmp = X.getcol(i)
            Col = Col_Tmp.toarray()
            probWordGivenPositive[i] = np.sum(Col[row] for row in Idx_Positive_Train)
            probWordGivenNegative[i] = np.sum(Col[row] for row in Idx_Negative_Train)
        
        # probWordGivenPositive = np.load("pos.npy")
        # probWordGivenNegative = np.load("neg.npy")
        probWordGivenPositive = (probWordGivenPositive+self.ALPHA)/(Num_Positive_Train+self.ALPHA)
        probWordGivenNegative = (probWordGivenNegative+self.ALPHA)/(Num_Negative_Train+self.ALPHA)
        
        return np.log(probWordGivenPositive), np.log(probWordGivenNegative), np.log(Prob_Positive_Train), np.log(Prob_Negative_Train)

    def Predict(self, X):
        #TODO: Implement Naive Bayes Classification
        logPositiveGivenProbWordPresent = 0
        logNegativeGivenProbWordPresent = 0
        Y = []
        Row = np.zeros((X.shape[1]),dtype='float')
        for f in tqdm(range(X.shape[0])):
            Row_Tmp = X.getrow(f)
            Row = Row_Tmp.toarray()
            judge = np.sum(Row[0,col]*self.probWordGivenPositive[0,col] for col in range(Row.shape[1])) + self.Prob_Positive_Train - np.sum(Row[0,col]*self.probWordGivenNegative[0,col] for col in range(Row.shape[1])) - self.Prob_Negative_Train
            if(judge > 0):
                Y.append(+1.0)
            else:
                Y.append(-1.0)
        return np.asarray(Y)

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    # sparse.save_npz('./train.npz', train.X)
    # sparse.save_npz('./test.npz', test.X)
    # np.save("train.npy",train.Y)
    # np.save("test.npy",test.Y)
    
    # train_x = sparse.load_npz('train.npz')
    # train_Y = np.load("train.npy")
    # test_x = sparse.load_npz('test.npz')
    # test_Y = np.load("test.npy")
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    #nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print(nb.Eval(test.X, test.Y))
