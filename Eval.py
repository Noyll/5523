# Eval.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import numpy as np

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        
    def Accuracy(self):
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))
