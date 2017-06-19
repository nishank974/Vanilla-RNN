#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:49:44 2017

@author: user
"""

def test(X,y):
    temp_test=[x.split(" ") for x in y ]
    count=0
    for i in xrange(len(X)):
        hprev=X[i]
        inputs = [char_to_ix[ch] for ch in temp_test[i][0:seq_length]]
        targets = [char_to_ix[ch] for ch in temp_test[i][1:]]
        sample_ix = sample(hprev,inputs[0],200)  #earlier was 200
        txt = ''.join(ix_to_char[ix]+' ' for ix in sample_ix)
        
        if("<s> "+str(ix_to_char[sample_ix[0]])+" </s>"==y[i]):
    
            count+=1
    print "Accuracy"
    print str((count/len(y))*100)+' %'
    
test(X,y)