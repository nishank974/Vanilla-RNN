# Vanilla-RNN
Recursive neural network code written without use of any library


rnn_q&a.py code is a vanilla rnn written from scratch.

sentence_to_vextor.py is a python code to convert sentences into vectors using word2vec and TFIDF model.

Data folder contains various datasets including with and without partial dependencies.

DataSet folder contains various other datasets on which the code can be tested.



Algorithm Explaination.

1 The sentence vector of the paragraph data is given as previous input i.e. feedback.
2 The question is asked on each unrolling rnn.
3 Once the question is completed then system starts predicting the answer word by word like a seq_to_seq generator.
4 Each answer is embedded in start(< s >) and end(< \ s >) mark ups to reduce the prediction to sentences.
5 The system is designed such a way that it is used to predict only one sentence. It can be modified easily

To run the code:
1 Gensim library needs the list of vocabulary which could be used before hand. 
2 Update the location of the file which you wish to use to train in sentence_to_vector.py
3 Then run rnn_q&a.py in the same address same/kernel.
4 Change the locations where you wish to save the predicted output.

Create the vocabulary file before training and if any new words occur update gensim model with the given new word.

