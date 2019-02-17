TP 1 : NLP
---------------------------



Step 1: Preprocessing.
---------------------------
For this step, we deleted numerical characters and punctiation from the text. We also lowered all the characters before separating the sentence. 


Step 2: Vocabulary Construction.
---------------------------
For this part, to be more efficient, we directly use the Counter object from the library 'collections' to count the occurences of each unique word in the vocabulary. We then use the result to filter the words that occur less than the selected threshold. The filtered words are stored in a separate list. We then use the unique words in the Counter object as the vocabulary (that we convert into a dictionnary for indexing purposes). 


Step 3: Set of Pairs Construction.
---------------------------
This is a pre-training step. Instead of extracting the context and the neighboring words at each epoch, we loop over the corpus and store the (context , [words]) pairs before the training. We don't use 2 words pairs (c , word) but we gather all the words in the window to speed up the training.


Step 3: Negative Sampling.
---------------------------
Before training the embedding matrices, we need to compute the negative sampling probabilities. We use the occurences matrix we computed before to do that (the formula is : w**0.75/sum(w**0.75))


Step 4: Training.
---------------------------
This is the part where the embedding matrices are trained.

We use two different matrices: One for the context word, and another one for the words in the pair list (that we call positive words) and the negative sampled words (that we call negative words). 

The context embedding and the word embeddings are used to obtain a conditional probability per word (through the sigmoid function). The loss we use is the cross-entropy loss, that we compute through a 'target' that we formulated to simplify the gradient computation (1 for a positive word and 0 for a negative one). 

The derivation we use is computed from scratch. The derivate of the context and the word embedding derivates have a common root that we use to gain performance.

The updates are made with a fixed step-size.