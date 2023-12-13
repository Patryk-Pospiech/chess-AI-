# chess-AI-

Currently training AI to reach an estamiated elo of 2700 with the model using 100M games from FICS Games Database for training 

# Simple explanation - 
Board is encode as 64 bytes and transformed into a 768 units wide float vector on the GPU this gives a higher performance boost as there are less I/O

The model start with f(p) function that approximates the value of the postition.
This model is currently using Negamax with alpha-beta pruining but if results do not come as predictated the New model would use MTD-F 
Durring testing training the model I used was a smaller version of the same neural network with 5 layer deep with 3072 wide vector condense down to single value 
this allowed for the model to see 4 times more positions evaluation.

 

# Limitation 

Durring playing the GPU is not used as its would cause hardware strain on users for this reason only the GPU was used for training
For a smarter and quicker model it would need to be written in C++ using  bitmaps 
The model would also need a better evaluatioon function as it accuracy per time unit slowed the AI by 17%

Using the (p,q,r) triplets  into a HDF5 data  file I was able to adapt the learning rate before finding the optimal learning rate 
but to achive fast results I adapted the learning rate scheme to 0.03 x exp(-time in days ) this was done as high training data was used and regularzation wasnt neccessary meaning dropout or L2 regularization was not used 
this caused overfitting and less gerneralization ablity of the model 

# Training network maths 

To train the network I used (p,q,r) triples which feed into the network denoted by S(x) = 1/(1 + exp(-x))

The F(r)>F(q)>F(p)> -F(q) and  f(p) < -F(q) with the  last to expression representing a soft equality. f(p) = -F(q) This allowed for 3 ReLu layers to fully connect 

but due to this it created 10m unkown parmaters in the network 
