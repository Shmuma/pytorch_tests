# English language questions model

This code trains language model on [quora questions kaggle dataset](https://www.kaggle.com/c/quora-question-pairs) 
using PyTorch. As embeddings pre-initialisation I've used [glove 6B 50d embeddings](https://nlp.stanford.edu/projects/glove/). 

## Model

Single-layer, unidirectional LSTM with 512 hidden neurons shown the best results. To speed up simuliation, 
I've implemented two-layer softmax approach following recent publication from Facebook research 
[Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309), which gave ~3 times speed 
up over standard softmax version. But due to more efficient GPU memory utilisation, it was possible to increase
batch size more than 30 times, which gave additional 4 times speed up. 

So, overall speed up on GTX 1080Ti is **11 times** (30 seconds per epoch versus 2:50).

## Two-layer softmax

Common problem of language models is the fact that on output, model needs to produce probability distribution over
vorcabulary. As vocabulary is usually large (100k of words, sometimes even millions), the output operation dominates
over the rest of computations.

There are several tricks invented how to handle this, such as Sampled Softmax or Hierarchical Softmax.

This code implements Two-Level hierarchical softmax, which, basically, splits the vocabulary into separate classes and
organizes those classes into two-level tree, factorising probability distribution over all words in vocabulary into
probability of predicted word to be in class times probability distribution of words withing predicted class.

In paper "Efficient softmax approximation for GPUs", authors proposes to leave most frequent words on top level and 
move less-frequent part of the dictionary into underlying classes.

### Implementation details

PyTorch supports dynamic calculations, which is beneficial in such cases. Two-level softmax is implemented in 
[questions/model.py, class TwoLevelSoftmaxMappingModule](https://github.com/Shmuma/pytorch_tests/blob/master/pract_rl/hw6.5/questions/questions/model.py#L142)

During training, our forward() function supposed to calculate loss from hidden's layer output and valid word indices.
As our batch is randomly sampled, indices can belong to different layers of our tree. Most of them will fall 
into L1 (top level), which allows us to calculate output probability by single matrix multiplication, but some 
(less frequent) will involve two multiplications, first to get probability for class and second to calculate
probability for words withing class.

To speed up this calculation, first, I sort our batch according to output indices, which makes our class ranges 
continuous, and then, do required matrix multiplications.

## Full hierarchical softmax implementation

There is other version of HSM, implemented according to "classical" hierarchical softmax definition, which 
assumes that we fully split our vocabulary into balanced binary tree 
[questions/model.py, class HierarchicalSoftmaxMappingModule](https://github.com/Shmuma/pytorch_tests/blob/master/pract_rl/hw6.5/questions/questions/model.py#L83)

But experiments gave zero speed up over normal softmax, and there is no practical usefulness of this code.  
 
