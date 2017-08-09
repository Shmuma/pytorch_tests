# Abstract

This code trains language model on [quora questions kaggle dataset](https://www.kaggle.com/c/quora-question-pairs) 
using PyTorch. Pre-trained glove embeddings were used as initial values [glove 6B 50d embeddings](https://nlp.stanford.edu/projects/glove/). 

Input data contains 537k questions with total of 7.5m tokens. Glove embeddings dictionary was filtered with tokens 
present in training data and final dictionary has 64517 words. 

The main goal was to implement and benchmark gpu-efficient softmax approximations.

## Model

Single-layer, unidirectional LSTM with 512 hidden neurons predicts next word in a train sequence. To speed up learning, 
I've implemented two most popular softmax approximations: hierarchical softmax and sampled softmax on pytorch and 
compared their speedups.  

To run the code, you need the following:
1. glove embeddings (or other embeddings you like in text format similar to glove). Words in embeddings need to be 
sorted by decrease of frequency (it's important for two-level softmax)
2. quora kaggle dataset with questions
3. gpu to use, as with cpu it will take ages to train.

## Softmax approximation

Common problem of neural language models is size of vocabulary we need to deal on the output of the network. 
In general, our model should return probability distribution over whole vocabulary to select next predicted word on 
every step of the sequence. Usually, it's achieved by output linear layer with softmax nonlinearity. Linear layer 
(matrix of size H*N, where H is a size of RNN output and N is count of words) transforms hidder RNN state into word 
'scores' and softmax normalizes it into probability distribution.

It works quite good, but even in moderate-size vocabulary of several hundred thousand words it requires giant matrices
to be multiplied and normalized for every step of every training example.  

### Sampled Softmax

Idea of sampled softmax is simple, the same as implementation. During training, we have valid word for every input 
sequence entry. Instead of calculating probability over whole vocabulary, we randomly sample some fixed small amount of 
words and using them as negative-samples, forcing our model to increase probability of correct words and decrease 
probability of random examples. 

This approach implemented in [questions/model.py, class SampledSoftmaxMappingModule](https://github.com/Shmuma/pytorch_tests/blob/master/pract_rl/hw6.5/questions/questions/model.py#L262), 
in those basic steps:
1. generated N random word indices and concatenate them to input batch indices,
2. by using Embedding layer, we convert every index into output matrix corresponding row,
3. perform scalar product of input matrix X (with RNN's output) to obtained matrix. This gives us unnormalized scores for every input sample,
4. calculate cross-entropy loss for our scores.

Very simple and straightforward. This method much simpler than Hierarchical Softmax I'm going to describe later. 

Benchmarks shown quite good speed up, ~5x times over naive softmax on vocabulary of 65k words.

Original paper on sampled softmax: [On using very large target vocabulary for Neural Machine Translation](https://arxiv.org/abs/1412.2007).

### Hierarchical Softmax

Hierarchical softmax puts vocabulary in a tree, which allows to avoid calculations of probabilities for full vocabulary,
replacing it several steps down the tree.

"Classical" approach is to put vocabulary in balanced binary tree, in which words will be placed in final leafs. 
It reduces our computations from O(n) to O(log n), but, unfortunately have several drawbacks:

1. final performance is heavily dependent on initial vocabulary splitting,
2. it's not very efficient on modern GPUs.

I've implemented full hierarchical softmax in [questions/model.py, class HierarchicalSoftmaxMappingModule](https://github.com/Shmuma/pytorch_tests/blob/master/pract_rl/hw6.5/questions/questions/model.py#L83),
but it worked slowly than original full softmax, so, it was excluded from consequent benchmarks.

### Two level-hierarchical softmax   

Better approach proposed in paper from Facebook team [Efficient softmax approximation for GPUs](https://arxiv.org/abs/1609.04309).
They propose to split vocabulary in small (4-10) amount of classes and organize them in two-level tree structure
with most frequent words to be placed in a root, and classes with less frequent words will be put on second-level.

It's implemented in [questions/model.py, class TwoLevelSoftmaxMappingModule](https://github.com/Shmuma/pytorch_tests/blob/master/pract_rl/hw6.5/questions/questions/model.py#L142), 
and it turned out to be the fastest of tried methods. With the same batch size (1000 tokens) it gave about 70% speed up, but
due to it's memory effectiveness, I was able to increase batch size to 100000 tokens (100 times!), which gave more than 14 times final speed increase.
Actual numbers are in the section below. 

## Benchmarks

I've benchmarked all implementation on GTX 1080Ti with 11GiB of memory and pytorch 0.1.2. I've measured how many tokens 
per second was processed during traning as well as wall clock epoch time. I've varied batch size which is measured in 
tokens of input sequences passed to network at once.

Speedup was calculated relative to full softmax with batch=1000.
 
Memory usage was also measured according to nvidia-smi tool output, but pytorch adjusts memory dynamically, so, it can be
misleading.

| Method  | Batch size | Tokens/s | Epoch time | Speedup | GPU memory usage |
|:-------:|-----------:|---------:|:----------:|--------:|-----------------:|
| Softmax | 1k         | 1271     | 7:02       | 1.00    | 5.5GiB           |
| Softmax | 3k         | 1568     | 5:42       | 1.23    | 9.6GiB           |
| Softmax | 5k         | 1668     | 5:21       | 1.31    | 10.5GiB          |
|         |            |          |            |         |                  |
| 2-HSM   | 1k         | 2186     | 4:05       | 1.72    | 1.2GiB           |  
| 2-HSM   | 3k         | 5600     | 1:35       | 4.41    | 1.4GiB           |  
| 2-HSM   | 5k         | 7686     | 1:09       | 6.05    | 1.8GiB           |  
| 2-HSM   | 10k        | 11398    | 0:47       | 8.97    | 2.6GiB           |  
| 2-HSM   | 20k        | 14745    | 0:36       | 11.60   | 3.5GiB           |  
| 2-HSM   | 30k        | 16284    | 0:32       | 12.81   | 5.0GiB           |  
| 2-HSM   | 50k        | 17626    | 0:30       | 13.87   | 7.6GiB           |  
| 2-HSM   | 100k       | 18700    | 0:28       | 14.71   | 10.8GiB          |  
|         |            |          |            |         |                  |
| SSM     | 1k         | 2682     | 3:20       | 2.11    | 1.5GiB           |
| SSM     | 3k         | 4662     | 1:55       | 3.67    | 2.5GiB           |
| SSM     | 5k         | 5541     | 1:36       | 4.36    | 6.5GiB           |
| SSM     | 10k        | 6590     | 1:21       | 5.18    | 8.4GiB           |
| SSM     | 20k        | 7131     | 1:15       | 5.61    | 7.9GiB           |
| SSM     | 30k        | 7318    |  1:13      |  5.76    | 10.5GiB          |

## Conclusions

Sampled softmax is very easy and straightforward to implement, but it's less efficient from GPU memory point of view,
which leads to less speed up that two-level hierarchical softmax.  
