Reimplementation of [A Novel Chaos Theory Inspired Neuronal Architecture"](https://arxiv.org/pdf/1905.12601.pdf).     

I reimplemented the architecture presented in the paper, but rather than focusing on the small training set
regime, I focused on standard 10-fold cross-validation. I feel like stopping here kind of for the moment, but
I'm happy to share the code for interested souls.

# Setup

To speed things up the functions of the neuron are written in Cython, so it requires compilation:

```bash
python3 setup.py build_ext --inplace
```


# Iris

The 10-fold cross-val experiments for iris are in ```iris.py```.

The best result I've got with the pure Python version (no Cython speedup) was a macro averaged F1-score of 0.93 averaged over 
10-folds of crossval with ```q = 0.59``` and ```b = 0.9```.   


I've found a large variance in performance searching through ```q``` and ```b```.    
Below the plot shows the distribution of average F1-scores with 10-fold cross-validation on Iris.   
The mean performance is around 0.6 and getting the best 0.93 is very lucky.

![distplot](dist.png)


This performance can be improved to 0.96 using 80-bit floating points and performance drops to 0.88 
with 32-bit representaton. Pretty dark

# MNIST

The experiments on the full MNIST take a long time, so far the best result so far is F1-score around 0.81 
10-fold cross-valling on the 60K training set. You can run this experiments with ```mnist.py```. It runs
parallellism across samples, because its super slow otherwise (super slow this way too, sorry).


# Take away

The model is super sensitive to the values of ```q``` and ```b```, not to mention that even ```epsilon``` is a hyperparameter.
I also couldn't find a way to implement it very fast: the iteration inside the neurons prevent me from thinking about a
good vectorized implementation. When I come back to it my plan is to randomly initialize ```b``` and ```q```, different for
each neuron and search for good values with  Evolution or Particle Swarm or something. Cheerio!
