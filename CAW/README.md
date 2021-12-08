# Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks

## Introduction

This is the simplified version of the original implementation. The original implementation could be found on the website https://github.com/snap-stanford/CAW. This implementation also reference the code of TGAT, link is https://drive.google.com/drive/folders/1GaH8vusCXJj4ucayfO-PyHpnNsJRkB78. 



The codes here are for project experiments. Hence the data analysis functions that are not used in the project are removed. Some optional modules that the authors provide are also removed, since they are not used in original paper experiments. Codes' structure and implementation are adjusted, so that it can be understood easily. 




## Dataset and preprocessing
#### Use preprocessed data provided by the authors.
The authors provide preprocessed datasets: Reddit, Wikipedia, Enron, and UCI. Download them from [here](https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing) to `processed/`. Then run the following:
```{bash}
cd processed/
unzip data.zip
```


## Training Commands

Training commands are not modified. We just paste the original commands here.

#### Examples:

* To train **CAW-N-mean** with Wikipedia dataset in inductive training, sampling 64 length-2 CAWs every node, and with alpha = 1e-5:
```bash
python main.py -d wikipedia --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0
```

* To train **CAW-N-attn** with UCI dataset in transductive mode, sampling 32 length-1 CAWs every node, with alpha = 1e-6, and using another random seed 123:
```bash
python main.py -d uci --pos_dim 100 --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --walk_pool attn --seed 123
```

Detailed logs can be found in `log/`, a one-line summary of the evaluation result will also be written to `log/oneline_summary.log` upon completion.

## Optional arguments

Some useless arguments are removed for code simplicity. We leave the rest here.

```{txt}
  -h, --help            show this help message and exit
  -d {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks}, --data {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks}
                        data sources to use, try wikipedia or reddit
  -m {t,i}, --mode {t,i}
                        transductive (t) or inductive (i)
  --n_degree [N_DEGREE [N_DEGREE ...]]
                        a list of neighbor sampling numbers for different hops, when only a single element is input n_layer
                        will be activated
  --n_layer N_LAYER     number of network layers
  --bias BIAS           the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is
                        uniform sampling
  --pos_dim POS_DIM     dimension of the positional embedding
  --pos_sample {multinomial,binary}
                        two practically different sampling methods that are equivalent in theory
  --walk_pool {attn,sum}
                        how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other
                        walk_ arguments
  --walk_n_head WALK_N_HEAD
                        number of heads to use for walk attention
  --walk_mutual         whether to do mutual query for source and target node random walks
  --walk_linear_out     whether to linearly project each node's

  --n_epoch N_EPOCH     number of epochs
  --bs BS               batch_size
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability for all dropout layers
  --tolerance TOLERANCE
                        toleratd margainal improvement for early stopper
  --seed SEED           random seed for all randomized algorithms
  --ngh_cache           (currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously
                        calculated to speed up repeated lookup
  --cpu_cores CPU_CORES
                        number of cpu_cores used for position encoding
  --verbosity VERBOSITY
                        verbosity of the program output
```

