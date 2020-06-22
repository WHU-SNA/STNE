# STNE
Python implementation of  the method proposed in "[Social trust Network Embedding](https://ieeexplore.ieee.org/document/8970926)", Pinghua Xu, Wenbin Hu, Jia Wu, Weiwei Liu, Bo Du and Jian Yang, ICDM 2019.

## Overview
This repository is organised as follows:
- `input/` contains four example graphs `WikiEditor` `WikiElec` `WikiRfa` `Slashdot` `Epinions`;
- `output/` is the directory to store the learned node embeddings;
- `src/` contains the implementation of the proposed STNE model.

## Requirements
The implementation is tested under Python 3.7, with the folowing packages installed:
- `networkx==2.3`
- `numpy==1.16.5`
- `scikit-learn==0.21.3`
- `texttable==1.6.2`
- `tqdm==4.36.1`

## Input
We investigated **social trust network**, which can be represented by a **directed signed (un)weighted** graph, in this work.

The code takes an input graph in `.txt` format. Every row indicates an edge between two nodes separated by a `space` or `\t`. The file does not contain a header. Nodes can be indexed starting with any non-negative number. Five example graphs are included in the `input/` directory. Among these graphs, `WikiElec`, `WikiRfa`, `Slashdot` and `Epinions` are donwloaded from [SNAP](http://snap.stanford.edu/data/#signnets), but node ID is resorted, and `WikiEditor` is generated according to the description in our paper. The structure of the input file is the following:

| Source node | Target node | Weight |
| :-----:| :----: | :----: |
| 0 | 1 | 4 |
| 1 | 3 | -5 |
| 1 | 2 | 2 |
| 2 | 4 | -1 |

## Options
#### Input and output options
```
--edge-path                 STR      Input file path                      Default=="./input/WikiElec.txt"
--outward-embedding-path    STR      Outward embedding path               Default=="./output/WikiElec_outward"
--inward-embedding-path     STR      Inward embedding path                Default=="./output/WikiElec_inward"
```
#### Model options
```
--dim                       INT      Dimension of latent factor vector    Default==32
--n                         INT      Number of noise samples              Default==5
--num_walks                 INT      Walks per node                       Default==20
--walk_len                  INT      Length per walk                      Default==10
--workers                   INT      Number of threads used for random walking    Default==4
--m                         FLOAT    Damping factor                       Default==1
--norm                      FLOAT    Normalization factor                 Default==0.01
--learning-rate             FLOAT    Leaning rate                         Default==0.02
```
#### Evaluation options
```
--test-size                 FLOAT    Test ratio                           Default==0.2
--split-seed                INT      Random seed for splitting dataset    Default==16
```

## Examples
Train an STNE model on the deafult `WikiElec` dataset, output the performance on sign prediction task, and save the embeddings:
```
python src/main.py
```

Train an SLF model with custom split seed and test ratio:
```
python src/main.py --split_seed 20 --test-size 0.3
```

Train an SLF model on the `WikiRfa` dataset:
```
python src/main.py --edge-path ./input/WikiRfa.txt  --outward-embedding-path ./output/WikiRfa_outward --inward-embedding-path ./output/WikiRfa_inward
```

## Output
We perform sign prediction to evaluate the node embeddings. And we use `AUC` and `macro-F1` as evaluation metric. Although `Micro-F1` is used in our paper, we admit that it is not a good choice for evaluation on a dataset with unbalanced labels.

Run `python src/main.py`, and the output is printed like the following:
```
Optimizing: 100%|█████████████████████████████████████████████| 75496/75496 [13:20<00:00, 94.32it/s]
Sign prediction: AUC 0.874, F1 0.747
```
Like the other methods in Skip-Gram family, we only perform one-epoch training.

## Baselines
In our paper, we used the following methods for comparison:
- `N2V`     "node2vec: Scalable Feature Learning for Networks" [[source](https://github.com/aditya-grover/node2vec)]
- `LINE`    "LINE: Large-scale information network embedding" [[source](https://github.com/tangjianpku/LINE)]
- `LSNE`    "Solving link-oriented tasks in signed network via an embedding approach"
- `MF`      "Low rank modeling of signed networks"
- `SIDE`    "Side: representation learning in signed directed networks" [[source](https://datalab.snu.ac.kr/side/)]
- `SIGNet`  "Signet: Scalable embeddings for signed networks" [[source](https://github.com/raihan2108/signet)]

`MF` and `LSNE` are not open-sourced, but if you are interested in our implementation of these methods, email to xupinghua@whu.edu.cn

## Cite
If you find this repository useful in your research, please cite our paper:

```
@INPROCEEDINGS{8970926,
  author={Pinghua Xu and Wenbin Hu and Jia Wu and Weiwei Liu and Bo Du and Jian Yang},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  title={Social Trust Network Embedding},
  year={2019},
  pages={678-687}
}
```

Moreover, if you are interested in the topic of **social trust network**, you may want to know our another work "[Opinion Maximization in Social Trust Networks](http://arxiv.org/abs/2006.10961)" (IJCAI 2020).
