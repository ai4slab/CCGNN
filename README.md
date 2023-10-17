# CCGNN

*Under review.*

## Datasets

Public datasets used in this paper:

|                    Dataset                    | #LncRNAs | #Proteins | #Interactions | Sparsity (%) |
|:---------------------------------------------:|:--------:|:---------:|:-------------:|:------------:|
|   [DB1](https://github.com/haichengyi/MAN)    |   2014   |    74     |     5115      |    96.57     |
| [DB2](https://github.com/NWPU-903PR/LPI_BLS)  |   1874   |    118    |     7317      |    96.69     |
| [DB3](http://39.100.104.29:8080/lpc/download) |   2356   |    90     |     6204      |    97.07     |
|  [DB4](https://github.com/zhaoqi106/LPICGAE)  |   3046   |    136    |     8112      |    98.04     |

## Requirements

Please ensure that all the libraries below are successfully installed:

* python==3.9.0
* torch==1.10.2
* torch_geometric==2.0.3
* numpy==1.22.2
* pandas==1.3.5
* scikit-learn==1.0.2

## Usage

You can directly run the following code to run the model:

```
cd ./code/
python main.py
```
