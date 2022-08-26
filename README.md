## Datasets

**Douban**: You can find processed data  in data/data or 

process the source datasets by data/preprocess_DoubanMovie.py

**epinions**:You can process the source code Extended_epinion by data/preprocess_Epinion.py



The specific processing methods can refer to https://github.com/trumpjoker/RecommenderSystems/blob/master/socialRec/README.md



## Environment

- python 3.6
- tensorflow 1.4.0
- numpy 1.13.1
- pandas  0.20.3



## Citation

Please cite our paper if you use the code:

```
@inproceedings{
title = {{GNNRec: gated graph neural network for session-based social recommendation model}},
author = {Chun Liu, Yuxiang Li, Hong Lin & Chaojie Zhang},
year = 2022,
booktitle = {Journal of Intelligent Information Systems},
location = {WUHAN, HUBEI, CHINA},
month = August,
url = {https://link.springer.com/article/10.1007/s10844-022-00733-5#Sec17},
doi = {https://doi.org/10.1007/s10844-022-00733-5},
}
```



## Acknowledgement

Thanks for Weiping Song's excellent project [DGREC](https://github.com/trumpjoker/RecommenderSystems/tree/master/socialRec/dgrec).

Thanks for Wu's excellent project [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).