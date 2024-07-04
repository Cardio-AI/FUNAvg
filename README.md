# FUNAvg: Federated Uncertainty Weighted Averaging for Datasets with Diverse Labels

Malte Tölle, Fernando Navarro, Sebastian Eble, Ivo Wolf, Bjoern Menze, Sandy Engelhardt

Code for paper accepted at MICCAI 2024.

Paper link: 

## Abstract

Federated learning is one popular paradigm to train a joint model in a distributed, privacy-preserving environment. 
But partial annotations pose an obstacle meaning that categories of labels are heterogeneous over clients.
We propose to learn a joint backbone in a federated manner while each site receives its own multi-label segmentation head.
By using Bayesian techniques we observe that the different segmentation heads although only trained on the individual client's labels also learn information about the other labels not present at the respective site. 
This information is encoded in their predictive uncertainty.
To obtain a final prediction we leverage this uncertainty and perform a weighted averaging of the ensemble of distributed segmentation heads, which allows us to segment "locally unknown" structures.
With our method, which we refer to as FUNAvg, we even outperform models trained and tested on the same dataset on average (0.868 vs. 0.848 DICE).

## BibTeX

```
@inproceedings{toelle2024funavg,
    title={FUNAvg: Federated Uncertainty Weighted Averaging for Datasets with Diverse Labels},
    author={T{\"o}lle, Malte and Navarro, Fernando and Eble, Sebastian and Wolf, Ivo and Menze, Bjoern and Engelhardt, Sandy},
    booktitle={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    volume={},
    pages={},
    year={2024},
    doi={}
}
```

## Contact

Malte Tölle<br>
[malte.toelle@med.uni-heidelberg.de](mailto:malte.toelle@med.uni-heidelberg.de)<br>
[@maltetoelle](https://x.com/maltetoelle)<br>

Group Artificial Intelligence in Cardiovascular Medicine (AICM)<br>
Heidelberg University Hospital<br>
Im Neuenheimer Feld 410, 69120 Heidelberg, Germany