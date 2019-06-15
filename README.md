# GeoMLE

[![Build Status](https://travis-ci.com/premolab/GeoMLE.svg?branch=master)](https://travis-ci.com/premolab/GeoMLE)

This repo contains code for our paper [Geometry-Aware Maximum Likelihood Estimation of Intrinsic Dimension](https://arxiv.org/abs/1904.06151)

## Abstract

The existing approaches to intrinsic dimension estimation usually are not reliable when the data are nonlinearly embedded in the high dimensional space. In this work, we show that the explicit accounting to geometric properties of unknown support leads to the polynomial correction to the standard maximum likelihood estimate of intrinsic dimension for flat manifolds. The proposed algorithm (GeoMLE) realizes the correction by regression of standard MLEs based on distances to nearest neighbors for different sizes of neighborhoods. Moreover, the proposed approach also efficiently handles the case of nonuniform sampling of the manifold. We perform numerous experiments on different synthetic and real-world datasets. The results show that our algorithm achieves state-of-the-art performance, while also being computationally efficient and robust to noise in the data.

## Quick Start with library GeoMLE

### Data generation
```python
from geomle import DataGenerator
DG = DataGenerator()
data = DG.gen_data('Sphere', 1000, 2, 1)
```
#### Algorithm Levina-Bickel (MLE)
```python
from geomle import mle
mle(data)
```
#### Algorithm GeoMLE
```python
from geomle import geomle
geomle(data)
```

## Experiments

All experiments you can find in [notebook](paper/FinalNtb.ipynb):

- [x] Decribing algorithms
- [x] Test with nonuniform distibution
- [x] Dependence on manifold dimension
- [x] Dependence on number of points
- [x] Comparing algorithms with Dolan-More curves
- [x] Dependence on noise
- [ ] Dependence on neigbors (k1 and k2)


## Algorithms

In this paper we compare our approch with many famous algorithms:
* MLE
* ESS
* MIND
* DANCo
* Local PCA

We use this [implementation](https://cran.r-project.org/web/packages/intrinsicDimension/index.html) in R.

## BibTex

```
@article{GeoMLE2019,
  title={Geometry-Aware Maximum Likelihood Estimation of Intrinsic Dimension},
  author={Marina Gomtsyan and Nikita Mokrov and Maxim Panov and Yury Yanovich},
  journal={arXiv preprint arXiv:1904.06151},
  year={2019},
  url = {https://arxiv.org/abs/1904.06151},
}
```


