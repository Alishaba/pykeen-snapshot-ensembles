<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>

<h1 align="center">
  PyKEEN
</h1>

<p align="center">
  <a href="https://travis-ci.com/mali-git/POEM_develop">
    <img src="https://travis-ci.com/mali-git/POEM_develop.svg?token=2tyMYiCcZbjqYscNWXwZ&branch=master"
         alt="Travis CI">
  </a>

  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>
</p>

<p align="center">
    <b>PyKEEN</b> (<b>P</b>ython <b>K</b>nowl<b>E</b>dge <b>E</b>mbeddi<b>N</b>gs) is a Python package designed to
    train and evaluate knowledge graph embedding models (incorporating multi-modal information). It is part of the
    <a href="https://github.com/SmartDataAnalytics/PyKEEN">KEEN Universe</a>.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#models">Models</a> •
  <a href="#datasets">Data Sets</a> •
  <a href="#supporters">Support</a>
</p>


## Installation

The development version of PyKEEN (POEM) can be downloaded and installed from
[GitHub](https://github.com/mali-git/POEM_develop) on Python 3.7+ with:

```bash
$ git clone https://github.com/mali-git/POEM_develop.git
$ cd poem
$ pip install -e .
$ # Install pre-commit
$ pip install pre-commit
$ pre-commit install
```

## Quickstart

This example shows how to train a model on a data set and test on another data set.

The fastest way to get up and running is to use the pipeline function. It
provides a high-level entry into the extensible functionality of this package.
The following example shows how to train and evaluate the TransE model on the
Nations dataset. By default, the training loop uses the open world assumption
and evaluates with rank-based evaluation.

```python
from poem.pipeline import pipeline
result = pipeline(
    model='TransE',
    data_set='nations',
)
```

The results are returned in a dataclass that has attributes for the trained
model, the training loop, and the evaluation.

POEM is extensible such that:

- Each model has the same API, so anything from ``poem.models`` can be dropped in
- Each training loop has the same API, so ``poem.training.CWATrainingLoop`` can be dropped in
- Triples factories can be generated by the user with ``from poem.triples.TriplesFactory``

## Implementation

Below are the models, data sets, training modes, evaluators, and metrics implemented
in POEM. These markdown tables can be regenerated with `poem ls`.

### Models (21)

| Name                | Reference                         | Citation                 |
|---------------------|-----------------------------------|--------------------------|
| ComplEx             | `poem.models.ComplEx`             | Trouillon *et al.*, 2016 |
| ComplExLiteralCWA   | `poem.models.ComplExLiteral`      | Agustinus *et al.*, 2018 |
| ConvE               | `poem.models.ConvE`               | Dettmers *et al.*, 2018  |
| ConvKB              | `poem.models.ConvKB`              | Nguyen *et al.*, 2018    |
| DistMult            | `poem.models.DistMult`            | Yang *et al.*, 2014      |
| DistMultLiteral     | `poem.models.DistMultLiteral`     | Agustinus *et al.*, 2018 |
| ERMLP               | `poem.models.ERMLP`               | Dong *et al.*, 2014      |
| HolE                | `poem.models.HolE`                | Nickel *et al.*, 2016    |
| KG2E                | `poem.models.KG2E`                | He *et al.*, 2015        |
| NTN                 | `poem.models.NTN`                 | Socher *et al.*, 2013    |
| ProjE               | `poem.models.ProjE`               | Shi *et al.*, 2017       |
| RESCAL              | `poem.models.RESCAL`              | Nickel *et al.*, 2011    |
| RotatE              | `poem.models.RotatE`              | Sun *et al.*, 2019       |
| SimplE              | `poem.models.SimplE`              | Kazemi *et al.*, 2018    |
| StructuredEmbedding | `poem.models.StructuredEmbedding` | Bordes *et al.*, 2011    |
| TransD              | `poem.models.TransD`              | Ji *et al.*, 2015        |
| TransE              | `poem.models.TransE`              | Bordes *et al.*, 2013    |
| TransH              | `poem.models.TransH`              | Wang *et al.*, 2014      |
| TransR              | `poem.models.TransR`              | Lin *et al.*, 2015       |
| TuckER              | `poem.models.TuckER`              | Balazevic *et al.*, 2019 |
| UnstructuredModel   | `poem.models.UnstructuredModel`   | Bordes *et al.*, 2014    |

### Data Sets (8)

| Name     | Reference                | Description                                                                                        |
|----------|--------------------------|----------------------------------------------------------------------------------------------------|
| fb15k    | `poem.datasets.fb15k`    | The FB15k data set.                                                                                |
| fb15k237 | `poem.datasets.fb15k237` | The FB15k-237 data set.                                                                            |
| kinship  | `poem.datasets.kinship`  | The Kinship data set.                                                                              |
| nations  | `poem.datasets.nations`  | The Nations data set.                                                                              |
| umls     | `poem.datasets.umls`     | The UMLS data set.                                                                                 |
| wn18     | `poem.datasets.wn18`     | The WN18 data set.                                                                                 |
| wn18rr   | `poem.datasets.wn18rr`   | The WN18-RR data set.                                                                              |
| yago310  | `poem.datasets.yago310`  | The YAGO3-10 data set is a subset of YAGO3 that only contains entities with at least 10 relations. |

### Training Modes (2)

| Name   | Reference                       | Description                                            |
|--------|---------------------------------|--------------------------------------------------------|
| cwa    | `poem.training.CWATrainingLoop` | A training loop that uses the closed world assumption. |
| owa    | `poem.training.OWATrainingLoop` | A training loop that uses the open world assumption.   |

### Evaluators (1)

| Name      | Reference                            | Description                            |
|-----------|--------------------------------------|----------------------------------------|
| rankbased | `poem.evaluators.RankBasedEvaluator` | A rank-based evaluator for KGE models. |

### Metrics (1)

| Name   | Reference                       | Description                     |
|--------|---------------------------------|---------------------------------|
| metric | `poem.evaluators.MetricResults` | Results from computing metrics. |

## Acknowledgements

### Supporters

This project has been supported by several organizations:

- [Smart Data Analytics (University of Bonn)](http://sda.cs.uni-bonn.de)
- [Fraunhofer Institute for Intelligent Analysis and Information Systems](https://www.iais.fraunhofer.de)
- [Bonn Aachen International Center for IT (University of Bonn)](http://www.b-it-center.de)
- [Fraunhofer Institute for Algorithms and Scientific Computing](https://www.scai.fraunhofer.de)
- [Fraunhofer Center for Machine Learning](https://www.cit.fraunhofer.de/de/zentren/maschinelles-lernen.html)
- [Munich Center for Machine Learning (MCML)](https://mcml.ai/)
- [Technical University of Denmark - DTU Compute - Section for Cognitive Systems](https://www.compute.dtu.dk/english/research/research-sections/cogsys)
- [Technical University of Denmark - DTU Compute - Section for Statistics and Data Analysis](https://www.compute.dtu.dk/english/research/research-sections/stat)

### Logo

The PyKEEN logo was designed by Carina Steinborn.
