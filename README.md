<p align="center">
  <img src="docs/source/logo.png" height="150">
</p>

<h1 align="center">
  SnapE-PyKEEN
</h1>

  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/242672435">
    <img src="https://zenodo.org/badge/242672435.svg" alt="DOI">
  </a>

  <a href="https://optuna.org">
    <img src="https://img.shields.io/badge/Optuna-integrated-blue" alt="Optuna integrated" height="20">
  </a>

  <a href="https://pytorchlightning.ai">
    <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="PyTorch Lightning">
  </a>

  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
  </a>

  <a href=".github/CODE_OF_CONDUCT.md">
    <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant">
  </a>
</p>

<p align="center">
    <b>SnapE-PyKEEN</b> (<b>S</b>napshot <b>E</b>nsembles <b>P</b>ython <b>K</b>nowl<b>E</b>dge <b>E</b>mbeddi<b>N</b>gs) is a Python package adapted from [PyKEEN](https://github.com/pykeen/pykeen) to
    allow training and evaluating snapshot ensembles of knowledge graph embedding models. Additionally, it also includes an extended negative sampler that iteratively creates negative examples using previous snapshot models. All functionalities and components provided in PyKEEN are also available in SnapE-PyKEEN.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#Evaluating an ensemble of KGEMs">Evaluating an ensemble of KGEMs</a> •
  <a href="#Extended Negative Sampler">Extended Negative Sampler</a> 
</p>

## Installation

SnapE-PyKEEN requires Python 3.8+. It can be downloaded
and installed directly from the
source code on [GitHub](https://github.com/Alishaba/pykeen-snapshot-ensembles) with:

```shell
pip install git+https://github.com/Alishaba/pykeen-snapshot-ensembles.git
```
More information about PyKEEN (e.g., installation, first steps, Knowledge Graph Embedding Models, extras) can be found in the [documentation](https://pykeen.readthedocs.io/en/latest/index.html).


## Training a Snapshot Ensemble of KGEMs

```python
from pykeen.datasets import FB15k237
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.training import ModelSavingCallback

# Get FB15k-237 dataset
dataset = FB15k237()

# Define model
model = TransE(
    triples_factory=dataset.training,
)

batch_size = 128
max_lr = 0.1
num_epochs = 500
step = 50
num_snapshots=10
dataset_name = 'FB15k237'
model_name = 'TransE'
method = 'paper'


# Train your model (code is omitted for brevity)
pipeline_result = pipeline(
    dataset=dataset,
    random_seed=10,
    model=model,
    training_kwargs=dict(
      drop_last=True,
      num_epochs=num_epochs,
      callbacks=ModelSavingCallback,
      callback_kwargs=dict(
          batch_size=batch_size,
          dataset_size=training.num_triples,
          max_lr=max_lr,
          min_lr=None,
          step=step,
          num_snapshots=num_snapshots,
          num_epochs=num_epochs,
          method=method,
          dataset_name=dataset_name,
          model_name=model_name
      )
    )
)
```


## Extended Negative Sampler

```python
from pykeen.datasets import FB15k237
from pykeen.sampling import ExtendedBasicNegativeSampler
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.training import ModelSavingCallback

# Get FB15k-237 dataset
dataset = FB15k237()

# Define model
model = TransE(
    triples_factory=dataset.training,
)

batch_size = 128
num_batches = math.floor(dataset.training.num_triples/batch_size),
models_to_load = 5,
max_lr = 0.1
num_epochs = 500
step = 50
num_snapshots=10
dataset_name = 'FB15k237'
model_name = 'TransE'
method = 'paper'

negative_sampler_kwargs = dict(
  num_batches=num_batches,
  models_to_load=models_to_load,
  num_epochs=num_epochs,
  step=step,
  dataset_name=dataset_name,
  model_name=model_name,
  method=method,
)

# Train your model
pipeline_result = pipeline(
    dataset=dataset,
    random_seed=10,
    model=model,
    negative_sampler=ExtendedBasicNegativeSampler,
    negative_sampler_kwargs=negative_sampler_kwargs,
    training_kwargs=dict(
      drop_last=True,
      num_epochs=num_epochs,
      callbacks=ModelSavingCallback,
      callback_kwargs=dict(
          batch_size=batch_size,
          dataset_size=training.num_triples,
          max_lr=max_lr,
          min_lr=None,
          step=step,
          num_snapshots=num_snapshots,
          num_epochs=num_epochs,
          method=method,
          dataset_name=dataset_name,
          model_name=model_name
      )
    )
)
```


## Evaluating a (snapshot) ensemble of KGEMs

The example provided in the previous section shows how to evaluate a single trained Knowledge Graph Embedding Model.  SnapE-PyKEEN extends this capability to provide rank-based evaluation for ensembles of Knowledge Graph Embedding Models.

The example below shows how to use EnsembleRankBasedEvaluator to evaluate an ensemble. In the example below, the scores are aggregated using a simple average.

```python
from pykeen.datasets import FB15k237
from pykeen.sampling import ExtendedBasicNegativeSampler
from pykeen.evaluation import EnsembleRankBasedEvaluator
from pykeen.models import TransE
from pykeen.pipeline import pipeline
import torch

# Get FB15k-237 dataset
dataset = FB15k237()

# Define model
model = TransE(
    triples_factory=dataset.training,
)

batch_size = 128
num_batches = math.floor(dataset.training.num_triples/batch_size),
models_to_load = 5,
max_lr = 0.1
num_epochs = 500
step = 50
num_snapshots=10
dataset_name = 'FB15k237'
model_name = 'TransE'
method = 'paper'

negative_sampler_kwargs = dict(
  num_batches=num_batches,
  models_to_load=models_to_load,
  num_epochs=num_epochs,
  step=step,
  dataset_name=dataset_name,
  model_name=model_name,
  method=method,
)

# Train your model
pipeline_result = pipeline(
    dataset=dataset,
    random_seed=10,
    model=model,
    negative_sampler=ExtendedBasicNegativeSampler,
    negative_sampler_kwargs=negative_sampler_kwargs,
    training_kwargs=dict(
      drop_last=True,
      num_epochs=num_epochs,
      callbacks=ModelSavingCallback,
      callback_kwargs=dict(
          batch_size=batch_size,
          dataset_size=training.num_triples,
          max_lr=max_lr,
          min_lr=None,
          step=step,
          num_snapshots=num_snapshots,
          num_epochs=num_epochs,
          method=method,
          dataset_name=dataset_name,
          model_name=model_name
      )
    )
)

# Load the last m=models_to_load models to ensemble
models = [torch.load(f"./models/trained_model_{dataset_name}_{model_name}_{method}_{num_epochs - i * step}.pkl") for i in range(0, models_to_load)]

# Define evaluator
evaluator = EnsembleRankBasedEvaluator()

# Evaluate your model with not only testing triples,
# but also filter on validation triples
results = evaluator.evaluate(
    model=models,
    mapped_triples=dataset.testing.mapped_triples,
    additional_filter_triples=[
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
    ],
)
```

Moreover, the scores of base models can be normalized and aggregated using weighted average. The example below shows how to pass weights and define a normalizer.

```python
from pykeen.datasets import FB15k237
from pykeen.sampling import ExtendedBasicNegativeSampler
from pykeen.evaluation import EnsembleRankBasedEvaluator
from pykeen.models import TransE
from pykeen.pipeline import pipeline
import torch

# Get FB15k-237 dataset
dataset = FB15k237()

# Define model
model = TransE(
    triples_factory=dataset.training,
)

# Train your model 
batch_size = 128
num_batches = math.floor(dataset.training.num_triples/batch_size),
models_to_load = 5,
max_lr = 0.1
num_epochs = 500
step = 50
num_snapshots=10
dataset_name = 'FB15k237'
model_name = 'TransE'
method = 'paper'

negative_sampler_kwargs = dict(
  num_batches=num_batches,
  models_to_load=models_to_load,
  num_epochs=num_epochs,
  step=step,
  dataset_name=dataset_name,
  model_name=model_name,
  method=method,
)

# Train your model
pipeline_result = pipeline(
    dataset=dataset,
    random_seed=10,
    model=model,
    negative_sampler=ExtendedBasicNegativeSampler,
    negative_sampler_kwargs=negative_sampler_kwargs,
    training_kwargs=dict(
      drop_last=True,
      num_epochs=num_epochs,
      callbacks=ModelSavingCallback,
      callback_kwargs=dict(
          batch_size=batch_size,
          dataset_size=training.num_triples,
          max_lr=max_lr,
          min_lr=None,
          step=step,
          num_snapshots=num_snapshots,
          num_epochs=num_epochs,
          method=method,
          dataset_name=dataset_name,
          model_name=model_name
      )
    )
)

# Load the last m=models_to_load models to ensemble
models = [torch.load(f"./models/trained_model_{dataset_name}_{model_name}_{method}_{num_epochs - i * step}.pkl") for i in range(0, models_to_load)]

# Define the weights. Weights will be normalized.
weights = [models_to_load - i for i in range(0,models_to_load)]

# Define wether to calculate Borda ranks
borda = True

# Define which normalizer to use out of ['MinMax', 'Standard', None]
normalize = 'MinMax'

# Define evaluator
evaluator = EnsembleRankBasedEvaluator()

# Evaluate your model with not only testing triples,
# but also filter on validation triples
results = evaluator.evaluate(
    model=models,
    mapped_triples=dataset.testing.mapped_triples,
    additional_filter_triples=[
        dataset.training.mapped_triples,
        dataset.validation.mapped_triples,
    ],
    weights=weights, 
    borda=borda, 
    normalize=normalize
)
```
