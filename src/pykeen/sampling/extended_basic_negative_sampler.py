# -*- coding: utf-8 -*-

"""Negative sampling algorithm based on the work of of Bordes *et al.*."""

import math
from typing import Collection, Optional

import torch

from .negative_sampler import NegativeSampler
from ..constants import LABEL_HEAD, LABEL_TAIL, TARGET_TO_INDEX
from ..typing import Target

__all__ = [
    "ExtendedBasicNegativeSampler",
    "random_replacement_",
    "adversarial_replacement_"
]


def adversarial_replacement_(model, batch: torch.LongTensor, index: int, selection: slice, k: int = 1) -> None:

    slice_batch = batch[selection, :]
    # get rt_batch for head scores and hr_batch for tail scores
    # TODO: expand to cover corrupting relations as well
    triples_to_scores = slice_batch[:,1:] if index==0 else slice_batch[:,:2]
    # get true heads or tails
    true_side = slice_batch[:,index]
    true_side = true_side.view(-1,1)
    # get scores of all possible heads for each triple
    scores_side = model.predict_h(triples_to_scores) if index==0 else model.predict_t(triples_to_scores)
    # create a tensor of indcies
    just_indices = torch.arange(0, true_side.size(0)).view(-1,1)
    # find the minimum score 
    min_score = scores_side.min()
    # replace scores of true heads with the minimal value to avoid getting them in top k
    scores_side[just_indices, true_side] = min_score - 1
    # find indices of top k heads per triple
    _, replacement = torch.topk(scores_side, k, dim=1)
    # create the negative triples 
    batch[selection, index] = torch.squeeze(replacement)


def random_replacement_(batch: torch.LongTensor, index: int, selection: slice, size: int, max_index: int) -> None:
    """
    Replace a column of a batch of indices by random indices.

    :param batch: shape: `(*batch_dims, d)`
        the batch of indices
    :param index:
        the index (of the last axis) which to replace
    :param selection:
        a selection of the batch, e.g., a slice or a mask
    :param size:
        the size of the selection
    :param max_index:
        the maximum index value at the chosen position
    """
    # At least make sure to not replace the triples by the original value
    # To make sure we don't replace the {head, relation, tail} by the
    # original value we shift all values greater or equal than the original value by one up
    # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
    replacement = torch.randint(
        high=max_index - 1,
        size=(size,),
        device=batch.device,
    )
    replacement += (replacement >= batch[selection, index]).long()
    batch[selection, index] = replacement


class ExtendedBasicNegativeSampler(NegativeSampler):
    r"""A basic negative sampler.

    This negative sampler that corrupts positive triples $(h,r,t) \in \mathcal{K}$ by replacing either $h$, $r$ or $t$
    based on the chosen corruption scheme. The corruption scheme can contain $h$, $r$ and $t$ or any subset of these.

    Steps:

    1. Randomly (uniformly) determine whether $h$, $r$ or $t$ shall be corrupted for a positive triple
       $(h,r,t) \in \mathcal{K}$.
    2. Randomly (uniformly) sample an entity $e \in \mathcal{E}$ or relation $r' \in \mathcal{R}$ for selection to
       corrupt the triple.

       - If $h$ was selected before, the corrupted triple is $(e,r,t)$
       - If $r$ was selected before, the corrupted triple is $(h,r',t)$
       - If $t$ was selected before, the corrupted triple is $(h,r,e)$
    3. If ``filtered`` is set to ``True``, all proposed corrupted triples that also exist as
       actual positive triples $(h,r,t) \in \mathcal{K}$ will be removed.
    """

    def __init__(
        self,
        *,
        corruption_scheme: Optional[Collection[Target]] = None,
        num_batches,
        models_to_load,
        num_epochs,
        step,
        dataset_name,
        model_name,
        method,
        **kwargs,
    ) -> None:
        """Initialize the basic negative sampler with the given entities.

        :param corruption_scheme:
            What sides ('h', 'r', 't') should be corrupted. Defaults to head and tail ('h', 't').
        :param kwargs:
            Additional keyword based arguments passed to :class:`pykeen.sampling.NegativeSampler`.
        """
        super().__init__(**kwargs)
        self.corruption_scheme = corruption_scheme or (LABEL_HEAD, LABEL_TAIL)
        # Set the indices
        self._corruption_indices = [TARGET_TO_INDEX[side] for side in self.corruption_scheme]
        self.model = None
        self.k = 1
        self.c = 0
        self.num_batches = num_batches
        self.models_to_load = models_to_load 
        self.num_epochs = num_epochs
        self.step=step
        self.num_epochs_passed = num_epochs - models_to_load * step
        self.epoch = 0
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.method = method
        self.num_epoch_model_loaded = 0
        self.start_adversarial_replacement = False

    # docstr-coverage: inherited
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        batch_shape = positive_batch.shape[:-1]

        # clone positive batch for corruption (.repeat_interleave creates a copy)
        negative_batch = positive_batch.view(-1, 3).repeat(self.num_negs_per_pos, 1)#.repeat_interleave(self.num_negs_per_pos, dim=0)

        self.c += 1
        self.epoch = math.ceil(self.c/self.num_batches)
        if self.epoch % self.step == 1 and self.epoch > self.num_epochs_passed + self.step and self.num_epoch_model_loaded != self.epoch:
            print('loading a model for negative sampling')
            print('epoch', self.epoch, 'batch', self.c)
            print(f"./models/trained_model_{self.dataset_name}_{self.model_name}_{self.method}_{self.epoch - 1}.pkl")
            self.num_epoch_model_loaded = self.epoch
            self.model = torch.load(f"./models/trained_model_{self.dataset_name}_{self.model_name}_{self.method}_{self.epoch - 1}.pkl")
            self.start_adversarial_replacement = True
                
        total_num_negatives = negative_batch.shape[0]
        split_idx = int(math.ceil(total_num_negatives / len(self._corruption_indices)))


        # linear equation to determine proportion of random triples starting from and linearly decreasing to 0
        random_portion = self.c/(self.num_batches * (self.num_epochs_passed + self.step - self.num_epochs)) - self.num_epochs/(self.num_epochs_passed + self.step - self.num_epochs)
        random_portion = min(1, random_portion) # if self.start_adversarial_replacement else 1

        for index, start in zip(self._corruption_indices, range(0, total_num_negatives, split_idx)):
            stop = min(start + split_idx, total_num_negatives)
            stop_random = math.ceil((stop - start) * random_portion) + start
            random_replacement_(
                batch=negative_batch,
                index=index,
                selection=slice(start, stop_random),
                size=stop_random - start,
                max_index=self.num_relations if index == 1 else self.num_entities,
            )
            if random_portion < 1 and stop_random < stop:
                print('start', start, 'stop', stop, 'stop_random', stop_random, 'index', index)
                # print(self.num_batches, self.models_to_load, self.num_epochs, self.step, 
                # self.num_epochs_passed, self.dataset_name, self.model_name, self.method, 
                # self.num_epoch_model_loaded, self.start_adversarial_replacement)
                # print('epoch', self.epoch, 'batch', self.c)
                # print('random portion', random_portion)
                # print('slice', stop_random, stop)
                adversarial_replacement_(
                    batch=negative_batch, 
                    index=index, 
                    selection=slice(stop_random, stop), 
                    model=self.model, 
                    k=1
                )
        
        return negative_batch.view(*batch_shape, self.num_negs_per_pos, 3)
