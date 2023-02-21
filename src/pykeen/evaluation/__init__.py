# -*- coding: utf-8 -*-

"""Evaluation."""

from class_resolver import ClassResolver

from .classification_evaluator import ClassificationEvaluator, ClassificationMetricResults
from .evaluation_loop import LCWAEvaluationLoop
from .evaluator import Evaluator, MetricResults, evaluate
from .ensemble_evaluator import EnsembleEvaluator
from .ogb_evaluator import OGBEvaluator
from .rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedEvaluator,
    RankBasedMetricResults,
    SampledRankBasedEvaluator,
)
from .rank_based_evaluator import (
    MacroRankBasedEvaluator,
    RankBasedEvaluator,
    RankBasedMetricResults,
    SampledRankBasedEvaluator,
)
from .ensemble_rank_based_evaluator import (
    EnsembleMacroRankBasedEvaluator,
    EnsembleRankBasedEvaluator,
    EnsembleRankBasedMetricResults,
    EnsembleSampledRankBasedEvaluator,
)

__all__ = [
    "evaluate",
    "Evaluator",
    "EnsembleEvaluator",
    "MetricResults",
    "RankBasedEvaluator",
    "EnsembleRankBasedEvaluator",
    "RankBasedMetricResults",
    "EnsembleRankBasedMetricResults",
    "MacroRankBasedEvaluator",
    "EnsembleMacroRankBasedEvaluator",
    "LCWAEvaluationLoop",
    "SampledRankBasedEvaluator",
    "EnsembleSampledRankBasedEvaluator",
    "OGBEvaluator",
    "ClassificationEvaluator",
    "ClassificationMetricResults",
    "evaluator_resolver",
    "ensemble_evaluator_resolver",
    "metric_resolver",
]

evaluator_resolver: ClassResolver[Evaluator] = ClassResolver.from_subclasses(
    base=Evaluator,
    default=RankBasedEvaluator,
)

ensemble_evaluator_resolver: ClassResolver[EnsembleEvaluator] = ClassResolver.from_subclasses(
    base=EnsembleEvaluator,
    default=EnsembleRankBasedEvaluator,
)

metric_resolver: ClassResolver[MetricResults] = ClassResolver.from_subclasses(MetricResults)
