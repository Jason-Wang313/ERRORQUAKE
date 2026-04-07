"""Top-level package for ERRORQUAKE."""

from errorquake.analyze import BValue, FitResult, PredictionResult
from errorquake.evaluate import ALL_MODELS, EvaluationEngine, ModelConfig, ModelResponse
from errorquake.magnitude import SCALE_5, SCALE_7, SCALE_11, SeverityLevel, get_scale
from errorquake.queries import Query, load_queries, load_reserve
from errorquake.score import ScoreResult, ScoringPipeline
from errorquake.synthetic import generate_synthetic_scores, run_experiment_0
from errorquake.utils import ProjectConfig

__all__ = [
    "ALL_MODELS",
    "BValue",
    "EvaluationEngine",
    "FitResult",
    "ModelConfig",
    "ModelResponse",
    "PredictionResult",
    "ProjectConfig",
    "Query",
    "SCALE_5",
    "SCALE_7",
    "SCALE_11",
    "ScoreResult",
    "ScoringPipeline",
    "SeverityLevel",
    "generate_synthetic_scores",
    "get_scale",
    "load_queries",
    "load_reserve",
    "run_experiment_0",
]

__version__ = "0.1.0"

