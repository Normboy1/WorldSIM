"""SIMLAB: AI-powered centralized scientific simulation platform."""

import os

# Force non-interactive Agg backend before any matplotlib import happens.
# Downstream code must not call matplotlib.use() again.
os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.1.0"
__author__ = "MaxOSL AI Research"

from simlab.core.engine.simlab_core import SimLabCore
from simlab.core.schemas.experiment import ExperimentRequest, ExperimentResult

__all__ = ["SimLabCore", "ExperimentRequest", "ExperimentResult"]
