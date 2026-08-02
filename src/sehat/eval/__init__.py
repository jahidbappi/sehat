"""Evaluation suite for Project Sehat: metrics, fairness, calibration, reporting.

Only stdlib + numpy are imported at package import time; torch/pandas are
always lazy so this package works on bare machines.
"""

from sehat.eval.calibration import reliability_diagram_data, temperature_scaling_available
from sehat.eval.fairness import subgroup_report
from sehat.eval.metrics import (
    auroc,
    average_precision,
    brier_score,
    expected_calibration_error,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)
from sehat.eval.model_card import render_model_card, write_model_card
from sehat.eval.report import evaluate_checkpoint, evaluate_predictions

__all__ = [
    "auroc",
    "average_precision",
    "brier_score",
    "evaluate_checkpoint",
    "evaluate_predictions",
    "expected_calibration_error",
    "reliability_diagram_data",
    "render_model_card",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity",
    "subgroup_report",
    "temperature_scaling_available",
    "write_model_card",
]
