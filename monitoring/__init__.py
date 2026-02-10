"""
Production Monitoring Module
============================
Provides monitoring capabilities for deployed ML models.
"""

from monitoring.data_collector import (
    ModelDataCollector,
    PredictionDataCollector,
    DataDriftDetector,
    get_collector,
    collect_input,
    collect_prediction,
    get_metrics,
)

__all__ = [
    "ModelDataCollector",
    "PredictionDataCollector",
    "DataDriftDetector",
    "get_collector",
    "collect_input",
    "collect_prediction",
    "get_metrics",
]
