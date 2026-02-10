"""
Flight Airfare ETL Package
==========================
Industry-standard PySpark-based ETL pipeline for distributed processing
of flight booking datasets with feature engineering, categorical encoding,
and data quality validation.

Modules:
    - config: Configuration management
    - extract: Data extraction layer
    - transform: Data transformation and feature engineering
    - load: Data loading and persistence
    - quality: Data quality validation
    - utils: Utility functions
    - pipelines: Pipeline orchestration

Author: Flight Airfare Prediction Team
Version: 1.0.0
"""

from etl.pipelines.flight_etl_pipeline import (
    FlightETLPipeline,
    PipelineBuilder,
    BatchPipeline,
    PipelineResult,
    PipelineStatus
)
from etl.config.etl_config import ETLConfig
from etl.utils.spark_utils import SparkSessionManager
from etl.utils.logging_utils import ETLLogger

__version__ = "1.0.0"
__all__ = [
    "FlightETLPipeline",
    "PipelineBuilder",
    "BatchPipeline",
    "PipelineResult",
    "PipelineStatus",
    "ETLConfig",
    "SparkSessionManager",
    "ETLLogger",
]
