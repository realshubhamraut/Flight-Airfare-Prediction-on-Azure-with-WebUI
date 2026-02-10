"""
ETL Configuration Management
============================
Centralized configuration for ETL pipeline parameters, paths, and Spark settings.
Uses dataclasses for type safety and YAML/environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class SparkConfig:
    """Spark session configuration."""
    app_name: str = "FlightAirfareETL"
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2
    shuffle_partitions: int = 200
    adaptive_enabled: bool = True
    
    # Additional Spark configs
    extra_configs: Dict[str, str] = field(default_factory=lambda: {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.parquet.compression.codec": "snappy",
        "spark.sql.shuffle.partitions": "200"
    })


@dataclass
class PathConfig:
    """File path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    # Input paths
    raw_data_dir: Path = field(init=False)
    train_data_path: Path = field(init=False)
    test_data_path: Path = field(init=False)
    
    # Output paths
    processed_data_dir: Path = field(init=False)
    staging_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.raw_data_dir = self.base_dir / "data"
        self.train_data_path = self.raw_data_dir / "train.csv"
        self.test_data_path = self.raw_data_dir / "test.csv"
        
        self.processed_data_dir = self.base_dir / "data" / "processed"
        self.staging_dir = self.base_dir / "data" / "staging"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
    def ensure_directories(self):
        """Create all necessary directories."""
        for path in [
            self.processed_data_dir, 
            self.staging_dir, 
            self.models_dir,
            self.logs_dir,
            self.checkpoints_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataSchemaConfig:
    """Data schema configuration for validation."""
    required_columns: List[str] = field(default_factory=lambda: [
        "Airline", "Date_of_Journey", "Source", "Destination",
        "Route", "Dep_Time", "Arrival_Time", "Duration",
        "Total_Stops", "Additional_Info", "Price"
    ])
    
    categorical_columns: List[str] = field(default_factory=lambda: [
        "Airline", "Source", "Destination", "Additional_Info"
    ])
    
    numerical_columns: List[str] = field(default_factory=lambda: [
        "Price", "Total_Stops"
    ])
    
    datetime_columns: List[str] = field(default_factory=lambda: [
        "Date_of_Journey"
    ])
    
    target_column: str = "Price"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Temporal features to extract
    extract_journey_day: bool = True
    extract_journey_month: bool = True
    extract_journey_year: bool = True
    extract_dep_hour: bool = True
    extract_dep_minute: bool = True
    extract_arrival_hour: bool = True
    extract_arrival_minute: bool = True
    extract_duration_hour: bool = True
    extract_duration_minute: bool = True
    
    # Route features
    max_route_segments: int = 5
    
    # Output feature columns (order matters for model)
    output_columns: List[str] = field(default_factory=lambda: [
        "Airline", "Source", "Destination", "Additional_Info",
        "Date", "Month", "Total_Stops",
        "Arrival_hour", "Arrival_min",
        "Duration_hour", "Duration_min"
    ])


@dataclass 
class QualityConfig:
    """Data quality thresholds and rules."""
    max_null_percentage: float = 5.0
    min_row_count: int = 1000
    price_min: float = 0
    price_max: float = 100000
    valid_airlines: Optional[List[str]] = None  # None means any
    valid_sources: Optional[List[str]] = None
    valid_destinations: Optional[List[str]] = None


@dataclass
class ETLConfig:
    """Master ETL configuration."""
    # Component configs
    spark: SparkConfig = field(default_factory=SparkConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    schema: DataSchemaConfig = field(default_factory=DataSchemaConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Pipeline settings
    pipeline_name: str = "flight_airfare_etl"
    environment: str = "development"  # development, staging, production
    debug_mode: bool = True
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000000  # rows
    
    # Processing settings
    batch_size: int = 100000
    coalesce_partitions: int = 1  # Final output partitions
    overwrite_output: bool = True
    
    @classmethod
    def from_json(cls, json_path: str) -> "ETLConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls._from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> "ETLConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("ETL_ENVIRONMENT"):
            config.environment = os.getenv("ETL_ENVIRONMENT")
        if os.getenv("ETL_DEBUG"):
            config.debug_mode = os.getenv("ETL_DEBUG").lower() == "true"
        if os.getenv("SPARK_DRIVER_MEMORY"):
            config.spark.driver_memory = os.getenv("SPARK_DRIVER_MEMORY")
        if os.getenv("SPARK_EXECUTOR_MEMORY"):
            config.spark.executor_memory = os.getenv("SPARK_EXECUTOR_MEMORY")
            
        return config
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "ETLConfig":
        """Create config from dictionary."""
        spark_config = SparkConfig(**config_dict.get("spark", {}))
        paths_config = PathConfig()
        schema_config = DataSchemaConfig(**config_dict.get("schema", {}))
        features_config = FeatureConfig(**config_dict.get("features", {}))
        quality_config = QualityConfig(**config_dict.get("quality", {}))
        
        return cls(
            spark=spark_config,
            paths=paths_config,
            schema=schema_config,
            features=features_config,
            quality=quality_config,
            pipeline_name=config_dict.get("pipeline_name", "flight_airfare_etl"),
            environment=config_dict.get("environment", "development"),
            debug_mode=config_dict.get("debug_mode", True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "spark": {
                "app_name": self.spark.app_name,
                "driver_memory": self.spark.driver_memory,
                "executor_memory": self.spark.executor_memory
            },
            "schema": {
                "required_columns": self.schema.required_columns,
                "target_column": self.schema.target_column
            },
            "features": {
                "output_columns": self.features.output_columns
            }
        }
    
    def validate(self) -> bool:
        """Validate configuration."""
        # Check paths exist
        if not self.paths.train_data_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.paths.train_data_path}")
        
        # Ensure output directories
        self.paths.ensure_directories()
        
        return True


# Default configuration instance
default_config = ETLConfig()
