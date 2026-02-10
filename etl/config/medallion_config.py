"""
Medallion Architecture Configuration
====================================
Bronze-Silver-Gold data lakehouse pattern for flight data ETL.

Architecture Overview:
    Bronze (Raw)    → Ingest raw data as-is with metadata
    Silver (Clean)  → Cleaned, validated, typed data
    Gold (Business) → Aggregated, ML-ready datasets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum


class DataLayer(Enum):
    """Data lakehouse layers."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class LayerConfig:
    """Configuration for a single data layer."""
    name: DataLayer
    path: Path
    format: str = "parquet"
    partition_by: List[str] = field(default_factory=list)
    compression: str = "snappy"
    description: str = ""
    
    @property
    def full_path(self) -> Path:
        """Get full path for this layer."""
        return self.path / self.name.value


@dataclass
class MedallionConfig:
    """
    Complete medallion architecture configuration.
    
    Example:
        config = MedallionConfig(base_path=Path("data/lakehouse"))
        bronze_path = config.bronze.full_path  # data/lakehouse/bronze
    """
    base_path: Path = field(default_factory=lambda: Path("data/lakehouse"))
    
    bronze: LayerConfig = field(default=None)
    silver: LayerConfig = field(default=None)
    gold: LayerConfig = field(default=None)
    
    def __post_init__(self):
        """Initialize layer configs."""
        if self.bronze is None:
            self.bronze = LayerConfig(
                name=DataLayer.BRONZE,
                path=self.base_path,
                format="parquet",
                partition_by=["ingestion_date"],
                description="Raw data ingested from source with metadata"
            )
        
        if self.silver is None:
            self.silver = LayerConfig(
                name=DataLayer.SILVER,
                path=self.base_path,
                format="parquet",
                partition_by=["Journey_Year", "Journey_Month"],
                description="Cleaned, validated, and typed data"
            )
        
        if self.gold is None:
            self.gold = LayerConfig(
                name=DataLayer.GOLD,
                path=self.base_path,
                format="parquet",
                partition_by=["Airline"],
                description="Business aggregations and ML-ready features"
            )
    
    def create_directories(self) -> None:
        """Create directory structure for all layers."""
        for layer in [self.bronze, self.silver, self.gold]:
            layer.full_path.mkdir(parents=True, exist_ok=True)
    
    def get_layer_path(self, layer: DataLayer, dataset: str = "flights") -> Path:
        """Get path for a specific layer and dataset."""
        layer_config = getattr(self, layer.value)
        return layer_config.full_path / dataset
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "base_path": str(self.base_path),
            "layers": {
                "bronze": {
                    "path": str(self.bronze.full_path),
                    "format": self.bronze.format,
                    "partition_by": self.bronze.partition_by,
                    "description": self.bronze.description
                },
                "silver": {
                    "path": str(self.silver.full_path),
                    "format": self.silver.format,
                    "partition_by": self.silver.partition_by,
                    "description": self.silver.description
                },
                "gold": {
                    "path": str(self.gold.full_path),
                    "format": self.gold.format,
                    "partition_by": self.gold.partition_by,
                    "description": self.gold.description
                }
            }
        }


# Layer-specific schemas
BRONZE_SCHEMA = {
    "description": "Raw flight booking data with ingestion metadata",
    "columns": [
        "Airline", "Date_of_Journey", "Source", "Destination",
        "Route", "Dep_Time", "Arrival_Time", "Duration",
        "Total_Stops", "Additional_Info", "Price",
        # Metadata columns
        "_ingestion_timestamp", "_source_file", "_batch_id"
    ]
}

SILVER_SCHEMA = {
    "description": "Cleaned and typed flight data",
    "columns": [
        "Airline", "Source", "Destination", "Route",
        "Journey_Day", "Journey_Month", "Journey_Year",
        "Dep_Hour", "Dep_Minute", "Arrival_Hour", "Arrival_Minute",
        "Duration_Minutes", "Total_Stops_Num",
        "Additional_Info", "Price",
        # Quality columns
        "_is_valid", "_quality_score"
    ]
}

GOLD_SCHEMA = {
    "description": "ML-ready features with encodings",
    "columns": [
        # Encoded categorical features
        "Airline_Encoded", "Source_Encoded", "Destination_Encoded",
        # Numeric features
        "Journey_Day", "Journey_Month", "Dep_Hour", "Dep_Minute",
        "Arrival_Hour", "Arrival_Minute", "Duration_Minutes",
        "Total_Stops_Num",
        # Derived features
        "IsWeekend", "IsMorningFlight", "IsEveningFlight",
        "Route_Segments", "Is_Direct", "City_Pair",
        # Target
        "Price",
        # Statistical features
        "Price_mean_by_airline", "Price_mean_by_route"
    ]
}
