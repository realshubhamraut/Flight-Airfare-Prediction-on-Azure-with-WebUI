"""
Medallion Pipeline
==================
Bronze-Silver-Gold ETL pipeline following industry-standard
data lakehouse architecture.

Pipeline Flow:
    Source CSV → Bronze (Raw + Metadata)
              → Silver (Cleaned + Typed)
              → Gold (ML-Ready Features)
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import uuid
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, StringType

from etl.config.medallion_config import MedallionConfig, DataLayer
from etl.utils.spark_utils import SparkSessionManager
from etl.utils.logging_utils import ETLLogger, log_execution_time
from etl.extract.data_extractor import DataExtractor
from etl.transform.feature_engineering import FeatureEngineer
from etl.transform.data_cleaners import DataCleaner
from etl.transform.encoders import CategoricalEncoder
from etl.quality.data_validators import DataValidator


class MedallionPipeline:
    """
    Production-grade medallion architecture ETL pipeline.
    
    Implements Bronze-Silver-Gold pattern:
    - Bronze: Raw data with ingestion metadata
    - Silver: Cleaned, validated, properly typed data
    - Gold: Feature-engineered, ML-ready data
    
    Example:
        pipeline = MedallionPipeline(
            source_path="data/train.csv",
            lakehouse_path="data/lakehouse"
        )
        pipeline.run_full_pipeline()
    """
    
    def __init__(
        self,
        source_path: str,
        lakehouse_path: str = "data/lakehouse",
        config: Optional[MedallionConfig] = None,
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize medallion pipeline.
        
        Args:
            source_path: Path to raw source data
            lakehouse_path: Base path for lakehouse layers
            config: Medallion configuration
            logger: Logger instance
        """
        self.source_path = Path(source_path)
        self.config = config or MedallionConfig(base_path=Path(lakehouse_path))
        self.logger = logger or ETLLogger("MedallionPipeline")
        
        # Initialize Spark
        self.spark_manager = SparkSessionManager()
        self.spark = self.spark_manager.get_or_create_session()
        
        # Create directory structure
        self.config.create_directories()
        
        # Pipeline state
        self.batch_id = str(uuid.uuid4())[:8]
        self.metrics: Dict[str, Any] = {}
    
    @log_execution_time
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete Bronze → Silver → Gold pipeline.
        
        Returns:
            Dict with pipeline metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("MEDALLION PIPELINE - Bronze → Silver → Gold")
        self.logger.info("=" * 70)
        self.logger.info(f"Batch ID: {self.batch_id}")
        self.logger.info(f"Source: {self.source_path}")
        self.logger.info(f"Lakehouse: {self.config.base_path}")
        
        start_time = datetime.now()
        
        try:
            # Bronze Layer
            bronze_df = self.ingest_to_bronze()
            
            # Silver Layer
            silver_df = self.transform_to_silver(bronze_df)
            
            # Gold Layer
            gold_df = self.transform_to_gold(silver_df)
            
            # Finalize
            end_time = datetime.now()
            self.metrics["total_duration_seconds"] = (end_time - start_time).total_seconds()
            self.metrics["status"] = "SUCCESS"
            
            self.logger.info("=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info(f"Duration: {self.metrics['total_duration_seconds']:.2f}s")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.metrics["status"] = "FAILED"
            self.metrics["error"] = str(e)
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        return self.metrics
    
    # =========================================================================
    # BRONZE LAYER - Raw Data Ingestion
    # =========================================================================
    
    @log_execution_time
    def ingest_to_bronze(self) -> DataFrame:
        """
        Ingest raw data to Bronze layer with metadata.
        
        Bronze layer contains:
        - Original data as-is (no transformations)
        - Ingestion metadata (_ingestion_timestamp, _source_file, _batch_id)
        - Partitioned by ingestion date
        
        Returns:
            DataFrame: Bronze layer data
        """
        self.logger.step("BRONZE LAYER - Raw Data Ingestion")
        
        # Extract raw data
        extractor = DataExtractor(self.source_path, logger=self.logger)
        df = extractor.extract(self.spark, use_schema=False)
        
        # Add ingestion metadata
        df = df.withColumn("_ingestion_timestamp", F.current_timestamp()) \
               .withColumn("_source_file", F.lit(str(self.source_path))) \
               .withColumn("_batch_id", F.lit(self.batch_id)) \
               .withColumn("_ingestion_date", F.current_date())
        
        # Save to Bronze
        bronze_path = self.config.get_layer_path(DataLayer.BRONZE, "flights")
        df.write.mode("append") \
          .partitionBy("_ingestion_date") \
          .parquet(str(bronze_path))
        
        row_count = df.count()
        self.metrics["bronze_rows"] = row_count
        
        self.logger.step_complete(
            "Bronze ingestion",
            metrics={"rows": row_count, "path": str(bronze_path)}
        )
        
        return df
    
    # =========================================================================
    # SILVER LAYER - Cleaned & Validated Data
    # =========================================================================
    
    @log_execution_time
    def transform_to_silver(self, bronze_df: Optional[DataFrame] = None) -> DataFrame:
        """
        Transform Bronze to Silver layer.
        
        Silver layer contains:
        - Cleaned data (nulls handled, strings trimmed)
        - Properly typed columns
        - Validated records (quality flags)
        - Parsed date/time fields
        - Partitioned by journey year/month
        
        Args:
            bronze_df: Bronze DataFrame (reads from storage if None)
            
        Returns:
            DataFrame: Silver layer data
        """
        self.logger.step("SILVER LAYER - Data Cleaning & Validation")
        
        # Read from Bronze if not provided
        if bronze_df is None:
            bronze_path = self.config.get_layer_path(DataLayer.BRONZE, "flights")
            bronze_df = self.spark.read.parquet(str(bronze_path))
        
        # Remove metadata columns for transformation
        df = bronze_df.drop("_ingestion_timestamp", "_source_file", 
                           "_batch_id", "_ingestion_date")
        
        # Clean data
        cleaner = DataCleaner(logger=self.logger)
        df = cleaner.clean_strings(df)
        df = cleaner.handle_nulls(df, strategy="drop")
        df = cleaner.standardize_values(df)
        
        # Parse temporal fields
        df = self._parse_temporal_fields(df)
        
        # Validate and add quality flags
        df = self._add_quality_flags(df)
        
        # Save to Silver
        silver_path = self.config.get_layer_path(DataLayer.SILVER, "flights")
        df.write.mode("overwrite") \
          .partitionBy("Journey_Year", "Journey_Month") \
          .parquet(str(silver_path))
        
        row_count = df.count()
        self.metrics["silver_rows"] = row_count
        
        self.logger.step_complete(
            "Silver transformation",
            metrics={"rows": row_count, "path": str(silver_path)}
        )
        
        return df
    
    def _parse_temporal_fields(self, df: DataFrame) -> DataFrame:
        """Parse date/time strings into structured fields."""
        
        # Parse Date_of_Journey
        df = df.withColumn(
            "Journey_Date_Parsed",
            F.to_date(F.col("Date_of_Journey"), "d/M/yyyy")
        )
        df = df.withColumn("Journey_Day", F.dayofmonth("Journey_Date_Parsed")) \
               .withColumn("Journey_Month", F.month("Journey_Date_Parsed")) \
               .withColumn("Journey_Year", F.year("Journey_Date_Parsed"))
        
        # Parse Dep_Time
        df = df.withColumn(
            "Dep_Hour", 
            F.split(F.col("Dep_Time"), ":").getItem(0).cast("int")
        ).withColumn(
            "Dep_Minute",
            F.split(F.col("Dep_Time"), ":").getItem(1).cast("int")
        )
        
        # Parse Arrival_Time
        df = df.withColumn(
            "Arrival_Time_Clean",
            F.split(F.col("Arrival_Time"), " ").getItem(0)
        ).withColumn(
            "Arrival_Hour",
            F.split(F.col("Arrival_Time_Clean"), ":").getItem(0).cast("int")
        ).withColumn(
            "Arrival_Minute",
            F.split(F.col("Arrival_Time_Clean"), ":").getItem(1).cast("int")
        )
        
        # Parse Duration to minutes
        df = df.withColumn(
            "Duration_Hours",
            F.when(
                F.col("Duration").contains("h"),
                F.regexp_extract(F.col("Duration"), r"(\d+)h", 1).cast("int")
            ).otherwise(0)
        ).withColumn(
            "Duration_Mins",
            F.when(
                F.col("Duration").contains("m"),
                F.regexp_extract(F.col("Duration"), r"(\d+)m", 1).cast("int")
            ).otherwise(0)
        ).withColumn(
            "Duration_Minutes",
            F.col("Duration_Hours") * 60 + F.col("Duration_Mins")
        )
        
        # Parse Total_Stops
        df = df.withColumn(
            "Total_Stops_Num",
            F.when(F.col("Total_Stops") == "non-stop", 0)
             .otherwise(F.regexp_extract(F.col("Total_Stops"), r"(\d+)", 1).cast("int"))
        )
        
        # Drop intermediate columns
        df = df.drop(
            "Journey_Date_Parsed", "Arrival_Time_Clean",
            "Duration_Hours", "Duration_Mins"
        )
        
        return df
    
    def _add_quality_flags(self, df: DataFrame) -> DataFrame:
        """Add data quality flags."""
        
        # Validity check
        df = df.withColumn(
            "_is_valid",
            F.when(
                (F.col("Price").isNotNull()) &
                (F.col("Price") > 0) &
                (F.col("Airline").isNotNull()) &
                (F.col("Source") != F.col("Destination")),
                True
            ).otherwise(False)
        )
        
        # Quality score (0-100)
        df = df.withColumn(
            "_quality_score",
            F.lit(100) -
            F.when(F.col("Route").isNull(), 10).otherwise(0) -
            F.when(F.col("Additional_Info") == "No Info", 5).otherwise(0) -
            F.when(F.col("Duration_Minutes") == 0, 15).otherwise(0)
        )
        
        return df
    
    # =========================================================================
    # GOLD LAYER - ML-Ready Features
    # =========================================================================
    
    @log_execution_time
    def transform_to_gold(self, silver_df: Optional[DataFrame] = None) -> DataFrame:
        """
        Transform Silver to Gold layer.
        
        Gold layer contains:
        - Feature-engineered data
        - Encoded categorical variables
        - Statistical aggregations
        - ML-ready format
        
        Args:
            silver_df: Silver DataFrame (reads from storage if None)
            
        Returns:
            DataFrame: Gold layer data
        """
        self.logger.step("GOLD LAYER - Feature Engineering & Encoding")
        
        # Read from Silver if not provided
        if silver_df is None:
            silver_path = self.config.get_layer_path(DataLayer.SILVER, "flights")
            silver_df = self.spark.read.parquet(str(silver_path))
        
        # Filter to valid records only
        df = silver_df.filter(F.col("_is_valid") == True)
        
        # Feature engineering
        engineer = FeatureEngineer(logger=self.logger)
        df = engineer.extract_route_features(df)
        df = self._add_time_based_features(df)
        
        # Add statistical features
        df = self._add_statistical_features(df)
        
        # Encode categorical variables
        encoder = CategoricalEncoder(
            columns=["Airline", "Source", "Destination"],
            strategy="label",
            logger=self.logger
        )
        df = encoder.fit_transform(df)
        
        # Select final columns for Gold
        df = self._select_gold_columns(df)
        
        # Save to Gold
        gold_path = self.config.get_layer_path(DataLayer.GOLD, "flights_ml_ready")
        df.write.mode("overwrite") \
          .parquet(str(gold_path))
        
        row_count = df.count()
        self.metrics["gold_rows"] = row_count
        self.metrics["gold_columns"] = len(df.columns)
        
        self.logger.step_complete(
            "Gold transformation",
            metrics={"rows": row_count, "columns": len(df.columns), "path": str(gold_path)}
        )
        
        return df
    
    def _add_time_based_features(self, df: DataFrame) -> DataFrame:
        """Add time-based derived features."""
        
        df = df.withColumn(
            "IsWeekend",
            F.when(F.dayofweek(F.to_date(F.col("Date_of_Journey"), "d/M/yyyy")).isin([1, 7]), 1).otherwise(0)
        ).withColumn(
            "IsMorningFlight",
            F.when((F.col("Dep_Hour") >= 5) & (F.col("Dep_Hour") < 12), 1).otherwise(0)
        ).withColumn(
            "IsEveningFlight",
            F.when((F.col("Dep_Hour") >= 17) & (F.col("Dep_Hour") < 21), 1).otherwise(0)
        ).withColumn(
            "Is_Direct",
            F.when(F.col("Total_Stops_Num") == 0, 1).otherwise(0)
        )
        
        return df
    
    def _add_statistical_features(self, df: DataFrame) -> DataFrame:
        """Add statistical aggregation features."""
        
        # Average price by airline
        airline_stats = df.groupBy("Airline").agg(
            F.mean("Price").alias("Price_mean_by_airline")
        )
        df = df.join(airline_stats, on="Airline", how="left")
        
        # Average price by route
        route_stats = df.groupBy("Source", "Destination").agg(
            F.mean("Price").alias("Price_mean_by_route")
        )
        df = df.join(route_stats, on=["Source", "Destination"], how="left")
        
        return df
    
    def _select_gold_columns(self, df: DataFrame) -> DataFrame:
        """Select and order final Gold layer columns."""
        
        gold_columns = [
            # Encoded features
            "Airline_Encoded", "Source_Encoded", "Destination_Encoded",
            # Numeric features
            "Journey_Day", "Journey_Month", "Journey_Year",
            "Dep_Hour", "Dep_Minute", "Arrival_Hour", "Arrival_Minute",
            "Duration_Minutes", "Total_Stops_Num",
            # Derived features
            "IsWeekend", "IsMorningFlight", "IsEveningFlight", "Is_Direct",
            "Route_Segments",
            # Statistical features
            "Price_mean_by_airline", "Price_mean_by_route",
            # Target
            "Price"
        ]
        
        # Select only columns that exist
        available_cols = [c for c in gold_columns if c in df.columns]
        return df.select(available_cols)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def read_layer(self, layer: DataLayer, dataset: str = "flights") -> DataFrame:
        """Read data from a specific layer."""
        path = self.config.get_layer_path(layer, dataset)
        return self.spark.read.parquet(str(path))
    
    def get_layer_stats(self, layer: DataLayer) -> Dict[str, Any]:
        """Get statistics for a layer."""
        df = self.read_layer(layer)
        return {
            "layer": layer.value,
            "row_count": df.count(),
            "column_count": len(df.columns),
            "columns": df.columns
        }
