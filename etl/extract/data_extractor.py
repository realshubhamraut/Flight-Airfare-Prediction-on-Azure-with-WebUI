"""
Data Extraction Layer
=====================
Handles reading data from various sources (CSV, Parquet, Delta, databases).
Implements schema validation and source abstraction.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, DateType, TimestampType
)
from etl.config.etl_config import ETLConfig, DataSchemaConfig
from etl.utils.logging_utils import ETLLogger


class BaseExtractor(ABC):
    """Abstract base class for data extractors."""
    
    @abstractmethod
    def extract(self, spark: SparkSession) -> DataFrame:
        """Extract data from source."""
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """Validate that source exists and is accessible."""
        pass


class DataExtractor(BaseExtractor):
    """
    Main data extractor for flight booking data.
    Supports CSV, Parquet, and partitioned data sources.
    """
    
    # Schema definition for flight data
    FLIGHT_SCHEMA = StructType([
        StructField("Airline", StringType(), True),
        StructField("Date_of_Journey", StringType(), True),
        StructField("Source", StringType(), True),
        StructField("Destination", StringType(), True),
        StructField("Route", StringType(), True),
        StructField("Dep_Time", StringType(), True),
        StructField("Arrival_Time", StringType(), True),
        StructField("Duration", StringType(), True),
        StructField("Total_Stops", StringType(), True),
        StructField("Additional_Info", StringType(), True),
        StructField("Price", IntegerType(), True)
    ])
    
    def __init__(
        self,
        source_path: Union[str, Path],
        config: Optional[ETLConfig] = None,
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize data extractor.
        
        Args:
            source_path: Path to source data
            config: ETL configuration
            logger: Logger instance
        """
        self.source_path = Path(source_path)
        self.config = config or ETLConfig()
        self.logger = logger or ETLLogger("Extractor")
        self._source_format = self._detect_format()
        
    def _detect_format(self) -> str:
        """Detect source data format from file extension."""
        suffix = self.source_path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".json": "json",
            ".orc": "orc"
        }
        return format_map.get(suffix, "csv")
    
    def validate_source(self) -> bool:
        """
        Validate source exists and is accessible.
        
        Returns:
            bool: True if source is valid
            
        Raises:
            FileNotFoundError: If source doesn't exist
        """
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source not found: {self.source_path}")
        
        self.logger.info(f"Source validated: {self.source_path}")
        return True
    
    def extract(
        self,
        spark: SparkSession,
        use_schema: bool = True,
        sample_fraction: Optional[float] = None
    ) -> DataFrame:
        """
        Extract data from source.
        
        Args:
            spark: Spark session
            use_schema: Whether to enforce schema
            sample_fraction: Optional fraction for sampling
            
        Returns:
            DataFrame: Extracted data
        """
        self.logger.step("Extracting data from source")
        self.validate_source()
        
        # Build reader
        reader = spark.read
        
        if self._source_format == "csv":
            reader = reader.option("header", "true") \
                          .option("inferSchema", "false" if use_schema else "true") \
                          .option("mode", "PERMISSIVE") \
                          .option("nullValue", "NA") \
                          .option("emptyValue", "")
            
            if use_schema:
                reader = reader.schema(self.FLIGHT_SCHEMA)
                
            df = reader.csv(str(self.source_path))
            
        elif self._source_format == "parquet":
            df = reader.parquet(str(self.source_path))
            
        elif self._source_format == "json":
            df = reader.json(str(self.source_path))
            
        else:
            raise ValueError(f"Unsupported format: {self._source_format}")
        
        # Apply sampling if specified
        if sample_fraction and 0 < sample_fraction < 1:
            df = df.sample(fraction=sample_fraction, seed=42)
            self.logger.info(f"Applied {sample_fraction*100}% sampling")
        
        # Log extraction metrics
        row_count = df.count()
        col_count = len(df.columns)
        self.logger.step_complete(
            "Data extraction",
            metrics={
                "source": str(self.source_path),
                "format": self._source_format,
                "rows": row_count,
                "columns": col_count
            }
        )
        
        return df
    
    def extract_with_partitioning(
        self,
        spark: SparkSession,
        partition_columns: List[str]
    ) -> DataFrame:
        """
        Extract partitioned data.
        
        Args:
            spark: Spark session
            partition_columns: Columns used for partitioning
            
        Returns:
            DataFrame: Partitioned data
        """
        self.logger.step("Extracting partitioned data")
        
        df = spark.read.parquet(str(self.source_path))
        
        self.logger.info(
            f"Loaded partitioned data",
            partition_columns=partition_columns
        )
        
        return df
    
    def get_schema(self) -> StructType:
        """Get the expected schema."""
        return self.FLIGHT_SCHEMA
    
    def preview(self, spark: SparkSession, rows: int = 5) -> None:
        """
        Preview source data.
        
        Args:
            spark: Spark session
            rows: Number of rows to preview
        """
        df = self.extract(spark)
        df.show(rows, truncate=False)
        df.printSchema()


class MultiSourceExtractor:
    """
    Extractor for multiple data sources.
    Handles merging and deduplication.
    """
    
    def __init__(
        self,
        sources: List[Union[str, Path]],
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize multi-source extractor.
        
        Args:
            sources: List of source paths
            logger: Logger instance
        """
        self.sources = [Path(s) for s in sources]
        self.logger = logger or ETLLogger("MultiExtractor")
        self.extractors = [DataExtractor(s, logger=self.logger) for s in self.sources]
    
    def extract_all(
        self,
        spark: SparkSession,
        deduplicate: bool = True,
        dedup_columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Extract and merge data from all sources.
        
        Args:
            spark: Spark session
            deduplicate: Whether to remove duplicates
            dedup_columns: Columns for deduplication
            
        Returns:
            DataFrame: Merged data
        """
        self.logger.step("Extracting from multiple sources")
        
        dfs = []
        for extractor in self.extractors:
            try:
                df = extractor.extract(spark)
                dfs.append(df)
            except Exception as e:
                self.logger.warning(
                    f"Failed to extract from {extractor.source_path}: {e}"
                )
        
        if not dfs:
            raise ValueError("No data extracted from any source")
        
        # Union all DataFrames
        result = dfs[0]
        for df in dfs[1:]:
            result = result.unionByName(df, allowMissingColumns=True)
        
        # Deduplicate if requested
        if deduplicate:
            if dedup_columns:
                result = result.dropDuplicates(dedup_columns)
            else:
                result = result.dropDuplicates()
            self.logger.info("Removed duplicates from merged data")
        
        self.logger.step_complete(
            "Multi-source extraction",
            metrics={
                "sources": len(self.sources),
                "total_rows": result.count()
            }
        )
        
        return result


class IncrementalExtractor(BaseExtractor):
    """
    Extractor for incremental data loads.
    Uses watermark columns for change data capture.
    """
    
    def __init__(
        self,
        source_path: Union[str, Path],
        watermark_column: str,
        last_watermark: Any,
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize incremental extractor.
        
        Args:
            source_path: Path to source data
            watermark_column: Column for tracking changes
            last_watermark: Last processed watermark value
            logger: Logger instance
        """
        self.source_path = Path(source_path)
        self.watermark_column = watermark_column
        self.last_watermark = last_watermark
        self.logger = logger or ETLLogger("IncrementalExtractor")
        
    def validate_source(self) -> bool:
        """Validate source exists."""
        return self.source_path.exists()
    
    def extract(self, spark: SparkSession) -> DataFrame:
        """
        Extract only new/changed records.
        
        Args:
            spark: Spark session
            
        Returns:
            DataFrame: Incremental data
        """
        self.logger.step("Extracting incremental data")
        
        # Read full data
        df = spark.read.csv(str(self.source_path), header=True, inferSchema=True)
        
        # Filter for new records
        if self.last_watermark:
            df = df.filter(df[self.watermark_column] > self.last_watermark)
            
        row_count = df.count()
        self.logger.step_complete(
            "Incremental extraction",
            metrics={
                "new_records": row_count,
                "watermark_column": self.watermark_column
            }
        )
        
        return df
