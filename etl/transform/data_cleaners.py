"""
Data Cleaning Module
====================
Handles data quality transformations:
- Null value handling
- Data type casting
- Outlier treatment
- Normalization
- String cleaning
"""

from typing import List, Optional, Dict, Any, Union
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, FloatType, DoubleType, StringType,
    BooleanType, DateType
)
from etl.utils.logging_utils import ETLLogger, log_execution_time


class DataCleaner:
    """
    Data cleaning operations for flight booking data.
    """
    
    def __init__(self, logger: Optional[ETLLogger] = None):
        """
        Initialize data cleaner.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or ETLLogger("DataCleaner")
        
    @log_execution_time
    def clean_all(self, df: DataFrame) -> DataFrame:
        """
        Apply all cleaning transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        self.logger.step("Starting data cleaning pipeline")
        
        initial_count = df.count()
        
        df = self.clean_strings(df)
        df = self.handle_nulls(df)
        df = self.standardize_values(df)
        df = self.remove_duplicates(df)
        
        final_count = df.count()
        
        self.logger.step_complete(
            "Data cleaning",
            metrics={
                "initial_rows": initial_count,
                "final_rows": final_count,
                "removed_rows": initial_count - final_count
            }
        )
        
        return df
    
    def clean_strings(
        self,
        df: DataFrame,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Clean string columns by trimming whitespace and standardizing case.
        
        Args:
            df: Input DataFrame
            columns: Columns to clean (all string columns if None)
            
        Returns:
            DataFrame: With cleaned strings
        """
        self.logger.step("Cleaning string values")
        
        if columns is None:
            columns = [
                f.name for f in df.schema.fields
                if isinstance(f.dataType, StringType)
            ]
        
        for col in columns:
            if col in df.columns:
                df = df.withColumn(
                    col,
                    F.trim(F.col(col))
                )
                # Replace empty strings with null
                df = df.withColumn(
                    col,
                    F.when(F.col(col) == "", None).otherwise(F.col(col))
                )
        
        self.logger.info(f"Cleaned {len(columns)} string columns")
        
        return df
    
    def handle_nulls(
        self,
        df: DataFrame,
        strategy: str = "drop",
        subset: Optional[List[str]] = None,
        fill_values: Optional[Dict[str, Any]] = None
    ) -> DataFrame:
        """
        Handle null values with various strategies.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill', 'fill_mean', 'fill_median', 'fill_mode'
            subset: Columns to apply strategy to
            fill_values: Dictionary of column -> fill value
            
        Returns:
            DataFrame: With nulls handled
        """
        self.logger.step(f"Handling nulls with strategy: {strategy}")
        
        null_counts = {
            col: df.filter(F.col(col).isNull()).count()
            for col in df.columns
        }
        total_nulls = sum(null_counts.values())
        
        if strategy == "drop":
            if subset:
                df = df.dropna(subset=subset)
            else:
                df = df.dropna()
                
        elif strategy == "fill" and fill_values:
            df = df.fillna(fill_values)
            
        elif strategy == "fill_mean":
            numeric_cols = [
                f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, FloatType, DoubleType))
            ]
            if subset:
                numeric_cols = [c for c in numeric_cols if c in subset]
                
            for col in numeric_cols:
                mean_val = df.select(F.mean(col)).collect()[0][0]
                if mean_val is not None:
                    df = df.fillna({col: mean_val})
                    
        elif strategy == "fill_mode":
            if subset:
                columns = subset
            else:
                columns = df.columns
                
            for col in columns:
                mode_row = df.groupBy(col).count().orderBy(
                    F.col("count").desc()
                ).first()
                if mode_row and mode_row[0] is not None:
                    df = df.fillna({col: mode_row[0]})
        
        self.logger.info(
            "Null handling complete",
            strategy=strategy,
            total_nulls_before=total_nulls
        )
        
        return df
    
    def standardize_values(self, df: DataFrame) -> DataFrame:
        """
        Standardize values in specific columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With standardized values
        """
        self.logger.step("Standardizing values")
        
        # Standardize airline names
        if "Airline" in df.columns:
            df = df.withColumn(
                "Airline",
                F.when(F.col("Airline").like("%Air India%"), "Air India")
                 .when(F.col("Airline").like("%Jet%"), "Jet Airways")
                 .when(F.col("Airline").like("%IndiGo%"), "IndiGo")
                 .when(F.col("Airline").like("%SpiceJet%"), "SpiceJet")
                 .otherwise(F.col("Airline"))
            )
        
        # Standardize city names
        city_columns = ["Source", "Destination"]
        for col in city_columns:
            if col in df.columns:
                df = df.withColumn(
                    col,
                    F.initcap(F.trim(F.col(col)))
                )
        
        # Standardize Additional_Info
        if "Additional_Info" in df.columns:
            df = df.withColumn(
                "Additional_Info",
                F.when(
                    F.col("Additional_Info").isin(["No info", "No Info", ""]),
                    "No Info"
                ).otherwise(F.col("Additional_Info"))
            )
        
        self.logger.info("Values standardized")
        
        return df
    
    def remove_duplicates(
        self,
        df: DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: 'first' or 'last'
            
        Returns:
            DataFrame: Deduplicated DataFrame
        """
        self.logger.step("Removing duplicates")
        
        initial_count = df.count()
        
        if subset:
            df = df.dropDuplicates(subset)
        else:
            df = df.dropDuplicates()
        
        final_count = df.count()
        duplicates_removed = initial_count - final_count
        
        self.logger.info(
            "Duplicates removed",
            initial_rows=initial_count,
            final_rows=final_count,
            removed=duplicates_removed
        )
        
        return df
    
    def remove_outliers(
        self,
        df: DataFrame,
        column: str,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> DataFrame:
        """
        Remove outliers from numeric column.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame: With outliers removed
        """
        self.logger.step(f"Removing outliers from {column}")
        
        initial_count = df.count()
        
        if method == "iqr":
            quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            df = df.filter(
                (F.col(column) >= lower_bound) & (F.col(column) <= upper_bound)
            )
            
        elif method == "zscore":
            stats = df.select(
                F.mean(column).alias("mean"),
                F.stddev(column).alias("stddev")
            ).collect()[0]
            
            mean = stats["mean"]
            stddev = stats["stddev"]
            
            df = df.filter(
                F.abs((F.col(column) - mean) / stddev) <= threshold
            )
        
        final_count = df.count()
        
        self.logger.info(
            f"Outliers removed from {column}",
            method=method,
            threshold=threshold,
            removed=initial_count - final_count
        )
        
        return df
    
    def cast_columns(
        self,
        df: DataFrame,
        type_mapping: Dict[str, str]
    ) -> DataFrame:
        """
        Cast columns to specified types.
        
        Args:
            df: Input DataFrame
            type_mapping: Dict of column -> type string
            
        Returns:
            DataFrame: With casted columns
        """
        self.logger.step("Casting column types")
        
        type_map = {
            "int": IntegerType(),
            "integer": IntegerType(),
            "float": FloatType(),
            "double": DoubleType(),
            "string": StringType(),
            "boolean": BooleanType(),
            "date": DateType()
        }
        
        for col, type_str in type_mapping.items():
            if col in df.columns and type_str.lower() in type_map:
                df = df.withColumn(col, F.col(col).cast(type_map[type_str.lower()]))
        
        self.logger.info(f"Casted {len(type_mapping)} columns")
        
        return df


class NullHandler:
    """
    Specialized null value handling with advanced strategies.
    """
    
    def __init__(self, logger: Optional[ETLLogger] = None):
        """Initialize null handler."""
        self.logger = logger or ETLLogger("NullHandler")
        
    def analyze_nulls(self, df: DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze null patterns in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict with null analysis per column
        """
        total_rows = df.count()
        analysis = {}
        
        for col in df.columns:
            null_count = df.filter(F.col(col).isNull()).count()
            analysis[col] = {
                "null_count": null_count,
                "null_percentage": (null_count / total_rows) * 100 if total_rows > 0 else 0,
                "has_nulls": null_count > 0
            }
        
        return analysis
    
    def smart_fill(self, df: DataFrame) -> DataFrame:
        """
        Intelligently fill nulls based on column type and content.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With nulls filled
        """
        for field in df.schema.fields:
            if isinstance(field.dataType, (IntegerType, FloatType, DoubleType)):
                # Numeric columns: fill with median
                median = df.approxQuantile(field.name, [0.5], 0.01)[0]
                df = df.fillna({field.name: median})
            elif isinstance(field.dataType, StringType):
                # String columns: fill with mode
                mode = df.groupBy(field.name).count().orderBy(
                    F.col("count").desc()
                ).first()
                if mode and mode[0]:
                    df = df.fillna({field.name: mode[0]})
                    
        return df
