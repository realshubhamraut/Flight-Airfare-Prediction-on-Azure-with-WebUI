"""
Feature Engineering Module
==========================
Implements feature extraction workflows for:
- Temporal patterns (journey dates, departure/arrival hours)
- Route analytics
- Duration parsing
- Derived metrics
"""

from typing import List, Optional, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType, StringType
from etl.utils.logging_utils import ETLLogger, log_execution_time


class FeatureEngineer:
    """
    Feature engineering for flight booking data.
    Extracts temporal patterns, route features, and derived metrics.
    """
    
    def __init__(self, logger: Optional[ETLLogger] = None):
        """
        Initialize feature engineer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or ETLLogger("FeatureEngineer")
        
    @log_execution_time
    def engineer_all_features(self, df: DataFrame) -> DataFrame:
        """
        Apply all feature engineering transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Transformed DataFrame with all features
        """
        self.logger.step("Starting feature engineering pipeline")
        
        # Apply transformations sequentially
        df = self.extract_temporal_features(df)
        df = self.extract_duration_features(df)
        df = self.extract_route_features(df)
        df = self.extract_stops_features(df)
        
        self.logger.step_complete(
            "Feature engineering",
            metrics={"total_features": len(df.columns)}
        )
        
        return df
    
    def extract_temporal_features(self, df: DataFrame) -> DataFrame:
        """
        Extract temporal features from date and time columns.
        
        Features extracted:
        - Journey_Day, Journey_Month, Journey_Year
        - Journey_DayOfWeek, Journey_WeekOfYear
        - Dep_Hour, Dep_Minute
        - Arrival_Hour, Arrival_Minute
        - IsWeekend, IsMorningFlight, IsEveningFlight
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With temporal features
        """
        self.logger.step("Extracting temporal features")
        
        # Parse Date_of_Journey (format: DD/MM/YYYY)
        df = df.withColumn(
            "Journey_Date_Parsed",
            F.to_date(F.col("Date_of_Journey"), "d/M/yyyy")
        )
        
        # Extract date components
        df = df.withColumn("Journey_Day", F.dayofmonth("Journey_Date_Parsed")) \
               .withColumn("Journey_Month", F.month("Journey_Date_Parsed")) \
               .withColumn("Journey_Year", F.year("Journey_Date_Parsed")) \
               .withColumn("Journey_DayOfWeek", F.dayofweek("Journey_Date_Parsed")) \
               .withColumn("Journey_WeekOfYear", F.weekofyear("Journey_Date_Parsed"))
        
        # Parse Dep_Time (format: HH:MM)
        df = df.withColumn(
            "Dep_Hour",
            F.split(F.col("Dep_Time"), ":").getItem(0).cast(IntegerType())
        ).withColumn(
            "Dep_Minute",
            F.split(F.col("Dep_Time"), ":").getItem(1).cast(IntegerType())
        )
        
        # Parse Arrival_Time (format: HH:MM or HH:MM DD Mon)
        df = df.withColumn(
            "Arrival_Time_Clean",
            F.split(F.col("Arrival_Time"), " ").getItem(0)
        ).withColumn(
            "Arrival_Hour",
            F.split(F.col("Arrival_Time_Clean"), ":").getItem(0).cast(IntegerType())
        ).withColumn(
            "Arrival_Minute",
            F.split(F.col("Arrival_Time_Clean"), ":").getItem(1).cast(IntegerType())
        )
        
        # Derived temporal features
        df = df.withColumn(
            "IsWeekend",
            F.when(F.col("Journey_DayOfWeek").isin([1, 7]), 1).otherwise(0)
        ).withColumn(
            "IsMorningFlight",
            F.when((F.col("Dep_Hour") >= 5) & (F.col("Dep_Hour") < 12), 1).otherwise(0)
        ).withColumn(
            "IsAfternoonFlight",
            F.when((F.col("Dep_Hour") >= 12) & (F.col("Dep_Hour") < 17), 1).otherwise(0)
        ).withColumn(
            "IsEveningFlight",
            F.when((F.col("Dep_Hour") >= 17) & (F.col("Dep_Hour") < 21), 1).otherwise(0)
        ).withColumn(
            "IsNightFlight",
            F.when((F.col("Dep_Hour") >= 21) | (F.col("Dep_Hour") < 5), 1).otherwise(0)
        )
        
        # Drop intermediate columns
        df = df.drop("Journey_Date_Parsed", "Arrival_Time_Clean")
        
        self.logger.info(
            "Temporal features extracted",
            features=[
                "Journey_Day", "Journey_Month", "Journey_Year",
                "Journey_DayOfWeek", "Journey_WeekOfYear",
                "Dep_Hour", "Dep_Minute", "Arrival_Hour", "Arrival_Minute",
                "IsWeekend", "IsMorningFlight", "IsAfternoonFlight",
                "IsEveningFlight", "IsNightFlight"
            ]
        )
        
        return df
    
    def extract_duration_features(self, df: DataFrame) -> DataFrame:
        """
        Parse duration string to total minutes.
        
        Handles formats like:
        - "2h 50m"
        - "1h"
        - "50m"
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With Duration_Minutes feature
        """
        self.logger.step("Extracting duration features")
        
        # Extract hours
        df = df.withColumn(
            "Duration_Hours",
            F.when(
                F.col("Duration").contains("h"),
                F.regexp_extract(F.col("Duration"), r"(\d+)h", 1).cast(IntegerType())
            ).otherwise(0)
        )
        
        # Extract minutes
        df = df.withColumn(
            "Duration_Mins",
            F.when(
                F.col("Duration").contains("m"),
                F.regexp_extract(F.col("Duration"), r"(\d+)m", 1).cast(IntegerType())
            ).otherwise(0)
        )
        
        # Calculate total minutes
        df = df.withColumn(
            "Duration_Minutes",
            F.col("Duration_Hours") * 60 + F.col("Duration_Mins")
        )
        
        # Drop intermediate columns
        df = df.drop("Duration_Hours", "Duration_Mins")
        
        self.logger.info("Duration features extracted", features=["Duration_Minutes"])
        
        return df
    
    def extract_route_features(self, df: DataFrame) -> DataFrame:
        """
        Extract route-based features.
        
        Features:
        - Number of stops/connections
        - Route complexity
        - City pairs
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With route features
        """
        self.logger.step("Extracting route features")
        
        # Number of cities in route (stops + origin + destination)
        df = df.withColumn(
            "Route_Segments",
            F.when(
                F.col("Route").isNotNull(),
                F.size(F.split(F.col("Route"), " → "))
            ).otherwise(2)
        )
        
        # Route length category
        df = df.withColumn(
            "Route_Complexity",
            F.when(F.col("Route_Segments") <= 2, "Direct")
             .when(F.col("Route_Segments") <= 3, "OneStop")
             .otherwise("MultiStop")
        )
        
        # Create city pair identifier
        df = df.withColumn(
            "City_Pair",
            F.concat_ws("_", F.col("Source"), F.col("Destination"))
        )
        
        self.logger.info(
            "Route features extracted",
            features=["Route_Segments", "Route_Complexity", "City_Pair"]
        )
        
        return df
    
    def extract_stops_features(self, df: DataFrame) -> DataFrame:
        """
        Convert Total_Stops to numeric.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: With numeric stops feature
        """
        self.logger.step("Extracting stops features")
        
        # Map string stops to numeric
        df = df.withColumn(
            "Total_Stops_Num",
            F.when(F.col("Total_Stops") == "non-stop", 0)
             .when(F.col("Total_Stops") == "1 stop", 1)
             .when(F.col("Total_Stops") == "2 stops", 2)
             .when(F.col("Total_Stops") == "3 stops", 3)
             .when(F.col("Total_Stops") == "4 stops", 4)
             .otherwise(F.regexp_extract(F.col("Total_Stops"), r"(\d+)", 1).cast(IntegerType()))
        )
        
        # Is direct flight
        df = df.withColumn(
            "Is_Direct",
            F.when(F.col("Total_Stops_Num") == 0, 1).otherwise(0)
        )
        
        self.logger.info(
            "Stops features extracted",
            features=["Total_Stops_Num", "Is_Direct"]
        )
        
        return df
    
    def add_statistical_features(
        self,
        df: DataFrame,
        group_columns: List[str],
        agg_column: str
    ) -> DataFrame:
        """
        Add statistical aggregation features.
        
        Args:
            df: Input DataFrame
            group_columns: Columns to group by
            agg_column: Column to aggregate
            
        Returns:
            DataFrame: With statistical features
        """
        self.logger.step(f"Adding statistical features for {agg_column}")
        
        # Calculate statistics per group
        stats_df = df.groupBy(group_columns).agg(
            F.mean(agg_column).alias(f"{agg_column}_mean"),
            F.stddev(agg_column).alias(f"{agg_column}_stddev"),
            F.min(agg_column).alias(f"{agg_column}_min"),
            F.max(agg_column).alias(f"{agg_column}_max"),
            F.count("*").alias(f"{agg_column}_count")
        )
        
        # Join back to original DataFrame
        df = df.join(stats_df, on=group_columns, how="left")
        
        # Add price deviation from mean
        if agg_column == "Price":
            df = df.withColumn(
                "Price_Deviation",
                (F.col("Price") - F.col("Price_mean")) / F.col("Price_stddev")
            )
        
        self.logger.info(
            f"Statistical features added for {agg_column}",
            group_by=group_columns
        )
        
        return df


class FeaturePipeline:
    """
    Orchestrates multiple feature engineering steps.
    """
    
    def __init__(self, steps: Optional[List[str]] = None):
        """
        Initialize feature pipeline.
        
        Args:
            steps: List of feature engineering steps to apply
        """
        self.steps = steps or [
            "temporal",
            "duration",
            "route",
            "stops"
        ]
        self.engineer = FeatureEngineer()
        self.logger = ETLLogger("FeaturePipeline")
        
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply feature pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Transformed DataFrame
        """
        self.logger.step(f"Running feature pipeline with steps: {self.steps}")
        
        step_methods = {
            "temporal": self.engineer.extract_temporal_features,
            "duration": self.engineer.extract_duration_features,
            "route": self.engineer.extract_route_features,
            "stops": self.engineer.extract_stops_features
        }
        
        for step in self.steps:
            if step in step_methods:
                df = step_methods[step](df)
            else:
                self.logger.warning(f"Unknown step: {step}")
        
        self.logger.step_complete("Feature pipeline", metrics={"steps": len(self.steps)})
        
        return df
