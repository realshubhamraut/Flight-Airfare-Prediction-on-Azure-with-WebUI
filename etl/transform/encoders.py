"""
Encoding Module
===============
Implements categorical encoding strategies:
- Label Encoding (ordinal)
- One-Hot Encoding
- Target Encoding
- Frequency Encoding
- Custom mappings
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StringIndexerModel
from pyspark.ml import Pipeline, PipelineModel
from etl.utils.logging_utils import ETLLogger, log_execution_time


class CategoricalEncoder:
    """
    Categorical variable encoder with multiple strategies.
    Wraps PySpark ML transformers and custom encoding logic.
    """
    
    # Default columns to encode for flight data
    DEFAULT_CATEGORICAL_COLUMNS = [
        "Airline",
        "Source",
        "Destination",
        "Total_Stops",
        "Additional_Info"
    ]
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "label",
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize encoder.
        
        Args:
            columns: Columns to encode
            strategy: 'label', 'onehot', 'target', or 'frequency'
            logger: Logger instance
        """
        self.columns = columns or self.DEFAULT_CATEGORICAL_COLUMNS
        self.strategy = strategy
        self.logger = logger or ETLLogger("CategoricalEncoder")
        self.fitted_models: Dict[str, Any] = {}
        self.label_mappings: Dict[str, Dict[str, int]] = {}
        
    @log_execution_time
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit encoder and transform data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Encoded DataFrame
        """
        self.logger.step(f"Encoding categorical columns with strategy: {self.strategy}")
        
        if self.strategy == "label":
            df = self._label_encode(df, fit=True)
        elif self.strategy == "onehot":
            df = self._onehot_encode(df, fit=True)
        elif self.strategy == "target":
            df = self._target_encode(df, fit=True)
        elif self.strategy == "frequency":
            df = self._frequency_encode(df, fit=True)
        else:
            raise ValueError(f"Unknown encoding strategy: {self.strategy}")
        
        self.logger.step_complete(
            "Categorical encoding",
            metrics={
                "columns": len(self.columns),
                "strategy": self.strategy
            }
        )
        
        return df
    
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform data using fitted encoder.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Encoded DataFrame
        """
        if not self.fitted_models and not self.label_mappings:
            raise RuntimeError("Encoder not fitted. Call fit_transform first.")
        
        if self.strategy == "label":
            df = self._label_encode(df, fit=False)
        elif self.strategy == "onehot":
            df = self._onehot_encode(df, fit=False)
        elif self.strategy == "target":
            df = self._target_encode(df, fit=False)
        elif self.strategy == "frequency":
            df = self._frequency_encode(df, fit=False)
            
        return df
    
    def _label_encode(self, df: DataFrame, fit: bool = True) -> DataFrame:
        """
        Apply label (ordinal) encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit or use existing mappings
            
        Returns:
            DataFrame: With label encoded columns
        """
        self.logger.info("Applying label encoding")
        
        for col in self.columns:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
                
            indexed_col = f"{col}_Encoded"
            
            if fit:
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=indexed_col,
                    handleInvalid="keep"  # Keep unknown values
                )
                model = indexer.fit(df)
                self.fitted_models[col] = model
                
                # Store mapping for interpretability
                self.label_mappings[col] = {
                    label: idx for idx, label in enumerate(model.labels)
                }
            else:
                model = self.fitted_models.get(col)
                if not model:
                    self.logger.warning(f"No fitted model for {col}")
                    continue
                    
            df = model.transform(df)
            
            # Cast to integer
            df = df.withColumn(indexed_col, F.col(indexed_col).cast("integer"))
        
        return df
    
    def _onehot_encode(self, df: DataFrame, fit: bool = True) -> DataFrame:
        """
        Apply one-hot encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit or use existing model
            
        Returns:
            DataFrame: With one-hot encoded columns
        """
        self.logger.info("Applying one-hot encoding")
        
        if fit:
            stages = []
            
            for col in self.columns:
                if col not in df.columns:
                    continue
                    
                indexed_col = f"{col}_Index"
                onehot_col = f"{col}_OneHot"
                
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=indexed_col,
                    handleInvalid="keep"
                )
                
                encoder = OneHotEncoder(
                    inputCol=indexed_col,
                    outputCol=onehot_col,
                    dropLast=True  # Avoid dummy variable trap
                )
                
                stages.extend([indexer, encoder])
            
            if stages:
                pipeline = Pipeline(stages=stages)
                model = pipeline.fit(df)
                self.fitted_models["pipeline"] = model
                df = model.transform(df)
        else:
            model = self.fitted_models.get("pipeline")
            if model:
                df = model.transform(df)
        
        return df
    
    def _target_encode(
        self,
        df: DataFrame,
        fit: bool = True,
        target_col: str = "Price"
    ) -> DataFrame:
        """
        Apply target (mean) encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit or use existing mappings
            target_col: Target column for mean calculation
            
        Returns:
            DataFrame: With target encoded columns
        """
        self.logger.info("Applying target encoding")
        
        for col in self.columns:
            if col not in df.columns:
                continue
                
            encoded_col = f"{col}_TargetEnc"
            
            if fit:
                # Calculate mean target per category
                means = df.groupBy(col).agg(
                    F.mean(target_col).alias(encoded_col)
                )
                
                # Store means for transform
                self.label_mappings[col] = {
                    row[col]: row[encoded_col]
                    for row in means.collect()
                    if row[col] is not None
                }
                
                # Global mean for unknown categories
                global_mean = df.select(F.mean(target_col)).collect()[0][0]
                self.label_mappings[f"{col}_default"] = global_mean
            
            # Apply encoding
            mapping = self.label_mappings.get(col, {})
            default_val = self.label_mappings.get(f"{col}_default", 0)
            
            # Create mapping expression
            mapping_expr = F.lit(default_val)
            for cat, val in mapping.items():
                mapping_expr = F.when(F.col(col) == cat, val).otherwise(mapping_expr)
                
            df = df.withColumn(encoded_col, mapping_expr)
        
        return df
    
    def _frequency_encode(self, df: DataFrame, fit: bool = True) -> DataFrame:
        """
        Apply frequency encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit or use existing mappings
            
        Returns:
            DataFrame: With frequency encoded columns
        """
        self.logger.info("Applying frequency encoding")
        
        total_count = df.count()
        
        for col in self.columns:
            if col not in df.columns:
                continue
                
            encoded_col = f"{col}_FreqEnc"
            
            if fit:
                # Calculate frequency per category
                freqs = df.groupBy(col).count()
                freqs = freqs.withColumn(
                    "frequency",
                    F.col("count") / total_count
                )
                
                # Store frequencies
                self.label_mappings[col] = {
                    row[col]: row["frequency"]
                    for row in freqs.collect()
                    if row[col] is not None
                }
            
            # Apply encoding
            mapping = self.label_mappings.get(col, {})
            
            mapping_expr = F.lit(0.0)
            for cat, freq in mapping.items():
                mapping_expr = F.when(F.col(col) == cat, freq).otherwise(mapping_expr)
                
            df = df.withColumn(encoded_col, mapping_expr)
        
        return df
    
    def get_mappings(self) -> Dict[str, Dict[str, int]]:
        """Get label mappings for interpretability."""
        return self.label_mappings
    
    def save_mappings(self, path: Union[str, Path]) -> None:
        """
        Save label mappings to JSON.
        
        Args:
            path: Output path
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.label_mappings, f, indent=2)
        self.logger.info(f"Mappings saved to {path}")
    
    def load_mappings(self, path: Union[str, Path]) -> None:
        """
        Load label mappings from JSON.
        
        Args:
            path: Input path
        """
        path = Path(path)
        with open(path, "r") as f:
            self.label_mappings = json.load(f)
        self.logger.info(f"Mappings loaded from {path}")


class EncoderFactory:
    """
    Factory for creating encoders based on configuration.
    """
    
    @staticmethod
    def create_encoder(
        strategy: str,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> CategoricalEncoder:
        """
        Create encoder instance.
        
        Args:
            strategy: Encoding strategy
            columns: Columns to encode
            **kwargs: Additional arguments
            
        Returns:
            CategoricalEncoder instance
        """
        return CategoricalEncoder(
            columns=columns,
            strategy=strategy,
            **kwargs
        )
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available encoding strategies."""
        return ["label", "onehot", "target", "frequency"]


class MultiEncoder:
    """
    Apply different encoding strategies to different column groups.
    """
    
    def __init__(
        self,
        encoding_config: Dict[str, List[str]],
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize multi-encoder.
        
        Args:
            encoding_config: Dict of strategy -> list of columns
            logger: Logger instance
            
        Example:
            encoding_config = {
                "label": ["Airline", "Source"],
                "onehot": ["Total_Stops"]
            }
        """
        self.encoding_config = encoding_config
        self.logger = logger or ETLLogger("MultiEncoder")
        self.encoders: Dict[str, CategoricalEncoder] = {}
        
    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit and transform using all encoders.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Encoded DataFrame
        """
        self.logger.step("Applying multi-strategy encoding")
        
        for strategy, columns in self.encoding_config.items():
            encoder = CategoricalEncoder(
                columns=columns,
                strategy=strategy,
                logger=self.logger
            )
            df = encoder.fit_transform(df)
            self.encoders[strategy] = encoder
        
        self.logger.step_complete(
            "Multi-strategy encoding",
            metrics={"strategies": list(self.encoding_config.keys())}
        )
        
        return df
    
    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform using fitted encoders.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Encoded DataFrame
        """
        for strategy, encoder in self.encoders.items():
            df = encoder.transform(df)
        return df
