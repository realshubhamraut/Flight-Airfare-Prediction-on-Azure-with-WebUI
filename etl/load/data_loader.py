"""
Data Loading Module
===================
Handles data output operations:
- Parquet/CSV/Delta Lake output
- Partitioning strategies
- Model serialization
- Checkpoint management
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from etl.utils.logging_utils import ETLLogger, log_execution_time


class DataLoader:
    """
    Data loader for writing processed data to various formats.
    """
    
    SUPPORTED_FORMATS = ["parquet", "csv", "json", "delta", "orc"]
    
    def __init__(
        self,
        output_path: Union[str, Path],
        format: str = "parquet",
        mode: str = "overwrite",
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize data loader.
        
        Args:
            output_path: Base output path
            format: Output format
            mode: Write mode ('overwrite', 'append', 'error', 'ignore')
            logger: Logger instance
        """
        self.output_path = Path(output_path)
        self.format = format.lower()
        self.mode = mode
        self.logger = logger or ETLLogger("DataLoader")
        
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
    
    @log_execution_time
    def save(
        self,
        df: DataFrame,
        partition_by: Optional[List[str]] = None,
        coalesce: Optional[int] = None,
        repartition: Optional[int] = None,
        compression: str = "snappy"
    ) -> Path:
        """
        Save DataFrame to output path.
        
        Args:
            df: DataFrame to save
            partition_by: Columns for partitioning
            coalesce: Number of partitions to coalesce to
            repartition: Number of partitions to repartition to
            compression: Compression codec
            
        Returns:
            Path: Output path
        """
        self.logger.step(f"Saving data to {self.output_path}")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply repartitioning
        if coalesce:
            df = df.coalesce(coalesce)
            self.logger.info(f"Coalesced to {coalesce} partitions")
        elif repartition:
            df = df.repartition(repartition)
            self.logger.info(f"Repartitioned to {repartition} partitions")
        
        # Build writer
        writer = df.write.mode(self.mode)
        
        # Apply partitioning
        if partition_by:
            writer = writer.partitionBy(*partition_by)
            self.logger.info(f"Partitioning by: {partition_by}")
        
        # Format-specific options
        if self.format == "parquet":
            writer = writer.option("compression", compression)
            writer.parquet(str(self.output_path))
            
        elif self.format == "csv":
            writer = writer.option("header", "true")
            writer.csv(str(self.output_path))
            
        elif self.format == "json":
            writer.json(str(self.output_path))
            
        elif self.format == "delta":
            writer.format("delta").save(str(self.output_path))
            
        elif self.format == "orc":
            writer = writer.option("compression", compression)
            writer.orc(str(self.output_path))
        
        # Log metrics
        row_count = df.count()
        self.logger.step_complete(
            "Data save",
            metrics={
                "output_path": str(self.output_path),
                "format": self.format,
                "rows": row_count,
                "partitioned_by": partition_by
            }
        )
        
        return self.output_path
    
    def save_with_schema(
        self,
        df: DataFrame,
        include_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Save DataFrame and its schema.
        
        Args:
            df: DataFrame to save
            include_metadata: Whether to save metadata
            
        Returns:
            Dict with paths to data and schema
        """
        paths = {}
        
        # Save data
        paths["data"] = self.save(df)
        
        # Save schema
        schema_path = self.output_path.parent / f"{self.output_path.stem}_schema.json"
        schema_json = df.schema.json()
        with open(schema_path, "w") as f:
            f.write(schema_json)
        paths["schema"] = schema_path
        
        # Save metadata
        if include_metadata:
            metadata_path = self.output_path.parent / f"{self.output_path.stem}_metadata.json"
            metadata = {
                "row_count": df.count(),
                "column_count": len(df.columns),
                "columns": df.columns,
                "created_at": datetime.now().isoformat(),
                "format": self.format
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            paths["metadata"] = metadata_path
        
        self.logger.info("Saved data with schema and metadata")
        return paths


class PartitionedLoader:
    """
    Loader with advanced partitioning strategies.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        partition_scheme: str = "date",
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize partitioned loader.
        
        Args:
            output_path: Base output path
            partition_scheme: 'date', 'airline', 'route', or 'custom'
            logger: Logger instance
        """
        self.output_path = Path(output_path)
        self.partition_scheme = partition_scheme
        self.logger = logger or ETLLogger("PartitionedLoader")
    
    def determine_partition_columns(self, df: DataFrame) -> List[str]:
        """
        Determine partition columns based on scheme.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of partition columns
        """
        schemes = {
            "date": ["Journey_Year", "Journey_Month"],
            "airline": ["Airline"],
            "route": ["Source", "Destination"],
            "full": ["Journey_Year", "Journey_Month", "Airline"]
        }
        
        columns = schemes.get(self.partition_scheme, [])
        
        # Filter to columns that exist in DataFrame
        available_columns = [c for c in columns if c in df.columns]
        
        return available_columns
    
    def save_partitioned(
        self,
        df: DataFrame,
        format: str = "parquet"
    ) -> Path:
        """
        Save with automatic partitioning.
        
        Args:
            df: DataFrame to save
            format: Output format
            
        Returns:
            Path: Output path
        """
        partition_cols = self.determine_partition_columns(df)
        
        loader = DataLoader(
            output_path=self.output_path,
            format=format,
            logger=self.logger
        )
        
        return loader.save(df, partition_by=partition_cols)


class CheckpointManager:
    """
    Manages ETL checkpoints for recovery and resumption.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            logger: Logger instance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or ETLLogger("CheckpointManager")
    
    def save_checkpoint(
        self,
        df: DataFrame,
        stage_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint for a pipeline stage.
        
        Args:
            df: DataFrame to checkpoint
            stage_name: Name of pipeline stage
            metadata: Additional metadata
            
        Returns:
            Path: Checkpoint path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_{timestamp}"
        
        # Save data
        df.write.mode("overwrite").parquet(str(checkpoint_path / "data"))
        
        # Save metadata
        meta = metadata or {}
        meta.update({
            "stage": stage_name,
            "timestamp": timestamp,
            "row_count": df.count()
        })
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        spark: SparkSession,
        stage_name: str,
        timestamp: Optional[str] = None
    ) -> Optional[DataFrame]:
        """
        Load checkpoint for a stage.
        
        Args:
            spark: Spark session
            stage_name: Stage name
            timestamp: Specific timestamp (latest if None)
            
        Returns:
            DataFrame if checkpoint exists, None otherwise
        """
        # Find matching checkpoints
        checkpoints = list(self.checkpoint_dir.glob(f"{stage_name}_*"))
        
        if not checkpoints:
            self.logger.warning(f"No checkpoint found for stage: {stage_name}")
            return None
        
        if timestamp:
            checkpoint_path = self.checkpoint_dir / f"{stage_name}_{timestamp}"
        else:
            # Get latest checkpoint
            checkpoint_path = sorted(checkpoints)[-1]
        
        if not checkpoint_path.exists():
            return None
        
        df = spark.read.parquet(str(checkpoint_path / "data"))
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return df
    
    def list_checkpoints(self, stage_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints.
        
        Args:
            stage_name: Filter by stage name
            
        Returns:
            List of checkpoint metadata
        """
        pattern = f"{stage_name}_*" if stage_name else "*"
        checkpoints = []
        
        for cp_dir in self.checkpoint_dir.glob(pattern):
            if not cp_dir.is_dir():
                continue
                
            meta_file = cp_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    meta["path"] = str(cp_dir)
                    checkpoints.append(meta)
        
        return sorted(checkpoints, key=lambda x: x.get("timestamp", ""))
    
    def cleanup_old_checkpoints(
        self,
        stage_name: str,
        keep_latest: int = 3
    ) -> int:
        """
        Remove old checkpoints, keeping only latest.
        
        Args:
            stage_name: Stage name
            keep_latest: Number of checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        import shutil
        
        checkpoints = list(self.checkpoint_dir.glob(f"{stage_name}_*"))
        checkpoints = sorted(checkpoints)
        
        to_remove = checkpoints[:-keep_latest] if len(checkpoints) > keep_latest else []
        
        for cp in to_remove:
            shutil.rmtree(cp)
            self.logger.info(f"Removed checkpoint: {cp}")
        
        return len(to_remove)


class ModelSerializer:
    """
    Serializes ML models and artifacts.
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize model serializer.
        
        Args:
            model_dir: Directory for model artifacts
            logger: Logger instance
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or ETLLogger("ModelSerializer")
    
    def save_spark_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save Spark ML model.
        
        Args:
            model: Spark ML model
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path: Model path
        """
        model_path = self.model_dir / model_name
        
        # Save model
        model.save(str(model_path))
        
        # Save metadata
        if metadata:
            meta_path = model_path / "custom_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def save_sklearn_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save scikit-learn model using joblib.
        
        Args:
            model: sklearn model
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path: Model path
        """
        import joblib
        
        model_path = self.model_dir / f"{model_name}.joblib"
        
        joblib.dump(model, model_path)
        
        if metadata:
            meta_path = self.model_dir / f"{model_name}_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"sklearn model saved: {model_path}")
        
        return model_path
    
    def load_sklearn_model(self, model_name: str) -> Any:
        """
        Load scikit-learn model.
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded model
        """
        import joblib
        
        model_path = self.model_dir / f"{model_name}.joblib"
        model = joblib.load(model_path)
        
        self.logger.info(f"Model loaded: {model_path}")
        
        return model
