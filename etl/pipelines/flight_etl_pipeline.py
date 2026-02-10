"""
Flight ETL Pipeline
==================
Orchestrates the complete ETL workflow:
1. Extract - Read data from source
2. Validate - Quality checks
3. Transform - Feature engineering & encoding  
4. Load - Save processed data
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pyspark.sql import DataFrame

from etl.config.etl_config import ETLConfig
from etl.utils.spark_utils import SparkSessionManager
from etl.utils.logging_utils import ETLLogger, log_execution_time
from etl.extract.data_extractor import DataExtractor
from etl.transform.feature_engineering import FeatureEngineer
from etl.transform.data_cleaners import DataCleaner
from etl.transform.encoders import CategoricalEncoder
from etl.load.data_loader import DataLoader, CheckpointManager
from etl.quality.data_validators import DataValidator, ValidationReport


class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    stages_completed: List[str] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    output_path: Optional[Path] = None
    row_count: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate pipeline duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "stages_completed": self.stages_completed,
            "validation_passed": self.validation_report.overall_passed if self.validation_report else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "row_count": self.row_count,
            "error_message": self.error_message,
            "metrics": self.metrics
        }


class FlightETLPipeline:
    """
    Main ETL pipeline for flight booking data.
    
    Orchestrates:
    - Data extraction from CSV/Parquet
    - Data quality validation
    - Feature engineering (temporal, duration, route)
    - Categorical encoding
    - Data loading to Parquet
    
    Example:
        pipeline = FlightETLPipeline(
            source_path="data/raw/flights.csv",
            output_path="data/processed/flights.parquet"
        )
        result = pipeline.run()
    """
    
    def __init__(
        self,
        source_path: str,
        output_path: str,
        config: Optional[ETLConfig] = None,
        enable_checkpointing: bool = True,
        checkpoint_dir: Optional[str] = None,
        skip_validation: bool = False,
        encoding_strategy: str = "label",
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize ETL pipeline.
        
        Args:
            source_path: Path to source data
            output_path: Path for output data
            config: ETL configuration
            enable_checkpointing: Whether to checkpoint stages
            checkpoint_dir: Directory for checkpoints
            skip_validation: Skip data validation
            encoding_strategy: Categorical encoding strategy
            logger: Logger instance
        """
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.config = config or ETLConfig()
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.skip_validation = skip_validation
        self.encoding_strategy = encoding_strategy
        self.logger = logger or ETLLogger("FlightETLPipeline")
        
        # Initialize components
        self.spark_manager = SparkSessionManager(self.config.spark)
        self.extractor = DataExtractor(self.source_path, config=self.config, logger=self.logger)
        self.cleaner = DataCleaner(logger=self.logger)
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.encoder = CategoricalEncoder(strategy=self.encoding_strategy, logger=self.logger)
        self.validator = DataValidator(fail_on_error=True, logger=self.logger)
        self.loader = DataLoader(self.output_path, format="parquet", logger=self.logger)
        
        if self.enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(self.checkpoint_dir, logger=self.logger)
        
        self._current_df: Optional[DataFrame] = None
        self._result: Optional[PipelineResult] = None
    
    @log_execution_time
    def run(self, resume_from: Optional[str] = None) -> PipelineResult:
        """
        Execute the complete ETL pipeline.
        
        Args:
            resume_from: Stage to resume from (if checkpointed)
            
        Returns:
            PipelineResult: Execution result
        """
        self._result = PipelineResult(
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Flight ETL Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Source: {self.source_path}")
        self.logger.info(f"Output: {self.output_path}")
        
        try:
            # Get Spark session
            spark = self.spark_manager.get_or_create_session()
            
            # Resume from checkpoint if specified
            if resume_from and self.enable_checkpointing:
                self._current_df = self.checkpoint_manager.load_checkpoint(spark, resume_from)
                if self._current_df:
                    self.logger.info(f"Resumed from checkpoint: {resume_from}")
            
            # Execute stages
            stages = [
                ("extract", self._extract),
                ("validate", self._validate),
                ("clean", self._clean),
                ("engineer_features", self._engineer_features),
                ("encode", self._encode),
                ("load", self._load)
            ]
            
            # Skip stages before resume point
            if resume_from:
                stage_names = [s[0] for s in stages]
                if resume_from in stage_names:
                    start_idx = stage_names.index(resume_from)
                    stages = stages[start_idx:]
            
            for stage_name, stage_func in stages:
                if stage_name == "validate" and self.skip_validation:
                    self.logger.info("Skipping validation (disabled)")
                    continue
                    
                self.logger.step(f"Executing stage: {stage_name}")
                stage_func(spark)
                self._result.stages_completed.append(stage_name)
                
                # Checkpoint after each stage
                if self.enable_checkpointing and self._current_df:
                    self.checkpoint_manager.save_checkpoint(
                        self._current_df,
                        stage_name,
                        {"rows": self._current_df.count()}
                    )
            
            # Finalize result
            self._result.status = PipelineStatus.COMPLETED
            self._result.end_time = datetime.now()
            self._result.output_path = self.output_path
            
            self.logger.info("=" * 60)
            self.logger.info(f"Pipeline completed successfully in {self._result.duration_seconds:.2f}s")
            self.logger.info(f"Output: {self.output_path}")
            self.logger.info(f"Rows processed: {self._result.row_count}")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self._result.status = PipelineStatus.FAILED
            self._result.end_time = datetime.now()
            self._result.error_message = str(e)
            self.logger.error(f"Pipeline failed: {e}")
            raise
            
        finally:
            # Cleanup can be handled here if needed
            pass
        
        return self._result
    
    def _extract(self, spark) -> None:
        """Extract data from source."""
        self._current_df = self.extractor.extract(spark)
        self._result.metrics["extracted_rows"] = self._current_df.count()
    
    def _validate(self, spark) -> None:
        """Validate data quality."""
        if self._current_df is None:
            raise RuntimeError("No data to validate. Run extract first.")
            
        report = self.validator.validate_all(self._current_df)
        self._result.validation_report = report
        self._result.metrics["validation_checks"] = report.total_checks
        self._result.metrics["validation_passed"] = report.overall_passed
    
    def _clean(self, spark) -> None:
        """Clean data."""
        if self._current_df is None:
            raise RuntimeError("No data to clean.")
            
        initial_count = self._current_df.count()
        self._current_df = self.cleaner.clean_all(self._current_df)
        final_count = self._current_df.count()
        
        self._result.metrics["rows_after_cleaning"] = final_count
        self._result.metrics["rows_removed_cleaning"] = initial_count - final_count
    
    def _engineer_features(self, spark) -> None:
        """Engineer features."""
        if self._current_df is None:
            raise RuntimeError("No data for feature engineering.")
            
        self._current_df = self.feature_engineer.engineer_all_features(self._current_df)
        self._result.metrics["features_engineered"] = len(self._current_df.columns)
    
    def _encode(self, spark) -> None:
        """Encode categorical variables."""
        if self._current_df is None:
            raise RuntimeError("No data to encode.")
            
        self._current_df = self.encoder.fit_transform(self._current_df)
        self._result.metrics["encoding_strategy"] = self.encoding_strategy
    
    def _load(self, spark) -> None:
        """Load data to output."""
        if self._current_df is None:
            raise RuntimeError("No data to load.")
        
        self._result.row_count = self._current_df.count()
        self.loader.save(self._current_df, coalesce=1)
    
    def get_current_dataframe(self) -> Optional[DataFrame]:
        """Get current DataFrame state."""
        return self._current_df


class PipelineBuilder:
    """
    Builder pattern for creating customized pipelines.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._source_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._config: Optional[ETLConfig] = None
        self._enable_checkpointing: bool = True
        self._skip_validation: bool = False
        self._encoding_strategy: str = "label"
        self._custom_stages: List[tuple] = []
    
    def source(self, path: str) -> "PipelineBuilder":
        """Set source path."""
        self._source_path = path
        return self
    
    def output(self, path: str) -> "PipelineBuilder":
        """Set output path."""
        self._output_path = path
        return self
    
    def config(self, config: ETLConfig) -> "PipelineBuilder":
        """Set ETL configuration."""
        self._config = config
        return self
    
    def with_checkpointing(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable/disable checkpointing."""
        self._enable_checkpointing = enabled
        return self
    
    def with_validation(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable/disable validation."""
        self._skip_validation = not enabled
        return self
    
    def with_encoding(self, strategy: str) -> "PipelineBuilder":
        """Set encoding strategy."""
        self._encoding_strategy = strategy
        return self
    
    def build(self) -> FlightETLPipeline:
        """Build the pipeline."""
        if not self._source_path:
            raise ValueError("Source path is required")
        if not self._output_path:
            raise ValueError("Output path is required")
            
        return FlightETLPipeline(
            source_path=self._source_path,
            output_path=self._output_path,
            config=self._config,
            enable_checkpointing=self._enable_checkpointing,
            skip_validation=self._skip_validation,
            encoding_strategy=self._encoding_strategy
        )


class BatchPipeline:
    """
    Run ETL pipeline on multiple files.
    """
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        file_pattern: str = "*.csv",
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize batch pipeline.
        
        Args:
            source_dir: Directory containing source files
            output_dir: Directory for output files
            file_pattern: Glob pattern for source files
            logger: Logger instance
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.file_pattern = file_pattern
        self.logger = logger or ETLLogger("BatchPipeline")
        self.results: List[PipelineResult] = []
    
    def run(self) -> List[PipelineResult]:
        """
        Run pipeline on all matching files.
        
        Returns:
            List of pipeline results
        """
        source_files = list(self.source_dir.glob(self.file_pattern))
        
        self.logger.info(f"Found {len(source_files)} files to process")
        
        for source_file in source_files:
            output_file = self.output_dir / f"{source_file.stem}_processed.parquet"
            
            self.logger.info(f"Processing: {source_file.name}")
            
            try:
                pipeline = FlightETLPipeline(
                    source_path=str(source_file),
                    output_path=str(output_file),
                    logger=self.logger
                )
                result = pipeline.run()
                self.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {source_file.name}: {e}")
                self.results.append(PipelineResult(
                    status=PipelineStatus.FAILED,
                    start_time=datetime.now(),
                    error_message=str(e)
                ))
        
        # Summary
        successful = sum(1 for r in self.results if r.status == PipelineStatus.COMPLETED)
        self.logger.info(f"Batch complete: {successful}/{len(source_files)} successful")
        
        return self.results
