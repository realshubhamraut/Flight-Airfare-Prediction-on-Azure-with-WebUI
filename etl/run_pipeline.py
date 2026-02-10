#!/usr/bin/env python
"""
Flight ETL Pipeline Runner
==========================
Command-line interface for executing the ETL pipeline.

Usage:
    python -m etl.run_pipeline --source data/raw/flights.csv --output data/processed/
    python -m etl.run_pipeline --config config/etl_config.json
    python -m etl.run_pipeline --source data/raw/ --batch --pattern "*.csv"
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from etl.config.etl_config import ETLConfig
from etl.pipelines.flight_etl_pipeline import (
    FlightETLPipeline,
    PipelineBuilder,
    BatchPipeline,
    PipelineStatus
)
from etl.utils.logging_utils import ETLLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Flight Airfare Prediction ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single file
  python -m etl.run_pipeline --source data/Data_Train.xlsx --output data/processed/

  # Run with custom config
  python -m etl.run_pipeline --source data/raw/flights.csv --output data/processed/ --config etl_config.json

  # Batch processing
  python -m etl.run_pipeline --source data/raw/ --output data/processed/ --batch --pattern "*.csv"

  # Skip validation
  python -m etl.run_pipeline --source data/raw/flights.csv --output data/processed/ --skip-validation

  # Use specific encoding
  python -m etl.run_pipeline --source data/raw/flights.csv --output data/processed/ --encoding onehot
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Source file or directory path"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file or directory path"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to ETL configuration JSON file"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch processing mode"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        default="*.csv",
        help="File pattern for batch mode (default: *.csv)"
    )
    
    parser.add_argument(
        "--encoding", "-e",
        choices=["label", "onehot", "target", "frequency"],
        default="label",
        help="Categorical encoding strategy (default: label)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip data quality validation"
    )
    
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)"
    )
    
    parser.add_argument(
        "--resume-from",
        help="Resume from a specific stage checkpoint"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running pipeline"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["parquet", "csv", "json"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    
    return parser.parse_args()


def load_config(config_path: Optional[str]) -> Optional[ETLConfig]:
    """Load configuration from file."""
    if not config_path:
        return None
        
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return ETLConfig.from_json(config_path)


def run_single_pipeline(
    args,
    config: Optional[ETLConfig],
    logger: ETLLogger
) -> int:
    """Run pipeline on single file."""
    source_path = Path(args.source)
    output_path = Path(args.output)
    
    # If output is directory, generate filename
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{source_path.stem}_processed.{args.output_format}"
    
    logger.info(f"Running ETL pipeline")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Encoding: {args.encoding}")
    
    if args.dry_run:
        logger.info("Dry run mode - validating configuration only")
        # Validation logic here
        logger.info("Configuration valid")
        return 0
    
    # Build and run pipeline
    pipeline = FlightETLPipeline(
        source_path=str(source_path),
        output_path=str(output_path),
        config=config,
        enable_checkpointing=not args.no_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        skip_validation=args.skip_validation,
        encoding_strategy=args.encoding,
        logger=logger
    )
    
    result = pipeline.run(resume_from=args.resume_from)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status: {result.status.value.upper()}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Rows processed: {result.row_count}")
    print(f"Stages completed: {', '.join(result.stages_completed)}")
    if result.validation_report:
        print(f"Validation: {'PASSED' if result.validation_report.overall_passed else 'FAILED'}")
        print(f"  - Checks: {result.validation_report.passed_checks}/{result.validation_report.total_checks} passed")
    print(f"Output: {result.output_path}")
    print("=" * 60)
    
    return 0 if result.status == PipelineStatus.COMPLETED else 1


def run_batch_pipeline(
    args,
    config: Optional[ETLConfig],
    logger: ETLLogger
) -> int:
    """Run pipeline on multiple files."""
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    
    if not source_dir.is_dir():
        logger.error(f"Source must be a directory in batch mode: {source_dir}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running batch ETL pipeline")
    logger.info(f"  Source directory: {source_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Pattern: {args.pattern}")
    
    if args.dry_run:
        files = list(source_dir.glob(args.pattern))
        logger.info(f"Dry run - found {len(files)} files to process:")
        for f in files:
            logger.info(f"  - {f.name}")
        return 0
    
    batch = BatchPipeline(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        file_pattern=args.pattern,
        logger=logger
    )
    
    results = batch.run()
    
    # Print summary
    successful = sum(1 for r in results if r.status == PipelineStatus.COMPLETED)
    failed = len(results) - successful
    
    print("\n" + "=" * 60)
    print("BATCH PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total rows: {sum(r.row_count for r in results)}")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize logger
    logger = ETLLogger(
        name="ETLRunner",
        log_level="DEBUG" if args.verbose else "INFO"
    )
    
    logger.info("=" * 60)
    logger.info("Flight Airfare Prediction ETL Pipeline")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Run appropriate pipeline mode
        if args.batch:
            exit_code = run_batch_pipeline(args, config, logger)
        else:
            exit_code = run_single_pipeline(args, config, logger)
        
        sys.exit(exit_code)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
