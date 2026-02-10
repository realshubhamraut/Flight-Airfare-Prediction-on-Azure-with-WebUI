"""
Data Quality Module
==================
Implements data quality validation:
- Schema validation
- Null checks
- Range validation
- Uniqueness constraints
- Business rule validation
- Data profiling
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from etl.utils.logging_utils import ETLLogger, log_execution_time


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    ERROR = "error"      # Pipeline should fail
    WARNING = "warning"  # Log but continue
    INFO = "info"        # Informational only


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    results: List[ValidationResult]
    overall_passed: bool
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warnings": self.warnings,
            "overall_passed": self.overall_passed,
            "execution_time_ms": self.execution_time_ms,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details
                }
                for r in self.results
            ]
        }


class DataValidator:
    """
    Comprehensive data quality validator for flight booking data.
    """
    
    # Expected columns for flight data
    REQUIRED_COLUMNS = [
        "Airline", "Date_of_Journey", "Source", "Destination",
        "Route", "Dep_Time", "Duration", "Total_Stops", "Price"
    ]
    
    # Valid values for categorical columns
    VALID_VALUES = {
        "Source": ["Delhi", "Kolkata", "Banglore", "Mumbai", "Chennai"],
        "Destination": ["Delhi", "Kolkata", "Banglore", "Mumbai", "Chennai", 
                       "Cochin", "Hyderabad", "New Delhi"],
        "Total_Stops": ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
    }
    
    def __init__(
        self,
        fail_on_error: bool = True,
        logger: Optional[ETLLogger] = None
    ):
        """
        Initialize validator.
        
        Args:
            fail_on_error: Whether to raise exception on ERROR severity
            logger: Logger instance
        """
        self.fail_on_error = fail_on_error
        self.logger = logger or ETLLogger("DataValidator")
        self.results: List[ValidationResult] = []
    
    @log_execution_time
    def validate_all(self, df: DataFrame) -> ValidationReport:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationReport: Complete validation report
        """
        import time
        start_time = time.time()
        
        self.logger.step("Running data quality validation")
        self.results = []
        
        # Run all validations
        self._validate_schema(df)
        self._validate_nulls(df)
        self._validate_data_types(df)
        self._validate_ranges(df)
        self._validate_categorical_values(df)
        self._validate_date_formats(df)
        self._validate_business_rules(df)
        
        # Generate report
        execution_time = (time.time() - start_time) * 1000
        report = self._generate_report(execution_time)
        
        # Log results
        self.logger.step_complete(
            "Data validation",
            metrics={
                "total_checks": report.total_checks,
                "passed": report.passed_checks,
                "failed": report.failed_checks,
                "overall_passed": report.overall_passed
            }
        )
        
        # Handle failures
        if not report.overall_passed and self.fail_on_error:
            failed_errors = [
                r for r in self.results 
                if not r.passed and r.severity == ValidationSeverity.ERROR
            ]
            error_msgs = [r.message for r in failed_errors]
            raise DataQualityError(
                f"Data validation failed: {'; '.join(error_msgs)}"
            )
        
        return report
    
    def _add_result(
        self,
        name: str,
        passed: bool,
        severity: ValidationSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add validation result."""
        result = ValidationResult(
            name=name,
            passed=passed,
            severity=severity,
            message=message,
            details=details or {}
        )
        self.results.append(result)
        
        if not passed:
            log_method = self.logger.warning if severity == ValidationSeverity.WARNING else self.logger.error
            log_method(f"Validation failed: {name} - {message}")
    
    def _validate_schema(self, df: DataFrame) -> None:
        """Validate schema has required columns."""
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]
        
        self._add_result(
            name="schema_check",
            passed=len(missing_columns) == 0,
            severity=ValidationSeverity.ERROR,
            message=f"Missing columns: {missing_columns}" if missing_columns else "Schema valid",
            details={"missing_columns": missing_columns}
        )
    
    def _validate_nulls(self, df: DataFrame) -> None:
        """Validate null percentages are acceptable."""
        total_rows = df.count()
        if total_rows == 0:
            self._add_result(
                name="null_check",
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="DataFrame is empty"
            )
            return
        
        critical_columns = ["Price", "Airline", "Source", "Destination"]
        null_thresholds = {"Price": 0.01, "Airline": 0.01, "Source": 0.01, "Destination": 0.01}
        
        for col in critical_columns:
            if col not in df.columns:
                continue
                
            null_count = df.filter(F.col(col).isNull()).count()
            null_pct = null_count / total_rows
            threshold = null_thresholds.get(col, 0.1)
            
            self._add_result(
                name=f"null_check_{col}",
                passed=null_pct <= threshold,
                severity=ValidationSeverity.ERROR if col == "Price" else ValidationSeverity.WARNING,
                message=f"{col}: {null_pct:.2%} nulls (threshold: {threshold:.2%})",
                details={"column": col, "null_count": null_count, "null_pct": null_pct}
            )
    
    def _validate_data_types(self, df: DataFrame) -> None:
        """Validate data types are correct."""
        expected_types = {
            "Price": ["int", "integer", "long", "double", "float"]
        }
        
        for col, valid_types in expected_types.items():
            if col not in df.columns:
                continue
                
            actual_type = str(df.schema[col].dataType).lower()
            is_valid = any(t in actual_type for t in valid_types)
            
            self._add_result(
                name=f"type_check_{col}",
                passed=is_valid,
                severity=ValidationSeverity.WARNING,
                message=f"{col}: type is {actual_type}",
                details={"column": col, "actual_type": actual_type}
            )
    
    def _validate_ranges(self, df: DataFrame) -> None:
        """Validate numeric values are in expected ranges."""
        if "Price" not in df.columns:
            return
            
        # Price range validation
        stats = df.select(
            F.min("Price").alias("min_price"),
            F.max("Price").alias("max_price"),
            F.mean("Price").alias("avg_price")
        ).collect()[0]
        
        min_price = stats["min_price"]
        max_price = stats["max_price"]
        
        # Price should be positive and reasonable
        price_valid = min_price is not None and min_price > 0 and max_price < 1000000
        
        self._add_result(
            name="price_range",
            passed=price_valid,
            severity=ValidationSeverity.WARNING,
            message=f"Price range: {min_price} - {max_price}",
            details={
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": stats["avg_price"]
            }
        )
    
    def _validate_categorical_values(self, df: DataFrame) -> None:
        """Validate categorical columns have expected values."""
        for col, valid_values in self.VALID_VALUES.items():
            if col not in df.columns:
                continue
            
            # Get distinct values
            distinct_values = [
                row[col] for row in df.select(col).distinct().collect()
                if row[col] is not None
            ]
            
            invalid_values = [v for v in distinct_values if v not in valid_values]
            
            self._add_result(
                name=f"categorical_check_{col}",
                passed=len(invalid_values) == 0,
                severity=ValidationSeverity.WARNING,
                message=f"{col}: {len(invalid_values)} invalid values" if invalid_values else f"{col}: all values valid",
                details={
                    "column": col,
                    "invalid_values": invalid_values[:10],  # Limit to 10
                    "valid_values": valid_values
                }
            )
    
    def _validate_date_formats(self, df: DataFrame) -> None:
        """Validate date columns have correct format."""
        if "Date_of_Journey" not in df.columns:
            return
        
        # Try to parse date
        parsed_df = df.withColumn(
            "parsed_date",
            F.to_date(F.col("Date_of_Journey"), "d/M/yyyy")
        )
        
        invalid_dates = parsed_df.filter(
            F.col("parsed_date").isNull() & F.col("Date_of_Journey").isNotNull()
        ).count()
        
        total_dates = df.filter(F.col("Date_of_Journey").isNotNull()).count()
        
        self._add_result(
            name="date_format_check",
            passed=invalid_dates == 0,
            severity=ValidationSeverity.WARNING,
            message=f"Invalid dates: {invalid_dates}/{total_dates}",
            details={"invalid_count": invalid_dates, "total_count": total_dates}
        )
    
    def _validate_business_rules(self, df: DataFrame) -> None:
        """Validate business logic rules."""
        # Rule: Source and Destination should be different
        if "Source" in df.columns and "Destination" in df.columns:
            same_city = df.filter(
                F.col("Source") == F.col("Destination")
            ).count()
            
            self._add_result(
                name="source_dest_different",
                passed=same_city == 0,
                severity=ValidationSeverity.ERROR,
                message=f"Rows with same source/destination: {same_city}",
                details={"same_city_count": same_city}
            )
        
        # Rule: Non-stop flights should have 0 stops
        if "Total_Stops" in df.columns and "Route" in df.columns:
            non_stop_with_stops = df.filter(
                (F.col("Total_Stops") == "non-stop") & 
                (F.size(F.split(F.col("Route"), " → ")) > 2)
            ).count()
            
            self._add_result(
                name="nonstop_consistency",
                passed=non_stop_with_stops < 10,  # Allow some tolerance
                severity=ValidationSeverity.WARNING,
                message=f"Non-stop flights with multiple route segments: {non_stop_with_stops}",
                details={"inconsistent_count": non_stop_with_stops}
            )
    
    def _generate_report(self, execution_time: float) -> ValidationReport:
        """Generate validation report from results."""
        passed = [r for r in self.results if r.passed]
        failed_errors = [
            r for r in self.results
            if not r.passed and r.severity == ValidationSeverity.ERROR
        ]
        warnings = [
            r for r in self.results
            if not r.passed and r.severity == ValidationSeverity.WARNING
        ]
        
        return ValidationReport(
            total_checks=len(self.results),
            passed_checks=len(passed),
            failed_checks=len(failed_errors),
            warnings=len(warnings),
            results=self.results,
            overall_passed=len(failed_errors) == 0,
            execution_time_ms=execution_time
        )


class DataProfiler:
    """
    Profiles data for quality insights.
    """
    
    def __init__(self, logger: Optional[ETLLogger] = None):
        """Initialize profiler."""
        self.logger = logger or ETLLogger("DataProfiler")
    
    def profile(self, df: DataFrame) -> Dict[str, Any]:
        """
        Generate data profile.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dict with profile statistics
        """
        self.logger.step("Profiling data")
        
        profile = {
            "row_count": df.count(),
            "column_count": len(df.columns),
            "columns": {}
        }
        
        for field in df.schema.fields:
            col_name = field.name
            col_type = str(field.dataType)
            
            col_profile = {
                "data_type": col_type,
                "nullable": field.nullable
            }
            
            # Null analysis
            null_count = df.filter(F.col(col_name).isNull()).count()
            col_profile["null_count"] = null_count
            col_profile["null_percentage"] = (null_count / profile["row_count"]) * 100
            
            # Distinct values
            distinct_count = df.select(col_name).distinct().count()
            col_profile["distinct_count"] = distinct_count
            
            # Numeric statistics
            if "int" in col_type.lower() or "double" in col_type.lower() or "float" in col_type.lower():
                stats = df.select(
                    F.min(col_name).alias("min"),
                    F.max(col_name).alias("max"),
                    F.mean(col_name).alias("mean"),
                    F.stddev(col_name).alias("stddev")
                ).collect()[0]
                
                col_profile.update({
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "stddev": stats["stddev"]
                })
            
            # String statistics
            elif "string" in col_type.lower():
                # Sample top values
                top_values = df.groupBy(col_name).count().orderBy(
                    F.col("count").desc()
                ).limit(5).collect()
                
                col_profile["top_values"] = [
                    {"value": row[col_name], "count": row["count"]}
                    for row in top_values
                ]
            
            profile["columns"][col_name] = col_profile
        
        self.logger.step_complete("Data profiling", metrics={"columns_profiled": len(df.columns)})
        
        return profile


class DataQualityError(Exception):
    """Exception raised for data quality failures."""
    pass


class QualityCheckBuilder:
    """
    Builder pattern for creating custom quality checks.
    """
    
    def __init__(self, df: DataFrame):
        """
        Initialize builder.
        
        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.checks: List[Tuple[str, Callable, ValidationSeverity]] = []
    
    def add_null_check(
        self,
        column: str,
        max_null_pct: float = 0.1,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "QualityCheckBuilder":
        """Add null percentage check."""
        def check():
            null_pct = self.df.filter(F.col(column).isNull()).count() / self.df.count()
            return null_pct <= max_null_pct, f"Null pct: {null_pct:.2%}"
        
        self.checks.append((f"null_{column}", check, severity))
        return self
    
    def add_range_check(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "QualityCheckBuilder":
        """Add range check."""
        def check():
            stats = self.df.select(F.min(column), F.max(column)).collect()[0]
            actual_min, actual_max = stats[0], stats[1]
            passed = True
            if min_val is not None and actual_min < min_val:
                passed = False
            if max_val is not None and actual_max > max_val:
                passed = False
            return passed, f"Range: [{actual_min}, {actual_max}]"
        
        self.checks.append((f"range_{column}", check, severity))
        return self
    
    def add_custom_check(
        self,
        name: str,
        check_func: Callable[[DataFrame], Tuple[bool, str]],
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ) -> "QualityCheckBuilder":
        """Add custom check function."""
        def check():
            return check_func(self.df)
        
        self.checks.append((name, check, severity))
        return self
    
    def run(self) -> List[ValidationResult]:
        """Run all checks and return results."""
        results = []
        for name, check_func, severity in self.checks:
            try:
                passed, message = check_func()
                results.append(ValidationResult(
                    name=name,
                    passed=passed,
                    severity=severity,
                    message=message
                ))
            except Exception as e:
                results.append(ValidationResult(
                    name=name,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Check failed with error: {str(e)}"
                ))
        return results
