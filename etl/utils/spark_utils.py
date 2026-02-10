"""
Spark Session Manager
=====================
Manages Spark session lifecycle with proper configuration and resource management.
Implements singleton pattern for session reuse.
"""

from typing import Optional
from pyspark.sql import SparkSession
from etl.config.etl_config import SparkConfig, ETLConfig


class SparkSessionManager:
    """
    Manages Spark session creation and lifecycle.
    Implements singleton pattern to ensure single session per application.
    """
    
    _instance: Optional["SparkSessionManager"] = None
    _session: Optional[SparkSession] = None
    
    def __new__(cls, config: Optional[SparkConfig] = None):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[SparkConfig] = None):
        """Initialize with configuration."""
        if config is None:
            config = SparkConfig()
        self.config = config
    
    def get_or_create_session(self) -> SparkSession:
        """
        Get existing Spark session or create new one.
        
        Returns:
            SparkSession: Active Spark session
        """
        if self._session is None or self._session.sparkContext._jsc is None:
            self._session = self._create_session()
        return self._session
    
    def _create_session(self) -> SparkSession:
        """
        Create new Spark session with configuration.
        
        Returns:
            SparkSession: Newly created Spark session
        """
        builder = SparkSession.builder \
            .appName(self.config.app_name) \
            .master(self.config.master) \
            .config("spark.driver.memory", self.config.driver_memory) \
            .config("spark.executor.memory", self.config.executor_memory) \
            .config("spark.executor.cores", str(self.config.executor_cores))
        
        # Apply extra configurations
        for key, value in self.config.extra_configs.items():
            builder = builder.config(key, value)
        
        session = builder.getOrCreate()
        
        # Set log level
        session.sparkContext.setLogLevel("WARN")
        
        return session
    
    def stop_session(self):
        """Stop the current Spark session."""
        if self._session is not None:
            self._session.stop()
            self._session = None
    
    @property
    def session(self) -> SparkSession:
        """Property accessor for session."""
        return self.get_or_create_session()
    
    def __enter__(self):
        """Context manager entry."""
        return self.get_or_create_session()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop session."""
        self.stop_session()


def get_spark_session(config: Optional[ETLConfig] = None) -> SparkSession:
    """
    Convenience function to get Spark session.
    
    Args:
        config: Optional ETL configuration
        
    Returns:
        SparkSession: Active Spark session
    """
    spark_config = config.spark if config else SparkConfig()
    manager = SparkSessionManager(spark_config)
    return manager.get_or_create_session()
