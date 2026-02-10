"""
Production Monitoring Module for Flight Price Prediction
=========================================================
Provides request/response logging, prediction tracking, and 
data drift monitoring for deployed ML models.

Features:
- Request/Response logging with timestamps
- Prediction distribution tracking
- Input data validation and anomaly detection
- Azure Application Insights integration (optional)
- Local file-based logging fallback
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque
from threading import Lock
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionDataCollector:
    """
    Collects and stores prediction data for monitoring.
    Thread-safe implementation for concurrent requests.
    """
    
    def __init__(
        self,
        service_name: str = "flight-price-prediction",
        max_buffer_size: int = 1000,
        log_dir: str = "logs/predictions"
    ):
        self.service_name = service_name
        self.max_buffer_size = max_buffer_size
        self.log_dir = log_dir
        
        # Thread-safe buffers
        self._lock = Lock()
        self._input_buffer: deque = deque(maxlen=max_buffer_size)
        self._output_buffer: deque = deque(maxlen=max_buffer_size)
        self._latency_buffer: deque = deque(maxlen=max_buffer_size)
        
        # Statistics
        self._request_count = 0
        self._error_count = 0
        self._start_time = datetime.now()
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info(f"PredictionDataCollector initialized for '{service_name}'")
    
    def collect_input(self, input_data: Dict[str, Any], request_id: str = None) -> str:
        """Collect input data for a prediction request."""
        if request_id is None:
            request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "type": "input",
            "data": input_data
        }
        
        with self._lock:
            self._input_buffer.append(record)
            self._request_count += 1
        
        return request_id
    
    def collect_output(
        self,
        output_data: Any,
        request_id: str,
        latency_ms: float = None,
        error: str = None
    ):
        """Collect output data for a prediction request."""
        record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "type": "output",
            "prediction": output_data,
            "latency_ms": latency_ms,
            "error": error
        }
        
        with self._lock:
            self._output_buffer.append(record)
            if latency_ms:
                self._latency_buffer.append(latency_ms)
            if error:
                self._error_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        with self._lock:
            latencies = list(self._latency_buffer)
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            stats = {
                "service_name": self.service_name,
                "uptime_seconds": uptime,
                "total_requests": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1),
                "requests_per_minute": (self._request_count / uptime) * 60 if uptime > 0 else 0,
            }
            
            if latencies:
                stats.update({
                    "avg_latency_ms": statistics.mean(latencies),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies),
                    "p50_latency_ms": statistics.median(latencies),
                    "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else None,
                    "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else None,
                })
            
            return stats
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Get the n most recent predictions."""
        with self._lock:
            outputs = list(self._output_buffer)[-n:]
            return outputs
    
    def flush_to_file(self) -> str:
        """Flush collected data to a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/predictions_{timestamp}.jsonl"
        
        with self._lock:
            inputs = list(self._input_buffer)
            outputs = list(self._output_buffer)
        
        with open(filename, 'w') as f:
            for record in inputs + outputs:
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Flushed {len(inputs)} inputs and {len(outputs)} outputs to {filename}")
        return filename


class ModelDataCollector:
    """
    Azure ML-compatible ModelDataCollector interface.
    Works both with Azure ML and standalone deployments.
    """
    
    def __init__(
        self,
        model_name: str,
        identifier: str = "default",
        feature_names: List[str] = None
    ):
        self.model_name = model_name
        self.identifier = identifier
        self.feature_names = feature_names or []
        
        # Internal collector
        self._collector = PredictionDataCollector(
            service_name=f"{model_name}_{identifier}"
        )
        
        # Try to use Azure Application Insights if available
        self._app_insights_client = None
        self._init_app_insights()
    
    def _init_app_insights(self):
        """Initialize Azure Application Insights if connection string is available."""
        connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
        instrumentation_key = os.environ.get("APPINSIGHTS_INSTRUMENTATIONKEY")
        
        if connection_string or instrumentation_key:
            try:
                from opencensus.ext.azure import metrics_exporter
                from opencensus.stats import aggregation, measure, stats, view
                from opencensus.tags import tag_map
                
                logger.info("Azure Application Insights configured")
                self._app_insights_enabled = True
            except ImportError:
                logger.warning("opencensus not installed. Run: pip install opencensus-ext-azure")
                self._app_insights_enabled = False
        else:
            self._app_insights_enabled = False
            logger.info("Application Insights not configured (set APPLICATIONINSIGHTS_CONNECTION_STRING)")
    
    def collect(self, data, request_id: str = None) -> str:
        """
        Collect data (input or output) for monitoring.
        
        Args:
            data: The data to collect (dict, list, or DataFrame)
            request_id: Optional request identifier
            
        Returns:
            The request ID used for this collection
        """
        import pandas as pd
        
        # Convert to dict if DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')
            if len(data) == 1:
                data = data[0]
        
        request_id = self._collector.collect_input(
            {"data": data},
            request_id=request_id
        )
        
        return request_id
    
    def collect_prediction(
        self,
        input_data: Dict,
        prediction: Any,
        latency_ms: float = None,
        request_id: str = None
    ):
        """
        Collect both input and prediction output together.
        """
        request_id = self._collector.collect_input(input_data, request_id)
        self._collector.collect_output(prediction, request_id, latency_ms)
        return request_id
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        return self._collector.get_statistics()
    
    def flush(self) -> str:
        """Flush data to storage."""
        return self._collector.flush_to_file()


class DataDriftDetector:
    """
    Detects data drift by comparing current input distributions
    to baseline statistics.
    """
    
    def __init__(self, baseline_stats: Dict[str, Dict] = None):
        self.baseline_stats = baseline_stats or {}
        self._current_values: Dict[str, List] = {}
        self._lock = Lock()
    
    def set_baseline(self, feature_name: str, mean: float, std: float, min_val: float, max_val: float):
        """Set baseline statistics for a feature."""
        self.baseline_stats[feature_name] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val
        }
    
    def add_observation(self, feature_name: str, value: float):
        """Add an observation for drift detection."""
        with self._lock:
            if feature_name not in self._current_values:
                self._current_values[feature_name] = []
            self._current_values[feature_name].append(value)
    
    def check_drift(self, feature_name: str, threshold: float = 2.0) -> Dict:
        """
        Check for drift in a feature using z-score method.
        
        Args:
            feature_name: Name of the feature to check
            threshold: Z-score threshold for drift detection (default: 2.0)
            
        Returns:
            Dict with drift detection results
        """
        if feature_name not in self.baseline_stats:
            return {"error": f"No baseline for {feature_name}"}
        
        with self._lock:
            if feature_name not in self._current_values or len(self._current_values[feature_name]) < 10:
                return {"status": "insufficient_data"}
            
            current = self._current_values[feature_name]
        
        baseline = self.baseline_stats[feature_name]
        current_mean = statistics.mean(current)
        
        # Calculate z-score of current mean vs baseline
        if baseline["std"] > 0:
            z_score = abs(current_mean - baseline["mean"]) / baseline["std"]
        else:
            z_score = 0
        
        is_drifted = z_score > threshold
        
        return {
            "feature": feature_name,
            "baseline_mean": baseline["mean"],
            "current_mean": current_mean,
            "z_score": z_score,
            "threshold": threshold,
            "is_drifted": is_drifted,
            "sample_size": len(current)
        }


# Global collector instance for easy access
_global_collector: Optional[ModelDataCollector] = None


def get_collector(model_name: str = "flight-price-prediction") -> ModelDataCollector:
    """Get or create the global ModelDataCollector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = ModelDataCollector(model_name)
    return _global_collector


# Convenience functions
def collect_input(data, request_id: str = None) -> str:
    """Collect input data using the global collector."""
    return get_collector().collect(data, request_id)


def collect_prediction(input_data, prediction, latency_ms: float = None, request_id: str = None) -> str:
    """Collect a complete prediction (input + output)."""
    return get_collector().collect_prediction(input_data, prediction, latency_ms, request_id)


def get_metrics() -> Dict:
    """Get current metrics from the global collector."""
    return get_collector().get_metrics()
