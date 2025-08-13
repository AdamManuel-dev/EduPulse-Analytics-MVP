"""
Performance metrics collection and analysis for e2e tests
Tracks detailed performance characteristics and generates reports
"""

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil


@dataclass
class SystemMetrics:
    """System-level performance metrics"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int

    @classmethod
    def capture(cls) -> "SystemMetrics":
        """Capture current system metrics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        io_counters = process.io_counters() if hasattr(process, "io_counters") else None

        return cls(
            timestamp=datetime.now(),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_mb=memory_info.rss / 1024 / 1024,
            disk_io_read_mb=io_counters.read_bytes / 1024 / 1024 if io_counters else 0,
            disk_io_write_mb=io_counters.write_bytes / 1024 / 1024 if io_counters else 0,
            network_sent_mb=0,  # Would need system-wide metrics
            network_recv_mb=0,
            open_files=len(process.open_files()) if hasattr(process, "open_files") else 0,
            threads=process.num_threads(),
        )


@dataclass
class RequestMetrics:
    """HTTP request performance metrics"""

    endpoint: str
    method: str
    start_time: datetime
    end_time: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    request_size_bytes: int = 0
    response_size_bytes: int = 0

    def complete(self, status_code: int, response_size: int = 0, error: str = None):
        """Mark request as complete"""
        self.end_time = datetime.now()
        self.response_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status_code = status_code
        self.response_size_bytes = response_size
        self.error = error


@dataclass
class WorkflowMetrics:
    """Workflow-level performance metrics"""

    workflow_name: str
    student_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0
    max_response_time_ms: float = 0
    min_response_time_ms: float = float("inf")
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    system_metrics_start: Optional[SystemMetrics] = None
    system_metrics_end: Optional[SystemMetrics] = None

    def add_request(self, request: RequestMetrics):
        """Add request metrics to workflow"""
        self.total_requests += 1

        if request.status_code and 200 <= request.status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if request.error:
                self.errors.append(request.error)

        if request.response_time_ms:
            self.response_times.append(request.response_time_ms)
            self.total_response_time_ms += request.response_time_ms
            self.max_response_time_ms = max(self.max_response_time_ms, request.response_time_ms)
            self.min_response_time_ms = min(self.min_response_time_ms, request.response_time_ms)

    def complete(self):
        """Mark workflow as complete"""
        self.end_time = datetime.now()
        self.system_metrics_end = SystemMetrics.capture()

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

    @property
    def avg_response_time_ms(self) -> float:
        return self.total_response_time_ms / self.total_requests if self.total_requests > 0 else 0

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0

    @property
    def p50_response_time_ms(self) -> float:
        return np.percentile(self.response_times, 50) if self.response_times else 0

    @property
    def p95_response_time_ms(self) -> float:
        return np.percentile(self.response_times, 95) if self.response_times else 0

    @property
    def p99_response_time_ms(self) -> float:
        return np.percentile(self.response_times, 99) if self.response_times else 0


class PerformanceCollector:
    """Collects and analyzes performance metrics"""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.workflows: List[WorkflowMetrics] = []
        self.requests: List[RequestMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.current_workflow: Optional[WorkflowMetrics] = None

        # Background system monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # seconds

    def start_monitoring(self):
        """Start background system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_system(self):
        """Background thread for system monitoring"""
        while self.monitoring:
            self.system_metrics.append(SystemMetrics.capture())
            time.sleep(self.monitor_interval)

    @contextmanager
    def workflow(self, name: str, student_type: str = "unknown"):
        """Context manager for tracking workflow performance"""
        workflow = WorkflowMetrics(
            workflow_name=name,
            student_type=student_type,
            start_time=datetime.now(),
            system_metrics_start=SystemMetrics.capture(),
        )
        self.current_workflow = workflow

        try:
            yield workflow
        finally:
            workflow.complete()
            self.workflows.append(workflow)
            self.current_workflow = None

    @contextmanager
    def request(self, endpoint: str, method: str = "GET", request_size: int = 0):
        """Context manager for tracking request performance"""
        request = RequestMetrics(
            endpoint=endpoint,
            method=method,
            start_time=datetime.now(),
            request_size_bytes=request_size,
        )

        try:
            yield request
        finally:
            if request.end_time is None:
                request.complete(status_code=500, error="Request did not complete properly")

            self.requests.append(request)

            # Add to current workflow if exists
            if self.current_workflow:
                self.current_workflow.add_request(request)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(),
            "workflows": self._analyze_workflows(),
            "endpoints": self._analyze_endpoints(),
            "student_types": self._analyze_student_types(),
            "system_metrics": self._analyze_system_metrics(),
            "performance_issues": self._identify_issues(),
        }

        # Save report
        report_path = os.path.join(
            self.output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Performance report saved to: {report_path}")
        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_requests = len(self.requests)
        successful_requests = sum(
            1 for r in self.requests if r.status_code and 200 <= r.status_code < 400
        )

        all_response_times = [r.response_time_ms for r in self.requests if r.response_time_ms]

        return {
            "total_workflows": len(self.workflows),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "overall_success_rate": successful_requests / total_requests
            if total_requests > 0
            else 0,
            "avg_response_time_ms": np.mean(all_response_times) if all_response_times else 0,
            "median_response_time_ms": np.median(all_response_times) if all_response_times else 0,
            "p95_response_time_ms": np.percentile(all_response_times, 95)
            if all_response_times
            else 0,
            "p99_response_time_ms": np.percentile(all_response_times, 99)
            if all_response_times
            else 0,
            "max_response_time_ms": max(all_response_times) if all_response_times else 0,
            "min_response_time_ms": min(all_response_times) if all_response_times else 0,
        }

    def _analyze_workflows(self) -> List[Dict[str, Any]]:
        """Analyze workflow performance"""
        workflow_stats = []

        for workflow in self.workflows:
            stats = {
                "name": workflow.workflow_name,
                "student_type": workflow.student_type,
                "duration_ms": workflow.duration_ms,
                "total_requests": workflow.total_requests,
                "success_rate": workflow.success_rate,
                "avg_response_time_ms": workflow.avg_response_time_ms,
                "p50_response_time_ms": workflow.p50_response_time_ms,
                "p95_response_time_ms": workflow.p95_response_time_ms,
                "p99_response_time_ms": workflow.p99_response_time_ms,
                "max_response_time_ms": workflow.max_response_time_ms,
                "error_count": len(workflow.errors),
            }

            # Add resource utilization if available
            if workflow.system_metrics_start and workflow.system_metrics_end:
                stats["cpu_usage_change"] = (
                    workflow.system_metrics_end.cpu_percent
                    - workflow.system_metrics_start.cpu_percent
                )
                stats["memory_usage_change_mb"] = (
                    workflow.system_metrics_end.memory_mb - workflow.system_metrics_start.memory_mb
                )

            workflow_stats.append(stats)

        return workflow_stats

    def _analyze_endpoints(self) -> Dict[str, Any]:
        """Analyze performance by endpoint"""
        endpoint_metrics = defaultdict(
            lambda: {
                "count": 0,
                "success_count": 0,
                "response_times": [],
                "status_codes": defaultdict(int),
            }
        )

        for request in self.requests:
            key = f"{request.method} {request.endpoint}"
            metrics = endpoint_metrics[key]
            metrics["count"] += 1

            if request.status_code:
                metrics["status_codes"][request.status_code] += 1
                if 200 <= request.status_code < 400:
                    metrics["success_count"] += 1

            if request.response_time_ms:
                metrics["response_times"].append(request.response_time_ms)

        # Calculate statistics
        endpoint_stats = {}
        for endpoint, metrics in endpoint_metrics.items():
            response_times = metrics["response_times"]
            endpoint_stats[endpoint] = {
                "count": metrics["count"],
                "success_rate": metrics["success_count"] / metrics["count"]
                if metrics["count"] > 0
                else 0,
                "avg_response_time_ms": np.mean(response_times) if response_times else 0,
                "p95_response_time_ms": np.percentile(response_times, 95) if response_times else 0,
                "p99_response_time_ms": np.percentile(response_times, 99) if response_times else 0,
                "status_codes": dict(metrics["status_codes"]),
            }

        return endpoint_stats

    def _analyze_student_types(self) -> Dict[str, Any]:
        """Analyze performance by student type"""
        student_type_metrics = defaultdict(
            lambda: {
                "workflow_count": 0,
                "total_duration_ms": 0,
                "total_requests": 0,
                "successful_requests": 0,
                "response_times": [],
            }
        )

        for workflow in self.workflows:
            metrics = student_type_metrics[workflow.student_type]
            metrics["workflow_count"] += 1
            metrics["total_duration_ms"] += workflow.duration_ms
            metrics["total_requests"] += workflow.total_requests
            metrics["successful_requests"] += workflow.successful_requests
            metrics["response_times"].extend(workflow.response_times)

        # Calculate statistics
        student_stats = {}
        for student_type, metrics in student_type_metrics.items():
            response_times = metrics["response_times"]
            student_stats[student_type] = {
                "workflow_count": metrics["workflow_count"],
                "avg_workflow_duration_ms": metrics["total_duration_ms"] / metrics["workflow_count"]
                if metrics["workflow_count"] > 0
                else 0,
                "avg_requests_per_workflow": metrics["total_requests"] / metrics["workflow_count"]
                if metrics["workflow_count"] > 0
                else 0,
                "overall_success_rate": metrics["successful_requests"] / metrics["total_requests"]
                if metrics["total_requests"] > 0
                else 0,
                "avg_response_time_ms": np.mean(response_times) if response_times else 0,
                "p95_response_time_ms": np.percentile(response_times, 95) if response_times else 0,
            }

        return student_stats

    def _analyze_system_metrics(self) -> Dict[str, Any]:
        """Analyze system resource utilization"""
        if not self.system_metrics:
            return {}

        cpu_values = [m.cpu_percent for m in self.system_metrics]
        memory_values = [m.memory_mb for m in self.system_metrics]

        return {
            "cpu": {
                "avg_percent": np.mean(cpu_values),
                "max_percent": max(cpu_values),
                "p95_percent": np.percentile(cpu_values, 95),
            },
            "memory": {
                "avg_mb": np.mean(memory_values),
                "max_mb": max(memory_values),
                "p95_mb": np.percentile(memory_values, 95),
            },
            "samples_collected": len(self.system_metrics),
        }

    def _identify_issues(self) -> List[Dict[str, Any]]:
        """Identify performance issues"""
        issues = []

        # Check for slow endpoints
        for request in self.requests:
            if request.response_time_ms and request.response_time_ms > 2000:
                issues.append(
                    {
                        "type": "slow_request",
                        "severity": "high" if request.response_time_ms > 5000 else "medium",
                        "endpoint": f"{request.method} {request.endpoint}",
                        "response_time_ms": request.response_time_ms,
                        "timestamp": request.start_time.isoformat(),
                    }
                )

        # Check for high error rates
        for workflow in self.workflows:
            if workflow.success_rate < 0.8:
                issues.append(
                    {
                        "type": "high_error_rate",
                        "severity": "high" if workflow.success_rate < 0.5 else "medium",
                        "workflow": workflow.workflow_name,
                        "student_type": workflow.student_type,
                        "success_rate": workflow.success_rate,
                        "errors": workflow.errors[:5],  # First 5 errors
                    }
                )

        # Check for resource issues
        if self.system_metrics:
            cpu_values = [m.cpu_percent for m in self.system_metrics]
            memory_values = [m.memory_mb for m in self.system_metrics]

            if max(cpu_values) > 80:
                issues.append(
                    {
                        "type": "high_cpu_usage",
                        "severity": "high" if max(cpu_values) > 90 else "medium",
                        "max_cpu_percent": max(cpu_values),
                        "avg_cpu_percent": np.mean(cpu_values),
                    }
                )

            if max(memory_values) > 1024:  # 1GB
                issues.append(
                    {
                        "type": "high_memory_usage",
                        "severity": "high" if max(memory_values) > 2048 else "medium",
                        "max_memory_mb": max(memory_values),
                        "avg_memory_mb": np.mean(memory_values),
                    }
                )

        return issues


# Decorator for easy performance tracking
def track_performance(collector: PerformanceCollector):
    """Decorator to track function performance"""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            endpoint = f"{func.__module__}.{func.__name__}"
            with collector.request(endpoint, method="FUNCTION"):
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator


# Global collector instance
global_collector = PerformanceCollector()


def get_global_collector() -> PerformanceCollector:
    """Get the global performance collector instance"""
    return global_collector
