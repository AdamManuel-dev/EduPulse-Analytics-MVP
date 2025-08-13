"""
End-to-end tests for complete student workflows
Simulates realistic student behavior patterns with performance metrics
"""

import pytest
import time
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_generation.generators import StudentSimulator, StudentProfile, StudentType  # noqa: E402


@dataclass
class PerformanceMetrics:
    """Track performance metrics for e2e tests"""

    workflow_name: str
    start_time: datetime
    end_time: datetime
    response_times: List[float]
    error_count: int
    success_count: int
    student_type: StudentType

    @property
    def total_duration(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def avg_response_time(self) -> float:
        return np.mean(self.response_times) if self.response_times else 0

    @property
    def p95_response_time(self) -> float:
        return np.percentile(self.response_times, 95) if self.response_times else 0

    @property
    def p99_response_time(self) -> float:
        return np.percentile(self.response_times, 99) if self.response_times else 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "student_type": self.student_type.value,
            "total_duration_seconds": self.total_duration,
            "avg_response_time_ms": self.avg_response_time * 1000,
            "p95_response_time_ms": self.p95_response_time * 1000,
            "p99_response_time_ms": self.p99_response_time * 1000,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
        }


class StudentWorkflowSimulator:
    """Simulates complete student workflows with realistic behavior"""

    def __init__(self, client, student_profile: StudentProfile):
        self.client = client
        self.profile = student_profile
        self.metrics = []
        self.session_data = {}

    async def _perform_registration(self, metrics: PerformanceMetrics):
        """Perform registration step of authentication"""
        start = time.time()
        try:
            response = await self.client.post(
                "/api/v1/auth/register",
                json={
                    "email": self.profile.email,
                    "password": f"SecurePass_{self.profile.id[:8]}",
                    "first_name": self.profile.name.split()[0],
                    "last_name": self.profile.name.split()[-1],
                    "student_id": self.profile.id,
                },
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code in [200, 201]:
                metrics.success_count += 1
                self.session_data["user_id"] = response.json().get("user_id")
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _perform_login(self, metrics: PerformanceMetrics):
        """Perform login step of authentication"""
        start = time.time()
        try:
            response = await self.client.post(
                "/api/v1/auth/login",
                json={"email": self.profile.email, "password": f"SecurePass_{self.profile.id[:8]}"},
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code == 200:
                metrics.success_count += 1
                self.session_data["token"] = response.json().get("token")
                self.session_data["session_id"] = response.json().get("session_id")
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _setup_2fa(self, metrics: PerformanceMetrics):
        """Setup 2FA if applicable"""
        if self.profile.student_type != StudentType.HIGH_ACHIEVER:
            return
        start = time.time()
        try:
            response = await self.client.post(
                "/api/v1/auth/enable-2fa",
                headers={"Authorization": f"Bearer {self.session_data.get('token', '')}"},
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _perform_password_reset(self, metrics: PerformanceMetrics):
        """Perform password reset if applicable"""
        if self.profile.student_type != StudentType.STRUGGLING or np.random.random() >= 0.3:
            return
        start = time.time()
        try:
            response = await self.client.post(
                "/api/v1/auth/reset-password", json={"email": self.profile.email}
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code in [200, 202]:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def simulate_authentication_flow(self) -> PerformanceMetrics:
        """Simulate complete authentication workflow"""
        metrics = PerformanceMetrics(
            workflow_name="authentication",
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_times=[],
            error_count=0,
            success_count=0,
            student_type=self.profile.student_type,
        )

        await self._perform_registration(metrics)
        await asyncio.sleep(np.random.uniform(0.5, 2.0))
        await self._perform_login(metrics)
        await self._setup_2fa(metrics)
        await self._perform_password_reset(metrics)

        metrics.end_time = datetime.now()
        return metrics

    def _get_browse_count(self) -> int:
        """Get number of courses to browse based on student type"""
        return {
            StudentType.HIGH_ACHIEVER: np.random.randint(5, 10),
            StudentType.AVERAGE_PERFORMER: np.random.randint(3, 6),
            StudentType.STRUGGLING: np.random.randint(1, 3),
            StudentType.AT_RISK: 1,
            StudentType.NON_TRADITIONAL: np.random.randint(2, 4),
        }.get(self.profile.student_type, 3)

    def _get_enrollment_count(self) -> int:
        """Get number of courses to enroll in based on student type"""
        return {
            StudentType.HIGH_ACHIEVER: np.random.randint(4, 6),
            StudentType.AVERAGE_PERFORMER: np.random.randint(3, 5),
            StudentType.STRUGGLING: np.random.randint(2, 4),
            StudentType.AT_RISK: np.random.randint(1, 3),
            StudentType.NON_TRADITIONAL: np.random.randint(2, 4),
        }.get(self.profile.student_type, 3)

    async def _browse_courses(self, metrics: PerformanceMetrics, headers: dict) -> list:
        """Browse and get available courses"""
        start = time.time()
        try:
            response = await self.client.get("/api/v1/courses", headers=headers)
            metrics.response_times.append(time.time() - start)
            if response.status_code == 200:
                metrics.success_count += 1
                return response.json().get("courses", [])
            else:
                metrics.error_count += 1
                return []
        except Exception:
            metrics.error_count += 1
            return []

    async def _view_course_details(self, course: dict, metrics: PerformanceMetrics, headers: dict):
        """View details for a specific course"""
        start = time.time()
        try:
            response = await self.client.get(f"/api/v1/courses/{course['id']}", headers=headers)
            metrics.response_times.append(time.time() - start)
            if response.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _enroll_in_course(self, course: dict, metrics: PerformanceMetrics, headers: dict) -> bool:
        """Enroll in a specific course"""
        start = time.time()
        try:
            response = await self.client.post(
                f"/api/v1/courses/{course['id']}/enroll",
                headers=headers,
                json={"student_id": self.profile.id},
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code in [200, 201]:
                metrics.success_count += 1
                return True
            else:
                metrics.error_count += 1
                return False
        except Exception:
            metrics.error_count += 1
            return False

    async def _drop_course_if_at_risk(self, enrolled_courses: list, metrics: PerformanceMetrics, headers: dict):
        """Drop a course if student is at-risk"""
        if (
            self.profile.student_type != StudentType.AT_RISK
            or not enrolled_courses
            or np.random.random() >= 0.4
        ):
            return

        course_to_drop = np.random.choice(enrolled_courses)
        start = time.time()
        try:
            response = await self.client.post(
                f"/api/v1/courses/{course_to_drop}/drop",
                headers=headers,
                json={"student_id": self.profile.id},
            )
            metrics.response_times.append(time.time() - start)
            if response.status_code == 200:
                metrics.success_count += 1
                enrolled_courses.remove(course_to_drop)
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def simulate_course_enrollment(self) -> PerformanceMetrics:
        """Simulate course browsing and enrollment"""
        metrics = PerformanceMetrics(
            workflow_name="course_enrollment",
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_times=[],
            error_count=0,
            success_count=0,
            student_type=self.profile.student_type,
        )

        headers = {"Authorization": f"Bearer {self.session_data.get('token', '')}"}

        # Browse available courses
        courses = await self._browse_courses(metrics, headers)

        # View course details based on student type
        browse_count = self._get_browse_count()
        for _ in range(min(browse_count, len(courses))):
            await asyncio.sleep(np.random.uniform(0.5, 2.0))
            if courses:
                course = np.random.choice(courses)
                await self._view_course_details(course, metrics, headers)

        # Enroll in courses
        enrolled_courses = []
        enrollment_count = self._get_enrollment_count()
        for i in range(min(enrollment_count, len(courses))):
            if courses:
                course = courses[i % len(courses)]
                if await self._enroll_in_course(course, metrics, headers):
                    enrolled_courses.append(course["id"])

        self.session_data["enrolled_courses"] = enrolled_courses

        # Drop course if at-risk
        await self._drop_course_if_at_risk(enrolled_courses, metrics, headers)

        metrics.end_time = datetime.now()
        return metrics

    async def _get_course_assignments(self, course_id: str, metrics: PerformanceMetrics, headers: dict) -> list:
        """Get assignments for a specific course"""
        start = time.time()
        try:
            response = await self.client.get(
                f"/api/v1/courses/{course_id}/assignments", headers=headers
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code == 200:
                metrics.success_count += 1
                return response.json().get("assignments", [])
            else:
                metrics.error_count += 1
                return []
        except Exception:
            metrics.error_count += 1
            return []

    async def _submit_assignment(self, assignment: dict, metrics: PerformanceMetrics, headers: dict) -> dict:
        """Submit a single assignment"""
        # Determine submission parameters
        submit_on_time = np.random.random() < self.profile.submission_timeliness
        work_duration = self.profile.avg_session_duration * np.random.uniform(0.5, 2.0)
        await asyncio.sleep(min(work_duration / 60, 1.0))  # Scale down for testing

        # Calculate submission quality
        base_quality = self.profile.base_performance / 100
        quality_variance = np.random.normal(0, self.profile.performance_variance)
        submission_quality = max(0, min(1, base_quality + quality_variance))

        # Create submission data
        submission_data = {
            "student_id": self.profile.id,
            "assignment_id": assignment["id"],
            "submission_text": f"Submission for {assignment['title']}",
            "quality_score": submission_quality,
            "time_spent_minutes": int(work_duration),
            "submitted_on_time": submit_on_time,
        }

        # Submit assignment
        start = time.time()
        try:
            response = await self.client.post(
                f"/api/v1/assignments/{assignment['id']}/submit",
                headers=headers,
                json=submission_data,
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code in [200, 201]:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

        return submission_data

    async def _resubmit_if_struggling(self, assignment: dict, submission_data: dict, metrics: PerformanceMetrics, headers: dict):
        """Handle resubmission for struggling students"""
        if self.profile.student_type != StudentType.STRUGGLING or np.random.random() >= 0.3:
            return

        await asyncio.sleep(0.5)
        start = time.time()
        try:
            response = await self.client.put(
                f"/api/v1/assignments/{assignment['id']}/resubmit",
                headers=headers,
                json={**submission_data, "revision": 2},
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def simulate_assignment_submission(self) -> PerformanceMetrics:
        """Simulate assignment viewing and submission workflow"""
        metrics = PerformanceMetrics(
            workflow_name="assignment_submission",
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_times=[],
            error_count=0,
            success_count=0,
            student_type=self.profile.student_type,
        )

        headers = {"Authorization": f"Bearer {self.session_data.get('token', '')}"}
        enrolled_courses = self.session_data.get("enrolled_courses", [])

        if not enrolled_courses:
            metrics.end_time = datetime.now()
            return metrics

        for course_id in enrolled_courses[:2]:  # Limit to 2 courses for testing
            # Get assignments for the course
            assignments = await self._get_course_assignments(course_id, metrics, headers)

            # Submit assignments based on student profile
            for assignment in assignments[:3]:  # Limit to 3 assignments per course
                submission_data = await self._submit_assignment(assignment, metrics, headers)
                await self._resubmit_if_struggling(assignment, submission_data, metrics, headers)

        metrics.end_time = datetime.now()
        return metrics

    def _get_analytics_frequency(self) -> int:
        """Get analytics check frequency based on student type"""
        return {
            StudentType.HIGH_ACHIEVER: 5,
            StudentType.AVERAGE_PERFORMER: 3,
            StudentType.STRUGGLING: 2,
            StudentType.AT_RISK: 1,
            StudentType.NON_TRADITIONAL: 3,
        }.get(self.profile.student_type, 2)

    async def _get_personal_analytics(self, metrics: PerformanceMetrics, headers: dict):
        """Get personal analytics for the student"""
        start = time.time()
        try:
            response = await self.client.get(
                f"/api/v1/analytics/student/{self.profile.id}", headers=headers
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code == 200:
                metrics.success_count += 1
                response.json()
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _get_performance_predictions(self, metrics: PerformanceMetrics, headers: dict):
        """Get performance predictions for the student"""
        start = time.time()
        try:
            response = await self.client.get(
                f"/api/v1/analytics/predictions/{self.profile.id}", headers=headers
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _get_recommendations(self, metrics: PerformanceMetrics, headers: dict):
        """Get recommendations for specific student types"""
        if self.profile.student_type not in [StudentType.HIGH_ACHIEVER, StudentType.STRUGGLING]:
            return

        start = time.time()
        try:
            response = await self.client.get(
                f"/api/v1/analytics/recommendations/{self.profile.id}", headers=headers
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code == 200:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def _export_report_occasionally(self, metrics: PerformanceMetrics, headers: dict):
        """Export report with low probability"""
        if np.random.random() >= 0.2:
            return

        start = time.time()
        try:
            response = await self.client.post(
                "/api/v1/analytics/export",
                headers=headers,
                json={
                    "student_id": self.profile.id,
                    "format": "pdf",
                    "include_predictions": True,
                },
            )
            metrics.response_times.append(time.time() - start)

            if response.status_code in [200, 202]:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
        except Exception:
            metrics.error_count += 1

    async def simulate_analytics_interaction(self) -> PerformanceMetrics:
        """Simulate interaction with analytics dashboard"""
        metrics = PerformanceMetrics(
            workflow_name="analytics_dashboard",
            start_time=datetime.now(),
            end_time=datetime.now(),
            response_times=[],
            error_count=0,
            success_count=0,
            student_type=self.profile.student_type,
        )

        headers = {"Authorization": f"Bearer {self.session_data.get('token', '')}"}
        check_frequency = self._get_analytics_frequency()

        for _ in range(check_frequency):
            await self._get_personal_analytics(metrics, headers)
            await self._get_performance_predictions(metrics, headers)
            await self._get_recommendations(metrics, headers)
            await asyncio.sleep(np.random.uniform(0.5, 2.0))

        await self._export_report_occasionally(metrics, headers)

        metrics.end_time = datetime.now()
        return metrics


class TestE2EStudentWorkflows:
    """End-to-end tests with multiple student profiles"""

    @pytest.fixture
    def student_profiles(self):
        """Generate diverse student profiles for testing"""
        simulator = StudentSimulator(seed=42)
        profiles = []

        # Generate 2 of each type for testing
        for student_type in StudentType:
            for _ in range(2):
                profiles.append(simulator.generate_student(student_type))

        return profiles

    @pytest.mark.asyncio
    async def test_complete_student_lifecycle(self, async_client, student_profiles):
        """Test complete student lifecycle with all workflows"""
        all_metrics = []

        for profile in student_profiles[:5]:  # Limit for testing
            simulator = StudentWorkflowSimulator(async_client, profile)

            # Run all workflows in sequence
            auth_metrics = await simulator.simulate_authentication_flow()
            all_metrics.append(auth_metrics)

            enrollment_metrics = await simulator.simulate_course_enrollment()
            all_metrics.append(enrollment_metrics)

            assignment_metrics = await simulator.simulate_assignment_submission()
            all_metrics.append(assignment_metrics)

            analytics_metrics = await simulator.simulate_analytics_interaction()
            all_metrics.append(analytics_metrics)

        # Analyze performance metrics
        self._analyze_metrics(all_metrics)

    @pytest.mark.asyncio
    async def test_concurrent_student_load(self, async_client, student_profiles):
        """Test system under concurrent student load"""
        simulators = [
            StudentWorkflowSimulator(async_client, profile) for profile in student_profiles[:10]
        ]

        # Run workflows concurrently
        tasks = []
        for simulator in simulators:
            tasks.append(simulator.simulate_authentication_flow())
            tasks.append(simulator.simulate_course_enrollment())

        metrics = await asyncio.gather(*tasks)

        # Verify system handles concurrent load
        success_rates = [m.success_rate for m in metrics]
        avg_success_rate = np.mean(success_rates)

        assert avg_success_rate > 0.8, f"Success rate too low under load: {avg_success_rate}"

        # Check response times
        p95_times = [m.p95_response_time for m in metrics]
        avg_p95 = np.mean(p95_times)

        assert avg_p95 < 2.0, f"P95 response time too high: {avg_p95}s"

    @pytest.mark.asyncio
    async def test_student_type_behaviors(self, async_client):
        """Test that different student types exhibit expected behaviors"""
        simulator = StudentSimulator(seed=42)

        # Test high achiever behavior
        high_achiever = simulator.generate_student(StudentType.HIGH_ACHIEVER)
        ha_simulator = StudentWorkflowSimulator(async_client, high_achiever)
        ha_metrics = await ha_simulator.simulate_assignment_submission()

        # Test at-risk student behavior
        at_risk = simulator.generate_student(StudentType.AT_RISK)
        ar_simulator = StudentWorkflowSimulator(async_client, at_risk)
        ar_metrics = await ar_simulator.simulate_assignment_submission()

        # High achievers should have better success rates
        assert ha_metrics.success_rate >= ar_metrics.success_rate

        # High achievers should interact more
        assert ha_metrics.success_count >= ar_metrics.success_count

    def _analyze_metrics(self, metrics: List[PerformanceMetrics]):
        """Analyze and report performance metrics"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_workflows": len(metrics),
            "by_workflow": {},
            "by_student_type": {},
            "overall": {
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "total_errors": 0,
                "total_successes": 0,
                "overall_success_rate": 0,
            },
        }

        # Group by workflow
        for metric in metrics:
            workflow = metric.workflow_name
            if workflow not in report["by_workflow"]:
                report["by_workflow"][workflow] = []
            report["by_workflow"][workflow].append(metric.to_dict())

        # Group by student type
        for metric in metrics:
            student_type = metric.student_type.value
            if student_type not in report["by_student_type"]:
                report["by_student_type"][student_type] = []
            report["by_student_type"][student_type].append(metric.to_dict())

        # Calculate overall metrics
        all_response_times = []
        total_errors = 0
        total_successes = 0

        for metric in metrics:
            all_response_times.extend(metric.response_times)
            total_errors += metric.error_count
            total_successes += metric.success_count

        if all_response_times:
            report["overall"]["avg_response_time_ms"] = np.mean(all_response_times) * 1000
            report["overall"]["p95_response_time_ms"] = np.percentile(all_response_times, 95) * 1000
            report["overall"]["p99_response_time_ms"] = np.percentile(all_response_times, 99) * 1000

        report["overall"]["total_errors"] = total_errors
        report["overall"]["total_successes"] = total_successes
        report["overall"]["overall_success_rate"] = (
            total_successes / (total_successes + total_errors)
            if (total_successes + total_errors) > 0
            else 0
        )

        # Save report
        with open("e2e_performance_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("\nPerformance Report Generated: e2e_performance_report.json")
        print(f"Overall Success Rate: {report['overall']['overall_success_rate']:.2%}")
        print(f"Avg Response Time: {report['overall']['avg_response_time_ms']:.2f}ms")
        print(f"P95 Response Time: {report['overall']['p95_response_time_ms']:.2f}ms")
