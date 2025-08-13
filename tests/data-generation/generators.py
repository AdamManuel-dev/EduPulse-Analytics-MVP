"""
Data generation utilities for comprehensive testing
Creates realistic student profiles, courses, and interaction patterns
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from faker import Faker

fake = Faker()


class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"


class StudentType(Enum):
    HIGH_ACHIEVER = "high_achiever"
    AVERAGE_PERFORMER = "average_performer"
    STRUGGLING = "struggling"
    AT_RISK = "at_risk"
    NON_TRADITIONAL = "non_traditional"


class EngagementPattern(Enum):
    CONSISTENT = "consistent"
    SPORADIC = "sporadic"
    DECLINING = "declining"
    IMPROVING = "improving"
    BINGE = "binge"


@dataclass
class StudentProfile:
    """Comprehensive student profile with hidden characteristics"""

    id: str
    name: str
    email: str
    student_type: StudentType
    learning_style: LearningStyle
    engagement_pattern: EngagementPattern

    # Academic characteristics
    base_performance: float  # 0-100
    performance_variance: float  # How much their performance varies
    submission_timeliness: float  # 0-1, probability of on-time submission
    participation_rate: float  # 0-1, how often they participate

    # Behavioral patterns
    preferred_study_hours: List[int]  # Hours of day they prefer to study
    avg_session_duration: int  # Minutes
    weekly_login_frequency: int
    help_seeking_probability: float  # 0-1

    # Technical characteristics
    primary_device: str
    network_quality: str  # 'excellent', 'good', 'poor'
    timezone: str

    # Hidden factors affecting performance
    stress_level: float  # 0-1, affects performance
    motivation_trend: str  # 'increasing', 'stable', 'decreasing'
    external_commitments: float  # 0-1, affects availability
    prior_knowledge: float  # 0-1, affects learning speed


class StudentSimulator:
    """Generates realistic student profiles with embedded characteristics"""

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        self.fake = Faker()
        if seed:
            Faker.seed(seed)

    def generate_student(self, student_type: Optional[StudentType] = None) -> StudentProfile:
        """Generate a student profile with realistic characteristics"""

        if student_type is None:
            # Realistic distribution of student types
            student_type = np.random.choice(
                list(StudentType),
                p=[
                    0.15,
                    0.50,
                    0.20,
                    0.10,
                    0.05,
                ],  # High achiever, average, struggling, at-risk, non-traditional
            )

        profile_config = self._get_profile_config(student_type)

        return StudentProfile(
            id=self.fake.uuid4(),
            name=self.fake.name(),
            email=self.fake.email(),
            student_type=student_type,
            learning_style=random.choice(list(LearningStyle)),
            engagement_pattern=profile_config["engagement_pattern"],
            base_performance=profile_config["base_performance"],
            performance_variance=profile_config["performance_variance"],
            submission_timeliness=profile_config["submission_timeliness"],
            participation_rate=profile_config["participation_rate"],
            preferred_study_hours=profile_config["study_hours"],
            avg_session_duration=profile_config["session_duration"],
            weekly_login_frequency=profile_config["login_frequency"],
            help_seeking_probability=profile_config["help_seeking"],
            primary_device=profile_config["device"],
            network_quality=profile_config["network"],
            timezone=self.fake.timezone(),
            stress_level=profile_config["stress"],
            motivation_trend=profile_config["motivation"],
            external_commitments=profile_config["commitments"],
            prior_knowledge=profile_config["prior_knowledge"],
        )

    def _get_profile_config(self, student_type: StudentType) -> Dict[str, Any]:
        """Get configuration for different student types"""

        configs = {
            StudentType.HIGH_ACHIEVER: {
                "engagement_pattern": EngagementPattern.CONSISTENT,
                "base_performance": np.random.uniform(85, 95),
                "performance_variance": np.random.uniform(0.02, 0.05),
                "submission_timeliness": np.random.uniform(0.95, 1.0),
                "participation_rate": np.random.uniform(0.8, 0.95),
                "study_hours": list(range(8, 22)),  # Studies throughout the day
                "session_duration": int(np.random.normal(90, 15)),
                "login_frequency": int(np.random.uniform(6, 7)),
                "help_seeking": np.random.uniform(0.3, 0.5),
                "device": np.random.choice(["laptop", "desktop"], p=[0.7, 0.3]),
                "network": "excellent",
                "stress": np.random.uniform(0.2, 0.4),
                "motivation": "stable",
                "commitments": np.random.uniform(0.1, 0.3),
                "prior_knowledge": np.random.uniform(0.7, 0.9),
            },
            StudentType.AVERAGE_PERFORMER: {
                "engagement_pattern": np.random.choice(
                    [EngagementPattern.CONSISTENT, EngagementPattern.SPORADIC]
                ),
                "base_performance": np.random.uniform(70, 85),
                "performance_variance": np.random.uniform(0.05, 0.10),
                "submission_timeliness": np.random.uniform(0.7, 0.9),
                "participation_rate": np.random.uniform(0.5, 0.7),
                "study_hours": list(range(10, 22)),
                "session_duration": int(np.random.normal(60, 20)),
                "login_frequency": int(np.random.uniform(4, 6)),
                "help_seeking": np.random.uniform(0.4, 0.6),
                "device": np.random.choice(["laptop", "tablet", "desktop"], p=[0.6, 0.2, 0.2]),
                "network": np.random.choice(["excellent", "good"], p=[0.6, 0.4]),
                "stress": np.random.uniform(0.3, 0.6),
                "motivation": np.random.choice(
                    ["stable", "increasing", "decreasing"], p=[0.6, 0.2, 0.2]
                ),
                "commitments": np.random.uniform(0.3, 0.5),
                "prior_knowledge": np.random.uniform(0.4, 0.7),
            },
            StudentType.STRUGGLING: {
                "engagement_pattern": np.random.choice(
                    [EngagementPattern.SPORADIC, EngagementPattern.DECLINING]
                ),
                "base_performance": np.random.uniform(55, 70),
                "performance_variance": np.random.uniform(0.10, 0.20),
                "submission_timeliness": np.random.uniform(0.4, 0.7),
                "participation_rate": np.random.uniform(0.2, 0.5),
                "study_hours": list(range(18, 24)) + list(range(0, 2)),  # Late night studying
                "session_duration": int(np.random.normal(45, 25)),
                "login_frequency": int(np.random.uniform(2, 4)),
                "help_seeking": np.random.uniform(0.6, 0.8),
                "device": np.random.choice(["laptop", "tablet", "phone"], p=[0.4, 0.3, 0.3]),
                "network": np.random.choice(["good", "poor"], p=[0.5, 0.5]),
                "stress": np.random.uniform(0.6, 0.9),
                "motivation": np.random.choice(["declining", "stable"], p=[0.7, 0.3]),
                "commitments": np.random.uniform(0.5, 0.8),
                "prior_knowledge": np.random.uniform(0.1, 0.4),
            },
            StudentType.AT_RISK: {
                "engagement_pattern": EngagementPattern.DECLINING,
                "base_performance": np.random.uniform(40, 60),
                "performance_variance": np.random.uniform(0.15, 0.30),
                "submission_timeliness": np.random.uniform(0.1, 0.4),
                "participation_rate": np.random.uniform(0.05, 0.2),
                "study_hours": random.sample(range(24), 4),  # Very irregular
                "session_duration": int(np.random.normal(30, 20)),
                "login_frequency": int(np.random.uniform(1, 3)),
                "help_seeking": np.random.uniform(0.1, 0.3),  # Often don't seek help
                "device": np.random.choice(["phone", "tablet", "laptop"], p=[0.5, 0.3, 0.2]),
                "network": np.random.choice(["poor", "good"], p=[0.7, 0.3]),
                "stress": np.random.uniform(0.7, 1.0),
                "motivation": "decreasing",
                "commitments": np.random.uniform(0.7, 1.0),
                "prior_knowledge": np.random.uniform(0.0, 0.3),
            },
            StudentType.NON_TRADITIONAL: {
                "engagement_pattern": EngagementPattern.BINGE,
                "base_performance": np.random.uniform(65, 85),
                "performance_variance": np.random.uniform(0.08, 0.15),
                "submission_timeliness": np.random.uniform(0.6, 0.85),
                "participation_rate": np.random.uniform(0.3, 0.6),
                "study_hours": list(range(20, 24)) + list(range(5, 8)),  # Evening/early morning
                "session_duration": int(np.random.normal(120, 30)),  # Long sessions
                "login_frequency": int(np.random.uniform(3, 5)),
                "help_seeking": np.random.uniform(0.2, 0.5),
                "device": np.random.choice(["laptop", "desktop"], p=[0.6, 0.4]),
                "network": np.random.choice(["excellent", "good"], p=[0.7, 0.3]),
                "stress": np.random.uniform(0.5, 0.8),
                "motivation": np.random.choice(["stable", "increasing"], p=[0.6, 0.4]),
                "commitments": np.random.uniform(0.6, 0.9),  # Work/family
                "prior_knowledge": np.random.uniform(0.3, 0.8),  # Variable background
            },
        }

        return configs[student_type]

    def simulate_behavior(
        self, student: StudentProfile, course_duration_days: int = 120
    ) -> Dict[str, Any]:
        """Simulate student behavior over a course duration"""

        behavior_log = {
            "student_id": student.id,
            "daily_activities": [],
            "assignments": [],
            "quiz_scores": [],
            "forum_posts": [],
            "help_requests": [],
        }

        current_date = datetime.now() - timedelta(days=course_duration_days)

        for day in range(course_duration_days):
            current_date += timedelta(days=1)

            # Determine if student logs in today
            login_probability = student.weekly_login_frequency / 7.0
            if random.random() < login_probability:
                # Generate session data
                session = self._generate_session(student, current_date)
                behavior_log["daily_activities"].append(session)

                # Check for assignments due
                if day % 7 == 0:  # Weekly assignments
                    assignment = self._generate_assignment_submission(student, day)
                    behavior_log["assignments"].append(assignment)

                # Quiz attempts
                if day % 14 == 0:  # Bi-weekly quizzes
                    quiz = self._generate_quiz_attempt(student, day)
                    behavior_log["quiz_scores"].append(quiz)

                # Forum participation
                if random.random() < student.participation_rate:
                    post = self._generate_forum_post(student, current_date)
                    behavior_log["forum_posts"].append(post)

                # Help seeking
                if random.random() < student.help_seeking_probability:
                    help_request = self._generate_help_request(student, current_date)
                    behavior_log["help_requests"].append(help_request)

        return behavior_log

    def _generate_session(self, student: StudentProfile, date: datetime) -> Dict[str, Any]:
        """Generate a study session"""
        hour = random.choice(student.preferred_study_hours)
        duration = max(5, int(np.random.normal(student.avg_session_duration, 15)))

        # Adjust duration based on stress and motivation
        if student.stress_level > 0.7:
            duration = int(duration * 0.8)
        if student.motivation_trend == "decreasing":
            duration = int(duration * 0.9)

        return {
            "timestamp": date.replace(hour=hour).isoformat(),
            "duration_minutes": duration,
            "pages_viewed": int(duration / 3),  # Approximate pages per session
            "videos_watched": int(duration / 20) if random.random() > 0.5 else 0,
            "device": student.primary_device,
            "network_quality": student.network_quality,
        }

    def _generate_assignment_submission(self, student: StudentProfile, day: int) -> Dict[str, Any]:
        """Generate assignment submission data"""

        # Calculate performance with variance and external factors
        performance = student.base_performance
        performance += np.random.normal(0, student.performance_variance * 100)

        # Adjust for stress and external commitments
        performance *= 1 - student.stress_level * 0.1
        performance *= 1 - student.external_commitments * 0.1

        # Improve over time if prior knowledge is high
        if student.prior_knowledge > 0.6:
            performance += (day / 120) * 5  # Gradual improvement

        performance = max(0, min(100, performance))

        # Determine if submitted on time
        on_time = random.random() < student.submission_timeliness

        return {
            "assignment_id": f"assignment_{day//7}",
            "score": round(performance, 2),
            "submitted_on_time": on_time,
            "late_hours": 0 if on_time else int(np.random.exponential(24)),
            "attempts": 1
            if student.student_type == StudentType.HIGH_ACHIEVER
            else random.randint(1, 3),
            "time_spent_minutes": int(np.random.normal(120, 30)),
        }

    def _generate_quiz_attempt(self, student: StudentProfile, day: int) -> Dict[str, Any]:
        """Generate quiz attempt data"""

        # Quiz performance similar to assignments but with more variance
        performance = student.base_performance
        performance += np.random.normal(0, student.performance_variance * 150)

        # Adjust for preparation (based on login frequency)
        prep_factor = student.weekly_login_frequency / 7.0
        performance *= 0.7 + prep_factor * 0.3

        performance = max(0, min(100, performance))

        return {
            "quiz_id": f"quiz_{day//14}",
            "score": round(performance, 2),
            "attempts": 1 if performance > 80 else random.randint(1, 3),
            "time_spent_minutes": int(np.random.normal(30, 10)),
            "questions_answered": 20,
            "questions_correct": int(20 * performance / 100),
        }

    def _generate_forum_post(self, student: StudentProfile, date: datetime) -> Dict[str, Any]:
        """Generate forum participation data"""

        post_types = ["question", "answer", "discussion"]
        weights = [0.4, 0.3, 0.3] if student.help_seeking_probability > 0.5 else [0.2, 0.5, 0.3]

        return {
            "timestamp": date.isoformat(),
            "type": np.random.choice(post_types, p=weights),
            "word_count": int(np.random.normal(50, 20)),
            "received_likes": random.randint(0, 5),
            "is_helpful": random.random() < (student.prior_knowledge * 0.7),
        }

    def _generate_help_request(self, student: StudentProfile, date: datetime) -> Dict[str, Any]:
        """Generate help-seeking behavior"""

        help_types = ["instructor_email", "ta_office_hours", "peer_tutoring", "online_resources"]

        return {
            "timestamp": date.isoformat(),
            "type": random.choice(help_types),
            "topic": random.choice(["assignment", "concept", "technical", "administrative"]),
            "urgency": "high"
            if student.stress_level > 0.7
            else random.choice(["low", "medium", "high"]),
            "resolved": random.random() < 0.8,
        }


class CourseDataGenerator:
    """Generate realistic course data"""

    def __init__(self):
        self.fake = Faker()

    def generate_course(self, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a course with assignments, quizzes, and materials"""

        difficulties = {
            "easy": {"assignments": 8, "quizzes": 4, "projects": 1},
            "medium": {"assignments": 12, "quizzes": 6, "projects": 2},
            "hard": {"assignments": 16, "quizzes": 8, "projects": 3},
        }

        config = difficulties[difficulty]

        return {
            "course_id": self.fake.uuid4(),
            "name": f"{self.fake.catch_phrase()} {random.choice(['101', '201', '301', '401'])}",
            "instructor": self.fake.name(),
            "credits": random.choice([3, 4]),
            "difficulty": difficulty,
            "assignments": [self._generate_assignment(i) for i in range(config["assignments"])],
            "quizzes": [self._generate_quiz(i) for i in range(config["quizzes"])],
            "projects": [self._generate_project(i) for i in range(config["projects"])],
            "duration_weeks": 16,
            "prerequisites": random.sample(
                ["MATH101", "CS101", "PHYS101", "CHEM101"], k=random.randint(0, 2)
            ),
        }

    def _generate_assignment(self, index: int) -> Dict[str, Any]:
        return {
            "id": f"assignment_{index}",
            "title": f"Assignment {index + 1}: {self.fake.bs()}",
            "points": random.choice([10, 20, 25, 50]),
            "due_week": index + 1,
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "estimated_hours": random.randint(2, 8),
        }

    def _generate_quiz(self, index: int) -> Dict[str, Any]:
        return {
            "id": f"quiz_{index}",
            "title": f"Quiz {index + 1}",
            "questions": random.randint(10, 30),
            "points": random.choice([20, 30, 40]),
            "time_limit_minutes": random.choice([30, 45, 60]),
            "attempts_allowed": random.randint(1, 3),
        }

    def _generate_project(self, index: int) -> Dict[str, Any]:
        return {
            "id": f"project_{index}",
            "title": f"Project {index + 1}: {self.fake.catch_phrase()}",
            "points": random.choice([100, 150, 200]),
            "due_week": 5 + (index * 5),
            "team_size": random.randint(1, 4),
            "estimated_hours": random.randint(20, 60),
        }


def generate_test_dataset(
    num_students: int = 100, num_courses: int = 10, output_file: str = "test_data.json"
) -> None:
    """Generate a complete test dataset"""

    simulator = StudentSimulator(seed=42)
    course_gen = CourseDataGenerator()

    dataset = {"students": [], "courses": [], "behaviors": []}

    # Generate students with realistic distribution
    for i in range(num_students):
        student = simulator.generate_student()
        dataset["students"].append(asdict(student))

        # Generate behavior for 2-3 courses
        for _ in range(random.randint(2, 3)):
            behavior = simulator.simulate_behavior(student, course_duration_days=120)
            dataset["behaviors"].append(behavior)

    # Generate courses
    for _ in range(num_courses):
        course = course_gen.generate_course(difficulty=random.choice(["easy", "medium", "hard"]))
        dataset["courses"].append(course)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2, default=str)

    print(f"Generated test dataset with {num_students} students and {num_courses} courses")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    # Generate test dataset
    generate_test_dataset(num_students=100, num_courses=10)
