"""
Benchmark Question Generator

Generates evaluation questions at 5 difficulty levels:
- L1: Single-hop entity lookup
- L2: Temporal constraint reasoning
- L3: Spatial multi-hop traversal
- L4: Joint spatio-temporal constraints
- L5: Conflict detection
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with ground truth."""

    question_id: str
    level: str
    question: str
    ground_truth: Any  # List of deed IDs or count
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "level": self.level,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuestion":
        """Create from dictionary."""
        return cls(
            question_id=data["question_id"],
            level=data["level"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            metadata=data.get("metadata", {})
        )


class BenchmarkQuestionGenerator:
    """
    Generates benchmark questions for RAG evaluation.

    Question Types:
    - L1_single_hop: Direct deed lookup by attribute
    - L2_temporal: Year or date range queries
    - L3_spatial_multihop: Multi-hop via shared streets
    - L4_spatiotemporal: Combined subdivision + decade
    - L5_conflict: Inconsistency detection
    """

    LEVEL_NAMES = [
        "L1_single_hop",
        "L2_temporal",
        "L3_spatial_multihop",
        "L4_spatiotemporal",
        "L5_conflict"
    ]

    def __init__(self, data: Dict[str, Any], seed: int = 42):
        """
        Initialize with generated data.

        Args:
            data: Output from SyntheticDeedGenerator.generate()
            seed: Random seed for reproducibility
        """
        self.data = data
        self.deeds = data.get("deeds", {})
        self.subdivisions = data.get("subdivisions", {})
        self.streets = data.get("streets", {})
        self.rng = random.Random(seed)

        # Build indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices for question generation."""
        self.deeds_by_year: Dict[int, List[str]] = {}
        self.deeds_by_subdivision: Dict[str, List[str]] = {}
        self.deeds_by_street: Dict[str, List[str]] = {}
        self.covenants_by_subdivision: Dict[str, List[str]] = {}

        for deed_id, deed in self.deeds.items():
            year = deed["signed_year"]
            sub_id = deed["subdivision_id"]
            street_id = deed["street_id"]

            # By year
            if year not in self.deeds_by_year:
                self.deeds_by_year[year] = []
            self.deeds_by_year[year].append(deed_id)

            # By subdivision
            if sub_id not in self.deeds_by_subdivision:
                self.deeds_by_subdivision[sub_id] = []
            self.deeds_by_subdivision[sub_id].append(deed_id)

            # By street
            if street_id not in self.deeds_by_street:
                self.deeds_by_street[street_id] = []
            self.deeds_by_street[street_id].append(deed_id)

            # Covenants
            if deed.get("has_covenant"):
                if sub_id not in self.covenants_by_subdivision:
                    self.covenants_by_subdivision[sub_id] = []
                self.covenants_by_subdivision[sub_id].append(deed_id)

    def generate_all(self, questions_per_level: int = 10) -> Dict[str, List[BenchmarkQuestion]]:
        """
        Generate questions for all difficulty levels.

        Args:
            questions_per_level: Number of questions per level

        Returns:
            Dict mapping level name to list of questions
        """
        questions = {}

        for level in self.LEVEL_NAMES:
            if level == "L1_single_hop":
                questions[level] = self._generate_l1_questions(questions_per_level)
            elif level == "L2_temporal":
                questions[level] = self._generate_l2_questions(questions_per_level)
            elif level == "L3_spatial_multihop":
                questions[level] = self._generate_l3_questions(questions_per_level)
            elif level == "L4_spatiotemporal":
                questions[level] = self._generate_l4_questions(questions_per_level)
            elif level == "L5_conflict":
                questions[level] = self._generate_l5_questions(questions_per_level)

        return questions

    def _generate_l1_questions(self, n: int) -> List[BenchmarkQuestion]:
        """
        L1: Single-hop entity lookup.

        Examples:
        - Find all deeds recorded in 1924
        - Which deeds mention Oak Street?
        """
        questions = []

        # Year lookup questions
        years = list(self.deeds_by_year.keys())
        selected_years = self.rng.sample(years, min(n // 2, len(years)))

        for i, year in enumerate(selected_years):
            deed_ids = self.deeds_by_year[year]
            q = BenchmarkQuestion(
                question_id=f"q_l1_{i+1:03d}",
                level="L1_single_hop",
                question=f"Find all deeds recorded in {year}.",
                ground_truth=deed_ids,
                metadata={
                    "type": "year_lookup",
                    "temporal_constraints": {"specific_year": year},
                    "spatial_constraints": {},
                    "required_hops": 1,
                    "expected_count": len(deed_ids)
                }
            )
            questions.append(q)

        # Subdivision lookup questions
        subs = list(self.subdivisions.keys())
        selected_subs = self.rng.sample(subs, min(n - len(questions), len(subs)))

        for i, sub_id in enumerate(selected_subs):
            sub_name = self.subdivisions[sub_id]["name"]
            deed_ids = self.deeds_by_subdivision.get(sub_id, [])
            q = BenchmarkQuestion(
                question_id=f"q_l1_{len(questions)+1:03d}",
                level="L1_single_hop",
                question=f"List all deeds in the {sub_name} subdivision.",
                ground_truth=deed_ids,
                metadata={
                    "type": "subdivision_lookup",
                    "temporal_constraints": {},
                    "spatial_constraints": {"subdivision_id": sub_id, "subdivision_name": sub_name},
                    "required_hops": 1,
                    "expected_count": len(deed_ids)
                }
            )
            questions.append(q)

        return questions[:n]

    def _generate_l2_questions(self, n: int) -> List[BenchmarkQuestion]:
        """
        L2: Temporal constraint reasoning.

        Examples:
        - List deeds signed between 1926 and 1939
        - Which deeds were recorded before 1925?
        """
        questions = []
        years = sorted(self.deeds_by_year.keys())
        min_year = min(years) if years else 1910
        max_year = max(years) if years else 1950

        for i in range(n):
            # Randomly choose question type
            q_type = self.rng.choice(["range", "before", "after", "decade"])

            if q_type == "range":
                start = self.rng.randint(min_year, max_year - 5)
                end = self.rng.randint(start + 3, min(start + 15, max_year))

                deed_ids = []
                for year in range(start, end + 1):
                    deed_ids.extend(self.deeds_by_year.get(year, []))

                q = BenchmarkQuestion(
                    question_id=f"q_l2_{i+1:03d}",
                    level="L2_temporal",
                    question=f"List all deeds signed between {start} and {end}.",
                    ground_truth=deed_ids,
                    metadata={
                        "type": "year_range",
                        "temporal_constraints": {"start_year": start, "end_year": end},
                        "spatial_constraints": {},
                        "required_hops": 1,
                        "expected_count": len(deed_ids)
                    }
                )

            elif q_type == "before":
                cutoff = self.rng.randint(min_year + 5, max_year)
                deed_ids = []
                for year in years:
                    if year < cutoff:
                        deed_ids.extend(self.deeds_by_year.get(year, []))

                q = BenchmarkQuestion(
                    question_id=f"q_l2_{i+1:03d}",
                    level="L2_temporal",
                    question=f"Which deeds were recorded before {cutoff}?",
                    ground_truth=deed_ids,
                    metadata={
                        "type": "before_year",
                        "temporal_constraints": {"end_year": cutoff - 1},
                        "spatial_constraints": {},
                        "required_hops": 1,
                        "expected_count": len(deed_ids)
                    }
                )

            elif q_type == "after":
                cutoff = self.rng.randint(min_year, max_year - 5)
                deed_ids = []
                for year in years:
                    if year > cutoff:
                        deed_ids.extend(self.deeds_by_year.get(year, []))

                q = BenchmarkQuestion(
                    question_id=f"q_l2_{i+1:03d}",
                    level="L2_temporal",
                    question=f"Find deeds recorded after {cutoff}.",
                    ground_truth=deed_ids,
                    metadata={
                        "type": "after_year",
                        "temporal_constraints": {"start_year": cutoff + 1},
                        "spatial_constraints": {},
                        "required_hops": 1,
                        "expected_count": len(deed_ids)
                    }
                )

            else:  # decade
                decade = self.rng.choice([1910, 1920, 1930, 1940])
                deed_ids = []
                for year in range(decade, decade + 10):
                    deed_ids.extend(self.deeds_by_year.get(year, []))

                q = BenchmarkQuestion(
                    question_id=f"q_l2_{i+1:03d}",
                    level="L2_temporal",
                    question=f"List all deeds from the {decade}s.",
                    ground_truth=deed_ids,
                    metadata={
                        "type": "decade",
                        "temporal_constraints": {"start_year": decade, "end_year": decade + 9, "decade": decade},
                        "spatial_constraints": {},
                        "required_hops": 1,
                        "expected_count": len(deed_ids)
                    }
                )

            questions.append(q)

        return questions

    def _generate_l3_questions(self, n: int) -> List[BenchmarkQuestion]:
        """
        L3: Spatial multi-hop traversal.

        Examples:
        - Which deeds share streets with deed_0001?
        - Find all deeds on the same street as deed_0005.
        """
        questions = []

        # Find deeds that have street neighbors
        deeds_with_neighbors = []
        for deed_id, deed in self.deeds.items():
            street_id = deed["street_id"]
            neighbors = [d for d in self.deeds_by_street.get(street_id, []) if d != deed_id]
            if neighbors:
                deeds_with_neighbors.append((deed_id, neighbors, street_id))

        # Sample deeds for questions
        selected = self.rng.sample(deeds_with_neighbors, min(n, len(deeds_with_neighbors)))

        for i, (deed_id, neighbors, street_id) in enumerate(selected):
            street_name = self.streets[street_id]["name"]

            # Alternate question phrasings
            if i % 2 == 0:
                question = f"Which deeds share a street with {deed_id}?"
            else:
                question = f"Find all deeds on the same street as {deed_id}."

            q = BenchmarkQuestion(
                question_id=f"q_l3_{i+1:03d}",
                level="L3_spatial_multihop",
                question=question,
                ground_truth=neighbors,
                metadata={
                    "type": "street_neighbors",
                    "temporal_constraints": {},
                    "spatial_constraints": {"reference_deed": deed_id, "street_name": street_name},
                    "required_hops": 2,
                    "expected_count": len(neighbors)
                }
            )
            questions.append(q)

        return questions[:n]

    def _generate_l4_questions(self, n: int) -> List[BenchmarkQuestion]:
        """
        L4: Joint spatio-temporal constraints.

        Examples:
        - How many covenants in Pine Valley during the 1920s?
        - List deeds in Oak Heights between 1925 and 1935.
        """
        questions = []

        for i in range(n):
            # Choose subdivision
            sub_id = self.rng.choice(list(self.subdivisions.keys()))
            sub_name = self.subdivisions[sub_id]["name"]

            # Choose decade
            decade = self.rng.choice([1910, 1920, 1930, 1940])

            # Question type: covenant count or deed list
            if i % 2 == 0:
                # Covenant count
                deed_ids = []
                for deed_id, deed in self.deeds.items():
                    if (deed["subdivision_id"] == sub_id and
                        deed.get("has_covenant") and
                        decade <= deed["signed_year"] < decade + 10):
                        deed_ids.append(deed_id)

                question = f"How many deeds with covenants were recorded in {sub_name} during the {decade}s?"

                q = BenchmarkQuestion(
                    question_id=f"q_l4_{i+1:03d}",
                    level="L4_spatiotemporal",
                    question=question,
                    ground_truth=deed_ids,  # Store IDs, evaluator extracts count
                    metadata={
                        "type": "covenant_count",
                        "temporal_constraints": {"start_year": decade, "end_year": decade + 9, "decade": decade},
                        "spatial_constraints": {"subdivision_id": sub_id, "subdivision_name": sub_name},
                        "required_hops": 2,
                        "expected_count": len(deed_ids)
                    }
                )
            else:
                # Deed list
                deed_ids = []
                for deed_id, deed in self.deeds.items():
                    if (deed["subdivision_id"] == sub_id and
                        decade <= deed["signed_year"] < decade + 10):
                        deed_ids.append(deed_id)

                question = f"List all deeds in {sub_name} from the {decade}s."

                q = BenchmarkQuestion(
                    question_id=f"q_l4_{i+1:03d}",
                    level="L4_spatiotemporal",
                    question=question,
                    ground_truth=deed_ids,
                    metadata={
                        "type": "subdivision_decade_list",
                        "temporal_constraints": {"start_year": decade, "end_year": decade + 9, "decade": decade},
                        "spatial_constraints": {"subdivision_id": sub_id, "subdivision_name": sub_name},
                        "required_hops": 2,
                        "expected_count": len(deed_ids)
                    }
                )

            questions.append(q)

        return questions

    def _generate_l5_questions(self, n: int) -> List[BenchmarkQuestion]:
        """
        L5: Conflict detection.

        Examples:
        - Identify deeds with date conflicts.
        - Which deeds have inconsistent review status?
        """
        questions = []

        # Find all deeds with conflicts
        date_conflicts = [d for d, deed in self.deeds.items() if deed.get("has_date_conflict")]
        review_conflicts = [d for d, deed in self.deeds.items() if deed.get("has_review_conflict")]
        all_conflicts = list(set(date_conflicts + review_conflicts))

        # Generate questions
        question_templates = [
            ("Identify all deeds with date conflicts (where recorded date is before signed date).", date_conflicts, "date_conflict"),
            ("Which deeds have inconsistent or conflicting review status?", review_conflicts, "review_conflict"),
            ("Find all deeds with any type of data inconsistency.", all_conflicts, "all_conflicts"),
            ("List deeds where the recorded date precedes the signed date.", date_conflicts, "date_conflict"),
            ("Identify deeds with annotation errors.", all_conflicts, "all_conflicts")
        ]

        for i in range(n):
            template_idx = i % len(question_templates)
            question_text, ground_truth, conflict_type = question_templates[template_idx]

            q = BenchmarkQuestion(
                question_id=f"q_l5_{i+1:03d}",
                level="L5_conflict",
                question=question_text,
                ground_truth=ground_truth,
                metadata={
                    "type": conflict_type,
                    "temporal_constraints": {},
                    "spatial_constraints": {},
                    "required_hops": 1,
                    "expected_count": len(ground_truth)
                }
            )
            questions.append(q)

        return questions

    def save(self, questions: Dict[str, List[BenchmarkQuestion]], path: str) -> None:
        """Save questions to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for level, q_list in questions.items():
            serializable[level] = [q.to_dict() for q in q_list]

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Dict[str, List[BenchmarkQuestion]]:
        """Load questions from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        questions = {}
        for level, q_list in data.items():
            questions[level] = [BenchmarkQuestion.from_dict(q) for q in q_list]

        return questions


if __name__ == "__main__":
    # Quick test
    from synthetic_generator import SyntheticDeedGenerator, GeneratorConfig

    config = GeneratorConfig(num_deeds=50, seed=42)
    gen = SyntheticDeedGenerator(config)
    data = gen.generate()

    q_gen = BenchmarkQuestionGenerator(data, seed=42)
    questions = q_gen.generate_all(questions_per_level=5)

    print("Generated questions:")
    for level, q_list in questions.items():
        print(f"\n{level}: {len(q_list)} questions")
        for q in q_list[:2]:
            print(f"  [{q.question_id}] {q.question}")
            print(f"    Ground truth: {len(q.ground_truth)} items")
