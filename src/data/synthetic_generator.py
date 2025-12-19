"""
Synthetic Deed Data Generator

Generates synthetic historical deed documents with:
- Spatio-temporal relationships (streets, subdivisions, dates)
- Covenant annotations
- Intentional data conflicts for L5 questions
- Deterministic output given fixed seed
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, timedelta
from pathlib import Path
import yaml


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""

    # Scale
    num_deeds: int = 100
    num_subdivisions: int = 5
    streets_per_subdivision: int = 15

    # Temporal range
    start_year: int = 1910
    end_year: int = 1950
    peak_years: List[int] = field(default_factory=lambda: [1924, 1926, 1942])

    # Spatial
    street_overlap_rate: float = 0.6

    # Conflicts
    date_conflict_rate: float = 0.07
    review_conflict_rate: float = 0.10

    # Text style for conversion
    text_style: str = "mixed"
    text_style_ratio: float = 0.6

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def load_config_from_yaml(path: str) -> GeneratorConfig:
    """Load generator config from YAML file."""
    with open(path, 'r') as f:
        config_data = yaml.safe_load(f)

    data_config = config_data.get('data', {})
    temporal = data_config.get('temporal_range', {})

    return GeneratorConfig(
        num_deeds=data_config.get('num_deeds', 100),
        num_subdivisions=data_config.get('num_subdivisions', 5),
        streets_per_subdivision=data_config.get('streets_per_subdivision', 15),
        start_year=temporal.get('start', 1910),
        end_year=temporal.get('end', 1950),
        peak_years=data_config.get('peak_years', [1924, 1926, 1942]),
        street_overlap_rate=data_config.get('street_overlap_rate', 0.6),
        date_conflict_rate=data_config.get('date_conflict_rate', 0.07),
        review_conflict_rate=data_config.get('review_conflict_rate', 0.10),
        text_style=data_config.get('text_style', 'mixed'),
        text_style_ratio=data_config.get('text_style_ratio', 0.6),
        seed=data_config.get('seed', 42)
    )


# Name generation data
FIRST_NAMES = [
    "John", "William", "James", "Robert", "Charles", "George", "Thomas", "Henry",
    "Edward", "Joseph", "Mary", "Anna", "Emma", "Elizabeth", "Margaret", "Alice",
    "Sarah", "Helen", "Dorothy", "Ruth", "Frank", "Arthur", "Walter", "Frederick",
    "Samuel", "Harold", "Raymond", "Albert", "Louis", "Carl", "Martha", "Florence",
    "Lillian", "Grace", "Louise", "Edith", "Annie", "Mildred", "Gertrude", "Hazel"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson",
    "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson",
    "Moore", "Young", "Allen", "King", "Wright", "Scott", "Green", "Baker", "Adams",
    "Nelson", "Hill", "Campbell", "Mitchell", "Roberts", "Carter", "Phillips", "Evans",
    "Turner", "Torres", "Parker", "Collins", "Edwards", "Stewart", "Morris", "Murphy"
]

STREET_PREFIXES = [
    "Oak", "Maple", "Pine", "Cedar", "Elm", "Birch", "Walnut", "Cherry", "Spruce",
    "Willow", "Ash", "Hickory", "Sycamore", "Chestnut", "Laurel", "Magnolia", "Holly",
    "Poplar", "Beech", "Cypress", "Main", "High", "Park", "Lake", "River", "Hill",
    "Valley", "Forest", "Garden", "Sunset", "Spring", "Summer", "Autumn", "Winter"
]

STREET_SUFFIXES = ["Street", "Avenue", "Road", "Lane", "Drive", "Boulevard", "Way", "Place"]

SUBDIVISION_PREFIXES = [
    "Pine", "Oak", "Maple", "Cedar", "Willow", "Green", "Pleasant", "Sunny", "Lake",
    "River", "Mountain", "Valley", "Rolling", "Highland", "Meadow", "Spring", "Forest"
]

SUBDIVISION_SUFFIXES = ["Valley", "Heights", "Hills", "Acres", "Park", "Estates", "Manor", "Grove"]

COVENANT_TEMPLATES = [
    "No lot shall be sold, conveyed, or rented to any person not of the Caucasian race.",
    "The premises herein conveyed shall not be sold, leased, or rented to any person other than of the white race.",
    "Said property shall never be sold, rented, or occupied by any person of African descent.",
    "This property shall not be used or occupied by any person except those of the Caucasian race.",
    "No part of said premises shall ever be sold, conveyed, or leased to any Negro or person of African descent."
]


class SyntheticDeedGenerator:
    """
    Generates synthetic historical deed documents.

    Features:
    - Deterministic given seed
    - Configurable temporal distribution
    - Street overlap for multi-hop queries
    - Intentional conflicts for L5 detection
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = random.Random(config.seed)

        # Generated data
        self.subdivisions: Dict[str, Dict] = {}
        self.streets: Dict[str, Dict] = {}
        self.persons: Dict[str, Dict] = {}
        self.deeds: Dict[str, Dict] = {}

        # Ground truth caches
        self._ground_truth: Optional[Dict] = None

    def generate(self) -> Dict[str, Any]:
        """
        Generate all synthetic data.

        Returns:
            Dict with keys: deeds, subdivisions, streets, persons, metadata
        """
        # Reset state
        self.subdivisions = {}
        self.streets = {}
        self.persons = {}
        self.deeds = {}
        self._ground_truth = None

        # Generate in order
        self._generate_subdivisions()
        self._generate_streets()
        self._generate_persons()
        self._generate_deeds()

        return {
            "deeds": self.deeds,
            "subdivisions": self.subdivisions,
            "streets": self.streets,
            "persons": self.persons,
            "metadata": {
                "config": self.config.to_dict(),
                "stats": {
                    "num_deeds": len(self.deeds),
                    "num_subdivisions": len(self.subdivisions),
                    "num_streets": len(self.streets),
                    "num_persons": len(self.persons),
                    "deeds_with_covenants": sum(1 for d in self.deeds.values() if d.get("has_covenant")),
                    "deeds_with_date_conflict": sum(1 for d in self.deeds.values() if d.get("has_date_conflict")),
                    "deeds_with_review_conflict": sum(1 for d in self.deeds.values() if d.get("has_review_conflict"))
                }
            }
        }

    def _generate_subdivisions(self) -> None:
        """Generate subdivision records."""
        used_names = set()

        for i in range(self.config.num_subdivisions):
            # Generate unique name
            while True:
                prefix = self.rng.choice(SUBDIVISION_PREFIXES)
                suffix = self.rng.choice(SUBDIVISION_SUFFIXES)
                name = f"{prefix} {suffix}"
                if name not in used_names:
                    used_names.add(name)
                    break

            sub_id = f"sub_{i+1}"

            # Generate plat book info
            plat_book = str(self.rng.randint(1, 50))
            plat_page = str(self.rng.randint(1, 200))

            # Recorded year
            recorded_year = self.rng.randint(
                self.config.start_year - 5,
                self.config.start_year + 10
            )

            self.subdivisions[sub_id] = {
                "subdivision_id": sub_id,
                "name": name,
                "plat_book": plat_book,
                "plat_page": plat_page,
                "recorded_year": recorded_year,
                "town": "Sample Town",
                "county": "Sample County"
            }

    def _generate_streets(self) -> None:
        """Generate street records within subdivisions."""
        used_names: Dict[str, set] = {sub_id: set() for sub_id in self.subdivisions}

        for sub_id in self.subdivisions:
            for j in range(self.config.streets_per_subdivision):
                # Generate unique street name within subdivision
                while True:
                    prefix = self.rng.choice(STREET_PREFIXES)
                    suffix = self.rng.choice(STREET_SUFFIXES)
                    name = f"{prefix} {suffix}"
                    if name not in used_names[sub_id]:
                        used_names[sub_id].add(name)
                        break

                street_id = f"street_{prefix.lower()}_{sub_id}"
                if street_id in self.streets:
                    street_id = f"{street_id}_{j}"

                self.streets[street_id] = {
                    "street_id": street_id,
                    "name": name,
                    "subdivision_id": sub_id,
                    "subdivision_name": self.subdivisions[sub_id]["name"]
                }

    def _generate_persons(self) -> None:
        """Generate person records."""
        # Generate enough persons for all deeds (2 per deed)
        num_persons = max(self.config.num_deeds * 2, 100)

        for i in range(num_persons):
            first = self.rng.choice(FIRST_NAMES)
            last = self.rng.choice(LAST_NAMES)

            person_id = f"person_{i+1}"
            self.persons[person_id] = {
                "person_id": person_id,
                "name": f"{first} {last}",
                "first_name": first,
                "last_name": last
            }

    def _generate_deeds(self) -> None:
        """Generate deed records."""
        person_ids = list(self.persons.keys())
        street_ids = list(self.streets.keys())

        # Build street pool by subdivision
        streets_by_sub: Dict[str, List[str]] = {}
        for street_id, street in self.streets.items():
            sub_id = street["subdivision_id"]
            if sub_id not in streets_by_sub:
                streets_by_sub[sub_id] = []
            streets_by_sub[sub_id].append(street_id)

        # Track used streets for overlap
        street_usage: Dict[str, int] = {s: 0 for s in street_ids}

        for i in range(self.config.num_deeds):
            deed_id = f"deed_{i+1:04d}"

            # Select subdivision
            sub_id = self.rng.choice(list(self.subdivisions.keys()))
            sub = self.subdivisions[sub_id]

            # Select street (with overlap preference)
            sub_streets = streets_by_sub[sub_id]
            if self.rng.random() < self.config.street_overlap_rate and sub_streets:
                # Prefer already-used streets
                used_sub_streets = [s for s in sub_streets if street_usage[s] > 0]
                if used_sub_streets:
                    street_id = self.rng.choice(used_sub_streets)
                else:
                    street_id = self.rng.choice(sub_streets)
            else:
                street_id = self.rng.choice(sub_streets)

            street_usage[street_id] += 1
            street = self.streets[street_id]

            # Select parties
            grantor_id = self.rng.choice(person_ids)
            grantee_id = self.rng.choice([p for p in person_ids if p != grantor_id])

            # Generate dates
            signed_date, signed_year = self._generate_date()

            # Recorded date (typically days to weeks after signing)
            days_until_recorded = self.rng.randint(1, 30)
            recorded_date = (
                date.fromisoformat(signed_date) + timedelta(days=days_until_recorded)
            ).isoformat()

            # Covenant (about 40% have covenants)
            has_covenant = self.rng.random() < 0.4
            covenant_text = self.rng.choice(COVENANT_TEMPLATES) if has_covenant else None

            # Conflicts for L5 questions
            has_date_conflict = self.rng.random() < self.config.date_conflict_rate
            has_review_conflict = self.rng.random() < self.config.review_conflict_rate

            # If date conflict, make recorded date before signed date
            if has_date_conflict:
                days_before = self.rng.randint(5, 30)
                recorded_date = (
                    date.fromisoformat(signed_date) - timedelta(days=days_before)
                ).isoformat()

            # Review status
            if has_review_conflict:
                review_status = "conflict"  # Marked as having issues
            else:
                review_status = self.rng.choice(["approved", "pending", "reviewed"])

            # Plan book info
            plan_book = str(self.rng.randint(1, 100))
            plan_page = str(self.rng.randint(1, 500))

            self.deeds[deed_id] = {
                "deed_id": deed_id,
                "signed_date": signed_date,
                "signed_year": signed_year,
                "recorded_date": recorded_date,
                "street_id": street_id,
                "street_name": street["name"],
                "subdivision_id": sub_id,
                "subdivision_name": sub["name"],
                "grantor_id": grantor_id,
                "grantor_name": self.persons[grantor_id]["name"],
                "grantee_id": grantee_id,
                "grantee_name": self.persons[grantee_id]["name"],
                "has_covenant": has_covenant,
                "covenant_text": covenant_text,
                "plan_book": plan_book,
                "plan_page": plan_page,
                "review_status": review_status,
                "has_date_conflict": has_date_conflict,
                "has_review_conflict": has_review_conflict
            }

    def _generate_date(self) -> Tuple[str, int]:
        """Generate a date within the temporal range, biased toward peak years."""
        # Choose year (biased toward peak years)
        if self.rng.random() < 0.3 and self.config.peak_years:
            year = self.rng.choice(self.config.peak_years)
        else:
            year = self.rng.randint(self.config.start_year, self.config.end_year)

        # Choose month and day
        month = self.rng.randint(1, 12)

        # Handle days per month
        if month in [4, 6, 9, 11]:
            max_day = 30
        elif month == 2:
            max_day = 28  # Simplify, ignore leap years
        else:
            max_day = 31

        day = self.rng.randint(1, max_day)

        date_str = f"{year}-{month:02d}-{day:02d}"
        return date_str, year

    def get_ground_truth(self) -> Dict[str, Any]:
        """
        Compute ground truth answers for benchmark questions.

        Returns:
            Dict with precomputed answers organized by query type
        """
        if self._ground_truth is not None:
            return self._ground_truth

        gt: Dict[str, Any] = {
            "by_year": {},
            "by_year_range": {},
            "by_subdivision": {},
            "by_street": {},
            "covenants_by_subdivision": {},
            "covenants_by_subdivision_decade": {},
            "street_neighbors": {},
            "conflicts": {
                "date": [],
                "review": [],
                "all": []
            }
        }

        # Index deeds by year
        for deed_id, deed in self.deeds.items():
            year = deed["signed_year"]
            if year not in gt["by_year"]:
                gt["by_year"][year] = []
            gt["by_year"][year].append(deed_id)

        # Index by year range (decades)
        for decade_start in range(1910, 1960, 10):
            decade_end = decade_start + 9
            key = f"{decade_start}-{decade_end}"
            gt["by_year_range"][key] = []
            for deed_id, deed in self.deeds.items():
                if decade_start <= deed["signed_year"] <= decade_end:
                    gt["by_year_range"][key].append(deed_id)

        # Index by subdivision
        for deed_id, deed in self.deeds.items():
            sub_id = deed["subdivision_id"]
            if sub_id not in gt["by_subdivision"]:
                gt["by_subdivision"][sub_id] = []
            gt["by_subdivision"][sub_id].append(deed_id)

        # Index by street
        for deed_id, deed in self.deeds.items():
            street_id = deed["street_id"]
            if street_id not in gt["by_street"]:
                gt["by_street"][street_id] = []
            gt["by_street"][street_id].append(deed_id)

        # Covenants by subdivision
        for deed_id, deed in self.deeds.items():
            if deed["has_covenant"]:
                sub_id = deed["subdivision_id"]
                if sub_id not in gt["covenants_by_subdivision"]:
                    gt["covenants_by_subdivision"][sub_id] = []
                gt["covenants_by_subdivision"][sub_id].append(deed_id)

        # Covenants by subdivision + decade
        for deed_id, deed in self.deeds.items():
            if deed["has_covenant"]:
                sub_id = deed["subdivision_id"]
                decade = (deed["signed_year"] // 10) * 10
                key = f"{sub_id}_{decade}"
                if key not in gt["covenants_by_subdivision_decade"]:
                    gt["covenants_by_subdivision_decade"][key] = []
                gt["covenants_by_subdivision_decade"][key].append(deed_id)

        # Street neighbors (deeds sharing same street)
        for deed_id, deed in self.deeds.items():
            street_id = deed["street_id"]
            neighbors = [d for d in gt["by_street"].get(street_id, []) if d != deed_id]
            gt["street_neighbors"][deed_id] = neighbors

        # Conflicts
        for deed_id, deed in self.deeds.items():
            if deed["has_date_conflict"]:
                gt["conflicts"]["date"].append(deed_id)
                gt["conflicts"]["all"].append(deed_id)
            if deed["has_review_conflict"]:
                gt["conflicts"]["review"].append(deed_id)
                if deed_id not in gt["conflicts"]["all"]:
                    gt["conflicts"]["all"].append(deed_id)

        self._ground_truth = gt
        return gt


if __name__ == "__main__":
    # Quick test
    config = GeneratorConfig(num_deeds=20, seed=42)
    generator = SyntheticDeedGenerator(config)
    data = generator.generate()

    print(f"Generated:")
    print(f"  {len(data['deeds'])} deeds")
    print(f"  {len(data['subdivisions'])} subdivisions")
    print(f"  {len(data['streets'])} streets")
    print(f"  {len(data['persons'])} persons")
    print(f"\nStats: {json.dumps(data['metadata']['stats'], indent=2)}")

    # Show sample deed
    sample = list(data['deeds'].values())[0]
    print(f"\nSample deed: {json.dumps(sample, indent=2)}")

    # Ground truth
    gt = generator.get_ground_truth()
    print(f"\nGround truth keys: {list(gt.keys())}")
    print(f"Deeds with date conflicts: {gt['conflicts']['date']}")
