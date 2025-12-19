# Data Schema Documentation

This document describes the data structures used in the STRAG thesis experiments.

## Generated Files

After running `01_generate_data.py`, the following files are created:

| File | Description |
|------|-------------|
| `synthetic_deeds.json` | Structured deed records |
| `ground_truth.json` | Precomputed answers for benchmarks |
| `benchmark_questions.json` | Evaluation questions (L1-L5) |
| `deeds_text/` | Individual text files per deed |
| `deeds_combined.txt` | All deeds in single file |

## Deed Record Schema

Each deed in `synthetic_deeds.json` has the following structure:

```json
{
  "deed_id": "deed_0001",

  "signed_date": "1924-03-15",
  "signed_year": 1924,
  "recorded_date": "1924-03-20",

  "street_id": "street_oak_sub_1",
  "street_name": "Oak Street",
  "subdivision_id": "sub_1",
  "subdivision_name": "Pine Valley",

  "grantor_id": "person_1",
  "grantor_name": "John Smith",
  "grantee_id": "person_2",
  "grantee_name": "Mary Johnson",

  "has_covenant": true,
  "covenant_text": "No lot shall be sold to any person not of the Caucasian race.",

  "plan_book": "123",
  "plan_page": "45",
  "review_status": "approved",

  "has_date_conflict": false,
  "has_review_conflict": false
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `deed_id` | string | Unique identifier (format: `deed_XXXX`) |
| `signed_date` | string | ISO date when deed was signed |
| `signed_year` | int | Year extracted from signed_date |
| `recorded_date` | string | ISO date when deed was recorded |
| `street_id` | string | Reference to street record |
| `street_name` | string | Human-readable street name |
| `subdivision_id` | string | Reference to subdivision record |
| `subdivision_name` | string | Human-readable subdivision name |
| `grantor_id` | string | Reference to person (seller) |
| `grantor_name` | string | Full name of grantor |
| `grantee_id` | string | Reference to person (buyer) |
| `grantee_name` | string | Full name of grantee |
| `has_covenant` | bool | Whether deed contains restrictive covenant |
| `covenant_text` | string | Text of covenant (null if none) |
| `plan_book` | string | Plan book reference number |
| `plan_page` | string | Page number in plan book |
| `review_status` | string | "approved", "pending", "reviewed", or "conflict" |
| `has_date_conflict` | bool | True if recorded_date < signed_date (data error) |
| `has_review_conflict` | bool | True if review_status is "conflict" |

## Subdivision Schema

```json
{
  "subdivision_id": "sub_1",
  "name": "Pine Valley",
  "plat_book": "12",
  "plat_page": "45",
  "recorded_year": 1908,
  "town": "Sample Town",
  "county": "Sample County"
}
```

## Street Schema

```json
{
  "street_id": "street_oak_sub_1",
  "name": "Oak Street",
  "subdivision_id": "sub_1",
  "subdivision_name": "Pine Valley"
}
```

## Person Schema

```json
{
  "person_id": "person_1",
  "name": "John Smith",
  "first_name": "John",
  "last_name": "Smith"
}
```

## Benchmark Question Schema

Questions in `benchmark_questions.json`:

```json
{
  "question_id": "q_l1_001",
  "level": "L1_single_hop",
  "question": "Find all deeds recorded in 1924.",
  "ground_truth": ["deed_0001", "deed_0005", "deed_0008"],
  "metadata": {
    "type": "year_lookup",
    "temporal_constraints": {"specific_year": 1924},
    "spatial_constraints": {},
    "required_hops": 1,
    "expected_count": 3
  }
}
```

### Question Levels

| Level | Type | Description |
|-------|------|-------------|
| `L1_single_hop` | Single entity lookup | Direct attribute matching |
| `L2_temporal` | Temporal reasoning | Year ranges, before/after, decades |
| `L3_spatial_multihop` | Spatial traversal | Finding neighbors via shared streets |
| `L4_spatiotemporal` | Combined constraints | Subdivision + decade filtering |
| `L5_conflict` | Conflict detection | Finding data inconsistencies |

### Metadata Fields

| Field | Description |
|-------|-------------|
| `type` | Specific question type (year_lookup, year_range, street_neighbors, etc.) |
| `temporal_constraints` | Dictionary with start_year, end_year, specific_year, decade |
| `spatial_constraints` | Dictionary with subdivision_id, subdivision_name, street_name |
| `required_hops` | Number of graph traversals needed (1 for direct, 2+ for multi-hop) |
| `expected_count` | Number of items in ground truth |

## Ground Truth Schema

`ground_truth.json` contains precomputed indices:

```json
{
  "by_year": {
    "1924": ["deed_0001", "deed_0005"],
    "1926": ["deed_0002", "deed_0003"]
  },
  "by_year_range": {
    "1920-1929": ["deed_0001", "deed_0002", ...]
  },
  "by_subdivision": {
    "sub_1": ["deed_0001", "deed_0004", ...]
  },
  "by_street": {
    "street_oak_sub_1": ["deed_0001", "deed_0003"]
  },
  "covenants_by_subdivision": {
    "sub_1": ["deed_0001", "deed_0004"]
  },
  "covenants_by_subdivision_decade": {
    "sub_1_1920": ["deed_0001"]
  },
  "street_neighbors": {
    "deed_0001": ["deed_0003", "deed_0007"]
  },
  "conflicts": {
    "date": ["deed_0010"],
    "review": ["deed_0005", "deed_0012"],
    "all": ["deed_0005", "deed_0010", "deed_0012"]
  }
}
```

## Text Document Format

Individual text files in `deeds_text/` use either concise or narrative format:

**Concise format:**
```
DEED: deed_0001
Signed: 1924-03-15 | Recorded: 1924-03-20
Location: Oak Street, Pine Valley
From: John Smith To: Mary Johnson
Plan Book: 123, Page: 45
COVENANT: No lot shall be sold to any person not of the Caucasian race.
Review Status: approved
```

**Narrative format:**
```
Deed deed_0001: On March 15, 1924, John Smith conveyed property located at
Oak Street in the Pine Valley subdivision to Mary Johnson. The deed was
officially recorded on March 20, 1924 in Plan Book 123, Page 45. This deed
contains a racial restrictive covenant stating: "No lot shall be sold to any
person not of the Caucasian race." Review status: approved.
```

## Configuration Reference

See `configs/experiment_config.yaml` for adjustable parameters:

```yaml
data:
  num_deeds: 100
  num_subdivisions: 5
  temporal_range: {start: 1910, end: 1950}
  date_conflict_rate: 0.07
  review_conflict_rate: 0.10
  seed: 42

benchmark:
  questions_per_level: 10
```
