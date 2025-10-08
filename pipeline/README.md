# Real Data Acquisition Pipeline (2025)

This folder provides a scaffold to collect REAL, CURRENT (2024-2025) data for Indian engineering colleges without fabricating values. Replace placeholder steps with actual extraction runs you perform manually or via scripts.

## Core Principles
- No synthetic/fabricated values committed.
- Every datum must have: `raw_value`, `extracted_at (UTC)`, `source_url`, optional `evidence_snippet`.
- Raw capture -> normalization -> validation -> consolidation -> quarterly snapshotting.

## Folders
```
raw/                 # Immutable raw JSON/HTML/text captures (timestamped)
extracted/           # Parsed structured atomic records (1 row per fact)
normalized/          # Cleaned & canonical field names/units
consolidated/        # Joined multi-category JSON objects per college
snapshots/           # Quarterly frozen datasets (Q1,Q2,Q3,Q4 2025)
logs/                # Harvest logs
scripts/             # Python scripts for scraping/parsing
schemas/             # JSON Schemas for validation
validation_reports/  # JSON reports from schema + cross-field checks
```

## High-Level Flow
1. discover_sources.py -> generates `sources_catalog.json`
2. fetch_pages.py -> stores raw responses under `raw/YYYYMMDD_hhmmss/`
3. parse_raw.py -> outputs atomic facts into `extracted/`
4. normalize.py -> harmonizes fields -> `normalized/`
5. consolidate.py -> builds category JSON -> `consolidated/{college}.json`
6. validate.py -> schema + referential checks -> `validation_reports/`
7. snapshot.py -> produce quarterly archive -> `snapshots/2025_Q1.json`

## Fact Record Example
```json
{
  "college_code": "KARE",
  "category": "nirf_ranking",
  "metric": "nirf_overall_rank",
  "raw_value": 74,
  "year": 2024,
  "extracted_at": "2025-01-15T09:12:44Z",
  "source_url": "https://nirfindia.org/2024/EngineeringRanking.html",
  "evidence_snippet": "74 Kalasalingam Academy of Research and Education Tamil Nadu",
  "confidence": 0.98,
  "parser_version": "1.0.0"
}
```

## Quarterly Refresh
Run `refresh_all.ps1` (to be added) which chains steps 1-6, then updates snapshot if quarter boundary.

## DO NOT add actual scraped proprietary HTML here if licensing forbids. Store only permissible structured facts.

## Next Steps
- Implement scripts (see `scripts/` stubs to be created).
- Define JSON Schemas in `schemas/`.
- Add PowerShell orchestrator for Windows environment.
