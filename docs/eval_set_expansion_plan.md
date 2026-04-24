# Eval Set Expansion Plan (toward 500 rows)

Current `data/eval/eval_tickets.csv` has balanced classes but limited size. In this environment, expanding safely to 300–500 rows is not feasible without ingesting additional structured-ticket rows and reviewing label quality.

This plan describes how to expand responsibly without fabricating labels.

Current snapshot (2026-04-24): `data/eval/eval_tickets.csv` has 99 rows with near-balanced classes (16–17 each), which is useful but still small for robust policy tuning.

## 1) Sampling strategy
1. Load structured ticket dataset rows with `Ticket Subject`, `Ticket Description`, `Ticket Type`, and `Ticket Priority`.
2. Map `Ticket Type` into project issue taxonomy using the same mapping logic used in pipeline code.
3. Keep only rows with high-confidence mapped issue labels.
4. Deduplicate on normalized `subject + description` text.
5. Stratified-sample by issue_type to target 500 rows total.

## 2) Label taxonomy
Allowed `issue_type` labels only:
- `billing`, `ads`, `login`, `technical`, `account`, `other`

Rows outside mapping confidence should be excluded or moved to review queue.

## 3) Review workflow
- **Pass 1 (metadata-derived auto-label)**: generate candidate labels from mapped `Ticket Type`.
- **Pass 2 (spot-check review)**: manually inspect at least 20% per class.
- **Pass 3 (adjudication)**: resolve uncertain or ambiguous rows.

Any manually touched labels should be marked with provenance metadata (e.g., `label_source=manual_review`).

## 4) Quality checks
- No missing text field (`text` or `subject/description`).
- No label outside taxonomy.
- Duplicate-rate check after normalization.
- Per-class minimum count threshold.
- Leakage check against keyword-rule triggers (see below).

## 5) Target class distribution
For 500 rows, initial target:
- 70–90 rows for each of six classes.

If a class is naturally sparse (e.g., `ads`), document shortfall explicitly instead of forcing synthetic examples.

## 6) Leakage avoidance (critical)
To avoid overestimating ML lift over keyword baseline:
- Flag rows where label is trivially recoverable via exact keyword rules.
- Limit over-representation of these easy rows in final eval sample.
- Include ambiguous/noisy rows so baseline-vs-ML comparison remains meaningful.

## 7) Output format
Expanded eval CSV should remain compatible with `evaluate_on_labeled_set()`:
- required: `issue_type`
- and either `text` or (`subject`, `description`)

Optional provenance columns are recommended for transparency.
