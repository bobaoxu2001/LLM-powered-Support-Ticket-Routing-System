# Eval Set Expansion — Status and Plan

## Current status (2026-04-24)

`data/eval/eval_tickets.csv` has been expanded to **399 rows** (6 issue types) using `scripts/build_eval_set.py`:

| Source | Rows | Label origin |
|---|---|---|
| Manually written | 99 | Hand-authored representative examples |
| Kaggle metadata (suraj520) | 300 | `Ticket Type` structured field → issue_type mapping |

**Important caveat — Kaggle dataset quality**: the `suraj520/customer-support-ticket-dataset` is a synthetic dataset. All 8,469 Ticket Descriptions contain a `{product_purchased}` placeholder that was never substituted. Ticket Subject and Ticket Type are randomly assigned and do not reflect actual description content. The 300 metadata-derived rows therefore have **limited text-label alignment**. Treat results on these rows as metadata-labeled evaluation, not a manually adjudicated gold standard.

Classes `ads` and `login` have only 17 rows each (manually written) because the Kaggle dataset contains no tickets mapping to those types.

## Original expansion plan (toward 500 rows)

This plan describes how to expand responsibly without fabricating labels.


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
