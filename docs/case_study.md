# Case Study: Support Operations Ticket Routing (Portfolio Prototype)

## 1) Business Problem
Support teams need to route incoming cases quickly and accurately so customers reach the right specialist queue with minimal delay. In practice, teams must balance three competing objectives:
- routing quality,
- manual review load,
- and LLM usage cost.

This project simulates a **gTech / Google Ads-style support operations workflow** using **public support datasets only**. It does **not** use proprietary Google data.

## 2) Why Not LLM-only?
An LLM-only design is flexible, but it is not always the most reliable or cost-efficient operating policy.

- **Rules** are cheapest and safest for deterministic patterns.
- **Calibrated ML** is cheaper and scalable for frequent, repeatable patterns.
- **LLM classification** is most useful for low-confidence ambiguity.
- **Human triage** protects the system against uncertain predictions and failure modes.

The result is a selective-LLM architecture rather than an all-LLM architecture.

## 3) System Design
The routing cascade is:
1. **Rule-based exact patterns**
2. **Calibrated TF-IDF + Logistic Regression** for high-confidence auto-routing
3. **LLM classification** for low-confidence cases
4. **Human triage** for middle-confidence uncertainty or failed LLM calls

For human-fallback tickets, optional enrichment hooks exist for summary, suggested path, escalation flag, and reason.

## 4) Data Strategy
- **Customer Support on Twitter** contributes noisy, real customer-authored language.
- **Structured support ticket data** contributes `Ticket Type` / `Ticket Priority` metadata.
- Twitter messages are filtered to **inbound customer-authored** content.
- Label provenance is preserved (`real` vs `weak`).
- Issue/urgency labels can come from real metadata mapping or weak fallback logic.
- **Complexity is a word-count heuristic proxy**, not semantic ground truth.

## 5) Evaluation Strategy
Evaluation is designed to show incremental signal over simple heuristics:
- ML vs keyword baseline comparison
- accuracy, macro-F1, weighted-F1
- per-class metrics
- confusion matrix outputs
- threshold sweep for policy analysis
- explicit separation of **measured vs estimated vs proxy** metrics

## 6) Operational Tradeoffs
This system is intended for policy discussion, not one-click automation claims:
- auto-route coverage
- LLM invocation rate
- human-triage/manual-review rate
- estimated LLM cost per ticket
- threshold recommendation as an **analytic cost–coverage policy guide** (not “optimal,” not automatically enforced)

## 7) Limitations
- Data is public support data, not Google Ads proprietary data.
- The eval set (399 tickets) uses metadata-derived labels: 99 manually-written rows and 300 rows labeled via Ticket Type metadata from the suraj520 Kaggle dataset (a synthetic dataset with template descriptions and random label assignments). Labels are **not** manually adjudicated gold labels. The keyword baseline currently matches or slightly outperforms ML on this set, which is expected given the limited text-label alignment in the Kaggle-derived rows.
- No true production AHT or downstream escalation outcome tracking in this repo.
- Complexity is heuristic, not annotated semantic truth.
- Threshold recommendation currently is not accuracy-aware by queue.
- LLM outputs still require monitoring and human review.

## 8) Next Steps
1. Human adjudication of the metadata-derived eval labels — the current 300 Kaggle-sourced rows use Ticket Type as the label, which has poor text-label alignment in this synthetic dataset. A manually reviewed gold set (500–2,000 rows) would yield more reliable per-class F1 estimates.
2. Add queue-specific SLA-aware threshold policies.
3. Add reliability/calibration plots.
4. Benchmark latency across routing stages.
5. Persist richer model/run metadata for reproducibility.
