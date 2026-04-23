# LLM-powered Support Ticket Routing System

Author: **Allen Xu**  
Language: **English**

This project is an end-to-end **Support Operations System** (not a chatbot), combining:

1. Real customer-support conversations (Twitter)
2. Large-scale structured support tickets (200K+)
3. LLM-powered summarization and reasoning for routing fallback

It implements a production-style multi-stage routing design:

```text
Rule-based (high confidence patterns)
      ↓
ML classifier (default routing)
      ↓
LLM reasoning (low-confidence cases)
      ↓
Human fallback
```

---

## 1) Datasets

### A. Customer Support on Twitter (real interactions)
- Kaggle slug used in code: `thoughtvector/customer-support-on-twitter`
- Purpose: real-world conversational language, noisy text, social support patterns

### B. Customer Support Tickets Dataset (200K+)
- Kaggle slug used in code: `suraj520/customer-support-ticket-dataset`
- Purpose: large-scale structured tickets for classification/routing training

### C. TWEETSUMM (for conversation summarization)
- You can add this as a third source in the same ingestion pattern.
- In this implementation, summarization is handled by LLM functions (`llm.py`) so you can plug TWEETSUMM in supervised fine-tuning later.

> Note: Kaggle API credentials are required to auto-download datasets.

---

## 2) System Architecture

### Core modules
- `src/llm_support_routing/data.py`  
  Data download, loading, and unified table construction.

- `src/llm_support_routing/features.py`  
  Text normalization + weak-label generation for:
  - issue type (`billing`, `ads`, `login`, `technical`, `account`, `other`)
  - urgency (`low`, `medium`, `high`, `critical`)
  - complexity (`low`, `medium`, `high`)

- `src/llm_support_routing/models.py`  
  Baseline classifier:
  - **TF-IDF + Logistic Regression**

- `src/llm_support_routing/routing.py`  
  Main routing logic with four stages:
  - rule-based
  - ML high-confidence
  - LLM reasoning (low confidence)
  - human fallback

- `src/llm_support_routing/llm.py`  
  LLM tasks:
  - few-shot style classification
  - ticket summarization
  - resolution path + escalation judgement

- `src/llm_support_routing/evaluation.py`  
  KPIs:
  - escalation rate
  - LLM invocation rate
  - estimated cost per ticket

- `scripts/run_pipeline.py`  
  End-to-end pipeline runner.

- `app.py`  
  Streamlit dashboard:
  - routing flow chart
  - route distribution
  - confidence distribution
  - key metrics

---

## 3) Environment Setup

## Option A: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Kaggle API (for dataset download)
Set credentials:
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

### OpenAI API (for LLM fallback/summarization)
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

---

## 4) Run End-to-End Pipeline

### Download + train + route + evaluate
```bash
python scripts/run_pipeline.py --download
```

Generated artifacts:
- `data/processed/unified_labeled_tickets.csv`
- `models/issue_type_tfidf_lr.joblib`
- `outputs/routed_tickets.csv`
- `outputs/routing_metrics.csv`
- `outputs/training_report.txt`

---

## 5) Launch Dashboard

```bash
streamlit run app.py
```

Dashboard includes:
- routing flow visualization (stage → queue)
- accuracy-related proxy outputs (via saved report + routing metrics)
- escalation rate
- cost per ticket

---

## 6) Baseline, Advanced, and LLM Layers

### Baseline (implemented)
- TF-IDF + Logistic Regression (issue type)

### Advanced (extension plan)
- DistilBERT / BERT fine-tuning for issue type, urgency, complexity
- Replace/augment weak labels with human annotations

### LLM layer (implemented)
- Low-confidence ticket reasoning
- Automatic ticket summary
- Suggested resolution path + escalation decision

---

## 7) Production Notes

- Keep confidence thresholds configurable (`RoutingThresholds` in `config.py`)
- Add monitoring by queue, language, and customer segment
- Add human-in-the-loop review for LLM outputs
- Validate synthetic-trained components on real ticket traffic before production rollout

---

## 8) Suggested Next Steps

1. Add TWEETSUMM ingestion and ROUGE/BERTScore evaluation for summarization.
2. Train separate classifiers for urgency and complexity.
3. Add calibration curves and confidence-based SLA policies.
4. Add per-queue capacity-aware routing (workload balancing).
5. Add model registry and CI/CD for retraining.

---

## 9) Quick Demo Without Download

If Kaggle API is unavailable, place CSVs manually:
- `data/raw/twitter_support/*.csv`
- `data/raw/support_tickets/*.csv`

Then run:
```bash
python scripts/run_pipeline.py
```

