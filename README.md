# 🛒 Hybrid E-commerce Recommendation System with Automated MLOps Pipeline

An end-to-end Hybrid Recommendation Engine built with a robust **MLOps pipeline**, featuring automated data/concept drift detection, **human-in-the-loop retraining triggers**, and automated **model validation gates** before production deployment.

---

## 🌟 Key Features

* **Hybrid Recommendation Architecture:** Combines text embeddings (Content-based filtering using Sentence-Transformers) and user interaction history (Collaborative Filtering) into a unified ranking layer.
* **Vector Database Integration:** High-speed approximate nearest neighbor (ANN) retrieval using Qdrant/Milvus.
* **Feature Store:** Online and offline feature serving powered by Feast.
* **Data Drift & Performance Monitoring:** Continuous tracking of `NDCG@K`, `Precision@K`, and data distribution shifts via **Evidently AI** and Statistical KS Tests.
* **Orchestration & Retraining:** Automated pipeline execution using **Apache Airflow** with a **Human-in-the-Loop** approval trigger mechanism.
* **Model Validation Gate (Champion vs. Challenger):** Automated evaluation comparing candidate models against active production models before promotion in **MLflow Model Registry**.

---

## 🏗 System Architecture

```text
[User Logs/Interactions] ➔ [Feast Feature Store] ➔ [Hybrid Recommender API (FastAPI)]
                                                              │
                                                              ▼
                                                    [Monitoring Layer]
                                                  (NDCG & Data Drift Logs)
                                                              │
                                                              ▼
                                                   [Alerting / Slack Notification]
                                                              │
                                                              ▼
                                                 [Human Approval Gate]
                                                              │
                                                              ▼
                                                  [Airflow Pipeline Trigger]
                                                              │
                                                              ▼
                                                  [Retraining & Validation]
                                             (Challenger vs. Champion Gate)
                                                              │
                                                              ▼
                                                 [MLflow Production Registry]
```

---

## 🚀 MLOps & Retraining Workflow

1. **Monitoring:** System logs user interactions and evaluates offline metrics (`NDCG@K`, `Precision@K`) and online proxy metrics (`CTR`) alongside feature distribution drift using **Evidently AI**.
2. **Alerting & Approval:** If a drift or metric degradation crosses defined thresholds, an alert is dispatched. An engineer reviews the drift report and approves retraining.
3. **Execution:** Upon approval, **Apache Airflow** executes the retraining DAG.
4. **Validation Gate:** The newly trained *Challenger* model is benchmarked against the current *Champion* model on a validation hold-out set. Only models exceeding performance baselines are transitioned from `Staging` to `Production` within **MLflow Model Registry**.

---

## 🛠 Tech Stack

* **ML & Recommendation:** PyTorch, Scikit-Learn, Sentence-Transformers, Qdrant/Milvus
* **MLOps & Feature Store:** MLflow, Feast, Evidently AI, Apache Airflow
* **Serving & Deployment:** FastAPI, Docker, Docker Compose
