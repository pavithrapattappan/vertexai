# MLOps Clustering Pipeline

A production-style machine learning pipeline for training, validating,
versioning, registering, and consuming segment-level clustering models.

The project integrates **Snowflake** for data processing with
**Google Cloud Vertex AI** for pipeline orchestration and model lifecycle
management.

The focus of this repository is not just model training. It implements
a controlled ML workflow with data validation, feature validation, model
quality gates, versioned artifacts, model registration, inference,
explainability, and drift monitoring.

---

## Overview

The pipeline processes customer data from Snowflake and trains clustering
models for individual business segments or industries.

A typical training run follows this lifecycle:

```text
Snowflake
    |
    v
Connection Check
    |
    v
Feature Extraction
    |
    v
Data Quality Validation
    |
    v
Preprocessing
    |
    v
Feature Validation
    |
    +-------------------+
    |                   |
   FAIL              OK / WARN
    |                   |
    v                   v
   STOP           Train Clusters
                         |
                         v
                  Model Artifacts
                         |
                         v
                  Model Validation
                         |
                  +------+------+
                  |             |
              FAIL/WARN         OK
                  |             |
                  v             v
                 STOP     Vertex AI Model
                              Registry




Architecture

The repository contains multiple pipelines that support different stages
of the ML lifecycle.


flowchart TD

    A[Snowflake Source Data] --> B[Connection Check]

    B --> C[Feature Extraction]

    C --> D[Data Quality Check]

    D --> E[Preprocessing]

    E --> F[Feature Validation]

    F -->|Fail| X[Stop Pipeline]

    F -->|OK or Warn| G[Cluster Training]

    G --> H[GCS Model Artifact Bundle]

    H --> I[Model Validation]

    I -->|Fail or Warn| X

    I -->|OK| J[Vertex AI Model Registry]

    H --> K[Inference Pipeline]

    H --> L[SHAP Explainability Pipeline]

    E --> M[Drift Detection Pipeline]
