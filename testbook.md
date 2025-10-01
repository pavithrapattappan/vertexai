9080857644



900



Define and enforce access control policies for models and pipelines

Generate and store model cards for all registered models

Document the end-to-end MLOps workflow for compliance audits



Resource and Cost Optimization

Goal: Optimize cost, auto-scale, and lifecycle management.



Automate resource scaling for endpoints and pipelines



Monitor, report, and alert on cost and usage anomalies



Implement automated model and feature artifact cleanup policies



Spss@H106E#$2





9001055

SpssXH106@$E2



Entity Types: customer, product, ...

Features: recency, frequency, ...

Online store: Bigtable (fast lookup)

store: BQ/Spanner/GCS (batch)

Time travel, versioning, lineage, RBAC



**Serves:**

**- Model Training (offline)**

**- Batch Scoring (offline)**

**- Real-time Prediction (online endpoint)**



* **Null/missing value handling**
* **Outlier removal/clipping**
* **Type conversion (dates, categories, numerics)**
* **Basic transformations (e.g., normalization, label encoding)**





* **Null/missing value handling**
* **Outlier removal/clipping**
* **Type conversion (dates, categories, numerics)**
* **Basic transformations (e.g., normalization, label encoding)**







* **Data splits (train/val/test)**
* **Model training**
* **Hyperparameter tuning (if needed)**
* **Save model artifact to GCS**









* **Run predictions**
* **Compare to ground truth**
* **Metrics calculation**
* **Fairness/bias checks**





* **Deploy as endpoint (REST/gRPC API)**
* **Configure machine type, autoscaling, traffic split**
* **Attach monitoring configs**



* **Fetch features for batch entities**
* **Invoke model (local or via endpoint)**
* **Save predictions to GCS/BigQuery**



**Feature Validation \& Drift Check   │**

**- Schema, nulls, range**

**- Drift detection (PSI, JS, etc.)**

**- Logs, alerts, block on fail**





**- Schema, nulls, range**

**- Drift detection (PSI, JS, etc.)**

**- Logs, alerts, block on fail**







**Build/Test Code \& Components**

**Build Docker Images**

**Deploy Pipeline Definitions \& Images**



**Build/Test Code \& Components**

**Build Docker Images**

**Deploy Pipeline Definitions \& Images**











**Infrastructure as Code (Terraform/IaC)**

    \*\*- Versioned definitions for:\*\*

     \\\*\\\*- VMs / VPC / Subnets\\\*\\\*

     \\\\\\\*\\\\\\\*- Artifact Registry (Docker images)\\\\\\\*\\\\\\\*

&nbsp;    \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*- GCS Buckets\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*

     \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*- BigQuery Datasets\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*

     \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*- Vertex AI resources\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*

     \\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\*- IAM roles \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\& Service Accounts\\\\\\\\\\\\\\\*\\\\\\\\\\\\\\\* 





















































































**Infrastructure as Code (Terraform/IaC)      |**

**Provisions all GCP \& Vertex AI resources**





**back**









**Silhouette Score: Measures how similar a point is to its own cluster compared to other clusters. Higher is better (ranges from -1 to 1).**



**Davies-Bouldin Index: Measures the average "similarity" between clusters (lower is better).**



**Calinski-Harabasz Index (Variance Ratio Criterion): Higher is better; evaluates cluster separation.**



**Cluster Size Distribution: Detects if clusters are too imbalanced (which may indicate poor segmentation).**





**-Cleansing/Filtering**

**-RFM Pre-aggregation**

**-Data Validation**

**-Logging**



**Channel exclusion**

**Order exclusion**

Date filter

Monetary filter

Industry filter



Feature Enrichment and Creation



| MATERIAL\\\_ID | MATERIAL\\\_WEB\\\_DESC      | BILLING\\\_QTY | BILLING\\\_DATE |

| ------------ | ------------------------ | ------------ | ------------- |

| 00000123     | “Industrial Cooling Fan” | 58           | 2024-07-01    |







MATERIAL\_ID	MONTH	CATEGORY	AGG\_BILL\_QTY

0000123	2024-07-01	FAN	238



mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000

