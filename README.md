# Credit Card Customer Segmentation --- Unsupervised Learning

An end-to-end practical project using K-Means, Agglomerative
Hierarchical Clustering, and DBSCAN to explore credit-card customer
behaviour.

## Business objective

Identify interpretable groups based on spending, balances, cash
advances, credit utilisation, and payment behaviour, then translate the
profiles into potential Cards and Payments engagement actions. This is
exploratory segmentation, not a credit-risk model.

## Dataset

-   Source: [Kaggle --- Credit Card Dataset for
    Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
-   File expected by the notebook: `data/CC GENERAL.csv`
-   Original dataset: 8,950 rows and 18 columns.
-   `CUST_ID` is excluded from modelling.

The dataset file is included in this project bundle for local
reproducibility. Before publishing it to a public GitHub repository,
verify the dataset's current licence/redistribution terms; if needed,
remove the CSV from the repository and instruct users to download it
from Kaggle.

## Repository structure

``` text
creditcard-segmentation-unsupervised-learning/
├── CreditCardSegmentation_UnsupervisedLearning.ipynb
├── data/
│   └── CC GENERAL.csv
├── artifacts/
│   ├── cc_scaler.pkl
│   ├── cc_segmentation_model.pkl
│   ├── cc_preprocessing_config.pkl
│   └── cluster_personas.pkl
├── summary_report.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

``` bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open `CreditCardSegmentation_UnsupervisedLearning.ipynb` and run all
cells from top to bottom. The notebook uses a fixed random state for
reproducibility. Some silhouette calculations use a sample of up to
2,500 customers to keep runtime manageable; this is labelled in the
outputs.

## Workflow

1.  Load and inspect the dataset; report missingness and duplicates.
2.  Perform histograms, TENURE boxplot, correlation heatmap, and
    customer-level analysis.
3.  Engineer monthly purchase, monthly cash-advance, limit-usage, and
    payment-to-minimum features.
4.  Impute missing values, cap specified outliers using IQR, transform
    skewed monetary columns with `log1p`, and standardise features.
5.  Tune K-Means over k=2--10 and visualise cluster profiles.
6.  Compare Ward, complete, and average hierarchical linkage; inspect a
    Ward dendrogram and cut line.
7.  Tune DBSCAN over the required 20 eps/min_samples combinations;
    report clusters and noise.
8.  Compare Silhouette, Davies--Bouldin, Calinski--Harabasz, and noise
    percentage; check K-Means stability over five seeds.
9.  Save the model and preprocessing artefacts and test scoring on five
    hypothetical customers.

## Findings from this run

-   K-Means k=4 was selected as a practical four-segment trade-off; k=2
    had the higher sampled silhouette (about 0.240 vs about 0.206 for
    k=4).
-   K-Means segment sizes: 2,352; 1,631; 2,708; and 2,259 customers.
-   Provisional personas: cash-advance/low-purchase behaviour;
    mixed-use/higher-balance behaviour; lower-balance/light-use
    behaviour; active purchasers.
-   DBSCAN at eps=1.5 and min_samples=10 found five non-noise clusters
    and labelled 2,147 customers (23.99%) as noise.
-   Complete and average linkage generated extremely imbalanced
    partitions with singleton clusters, so their high silhouette should
    not be interpreted as useful segmentation quality.

See `summary_report.md` for the measured results and limitations.

## Artefacts and scoring

The `artifacts/` directory contains the fitted StandardScaler, final
K-Means model, preprocessing configuration, and cluster-to-persona
mapping. The notebook defines `predict_customer()` to apply the same
preprocessing and return a cluster label and persona for new raw
customer behaviour inputs.

Only load trusted `.pkl`/`.joblib` files. Model artefacts created with
one scikit-learn version may not be portable to every other version; if
loading fails, rerun the notebook in the target environment.

## Business personas and potential actions

-   **Cash-advance / low-purchase behaviour:** provide clear information
    about fees, interest, and repayment options.
-   **Mixed-use, higher-balance behaviour:** test relevant product
    education and balance-management communications.
-   **Lower-balance, light-use behaviour:** consider optional
    card-benefit education and relevant activation campaigns.
-   **Active purchasers:** test relevant rewards and merchant offers.

These are group-level hypotheses. Cluster membership must not be used
alone to determine credit eligibility, pricing, fraud status, or other
consequential decisions.

## Limitations and future improvements

The model uses aggregated monthly behaviour and has no ground-truth
segment labels. Results depend on feature engineering, scaling, and
chosen hyperparameters. Validate future campaign actions against
observed outcomes, and consider adding appropriately governed
transaction categories, temporal patterns, repayment history, and
customer response data.

## Recorded video link

**Add your actual Google Drive or YouTube Unlisted video URL here before
submission.** The assignment requires a 5--10 minute MP4/MOV recording
with your face overlay and full screen visible throughout.
