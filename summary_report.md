# Credit Card Customer Segmentation --- Summary Report

## Business problem and dataset

This project explores behavioural segmentation for a bank's Cards and
Payments division using the Kaggle Credit Card Dataset for Clustering.
It contains 8,950 customer records and 18 columns. `CUST_ID` was
excluded because it is an identifier, not a behavioural feature. The
dataset contains 313 missing `MINIMUM_PAYMENTS` values and one missing
`CREDIT_LIMIT` value; no exact duplicate rows were found.

## Feature engineering and preprocessing

Four features were engineered: average monthly purchases, average
monthly cash advances, balance-to-credit-limit utilisation, and payments
relative to minimum payments. Missing numeric values were
median-imputed. IQR bounds were applied to `BALANCE`, `PURCHASES`,
`CASH_ADVANCE`, `CREDIT_LIMIT`, and `PAYMENTS`; outliers were capped,
not deleted. Skewed non-negative monetary features were transformed
using `log1p`, and all model features were standardised with
`StandardScaler`. The matching preprocessing configuration is saved for
repeatable scoring. Silhouette scores use a reproducible sample of up to
2,500 customers to keep runtime manageable.

## Algorithm comparison

K-Means was tested for k=2 through k=10. The k=2 candidate had the
highest sampled silhouette (about 0.240), while k=4 scored about 0.206.
Four clusters were selected as a practical interpretability trade-off,
not because k=4 maximises silhouette. The K-Means groups contain 2,352,
1,631, 2,708, and 2,259 customers. For k=4, sampled silhouette is about
0.206, Davies--Bouldin Index about 1.738, and Calinski--Harabasz Index
about 1,955.5. Ward agglomerative clustering scores about 0.174 on
sampled silhouette. Complete and average linkage each produce one
cluster of 8,943 customers plus clusters of 5, 1, and 1; their high
silhouette (about 0.737) is misleading for balanced segmentation. DBSCAN
with eps=1.5 and min_samples=10 finds five non-noise clusters and marks
2,147 customers (23.99%) as noise. K-Means silhouette is stable across
five random seeds, with standard deviation around 0.0001.

## Personas and actions

Cluster 0 (2,352 customers) has very low purchases and purchase
frequency, with appreciable cash advances and balances; consider clear
fee and repayment information. Cluster 1 (1,631) has higher balances and
credit limits with both purchases and cash advances; test relevant
product education and balance-management communications. Cluster 2
(2,708) has lower balances and cash advances, moderate purchase
frequency, and a higher full-payment share; consider optional
card-benefit education. Cluster 3 (2,259) has the highest average
purchase amount and purchase frequency among these groups, with
relatively low cash advances; test relevant rewards or merchant offers.
These labels describe group averages, not every customer.

## Limitations and next steps

K-Means is selected for this prototype because it yields reasonably
sized, interpretable groups and supports `.predict()` for new
observations. Its moderate silhouette and the higher k=2 score are
acknowledged trade-offs. DBSCAN is parameter-sensitive and excludes
noise from its silhouette calculation. These are exploratory segments,
not validated measures of profitability, fraud, or credit risk. Future
work could add appropriately governed transaction categories, temporal
patterns, repayment history, and campaign outcomes, then validate
segment stability and business impact before operational use.
