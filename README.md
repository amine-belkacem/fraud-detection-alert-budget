# Fraud detection under an alert budget

Work in progress. This repository evaluates credit card fraud models the way a bank
would use them: how many frauds get caught when analysts can only review a fixed
number of alerts, and what the errors cost in euros. Accuracy is not used as a
criterion, because on this dataset a model that never flags anything already
scores 99.83 %.

## Data

ULB / Worldline credit card fraud dataset: 284,807 European card transactions over
two days in September 2013, 492 of them fraudulent (0.173 %). Features V1 to V28 are
PCA components released by the data provider; only `Time` and `Amount` are in clear.
Distributed on Kaggle (mlg-ulb/creditcardfraud) under ODbL 1.0 (database) / DbCL 1.0 (contents)
licence. The CSV is not stored here:

```bash
python scripts/download_data.py
```

The script checks the SHA-256 of the file and its row and fraud counts.

Source: A. Dal Pozzolo, O. Caelen, R. A. Johnson, G. Bontempi, *Calibrating
Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015.

## Split

Rows are cut in time order, never shuffled: the model is trained on the past and
tested on the most recent transactions, as it would be in production. Each cut is
moved to the next change of timestamp so that no second of activity lands in two sets.

| Set | Hours | Transactions | Frauds | Prevalence |
|---|---|---|---|---|
| train | 0 to 33.4 | 170,888 | 360 | 0.211 % |
| validation | 33.4 to 40.3 | 56,957 | 57 | 0.100 % |
| test | 40.3 to 48.0 | 56,962 | 75 | 0.132 % |

The fraud rate in training is twice the one in validation. A PR-AUC measured on one
set is therefore not directly comparable with one measured on another.

About duplicates: the file has 1,081 exact duplicate rows. None of them straddle a
cut (copies share the same timestamp). Another 1,209 validation and test rows have
the same features and amount as some training row at a different time, which looks
like recurring payments of 1 to 10 EUR. None of these are frauds, so I keep every row.

## First results: the accuracy trap (milestone J1)

Trained on train, evaluated once on test. Single run, no confidence intervals yet
(they come with the bootstrap in J2), so differences of a few points are not
meaningful at this stage.

![accuracy trap](figures/baseline_accuracy_trap.png)

| Model | Accuracy @0.5 | Recall @0.5 | Alerts @0.5 | ROC-AUC | PR-AUC | Recall @FPR 0.1 % | Cost @0.5 (EUR) |
|---|---|---|---|---|---|---|---|
| Dummy (always legit) | 99.87 % | 0 % | 0 | 0.500 | 0.0013 | 0 % | 7,729 |
| Logistic regression | 99.91 % | 49 % | 52 | 0.972 | 0.684 | 75 % | 7,133 |
| Logistic regression, class-weighted | 97.32 % | 89 % | 1,588 | 0.982 | 0.747 | 80 % | 10,281 |

What I take from it:

- The dummy model is 99.87 % accurate and catches no fraud at all. Accuracy ranks it
  above the class-weighted model, which catches 89 % of them.
- Class weighting raises recall at threshold 0.5 but floods the analysts with 1,588
  alerts. At 5 EUR per alert this costs more than letting every fraud through
  (10,281 vs 7,729 EUR). The model is not the problem, the 0.5 threshold is:
  choosing it properly is milestone J3.
- The class-weighted model's single highest score is a legitimate transaction: the
  largest one in the dataset (25,691 EUR), with a PCA component at 120 when the
  median row stays around 2. A linear model extrapolates on that kind of outlier,
  which is one reason to try trees in J2.
- With 100 alerts per 100k transactions the test slice allows 56 alerts for 75
  frauds, so no model can exceed 75 % recall under that budget.

Reproduce with `python scripts/run_baseline.py` (CSV in `results/`).

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

The unit tests run on synthetic data, so the CI never downloads the dataset.
