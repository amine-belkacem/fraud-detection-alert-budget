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

## Tests

```bash
pip install -r requirements.txt
pytest -v
```

The unit tests run on synthetic data, so the CI never downloads the dataset.
