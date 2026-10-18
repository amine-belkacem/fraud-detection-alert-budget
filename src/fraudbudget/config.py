"""Operating assumptions shared by every script. Changing one here changes all results."""

# At most 0.1 % of legitimate transactions may be flagged.
MAX_FPR = 0.001

# The analysts can review 200 alerts per 100,000 transactions (113 on the test
# slice, for 75 frauds). The tight scenario, 100 per 100k, gives 56 alerts for 75
# frauds: even a perfect model is capped at 75 % recall there, which is why it is
# kept as a stress case and not as the main operating point.
ALERTS_PER_100K = 200
ALERTS_PER_100K_TIGHT = 100

# Analyst time per alert, in euros. An assumption, not a figure from the data:
# J3 sweeps it from 1 to 20 EUR to check that conclusions do not hinge on it.
COST_PER_ALERT_EUR = 5.0

# Threshold of predict_proba used for the "out of the box" accuracy numbers.
DEFAULT_THRESHOLD = 0.5

SEED = 0
