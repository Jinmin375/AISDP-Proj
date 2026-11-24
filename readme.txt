Data quality

There are no duplicates, and timestamps are exactly 1 minute apart.
temperature is 14% missing, and the missingness is completely random. It doesn't vary by target, machine, status or day (chi-square p = 0.31). The notebook fills it with the median, learned on the training set only.
6.7% of vibration values are negative. After removing the sign, they match the positive distribution (KS p = 0.58), so this is a sign error and is fixed with abs().
Outliers in temperature and vibration are real events: 54–58% of them need maintenance, against 20% overall. They are kept.

Target

The classes are imbalanced, about 80% No and 20% Yes. Accuracy is misleading, so the notebook uses stratified splits and judges models on recall, precision, F1 and PR-AUC.

What drives the target

Only three numeric features carry signal: predicted_remaining_life (strongest, r = −0.34), temperature (+0.28) and vibration (+0.11).
They act as thresholds, not straight lines. Above 90 °C, above 80 mm/s, or below 20 h of remaining life, maintenance is needed 100% of the time. Below 50 h the risk rises sharply.
humidity, pressure and energy_consumption are roughly uniform random and unrelated to the target (p > 0.05).
Sensors are uncorrelated with each other, so there is no multicollinearity.
machine_id, hour, weekday and trend have no effect, and consecutive readings are independent (autocorrelation ≈ 0). The notebook adds no time or lag features.

Leakage (the key finding)

The target matches this rule on 100% of rows: anomaly_flag OR machine_status == 2 OR remaining_life < 20.
anomaly_flag is itself almost exactly a sensor rule. Of the 733 anomalies the rule misses, 99.3% have a missing temperature.
The notebook drops failure_type, downtime_risk (a duplicate of anomaly_flag), anomaly_flag and machine_status. It keeps predicted_remaining_life and the raw sensors.
With machine_status included, a baseline reaches ~0.99 F1. Without it, F1 is ~0.69 (recall ~0.53, precision ~0.99).
45% of positive cases are flagged only because the machine has already failed, and failing machines look identical to healthy ones in every sensor. That caps honest recall at about 55%.

Feature engineering

The notebook adds threshold flags: temp_over_90, vibration_over_80, rul_under_20, rul_under_50 and sensor_alarm.
For logistic regression, F1 rises from 0.47 to 0.69, and precision from 0.36 to 0.94. Recall falls from 0.68 to 0.55.
Tree models can learn these thresholds on their own, so the flags help them less.