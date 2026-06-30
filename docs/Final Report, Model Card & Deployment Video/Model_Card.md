# Model Card: Student Academic Performance Predictor
## Gaming Behavior & Academic Performance — Gradient Boosting Regressor
### Group 5


## Model Details

| Attribute | Specification |
|-----------|-------------|
| **Model** | Gradient Boosting Regressor |
| **Architecture** | 300 decision trees with sequential error correction |
| **Hyperparameters** | learning_rate=0.05, max_depth=3, subsample=0.9, random_state=42 |
| **Input Features** | 23 behavioural, demographic, and engineered variables |
| **Target** | grades (0–100) |
| **Training Data** | 6,400 student records |
| **Test Data** | 1,600 held-out records |
| **Developed By** | Group 5, INFO442 Data Science Projects, June 2026 |

---

## Intended Use

**Primary Purpose:** Educational awareness and decision-support tool for exploring how gaming behaviour, study habits, and health variables relate to academic performance.

**Appropriate Users:**
- Students examining personal behavioural balance
- Educators and counsellors identifying at-risk profiles for guidance
- Parents seeking data-informed context on gaming-academia trade-offs
- Researchers investigating behavioural predictors of student outcomes

**Out-of-Scope Uses:**
- Automated disciplinary actions or punitive measures
- Permanent risk labeling or stigmatization of individuals
- High-stakes decisions (admissions, scholarships, grading)
- Clinical diagnosis of gaming disorder or addiction
- Populations outside the training demographic

---

## Performance

### Test Set Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | 0.936 | 93.6% of grade variance explained |
| **RMSE** | 5.60 | ~5.6 grade points average error |
| **MAE** | 4.42 | Typical prediction within ±4.4 points |
| **CV RMSE** | 5.78 ± 0.06 | Stable across 5 folds |

### Feature Importance (Top 5)

| Rank | Feature | Importance | Description |
|------|---------|-----------|-------------|
| 1 | `gaming_study_ratio` | 77.0% | Gaming hours / study hours |
| 2 | `study_hours` | 10.4% | Daily study duration |
| 3 | `sleep_hours` | 6.2% | Nightly sleep duration |
| 4 | `attendance` | 2.4% | Class attendance percentage |
| 5 | `total_load` | 0.8% | Gaming + study hours |

Top 3 features account for **93.6%** of predictive power.

---

## Limitations

### Data Limitations
- **Cross-sectional design:** Single time-point snapshots prevent causal inference; predictions are associational.
- **Self-report bias:** Gaming, study, and sleep hours rely on student recall, susceptible to measurement error.
- **Synthetic artifacts:** Near-perfect correlation between reaction time and gaming hours (r = −0.94) suggests data generation issues; the variable was excluded.
- **Grade ambiguity:** Some grades exceed 100, indicating non-standard scoring.
- **Limited scope:** Dataset demographic may not generalize to other populations.

### Model Limitations
- **Correlation ≠ causation:** Low predicted grades do not prove gaming caused poor performance; reverse causation and confounding are equally plausible.
- **Threshold uncertainty:** The ~2-hour gaming breakpoint is data-derived, not a universal rule.
- **Individual variation:** The model explains 93.6% of population variance but cannot account for all individual differences.
- **Extrapolation risk:** Predictions unreliable for profiles far outside training distribution.

---

## Ethical Considerations

- **Privacy:** No PII processed; all inputs are behavioural/demographic without identifiers.
- **Fairness:** Gender and genre contribute <0.1% combined; formal subgroup fairness audit pending.
- **Transparency:** Feature importance provides interpretability, but individual tree paths are complex.
- **Human oversight required:** Predictions are statistical expectations, not deterministic judgments.

---

## Recommendations for Use

| User | Guidance |
|------|----------|
| **Students** | Explore behavioural trade-offs; treat predictions as awareness tools, not fate. |
| **Educators** | Focus on `gaming_study_ratio`: risk is about imbalance, not gaming alone. |
| **Researchers** | Replicate on longitudinal data; priority: causal inference and external validation. |





```python

```
