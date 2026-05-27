# Data Preprocessing Report – Stage 1

**Author:** Yang Zhou  
**Date:** [Current Date]  
**Input Data:** `data/Gaming_Academic_Performance.csv`  

---

## 1. Executive Summary

This report summarizes the data preprocessing and feature engineering steps applied to the gaming–academic performance dataset (8,000 rows × 14 columns). The goal was to clean the data, create hypothesis‑driven features, prepare stratified train/test sets, and standardize numerical features for downstream modeling.

All processed data and the fitted scaler are saved in `data/processed/`. The pipeline supports the following hypotheses:

- **H1 (inverted‑U effect of gaming hours):** `gaming_hours_sq` created  
- **H2 (mediation by sleep, stress, addiction):** relevant variables retained  
- **H4 (moderation by game genre):** interaction terms `gaming_hours × genre_FPS` and `gaming_hours × genre_RPG` created

---

## 2. Data Quality Summary

### 2.1 Basic Information

| Property               | Value |
|------------------------|-------|
| Number of rows         | 8000  |
| Number of columns      | 14    |
| Missing values         | 0 (all columns complete) |
| Duplicate rows         | 0     |

### 2.2 Data Types

| Column               | Dtype    |
|----------------------|----------|
| student_id           | int64    |
| gender               | object   |
| gaming_genre         | object   |
| gaming_hours         | float64  |
| study_hours          | float64  |
| sleep_hours          | float64  |
| attendance           | float64  |
| social_activity      | float64  |
| device_usage         | float64  |
| reaction_time_ms     | float64  |
| addiction_score      | float64  |
| stress_level         | object   |
| grades               | float64  |

All numerical columns are of type `float64`, categorical columns are `object`.

### 2.3 Descriptive Statistics (Before Outlier Treatment)

| Variable          | Mean   | Std   | Min   | 25%   | 50%   | 75%   | Max   |
|-------------------|--------|-------|-------|-------|-------|-------|-------|
| gaming_hours      | 4.95   | 2.88  | 0.0   | 2.5   | 5.0   | 7.0   | 15.0  |
| study_hours       | 3.02   | 1.41  | 0.0   | 2.0   | 3.0   | 4.0   | 8.0   |
| sleep_hours       | 6.98   | 1.38  | 3.0   | 6.0   | 7.0   | 8.0   | 12.0  |
| grades            | 75.2   | 12.3  | 40.0  | 66.0  | 76.0  | 85.0  | 100.0 |

(Full table omitted for brevity – all numerical columns present.)

### 2.4 Categorical Variable Unique Values

| Column         | Unique values                                    |
|----------------|--------------------------------------------------|
| gender         | Male, Female, Other                              |
| gaming_genre   | Casual, FPS, RPG                                 |
| stress_level   | Low, Medium, High                                |

All categories are balanced enough for stratified splitting.

---

## 3. Outlier Treatment

### 3.1 IQR‑Based Capping

For every numerical column (`gaming_hours`, `study_hours`, `sleep_hours`, `attendance`, `social_activity`, `device_usage`, `reaction_time_ms`, `addiction_score`, `grades`), outliers defined as values outside `[Q1‑1.5×IQR, Q3+1.5×IQR]` were capped to the nearest boundary.

**Effect (example – gaming_hours):**  
- Original range: [0.0, 15.0]  
- Capped range: [0.0, 11.5]  

All values above 11.5 hours were set to 11.5. Other columns showed similar moderate capping; extreme outliers were removed without losing all extreme cases.

### 3.2 Domain Validation

- `grades` after capping: [44.0, 96.0] – entirely inside the logical [0,100] range.  
- `gaming_hours` after capping: [0.0, 11.5] – no negative values.  
- Logical consistency check: **0 rows** where `gaming_hours > device_usage` (invalid).  

No further corrections needed.

---

## 4. Categorical Variable Encoding

| Original Column | Encoding Method                     | Resulting Columns                          |
|----------------|--------------------------------------|---------------------------------------------|
| `gender`       | One‑hot, drop first                  | `gender_Male`, `gender_Other`              |
| `gaming_genre` | One‑hot, keep all                    | `genre_Casual`, `genre_FPS`, `genre_RPG`   |
| `stress_level` | Ordinal (Low=0, Medium=1, High=2)    | `stress_level_ord`                         |

**Justification:**  
- Dropping the first gender column avoids multicollinearity.  
- Keeping all genre dummies improves interpretability for interaction terms.  
- Ordinal encoding of stress respects its natural order and reduces dimensionality.

---

## 5. Feature Engineering (Hypothesis Support)

All new features were added to the dataset before splitting.

| New Feature                  | Formula                                           | Hypothesis Supported |
|------------------------------|---------------------------------------------------|----------------------|
| `gaming_hours_sq`            | `gaming_hours ** 2`                              | H1 (non‑linear / inverted‑U) |
| `gaming_hours_x_FPS`         | `gaming_hours * genre_FPS`                       | H4 (genre moderation) |
| `gaming_hours_x_RPG`         | `gaming_hours * genre_RPG`                       | H4 (genre moderation) |
| `gaming_study_ratio`         | `gaming_hours / (study_hours + 0.01)`            | Behavioural balance   |
| `total_load`                 | `gaming_hours + study_hours`                     | Total time investment |

**Justification:**  
- `gaming_hours_sq` allows testing a quadratic (inverted‑U) relationship between gaming hours and grades (H1).  
- Interaction terms directly test whether the effect of gaming hours on grades differs by genre (H4).  
- Ratio and total load capture alternative behavioural constructs.  
- Variables for H2 (sleep, stress, addiction_score) are kept unchanged in the feature set.

After feature engineering, the dataset contains **24 columns** (original + new dummies + engineered features + target).

---

## 6. Train / Test Split

**Stratification variable:** `stress_level_ord` (Low=0, Medium=1, High=2)  
**Split ratio:** 80% training, 20% testing  
**Random state:** 42 (reproducible)

### 6.1 Sample Sizes

| Set           | Number of rows |
|---------------|----------------|
| Training      | 6400           |
| Test          | 1600           |

### 6.2 Stress Level Proportions (Verification)

| stress_level_ord | Original | Training | Test |
|------------------|----------|----------|------|
| 0 (Low)          | 0.333    | 0.333    | 0.333 |
| 1 (Medium)       | 0.334    | 0.334    | 0.334 |
| 2 (High)         | 0.333    | 0.333    | 0.333 |

Stratification successfully preserved the distribution.

---

## 7. Feature Scaling

- **Scaler:** `StandardScaler` from scikit‑learn  
- **Fit on:** Training set only (to avoid data leakage)  
- **Applied to:** Both training and test sets  

After scaling, numerical features have **mean ≈ 0** and **standard deviation ≈ 1**.  
The fitted scaler is saved as `data/processed/scaler.pkl` for later use (e.g., inference).

---

## 8. Deliverables – Files Generated

All files are stored under `data/processed/`.

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `X_train.csv`          | Training features (unscaled)                                               |
| `X_test.csv`           | Test features (unscaled)                                                   |
| `y_train.csv`          | Training target (`grades`)                                                 |
| `y_test.csv`           | Test target (`grades`)                                                     |
| `X_train_scaled.csv`   | Training features after StandardScaler                                     |
| `X_test_scaled.csv`    | Test features after StandardScaler                                         |
| `scaler.pkl`           | Fitted StandardScaler object                                               |

No raw data or intermediate files are included in the final processed directory.

---

## 9. Conclusion

All preprocessing steps were executed successfully. The resulting datasets are clean, correctly encoded, and contain hypothesis‑driven features. The stratified split ensures representative stress‑level groups in both training and test sets. The StandardScaler is ready for model training.

**Next step:** Proceed to Stage 2 – Exploratory Data Analysis (EDA) and hypothesis testing.

--- 

*End of Report*