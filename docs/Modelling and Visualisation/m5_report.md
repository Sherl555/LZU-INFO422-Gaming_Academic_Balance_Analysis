# M5 Part 1 — Modelling Setup and Model Training

## 1. Connection to Previous Milestones

This modelling stage is built directly on the work completed in M1–M4. In M1, the project defined the central research problem as the relationship between gaming behaviour and academic performance, with `grades` as the main academic outcome. M2 established the dataset source and structure: one row represents one student, and the dataset contains behavioural, study-related, and health-related variables. M3 converted the raw data into model-ready train/test files through cleaning, encoding, feature engineering, and scaling. M4 then used exploratory data analysis to revise the original hypotheses and guide the modelling strategy.

The EDA findings are especially important for M5. The original H1 proposed an inverted-U relationship between gaming time and grades, but EDA suggested a threshold-negative pattern instead: gaming appears relatively low-risk up to about two hours per day, followed by a clearer academic decline at higher gaming levels. EDA also suggested that the addiction pathway is more convincing than the sleep or stress pathways, while game genre showed little evidence of moderating the gaming–grades relationship. Therefore, the modelling stage focuses on predicting `grades` while testing whether nonlinear models better capture the threshold pattern found during EDA.

## 2. Modelling Question
The modelling question for M5 is:

> Can student grades be predicted from gaming behaviour, study habits, and health-related variables, and do nonlinear or threshold-aware models better capture the gaming-risk pattern identified during EDA?

This question keeps the project aligned with the original proposal while incorporating the revised hypotheses from EDA. The task is treated as a regression problem because the target variable, `grades`, is continuous.

## 3. Data Used for Modelling

The modelling code uses the cleaned and preprocessed datasets from M3:

- `X_train.csv`
- `X_test.csv`
- `X_train_scaled.csv`
- `X_test_scaled.csv`
- `y_train.csv`
- `y_test.csv`
- `scaler.pkl`

The original train/test split is preserved instead of creating a new split. This keeps the experimental design consistent across preprocessing, EDA, and modelling. It also ensures that all models are compared fairly on the same test set.

Although M3 already provides scaled feature matrices, the modelling notebook mainly trains from the unscaled processed features and applies scaling inside sklearn pipelines for linear models. This is because M5 adds EDA-informed threshold features, and any new scaling step should be fitted only on the training data. Tree-based models use unscaled features because decision trees split on feature thresholds and do not require standardization.

## 4. Modelling Data and Feature Audit

Before training, the notebook performs a modelling data audit. This records the number of train/test rows, the number of features, train/test target means and standard deviations, column alignment, and missing-value counts. This confirms that the M3 output is usable for modelling and creates a transparent handoff record.

The notebook also creates a feature audit table that groups variables into gaming behaviour, study behaviour, health/wellbeing, and demographic/categorical features. This makes later feature-importance interpretation more meaningful.

## 5. EDA-Informed Feature Engineering

M4 suggested that the gaming–grades relationship is not simply linear or inverted-U shaped. Instead, grades appear relatively stable at low gaming levels and decline more clearly after around two hours per day. To connect this EDA finding to modelling, the notebook adds threshold features:

```text
gaming_within_2h = min(gaming_hours, 2)
gaming_over_2h = max(0, gaming_hours - 2)
```

It also adds nearby sensitivity features:

```text
gaming_over_1_5h = max(0, gaming_hours - 1.5)
gaming_over_2_5h = max(0, gaming_hours - 2.5)
```

These features allow linear models to represent different slopes before and after the EDA threshold. This does not claim that two hours is a universal causal boundary; it is an EDA-informed modelling feature that should be evaluated in the second part of M5.

The variable `reaction_time_ms` is excluded from the primary model by default. In EDA, this variable showed an unusually strong relationship with `gaming_hours`, raising a possible leakage or mechanical-generation concern. Excluding it makes the primary modelling results more conservative.

## 6. Modelling Design Matrices

To make the modelling comparison more informative, the notebook prepares several design matrices:

1. **Core behavioural features**: a small, interpretable set of main variables.
2. **Core + 2h threshold features**: directly tests the M4 threshold finding.
3. **Core + threshold sensitivity features**: compares nearby threshold choices.
4. **Full processed features**: uses all available processed variables except leakage-risk features.
5. **Full processed features without device usage**: supports a multicollinearity sensitivity check.

This design allows the evaluation teammate to compare not only model families, but also feature representations.

## 7. Model Selection Rationale

The model set is designed to move from simple baselines to more flexible nonlinear models. This follows the Week 7 modelling principle that a strong modelling pipeline should build sensible baselines, compare model families fairly, evaluate responsibly, and report results clearly.

### 7.1 Dummy Mean Baseline

The Dummy Regressor predicts the mean grade from the training set for every student. It provides a non-learning reference point. Any useful model should perform substantially better than this baseline.

### 7.2 Linear Regression Core

The core linear model uses a small set of theoretically important variables, including gaming hours, study hours, sleep hours, attendance, addiction score, and stress level. This model is easy to explain and shows how much predictive signal exists in the most important project variables.

### 7.3 Linear Regression Full

The full linear model uses the larger processed feature set. It tests whether engineered variables and encoded categorical features improve a simple additive model. This model is still interpretable, but it may be sensitive to correlated predictors.

### 7.4 Piecewise Linear Models

Piecewise linear models are included to connect M5 directly to the revised H1 from M4. By using threshold features, they can model a change in slope after the EDA-identified gaming threshold. A sensitivity version also checks nearby thresholds so that the analysis does not rely on one arbitrary cutoff.

### 7.5 Ridge Regression

Ridge Regression is included as a regularized linear model. It is useful because the dataset contains correlated behavioural variables and engineered features. Ridge shrinks coefficients smoothly, which helps reduce instability caused by multicollinearity while preserving the linear modelling framework.

### 7.6 Lasso and ElasticNet

Lasso and ElasticNet are included as additional regularized models. Lasso can shrink weak coefficients to zero, while ElasticNet combines L1 and L2 regularization. These models help test whether a smaller or more stable feature set can preserve predictive performance.

### 7.7 Random Forest Regressor

Random Forest is included as a nonlinear tree-based ensemble. It is appropriate for structured tabular data and can capture nonlinear effects, threshold patterns, and interactions without requiring all of them to be manually specified.

### 7.8 Gradient Boosting Regressor

Gradient Boosting is included as a stronger nonlinear benchmark. It builds trees sequentially, where each new tree focuses on correcting the errors of previous trees. This model is often effective for tabular data and can capture complex patterns, but the final interpretation is left for the evaluation section.

## 8. Evaluation Preparation for Part 2

This part trains models and produces raw evaluation outputs, but it does not write the final model comparison or stakeholder interpretation. Those tasks are left for the second teammate.

The notebook prepares three regression metrics:

- **RMSE**, which penalizes large errors more strongly.
- **MAE**, which reports average grade-point error and is easy to explain to stakeholders.
- **R²**, which shows the proportion of grade variance explained by the model.

The notebook also performs 5-fold cross-validation on the training set. The held-out test set is used to generate predictions and raw metric values for Part 2.

The following files are saved for the second teammate:

- `modelling_data_audit.csv`
- `feature_audit.csv`
- `design_matrix_plan.csv`
- `model_training_plan.csv`
- `model_performance_raw.csv`
- `test_predictions.csv`
- `trained_model_metadata.csv`
- `linear_model_coefficients.csv`
- `tree_feature_importance_raw.csv`
- saved model files under `m5_outputs_part1/models/`

This division keeps Part 1 focused on modelling design, feature preparation, and model training, while leaving Part 2 enough space to complete final evaluation, comparison charts, stakeholder-facing visualisations, residual analysis, and limitations.