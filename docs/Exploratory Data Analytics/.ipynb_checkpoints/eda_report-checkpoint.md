# Exploratory Data Analysis Report — Gaming Behavior & Academic Performance

**INFO 442 · Data Science Projects · M4 — EDA Report**
**Group 5** -  Jingxu Li 

---

## 1. Introduction

This report presents the exploratory data analysis of the Gaming vs Academic Performance dataset — 8,000 students and 14 features. The analysis is guided by the four hypotheses defined in the project proposal:

- **H1:** Gaming and learning exhibit a nonlinear relationship with an optimizable balance point (inverted U-shaped or threshold pattern)
- **H2:** Gaming behavior affects academic performance through physical and mental health pathways (sleep quality and stress levels as mediators)
- **H3:** Distinct student profiles exist, enabling early identification of at-risk groups (behavioral clusters)
- **H4:** Game genre significantly moderates the gaming–academia relationship (Casual players show weaker negative associations compared to FPS/RPG)

All visualisations come from the companion notebook `02_exploratory_data_analysis.ipynb`.

The dataset is sourced from Kaggle ("Gaming vs Academic Performance"). It contains 10 numeric variables, 3 categorical variables, and a student ID column. Our target is `grades`. Key predictors include `gaming_hours`, `study_hours`, `sleep_hours`, `attendance`, `gaming_genre`, `stress_level`, and `addiction_score`. Stage 1 preprocessing confirmed zero missing values and zero duplicates, capped outliers using the IQR method, one-hot encoded categorical variables, and added 6 engineered features — bringing the processed feature set to 20. The full raw dataset (8,000 rows) is used here for exploration; processed data is reserved for modelling.

---

## 2. Data Overview & Quality

The dataset is clean with no missing values or duplicate rows. It includes 10 numeric variables (`age`, `gaming_hours`, `study_hours`, `sleep_hours`, `attendance`, `social_activity`, `device_usage`, `reaction_time_ms`, `addiction_score`, `grades`) and 3 categorical variables (`gender`, `gaming_genre`, `stress_level`).

For the categorical features, the gender split is nearly balanced at roughly 49–51% across Male and Female, with "Other" making up about 3.7% of the sample (293 students). Gaming genre is more uneven — FPS accounts for nearly 40% of students (3,187), while RPG and Casual each sit around 30% (2,408 and 2,405 respectively). Stress level shows a notable class imbalance: over half the students report Medium stress (4,247), while only 12% report High stress (1,010). This imbalance will need to be considered in stratified analyses.

Numerically, grades average 66.18 with a standard deviation of 22.42 and a median of 67.07, suggesting a leftward skew — most students cluster around moderate performance with a long tail toward the lower end. Gaming hours average 4.09 per day, spread relatively evenly from 0 to 8, with a roughly symmetric shape. Study hours average 5.46, mildly right-skewed. Sleep hours cluster tightly around 6.5, with most students sleeping between 5 and 8 hours. Attendance centers near 80%, but extends down to 60% and up to 100%, with a leftward skew. Device usage (7.59 hours average) substantially exceeds gaming hours, reflecting non-gaming screen time. Reaction time sits around 271 ms with a symmetric distribution. The addiction score averages 9.91 with a fairly even spread across its range.

---

## 3. Univariate Analysis

### 3.1 Target Variable: Grades

![alt text](images/image.png)

Grades follow a left-skewed, unimodal distribution. Most students score between 40 and 100, with a median of 67.07 sitting just above the mean of 66.18. A long lower tail extends toward zero, and a few values exceed 100 — possibly reflecting bonus scoring or a data recording issue. There is no evidence of distinct high-achieving and low-achieving subgroups; the distribution is a single peak.

### 3.2 Numeric Feature Distributions

![alt text](images/image-1.png)

Gaming hours range from 0 to 8 per day with a median near 4. The distribution is slightly right-skewed but broadly symmetric. Study hours span 1 to 10 per day with a similar mild right skew and a median around 5.5. Sleep hours are the most concentrated variable — nearly all students fall between 5 and 8 hours, with a median of 6.5 and very few outliers. Attendance is left-skewed with a median of 80%; a noticeable subset of students has attendance below 70%, which could indicate chronic absenteeism. Device usage ranges from 1 to 14 hours with a strong right skew — many students spend substantially more time on screens than the 4 hours they report for gaming alone. Reaction time is roughly symmetric around 270 ms with balanced whiskers. The addiction score spreads broadly from 0 to 23 with no strong floor or ceiling effects.

### 3.3 Categorical Features

![alt text](images/image-2.png)

Gender shows near-parity between Male (3,904) and Female (3,803), with Other as a small category (293). Gaming genre is dominated by FPS (3,187), while RPG and Casual are nearly tied at around 2,400 each. Stress level is imbalanced: Medium dominates at 4,247, Low accounts for 2,743, and only 1,010 students report High stress. This roughly 50:33:12 split means comparisons involving the High stress group will have less statistical power.

---

## 4. Multivariate: Correlation Structure

![alt text](images/image-3.png)

The correlation matrix reveals several findings worth flagging before we move to hypothesis testing.

The strongest positive signal is study hours versus grades at r ≈ 0.73 — students who study more earn higher grades, which is expected but useful to confirm. Gaming hours and grades show a moderate negative correlation of r ≈ -0.55, the primary relationship this project investigates.

A few patterns stand out as surprising. Gaming hours and study hours are essentially uncorrelated at r ≈ -0.01. This contradicts the common "time displacement" narrative — students do not appear to substitute study time for gaming time in any linear sense. Similarly, gaming hours and sleep hours show zero linear association, which weakens the case for sleep as a mediation pathway before we even begin.

Two correlations raise technical concerns. Reaction time and gaming hours are nearly perfectly correlated at r ≈ -0.94 — this is suspicious and may indicate data leakage (reaction time could be derived from or mechanically linked to gaming hours). Device usage and gaming hours correlate at r ≈ 0.85, which creates a multicollinearity risk if both enter the same model.

Other notable correlations: addiction score with grades sits at r ≈ -0.50, sleep with grades at a weak r ≈ 0.25, attendance with grades at a surprisingly low r ≈ 0.13, and social activity with grades at essentially zero.

---

## 5. Bivariate & Multivariate: Hypothesis Testing

### 5.1 H1 — Nonlinear Gaming–Academia Relationship

**Original hypothesis:** Gaming and learning exhibit a nonlinear relationship with an optimizable balance point — an inverted-U shape where moderate gaming is associated with higher grades than no gaming at all.

#### Scatter + LOESS Smooth

![alt text](images/image-4.png)

The LOESS smooth curve tells a clear story: the relationship is monotonically negative, not an inverted-U. Three segments are visible. Between 0 and 2 hours of daily gaming, grades hold steady around 82–83 — this is a safe threshold zone where gaming appears to have minimal academic cost. From 2 to 5 hours, grades drop steeply from roughly 82 to 65. Beyond 5 hours, the decline continues but at a reduced rate, reaching around 40 at the upper end. There is no moderate-gaming group that outperforms the lowest-gaming group, which rules out the inverted-U.

#### Binned Mean Grades

![alt text](images/image-5.png)

Binning gaming hours into one-hour intervals confirms the pattern numerically. The 1–2 hour bin shows the highest mean grade at 82.9, nearly identical to the 0–1 hour bin at 81.6. The critical drop comes at the 2–3 hour bin where the mean falls to 75.0 — a loss of about 8 points. From there the decline is steady: 69.8 at 3–4 hours, 64.5 at 4–5, 60.5 at 5–6, 53.4 at 6–7, and 45.6 at 7–8 hours. The total spread from peak to trough is roughly 37 points, and the sharpest deterioration happens between 2 and 5 hours.

#### Assessment

**H1 is not supported** as an inverted-U. The relationship is better characterized as a threshold-negative model: a safe zone up to about 2 hours per day, followed by steep linear decline, then decelerating harm at very high levels. The quadratic specification originally proposed may not capture this shape well — a piecewise linear or threshold regression with a breakpoint near 2 hours would be more appropriate.

### 5.2 H2 — Mediation Pathways

**Original hypothesis:** Gaming behavior affects academic performance through physical and mental health pathways, specifically sleep quality and stress levels, with addiction as a related factor.

#### Grades vs Study Hours, Sleep Hours, Addiction Score

![alt text](images/image-6.png)

Study hours show a strong positive, near-monotonic relationship with grades. Scores rise from about 40 at 1–2 hours of daily study to about 85 at 7–8 hours, then plateau. This is the strongest positive driver in the dataset.

Sleep hours tell a weaker story. Grades do increase gradually from about 55 at 4 hours of sleep to about 75 at 9 hours, but the variance is large at every level. Many students with low sleep achieve high grades, and vice versa. The signal is positive but noisy.

Addiction score shows a clear threshold structure. Grades hold flat around 80 for scores below 5, then drop sharply and linearly to around 40 at the upper end. This is the most visually compelling of the three potential mediators.

#### Grades by Stress Level

![alt text](images/image-7.png)

The stress level results run counter to expectations. Students reporting High stress have the highest median grades at 82.05, followed by Medium at 72.08, while Low-stress students show the lowest median at 47.36. This ordering directly contradicts a stress-mediation hypothesis. A plausible interpretation is that stress here reflects academic engagement — students who push themselves harder achieve more but also feel more pressure — rather than functioning as a mediator of gaming harm.

#### Pairplot: Mediation Chain

![alt text](images/image-13.png)

A corner pairplot of gaming hours, sleep hours, addiction score, stress level, and grades summarizes the bivariate picture. Gaming and addiction show a very strong positive association (tight diagonal band), and addiction and grades show a clear negative association — this chain is compelling. But gaming and sleep show no discernible pattern, and gaming and stress show no clear trend either. The diagonal distributions confirm that gaming hours and sleep are roughly uniform, addiction is bell-shaped, grades are mildly left-skewed, and stress forms three distinct peaks.

#### Assessment

**H2 is partially supported.** The addiction pathway is convincing — gaming is strongly linked to addiction score, which in turn is strongly linked to lower grades. But the sleep pathway has minimal support (no gaming–sleep association), and the stress pathway is actively contradicted by the data (high-stress students have the highest grades). The direct gaming-to-grades path remains visually dominant. For Stage 3, mediation analysis should focus on addiction while critically reassessing sleep and stress.

### 5.3 H4 — Genre Moderation

**Original hypothesis:** Game genre significantly moderates the gaming–academia relationship, with Casual gaming carrying a smaller academic penalty than FPS or RPG.

#### Grades by Gaming Genre

![alt text](images/image-8.png)

A boxplot of grades split by genre shows medians of 67.33 for Casual, 66.94 for FPS, and 67.04 for RPG — a spread of less than half a grade point. Genre alone explains essentially nothing about academic performance. The jittered strip overlay shows that all three distributions span the full grade range with similar shapes.

#### Faceted LOESS by Genre

![alt text](images/image-9.png)

Plotting grades against gaming hours separately for each genre puts the moderation question to a more direct test. The three LOESS curves are qualitatively identical: each shows the same threshold-plus-decline structure identified in the H1 analysis. The slopes are visually parallel, and all three converge toward the 40–50 range at 6–8 hours of daily gaming. No genre shows a uniquely elevated safe threshold or a flatter decline.

#### Genre × Hours Interaction Heatmap

![alt text](images/image-10.png)

A heatmap of mean grades cross-tabulated by genre and gaming-hour bin confirms what the LOESS curves suggest. Within any given gaming bin, the three genres differ by at most 2–3 points — negligible compared to the roughly 39-point swing from lowest to highest gaming hours. The color gradient runs vertically across hours, not horizontally across genres. The RPG 2–3 hour bin (72.7) is slightly lower than Casual (76.8) and FPS (75.3), but this is a small deviation relative to the overall pattern.

#### Assessment

**H4 is not supported.** Genre effects are negligible across three different visualizations. Median grades differ by less than 0.4 points, LOESS curves are parallel, and within-bin differences are dwarfed by the gaming-hours gradient. Gaming duration, not genre, is the primary driver of academic performance. The genre interaction terms in the modelling pipeline are unlikely to add meaningful predictive power.

### 5.4 H3 — Student Clustering (Deferred to Stage 3)

**Original hypothesis:** Distinct student profiles exist, enabling early identification of at-risk groups.

The gaming × grades space shows a continuous distribution with no obvious multimodality, so this hypothesis cannot be evaluated visually. It will be addressed in Stage 3 using K-means clustering on gaming hours, study hours, sleep hours, attendance, and addiction score, with elbow method and silhouette analysis to determine the optimal number of clusters.

**Status: To Be Determined.**

---

## 6. Bivariate: Supplementary Findings

### 6.1 Attendance

![alt text](images/image-12.png)

Grades rise gradually from about 61 at 60% attendance to about 72 at 100%, for a total gain of roughly 11 points across the full range. The relationship is positive but weak, with substantial scatter at every attendance level. The correlation is r ≈ 0.13 — attendance alone is a poor predictor of academic performance. Other behavioral factors clearly play a larger role.

### 6.2 Age Groups

![alt text](images/image-11.png)

Splitting students into three age groups (16–18 Teen, 19–21 Young Adult, 22–24 Adult) reveals nearly identical grade distributions. Medians cluster tightly around 67–68 across all three groups. When we plot mean grades against gaming hours separately for each age group, the three lines follow the same threshold-plus-decline trajectory. Age-stratified modelling does not appear necessary.

---

## 7. Revised Hypotheses

The EDA evidence leads to revised assessment of all four hypotheses:

**H1** originally proposed an inverted-U relationship, but the data shows a monotonically negative threshold pattern. Grades hold steady up to about 2 hours of daily gaming, then decline sharply with no recovery at higher levels. The hypothesis is not supported in its original form and should be reframed as a threshold model.

**H2** receives partial support. The addiction pathway is compelling — gaming correlates strongly with addiction score, which in turn correlates strongly with lower grades. But the sleep pathway shows minimal evidence (gaming and sleep hours are uncorrelated), and the stress pathway is contradicted (high-stress students have the highest grades). The direct gaming-to-grades path remains dominant.

**H3** cannot be assessed from the current EDA. The distribution of gaming behavior and grades is continuous without obvious cluster structure. This hypothesis will be tested in Stage 3 using K-means clustering.

**H4** is not supported. Median grades differ by less than half a point across Casual, FPS, and RPG genres. Faceted LOESS curves are visually parallel, and within-bin differences are negligible compared to the effect of gaming hours. Gaming duration, not genre, drives the relationship with grades.

---

## 8. Modelling Question

Based on the EDA evidence, our team will pursue the following modelling question for Stage 3:

> **Can student academic performance be predicted from gaming behaviour, study habits, and health variables, and can the data distinguish: (1) the threshold at which gaming becomes academically harmful (H1), (2) the degree to which addiction mediates the gaming–grades relationship (H2), (3) whether distinct student behavioural clusters exist (H3), and (4) whether game genre meaningfully moderates the effect (H4)?**

**Planned Pipeline:**

1. **Baseline Linear Model:** `grades ~ gaming_hours + study_hours + sleep_hours + attendance` — establish reference R² and RMSE.

2. **H1 — Threshold Regression:** Piecewise linear model with breakpoint at ~2 hours per day. Compare against a linear specification and a quadratic specification. If quadratic is retained, expect a negative linear coefficient with an insignificant or positive quadratic term (deceleration, not an optimum).

3. **H2 — Mediation Analysis:** Focus on the addiction pathway. Bootstrap indirect effects for `gaming_hours → addiction_score → grades`. Test sleep and stress pathways but expect minimal or contradictory findings.

4. **H4 — Interaction Test:** F-test for joint significance of `gaming_hours × genre_FPS` and `gaming_hours × genre_RPG`. Compare model R² with and without interaction terms. Expected: non-significant.

5. **Random Forest Benchmark:** Full 20-feature processed set. Extract feature importance to confirm whether gaming hours, study hours, and addiction score dominate.

6. **H3 — K-means Clustering:** k = 2–6 on gaming_hours, study_hours, sleep_hours, attendance, addiction_score. Elbow method and silhouette score for k selection. Cluster profiling via radar charts.

**Evaluation Metrics:**
- Regression: R², Adjusted R², RMSE, MAE
- Mediation: indirect effect proportion, bootstrapped confidence intervals
- Clustering: silhouette score, elbow method
- Interaction: F-test p-value, ΔR²

---

## 9. Limitations & Next Steps

These findings are exploratory, not confirmatory. All observations are pattern-based visual evidence — formal statistical testing and out-of-sample validation are reserved for Stages 3 and 4. The dataset is cross-sectional and observational, so no causal claims can be made from these correlations alone.

A few data quality points worth noting. Some grades exceed 100, which may indicate bonus scoring or a recording issue. The near-perfect correlation between reaction time and gaming hours (r ≈ -0.94) raises a potential data leakage concern that should be investigated before modelling. And the stress level variable shows a counter-intuitive relationship with grades that may reflect reverse causality rather than mediation.

---

