# Gaming Behavior and Academic Achievement: Exploring the Impacts of Video Games on Academic Performance
## Group Project Proposal for INFO442 Data Science
### Members From Group 5
| Name | Student ID | 
|------|------------|
| Jingxu Li | 320230942071  
| Bohan Liu | 320230942141
| Shuyang Zhang | 320230942711  
| Jiayu Wang | 320230942421  
| Yang Zhou | 320230942801 

## Domain and Motivation
At present, video games have been deeply integrated into the daily lives of contemporary students. There are over 2 billion active video game players worldwide, with a significant proportion being teenagers and college students. With the popularity of this form of entertainment, a series of social discussions have emerged: parents and schools are concerned that excessive gaming will erode students' study time and lead to a decline in academic performance; students, on the other hand, hope to have a reasonable relaxation space after the pressure of studies. For a long time, discussions on whether "video games affect learning" have mostly relied on subjective judgments and individual case observations, lacking theoretical support based on large-scale sample quantitative data. Is the impact of video games on students' learning purely negative? Does moderate gaming have a positive effect on students' learning and health? What are the degrees of influence of different game types and gaming durations on students' academic performance and psychological and physiological conditions? And, can we find a balance point between gaming and academic learning? Currently, we cannot provide a definite, forceful answer supported by data and theory to these questions.

To address this series of issues, the public dataset "Gaming vs Academic Performance" selected by this research from Kaggle provides a solid data foundation for breaking through this predicament. This dataset includes 8,000 student records and 14 core features, systematically covering the dimensions of students' gaming behavior: including game duration, game type, device usage duration, reaction time, and addiction score. As well as academic performance dimensions: including study duration, attendance rate, sleep quality, stress level, and final academic performance. This data structure enables us to go beyond simple correlation descriptions and delve into the complex interaction mechanisms between gaming behavior patterns and academic performance.

## Dataset Description
For this research, the data we used is from the public dataset "Gaming vs Academic Performance" on the Kaggle website:

- **Source**: Kaggle Open Datasets (https://www.kaggle.com/datasets/aiexplorer77/gaming-vs-academic-performance)
- **Size**: 8,000 rows × 14 features (approximately 1.2 MB)
- **Format**: CSV (Comma-Separated Values), compatible with pandas, R, and Excel
- **Access Method**: Publicly available; free download after Kaggle account registration (no API token required for direct download)
- **Ethical Considerations**: 
  - The dataset is fully anonymized with no personally identifiable information (PII); `student_id` is a synthetic identifier with no linkage to real individuals
  - Published under the CC0 (Public Domain) or applicable Kaggle dataset license, permitting academic and research use
  - No sensitive demographic data beyond basic age and gender; no medical or financial records involved
  - As secondary data analysis, no informed consent from original participants is required; proper attribution to the original dataset and Kaggle platform is maintained

- **Features**:
  1) student_id - Unique identifier assigned to each student
  2) age - Age of the student
  3) gender - Gender category: Male , Female , Other
  4) gaming_hours - Average number of hours spent gaming per day
  5) study_hours - Average number of hours spent studying per day
  6) sleep_hours - Average number of hours slept per day
  7) attendance - Percentage of classes attended
  8) gaming_genre - Preferred gaming category: FPS, RPG, Casual
  9) social_activity - Time spent on social activities per day
  10) device_usage - Total daily screen time including gaming
  11) reaction_time_ms - Cognitive reaction time in milliseconds
  12) addiction_score - Estimated gaming addiction level
  13) stress_level - Stress category: Low , Medium , High
  14) grades - Academic performance score on a scale from 0 to 100

Preliminary data auditing indicates that this dataset has a high degree of structural integrity and internal consistency. There are no missing values for all variables, eliminating the potential bias caused by missing data imputation. The distribution range of numerical variables is reasonable, and no obvious input errors or logical contradictions have been found. The fourteen-dimensional features provide a solid foundation for modeling and reserve a rich space for derivative development.

## Scientific Question
**Core Research Question:** What is the optimal daily gaming duration (if any) that maximizes academic performance, and which gaming behavior features (such as genre, addiction score, sleep displacement) most strongly predict deviation from this optimum?

**Specific Research Directions:** 
1. What is the specific relationship between gaming duration and academic performance, and can we estimate a threshold point beyond which marginal gaming time becomes academically harmful?

2. To what extent do sleep hours and stress level mediate the gaming-to-grades relationship, and does controlling for these health variables alter the estimated optimal gaming duration?

3. Does the type of game played moderate the relationship between gaming duration and academic outcomes, and should intervention strategies account for this heterogeneity?

4. Do students with different gaming behaviors form distinct risk groups, and can these groups be identified early for targeted support?

## Preliminary Hypothesis

**Hypothesis 1: Gaming and learning exhibit a nonlinear relationship with an optimizable balance point**

We expect to find that the relationship between gaming hours and academic performance is not purely negative, but follows an inverted U-shaped or threshold pattern where moderate gaming (likely 1–2 hours daily) correlates with better academic outcomes than both zero gaming and excessive gaming. 

**Reason**: Students face substantial academic pressure; complete deprivation of recreational outlets may elevate stress and reduce study efficiency, while excessive gaming displaces study and sleep time. We anticipate a quantifiable balance point where gaming serves as effective stress relief without encroaching on academic development.

---

**Hypothesis 2: Gaming behavior affects academic performance through physical and mental health pathways**

We expect that gaming hours and addiction scores will show significant associations with sleep quality and stress levels, which in turn mediate the effect on grades. Students with high gaming addiction scores might cluster in the low-sleep, high-stress, low-grade group.

**Reason**: Preliminary observations suggest that gaming displacement of sleep is common among adolescents. If the direct gaming-to-grades link weakens after controlling for sleep and stress, this indicates that health variables are critical intervention targets for parents and schools, not merely gaming duration itself.

---

**Hypothesis 3: Distinct student profiles exist, enabling early identification of at-risk groups**

We expect to identify distinct behavioral clusters (e.g., "balanced gamers," "high-risk addicts," "non-gaming high-stress") through unsupervised or supervised classification. The high-risk group might be characterized by high addiction scores, high FPS/RPG preference, low attendance, and poor sleep.

**Reason**: If such clusters emerge with clear feature boundaries, schools and parents can intervene preemptively using easily observable proxies (device usage, attendance drops, genre preference) rather than waiting for grade deterioration.

---

**Hypothesis 4: Game genre significantly moderates the gaming-academia relationship**

We expect that **Casual** game players will show weaker negative associations with academic performance compared to **FPS** and **RPG** players at equivalent time investments.

**Reason**: Casual games typically demand less sustained cognitive load and are less prone to compulsive engagement loops, potentially serving as genuine recreational breaks. FPS and RPG games, by contrast, emphasize rapid reactive responses or immersive narrative progression that may induce higher cognitive fatigue or addictive retention patterns. This would support genre-specific guidance rather than uniform gaming restrictions.



## Roles of Group Members
To make the project organized and efficient, our team will divide the work based on the main stages of the data science workflow. Each member will take responsibility for a specific part of the project, while all members will also participate in group discussions, peer review, and final integration. This arrangement will help us complete each milestone on time and maintain the consistency of the whole project.

| Team Member | Role | Responsibilities |
|---|---|---|
| Jingxu Li | Project Coordinator & Research Lead | Jingxu Li will coordinate the overall progress of the project. His responsibilities include organizing team discussions, tracking deadlines, refining the scientific question, and making sure the project remains aligned with the course requirements and research goals. |
| Yang Zhou | Dataset & Preprocessing Lead | Yang Zhou will be responsible for managing the dataset and preparing it for analysis. His tasks include documenting the dataset source, size, format, access method, and ethical considerations, as well as checking data quality, cleaning the data, encoding categorical variables, and preparing the processed dataset for later analysis. |
| Shuyang Zhang | Exploratory Data Analysis & Visualization Lead | Shuyang Zhang will focus on exploratory data analysis and visualization. His responsibilities include analyzing the distribution of key variables, exploring relationships between gaming behavior and academic performance, creating visualizations, and summarizing the main findings from the EDA process. |
| Jiayu Wang | Modeling & Evaluation Lead | Jiayu Wang will be responsible for building and evaluating predictive models. Her tasks include selecting suitable models, training baseline and advanced models, comparing model performance with appropriate evaluation metrics, and interpreting model results through feature importance or other explanation methods. |
| Bohan Liu | Documentation, GitHub & Presentation Lead | Bohan Liu will manage project documentation and GitHub organization. His responsibilities include maintaining the repository structure, organizing notebooks and reports, assisting with the proposal and final report, preparing presentation materials, and ensuring that all project deliverables are clear, complete, and reproducible. |

Although each member has a primary role, our team will work collaboratively throughout the project. Important decisions, such as feature engineering strategies, model selection, evaluation methods, and final interpretation of results, will be discussed and decided by the whole team. All members will also review each other's work to improve the quality and reliability of the final project.



