# 数据预处理报告

## 1. 初始数据质量

- **原始数据形状**：8000 行 × 14 列
- **缺失值**：0
- **重复行**：0
- **数据类型**：数值型 + 分类型

### 分类变量唯一值
| 变量 | 唯一值 |
|------|--------|
| gender | Male, Female, Other |
| gaming_genre | FPS, Casual, RPG |
| stress_level | Low, Medium, High |

## 2. 异常值处理

采用 IQR 截尾法。

| 变量 | 截尾前范围 | 截尾后范围 |
|------|------------|------------|
| grades | [0.00, 118.63] | 无变化 |
| gaming_hours | [0.00, 8.00] | 无变化 |
| addiction_score | [-3.91, 23.01] | [-2.96, 22.04] |

逻辑一致性：`gaming_hours > device_usage` 行数为 0。

## 3. 分类变量编码

| 原始变量 | 编码方式 | 生成新列 |
|----------|----------|----------|
| gender | One-hot (drop first) | gender_Male, gender_Other |
| gaming_genre | One-hot | genre_Casual, genre_FPS, genre_RPG |
| stress_level | 序数 | stress_level_ord (Low=0, Medium=1, High=2) |

编码后 `stress_level_ord` 分布：Low 34.29%, Medium 53.09%, High 12.63%。

## 4. 特征工程

| 特征名 | 计算公式 | 用途 |
|--------|----------|------|
| gaming_hours_sq | gaming_hours² | H1 倒U形检验 |
| gaming_hours_x_FPS | gaming_hours × genre_FPS | H4 调节效应 |
| gaming_hours_x_RPG | gaming_hours × genre_RPG | H4 调节效应 |
| gaming_study_ratio | gaming_hours / (study_hours+0.01) | 行为平衡 |
| total_load | gaming_hours + study_hours | 总投入 |

处理后总特征数：22

## 5. 训练/测试集划分

- 分层抽样（按 stress_level_ord），测试集 20%
- 训练集：6400 样本，测试集：1600 样本
- 分层后各压力水平比例与原始数据一致

## 6. 特征标准化

- 使用 StandardScaler，仅在训练集拟合
- 标准化后训练集各特征均值 ≈ 0，标准差 ≈ 1

## 7. 生成的文件

保存在 `data/processed/` 目录：

- X_train.csv, X_test.csv
- y_train.csv, y_test.csv
- X_train_scaled.csv, X_test_scaled.csv
- scaler.pkl

## 8. 结论

数据清洗、编码、特征工程、分层划分和标准化已完成，可用于后续建模与假设检验。