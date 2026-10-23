# 房价预测 (House Price Prediction)

基于 **Ames 房价数据集**（Kaggle 经典题目 *House Prices: Advanced Regression Techniques*）的端到端房价预测项目，涵盖数据探索、缺失值清洗、MySQL 入库与 SQL 分析、可视化，以及机器学习建模。

## 数据集

| 项目 | 说明 |
| --- | --- |
| 来源 | Ames, Iowa 市住宅销售数据 |
| 样本数 | 1,460 行 |
| 特征数 | 80 个（含 43 个类别特征 + 35 个数值特征） |
| 目标变量 | `SalePrice`（售价，美元） |

目标变量分布：均值 180,921，中位数区间 15–16 万，范围 34,900 ~ 755,000，呈右偏长尾分布。

## 文件结构

```
房价预测/
├── Untitled.ipynb          # 主分析 notebook（数据清洗 → SQL → 建模全流程）
├── house_prices_raw.csv    # 原始数据（1460 × 81）
├── house_prices_clean.csv  # 清洗后的数据（无缺失值）
└── README.md               # 本文件
```

## 分析流程

### 1. 数据探索 (EDA)

- 查看数据形状、前 5 行、字段类型与缺失值分布。
- 发现 19 列存在缺失值，缺失最多的为 `PoolQC`(1453)、`MiscFeature`(1406)、`Fence`(1179)、`Alley`(1369) 等。

### 2. 缺失值处理

按缺失语义分三类处理：

| 处理策略 | 字段 |
| --- | --- |
| 填 `"None"`（缺失 = 没有该设施） | `BsmtQual`、`BsmtCond`、`BsmtExposure`、`BsmtFinType1/2`、`GarageType`、`GarageFinish`、`GarageQual`、`GarageCond`、`FireplaceQu`、`PoolQC`、`Fence`、`Alley`、`MiscFeature`、`MasVnrType` |
| 填 `0`（缺失 = 无对应面积/年份） | `MasVnrArea`、`GarageYrBlt` |
| 统计量填补 | `LotFrontage`（中位数）、`Electrical`（众数） |

清洗后缺失值总数为 0，结果保存至 `house_prices_clean.csv`。

### 3. MySQL 入库与 SQL 分析

通过 SQLAlchemy + PyMySQL 写入 MySQL（库：`house_price`，表：`house_prices`），并用 SQL 做了初步洞察：

- **房价 Top 社区**：NoRidge(33.5 万)、NridgHt(31.6 万)、StoneBr(31.0 万)。
- **整体质量与房价强正相关**：`OverallQual` 从 1 级（5.0 万）单调上升到 10 级（43.9 万）。
- **居住面积与房价正相关**：`GrLivArea` <1000 平均 11.3 万，≥2500 平均 32.9 万。

### 4. 可视化

- `SalePrice` 分布直方图（右偏长尾）。
- `GrLivArea` 与 `SalePrice` 散点图（正相关，可见少量高面积低价的离群点）。

### 5. 机器学习建模

对目标做 `log1p` 对数变换以压缩长尾、近似正态，训练集 80% / 测试集 20%（`random_state=42`）。

| 模型 | 特征 | 目标变换 | R² | RMSE（美元） |
| --- | --- | --- | --- | --- |
| 线性回归 | 仅数值特征（36 个） | 无 | 0.8274 | 36,387 |
| 线性回归 | 全部特征 one-hot（302 个） | 无 | 0.8875 | 29,381 |
| **线性回归** | 全部特征（302 个） | **log1p** | **0.9327** | **22,715** |
| 随机森林（200 棵树） | 全部特征（302 个） | log1p | 0.8845 | 29,767 |

**结论**：对数变换 + 线性回归表现最佳（R² 0.9327）。特征重要性显示 `OverallQual`（54%）、`GrLivArea`（11.6%）、`TotalBsmtSF`（4.4%）、`GarageCars`（4.0%）对房价贡献最大。

### 5.1 优化：剔除离群点 + 交叉验证

单次 80/20 划分的 R² 可能偏高。用 5 折交叉验证复查后发现：含离群点的一折 R² 会跌到负值，说明模型对极端样本敏感。据此做了两项优化：

- **剔除离群点**：去掉 `GrLivArea > 4000` 且 `SalePrice < 300000` 的 2 条异常样本。
- **交叉验证**：改用 5 折交叉验证评估，每折轮流当测试集。

| 指标 | 优化前（单次划分） | 优化后（5 折交叉验证） |
| --- | --- | --- |
| R² | 0.9327（单一测试集，偏高） | **0.9148 ± 0.0099**（各折约 0.90–0.93） |

剔除离群点后，交叉验证各折 R² 稳定在 0.90–0.93，说明模型泛化能力稳定，而非单次划分的偶然结果。

## 环境与运行

- Python 3.13（Anaconda）
- 依赖：`pandas`、`numpy`、`matplotlib`、`scikit-learn`、`SQLAlchemy`、`PyMySQL`
- MySQL（库 `house_price`，连接字符串见 notebook）

```bash
# 安装依赖
pip install pandas numpy matplotlib scikit-learn sqlalchemy pymysql

# 启动 notebook 依次运行各 cell
jupyter notebook Untitled.ipynb
```

> 注意：运行「MySQL 入库」相关 cell 前，需先创建数据库 `house_price` 并确认本机 MySQL 已启动、账号密码与连接字符串一致。

## 后续可优化方向

- 特征工程：构造 `TotalSF`、总浴室数、房龄等组合特征。
- 换用梯度提升模型（`HistGradientBoostingRegressor` / XGBoost / LightGBM）并调参。
