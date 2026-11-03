<div align="center">

# 📈 Sales Prediction using Python

**CodeAlpha Data Science Internship · Task 4**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C) ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0) ![CodeAlpha](https://img.shields.io/badge/CodeAlpha-Data%20Science%20Internship-6A1B9A)

A model that predicts product sales from advertising spend on TV, Radio and Newspaper with **R² = 0.99**, and finds a budget split that could raise sales by about **35% at no extra cost**.

</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Visualizations](#-visualizations)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

## 📌 Overview
Businesses spend large amounts on advertising, but which channel actually brings in sales? This project
uses spending data from 200 markets to **predict sales**, measure how much each advertising channel
contributes, and test **what-if budget scenarios** that marketing teams can act on.

## 📂 Dataset
| Property | Value |
|---|---|
| Source | [Advertising dataset on Kaggle](https://www.kaggle.com/datasets/bumba5341/advertisingcsv) |
| Rows | 200 markets |
| Features | `TV`, `Radio`, `Newspaper` (advertising spend in $ thousands) |
| Target | `Sales` (in thousands of units) |

The dataset file is included in this repository, so the notebook runs without any extra downloads.

## 🔄 Project Workflow
1. **Data cleaning**: dropped the index column; checked for missing values, duplicates and negative values
2. **EDA**: distributions, sales vs spend on each channel, correlation heatmap
3. **Feature selection**: models with and without `Newspaper` compared using cross-validation
4. **Modelling**: Linear Regression, an **interaction + curve model** (TV × Radio and squared terms), Random Forest and Gradient Boosting
5. **Evaluation**: R², MAE and RMSE on a held-out test set
6. **Business analysis**:
   - Sales uplift from an extra $10k on each channel
   - Budget reallocation with the same total spend
   - A TV × Radio sales map showing how the two channels work together

## 📊 Results
| Model | CV R² | Test R² | Test MAE | Test RMSE |
|---|:---:|:---:|:---:|:---:|
| 🏆 **Interaction + curve (TV, Radio)** | **0.982** | **0.989** | **0.49** | **0.60** |
| Gradient Boosting | 0.968 | 0.984 | 0.60 | 0.72 |
| Random Forest | 0.966 | 0.983 | 0.61 | 0.74 |
| Linear (TV, Radio) | 0.874 | 0.901 | 1.44 | 1.77 |
| Linear (TV, Radio, Newspaper) | 0.870 | 0.899 | 1.46 | 1.78 |

> Models are compared using **5-fold cross-validation** on the training set. The test set is held out until the end for a fair final check.

## 💡 Key Insights
- 📻 **Radio gives the best return:** about **190 extra units per extra $1,000**
