<div align="center">

# Supply Chain Delivery Delay Analysis and Prediction

### Exploratory analysis and machine learning on 170K+ orders to understand and predict late deliveries

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Key Results at a Glance](#key-results-at-a-glance)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Insights](#key-insights)
- [Machine Learning Model](#machine-learning-model)
- [Recommendations](#recommendations)
- [Limitations and Future Work](#limitations-and-future-work)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Overview

Late deliveries hurt customer satisfaction, increase operating costs, and weaken trust in a supply chain. This project analyzes the **DataCo Smart Supply Chain dataset** to measure how often orders ship late, identify where and when delays concentrate, understand the profit impact, and build a model that predicts late-delivery risk before an order is shipped.

The work is delivered as a single, well-structured Jupyter notebook that covers data cleaning, feature engineering, business KPI calculation, visual analysis, and a classification model.

---

## Business Problem

Operations and logistics teams need answers to questions such as:

- What share of orders are delivered late, and by how many days?
- Do delays differ by region, shipping mode, customer segment, or department?
- Are there seasonal, weekday, or time-of-day patterns in delays?
- Do delayed orders earn less profit?
- Can late-delivery risk be predicted early enough to act on it?

---

## Key Results at a Glance

| Metric | Result |
|--------|--------|
| **Orders analyzed** | 172,765 (after removing canceled shipments) |
| **Late delivery rate** | 54.71% |
| **On-time delivery rate** | 45.29% |
| **90th percentile delay** | 3 days (among late orders) |
| **Profitable orders** | 80.7% (loss-making: 18.7%, break-even: 0.6%) |
| **Total profit (profitable orders)** | $7.5M |
| **Profit from delayed orders** | $2.1M |
| **Model accuracy (Random Forest)** | 74% |
| **Model precision / recall (late class)** | 79% / 75% |

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | DataCo Smart Supply Chain for Big Data Analysis |
| **Source** | [Kaggle / Mendeley Data link] |
| **Raw size** | 180,519 rows, 53 columns |
| **After cleaning** | 172,765 rows |
| **Time period** | January 2015 to January 2018 |
| **Granularity** | One row per order item |

**Key fields used:** order and shipping dates, scheduled vs real shipping days, shipping mode, delivery status, order region, customer segment, department and category, payment type, order status, sales, and profit per order.

> See [How to Run](#how-to-run) for download instructions.

---

## Project Workflow

```
Raw Data  →  Cleaning  →  Feature Engineering  →  KPIs  →  EDA  →  Modeling  →  Recommendations
```

1. **Load and inspect** the data (shape, duplicates, missing values)
2. **Clean** by dropping personal, redundant, and sparse columns and excluding canceled shipments
3. **Engineer features** for delay, processing time, and time-based patterns
4. **Compute business KPIs** for delivery performance and profitability
5. **Analyze visually** across categories, regions, and time
6. **Model** late-delivery risk with a balanced Random Forest classifier

---

## Data Cleaning and Feature Engineering

**Cleaning**
- Confirmed there are **no duplicate rows**
- Removed columns that were fully empty (`Product Description`) or mostly empty (`Order Zipcode`)
- Dropped personally identifiable fields (customer name, email, password, street) for privacy
- Dropped redundant ID, location, and order-item columns not needed for the analysis
- Excluded orders with **"Shipping canceled"** status, since they have no delivery outcome
- Parsed order and shipping timestamps into proper datetime fields

**Engineered features**

| Feature | Definition |
|---------|------------|
| `Order Processing Time` | Shipping date minus order date, in days |
| `Delay` | Processing time minus scheduled shipment days (positive = late) |
| `Is_Delayed` | Boolean flag where `Delay > 0` |
| `order_month`, `order_day`, `order_hour` | Calendar and time-of-day features from the order timestamp |
| Profit flag | Each order labeled Profit, Loss, or Break-even from profit per order |

---

## Exploratory Data Analysis

The notebook includes the following analyses:

- **Profit distribution:** share of profitable, loss-making, and break-even orders
- **Delay distribution:** percentage of orders by days early or late
- **Profit by delay days:** total and mean profit for each delay bucket
- **Delay rate by category:** order region, customer segment, shipping mode, order status, payment type, and department
- **Regional driver analysis:** a reusable function that ranks the top factors behind late delivery for any region (demonstrated on Central Africa)
- **Time-based analysis:** delay rate by month, day of week, and hour of day

![Delay Distribution]
<img width="1590" height="590" alt="delay_distribution" src="https://github.com/user-attachments/assets/868aae69-9c09-4cea-b488-b07d0da6c934" />


![Delay by Category](Images/delay_by_category.png)
![Time Analysis](Images/time_analysis.png)
-->

---

## Key Insights

- **More than half of all orders arrive late.** The late rate is 54.71%, leaving only 45.29% on time.
- **Most delays are short.** A one-day delay is the most common outcome (31% of orders), and the 90th percentile among late orders is 3 days.
- **Delays do not reduce profit per order.** Average profit per order stays around $20 to $23 regardless of delay length, so the cost of lateness is more likely to appear in customer experience than in margin per order.
- **Profitability is healthy overall.** About 81% of orders are profitable, though roughly 19% lose money, which is worth investigating separately.
- **Delay rates are very uniform across segments and seasons.** Late rates sit between 54.5% and 55.4% across customer segments and between 53.7% and 55.4% across months, which points to a **systemic process issue rather than a seasonal or segment-specific one**.
- **Scheduling is a major lever.** Most orders (about 60%) are scheduled under Standard Class with a 4-day window, so shipping mode and scheduled days are central to delay performance.

---

## Machine Learning Model

**Objective:** Predict `Late_delivery_risk` (late vs not late) using only information available at order time.

**Features:** payment type, scheduled shipment days, category, customer segment, department, order region, shipping mode, order month, and order hour.

**Approach**
- **Encoding:** frequency encoding for categorical variables
- **Split:** 80/20 stratified train-test split (`random_state=42`)
- **Class imbalance:** SMOTE applied to the training set only (79,182 samples per class after balancing)
- **Model:** Random Forest Classifier

**Results (test set, 34,553 orders)**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| On time (0) | 0.68 | 0.73 | 0.70 |
| Late (1) | 0.79 | 0.75 | 0.77 |
| **Overall accuracy** | | | **0.74** |

The model correctly flags about **three out of four late orders** before they ship, giving operations teams a window to intervene.

---

## Recommendations

1. **Review scheduled delivery windows.** With over half of orders late and delays concentrated at 1 to 2 days, promised timelines may be unrealistically tight.
2. **Fix the process, not the segment.** Uniform delay rates across segments and months suggest focusing on warehouse, carrier, and fulfilment workflows.
3. **Deploy the risk model as an early-warning tool.** Flag high-risk orders at checkout for expedited handling or proactive customer communication.
4. **Audit loss-making orders.** Roughly 19% of orders lose money, which is a separate opportunity to improve pricing and discounting.
5. **Use the regional driver function** to build region-specific action plans.

---

## Limitations and Future Work

- Model performance (74% accuracy) could improve with **hyperparameter tuning** and comparison against Gradient Boosting, XGBoost, or LightGBM.
- Frequency encoding could be compared with target encoding or one-hot encoding.
- Add **feature importance and SHAP analysis** to explain predictions.
- Predict the **number of delay days** as a regression task rather than a binary outcome.
- Add **geographic and carrier-level data** to find root causes behind regional delays.
- Package the model into a **Streamlit app** or API for real-time scoring.

---

## Tech Stack

- **Language:** Python 3
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn (Random Forest), imbalanced-learn (SMOTE)
- **Environment:** Jupyter Notebook

---

## Repository Structure

```
Supply-Chain-Delivery-Delay-Analysis/
│
├── Data/
│   └── DataCoSupplyChainDataset.csv      # download separately
│
├── Notebook/
│   └── supply_chain.ipynb
│
├── Images/
│   └── (chart screenshots)
│
├── requirements.txt
└── README.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Supply-Chain-Delivery-Delay-Analysis.git
   cd Supply-Chain-Delivery-Delay-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```

3. **Download the dataset** (`DataCoSupplyChainDataset.csv`) from [Kaggle / Mendeley Data link] and place it in the same folder as the notebook, or update the path in the notebook.

4. **Launch the notebook**
   ```bash
   jupyter notebook supply_chain.ipynb
   ```

> The dataset must be read with `encoding='latin-1'`, which is already set in the notebook.

---

<div align="center">

If you found this project useful, please consider giving it a star.

</div>
