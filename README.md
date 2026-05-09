# Black Friday Sales Exploratory Data Analysis (EDA)

## Project Overview

This project performs an Exploratory Data Analysis (EDA) on the Black Friday Sales dataset to uncover customer purchasing behaviour, product trends, and business insights that can support data-driven retail decisions.

The analysis focuses on understanding:

* Customer demographics and spending patterns
* Product category performance
* Purchase distribution and outliers
* Relationships between customer features and purchase amount
* Correlation between numerical variables

The project was built using Python and common data analysis libraries such as Pandas, NumPy, Matplotlib, and Seaborn.

---

## Business Problem

Retail businesses generate large volumes of transactional data during Black Friday sales events. Understanding customer behaviour and purchasing trends is important for:

* Improving targeted marketing campaigns
* Identifying high-value customer groups
* Optimising product recommendations
* Enhancing inventory planning
* Increasing overall sales performance

This project explores the dataset to identify patterns that can help businesses make strategic retail decisions.

---

## Dataset Information

The dataset contains customer demographic information, product categories, and purchase amounts.

### Key Features

| Feature                    | Description                                     |
| -------------------------- | ----------------------------------------------- |
| User_ID                    | Unique customer identifier                      |
| Product_ID                 | Unique product identifier                       |
| Gender                     | Customer gender                                 |
| Age                        | Age group of customer                           |
| Occupation                 | Occupation category                             |
| City_Category              | Customer city category                          |
| Stay_In_Current_City_Years | Number of years customer stayed in current city |
| Marital_Status             | Marital status                                  |
| Product_Category_1         | Primary product category                        |
| Product_Category_2         | Secondary product category                      |
| Product_Category_3         | Third product category                          |
| Purchase                   | Purchase amount                                 |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Loading

* Imported required libraries
* Loaded the Black Friday dataset using Pandas

### 2. Data Understanding

* Checked dataset shape
* Reviewed data types and missing values
* Generated descriptive statistics

### 3. Data Cleaning

* Handled missing values in product category columns
* Removed rows with large missing data where necessary

### 4. Exploratory Data Analysis

Performed multiple visual and statistical analyses including:

* Histograms for numerical features
* Boxplots for outlier detection
* Purchase distribution analysis
* Correlation heatmap
* Feature relationship analysis against purchase amount

### 5. Insights Generation

Generated insights from customer demographics and purchasing patterns.

---

## Key Insights

### Customer Spending Behaviour

* Certain age groups contribute significantly more to total purchases.
* Purchase amount distribution is positively skewed, indicating the presence of high spenders.

### Product Category Trends

* Some product categories show stronger relationships with purchase values.
* Product category combinations influence customer spending patterns.

### Outlier Detection

* Boxplots revealed several high-value purchases that may represent premium customers or bulk purchases.

### Correlation Analysis

* Numerical feature relationships were explored using correlation heatmaps.
* Some product category variables showed moderate relationships with purchase amount.

---

## Example Visualisations

The project includes:

* Histograms
* Boxplots
* Correlation heatmaps
* Distribution plots
* Feature comparison charts

---

## Repository Structure

```text
Black-Friday-Sales-EDA/
│
├── Black_Friday_Sales_EDA.ipynb
├── README.md
├── requirements.txt
├── dataset/
│   └── train.csv
└── images/
    ├── purchase_distribution.png
    ├── heatmap.png
    └── boxplot.png
```


## Future Improvements

Potential future enhancements include:

* Feature engineering
* Predictive modelling for purchase amount forecasting
* Customer segmentation
* Recommendation system development
* Interactive dashboard creation using Power BI or Tableau
<img width="713" height="477" alt="output_10_1" src="https://github.com/user-attachments/assets/ae362b8f-1041-47ac-ba46-cbbc173c449a" />
<img width="894" height="651" alt="output_9_0" src="https://github.com/user-attachments/assets/633057d6-5543-4559-b408-668aa8e3d3bb" />
<img width="732" height="468" alt="output_8_0" src="https://github.com/user-attachments/assets/ee996a73-0f55-49ab-b243-a0e55aa8e8db" />
<img width="587" height="433" alt="output_7_0" src="https://github.com/user-attachments/assets/47a54320-4c25-47f8-9bdf-1f83f7915ad7" />

<img width="687" height="390" alt="output_6_1" src="https://github.com/user-attachments/assets/b95affde-4a61-410a-8357-e9d8a8ac2ebd" />
<img width="1004" height="913" alt="output_4_0" src="https://github.com/user-attachments/assets/83c21e44-e6f4-4dcf-9291-2f547a07a8b7" />


---

## Author

Ogodo Barnabas

Data Analyst | Data Scientist | Machine Learning Enthusiast
