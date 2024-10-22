# Predictive-Customer-Churn-Analysis-Using-Machine-Learning-to-Improve-Customer-Retention

## Project Overview:
Customer churn is a critical concern for businesses, especially in the telecommunications industry. This project aims to develop a **machine learning-based prediction system** to identify customers who are likely to churn. By accurately predicting customer churn, businesses can implement proactive retention strategies, thereby reducing customer loss and increasing revenue.

## Key Objectives:
1. **Predict Customer Churn**: Build a model that accurately predicts whether a customer will churn based on historical data.
2. **Feature Importance**: Understand which customer attributes (e.g., age, subscription length, monthly bill) are most indicative of churn.
3. **Optimize Model Performance**: Compare multiple machine learning algorithms to find the best model with high accuracy and minimal overfitting.

## Dataset:
The dataset contains various customer-related information, including:
- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Male or Female.
- **Location**: Geographical location (e.g., Houston, Los Angeles).
- **Subscription Length**: Number of months the customer has been subscribed.
- **Monthly Bill**: Monthly bill amount.
- **Total Usage (GB)**: Total service usage in gigabytes.
- **Churn**: Whether the customer has churned (1) or not (0).

## Approach:

### 1. Data Preprocessing:
- Removed irrelevant columns like `CustomerID`.
- Handled missing values and ensured data consistency.
- Scaled continuous features such as `Age`, `Subscription Length`, `Monthly Bill`, and `Total Usage GB` using **MinMaxScaler**.
- One-hot encoded categorical variables like `Gender` and `Location`.

### 2. Exploratory Data Analysis (EDA):
- **Statistical Summary**: Described key numerical variables (e.g., average customer age of 44 years).
- **Correlation Analysis**: Examined relationships between features using correlation heatmaps.
- **Class Imbalance**: Checked for class balance, finding an approximately 50/50 split between churned and non-churned customers.

### 3. Feature Selection:
- Applied **Random Forest** for feature importance. The most important features were:
  1. Monthly Bill
  2. Total Usage (GB)
  3. Age
  4. Subscription Length

### 4. Model Building:
- Trained multiple machine learning models, including:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
  - XGBoost
  - Support Vector Machine (SVM)
  - Neural Networks
- Evaluated models based on accuracy, precision, recall, F1-score, and training time.
- Addressed overfitting using cross-validation and feature reduction techniques.
- Introduced **Neural Networks** with early stopping and model checkpointing to further improve performance.

## Key Results:
- **Random Forest and Decision Tree**: Achieved perfect accuracy on training data, indicating possible overfitting.
- **Logistic Regression, KNN, and XGBoost**: Showed moderate performance.
- **Neural Network**: Provided better accuracy with techniques to avoid overfitting.

### Visualizations:
- **Correlation Heatmap**: Displayed relationships between features.
- **Class Distribution**: Illustrated the balanced churn distribution.
- **Feature Importance Plot**: Highlighted the importance of features like Monthly Bill and Total Usage in predicting churn.

## Conclusion:
The **Random Forest** model demonstrated the highest accuracy on training data, but steps were taken to avoid overfitting. A refined **Neural Network** model was introduced to further improve prediction accuracy. This project is a valuable tool for predicting customer churn and provides actionable insights for businesses to implement effective customer retention strategies.

## Future Work:
- Fine-tune the **Neural Network** model to further enhance performance.
- Experiment with advanced techniques such as **ensemble learning** and **hyperparameter tuning**.
- Consider additional business-specific features to boost prediction accuracy.
