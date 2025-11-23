# UberEats Pricing & Restaurant Classification Models

This project builds two complete machine learning pipelines using the Uber Eats USA Restaurants & Menus Dataset. It includes full data ingestion, cleaning, merging, feature engineering, exploratory analysis, model training, evaluation, and production-ready preprocessing pipelines for both regression and classification tasks.

The full end-to-end notebook is available here:
https://www.kaggle.com/code/yousefzahran1/ubereats-pricing-and-classification-models-ipynb

---

## Project Description

This project uses the Uber Eats USA restaurant and menu dataset to study pricing patterns and classify restaurants into categories based on menu content. It implements two full machine learning objectives:

1. **Menu Price Prediction (Regression)**  
   Predict the price of a menu item using restaurant metadata, menu text, and aggregated statistics.

2. **Restaurant Category Classification (NLP-based Multi-Class Classification)**  
   Classify restaurants into 20 categories based on aggregated menu descriptions, numeric features, and metadata.

The project follows a real-world ML workflow:
- Data ingestion
- Cleaning inconsistent fields (price strings, missing values, malformed text)
- Merging two large datasets efficiently
- Exploratory Data Analysis (EDA)
- Feature engineering for numeric, categorical, and text fields
- TF-IDF vectorization across menu text
- Model training + evaluation
- Saving trained preprocessors and best-performing models

This project demonstrates strong practical ML engineering skills, including data preparation, text modeling, pipeline design, and regression/classification modeling.

---

# Dataset Contents

**restaurants.csv**
- restaurant_id (unique identifier)
- restaurant_name, restaurant_categories
- customer_rating, customer_review_count, restaurant_price_range
- restaurant_latitude, restaurant_longitude
- estimated_delivery_time_metadata

**restaurant-menus.csv**
- restaurant_id (foreign key linking to restaurants.csv)
- menu_item_name, menu_item_description
- menu_item_price
- menu_item_category

Both files are merged on **restaurant_id**.

---

# 1. Data Ingestion, Loading & Cleaning Process

### Data Cleaning Steps:
- Import both CSV files
- Clean column names (lowercase, standardized)
- Remove currency and non-numeric characters from price
- Normalize text fields
- Handle missing values and duplicates
- Remove entries with no usable menu descriptions
- Convert text features into consistent format

---

# 2. Multi-Dataset Merging & Exploratory Data Analysis

### Exploratory Data Analysis Performed:
- Merge restaurant and menu datasets
- Category distribution
- Price distribution across cuisines
- Outlier detection using IQR
- Correlation analysis (numeric)
- Text-length distributions
- Restaurant statistics distribution
- Top cuisines by volume

All major analyses are supported by visualizations in the notebook.

---

# 3. Feature Engineering

## Feature Engineering for Menu Item Price Prediction (Regression Pipeline)

- TF-IDF text vectorization over combined item name + description
- Frequency encoding for restaurant category
- Price range encoding
- Text length features:
  - item_name_character_length
  - item_description_character_length
  - word_count_per_description
- Numeric aggregation features:
  - restaurant_average_item_price
  - category_average_price
  - items_count_per_restaurant_category
- Log-scaled features for skewed distributions

Resulting feature vector: **hundreds of engineered features per menu item**

---

## Feature Engineering for Restaurant Classification (Text-Based Multi-Class Classification Pipeline)

- Aggregate all menu descriptions per restaurant
- Clean and normalize text
- TF-IDF vectorization at 500 features
- Aggregated statistical features per restaurant:
  - restaurant_mean_menu_item_price
  - restaurant_min_menu_item_price
  - restaurant_max_menu_item_price
  - restaurant_price_standard_deviation
  - restaurant_total_menu_items_count

Dataset transformed to:
- **Train shape:** (12133, 512)
- **Test shape:**  (3034, 512)

---

# 4. Machine Learning Models

# Machine Learning Pipeline 1: Menu Item Price Prediction (Regression)

### Regression Models Trained and Evaluated:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Regression Evaluation Metrics:
- RMSE
- MAE
- R² score
- Training time

### Outputs:
- `price_prediction_best_model.pkl`
- `price_prediction_preprocessor.pkl`
- `price_prediction_X_train_features.csv`, `price_prediction_y_train_labels.csv`
- `price_prediction_model_comparison.csv`

---

# Machine Learning Pipeline 2: Restaurant Multi-Category Classification (20 Classes)

### Training and Test Dataset Specifications:
- Train: (12133 samples, 13 raw columns → 512 transformed features)
- Test:  (3034 samples)
- Total classes: **20**

### Label Encoding and Class Balancing Strategy:
- LabelEncoder applied to all 20 categories
- Highly imbalanced categories handled with weighting

---

# 5. Classification Model Training Results and Performance Metrics

## Logistic Regression Classification Model (Best Performing Model)
- Train Accuracy: **0.8255**
- Test Accuracy: **0.8088**
- Test F1-Score: **0.8014**
- Precision: 0.8025
- Recall: 0.8088
- Training Time: 13.44s

## Linear Support Vector Machine Classification Model
- Test Accuracy: **0.8045**
- Test F1-Score: **0.7950**
- Training Time: 11.77s

## Random Forest Classification Model
- Test Accuracy: **0.7989**
- Test F1-Score: **0.7771**
- Training Time: 6.97s

---

# Multi-Model Performance Comparison Table

| Classification Model | Training Accuracy | Testing Accuracy | Training F1-Score | Testing F1-Score | Training Precision | Testing Precision | Training Recall | Testing Recall | Model Training Time |
|--------------------|----------------|---------------|----------|---------|------------------|----------------|--------------|-------------|------------|
| Logistic Regression | 0.8255         | 0.8088        | 0.8183   | 0.8014  | 0.8241           | 0.8025         | 0.8255       | 0.8088      | 13.44s     |
| Linear SVM         | 0.8342         | 0.8045        | 0.8249   | 0.7949  | 0.8328           | 0.7941         | 0.8342       | 0.8045      | 11.77s     |
| Random Forest      | 0.9222         | 0.7989        | 0.9198   | 0.7770  | 0.9316           | 0.8207         | 0.9222       | 0.7989      | 6.96s      |

---

# BEST PERFORMING CLASSIFICATION MODEL
**Logistic Regression Classification Model**  
Testing Accuracy: **0.8088**  
Testing F1-Score: **0.8014**  
Testing Precision: **0.8025**  
Testing Recall: **0.8088**

---

# Full Detailed Classification Report (Logistic Regression Model)

precision recall f1-score support

American, Breakfast and Brunch, Desserts 0.66 0.66 0.66
American, Burgers, Fast Food 0.84 0.93 0.88
American, Burgers, Sandwiches 0.43 0.22 0.29
American, Fast Food, Burgers 0.87 0.83 0.85
American, burger, Fast Food 0.90 0.76 0.82
American, burger, Fast Food, Family Meals 0.97 0.97 0.97
Bakery, Breakfast and Brunch, Cafe 0.96 1.00 0.98
Breakfast and Brunch, American, Sandwiches 0.43 0.33 0.38
Burgers, American, Sandwiches 0.65 0.80 0.71
Burritos, Fast Food, Mexican 1.00 1.00 1.00
Chinese, Asian, Asian Fusion 0.92 0.94 0.93
Coffee and Tea, American, Breakfast 0.74 0.69 0.71
Everyday Essentials, Convenience, Snacks 1.00 1.00 1.00
Fast Food, Sandwich, American 1.00 1.00 1.00
Italian, Pasta, Comfort Food 0.67 0.47 0.56
Mexican, Latin American, New Mexican 0.90 0.93 0.91
Pharmacy, Convenience, Everyday Essentials 1.00 1.00 1.00
Pizza, American, Italian 0.75 0.78 0.77
Sandwiches, American, Healthy 0.62 0.44 0.52
Seafood, American, Southern 0.79 0.68 0.73

Overall Accuracy: 0.81
Macro F1: 0.78
Weighted Avg F1: 0.80


---

# Project Value and Key Technical Achievements

- Real-world data cleaning and preprocessing  
- Strong feature engineering with text, numeric, and categorical signals  
- TF-IDF modeling over restaurant menus  
- Multiple models evaluated with meaningful metrics  
- End-to-end ML pipeline with saved models and preprocessors  
- Fully reproducible notebook  
- Production-oriented mindset suitable for ML engineering roles  

---

