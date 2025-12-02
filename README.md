# UberEats Pricing & Restaurant Classification Models

This project builds two complete machine learning pipelines using the Uber Eats USA Restaurants & Menus Dataset.
It includes full preprocessing, feature engineering, model training, evaluation, and visualization—including the confusion matrix below.

## Confusion Matrix (Top 10 Classes)

![Confusion Matrix](matrix.png)

## Project Description
This project uses the Uber Eats USA restaurant and menu dataset to study pricing patterns and classify restaurants into categories based on menu content.

Two ML tasks are included:

1. **Menu Price Prediction** — Regression  
2. **Restaurant Category Classification** — Text-based multi-class classification over 20 categories  

The workflow includes:
- Data ingestion  
- Cleaning & standardization  
- Merging datasets  
- Full EDA  
- Feature engineering  
- TF‑IDF text vectorization  
- Model comparisons  
- Saving models and preprocessors  

## Task 2 Results Summary

**Train set:** 12133  
**Test set:** 3034  
**Number of categories:** 20  

### Best Model: Logistic Regression
- Test Accuracy: 0.8088  
- F1‑Score: 0.8014  
- Precision: 0.8025  
- Recall: 0.8088  

### Model Comparison
| Model               | Test Accuracy | Test F1 |
|--------------------|--------------|---------|
| Logistic Regression | 0.8088       | 0.8014  |
| Linear SVM         | 0.8045       | 0.7950  |
| Random Forest      | 0.7989       | 0.7771  |

The Logistic Regression classifier provides the strongest generalization across all 20 restaurant categories.
