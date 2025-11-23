# UberEats Pricing & Restaurant Classification Models

End-to-end ML pipelines for the Uber Eats USA Restaurants & Menus dataset.  
The work covers regression to estimate average menu prices and NLP-driven multi-class classification to assign restaurants to Uber Eats cuisine categories.

## Dataset
- **Source:** [Uber Eats USA Restaurants & Menus Dataset](https://www.kaggle.com/datasets/), containing restaurant-level metadata and scraped menus.  
- **Observation count:** 15k+ restaurants after cleaning.  
- **Menu text:** Individual items concatenated per restaurant, allowing TF-IDF and n-gram modeling.  
- **Labels:** Price buckets for regression, 20 cuisine/category labels for classification.

## Repository Contents
- `README.md` – documentation of the pipelines, metrics, and how to reproduce the work.  
- `matrix.png` – confusion matrix visualizing the top-10 restaurant categories from the classification task.

## Modeling Tasks

### 1. Menu Price Prediction (Regression)
- **Goal:** Predict the median menu item price for each restaurant.  
- **Inputs:** Menu item descriptions, number of menu items, rating count, delivery estimates, and engineered signals such as unique ingredient counts.  
- **Feature engineering:** Numeric scaling, one-hot encoding for city/state, and TF-IDF vectors on menu text plus metadata concatenation.  
- **Algorithms evaluated:** Linear Regression, ElasticNet, Gradient Boosted Trees, and Random Forests.  
- **Evaluation:** 80/20 train-test split, MAE & RMSE. ElasticNet provided the most stable generalization with sub-$1.50 MAE on the held-out set.

### 2. Restaurant Category Classification (NLP)
- **Goal:** Map every restaurant to its primary Uber Eats cuisine label using menu text only.  
- **Text pipeline:**  
  1. Tokenization + stopword removal  
  2. Lemmatization (spaCy)  
  3. Character + word level TF-IDF (1-3 n-grams)  
  4. Max-feature cap at 50k for tractable training  
- **Class balance:** 20 categories; imbalance addressed with class weights.  
- **Dataset split:** 12,133 train / 3,034 test examples.  
- **Best model:** Logistic Regression (saga) with L2 regularization.  

| Model               | Test Accuracy | Test F1 |
|---------------------|---------------|---------|
| Logistic Regression | 0.8088        | 0.8014  |
| Linear SVM          | 0.8045        | 0.7950  |
| Random Forest       | 0.7989        | 0.7771  |

**Precision:** 0.8025 **Recall:** 0.8088 **Number of categories:** 20  
The confusion matrix below highlights misclassifications concentrated between similar cuisine types (e.g., Mexican vs. Tex-Mex).

![Confusion Matrix](matrix.png)

## Pipeline Overview
1. **Ingest:** Load raw CSVs, normalize column names, drop noisy entries, unify time zones.  
2. **Clean:** Remove duplicates, impute missing price/rating signals, merge restaurant and menu tables.  
3. **EDA:** Explore price distributions, ingredient frequencies, text length, and class imbalance.  
4. **Engineer:** Create price bucket labels, encode location, compute menu diversity scores, and build TF-IDF matrices.  
5. **Train:** Compare baseline and advanced models with cross-validation; log metrics via MLflow.  
6. **Evaluate:** Produce regression error plots and classification confusion matrices (sample shown above).  
7. **Persist:** Save scalers, vectorizers, and trained models with `joblib` for downstream inference.

## How to Reproduce
```bash
# 1. Clone the repo and enter it.
git clone https://github.com/<your-user>/ubereats-ml.git
cd ubereats-ml

# 2. Create a Python environment.
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (adjust as needed).
pip install pandas numpy scikit-learn scipy nltk spacy seaborn matplotlib joblib

# 4. Download the Uber Eats dataset to data/raw/ and update paths in the notebooks/scripts.

# 5. Run the training scripts or notebooks for both tasks.
python scripts/train_regression.py
python scripts/train_classification.py

# 6. Regenerate evaluation plots (confusion matrix, error plots).
python scripts/plot_confusion_matrix.py
```
> The repo currently stores documentation and the classification confusion matrix. Add your training scripts/notebooks under `scripts/` or `notebooks/` before running the commands above.

## Future Enhancements
- Add hyperparameter sweeps with Optuna for both models.  
- Deploy lightweight FastAPI endpoints for real-time predictions.  
- Incorporate delivery fee, prep time, and user rating sentiment for richer features.  
- Experiment with transformer-based text encoders (DistilBERT) for classification.
