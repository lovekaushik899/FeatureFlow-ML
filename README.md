# FeatureFlow-ML

A comprehensive machine learning framework providing optimized classification and regression pipelines with automated feature engineering, preprocessing, and model evaluation capabilities.

## 🎯 Overview

FeatureFlow-ML streamlines machine learning workflows by providing production-ready pipelines for both classification and regression tasks. It abstracts away boilerplate code while maintaining full flexibility for custom configurations and advanced use cases.

**Key Strengths:**
- ⚡ **Fast Implementation**: Build ML models with minimal code
- 🔄 **Automated Pipelines**: Complete data-to-prediction workflows
- 📊 **Multiple Algorithms**: Ensemble and traditional models
- 🎛️ **Configurable**: Fine-grained control over pipeline stages
- 📈 **Performance Tracking**: Built-in evaluation and metrics
- 🔍 **Interpretability**: Feature importance and model explanations

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Core Modules](#core-modules)
- [Classification Pipeline](#classification-pipeline)
- [Regression Pipeline](#regression-pipeline)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Algorithm Comparison](#algorithm-comparison)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Contributing](#contributing)

## 🚀 Quick Start

### Classification in 5 Lines

```python
from classification_pipeline import ClassificationPipeline

pipeline = ClassificationPipeline(model_type='random_forest')
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression in 5 Lines

```python
from regression_pipeline import RegressionPipeline

pipeline = RegressionPipeline(model_type='xgboost')
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
r2_score = pipeline.evaluate(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

## 📦 Installation

### Prerequisites
- Python 3.7+
- pip or conda

### From Source

```bash
git clone https://github.com/lovekaushik899/FeatureFlow-ML.git
cd FeatureFlow-ML
pip install -r requirements.txt
```

### Dependencies

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

**Dependency Breakdown:**
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms and utilities
- `xgboost` - Gradient boosting framework
- `lightgbm` - Light gradient boosting (alternative to XGBoost)
- `matplotlib`, `seaborn` - Visualization

## 📚 Core Modules

### ClassificationPipeline

Handles multi-class and binary classification tasks with automated preprocessing and multiple algorithm support.

**Supported Algorithms:**
- Logistic Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Light Gradient Boosting (LightGBM)
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)

### RegressionPipeline

Handles continuous target prediction with support for linear and non-linear models.

**Supported Algorithms:**
- Linear Regression
- Ridge/Lasso Regression
- Random Forest Regression
- Gradient Boosting (XGBoost)
- Light Gradient Boosting (LightGBM)
- Support Vector Regression (SVR)

## 🔧 Classification Pipeline

### Basic Usage

```python
from classification_pipeline import ClassificationPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize pipeline
pipeline = ClassificationPipeline(
    model_type='random_forest',
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Predictions: {predictions}")
```

### Advanced Configuration

```python
pipeline = ClassificationPipeline(
    model_type='xgboost',
    # Preprocessing
    handle_missing='mean',           # 'mean', 'median', 'drop'
    scale_features=True,              # Standardize features
    encode_categorical=True,          # One-hot encoding
    # Model parameters
    n_estimators=150,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    # Training
    validation_split=0.2,
    early_stopping_rounds=20,
    random_state=42
)

# Cross-validation
cv_scores = pipeline.cross_validate(X_train, y_train, cv=5)
print(f"CV Scores: {cv_scores}")

# Feature importance
importance = pipeline.feature_importance()
print(f"Top features: {importance.head()}")
```

## 📈 Regression Pipeline

### Basic Usage

```python
from regression_pipeline import RegressionPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('housing_data.csv')
X = data.drop('price', axis=1)
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize pipeline
pipeline = RegressionPipeline(
    model_type='xgboost',
    n_estimators=100
)

# Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
r2 = pipeline.evaluate(X_test, y_test, metric='r2')
rmse = pipeline.evaluate(X_test, y_test, metric='rmse')
mae = pipeline.evaluate(X_test, y_test, metric='mae')

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
```

### Advanced Configuration

```python
pipeline = RegressionPipeline(
    model_type='lightgbm',
    # Preprocessing
    handle_missing='median',
    scale_features=True,
    outlier_removal=True,           # Remove statistical outliers
    outlier_threshold=3,             # Standard deviations
    # Model parameters
    n_estimators=200,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    # Training
    validation_split=0.2,
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=-1                        # Parallel processing
)

# Evaluation with multiple metrics
metrics = pipeline.evaluate_all(X_test, y_test)
print(metrics)  # {'r2': ..., 'rmse': ..., 'mae': ..., 'mape': ...}
```

## 🔬 Advanced Usage

### 1. Custom Feature Engineering

```python
def custom_features(X):
    X_new = X.copy()
    X_new['feature_ratio'] = X['col1'] / (X['col2'] + 1)
    X_new['feature_interaction'] = X['col1'] * X['col3']
    return X_new

pipeline = ClassificationPipeline(model_type='random_forest')
X_train_engineered = custom_features(X_train)
X_test_engineered = custom_features(X_test)
pipeline.fit(X_train_engineered, y_train)
predictions = pipeline.predict(X_test_engineered)
```

### 2. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.5]
}

pipeline = ClassificationPipeline(model_type='xgboost')
best_params = pipeline.grid_search(X_train, y_train, param_grid, cv=5)
print(f"Best parameters: {best_params}")
```

### 3. Class Imbalance Handling

```python
from imblearn.over_sampling import SMOTE

pipeline = ClassificationPipeline(
    model_type='random_forest',
    handle_imbalance=True,           # Enable SMOTE
    imbalance_sampling_strategy='auto'
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### 4. Model Persistence

```python
import pickle

# Save model
pipeline.save('my_model.pkl')

# Load model
loaded_pipeline = pickle.load(open('my_model.pkl', 'rb'))
predictions = loaded_pipeline.predict(X_test)
```

## 📖 API Reference

### ClassificationPipeline

```python
ClassificationPipeline(
    model_type='random_forest',      # str: Algorithm choice
    n_estimators=100,                 # int: Number of trees/boosting rounds
    max_depth=None,                   # int or None: Max tree depth
    learning_rate=0.1,                # float: Learning rate for boosting
    random_state=None,                # int or None: Random seed
    scale_features=True,              # bool: Standardize features
    handle_missing='mean',            # str: 'mean', 'median', 'drop'
    encode_categorical=True,          # bool: One-hot encode
    validation_split=0.2,             # float: Validation set fraction
    n_jobs=-1                         # int: Parallel jobs (-1 = all cores)
)
```

**Key Methods:**
- `fit(X, y)` - Train the pipeline
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get prediction probabilities
- `evaluate(X, y, metric='accuracy')` - Evaluate performance
- `feature_importance()` - Get feature rankings
- `cross_validate(X, y, cv=5)` - K-fold cross-validation
- `save(filepath)` - Save trained model
- `load(filepath)` - Load trained model

### RegressionPipeline

```python
RegressionPipeline(
    model_type='xgboost',             # str: Algorithm choice
    n_estimators=100,                 # int: Number of boosting rounds
    learning_rate=0.1,                # float: Learning rate
    max_depth=6,                      # int: Max tree depth
    random_state=None,                # int or None: Random seed
    scale_features=True,              # bool: Standardize features
    handle_missing='median',          # str: 'mean', 'median', 'drop'
    outlier_removal=False,            # bool: Remove outliers
    outlier_threshold=3,              # float: Std deviation threshold
    validation_split=0.2,             # float: Validation fraction
    n_jobs=-1                         # int: Parallel jobs
)
```

**Key Methods:**
- `fit(X, y)` - Train the pipeline
- `predict(X)` - Make predictions
- `evaluate(X, y, metric='r2')` - Evaluate performance
- `evaluate_all(X, y)` - Get all metrics (R², RMSE, MAE, MAPE)
- `feature_importance()` - Get feature rankings
- `cross_validate(X, y, cv=5)` - K-fold cross-validation
- `residuals(X, y)` - Get prediction errors
- `save(filepath)` - Save trained model
- `load(filepath)` - Load trained model

## 📊 Algorithm Comparison

### Classification Models

| Algorithm | Speed | Accuracy | Interpretability | Memory | Best For |
|-----------|-------|----------|------------------|--------|----------|
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Linear separable data |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced performance |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Competitive performance |
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Large datasets |
| SVM | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | High-dimensional data |
| KNN | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Small to medium data |

### Regression Models

| Algorithm | Speed | Accuracy | Interpretability | Memory | Best For |
|-----------|-------|----------|------------------|--------|----------|
| Linear Regression | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Linear relationships |
| Ridge/Lasso | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Regularized prediction |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Non-linear patterns |
| XGBoost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Competitive datasets |
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Large datasets |
| SVR | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Non-linear regression |

## ⚡ Performance Optimization

### 1. Data Preprocessing

```python
# Efficient feature scaling
pipeline = ClassificationPipeline(
    scale_features=True,           # Much faster with scaled features
    handle_missing='drop'          # Faster than 'mean'
)
```

### 2. Parallel Processing

```python
# Use all available cores
pipeline = ClassificationPipeline(
    n_jobs=-1,                     # Parallelization
    model_type='xgboost'
)
```

### 3. Early Stopping

```python
# Stop training when validation performance plateaus
pipeline = ClassificationPipeline(
    model_type='xgboost',
    validation_split=0.2,
    early_stopping_rounds=20       # Stop after 20 rounds without improvement
)
```

### 4. Model Selection by Dataset Size

```
Small Dataset (< 10K samples):     Use Logistic Regression or SVM
Medium Dataset (10K - 1M):          Use Random Forest or XGBoost
Large Dataset (> 1M):               Use LightGBM with parallel processing
```

### 5. Memory Optimization

```python
# Reduce memory footprint
pipeline = RegressionPipeline(
    model_type='lightgbm',
    num_leaves=31,                 # Lower = less memory
    max_depth=10,
    n_jobs=-1
)
```

## 🔧 Troubleshooting

### Issue: Out of Memory Error

**Solution 1:** Use LightGBM instead of XGBoost
```python
pipeline = ClassificationPipeline(model_type='lightgbm')
```

**Solution 2:** Reduce model complexity
```python
pipeline = ClassificationPipeline(
    n_estimators=50,               # Fewer trees
    max_depth=5,                   # Shallower trees
    subsample=0.5                  # Use fraction of data
)
```

### Issue: Low Accuracy / R² Score

**Check 1:** Verify data quality
```python
# Check for missing values
print(X_train.isnull().sum())

# Check for class imbalance (classification)
print(y_train.value_counts())
```

**Check 2:** Try different algorithms
```python
for model in ['random_forest', 'xgboost', 'lightgbm']:
    pipeline = ClassificationPipeline(model_type=model)
    pipeline.fit(X_train, y_train)
    score = pipeline.evaluate(X_test, y_test)
    print(f"{model}: {score:.4f}")
```

**Check 3:** Increase model complexity
```python
pipeline = ClassificationPipeline(
    n_estimators=200,              # More trees
    max_depth=15,                  # Deeper trees
    learning_rate=0.05             # Lower rate for finer tuning
)
```

### Issue: Overfitting

**Solution 1:** Regularization
```python
pipeline = ClassificationPipeline(
    model_type='xgboost',
    reg_alpha=1.0,                 # L1 regularization
    reg_lambda=1.0,                # L2 regularization
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Solution 2:** Early stopping
```python
pipeline = ClassificationPipeline(
    validation_split=0.2,
    early_stopping_rounds=15
)
```

### Issue: Underfitting

**Solution:** Increase model capacity
```python
pipeline = ClassificationPipeline(
    n_estimators=300,
    max_depth=15,
    learning_rate=0.1               # Faster learning
)
```

### Issue: Slow Training

**Solution:** Use LightGBM and parallelization
```python
pipeline = ClassificationPipeline(
    model_type='lightgbm',
    n_jobs=-1                       # All CPU cores
)
```

## 💡 Best Practices

### 1. Data Splitting Strategy

```python
from sklearn.model_selection import train_test_split

# Always use stratified split for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. Preprocessing Before Modeling

```python
# Fit preprocessing on training data only
pipeline = ClassificationPipeline()
pipeline.fit(X_train, y_train)         # Scales/encodes based on training data
predictions = pipeline.predict(X_test)  # Applies same transformations
```

### 3. Cross-Validation for Robustness

```python
# Use k-fold CV to assess model stability
cv_scores = pipeline.cross_validate(X, y, cv=5)
print(f"Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 4. Feature Importance Analysis

```python
# Understand which features drive predictions
importance = pipeline.feature_importance()
importance.plot(kind='barh')
plt.title('Feature Importance')
plt.show()
```

### 5. Monitor for Class Imbalance

```python
# Check training data
if y.value_counts().min() < y.value_counts().max() * 0.1:
    print("⚠️ Severe class imbalance detected")
    pipeline = ClassificationPipeline(handle_imbalance=True)
```

### 6. Document Your Pipeline

```python
# Save pipeline configuration for reproducibility
config = {
    'model_type': 'xgboost',
    'n_estimators': 100,
    'max_depth': 8,
    'random_state': 42
}

import json
with open('pipeline_config.json', 'w') as f:
    json.dump(config, f)
```

## 📋 Examples

### Example 1: Binary Classification (Customer Churn)

```python
from classification_pipeline import ClassificationPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('churn_data.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
pipeline = ClassificationPipeline(
    model_type='xgboost',
    n_estimators=150,
    handle_imbalance=True
)
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.evaluate(X_test, y_test)
prob = pipeline.predict_proba(X_test)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sample probabilities: {prob[:5]}")
```

### Example 2: Multi-Class Classification (Iris Dataset)

```python
from classification_pipeline import ClassificationPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
pipeline = ClassificationPipeline(model_type='random_forest')
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.evaluate(X_test, y_test)
importance = pipeline.feature_importance()

print(f"Accuracy: {accuracy:.4f}")
print(f"Top feature: {importance.index[0]}")
```

### Example 3: Regression (Housing Price Prediction)

```python
from regression_pipeline import RegressionPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Load
df = pd.read_csv('housing.csv')
X = df.drop('price', axis=1)
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline = RegressionPipeline(
    model_type='lightgbm',
    n_estimators=200,
    n_jobs=-1
)
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate_all(X_test, y_test)
predictions = pipeline.predict(X_test)

print(f"R² Score: {metrics['r2']:.4f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** changes (`git commit -am 'Add feature'`)
4. **Push** to branch (`git push origin feature/improvement`)
5. **Submit** a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update README with changes

## 📄 License

This project is open-source and available under the MIT License.

## 📚 Resources

- **scikit-learn Documentation**: https://scikit-learn.org/stable/
- **XGBoost Docs**: https://xgboost.readthedocs.io/
- **LightGBM Docs**: https://lightgbm.readthedocs.io/
- **Pandas Guide**: https://pandas.pydata.org/docs/

## 🆘 Support

For issues, questions, or suggestions:
- Open a GitHub issue
- Check existing documentation
- Review example notebooks

---

**Version:** 2.0.0  
**Last Updated:** 2026-03-09  
**Maintainer:** Love Kaushik (@lovekaushik899)
