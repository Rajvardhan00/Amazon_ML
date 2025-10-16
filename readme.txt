Overview
This solution implements an optimized machine learning pipeline for price prediction using only text data from product listings. The approach focuses on robust feature engineering and ensemble modeling to achieve competitive performance without image data.

Technical Approach
Data Processing
Input: 75,000 training samples, 75,000 test samples

Target Variable: Product price (log-transformed for modeling)

Validation: 5-fold cross-validation strategy

Feature Engineering
Text-Based Features
Product Title Analysis: Length, word count, character count

Brand Indicators: Presence of brand mentions, luxury indicators

Product Characteristics: Color, material, size patterns

Condition Analysis: New, used, refurbished status

Shipping Information: Free shipping indicators

TF-IDF Vectorization
Vocabulary Size: 10,000 most frequent words

N-gram Range: Unigrams, bigrams, and trigrams

Text Preprocessing: Lowercasing, special character handling

Model Architecture
Ensemble of Gradient Boosting Models
LightGBM

CPU implementation with early stopping

500 boosting rounds

XGBoost

GPU-accelerated training

500 boosting rounds with early stopping

CatBoost

GPU-accelerated training

500 boosting rounds with early stopping

Ensemble Strategy
Simple averaging of model predictions

Equal weighting across all three models

Performance Metrics
Primary Metric: Symmetric Mean Absolute Percentage Error (SMAPE)

Cross-Validation: 5-fold CV with consistent fold splits

Current Performance: 56.88% SMAPE

Performance Range: 55.96% - 57.53% across folds

Implementation Details
Key Parameters
Learning Rate: 0.05 across all models

Early Stopping: 50 rounds without improvement

Maximum Depth: 8-10 trees

Regularization: L1/L2 regularization to prevent overfitting

Computational Requirements
Runtime: Approximately 18 minutes with GPU acceleration

Memory: Efficient feature representation

GPU Support: XGBoost and CatBoost utilize GPU acceleration

File Structure
text
project/
├── train.csv                 # Training data
├── test.csv                  # Test data
├── solution.py              # Main implementation
└── test_out.csv             # Prediction output
Output Format
The solution generates a submission file with:

sample_id: Unique identifier from test data

price: Predicted price values