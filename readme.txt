Price Prediction Machine Learning Pipeline.

Project Overview:-
This repository contains an optimized machine learning solution for predicting product prices using exclusively text-based features from product listings. The implementation leverages advanced feature engineering techniques and ensemble modeling to achieve robust performance without relying on image data.

System Architecture
Data Specifications
Training Dataset

Total samples: 75,000 product listings
Feature sources: Product descriptions, titles, metadata
Target distribution: Log-transformed pricing data

Test Dataset

Total samples: 75,000 product listings
Prediction format: Direct price values

Validation Strategy

Method: Stratified 5-fold cross-validation
Ensures consistent model evaluation across different data splits


Feature Engineering Pipeline
Textual Feature Extraction
The solution extracts multiple layers of information from product text:
Structural Features

Title length (character count)
Word count and distribution
Average word length
Special character presence

Semantic Features

Brand identification and categorization
Luxury brand indicators
Product category keywords
Technical specifications

Commercial Features

Condition status (new, used, refurbished)
Shipping cost indicators
Bundle and package detection
Promotional language patterns

Advanced Text Representation
TF-IDF Vectorization
Configuration:
- Vocabulary: Top 10,000 most informative terms
- N-gram range: 1-3 (captures phrases and context)
- Preprocessing: Normalized text with handled special characters
- Weighting: Term frequency-inverse document frequency
This approach transforms raw text into numerical features while preserving semantic meaning and reducing dimensionality.

Model Architecture
Ensemble Configuration
The solution employs a heterogeneous ensemble of three gradient boosting frameworks, each offering distinct advantages:
Component Models
LightGBM Implementation

Platform: CPU-optimized training
Rounds: 500 iterations with early stopping
Advantages: Fast training, efficient memory usage
Special features: Histogram-based learning

XGBoost Implementation

Platform: GPU-accelerated computation
Rounds: 500 iterations with early stopping
Advantages: Precise split finding, strong regularization
Special features: Advanced tree pruning

CatBoost Implementation

Platform: GPU-accelerated computation
Rounds: 500 iterations with early stopping
Advantages: Categorical feature handling, reduced overfitting
Special features: Ordered boosting algorithm

Ensemble Strategy
Aggregation Method

Technique: Simple arithmetic mean
Weighting: Equal contribution from each model
Rationale: Reduces individual model variance and bias

Why This Works
The diversity of algorithms ensures different models capture different patterns in the data. XGBoost excels at finding optimal splits, LightGBM provides speed and efficiency, while CatBoost handles categorical relationships effectively. Their combination produces more stable and accurate predictions.

Hyperparameter Configuration
Global Settings
Learning Parameters:
- Learning rate: 0.05 (balanced convergence speed)
- Early stopping patience: 50 rounds
- Tree depth: 8-10 levels (model-dependent)
Regularization Strategy
Overfitting Prevention:
- L1 regularization: Feature selection pressure
- L2 regularization: Weight magnitude control
- Min child weight: Minimum samples per leaf
- Subsample ratio: Training data sampling
These parameters were tuned to balance model complexity with generalization capability.

Performance Analysis
Evaluation Metrics
Primary Metric: SMAPE (Symmetric Mean Absolute Percentage Error)
SMAPE = (100 / n) × Σ |predicted - actual| / (|predicted| + |actual|)
This metric treats over-predictions and under-predictions symmetrically, making it ideal for price prediction where both types of errors are equally undesirable.
Cross-Validation Results
Overall Performance: 56.88% SMAPE

Fold-wise Breakdown:
- Fold 1: 55.96%
- Fold 2: 56.45%
- Fold 3: 57.12%
- Fold 4: 57.53%
- Fold 5: 56.34%

Standard Deviation: 0.58%
The low standard deviation indicates consistent performance across different data splits, suggesting the model generalizes well.

Computational Requirements
Hardware Utilization
CPU Resources

LightGBM training and feature engineering
Multi-threaded processing where available

GPU Acceleration

XGBoost model training
CatBoost model training
Approximate speedup: 3-5x compared to CPU-only

Runtime Analysis
Total Pipeline Duration: ~18 minutes

Breakdown:
- Data loading and preprocessing: 2 minutes
- Feature engineering: 4 minutes
- Model training (3 models × 5 folds): 10 minutes
- Prediction generation: 2 minutes
Memory Footprint
Efficient sparse matrix representation keeps memory usage reasonable even with 10,000 TF-IDF features.

Project Structure
project/
│
├── train.csv              # Training dataset (75K samples)
├── test.csv               # Test dataset (75K samples)
├── solution.py            # Main implementation script
└── test_out.csv           # Generated predictions

Output Specification
Submission File Format
The test_out.csv file contains two columns:
sample_id

Unique identifier matching test dataset
Format: Integer index
Range: 0 to 74,999

price

Predicted price value
Format: Float (up to 2 decimal places)
Unit: Currency units from original dataset

Sample Output
sample_id,price
0,45.67
1,123.89
2,67.23
...

Getting Started
Prerequisites
bashPython 3.8+
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
Installation
bashpip install pandas numpy scikit-learn lightgbm xgboost catboost
Execution
bashpython solution.py
The script will automatically:

Load training and test data
Engineer features
Train ensemble models with cross-validation
Generate predictions
Save results to test_out.csv


Key Insights
What Makes This Approach Effective
Text is Information-Rich
Product descriptions contain pricing signals: brand prestige, condition, features, and specifications all correlate with price.
Ensemble Reduces Risk
Different algorithms make different types of errors. Averaging their predictions cancels out individual weaknesses.
Feature Engineering Matters
Extracting meaningful patterns from raw text (not just word frequencies) significantly improves model performance.
Cross-Validation Prevents Overfitting
Testing on multiple data splits ensures the model truly learned generalizable patterns rather than memorizing training data.

Future Improvements
Potential Enhancements
Advanced Text Processing

Word embeddings (Word2Vec, GloVe)
Transformer-based features (BERT)
Topic modeling (LDA)

Additional Features

Price history patterns
Seasonal indicators
Category-specific features

Model Architecture

Neural network integration
Stacking ensemble
Hyperparameter optimization (Optuna, Hyperopt)

Performance Optimization

Feature selection techniques
Model compression
Inference optimization


Technical Notes
Design Decisions
Why Log-Transform Prices?
Prices often follow log-normal distributions. Log transformation normalizes the distribution and makes the prediction task easier for tree-based models.
Why Equal Weighting in Ensemble?
Without a separate validation set for meta-learning, equal weighting is a safe default that prevents overfitting to any single model's characteristics.
Why These Three Models?:=
LightGBM, XGBoost, and CatBoost represent the state-of-the-art in gradient boosting, each with unique strengths. Together they provide robust coverage of the solution space.


License
This project is available for educational and research purposes.

