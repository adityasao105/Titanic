# Titanic Survival Prediction

An end-to-end machine learning project predicting passenger survival aboard the Titanic using the Kaggle **Titanic: Machine Learning from Disaster** dataset.

This project focuses on building a structured ML workflow—from exploratory data analysis and feature engineering through model selection, cross-validation, hyperparameter tuning, interpretation, and final Kaggle submission.

## Overview

The goal is to predict whether a passenger survived the Titanic disaster using demographic, socioeconomic, and travel information such as:

* Passenger class
* Sex
* Age
* Fare
* Family relationships
* Embarkation port
* Cabin information
* Ticket information

Rather than focusing solely on leaderboard performance, this project explores how different modeling decisions affect generalization and demonstrates a reproducible approach to tabular classification.

## Machine Learning Workflow

The project follows a standard supervised machine learning workflow:

1. **Data exploration and cleaning**
2. **Exploratory data analysis**
3. **Feature engineering**
4. **Baseline modeling**
5. **Cross-validation**
6. **Feature experimentation**
7. **Model comparison**
8. **Hyperparameter tuning**
9. **Model interpretation**
10. **Final training and Kaggle submission**

Preprocessing and modeling are contained within Scikit-learn pipelines to reduce data leakage and make experiments reproducible.

## Exploratory Data Analysis

EDA was used to investigate how passenger characteristics related to survival.

Some of the strongest patterns appeared across:

* **Sex** — female passengers had substantially higher survival rates.
* **Passenger class** — first-class passengers were more likely to survive than third-class passengers.
* **Age** — children showed different survival patterns from adults, particularly among male passengers.
* **Family structure** — passengers traveling in small family groups tended to perform differently from passengers traveling alone or in large groups.
* **Fare and cabin information** — both contained information related to passenger class and socioeconomic status.

These observations were then used to generate hypotheses for feature engineering.

## Feature Engineering

Several candidate features were created from the original variables:

* `FamilySize` — total immediate family members traveling together
* `IsAlone` — whether the passenger traveled without immediate family
* `HasCabin` — whether cabin information was available
* `Deck` — extracted from the passenger's cabin
* `TicketGroupSize` — number of passengers sharing a ticket
* `TicketPrefix` — normalized prefix extracted from ticket numbers
* `IsChild` — indicator for younger passengers
* `MaleChild` — interaction between age and sex motivated by EDA

Engineered features were evaluated using cross-validation rather than automatically being included in the final model.

One important result was that features useful for a linear model were not necessarily useful for tree-based models. For example, the `MaleChild` interaction improved Logistic Regression substantially, while Random Forest could largely learn the same interaction directly from `Sex` and `Age`.

## Models

Three classification algorithms were investigated.

### Logistic Regression

Logistic Regression provided an interpretable baseline and helped evaluate whether explicitly engineered interactions improved a linear model.

### Random Forest

Random Forest captured nonlinear relationships and feature interactions without requiring them to be manually specified.

Hyperparameters including:

* Number of trees
* Maximum tree depth
* Minimum samples per leaf

were evaluated using cross-validation and `GridSearchCV`.

### XGBoost

XGBoost was tested as a boosted-tree alternative.

Experiments included:

* Learning rate
* Number of estimators
* Tree depth
* Minimum child weight
* Subsampling
* Column sampling

Despite its additional complexity, XGBoost did not clearly outperform the simpler alternatives, reinforcing that more complex models are not automatically better.

## Evaluation

Model selection primarily used **Stratified 5-Fold Cross-Validation** so that each fold maintained approximately the same survival distribution.

The selected Random Forest achieved approximately:

| Metric                    |   Score |
| ------------------------- | ------: |
| Cross-Validation Accuracy |    ~84% |
| Out-of-Fold ROC-AUC       |   ~0.87 |
| Initial Kaggle Score      | 0.76555 |

Out-of-fold predictions were also used to examine classification thresholds and ROC-AUC without depending on a single arbitrary validation split.

## Model Interpretation

Permutation importance was used to understand which original variables contributed most strongly to the Random Forest's predictions.

The most influential features included:

1. **Sex**
2. **Passenger Class**
3. **Age**
4. **Fare**

An important takeaway was that feature importance does not imply causation. Several Titanic variables are correlated—for example, passenger class, fare, and cabin availability—so their individual importance must be interpreted in context.

## Key Takeaways

This project reinforced several practical machine learning lessons:

* Start with simple, interpretable baselines before increasing model complexity.
* Use EDA to generate feature-engineering hypotheses rather than creating features arbitrarily.
* Validate engineered features instead of assuming they improve the model.
* Keep preprocessing inside ML pipelines to prevent leakage.
* Use consistent cross-validation when comparing experiments.
* Tree-based models can learn interactions that linear models may require explicitly.
* More complex algorithms do not necessarily produce better generalization.
* Local cross-validation performance does not guarantee equivalent performance on unseen competition data.

Most importantly, the project demonstrated that machine learning development is an **iterative experimental process**:

**Explore → Hypothesize → Experiment → Evaluate → Refine**

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

## Repository Structure

```text
titanic-survival-ml/
│
├── titanic_ml_portfolio.ipynb   # EDA, modeling, evaluation, and submission
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

The Kaggle competition data is not included in this repository and can be obtained directly from the Titanic competition page.

## Future Improvements

Potential extensions include:

* Comparing soft-voting ensembles of Logistic Regression, Random Forest, and XGBoost
* Testing weighted probability ensembles
* Using repeated or nested cross-validation for a more conservative estimate of generalization
* Further analyzing out-of-fold model disagreements
* Exploring additional feature interactions while carefully controlling for overfitting
