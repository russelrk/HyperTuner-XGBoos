# XGBoost_Model_Train_Hyper_Tuner


markdown
# XGBoost Model Training and Hyperparameter Tuning

This repository provides scripts and notebooks that demonstrate how to train machine learning models using XGBoost on tabular data and perform hyperparameter tuning using RandomizedSearchCV and GridSearchCV to find the optimal model.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Ensure that you have the following installed:
- Python 3.8 or later
- Scikit-learn
- XGBoost
- NumPy
- pandas

You can install the required packages using the following command:

sh
pip install scikit-learn xgboost numpy pandas



## Usage

### Data Preparation

Before training your model, ensure your data is cleaned and prepared in a tabular format (e.g., CSV). You can load your data using pandas:

```python
import pandas as pd

data = pd.read_csv('path/to/yourdata.csv')
```

### Model Training

To train an XGBoost model using default parameters:

```python
from xgboost import XGBClassifier
import numpy as np

X, y = data.iloc[:, :-1], data.iloc[:, -1]

model = XGBClassifier()
model.fit(X, y)
```

### Hyperparameter Tuning

This repository offers two scripts for hyperparameter tuning: `grid_search_tuning.py` and `randomized_search_tuning.py`. You can use them as follows:

#### Grid Search

```python
from grid_search_tuning import fine_tune

best_params, optimal_num_rounds = fine_tune_grid_search(X_train, y_train)
```

#### Randomized Search

```python
from randomized_search_tuning import fine_tune

best_params, optimal_num_rounds = fine_tune_randomize_search(X_train, y_train)
```

## Scripts

- `grid_search_tuning.py`: Script for hyperparameter tuning using GridSearchCV.
- `randomized_search_tuning.py`: Script for hyperparameter tuning using RandomizedSearchCV.

## Contributing

Feel free to open issues or PRs if you encounter any problems or have suggestions for improvements.

## License

MIT License. See [LICENSE](LICENSE) for more details.
```
