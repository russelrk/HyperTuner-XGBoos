from typing import List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier, DMatrix, cv as xgb_cv


def fine_tune_randomize_search(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    eta: Optional[List[float]] = None, 
    max_depth: Optional[List[int]] = None, 
    sub_sample: Optional[List[float]] = None, 
    col_sample: Optional[List[float]] = None, 
    alpha_: Optional[List[float]] = None, 
    lambda_: Optional[List[float]] = None,  
    cv: int = 10,
    n_iter: int = 100,
    verbose: bool = False
) -> Tuple[dict, int]: 
    
    """
    Fine-tunes an XGBoost classifier using grid search to find the optimal hyperparameters.

    Parameters:
    - X_train (np.ndarray): The training data features, a NumPy array of shape (n_samples, n_features).
    - y_train (np.ndarray): The training data labels, a NumPy array of shape (n_samples,).
    - eta (Optional[List[float]]): List of learning rates to try, by default considers a preset list.
    - max_depth (Optional[List[int]]): List of max depths to try, by default considers a preset list.
    - sub_sample (Optional[List[float]]): List of sub-sample values to try, by default considers a preset list.
    - col_sample (Optional[List[float]]): List of colsample_bylevel values to try, by default considers a preset list.
    - alpha_ (Optional[List[float]]): List of alpha values to try, by default considers a preset list.
    - lambda_ (Optional[List[float]]): List of lambda values to try, by default considers a preset list.
    - cv (int): number of cross validation to run to find optimal parameters
    - n_iter: number of iteration 
    - sequential (bool): If True, performs sequential grid search, else performs a regular grid search. Default is True.
    - verbose (bool): If True, prints intermediate results during the grid search. Default is False.

    Returns:
    - best_params (dict): Dictionary of the best parameters found through the grid search.
    - optimal_num_rounds (int): Optimal number of boosting rounds determined through the grid search.
    """
    
    # Define default parameter distributions
    param_distributions = {
        'eta': eta or [0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': max_depth or [1, 2, 3, 4, 6],
        'subsample': sub_sample or [0.4, 0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bylevel': col_sample or [0.2, 0.4, 0.6, 0.8, 1],
        'alpha': alpha_ or [0.01, 0.1, 0.5, 1, 5],
        'lambda': lambda_ or [0.1, 0.5, 1, 5, 10],
    }
    
    # Calculate weight
    weight = np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1)
    
    
    # Initialize and fit the RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=XGBClassifier(use_label_encoder=False, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=weight), 
                                       param_distributions=param_distributions, 
                                       cv=cv, n_iter=n_iter)
    random_search.fit(X_train, y_train)
    
    
    # Get the best parameters and estimator
    best_params = random_search.best_params_
    best_estimator = random_search.best_estimator_
    
    # Get the optimal number of rounds
    optimal_num_rounds = best_estimator.get_booster().best_ntree_limit
        
    return best_params, optimal_num_rounds

