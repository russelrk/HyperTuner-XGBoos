from typing import List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, DMatrix, cv as xgb_cv

def fine_tune_grid_search(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    eta: Optional[List[float]] = None, 
    max_depth: Optional[List[int]] = None, 
    sub_sample: Optional[List[float]] = None, 
    col_sample: Optional[List[float]] = None, 
    alpha_: Optional[List[float]] = None, 
    lambda_: Optional[List[float]] = None, 
    cv: int = 10,
    sequential: bool = True,
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
    - sequential (bool): If True, performs sequential grid search, else performs a regular grid search. Default is True.
    - verbose (bool): If True, prints intermediate results during the grid search. Default is False.

    Returns:
    - best_params (dict): Dictionary of the best parameters found through the grid search.
    - optimal_num_rounds (int): Optimal number of boosting rounds determined through the grid search.
    """
    
    # Define default parameter grids
    param_grids = {
        'eta': eta or [0.01, 0.05, 0.1, 0.5, 1],
        'max_depth': max_depth or [1, 2, 3, 4, 6],
        'subsample': sub_sample or [0.4, 0.6, 0.7, 0.8, 0.9, 1],
        'colsample_bylevel': col_sample or [0.2, 0.4, 0.6, 0.8, 1],
        'alpha': alpha_ or [0.01, 0.1, 0.5, 1, 5],
        'lambda': lambda_ or [0.1, 0.5, 1, 5, 10],
    }
    
    num_rounds = num_rounds or 5000
    
    weight = np.count_nonzero(y_train == 0) / np.count_nonzero(y_train == 1)

    # Initial base parameters
    base_params = {
        'objective': ['binary:logistic'],
        'eval_metric': ['logloss'],
        'scale_pos_weight': [weight],
    }
    

    best_params = {}
    
    if sequential:
        for param_name, param_values in param_grids.items():
            grid_param = {param_name: param_values}
            grid_param.update(base_params)
            
            # print(grid_param)

            grid_search = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False), param_grid=grid_param, cv=cv)
            grid_search.fit(X_train, y_train)
            
            # Updating the base_params with the best parameter found in this iteration
            base_params[param_name] = [grid_search.best_params_[param_name]]

            if verbose:
                print(f"Best value for {param_name}: {best_params[param_name]}")
                
    else:
        param_grids.update(base_params)
        grid_search = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False), param_grid=param_grids, cv=cv)
        grid_search.fit(X_train, y_train)
        
        
    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    cv_result = xgb.cv(
        best_params,
        xgb.DMatrix(X_train, label=y_train),
        num_boost_round=1000,
        early_stopping_rounds=50,
        metrics="logloss",
        stratified=True,
        seed=42,
        verbose_eval=False
    )
     
    optimal_num_rounds = cv_result.shape[0]
        
        
    return best_params, optimal_num_rounds
