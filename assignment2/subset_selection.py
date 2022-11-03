import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def forward_selection(X, y, stopping_criterion='AIC'):
    # Create a dataframe containing only the intercept
    df_forward = pd.DataFrame(X['bias'])

    # Find list of all possible features
    columns = X.columns.drop('bias')
    
    # Initiate initial criterion to infinity
    criterion_prev = float('inf')

    # Find value of AIC / BIC for model with only intercept
    linear_model = sm.OLS(y, df_forward)
    lm_result = linear_model.fit()
    if stopping_criterion == 'AIC':
        criterion = lm_result.aic
    elif stopping_criterion == 'BIC':
        criterion = lm_result.bic

    # Run forward selection 
    while len(columns) > 0: 
        best_col = ''
        lowest_error = float('inf')

        for col in columns:

            df_selected = df_forward.copy(deep=True)
            df_selected[col] = X[col] 

            # Train model with selected columns
            linear_model = sm.OLS(y, df_selected)
            lm_result = linear_model.fit()

            y_pred = lm_result.predict(df_selected)
            mse = mean_squared_error(y, y_pred)
            if mse < lowest_error:
                lowest_error = mse
                best_col = col

            if stopping_criterion == 'AIC':
                criterion = lm_result.aic
            elif stopping_criterion == 'BIC':
                criterion = lm_result.bic
                
        # Stop loop if criterion not decreasing
        if criterion > criterion_prev:
            break
        criterion_prev = criterion
        # Remove added feature from list of columns
        columns = columns.drop(best_col)
        df_forward[best_col] = X[best_col]

    return df_forward

def backward_selection(X, y, stopping_criterion='AIC'):
    # Copy dataframe
    df_backward = X.copy(deep=True)

    # Find list of all possible features
    columns = X.columns.drop('bias')
    
    # Initiate initial criterion to infinity
    criterion_prev = float('inf')

    # Find value of AIC / BIC for model with only intercept
    linear_model = sm.OLS(y, df_backward)
    lm_result = linear_model.fit()
    if stopping_criterion == 'AIC':
        criterion = lm_result.aic
    elif stopping_criterion == 'BIC':
        criterion = lm_result.bic

    # Run backward selection while we still have columns
    while len(df_backward) > 0: 
        for col in columns:
            # Train model with selected columns
            linear_model = sm.OLS(y, df_backward)
            lm_result = linear_model.fit()

            worst_col = lm_result.pvalues.idxmax()
                
        # Remove column from dataframe
        df_backward = df_backward.drop(worst_col, axis=1)
        
        if stopping_criterion == 'AIC':
            criterion = lm_result.aic
        elif stopping_criterion == 'BIC':
            criterion = lm_result.bic
            
        # Stop loop if criterion not decreasing
        if criterion > criterion_prev:
            break
        criterion_prev = criterion

    return df_backward
