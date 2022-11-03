import numpy as np
import pandas as pd

def make_bootstrap_sample(X, z, sample_size=1):
    """
    Function that generates one bootstrap sample.
    
    Input
    -----
    X (dataframe or matrix): design matrix containing all input data
    z (array): array of outputs
    sample_size (float): percentage of input to use for bootstrap sample
    
    Returns
    -------
    X_sample (dataframe): bootstrap sample of input data
    z_sample (array): output data corresponding to input data in bootstrap sample
    X_test (dataframe): input data not sampled, used as test data
    z_test (dataframe): output data corresponding to input data not sampled
    """
    # Randomly draw n rows from design matrix X with replacement
    X = pd.DataFrame(X)
    X_sample = X.sample(n=int(sample_size*len(X)), replace=True)
    rows_chosen = X_sample.index
    # Choose same rows from z to get output training data
    z_sample = z[rows_chosen]
    
    # Use rows not sampled as test set
    X_test = X[~X.index.isin(rows_chosen)]
    z_test = z.drop(rows_chosen)

    
    return X_sample, z_sample, X_test, z_test