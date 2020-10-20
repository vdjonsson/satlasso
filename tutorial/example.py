import numpy as np
import pandas as pd
from SatLasso import SatLasso, SatLassoCV

def get_data():
    # read in data
    data = pd.read_csv('NeutSeqData_3BNC117_encoded.csv', sep=',', header=0, index_col=0)
    
    # read in metadata to get labels
    metadata = pd.read_csv('NeutSeqData_3BNC117.csv', sep=',', header=0)
    labels = metadata['ic50_ngml'].values
    
    return data, labels

# satlasso code
def run_satlasso(data, labels):
    # set up satlasso object
    satlasso = SatLasso(lambda_1 = 1., lambda_2 = 7.75, lambda_3 = 10., saturation='max', fit_intercept=True, normalize=False, copy_X=True)
    
    # fit satlasso regressor with data and labels
    satlasso.fit(data, labels)
    
    # get coefficients, intercept
    coef, intercept = satlasso.coef_, satlasso.intercept_
    
    # get predicted values
    predicted = satlasso.predict(data)
    
    # get error of prediction
    error = ((predicted-labels)**2).mean()

    return coef, intercept, predicted, error

# satlassocv code
def run_satlassocv(data, labels):
    lambda1s = np.linspace(1,10,3)
    lambda2s = np.linspace(1,10,3)
    lambda3s = np.linspace(1,10,3)
    
    # set up satlasso cross-validation object
    satlassocv = SatLassoCV(lambda_1s = lambda1s, lambda_2s = lambda2s, lambda_3 = lambda3s, saturation='max', fit_intercept=True, normalize=False, copy_X=True, cv=3)
    
    # fit satlasso regressor with data and labels
    satlassocv.fit(data, labels)
    
    # get coefficients, intercept, optimal lambdas and cross-validation error for lambda combinations tested
    coef, intercept, lambda1, lambda2, lambda3, mse_dict = satlassocv.coef_, satlassocv.intercept_, satlassocv.lambda_1_, satlassocv.lambda_2_, satlassocv.lambda_3_, satlassocv.mse_dict_
    
    # get predicted values
    predicted = satlassocv.predict(data)
    
    # get error of prediction
    error = ((predicted-labels)**2).mean()

    return coef, intercept, (lambda1, lambda2, lambda3), mse_dict, predicted, error

def main():
    # load data
    data, labels = get_data()
    
    # run satlasso, get metrics and output of satlasso
    coef, intercept, predicted, error = run_satlasso(data, labels)
    
    # run satlassocv, get metrics and output of satlassocv
    cv_coef, cv_intercept, optimal_lambdas, mse_dict, cv_predicted, cv_error = run_satlassocv(data, labels)
