import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum
from parse_header import get_header

if __name__ == "__main__":

    files = glob.glob("noisy_train_1/*.txt") #+ glob.glob("noisy_test_2/*.txt")

    bs = np.zeros(len(files))

    for i in range(len(files)):
        data = np.loadtxt(files[i])
    
        # loop through smoothing factors 
        depths, res = estimate_spectrum(data, smooth=10)
        # mse = np.sum( (depths - truth[i])**2 )
        # save best smoothing factor
        # save parameters from header
            # also save semi-major axis and inclination limit from tools.py 
    
    # create histogram of best smoothing factors
    # create corner plot with smoothing factor and stellar parameters
    # create correlation matrix 
    # if there is a correlation: Use a spline as an estimate for an interpolation 
    #       input: whatever parameters, output: smoothing factor 

    # check scikit.learn mutual_information_regression, does that work for regression or classification?
    # if regression, what are the correlations amonst the parameters wrt to optimal smoothing factor 