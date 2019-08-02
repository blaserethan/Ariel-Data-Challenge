import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum
from parse_header import get_header
from tools import *

if __name__ == "__main__":
    bestMSE = np.Inf
    files = glob.glob("noisy_train_1/*.txt") #+ glob.glob("noisy_test_2/*.txt")
    truthFiles = 
    bs = np.zeros(len(files)) 
    paramsList= []  #list of dictionaries of all of the parameters we want
    #get truths for all the files into an array

    import pdb; pdb.set_trace()
    for i in range(len(files)):
        data = np.loadtxt(files[i])
        fileHeaders= get_header(files[i])
        checkRange = np.linspace(1, 20, num = 20)

        import pdb; pdb.set_trace()
        # loop through smoothing factors to test in checkRange array
        for factor in checkRange:
            depths, res = estimate_spectrum(data, smooth=factor)
            mse = np.sum( (depths - truth[i])**2 )
            #if its the best smoothing factor
            if mse<bestMSE :
                bestMSE= mse
                bs[i]= mse
                # save parameters from header
                    # also save semi-major axis and inclination limit from tools.py 
                    # also calculate the standard deviation of the residuals to see how noisy the data is
                fileHeaders["stdRes"] = np.std(res)
                fileHeaders["a"]= sa(fileHeaders["star_mass"],fileHeaders["period"])
                fileHeaders["incLim"]= incLim(fileHeaders["a"], fileHeaders["star_rad"])
                import pdb; pdb.set_trace()
                #add to the list of dictionaries
                paramsList.append(fileHeaders)

                stdRes = np.std(res)
                
                
    
    # create histogram of best smoothing factors
    plt.figure()
    plt.title('Histogram of Best Smoothing Factors')
    plt.hist(bs)
    plt.show()

    # create corner plot with smoothing factor and stellar parameters
    # create correlation matrix 
    # if there is a correlation: Use a spline as an estimate for an interpolation 
    #       input: whatever parameters, output: smoothing factor 

    # check scikit.learn mutual_information_regression, does that work for regression or classification?
    # if regression, what are the correlations amonst the parameters wrt to optimal smoothing factor 