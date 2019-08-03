import glob
import numpy as np
import pickle 
import pandas as pd
import matplotlib.pyplot as plt

from estimator import estimate_spectrum
from parse_header import get_header
from tools import *
from pandas.plotting import scatter_matrix

if __name__ == "__main__":

    files = glob.glob("noisy_train_1/*.txt") #+ glob.glob("noisy_test_2/*.txt")
    bs = np.zeros(len(files)) 
    tempList= []  
    loggList= []
    radList= []
    massList= []
    kmagList= []
    periodList= []
    stdResList= []
    aList= []
    incLimList= []
    #get truths for all the files into an array

    import pdb; pdb.set_trace()
    for i in range(len(files)):
        bestMSE = np.Inf
        if (i%100 == 0):
            print (i)
        data = np.loadtxt(files[i])
        fileHeaders= get_header(files[i])
        checkRange = np.linspace(1, 20, num = 20)
        truth = np.loadtxt("params_train/"+files[i].split('/')[1])**2

        #import pdb; pdb.set_trace()
        # loop through smoothing factors to test in checkRange array
        for factor in checkRange:
            depths, res = estimate_spectrum(data, smooth=factor)
            mse = np.sum( (depths - truth)**2 )
            #if its the best smoothing factor
            if mse<bestMSE :
                bestMSE= mse
                bs[i]= factor
                # save parameters from header
                    # also save semi-major axis and inclination limit from tools.py 
                    # also calculate the standard deviation of the residuals to see how noisy the data is
                #import pdb; pdb.set_trace()
                #add to the list of dictionaries
        fileHeaders["stdRes"] = np.std(data[:50])
        fileHeaders["a"]= sa(fileHeaders["star_mass"],fileHeaders["period"])
        fileHeaders["incLim"]= incLim(fileHeaders["a"], fileHeaders["star_rad"])
        tempList.append(fileHeaders['star_temp'])
        loggList.append(fileHeaders['star_logg'])
        radList.append(fileHeaders['star_rad'])
        massList.append(fileHeaders['star_mass'])
        kmagList.append(fileHeaders['star_k_mag'])
        periodList.append(fileHeaders['period'])
        stdResList.append(fileHeaders['stdRes'])
        aList.append(fileHeaders['a'])
        incLimList.append(fileHeaders['incLim'])

    paramsDict = {'temps': tempList, 'logg': loggList, 'st_rad': radList, 'st_mass': massList, 'kmag': kmagList, 'period': periodList, 'stdRes': stdResList, 'a': aList, 'incLim': incLimList,'sf': bs.tolist()}

    #build pandas dataframe
    df = pd.DataFrame(data=paramsDict)

    # create corner plot with smoothing factor and stellar parameters  
    scatter_matrix(df)
    plt.show()

    # create correlation matrix 
    plt.figure()
    plt.matshow(df.corr())
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(bs.tolist(), periodList, 'bo')
    plt.show()

    plt.hist(periodList, bins = 100)
    plt.show()
                
    plt.plot(stdResList, bs.tolist(), 'bo')
    plt.show()

    # create histogram of best smoothing factors
    plt.figure()
    plt.title('Histogram of Best Smoothing Factors')
    plt.hist(bs, bins= 50)
    plt.show()


    # if there is a correlation: Use a spline as an estimate for an interpolation 
    #       input: whatever parameters, output: smoothing factor 

    # check scikit.learn mutual_information_regression, does that work for regression or classification?
    # if regression, what are the correlations amonst the parameters wrt to optimal smoothing factor 