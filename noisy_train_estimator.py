import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum

if __name__ == "__main__":

    files = glob.glob("noisy_train_1/*.txt") #+ glob.glob("noisy_test_2/*.txt")

    estimates = np.zeros( (len(files),55) )
    residuals = np.zeros( (len(files),55,300) )
    truths = np.zeros( (len(files),55) )

    for i in range(len(files)):
        data = np.loadtxt(files[i])
        # parse header
        # get best smoothing factor 
        depths, res = estimate_spectrum(data, smooth=9)

        truths[i] = np.loadtxt("params_train/"+files[i].split('/')[1])**2
        estimates[i] = depths
        residuals[i] = res
        if i%100 == 0:
            print(i)
        
    pickle.dump( truths, open("pickle_files/train_truths.pkl",'wb') )
    pickle.dump( estimates, open("pickle_files/train_estimates.pkl",'wb') )
    pickle.dump( residuals, open("pickle_files/train_residuals.pkl",'wb') )

    print('mse :',np.sum( (truths-estimates)**2 ) )

    # MSE values for different fits 
    # template smoothing factor 10 -> 5.1 
    # template smoothing factor 9  -> 5.08
    # template smoothing factor 9, data smoothing factor 1 - > 5.07
    # MSE of white light flat spectrum - > 