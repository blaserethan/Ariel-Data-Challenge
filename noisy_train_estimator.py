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
        depths, res = estimate_spectrum(data, smooth=10)

        truths[i] = np.loadtxt("params_train/"+files[i].split('/')[1])**2
        estimates[i] = depths
        residuals[i] = res
        if i%100 == 0:
            print(i)
        
    pickle.dump( truths, open("pickle_files/train_truths.pkl",'wb') )
    pickle.dump( estimates, open("pickle_files/train_estimates.pkl",'wb') )
    pickle.dump( residuals, open("pickle_files/train_residuals.pkl",'wb') )

    print('mse :',np.sum(residuals**2))
    # baseline: stdev 10 -> 5.1 