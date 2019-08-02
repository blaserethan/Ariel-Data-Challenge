import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum

if __name__ == "__main__":

    files = glob.glob("noisy_test_1/*.txt") + glob.glob("noisy_test_2/*.txt")

    spectra = np.zeros( (len(files),55) )
    residuals = np.zeros( (len(files),55,300) )

    for i in range(len(files)):
        data = np.loadtxt(files[i])
        # parse header
        # get best smoothing factor
        depths, res = estimate_spectrum(data, smooth=10)
        spectra[i] = depths
        residuals[i] = res
        if i%100 == 0:
            print(i)

    pickle.dump( [spectra, residuals], open("test_spectra_residuals.pkl","wb"), protocol=4)