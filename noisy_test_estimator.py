import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum
from  network_design import build_FullyConnected
import tensorflow as tf

if __name__ == "__main__":

    files = glob.glob("noisy_test_1/*.txt") + glob.glob("noisy_test_2/*.txt")
    estimates = np.zeros( (len(files),55) )
    residuals = np.zeros( (len(files),55,300) )

    dc = 0
    for i in range(5,2096+1):
        for j in range(1,10+1):
            for k in range(1,10+1):
                try:
                    filename = "noisy_test_1/{:04}_{:02}_{:02}.txt".format(i,j,k)
                    data = np.loadtxt(filename)
                except:
                    try:
                        filename = "noisy_test_2/{:04}_{:02}_{:02}.txt".format(i,j,k)
                        data = np.loadtxt(filename)
                    except:
                        #print('file not found: ', filename, i,j,k)
                        continue

                estimates[dc], residuals[dc], snr = estimate_spectrum(data)
                if dc%100 == 0:
                    print(i)
                dc +=1 

    #save the estimates into a pickle file
    pickle.dump( [estimates, residuals], open("test_set.pkl","wb"), protocol=4)

    #load the neural network
    model = build_FullyConnected()
    model.load_weights('regressor.h5')
    model.summary() 

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        #loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )
    
    scaled_residuals = np.copy(residuals)
    scaled_estimates = np.copy(estimates)

    #scale the data s.t. the mean is 0 and std ~1
    for k in range(residuals.shape[0]):
        scaled_residuals[k] = residuals[k] / residuals[k].std()
        #also scale estimates by scatter in residual to prevent offset bias
        scaled_estimates[k] = (estimates[k] - estimates[k].mean()) / residuals[k].std()
                
    debiased = model.predict( [scaled_residuals,scaled_estimates] )
    for k in range(debiased.shape[0]):
        debiased[k] *= residuals[k].std()
        debiased[k] += estimates[k].mean()

    np.savetxt('test_output.txt',debiased)
    print(debiased.shape)
    np.savetxt('test_output_transpose.txt',debiased)
