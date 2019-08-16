import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum

if __name__ == "__main__":

    files = glob.glob("noisy_test_1/*.txt") #+ glob.glob("noisy_test_2/*.txt")
    estimates = np.zeros( (len(files),55) )
    residuals = np.zeros( (len(files),55,300) )

    #for each file load in the data, get estimates and resuiduals
    for i in range(len(files)):
        data = np.loadtxt(files[i])
        
        depths, res, snr = estimate_spectrum(data)
        estimates[i] = depths
        residuals[i] = res
        if i%100 == 0:
            print(i)

    #save the estimates into a pickle file
    #pickle.dump( [spectra, residuals], open("test_set1.pkl","wb"), protocol=4)

    #load the neural network
    model = build_FullyConnected()
    model.summary() 

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # TODO parameterize optimizer 
        loss=tf.keras.losses.MeanSquaredError(),
        #loss=tf.keras.losses.MeanAbsolutePercentageError(),
    )

    scaled_residuals = np.copy(residuals)
    scaled_estimates = np.copy(estimates)

    #scale the data s.t. the mean is 0 and std ~1
    for j in range(residual.shape[0]):
        scaled_residuals[k] = residuals[k] / residuals[k].std()
        #also scale estimates by scatter in residual to prevent offset bias
        scaled_estimates[k] = (estimates[k] - np.mean(estimates[k]))/residuals[k].std() 

    # history = model.fit(
    # [scaled_residuals,scaled_estimates], 
    # scaled_truths,
    # epochs=args.epochs, 
    # batch_size=32,
    # validation_split=0.1
    # )

    debiased = model.predict([scaled_residuals, scaled_estimates])

    #scale data back by multiplying by the scatter in the residuals 
    for k in range(debiased.shape[0]):
        debiased[k] *= residuals[k].std()
