# applies estimator to training data prior to NN input
# preprocessing step 
import glob
import numpy as np
import pickle 

from estimator import estimate_spectrum

if __name__ == "__main__":

    file_sets = [
        glob.glob("noisy_train_1/*.txt"),
        glob.glob("noisy_train_2/*.txt"),
        glob.glob("noisy_train_3/*.txt"),
        glob.glob("noisy_train_4/*.txt"),
        #glob.glob("noisy_train_5/*.txt")
    ]
    NUM = 20000
    estimates = np.zeros( (NUM,55) )
    residuals = np.zeros( (NUM,55,300) )
    truths = np.zeros( (NUM,55) )

    for j in range(len(file_sets)):
        files = file_sets[j]
        print('processing file:',file_sets[j])

        for i in range(NUM):
            data = np.loadtxt(files[i])

            estimates[i], residuals[i], snr = estimate_spectrum(data)
            truths[i] = np.loadtxt("params_train/"+files[i].split('/')[1])**2

            if i%500 == 0:
                print(i)
            if i == NUM:
                break

        pickle.dump( truths, open("pickle_files/train_{}_truths.pkl".format(j+1),'wb') )
        pickle.dump( estimates, open("pickle_files/train_{}_estimates.pkl".format(j+1),'wb') )
        pickle.dump( residuals, open("pickle_files/train_{}_residuals.pkl".format(j+1),'wb') )

        print('{} mse :{:.2f}'.format(j, np.sum( (truths-estimates)**2 ) ))