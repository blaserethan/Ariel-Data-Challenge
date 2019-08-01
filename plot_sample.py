import glob
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    files = glob.glob("noisy_train_1/*.txt")

    i = np.random.randint(len(files))

    data = np.loadtxt(files[i])
    pars_file = "params_train/"+files[i].split('/')[-1]
    pars = np.loadtxt(pars_file)

    cm = plt.get_cmap('jet') 

    f,ax = plt.subplots(1)
    for j in range(data.shape[0]):
        ax.plot(data[j]+0.01*j,color=cm(j/data.shape[0]))
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Flux')
    ax.set_title('Spectral Time-series ({})'.format(files[i]))
    plt.show()

    f,ax = plt.subplots(1)
    ax.plot(data.mean(0),'ko')
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Flux')
    ax.set_title('White Light Curve ({})'.format(files[i]))
    plt.show() 