import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

def estimate_spectrum(data, smooth=10): 

    whitelight = data.mean(0)
    swl = gaussian_filter(whitelight,smooth)
    template = -1*(swl-swl.max())/(swl-swl.max()).min()

    A = np.vstack([np.ones(swl.shape[0]), template]).T

    depths = np.zeros(data.shape[0])
    residuals = np.zeros(data.shape)
    
    for j in range(data.shape[0]):
        b, m = np.linalg.lstsq(A, data[j], rcond=None)[0]
        depths[j] = m
        residuals[j] = data[j]- (m*template +b)

    # depths[depths<0] = 0 # will this bias ML with residuals? 
    return depths, residuals 


if __name__ == "__main__":

    files = glob.glob("noisy_train_1/*.txt")
    i = np.random.randint(len(files))
    data = np.loadtxt(files[i])
    pars_file = "params_train/"+files[i].split('/')[-1]
    pars = np.loadtxt(pars_file)


    whitelight = data.mean(0)
    swl = gaussian_filter(whitelight,10)
    template = -1*(swl-swl.max())/(swl-swl.max()).min()
    A = np.vstack([np.ones(swl.shape[0]), template]).T
    bw, mw = np.linalg.lstsq(A, whitelight, rcond=None)[0]


    plt.plot(whitelight,'ko',label='data')
    plt.plot(swl,'r-', label='template')
    plt.plot(mw*template +bw,'g--', label='best fit (depth={:.3f})'.format(mw))
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Relative Flux')
    plt.title('White Light Curve ({})'.format(files[i]))
    plt.show()


    depths, residuals = estimate_spectrum(data)
    
    plt.plot(pars**2,'g-',label='truth')
    plt.plot(depths,'r-',label='estimate')
    plt.legend(loc='best')
    plt.xlabel('wavelength')
    plt.ylabel('Transit Depth')
    plt.title('Estimated Transmission Spectrum ({})'.format(files[i]))
    plt.show()