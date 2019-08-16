import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

def estimate_spectrum(data, smooth=9, data_smooth=2): 

    whitelight = data.mean(0) #make a white light curve
    swl = gaussian_filter(whitelight, smooth) #smooth it out
    template = -1*(swl-swl.max())/(swl-swl.max()).min() #scale smooth light curve between -1 and 0
    A = np.vstack([np.ones(swl.shape[0]), template]).T #makes template matrix
    bw, mw = np.linalg.lstsq(A, whitelight, rcond=None)[0] #fit for m is depth and b is offset of white lightcurve
    resw = data.mean(0) - (mw*template+bw)
    snr = mw/np.std(resw)

    # smooth data in time and wavelength to reduce noise    
    sdata = gaussian_filter(data,data_smooth)
    
    #f,ax = plt.subplots(2); ax[0].imshow(data,vmin=0.999,vmax=1.001); ax[0].set_ylabel('Wavelength Index'); ax[1].set_xlabel('Time Index'); ax[0].set_title('Raw Data'); im=ax[1].imshow(sdata,vmin=0.999,vmax=1.001); ax[1].set_title('Smoothed Data'); cbar = plt.colorbar(im); cbar.ax.get_yaxis().labelpad = 15; cbar.ax.set_ylabel('Relative flux'); plt.show()

    depths = np.zeros(data.shape[0]) #alloc for transit depth vector
    residuals = np.zeros(data.shape) #Data- model

    for j in range(data.shape[0]):
        #depths[j] = mw
        #residuals[j] = data[j] - (mw*template+bw) #residuals from white light curve fit
        #continue 

        b, m = np.linalg.lstsq(A, sdata[j], rcond=None)[0]

        if m < 0: # if negative transit depth, replace with white light 
            depths[j] = mw
            residuals[j] = data[j] - (mw*template+bw) #residuals from white light curve fit
        else:
            depths[j] = m
            residuals[j] = data[j] - (m*template+b) #residuals for each template model
    
    return depths, residuals, snr

# average with mirror image? 

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

    depths, residuals, snr = estimate_spectrum(data)
    
    plt.plot(pars**2,'g-',label='truth')
    plt.plot(depths,'r-',label='estimate')
    plt.plot([0,len(depths)-1],[mw,mw],'k--',label='White Light')
    plt.legend(loc='best')
    plt.xlabel('wavelength')
    plt.ylabel('Transit Depth')
    plt.title('Estimated Transmission Spectrum ({})'.format(files[i]))
    plt.show()
