import glob
import numpy as np
import matplotlib.pyplot as plt
import corner

from parse_header import get_header

if __name__ == "__main__":
    files = glob.glob("noisy_train_1/*.txt")

    alldata = {}

    for i in range(len(files)):

        #data = np.loadtxt(files[i])
        pars_file = "params_train/"+files[i].split('/')[-1]
        #pars = np.loadtxt(pars_file)

        spars = get_header(files[i])
        ppars = get_header(pars_file)

        if i == 0:
            for k in spars:
                alldata[k] = []
            for k in ppars:
                alldata[k] = []
        
        for k in spars:
            alldata[k].append(spars[k])
        for k in ppars:
            alldata[k].append(ppars[k])
    
