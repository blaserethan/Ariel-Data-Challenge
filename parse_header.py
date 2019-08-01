import glob
import numpy as np
import matplotlib.pyplot as plt

def get_header(filename): 
    with open(filename) as f: # Use file to refer to the file object
        lines = f.readlines()
        ddict = {}
        for j in range(len(lines)):
            if '#' in lines[j]:
                split = lines[j][1:].split(':')
                ddict[split[0].strip()] = float(split[1].strip())
            else:
                break
    return ddict

if __name__ == "__main__":
    files = glob.glob("noisy_train_1/*.txt")

    i = np.random.randint(len(files))

    data = np.loadtxt(files[i])
    pars_file = "params_train/"+files[i].split('/')[-1]
    pars = np.loadtxt(pars_file)

    spars = get_header(files[i])
    ppars = get_header(pars_file)