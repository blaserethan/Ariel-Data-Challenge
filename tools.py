import numpy as np

rsun = 6.955e8 # m
msun = 1.989e30 # kg
mjup = 1.898e27 # kg 
rjup = 7.1492e7 # m
mearth = 5.972e24 # kg
rearth = 6.3781e6 # m
au=1.496e11 # m 
G = 0.00029591220828559104 # day, AU, Msun

# logg[cm/s2], rs[rsun]
stellar_mass = lambda logg,rs: ((rs*6.955e10)**2 * 10**logg / 6.674e-8)/1000./msun

# keplerian semi-major axis (au)
sa = lambda m,P : (G*m*P**2/(4*np.pi**2) )**(1./3) 

incLim = lambda a, rs: np.arctan((a*au)/(rs*rsun))
# TODO inclination limit 