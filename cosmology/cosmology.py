import numpy as np
import scipy.interpolate
import scipy.integrate

############################################################

class Constants:
    # Unit conversions
    MPC_TO_M = 3.086e22
    KM_TO_M = 1.e3 
    YR_TO_S = 3.154e7
    SR_TO_DEG2 = (180. / np.pi)**2

    # Physical constants
    C_LIGHT = 2.998e8 # m s^-1

############################################################
    
class Cosmology:
    
    def __init__(self, **kwargs):
   
        # Default parameter values
        self.omega_matter = 0.3
        self.omega_radiation = 9.0e-5 # photons + neutrinos
        self.omega_lambda = 1. - (self.omega_matter + self.omega_radiation)
        self.hubble = 70. # km s^-1 Mpc^-1
        
        # Set parameters
        self.setParams(**kwargs)
    
    def setParams(self, **kwargs):
        self.__dict__.update(kwargs)
         
        self._precompute()
        
    def _precompute(self):
        self.hubbleTime = (self.hubble * Constants.KM_TO_M / Constants.MPC_TO_M)**(-1) # s
        self.hubbleDistance = Constants.C_LIGHT * self.hubbleTime # m
        
    def _comovingDistance(self, z):
        return self.hubbleDistance * scipy.integrate.quad(self._EReciprocal, 0, z)[0]
         
    def show(self):
        print(self.__dict__)
        
    def _E(self, z):
        return np.sqrt(self.omega_matter * (1. + z)**3 + self.omega_lambda + self.omega_radiation * (1 + z)**4)
    
    def _EReciprocal(self, z):
        return 1. / self._E(z)

    def luminosityDistance(self, z):
        """
        Return the luminosity distance (m) for a set of input redshift values
        """
        return (1. + z) * self._comovingDistance(z) # m
        
    def angularDistance(self, z):
        """
        Return the angular distance (m) for a set of input redshift values
        """
        return (1. + z) * self._comovingDistance(z) # m
        
############################################################