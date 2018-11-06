import numpy as np
import scipy.interpolate

############################################################

class Constants:
    # Unit conversions
    MPC_TO_CM = 3.086e24
    KM_TO_CM = 1.e5 
    YR_TO_S = 3.154e7
    SR_TO_DEG2 = (180. / np.pi)**2

    # Physical constants
    C_LIGHT = 2.998e10 # cm s^-1

############################################################
    
class Cosmology:
    
    def __init__(self, **kwargs):
   
        # Default parameter values
        self.omega_m = 0.3
        self.omega_lambda = 0.7
        self.hubble = 70 # km s^-1 Mpc^-1
        
        # Set parameters
        self.setParams(**kwargs)
    
    def setParams(self, **kwargs):
        self.__dict__.update(kwargs)
   
        # Special case
        self._hubble = self.hubble * Constants.KM_TO_CM / Constants.MPC_TO_CM
        
        self._precompute()
        
    def _precompute(self):
        self.D_H = Constants.C_LIGHT / self._hubble # cm
        
        # The numerical precision could be improved
        z_array = np.linspace(0., 1000., 1000000)
        dz = z_array[1] - z_array[0]
        self.D_C = scipy.interpolate.interp1d(z_array, 
                                              self.D_H * dz * np.cumsum(self.E(z_array)**(-1)))
         
    def show(self):
        print(self.__dict__)
        
    def E(self, z):
        return np.sqrt(self.omega_m * (1. + z)**3 + self.omega_lambda)

    def D_L(self, z):
        return (1. + z) * self.D_C(z)

    def D_A(self, z):
        return (1. + z)**(-1) * self.D_C(z)
        
############################################################