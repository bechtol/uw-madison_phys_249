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
         
        self._precompute()
        
    def _precompute(self):
        self.hubbleTime = (self.hubble * Constants.KM_TO_CM / Constants.MPC_TO_CM)**(-1) # s
        self.hubbleDistance = Constants.C_LIGHT * self.hubbleTime # cm
        
        # The numerical precision could be improved
        z_array = np.linspace(0., 1000., 1000000)
        dz = z_array[1] - z_array[0]
        self.comovingDistance = scipy.interpolate.interp1d(z_array, 
                                                           self.hubbleDistance * dz * np.cumsum(self._E(z_array)**(-1)))
         
    def show(self):
        print(self.__dict__)
        
    def _E(self, z):
        return np.sqrt(self.omega_m * (1. + z)**3 + self.omega_lambda)

    def luminosityDistance(self, z):
        """
        Return the luminosity distance (cm) for a set of input redshift values
        """
        return (1. + z) * self.comovingDistance(z) # cm

    def angularDistance(self, z):
        """
        Return the angular distance (cm) for a set of input redshift values
        """
        return (1. + z)**(-1) * self.comovingDistance(z) # cm
        
############################################################