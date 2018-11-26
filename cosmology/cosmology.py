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
        self.omega_matter_0 = 0.3
        self.omega_radiation_0 = 9.0e-5 # photons + neutrinos
        self.omega_lambda_0 = 1. - (self.omega_matter_0 + self.omega_radiation_0)
        self.hubble_0 = 70. # km s^-1 Mpc^-1
        
        # Set parameters
        self.setParams(**kwargs)
    
    def setParams(self, **kwargs):
        self.__dict__.update(kwargs)
         
        self._precompute()
        
    def _precompute(self):
        """
        Precompute a few derived quantities
        """
        self.hubbleTime = (self.hubble_0 * Constants.KM_TO_M / Constants.MPC_TO_M)**(-1) # s
        self.hubbleDistance = Constants.C_LIGHT * self.hubbleTime # m
        self.omega_0 = self.omega_matter_0 + self.omega_radiation_0 + self.omega_lambda_0
        if not np.isclose(self.omega_0, 1.):
            print('WARNING: input parameters correspond to universe with non-zero curvature')
        
    def _comovingDistance(self, z):
        return self.hubbleDistance * scipy.integrate.quad(self._EReciprocal, 0, z)[0]
         
    def show(self):
        print(self.__dict__)
        
    def _E(self, z):
        return np.sqrt(self.omega_matter_0 * (1. + z)**3 \
                       + self.omega_lambda_0 \
                       + self.omega_radiation_0 * (1 + z)**4 \
                       + (1. - self.omega_0) * (1 + z)**2)
    
    def _EReciprocal(self, z):
        return 1. / self._E(z)

    def _Ea(self, a):
        return np.sqrt(self.omega_radiation_0 * a**(-2) \
                       + self.omega_matter_0 * a**(-1) \
                       + self.omega_lambda_0 * a**2 \
                       + (1. - self.omega_0))
    
    def _EaReciprocal(self, a):
        return 1. / self._Ea(a)
    
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
    
    def cosmicTime(self, a, a_init=1.e-10):
        """
        Return the cosmic time H_0 t (dimensionless) corresponding to a given scale factor
        """
        return scipy.integrate.quad(self._EaReciprocal, a_init, a)[0]
        
############################################################