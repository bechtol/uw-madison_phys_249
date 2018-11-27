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
        """Initialize a cosmology with optional parameter values.
       
        Args:
            omega_matter_0 (float, optional): matter density in units of critical density
            omega_radiation_0 (float, optional): radiation density in units of critical density
            omega_lambda_0 (float, optional): dark energy density in units of critical density
            hubble_0 (float, optional): Hubble contant today (km s^-1 Mpc^-1)
        """
        # Default parameter values
        self.omega_matter_0 = 0.3
        self.omega_radiation_0 = 9.0e-5 # photons + neutrinos
        self.omega_lambda_0 = 1. - (self.omega_matter_0 + self.omega_radiation_0)
        self.hubble_0 = 70. # km s^-1 Mpc^-1
        
        # Set parameters
        self.setParams(**kwargs)
    
    def setParams(self, **kwargs):
        """
        Set or update cosmological parameter values.
        """
        self.__dict__.update(kwargs)
         
        self._precompute()
        
    def _precompute(self):
        """
        Precompute a few derived quantities.
        """
        self.hubbleTime = (self.hubble_0 * Constants.KM_TO_M / Constants.MPC_TO_M)**(-1) # s
        self.hubbleDistance = Constants.C_LIGHT * self.hubbleTime # m
        self.omega_0 = self.omega_matter_0 + self.omega_radiation_0 + self.omega_lambda_0
        if not np.isclose(self.omega_0, 1.):
            print('WARNING: input parameters correspond to universe with non-zero curvature')
        
    def comovingDistance(self, z):
        """Compute the comoving distance for a given redshift.
        
        Args:
            z (float): redshift
            
        Returns:
            float: comoving distance (m)
        """
        return self.hubbleDistance * scipy.integrate.quad(self._EReciprocal, 0, z)[0]
         
    def show(self):
        """
        Display the current cosmological parameter values. 
        """
        print(self.__dict__)
        
    def _E(self, z):
        return np.sqrt(self.omega_matter_0 * (1. + z)**3 \
                       + self.omega_lambda_0 \
                       + self.omega_radiation_0 * (1 + z)**4 \
                       + (1. - self.omega_0) * (1 + z)**2)
    
    def _EReciprocal(self, z):
        return 1. / self._E(z)
    
    def _EReciprocalLookbackTime(self, z):
        return 1. / ((1. + z) * self._E(z))

    def _Ea(self, a):
        return np.sqrt(self.omega_radiation_0 * a**(-2) \
                       + self.omega_matter_0 * a**(-1) \
                       + self.omega_lambda_0 * a**2 \
                       + (1. - self.omega_0))
    
    def _EaReciprocal(self, a):
        return 1. / self._Ea(a)
    
    def luminosityDistance(self, z):
        """Compute the luminosity distance for a given redshift.
        
        Args:
            z (float): redshift
            
        Returns:
            float: luminosity distance (m)
        """
        return (1. + z) * self.comovingDistance(z) # m
        
    def angularDistance(self, z):
        """Compute the angular distance for a given redshift.
        
        Args:
            z (float): redshift
            
        Returns:
            float: angular distance (m)
        """
        return (1. + z) * self.comovingDistance(z) # m
    
    def cosmicTime(self, a, a_init=1.e-10):
        """Compute the cosmic time corresponding to a given scale factor.
        
        Args:
            a (float): scale factor.
            a_init (float): scale factor at starting time.
                Defaults to 1.e-10
        
        Returns:
            float: cosmic time, H_0 t (dimensionless)
        """
        return scipy.integrate.quad(self._EaReciprocal, a_init, a)[0]
    
    def lookbackTime(self, z):
        """Compute the lookback time for a given redshift.
        
        Args:
            z (float): redshift
            
        Returns:
            float: lookback time (s)
        """
        return self.hubbleTime * scipy.integrate.quad(self._EReciprocalLookbackTime, 0, z)[0] # s
        
############################################################