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
    G_GRAVITATION = 6.674e-11 # m^3 kg^-1 s^-2
    M_PROTON = 1.673e-27 # kg
    
############################################################
    
class Cosmology:
    
    def __init__(self, **kwargs):
        """Initialize a cosmology with optional parameter values.
       
        Args:
            omega_matter_0 (float, optional): matter density in units of critical density
            omega_radiation_0 (float, optional): radiation density in units of critical density
            omega_darkenergy_0 (float, optional): dark energy density in units of critical density
            hubble_0 (float, optional): Hubble contant today (km s^-1 Mpc^-1)
        """
        # Default parameter values
        self.omega_matter_0 = 0.3
        self.omega_radiation_0 = 9.0e-5 # photons + neutrinos
        self.omega_darkenergy_0 = 1. - (self.omega_matter_0 + self.omega_radiation_0)
        self.w_darkenergy = -1. # equation of state of dark energy
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
        self.hubble_si = self.hubble_0 * Constants.KM_TO_M / Constants.MPC_TO_M # s^-1, Hubble Constant in SI units
        self.hubble_time = (self.hubble_si)**(-1) # s
        self.hubble_distance = Constants.C_LIGHT * self.hubble_time # m
        self.omega_0 = self.omega_matter_0 + self.omega_radiation_0 + self.omega_darkenergy_0
        if not np.isclose(self.omega_0, 1.):
            print('WARNING: input parameters correspond to universe with non-zero curvature')
        
    def _multiRedshift(self, z, z_second=None):
        """Handle a second redshift if provided.
        
        Args:
            z (float): redshift
            z_second (float): a second optional redshift
            
        Returns:
            float: lower redshift (m)
            float: higher redshift (m)
        """
        if z_second is not None and z_second > z:
            z_1 = z
            z_2 = z_second
        else:
            z_1 = 0.
            z_2 = z
        return z_1, z_2
        
    def comovingDistance(self, z, z_second=None):
        """Compute the comoving distance for a given redshift.
        
        Args:
            z (float): redshift
            z_second (float): a second redshift
            
        Returns:
            float: comoving distance (m)
        """
        z_1, z_2 = self._multiRedshift(z, z_second)
        
        omega_kappa = 1. - self.omega_0
        
        if np.isclose(omega_kappa, 0.):
            return self.hubble_distance * scipy.integrate.quad(self._EReciprocal, z_1, z_2)[0]
        elif omega_kappa > 0.:
            return self.hubble_distance \
                * np.sinh(np.sqrt(omega_kappa) * scipy.integrate.quad(self._EReciprocal, z_1, z_2)[0]) \
                / np.sqrt(omega_kappa)
        else:
            return self.hubble_distance \
                * np.sin(np.sqrt(np.fabs(omega_kappa)) * scipy.integrate.quad(self._EReciprocal, z_1, z_2)[0]) \
                / np.sqrt(np.fabs(omega_kappa))
            
    def show(self):
        """
        Display the current cosmological parameter values. 
        """
        print(self.__dict__)
        
    def _E(self, z):
        return np.sqrt(self.omega_matter_0 * (1. + z)**3 \
                       + self.omega_darkenergy_0 * (1. + z)**(3. * (1. + self.w_darkenergy)) \
                       + self.omega_radiation_0 * (1 + z)**4 \
                       + (1. - self.omega_0) * (1 + z)**2)
    
    def _EReciprocal(self, z):
        return 1. / self._E(z)
    
    def _EReciprocalLookbackTime(self, z):
        return 1. / ((1. + z) * self._E(z))

    def _Ea(self, a):
        return np.sqrt(self.omega_radiation_0 * a**(-2) \
                       + self.omega_matter_0 * a**(-1) \
                       + self.omega_darkenergy_0 * a**(-1. - (3. * self.w_darkenergy)) \
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
        
    def angularDistance(self, z, z_second=None):
        """Compute the angular distance for a given redshift.
        
        Args:
            z (float): redshift
            z_second (float): a second redshift
            
        Returns:
            float: angular distance (m)
        """
        z_1, z_2 = self._multiRedshift(z, z_second)
        return (1. + z_2)**(-1) * self.comovingDistance(z_1, z_2) # m
    
    def cosmicTime(self, a, a_init=1.e-15):
        """Compute the cosmic time corresponding to a given scale factor.
        
        Args:
            a (float): scale factor.
            a_init (float): scale factor at starting time.
                Defaults to 1.e-15
        
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
        return self.hubble_time * scipy.integrate.quad(self._EReciprocalLookbackTime, 0, z)[0] # s
        
    def horizonDistance(self, t):
        """Compute the horizon distance for a given redshift.
        
        Args:
            t (float): age of the Universe (s)
            
        Returns:
            float: horizon distance in comoving coordinates (m)
        """
        # Create a log-spaced set of scale factor values
        a_array = np.logspace(-15, 3, 10000)

        # For each scale factor value, compute the associated cosmic time
        t_array = np.empty(len(a_array))
        for ii in range(0, len(a_array)):
            t_array[ii] = self.cosmicTime(a_array[ii]) / self.hubble_si
        
        # Create interpolation function to use an integrand, scale factor as function of time
        a_reciprocal = scipy.interpolate.interp1d(t_array, 1. / a_array)
        
        return Constants.C_LIGHT * scipy.integrate.quad(a_reciprocal, 0., t)[0]
        
############################################################