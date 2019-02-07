from scipy import constants
import numpy as np
from pythonradex import helpers,escape_probability,atomic_transition
from pythonradex.molecule import EmittingMolecule


class RateEquations():

    '''Represents the equations of statistical equilibrium for the level populations
    of a molecule'''

    def __init__(self,molecule,coll_partner_densities,Tkin,mode='std'):
        '''molecule is an instance of the Molecule class
        coll_partner_densities is a dict with the densities of the collision partners
        considered
        Tkin is the kinetic temperature
        mode is the method to solve the radiative transfer: either std (i.e. lambda iteration)
        or ALI (i.e. accelerated lambda iteration)'''
        self.molecule = molecule
        self.coll_partner_densities = coll_partner_densities
        self.Tkin = Tkin
        assert mode in ('std','ALI')
        if mode == 'std':
            self.rad_rates = self.rad_rates_std
        elif mode == 'ALI':
            self.rad_rates = self.rad_rates_ALI

    def empty_rate_matrix(self):
        '''Return a zero matrix with the dimensions of the rate equation matrix'''
        return np.zeros((self.molecule.n_levels,self.molecule.n_levels))

    def rad_rates_std(self,Jbar_lines):
        '''compute the rates for the Lambda iteration method for radiative transitions,
        from the average radiation field for all lines given by Jbar_lines.'''
        uprate = [line.B12*Jbar_lines[i] for i,line in
                  enumerate(self.molecule.rad_transitions)]
        downrate = [line.A21+line.B21*Jbar_lines[i] for i,line in
                    enumerate(self.molecule.rad_transitions)]
        return np.array(uprate),np.array(downrate)

    def rad_rates_ALI(self,beta_lines,I_ext_lines):
        '''compute the rates for the accelerated Lambda iteration (ALI) method
        for radiative transitions, from the escape probability for all lines (beta_lines)
        and the external intensity for all lines (I_ext_lines).'''
        #see section 7.10 of the Dullemond radiative transfer lectures (in the
        #ALI_explained.pdf document)
        uprate = [line.B12*I_ext*beta for line,I_ext,beta in
                     zip(self.molecule.rad_transitions,I_ext_lines,beta_lines)]
        downrate = [line.A21*beta+line.B21*I_ext*beta for line,I_ext,beta in
                     zip(self.molecule.rad_transitions,I_ext_lines,beta_lines)]
        return np.array(uprate),np.array(downrate)

    def add_rad_rates(self,matrix,rad_rates):
        '''add the radiative rates to the matrix representing the equations of SE'''
        uprates,downrates = rad_rates
        for up,down,line in zip(uprates,downrates,self.molecule.rad_transitions):
            #production of low level from upper level
            matrix[line.low.number,line.up.number] +=  down
            #descruction of upper level towards lower level
            matrix[line.up.number,line.up.number] += -down
            #production of upper level from lower level
            matrix[line.up.number,line.low.number] += up
            #descruction of lower level towards upper level
            matrix[line.low.number,line.low.number] += -up

    def add_coll_rates(self,matrix):
        '''add the rates of collisional transitions to the matrix representing
        the equations of SE'''
        for coll_part_name,coll_part_dens in self.coll_partner_densities.items():
            for coll_trans in self.molecule.coll_transitions[coll_part_name]:
                coeffs = coll_trans.coeffs(self.Tkin)
                K12,K21 = coeffs['K12'],coeffs['K21']
                matrix[coll_trans.up.number,coll_trans.low.number]\
                                               += K12*coll_part_dens
                matrix[coll_trans.low.number,coll_trans.low.number]\
                                             += -K12*coll_part_dens
                matrix[coll_trans.low.number,coll_trans.up.number]\
                                               += K21*coll_part_dens
                matrix[coll_trans.up.number,coll_trans.up.number]\
                                              += -K21*coll_part_dens

    def solve(self,**kwargs):
        '''Solve the SE equations
           for std mode: kwargs = Jbar_lines
           for ALI mode: kwargs = beta_lines,I_ext_lines'''
        matrix = self.empty_rate_matrix()
        rad_rates = self.rad_rates(**kwargs)
        self.add_rad_rates(matrix=matrix,rad_rates=rad_rates)
        self.add_coll_rates(matrix)
        # the system of equations is not linearly independent
        #thus, I replace one equation by the normalisation condition,
        #i.e. x1+...+xn=1, where xi is the fractional population of level i
        #I replace the first equation (arbitrary choice):
        matrix[0,:] = np.ones(self.molecule.n_levels)
        #steady state; A*x=b, x=fractional population that we search:
        b = np.zeros(self.molecule.n_levels)
        b[0] = 1
        fractional_population = np.linalg.solve(matrix,b)
        return fractional_population


class Nebula():

    '''Represents an emitting gas cloud

    Attributes:
    ---------------

    - emitting_molecule: EmittingMolecule
        An object containing atomic data and line profile information
    
    - geometry: str
        geometry of the gas cloud

    - ext_background: func
        function returning the external background radiation field for given frequency

    - Tkin: float
        kinetic temperature of colliders

    - coll_partner_densities: dict
        densities of the collision partners

    - Ntot: float
        total column density

    - rate_equations: RateEquations
        object used to set up and solve the equations of statistical equilibrium

    - verbose: bool
        if True, additional information is printed out
        
    The following attributes are available after the radiative transfer has been solved:
    
    - tau_nu0: numpy.ndarray
        optical depth of each transition at the central frequency.
    - level_pop: numpy.ndarray
        fractional population of levels.
    - Tex: numpy.ndarray
        excitation temperature of each transition.
    '''

    relative_convergence = 1e-2
    min_iter = 30
    max_iter = 10000
    underrelaxation = 0.3
    geometries = {'uniform sphere':escape_probability.UniformSphere,
                  'uniform sphere RADEX':escape_probability.UniformSphereRADEX}
    line_profiles = {'Gaussian':atomic_transition.GaussianLineProfile,
                     'square':atomic_transition.SquareLineProfile}

    def __init__(self,data_filepath,geometry,ext_background,Tkin,
                 coll_partner_densities,Ntot,line_profile,width_v,verbose=False):
        '''
        Parameters:    
        ---------------        
        
        data_filepath: str
             path to the LAMDA data file that contains the atomic data

        geometry: str
            geometry of the gas cloud. Currently available are "uniform sphere"and "uniform sphere RADEX". The latter uses the forumla for a uniform sphere for the escape probability and the formula for a uniform slab to calculate the flux, as in RADEX.

        ext_background: func
            The function should take the frequency in Hz as input and return the background radiation field in [W/m2/Hz/sr]

        Tkin: float
            kinetic temperature of the colliders

        coll_partner_densities: dict
            number densities of the collision partners in [1/m3]. Following keys are recognised: "H2", "para-H2", "ortho-H2", "e", "H", "He", "H+"

        Ntot: float
            total column density in [1/m2]

        line_profile: str
            type of line profile. Available are "Gaussian" and "square".

        width_v: float
            width of the line in [m/s]. For Gaussian, this is the FWHM.

        verbose: bool
            if True, additional information is printed out
        '''
        self.emitting_molecule = EmittingMolecule.from_LAMDA_datafile(
                                    data_filepath=data_filepath,
                                    line_profile_cls=self.line_profiles[line_profile],
                                    width_v=width_v)
        self.geometry = self.geometries[geometry]()
        self.ext_background = ext_background
        self.Tkin = Tkin
        self.coll_partner_densities = coll_partner_densities
        self.Ntot = Ntot
        self.rate_equations = RateEquations(
                                  molecule=self.emitting_molecule,
                                  coll_partner_densities=self.coll_partner_densities,
                                  Tkin=self.Tkin,mode='ALI')
        self.verbose = verbose

    def beta_alllines(self,level_populations):
        '''compute the escape probability for all lines, given the level population'''
        beta = []
        for line in self.emitting_molecule.rad_transitions:
            N1 = self.Ntot * level_populations[line.low.number]
            N2 = self.Ntot * level_populations[line.up.number]
            beta_nu_array = self.geometry.beta(
                                  line.tau_nu_array(N1=N1,N2=N2))
            averaged_beta = line.line_profile.average_over_nu_array(beta_nu_array)
            beta.append(averaged_beta)
        return np.array(beta)

    def solve_radiative_transfer(self):
        """Solves the radiative transfer by iterating and initialises
        new attributes that contain the solution.
        """
        beta_lines = np.ones(self.emitting_molecule.n_rad_transitions)
        I_ext_lines = np.array([self.ext_background(line.nu0) for line in
                                self.emitting_molecule.rad_transitions])
        level_pop = self.rate_equations.solve(beta_lines=beta_lines,
                                              I_ext_lines=I_ext_lines)
        Tex_residual = np.ones(self.emitting_molecule.n_rad_transitions) * np.inf
        old_Tex = 0
        counter = 0
        while np.any(Tex_residual > self.relative_convergence) or\
                  counter < self.min_iter:
            counter += 1
            if counter%10 == 0 and self.verbose:
                print('iteration {:d}'.format(counter))
            if counter > self.max_iter:
                raise RuntimeError('maximum number of iterations reached')
            new_level_pop = self.rate_equations.solve(
                                 beta_lines=beta_lines,I_ext_lines=I_ext_lines)
            Tex = self.emitting_molecule.get_Tex(new_level_pop)
            Tex_residual = helpers.relative_difference(Tex,old_Tex)
            if self.verbose:
                print('max relative Tex residual: {:g}'.format(np.max(Tex_residual)))
            old_Tex = Tex.copy()
            level_pop = self.underrelaxation*new_level_pop\
                        + (1-self.underrelaxation)*level_pop
            beta_lines = self.beta_alllines(level_pop)
        if self.verbose:
            print('converged in {:d} iterations'.format(counter))
        self.tau_nu0 = self.emitting_molecule.get_tau_nu0(
                                   N=self.Ntot,level_population=level_pop)
        self.level_pop = level_pop
        self.Tex = self.emitting_molecule.get_Tex(level_pop)
        self.compute_line_fluxes()

    def compute_line_fluxes(self):
        '''Compute the flux for each line. This requires that the radiative transfer
        has been solved'''
        self.line_fluxes = []
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            nu_array = line.line_profile.nu_array
            x1 = self.level_pop[line.low.number]
            x2 = self.level_pop[line.up.number]
            source_function = helpers.B_nu(T=self.Tex[i],nu=nu_array)
            tau_nu = line.tau_nu_array(N1=x1*self.Ntot,N2=x2*self.Ntot)
            line_flux_nu = self.geometry.compute_flux_nu(
                                              tau_nu=tau_nu,
                                              source_function=source_function)
            line_flux = np.trapz(line_flux_nu,line.line_profile.nu_array) #[W/m2]
            self.line_fluxes.append(line_flux)

    def observed_fluxes(self,source_surface,d_observer):
        '''Compute the flux recorded at the telescope. Can only be called if
        the radiative transfer has been solved.
        
        Parameters:
        ---------------
        source_surface: float
            the surface of the emitting cloud in [m2]
        d_observer:
            the distance between observer and emitting cloud in [m]
        
        Returns:
        -----------
        numpy.ndarray
            the flux in W/m2 seen by the observer, for each radiative transition'''
        obs_fluxes = np.empty(len(self.emitting_molecule.rad_transitions))
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            flux = self.line_fluxes[i]*source_surface/(4*np.pi*d_observer**2)
            obs_fluxes[i] = flux
        return obs_fluxes

    def print_results(self):
        '''print out the results from the radiative transfer computation. Can
        only be called if the radiative transfer has been solved.'''
        print('\n')
        print('  up   low      nu [GHz]    T_ex [K]      poplow         popup          tau_nu0')
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            rad_trans_string = '{:>4d} {:>4d} {:>14.6f} {:>10.2f} {:>14g} {:>14g} {:>14g}'
            rad_trans_format = (line.up.number,line.low.number,
                                line.nu0/constants.giga,self.Tex[i],
                                self.level_pop[line.low.number],
                                self.level_pop[line.up.number],
                                self.tau_nu0[i])
            print(rad_trans_string.format(*rad_trans_format))
        print('\n')