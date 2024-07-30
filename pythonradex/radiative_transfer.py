from scipy import constants
import numpy as np
from pythonradex import helpers,escape_probability,atomic_transition
from pythonradex.molecule import EmittingMolecule
import warnings
import numba as nb
import itertools


class RateEquations():

    def __init__(self,molecule,collider_densities,Tkin,mode):
        self.molecule = molecule
        self.collider_densities = collider_densities
        self.Tkin = Tkin
        def is_requested(collider):
            return collider in collider_densities
        self.collider_selection = [is_requested(collider) for collider in
                                   self.molecule.ordered_colliders]
        self.collider_selection = nb.typed.List(self.collider_selection)
        self.collider_densities_list = nb.typed.List([])
        #important to iterate over ordered_colliders, not collider_densities.items()
        for collider in self.molecule.ordered_colliders:
            if is_requested(collider):
                #need to convert to float, otherwise numba can get confused
                self.collider_densities_list.append(float(collider_densities[collider]))
            else:
                self.collider_densities_list.append(np.inf)
        self.coll_rate_matrix = self.construct_coll_rate_matrix(
                    Tkin=np.atleast_1d(self.Tkin),
                    collider_selection=self.collider_selection,
                    collider_densities_list=self.collider_densities_list,
                    n_levels=self.molecule.n_levels,
                    coll_trans_low_up_number=self.molecule.coll_trans_low_up_number,
                    Tkin_data = self.molecule.coll_Tkin_data,
                    K21_data=self.molecule.coll_K21_data,gups=self.molecule.coll_gups,
                    glows=self.molecule.coll_glows,DeltaEs=self.molecule.coll_DeltaEs)
        assert mode in ('LI','ALI')
        self.mode = mode
        if self.mode == 'LI':
            self.rad_rates = self.rad_rates_LI
        elif self.mode == 'ALI':
            self.rad_rates = self.rad_rates_ALI
        self.up_number_lines = np.array([line.up.number for line in
                                         self.molecule.rad_transitions])
        self.low_number_lines = np.array([line.low.number for line in
                                          self.molecule.rad_transitions])
        #steady state; A*x=b, x=fractional population that we search
        self.b = np.zeros(self.molecule.n_levels)
        self.b[0] = 1

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def construct_coll_rate_matrix(
              Tkin,collider_selection,collider_densities_list,n_levels,
              coll_trans_low_up_number,Tkin_data,K21_data,gups,glows,DeltaEs):
        '''pre-calculate the matrix with the collsional rates, so that it can
        easily be added during solving the radiative transfer'''
        coll_rate_matrix = np.zeros((n_levels,n_levels))
        for i in range(len(collider_selection)):
            if not collider_selection[i]:
                continue
            coll_density = collider_densities_list[i]
            n_transitions = len(DeltaEs[i])
            for j in range(n_transitions):
                K12,K21 = atomic_transition.fast_coll_coeffs(
                          Tkin=Tkin,Tkin_data=Tkin_data[i][j],
                          K21_data=K21_data[i][j],gup=gups[i][j],glow=glows[i][j],
                          Delta_E=DeltaEs[i][j])
                #K12 and K21 are 1D arrays because Tkin is a 1D array
                K12 = K12[0]
                K21 = K21[0]
                n_low = coll_trans_low_up_number[i][j,0]
                n_up = coll_trans_low_up_number[i][j,1]
                coll_rate_matrix[n_up,n_low] += K12*coll_density
                coll_rate_matrix[n_low,n_low] += -K12*coll_density
                coll_rate_matrix[n_low,n_up] += K21*coll_density
                coll_rate_matrix[n_up,n_up] += -K21*coll_density
        assert np.all(np.isfinite(coll_rate_matrix))
        return coll_rate_matrix

    @staticmethod
    #@nb.jit(nopython=True,cache=True) #doesn't help
    def rad_rates_LI(Jbar_lines,A21_lines,B12_lines,B21_lines):
        '''compute the rates for the Lambda iteration method for radiative transitions,
        from the average radiation field for all lines given by Jbar_lines.'''
        uprate = B12_lines*Jbar_lines
        downrate = A21_lines+B21_lines*Jbar_lines
        return uprate,downrate

    @staticmethod
    #@nb.jit(nopython=True,cache=True)#doesn't help
    def rad_rates_ALI(beta_lines,betaIext_lines,A21_lines,B12_lines,B21_lines):
        #see section 7.10 of the Dullemond radiative transfer lectures (in the
        #ALI_explained.pdf document)
        uprate = B12_lines*betaIext_lines
        downrate = A21_lines*beta_lines + B21_lines*betaIext_lines
        return uprate,downrate

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def add_rad_rates(matrix,rad_rates,low_number_lines,up_number_lines):
        uprates,downrates = rad_rates
        for i in range(low_number_lines.size):
            ln = low_number_lines[i]
            un = up_number_lines[i]
            down = downrates[i]
            up = uprates[i]
            #production of low level from upper level
            matrix[ln,un] +=  down
            #descruction of upper level towards lower level
            matrix[un,un] += -down
            #production of upper level from lower level
            matrix[un,ln] += up
            #descruction of lower level towards upper level
            matrix[ln,ln] += -up
        return matrix

    @staticmethod
    #@nb.jit(nopython=True,cache=True) #doesn't help
    def add_coll_rates(matrix,coll_rate_matrix):
        return matrix + coll_rate_matrix

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def apply_normalisation_condition(matrix,n_levels):
        # the system of equations is not linearly independent
        #thus, I replace one equation by the normalisation condition,
        #i.e. x1+...+xn=1, where xi is the fractional population of level i
        #I replace the first equation (arbitrary choice):
        matrix[0,:] = np.ones(n_levels)
        return matrix

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_solve(matrix,b):
        return np.linalg.solve(matrix,b)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def assert_frac_pop_positive(fractional_population):
        assert np.all(fractional_population >= 0),\
                  'negative level population, potentially due to high column'\
                  +'density and/or low collider density'

    def solve(self,**kwargs):
        matrix = np.zeros((self.molecule.n_levels,self.molecule.n_levels))
        rad_rates = self.rad_rates(**kwargs)
        matrix = self.add_rad_rates(
                     matrix=matrix,rad_rates=rad_rates,
                     low_number_lines=self.low_number_lines,
                     up_number_lines=self.up_number_lines)
        matrix = self.add_coll_rates(matrix=matrix,coll_rate_matrix=self.coll_rate_matrix)
        matrix = self.apply_normalisation_condition(
                                         matrix=matrix,n_levels=self.molecule.n_levels)
        fractional_population = self.fast_solve(matrix=matrix,b=self.b)
        self.assert_frac_pop_positive(fractional_population=fractional_population)
        return fractional_population


class Cloud():

    '''
    Solving the non-LTE radiative transfer using escape probabilities.

    Attributes:

        emitting_molecule (pythonradex.molecule.EmittingMolecule): object
            containing all the information about the emitting atom or molecule
        tau_nu0 (numpy.ndarray): optical depth of each transition at the central frequency
        level_pop (numpy.ndarray): fractional population of each level
        Tex (numpy.ndarray): excitation temperature of each transition

    Note:
        The attributes tau_nu0, level_pop and Tex are available only after solving the radiative transfer by calling solve_radiative_transfer
        
    '''
    relative_convergence = 1e-6
    min_iter = 10
    max_iter = 1000
    underrelaxation = 0.3 #RADEX uses 0.3
    min_tau_considered_for_convergence = 1e-2
    min_iter_before_ng_acceleration = 4
    ng_acceleration_interval = 4
    #for calculation of flux:
    tau_peak_fraction = 1e-2
    nu_per_FHWM = 20
    geometries = {'uniform sphere':escape_probability.UniformSphere,
                  'uniform sphere RADEX':escape_probability.UniformSphereRADEX,
                  'uniform slab':escape_probability.UniformSlab,
                  'LVG slab':escape_probability.UniformLVGSlab,
                  'LVG sphere':escape_probability.UniformLVGSphere,
                  'LVG sphere RADEX':escape_probability.LVGSphereRADEX}
    line_profiles = {'Gaussian':atomic_transition.GaussianLineProfile,
                     'rectangular':atomic_transition.RectangularLineProfile}

    def __init__(self,datafilepath,geometry,line_profile_type,width_v,iteration_mode='ALI',
                 use_NG_acceleration=True,average_over_line_profile=False,
                 treat_overlapping_lines=False,warn_negative_tau=True,verbose=False,
                 test_mode=False):
        '''Initialises a new instance of the Cloud class.

        Args:
            datafilepath (:obj:`str`): filepath to the LAMDA file that contains
                the atomic / molecular data
            geometry (:obj:`str`): determines how the escape probability
                and flux are calculated. Available options: "uniform sphere", 
                "uniform sphere RADEX", "uniform slab", "LVG slab", "LVG sphere" and
                "LVG sphere RADEX". The options containing "RADEX" are meant to
                mimic the behaviour of the original RADEX code by using the same
                equations as RADEX.
            line_profile_type (:obj:`str`): The type of the line profile. Available options:
                "rectangular" or "Gaussian". Note that for geometries "LVG sphere"
                and "LVG slab", only "rectangular" is allowed.
            width_v (:obj:`float`): The width of the line profile in [m/s]. For a Gaussian
                profile, this is interpreted as the FWHM. Note that the intepretation
                of this parameter depends on the adopted geometry. For the static
                geometries, width_v is the width of the intrinsic emission profile.
                On the other hand, for geometries "LVG sphere" and "LVG slab",
                width_v corresponds to the total velocity witdh of the cloud.
                So, for "LVG sphere", width_v=2*V, where V is the velocity at the surface
                of the sphere. In terms of the constant velocity gradient dv/dr=V/R (with
                R the radius of the sphere), we can also say width_v=dv/dr*2*R. For
                "LVG slab", width_v=dv/dz*Z where Z is the depth of the slab and dv/dz
                the constant velocity gradient of the slab.
            iteration_mode (:obj:`str`): Method used to solve the radiative transfer:
                "LI" for standard Lambda iteration, or "ALI" for Accelerated Lambda
                Iteration. ALI is recommended. Defaults to "ALI".
            use_NG_acceleration (:obj:`bool`): Whether to use Ng acceleration. Defaults
                to True.
            average_over_line_profile (:obj:`bool`): Whether to average the escape
                probability, source function etc. over the line profile, or just take the value
                at the rest frequency (like RADEX). Defaults to False. Setting this to true makes the
                calculation slower. To treat overlapping lines, this needs to be activated.
            treat_overlapping_lines (:obj:`bool`): Whether to treat overlapping lines.
                Defaults to False. Can only be used in combination with static geometries
                (i.e. not with LVG)
            warn_negative_tau (:obj:`bool`): Whether the raise a warning when negative
                optical depth is encountered. Defaults to True. Setting this to False
                is useful when calculating a grid of models.
            verbose (:obj:`bool`): Whether to print additional information. Defaults to False.
            test_mode (:obj:`bool`): Enter test mode. Only for developer, should not be used
                by general user. Defaults to False.
        
        '''
        self.line_profile_type = line_profile_type
        assert width_v < 10000*constants.kilo,\
                      'assumption of small nu0/Delta_nu for flux calculation not satisfied'
        self.emitting_molecule = EmittingMolecule(
                                    datafilepath=datafilepath,
                                    line_profile_type=self.line_profile_type,
                                    width_v=width_v)
        self.average_over_line_profile = average_over_line_profile
        self.treat_overlapping_lines = treat_overlapping_lines
        self.identify_overlapping_lines()
        if self.treat_overlapping_lines:
            if not self.average_over_line_profile:
                raise ValueError('treating overlapping lines requires activation'
                                 ' line profile averaging')
        if self.any_line_has_overlap(
                 line_indices=range(self.emitting_molecule.n_rad_transitions))\
            and not self.treat_overlapping_lines:
            warnings.warn('lines of input molecule are overlapping, '
                          +'but treatment of overlapping lines not activated')
        if verbose:
            for i in range(len(self.overlapping_lines)):
                if len(self.overlapping_lines[i]) > 0:
                    print(f'identified overlaps for line {i}: {self.overlapping_lines[i]}')
        self.geometry_name = geometry
        if self.geometry_name in ('LVG sphere','LVG slab') and not test_mode:
            if not self.line_profile_type == 'rectangular':
                #this is to get the right optical depth which is independent of
                #velocity in this model (e.g. Eq. 13 in de Jong et al. 1975)
                raise ValueError(f'{self.geometry_name} requires a rectangular profile')
        if self.geometry_name == 'LVG sphere':
            #velocity at the surface of the LVG sphere
            self.V_LVG_sphere = self.emitting_molecule.width_v/2
        self.geometry = self.geometries[geometry]()
        self.iteration_mode = iteration_mode
        self.use_NG_acceleration = use_NG_acceleration
        # if average_beta_over_line_profile:
        #     self.beta_alllines = self.beta_alllines_average
        # else:
        #     self.beta_alllines = self.beta_alllines_nu0
        self.warn_negative_tau = warn_negative_tau
        self.verbose = verbose
        #the following attributes are needed for the compiled functions:
        rad_trans = self.emitting_molecule.rad_transitions
        self.A21_lines = np.array([line.A21 for line in rad_trans])
        self.B12_lines = np.array([line.B12 for line in rad_trans])
        self.B21_lines = np.array([line.B21 for line in rad_trans])
        self.DeltaE_lines = np.array([line.Delta_E for line in rad_trans])
        # self.coarse_nu_lines = np.array([line.line_profile.coarse_nu_array for line in
        #                           rad_trans])
        self.nu0_lines = np.array([line.nu0 for line in rad_trans])
        # self.phi_nu_lines = np.array([line.line_profile.coarse_phi_nu_array for
        #                               line in rad_trans])
        self.phi_nu0_lines = np.array([line.line_profile.phi_nu0 for line in
                                       rad_trans])
        self.gup_lines = np.array([line.up.g for line in rad_trans])
        self.glow_lines = np.array([line.low.g for line in rad_trans])
        self.low_number_lines = np.array([line.low.number for line in rad_trans])
        self.up_number_lines = np.array([line.up.number for line in rad_trans])

    def set_parameters(self,ext_background,N,Tkin,collider_densities):
        r'''Set the parameters for a new radiative transfer calculation.

        Args:
            ext_background (func): A function taking the frequency in Hz as input
                and returning the background radiation field in [W/m\ :sup:`2`/Hz/sr].
            N (:obj:`float`): The column density in [m\ :sup:`-2`].
            Tkin (:obj:`float`): The kinetic temperature of the gas in [K].
            collider_densities (dict): A dictionary of the number densities of
               each collider that should be considered, in units of [m\ :sup:`-3`]. The
               following keys are recognised: "H2", "para-H2", "ortho-H2", "e",
               "H", "He", "H+"
        '''
        #why do I put this into a seperate method rather than __init__? The reason
        #is that the stuff in __init__ is expensive (in particular reading the
        #LAMDA file). So if I explore e.g. a grid of N, that stuff in __init__
        #should only be done once to save computation time. Then this method
        #can be called inside the loop over e.g. the N values
        #note that setting up the rate equations is expensive, so I only do it if
        #necessary
        self.ext_background = ext_background
        if not self.ext_background_is_slowly_varying() and\
                                      not self.average_over_line_profile:
            raise ValueError('external background is varying over line profile. Please'
                             +' activate averaging over line profile')
        #needed for the compiled functions:
        self.I_ext_lines = np.array([self.ext_background(line.nu0) for line in
                                     self.emitting_molecule.rad_transitions])
        self.N = N
        if (not hasattr(self,'Tkin')) or (not hasattr(self,'collider_densities')):
            Tkin_or_collider_density_changed = True
        else:
            Tkin_or_collider_density_changed\
                    = (self.Tkin != Tkin) or (self.collider_densities != collider_densities)
        if Tkin_or_collider_density_changed:
            self.Tkin = Tkin
            self.collider_densities = collider_densities
            self.rate_equations = RateEquations(
                                      molecule=self.emitting_molecule,
                                      collider_densities=self.collider_densities,
                                      Tkin=self.Tkin,mode=self.iteration_mode)

    def ext_background_is_slowly_varying(self):
        nu0 = self.emitting_molecule.nu0_lines
        Delta_nu = self.emitting_molecule.width_v/constants.c*nu0
        Iext_nu0 = np.array((self.ext_background(nu0),))
        Iext_nu0_plus_Deltanu = np.array((self.ext_background(nu0+Delta_nu),))
        relative_diff = helpers.relative_difference(Iext_nu0,Iext_nu0_plus_Deltanu)
        if np.any(relative_diff>1e-2):
            return False
        else:
            return True

    # @staticmethod
    # @nb.jit(nopython=True,cache=True)
    # def fast_averaged_beta_lines(level_populations,low_number_lines,up_number_lines,N,
    #                              A21_lines,phi_nu_lines,gup_lines,glow_lines,nu_lines,
    #                              beta_function):
    #     n_lines = low_number_lines.size
    #     averaged_beta = np.empty(n_lines)
    #     for i in range(n_lines):
    #         N1 = N * level_populations[low_number_lines[i]]
    #         N2 = N * level_populations[up_number_lines[i]]
    #         tau_nu = atomic_transition.fast_tau_nu(
    #                       A21=A21_lines[i],phi_nu=phi_nu_lines[i,:],
    #                       g_up=gup_lines[i],g_low=glow_lines[i],N1=N1,N2=N2,
    #                       nu=nu_lines[i,:])
    #         beta_nu = beta_function(tau_nu)
    #         #phi_nu is already normalised, but since I use the coarse grid here,
    #         #I re-normalise
    #         phi_nu = phi_nu_lines[i,:]
    #         nu = nu_lines[i,:]
    #         norm = helpers.fast_trapz(y=phi_nu,x=nu)
    #         averaged_beta[i] = helpers.fast_trapz(y=beta_nu*phi_nu,x=nu)/norm
    #     return averaged_beta

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_beta_nu0_lines(level_populations,low_number_lines,up_number_lines,N,
                            A21_lines,phi_nu0_lines,gup_lines,glow_lines,nu0_lines,
                            beta_function):
        n_lines = low_number_lines.size
        tau_nu0_lines = np.empty(n_lines)
        for i in range(n_lines):
            N1 = N * level_populations[low_number_lines[i]]
            N2 = N * level_populations[up_number_lines[i]]
            tau_nu0_lines[i] = atomic_transition.fast_tau_nu(
                                 A21=A21_lines[i],phi_nu=phi_nu0_lines[i],
                                 g_up=gup_lines[i],g_low=glow_lines[i],N1=N1,N2=N2,
                                 nu=nu0_lines[i])
        return beta_function(tau_nu0_lines)

    # def beta_alllines_average(self,level_populations):
    #     return self.fast_averaged_beta_lines(
    #                     level_populations=level_populations,
    #                     low_number_lines=self.low_number_lines,
    #                     up_number_lines=self.up_number_lines,N=self.N,
    #                     A21_lines=self.A21_lines,phi_nu_lines=self.phi_nu_lines,
    #                     gup_lines=self.gup_lines,glow_lines=self.glow_lines,
    #                     nu_lines=self.coarse_nu_lines,beta_function=self.geometry.beta)

    def beta_alllines_without_average(self,level_populations):
        return self.fast_beta_nu0_lines(
                       level_populations=level_populations,
                       low_number_lines=self.low_number_lines,
                       up_number_lines=self.up_number_lines,N=self.N,
                       A21_lines=self.A21_lines,phi_nu0_lines=self.phi_nu0_lines,
                       gup_lines=self.gup_lines,glow_lines=self.glow_lines,
                       nu0_lines=self.nu0_lines,beta_function=self.geometry.beta)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_source_function_alllines_without_average(
               level_populations,low_number_lines,up_number_lines,A21_lines,
               B21_lines,B12_lines):
        n_lines = low_number_lines.size
        source_func = np.zeros(n_lines)
        for i in range(n_lines):
            x1 = level_populations[low_number_lines[i]]
            x2 = level_populations[up_number_lines[i]]
            if x1==x2==0:
                continue
            else:
                s = A21_lines[i]*x2/(x1*B12_lines[i]-x2*B21_lines[i])
                source_func[i] = s
        return source_func

    def source_function_alllines_without_average(self,level_populations):
        return self.fast_source_function_alllines_without_average(
                 level_populations=level_populations,
                 low_number_lines=self.low_number_lines,
                 up_number_lines=self.up_number_lines,A21_lines=self.A21_lines,
                 B21_lines=self.B21_lines,B12_lines=self.B12_lines)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Jbar_allines_without_average(beta_lines,I_ext_lines,source_func):
        return beta_lines*I_ext_lines + (1-beta_lines)*source_func

    def Jbar_alllines_without_average(self,level_populations):
        beta_lines = self.beta_alllines_without_average(level_populations=level_populations)
        source_func = self.source_function_alllines_without_average(
                                       level_populations=level_populations)
        return self.fast_Jbar_allines_without_average(
                    beta_lines=beta_lines,I_ext_lines=self.I_ext_lines,
                    source_func=source_func)

    def Jbar_alllines_averaged(self,level_populations):
        Jbar = np.empty(self.emitting_molecule.n_rad_transitions)
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            x1,x2 = level_populations[line.low.number],level_populations[line.up.number]
            N1,N2 = x1*self.N,x2*self.N
            if self.treat_overlapping_lines:
                raise NotImplementedError
            else:
                def tau_func(nu):
                    return line.tau_nu(N1=N1,N2=N2,nu=nu)
                def source_func(nu):
                    return line.A21*x2/(x1*line.B12-x2*line.B21)
            def J(nu):
                beta = self.geometry.beta(tau_func(nu))
                Iext = self.ext_background(nu)
                S = source_func(nu)
                return beta*Iext + (1-beta)*S
            Jbar[i] = line.line_profile.average_over_phi_nu(J)
        return Jbar

    def beta_allines_averaged(self,level_populations):
        beta_bar = np.empty(self.emitting_molecule.n_rad_transitions)
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            x1,x2 = level_populations[line.low.number],level_populations[line.up.number]
            N1,N2 = x1*self.N,x2*self.N
            if self.treat_overlapping_lines:
                raise NotImplementedError
            else:
                def tau_func(nu):
                    return line.tau_nu(N1=N1,N2=N2,nu=nu)
            def beta(nu):
                return self.geometry.beta(tau_func(nu))
            beta_bar[i] = line.line_profile.average_over_phi_nu(beta)
        return beta_bar

    def betaIext_alllines_averaged(self,level_populations):
        betaIext_bar = np.empty(self.emitting_molecule.n_rad_transitions)
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            x1,x2 = level_populations[line.low.number],level_populations[line.up.number]
            N1,N2 = x1*self.N,x2*self.N
            if self.treat_overlapping_lines:
                raise NotImplementedError
            else:
                def tau_func(nu):
                    return line.tau_nu(N1=N1,N2=N2,nu=nu)
            def betaIext(nu):
                return self.geometry.beta(tau_func(nu))*self.ext_background(nu)
            betaIext_bar[i] = line.line_profile.average_over_phi_nu(betaIext)
        return betaIext_bar

    def get_new_level_pop(self,old_level_pop):
        assert self.rate_equations.mode in ('ALI','LI')
        Einstein_kwargs = {'B12_lines':self.B12_lines,'B21_lines':self.B21_lines,
                           'A21_lines':self.A21_lines}
        if old_level_pop is None:
            #assume optically thin emission for the first iteration
            if self.rate_equations.mode == 'ALI':
                beta_lines = np.ones(self.emitting_molecule.n_rad_transitions)
                return self.rate_equations.solve(beta_lines=beta_lines,
                                                 betaIext_lines=beta_lines*self.I_ext_lines,
                                                 **Einstein_kwargs)
            elif self.rate_equations.mode == 'LI':
                Jbar_lines = self.I_ext_lines
                return self.rate_equations.solve(Jbar_lines=Jbar_lines,
                                                 **Einstein_kwargs)
        else:
            if self.average_over_line_profile:
                if self.rate_equations.mode == 'ALI':
                    if self.treat_overlapping_lines:
                        raise NotImplementedError
                    else:
                        beta_lines = self.beta_allines_averaged(
                                                level_populations=old_level_pop)
                        betaIext_lines = self.betaIext_alllines_averaged(
                                                level_populations=old_level_pop)
                        return self.rate_equations.solve(beta_lines=beta_lines,
                                                         betaIext_lines=betaIext_lines,
                                                         **Einstein_kwargs)
                else:
                    Jbar_lines = self.Jbar_alllines_averaged(level_populations=old_level_pop)
                    return self.rate_equations.solve(Jbar_lines=Jbar_lines,
                                                     **Einstein_kwargs)
            else:
                if self.rate_equations.mode == 'ALI':
                    beta_lines = self.beta_alllines_without_average(
                                                level_populations=old_level_pop)
                    return self.rate_equations.solve(
                                   beta_lines=beta_lines,
                                   betaIext_lines=beta_lines*self.I_ext_lines,
                                   **Einstein_kwargs)
                elif self.rate_equations.mode == 'LI':
                    Jbar_lines = self.Jbar_alllines_without_average(
                                          level_populations=old_level_pop)
                    return self.rate_equations.solve(Jbar_lines=Jbar_lines,
                                                     **Einstein_kwargs)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def ng_accelerate(level_pop,old_level_pops):
        #implemented in the same way as JADEX
        #https://github.com/autocorr/Jadex.jl/blob/main/src/solver.jl
        #see Dullemond lecture notes (ALI_basics_and_NGAcceleratin.pdf) for the
        #formulas
        #the first element of old_level_pops is the oldest level population
        x1 = level_pop
        x2 = old_level_pops[2]
        x3 = old_level_pops[1]
        x4 = old_level_pops[0]
        q1 = x1 - 2*x2 + x3
        q2 = x1 - x2 - x3 + x4
        q3 = x1 - x2
        A1 = np.sum(q1*q1)
        A2 = np.sum(q2*q1)
        B1 = A2  # q1*q2
        B2 = np.sum(q2*q2)
        C1 = np.sum(q1*q3)
        C2 = np.sum(q2*q3)
        denom = A1 * B2 - A2 * B1
        if denom > 0:
            a = (C1 * B2 - C2 * B1) / denom
            b = (C2 * A1 - C1 * A2) / denom
            accelerated_level_pop = (1 - a - b) * x1 + a * x2 + b * x3
        else:
            accelerated_level_pop = level_pop
        return accelerated_level_pop

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def compute_residual(Tex_residual,tau_lines,min_tau_considered_for_convergence):
        #same as RADEX: consider only lines above a minimum tau for convergence
        selection = tau_lines > min_tau_considered_for_convergence
        n_selected = selection.sum()
        if n_selected > 0:
            return np.sum(Tex_residual[selection]) / n_selected
        else:
            return 0

    def solve_radiative_transfer(self):
        """Solves the radiative transfer."""
        level_pop = self.get_new_level_pop(old_level_pop=None)
        old_level_pops = nb.typed.List([])
        Tex_residual = np.ones(self.emitting_molecule.n_rad_transitions) * np.inf
        old_Tex = 0
        counter = 0
        residual = np.inf
        while residual>self.relative_convergence or counter < self.min_iter:
            counter += 1
            if counter%10 == 0 and self.verbose:
                print(f'iteration {counter}')
            if counter > self.max_iter:
                raise RuntimeError('maximum number of iterations reached')
            new_level_pop = self.get_new_level_pop(old_level_pop=level_pop)
            Tex = self.emitting_molecule.get_Tex(new_level_pop)
            Tex_residual = helpers.relative_difference(Tex,old_Tex)
            if self.verbose:
                print(f'max relative Tex residual: {np.max(Tex_residual):.3g}')
            tau_nu0_lines = self.emitting_molecule.get_tau_nu0(
                                       N=self.N,level_population=new_level_pop)
            residual = self.compute_residual(
                 Tex_residual=Tex_residual,tau_lines=tau_nu0_lines,
                 min_tau_considered_for_convergence=self.min_tau_considered_for_convergence)
            old_Tex = Tex.copy()
            old_level_pops.append(level_pop.copy())
            if len(old_level_pops) > 3:
                old_level_pops.pop(0)
            level_pop = self.underrelaxation*new_level_pop\
                                    + (1-self.underrelaxation)*level_pop
            if self.use_NG_acceleration and counter>self.min_iter_before_ng_acceleration\
                and counter%self.ng_acceleration_interval==0:
                level_pop = self.ng_accelerate(level_pop=level_pop,
                                               old_level_pops=old_level_pops)
        if self.verbose:
            print(f'converged in {counter} iterations')
        self.tau_nu0 = self.emitting_molecule.get_tau_nu0(
                                   N=self.N,level_population=level_pop)
        if self.warn_negative_tau:
            if np.any(self.tau_nu0 < 0):
                negative_tau_transition_indices = np.where(self.tau_nu0 < 0)[0]
                negative_tau_transitions = [self.emitting_molecule.rad_transitions[i]
                                            for i in negative_tau_transition_indices]
                warnings.warn('negative optical depth!')
                for i,trans in zip(negative_tau_transition_indices,
                                    negative_tau_transitions):
                    print(f'{trans.name}: tau_nu0 = {self.tau_nu0[i]:.3g}')
        self.level_pop = level_pop
        self.Tex = self.emitting_molecule.get_Tex(level_pop)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_fluxes_rectangular(
                 solid_angle,transitions,tau_nu0_lines,Tex,nu0_lines,compute_flux_nu,
                 width_v):
        #I am neglecting the dependence of tau on 1/nu**2 (and corresponding dependence
        #of beta), and the dependence on nu of the source function
        obs_line_fluxes = []
        for i in transitions:
            nu0 = nu0_lines[i]
            tau_nu0 = tau_nu0_lines[i]
            width_nu = width_v/constants.c*nu0
            source_function_nu0 = helpers.B_nu(T=Tex[i],nu=nu0)
            flux_nu0 = compute_flux_nu(tau_nu=np.array((tau_nu0,)),
                                       source_function=source_function_nu0,
                                       solid_angle=solid_angle)
            flux = flux_nu0*width_nu
            obs_line_fluxes.append(flux[0])
        return np.array(obs_line_fluxes)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_fluxes_LVG_sphere(
                 solid_angle,transitions,tau_nu0_lines,Tex,nu0_lines,
                 V_LVG_sphere):
        #here again I neglect the dependence of tau and the source function on nu
        obs_line_fluxes = []
        for i in transitions:
            nu0 = nu0_lines[i]
            tau_nu0 = tau_nu0_lines[i]
            source_function_nu0 = helpers.B_nu(T=Tex[i],nu=nu0)
            tau_factor = escape_probability.exp_tau_factor(tau_nu=tau_nu0)
            #the following formula is derived by analytically integrating the formula
            #for the spectrum
            flux = solid_angle*source_function_nu0*tau_factor\
                                     *4/3*V_LVG_sphere*nu0/constants.c
            obs_line_fluxes.append(flux)
        return np.array(obs_line_fluxes)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_fluxes_Gaussian(
                   solid_angle,transitions,tau_nu0_lines,Tex,nu0_lines,
                   compute_flux_nu,width_v,tau_peak_fraction,nu_per_FHWM):
        obs_line_fluxes = []
        for i in transitions:
            nu0 = nu0_lines[i]
            tau_nu0 = tau_nu0_lines[i]
            FWHM_nu = width_v/constants.c*nu0
            sigma_nu = helpers.FWHM2sigma(FWHM_nu)
            #for thin emission, I just want to include out to a certain fraction of the peak
            #but for thick emission, the spectrum is saturated, so a fraction of the
            #peak is not useful; in those cases, I need to set an absolute value
            #for the minimum tau to include
            min_tau = np.min(np.array([0.01,tau_peak_fraction*tau_nu0]))
            Delta_nu = sigma_nu*np.sqrt(-2*np.log(min_tau/tau_nu0))
            n_nu = int(2*Delta_nu/FWHM_nu * nu_per_FHWM)
            nu = np.linspace(nu0-Delta_nu,nu0+Delta_nu,n_nu)
            #doing an approximation: in principle, tau has an additional 1/nu**2 dependence,
            #but if Delta_nu is small compared to nu0, that dependence is negligible
            phi_nu_shape = np.exp(-(nu-nu0)**2/(2*sigma_nu**2))
            tau_nu = tau_nu0*phi_nu_shape
            source_function = helpers.B_nu(T=Tex[i],nu=nu)
            spectrum = compute_flux_nu(tau_nu=tau_nu,source_function=source_function,
                                       solid_angle=solid_angle)
            flux = helpers.fast_trapz(spectrum,nu)
            obs_line_fluxes.append(flux)
        return np.array(obs_line_fluxes)

    def all_transitions(self):
        return list(range(self.emitting_molecule.n_rad_transitions))

    def fluxes(self,solid_angle,transitions=None):
        r'''Calculate the observed fluxes.

        Args:
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
            transitions (:obj:`list` of :obj:`str` or None): The indices of the
                transitions for which to calculate the fluxes. If None, then the
                fluxes of all transitions are calculated. Defaults to None. The
                indices are relative to the list of transitions in the LAMDA file,
                starting with 0.
        
        Returns:
            list: The list of fluxes in [W/m\ :sup:`2`] corresponding to the input list of requested
            transitions. If not specific transitions where requested (transitions=None),
            then the list of fluxes corresponds to the transitions as listed in the LAMDA
            file.
        '''
        if transitions is None:
            transitions = self.all_transitions()
        transitions = nb.typed.List(transitions)
        general_kwargs = {'tau_nu0_lines':self.tau_nu0,'nu0_lines':self.nu0_lines,
                          'solid_angle':solid_angle,'transitions':transitions,
                          'Tex':self.Tex}
        if self.geometry_name == 'LVG sphere':
            fast_flux = self.fast_fluxes_LVG_sphere(**general_kwargs,
                                                    V_LVG_sphere=self.V_LVG_sphere)
        else:
            if self.line_profile_type == 'Gaussian':
                fast_flux = self.fast_fluxes_Gaussian(
                               **general_kwargs,
                               width_v=self.emitting_molecule.width_v,
                               compute_flux_nu=self.geometry.compute_flux_nu,
                               tau_peak_fraction=self.tau_peak_fraction,
                               nu_per_FHWM=self.nu_per_FHWM)
            elif self.line_profile_type == 'rectangular':
                fast_flux = self.fast_fluxes_rectangular(
                              **general_kwargs,compute_flux_nu=self.geometry.compute_flux_nu,
                              width_v=self.emitting_molecule.width_v,)
            else:
                raise ValueError(f'line profile {self.line_profile_type} unknown')
        return np.squeeze(fast_flux)

    def identify_lines(self,nu):
        nu_min,nu_max = np.min(nu),np.max(nu)
        selected_lines = []
        selected_line_indices = []
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            if nu_min <= line.nu0 and line.nu0 <= nu_max:
                selected_line_indices.append(i)
                selected_lines.append(line)
        return selected_lines,selected_line_indices

    def get_tau_nu_lines(self,nu):
        selected_lines,selected_line_indices = self.identify_lines(nu=nu)
        tau_nu_lines = []
        for line_index,line in zip(selected_line_indices,selected_lines):
            x1 = self.level_pop[line.low.number]
            x2 = self.level_pop[line.up.number]
            N1 = self.N*x1
            N2 = self.N*x2
            tau_nu_lines.append(line.tau_nu(N1=N1,N2=N2,nu=nu))
        return tau_nu_lines

    def tau_nu(self,nu):
        '''Calculate the optical depth profile at the requested frequencies.

        Args:
            nu (numpy.ndarray): The frequencies in [Hz].
        
        Returns:
            np.ndarray: The optical depth at the requested frequencies.
        '''
        tau_nu_lines = self.get_tau_nu_lines(nu=nu)
        return np.sum(tau_nu_lines,axis=0)

    def identify_overlapping_lines(self):
        self.overlapping_lines = []
        width_v = self.emitting_molecule.width_v
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            nu0 = line.nu0
            width_nu = width_v/constants.c*nu0
            overlapping_lines = []
            for j,overlap_line in enumerate(self.emitting_molecule.rad_transitions):
                if j == i:
                    continue
                nu0_overlap_line = overlap_line.nu0
                width_nu_overlap_line = width_v/constants.c*nu0_overlap_line
                if self.line_profile_type == 'rectangular':
                    if np.abs(nu0-nu0_overlap_line) <= width_nu/2 + width_nu_overlap_line/2:
                        overlapping_lines.append(j)
                elif self.line_profile_type == 'Gaussian':
                    #here I need to be conservative because the Gaussian profile can,
                    #in theory, be arbitrarily broad (for arbitrarily high optical depth)
                    #take +- 1.57 FWHM, where the Gaussian is 0.1% of the peak
                    if np.abs(nu0-nu0_overlap_line) <= 1.57*width_nu + 1.57*width_nu_overlap_line:
                        overlapping_lines.append(j)
            self.overlapping_lines.append(overlapping_lines)

    def any_line_has_overlap(self,line_indices):
        for index in line_indices:
            if len(self.overlapping_lines[index]) > 0:
                return True
        return False

    def spectrum(self,solid_angle,nu):
        r'''Calculate the flux at the requested frequencies.

        Args:
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
            nu (numpy.ndarray): The frequencies in [Hz].
        
        Returns:
            np.ndarray: The flux at the requested frequencies in [W/m\ :sup:`2`/Hz].
        '''
        #assumption: lines do not overlap, so I can just sum them up
        #this also works for overlapping lines in the optically thin regime
        #but this is invalid for overlapping thick lines
        selected_lines,selected_line_indices = self.identify_lines(nu=nu)
        if self.any_line_has_overlap(line_indices=selected_line_indices):
            warnings.warn('lines are overlapping, spectrum might not be correct!')
        tau_nu_lines = self.get_tau_nu_lines(nu=nu)
        spectrum = np.zeros_like(nu)
        LVG_sphere_kwargs = {}
        for line_index,line,tau_nu_line in zip(selected_line_indices,selected_lines,
                                               tau_nu_lines):
            if self.geometry_name == 'LVG sphere':
                LVG_sphere_kwargs = {'nu':nu,'nu0':line.nu0,'V':self.V_LVG_sphere}
            source_function = helpers.B_nu(T=self.Tex[line_index],nu=nu)
            spectrum += self.geometry.compute_flux_nu(
                               tau_nu=tau_nu_line,source_function=source_function,
                               solid_angle=solid_angle,**LVG_sphere_kwargs)
        return spectrum

    def model_grid(self,ext_backgrounds,N_values,Tkin_values,collider_densities_values,
                   requested_output,solid_angle=None,transitions=None,nu=None):
        r'''Iterator over a grid of models. Models are calculated for all possible combinations
            of the input parameters.

        Args:
            ext_backgrounds (:obj:`dict`): A dictionary of functions, one entry for
                each background that should be used. The keys of the dictionary are used
                in the output to identify which background was used for a certain output
            N_values (:obj:`list` or numpy.ndarray): The list of column densities to
                compute models for, in [m\ :sup:`-2`].
            Tkin_values (:obj:`list` or numpy.ndarray): The list of kinetic temperatures
                to compute models for, in [K].
            collider_densities_values (:obj:`dict`): A dictionary with one entry for each
                collider. Each entry is a list of densities for which models should be
                computed for. Units are [m\ :sup:`-3`].
            requested_output (:obj:`list`): The list of requested outputs. Possible entries
                are 'level_pop','Tex','tau_nu0','fluxes','tau_nu', and 'spectrum'
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
                Defaults to None. Compulsory if 'fluxes' or 'spectrum'
                are requested.
            transitions (:obj:`list`): The indices of the
                transitions for which to calculate Tex, tau_nu and fluxes. If None, then
                values for all transitions are calculated. Defaults to None. The
                indices are relative to the list of transitions in the LAMDA file,
                starting with 0.
            nu (numpy.ndarray): The frequencies in [Hz]. Compulsory if 'tau_nu'
                or 'spectrum' is requested.

        Returns:
            dict: dictionary with fields 'ext_background','N','Tkin' and 'collider_densities'
                to identify the input parameters of the model, as well as any requested
                output. Units of outputs: 'level_pop': no units; 'Tex': [K];
                'tau_nu0': no units;'fluxes': [W/m\ :sup:`2`]; 'tau_nu': no units;
                'spectrum': [W/m\ :sup:`2`/Hz]
        '''
        #it is expensive to update Tkin and collider densities, so those should be in
        #the outermost loops
        allowed_outputs = ('level_pop','Tex','tau_nu0','fluxes','tau_nu','spectrum')
        for request in requested_output:
            assert request in allowed_outputs,f'requested output "{request}" is invalid'
        if 'fluxes' in requested_output or 'spectrum' in requested_output:
            assert solid_angle is not None
        if 'tau_nu' in requested_output or 'spectrum' in requested_output:
            assert nu is not None
        if transitions is None:
            transitions = self.all_transitions()
        colliders = list(collider_densities_values.keys())
        ordered_collider_density_values = [collider_densities_values[coll] for coll
                                           in colliders]
        for Tkin in Tkin_values:
            for collider_density_set in itertools.product(*ordered_collider_density_values):
                collider_densities = {collider:value for collider,value in
                                      zip(colliders,collider_density_set)}
                for ext_background_name,ext_background in ext_backgrounds.items():
                    for N in N_values:
                        self.set_parameters(
                               ext_background=ext_background,N=N,Tkin=Tkin,
                               collider_densities=collider_densities)
                        self.solve_radiative_transfer()
                        output = {'ext_background':ext_background_name,'N':N,
                                  'Tkin':Tkin,'collider_densities':collider_densities}
                        if 'level_pop' in requested_output:
                            output['level_pop'] = self.level_pop
                        if 'Tex' in requested_output:
                            output['Tex'] = self.Tex[transitions]
                        if 'tau_nu0' in requested_output:
                            output['tau_nu0'] = self.tau_nu0[transitions]
                        if 'fluxes' in requested_output:
                            output['fluxes'] = self.fluxes(solid_angle=solid_angle,
                                                           transitions=transitions)
                        if 'tau_nu' in requested_output:
                            output['tau_nu'] = self.tau_nu(nu=nu)
                        if 'spectrum' in requested_output:
                            output['spectrum'] = self.spectrum(
                                                    solid_angle=solid_angle,nu=nu)
                        yield output

    def print_results(self):
        '''Prints the results from the radiative transfer computation.'''
        print('\n')
        print('  up   low      nu [GHz]    T_ex [K]      poplow         popup'\
              +'         tau_nu0')
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            rad_trans_string = '{:>4d} {:>4d} {:>14.6f} {:>10.2f} {:>14g} {:>14g} {:>14g}'
            rad_trans_format = (line.up.number,line.low.number,
                                line.nu0/constants.giga,self.Tex[i],
                                self.level_pop[line.low.number],
                                self.level_pop[line.up.number],
                                self.tau_nu0[i])
            print(rad_trans_string.format(*rad_trans_format))
        print('\n')