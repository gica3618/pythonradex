from scipy import constants
import numpy as np
from pythonradex import helpers,escape_probability,atomic_transition,flux,rate_equations
from pythonradex.molecule import EmittingMolecule
import warnings
import numba as nb
import itertools


class Cloud():

    '''
    Solving the non-LTE radiative transfer using escape probabilities.

    Attributes:

        emitting_molecule (pythonradex.molecule.EmittingMolecule): object
            containing all the information about the emitting atom or molecule
        tau_nu0_individual_transitions (numpy.ndarray): optical depth of each individual
            transition at the central frequency (contributions of dust or overlapping lines
            to the optical depth are not included)
        level_pop (numpy.ndarray): fractional population of each level
        Tex (numpy.ndarray): excitation temperature of each transition

    Note:
        The attributes tau_nu0_individual_transitions, level_pop and Tex are available
        only after solving the radiative transfer by calling solve_radiative_transfer
        
    '''
    relative_convergence = 1e-6
    min_iter = 10
    max_iter = 1000
    underrelaxation = 0.3 #RADEX uses 0.3
    min_tau_considered_for_convergence = 1e-2
    min_iter_before_ng_acceleration = 4
    ng_acceleration_interval = 4
    slow_variation_limit = 0.1
    geometries = {'uniform sphere':escape_probability.UniformSphere,
                  'uniform sphere RADEX':escape_probability.UniformSphereRADEX,
                  'uniform slab':escape_probability.UniformSlab,
                  'LVG slab':escape_probability.UniformLVGSlab,
                  'LVG sphere':escape_probability.UniformLVGSphere,
                  'LVG sphere RADEX':escape_probability.LVGSphereRADEX}
    line_profiles = {'Gaussian':atomic_transition.GaussianLineProfile,
                     'rectangular':atomic_transition.RectangularLineProfile}

    def __init__(self,datafilepath,geometry,line_profile_type,width_v,
                 iteration_mode='ALI',use_NG_acceleration=True,
                 average_over_line_profile=False,treat_line_overlap=False,
                 warn_negative_tau=True,verbose=False,test_mode=False):
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
                width_v corresponds to the total velocity width of the cloud.
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
            treat_line_overlap (:obj:`bool`): Whether to treat overlapping lines.
                Defaults to False. Can only be used in combination with static geometries
                (i.e. not with LVG)
            warn_negative_tau (:obj:`bool`): Whether the raise a warning when negative
                optical depth is encountered. Defaults to True. Setting this to False
                is useful when calculating a grid of models.
            verbose (:obj:`bool`): Whether to print additional information. Defaults to False.
            test_mode (:obj:`bool`): Enter test mode. Only for developer, should not be used
                by general user. Defaults to False.
        
        '''
        assert width_v < 10000*constants.kilo,\
                      'assumption of small nu0/Delta_nu for flux calculation not satisfied'
        self.emitting_molecule = EmittingMolecule(
                                    datafilepath=datafilepath,
                                    line_profile_type=line_profile_type,
                                    width_v=width_v)
        self.average_over_line_profile = average_over_line_profile
        self.treat_line_overlap = treat_line_overlap
        self.geometry_name = geometry
        if self.treat_line_overlap:
            if not self.average_over_line_profile:
                raise ValueError('treating overlapping lines requires'
                                 ' line profile averaging')
            if 'LVG' in self.geometry_name:
                #this is because in LVG, the assumption is that once the radiation
                #escapes from a local slab, it also escapes the cloud. With overlapping
                #lines this assumption breaks down
                raise ValueError('treatment of overlapping lines currently not'
                                 +' supported for LVG geometries')
        if self.emitting_molecule.has_overlapping_lines and not self.treat_line_overlap:
            warnings.warn('some lines are overlapping, but treatement of'
                          +' overlapping lines not activated')
        if verbose:
            for i in range(len(self.emitting_molecule.overlapping_lines)):
                if len(self.emitting_molecule.overlapping_lines[i]) > 0:
                    print(f'identified overlaps for line {i}:'
                          +f' {self.emitting_molecule.overlapping_lines[i]}')
        if 'LVG' in self.geometry_name and not test_mode:
            if not self.emitting_molecule.line_profile_type == 'rectangular':
                #this is to get the right optical depth which is independent of
                #velocity in this model (e.g. Eq. 13 in de Jong et al. 1975)
                raise ValueError(f'{self.geometry_name} requires a rectangular profile')
        #velocity at the surface of a LVG sphere
        self.V_LVG_sphere = self.emitting_molecule.width_v/2
        self.geometry = self.geometries[geometry]()
        self.iteration_mode = iteration_mode
        self.use_NG_acceleration = use_NG_acceleration
        self.warn_negative_tau = warn_negative_tau
        self.verbose = verbose
        self.Einstein_kwargs = {'B12':self.emitting_molecule.B12,
                                'B21':self.emitting_molecule.B21,
                                'A21':self.emitting_molecule.A21}

    @staticmethod
    def zero_field(nu):
        return np.zeros_like(nu)

    def set_parameters(self,ext_background,N,Tkin,collider_densities,T_dust=None,
                       tau_dust=None):
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
            T_dust (func): The dust temperature in [K] as a function of frequency.
                 It is assumed that the source function of the dust is a black body
                 at temperature T_dust. Defaults to None (i.e. no internal dust
                 radiation field). Can only be used with static geometries (i.e.
                 not with LVG geometries).
            tau_dust (func): optical depth of the dust as a function of frequency.
                Defaults to None (i.e. no internal dust radiation field). Can
                only be used with static geometries (i.e. not with LVG geometries).
        '''
        #why do I put this into a seperate method rather than __init__? The reason
        #is that the stuff in __init__ is expensive (in particular reading the
        #LAMDA file). So if I explore e.g. a grid of N, the stuff in __init__
        #should only be done once to save computation time. Then this method
        #can be called inside the loop over e.g. the N values
        #note that setting up the rate equations is expensive, so I only do it if
        #necessary
        for collider in collider_densities.keys():
            if collider not in self.emitting_molecule.ordered_colliders:
                raise ValueError(f'no data for collider "{collider}" available')
        self.ext_background = ext_background
        self.N = N
        if (T_dust is None and tau_dust is not None) or\
                (T_dust is not None and tau_dust is None):
            raise ValueError('both T_dust and tau_dust are needed to specify the'
                             +' dust radiation field')
        if 'LVG' in self.geometry_name:
            if not (T_dust is None and tau_dust is None):
                #this is because for LVG, it is assumed that radiation escaping
                #the local slab will escape the entire cloud, which is not true
                #if there is dust
                raise ValueError('including dust continuum is currently not'
                                 +' supported for LVG geometries')
        if T_dust is None:
            self.T_dust = self.zero_field
        else:
            self.T_dust = T_dust
        if tau_dust is None:
            self.tau_dust = self.zero_field
        else:
            self.tau_dust = tau_dust
        for func_name,func in {'external background':self.ext_background,
                               'T_dust':self.T_dust,'tau_dust':self.tau_dust}.items():
            if not self.is_slowly_varying_over_linewidth(func)\
                                      and not self.average_over_line_profile:
                warnings.warn(f'{func_name} is significantly varying over'
                                 +' line profile. Please activate averaging over'
                                 +' line profile.')
        #needed for the compiled functions:
        self.I_ext_nu0 = np.array([self.ext_background(nu0) for nu0 in
                                   self.emitting_molecule.nu0])
        self.tau_dust_nu0 = np.array([self.tau_dust(nu0) for nu0 in
                                      self.emitting_molecule.nu0])
        self.S_dust_nu0 = np.array([self.S_dust(nu=nu0) for nu0 in
                                    self.emitting_molecule.nu0])
        if (not hasattr(self,'Tkin')) or (not hasattr(self,'collider_densities')):
            Tkin_or_collider_density_changed = True
        else:
            Tkin_or_collider_density_changed\
                    = (self.Tkin != Tkin) or (self.collider_densities != collider_densities)
        if Tkin_or_collider_density_changed:
            self.Tkin = Tkin
            self.collider_densities = collider_densities
            self.rate_equations = rate_equations.RateEquations(
                                      molecule=self.emitting_molecule,
                                      collider_densities=self.collider_densities,
                                      Tkin=self.Tkin,mode=self.iteration_mode)

    def S_dust(self,nu):
        T = np.atleast_1d(self.T_dust(nu))
        return np.squeeze(helpers.B_nu(nu=nu,T=T))

    def is_slowly_varying_over_linewidth(self,func):
        nu0 = self.emitting_molecule.nu0
        Delta_nu = self.emitting_molecule.width_v/constants.c*nu0
        func_nu0 = np.array((func(nu0),))
        func_nu0_plus_Deltanu = np.array((func(nu0+Delta_nu),))
        relative_diff = helpers.relative_difference(func_nu0,func_nu0_plus_Deltanu)
        if np.any(relative_diff>self.slow_variation_limit):
            return False
        else:
            return True

    ############  preparing quantities used to solve the rate equations  ###############
    # We need to consider three cases:
    #1. no averaging over line profile (i.e. evaluate everything at nu0);
    #this cannot be used for overlapping lines, but dust is ok
    #2. with averaging, no treatment of overlapping lines
    #3. with averaging, with treatement of overlapping lines
    #all three cases need to include dust continuum

    #Case 1: no average, everything evaluated at nu0; can assume that treatment
    #of overlapping lines is not requested (if requested, averaging is enforced)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_tau_nu0_onlyline(
                       level_population,nlow_rad_transitions,nup_rad_transitions,N,
                       A21,phi_nu0,gup_rad_transitions,glow_rad_transitions,nu0):
        n_lines = nlow_rad_transitions.size
        tau_nu0 = np.empty(n_lines)
        for i in range(n_lines):
            N1 = N * level_population[nlow_rad_transitions[i]]
            N2 = N * level_population[nup_rad_transitions[i]]
            tau_nu0[i] = atomic_transition.fast_tau_nu(
                                 A21=A21[i],phi_nu=phi_nu0[i],g_up=gup_rad_transitions[i],
                                 g_low=glow_rad_transitions[i],N1=N1,N2=N2,nu=nu0[i])
        return tau_nu0

    def tau_nu0_onlyline(self,level_population):
        return self.fast_tau_nu0_onlyline(
                           level_population=level_population,
                           nlow_rad_transitions=self.emitting_molecule.nlow_rad_transitions,
                           nup_rad_transitions=self.emitting_molecule.nup_rad_transitions,
                           N=self.N,A21=self.emitting_molecule.A21,
                           phi_nu0=self.emitting_molecule.phi_nu0,
                           gup_rad_transitions=self.emitting_molecule.gup_rad_transitions,
                           glow_rad_transitions=self.emitting_molecule.glow_rad_transitions,
                           nu0=self.emitting_molecule.nu0)

    def tau_nu0_including_dust(self,level_population):
        tau_nu0_line = self.tau_nu0_onlyline(level_population=level_population)
        return tau_nu0_line + self.tau_dust_nu0

    # @staticmethod
    # @nb.jit(nopython=True,cache=True)
    # def fast_beta_nu0_without_overlap(
    #                    level_population,nlow_rad_transitions,nup_rad_transitions,N,
    #                    A21,phi_nu0,gup_rad_transitions,glow_rad_transitions,nu0,
    #                    beta_function,tau_dust_nu0):
    #     #compute beta at nu0
    #     n_lines = nlow_rad_transitions.size
    #     tau_nu0 = np.empty(n_lines)
    #     for i in range(n_lines):
    #         N1 = N * level_population[nlow_rad_transitions[i]]
    #         N2 = N * level_population[nup_rad_transitions[i]]
    #         tau_nu0[i] = atomic_transition.fast_tau_nu(
    #                              A21=A21[i],phi_nu=phi_nu0[i],
    #                              g_up=gup_rad_transitions[i],g_low=glow_rad_transitions[i],N1=N1,
    #                              N2=N2,nu=nu0[i])
    #         tau_nu0[i] += tau_dust_nu0[i]
    #     return beta_function(tau_nu0)

    # def beta_alllines_average(self,level_population):
    #     return self.fast_averaged_beta(
    #                     level_population=level_population,
    #                     nlow_rad_transitions=self.nlow_rad_transitions,
    #                     nup_rad_transitions=self.nup_rad_transitions,N=self.N,
    #                     A21=self.A21,phi_nu=self.phi_nu,
    #                     gup_rad_transitions=self.gup_rad_transitions,glow_rad_transitions=self.glow_rad_transitions,
    #                     nu=self.coarse_nu,beta_function=self.geometry.beta)

    def beta_nu0(self,level_population):
        tau_nu0 = self.tau_nu0_including_dust(level_population=level_population)
        return self.geometry.beta(tau_nu0)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_S_nu0(level_population,nlow_rad_transitions,nup_rad_transitions,A21,
                   B21,B12,tau_dust_nu0,S_dust_nu0,tau_nu0_onlyline):
        n_lines = nlow_rad_transitions.size
        source_func = np.zeros(n_lines)
        for i in range(n_lines):
            x1 = level_population[nlow_rad_transitions[i]]
            x2 = level_population[nup_rad_transitions[i]]
            if x1 == x2 == 0:
                source_func[i] = S_dust_nu0[i]
            else:
                S_line = A21[i]*x2/(x1*B12[i]-x2*B21[i])
                tau_line = tau_nu0_onlyline[i]
                tau_dust = tau_dust_nu0[i]
                tau_tot = tau_line + tau_dust
                source_func[i] = (S_line*tau_line+S_dust_nu0[i]*tau_dust)\
                                    /tau_tot
        return source_func

    def S_nu0(self,level_population):
        tau_nu0_onlyline = self.tau_nu0_onlyline(level_population=level_population)
        return self.fast_S_nu0(
                 level_population=level_population,
                 nlow_rad_transitions=self.emitting_molecule.nlow_rad_transitions,
                 nup_rad_transitions=self.emitting_molecule.nup_rad_transitions,
                 A21=self.emitting_molecule.A21,B21=self.emitting_molecule.B21,
                 B12=self.emitting_molecule.B12,tau_dust_nu0=self.tau_dust_nu0,
                 S_dust_nu0=self.S_dust_nu0,tau_nu0_onlyline=tau_nu0_onlyline)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Jbar_nu0(beta_nu0,I_ext_nu0,S_nu0):
        return beta_nu0*I_ext_nu0 + (1-beta_nu0)*S_nu0

    def Jbar_nu0(self,level_population):
        beta_nu0 = self.beta_nu0(level_population=level_population)
        S_nu0 = self.S_nu0(level_population=level_population)
        return self.fast_Jbar_nu0(beta_nu0=beta_nu0,I_ext_nu0=self.I_ext_nu0,
                                  S_nu0=S_nu0)

    def A21_factor_nu0(self,level_population):
        beta_nu0 = self.beta_nu0(level_population=level_population)
        tau_nu0_onlyline = self.tau_nu0_onlyline(level_population=level_population)
        tau_tot_nu0 = self.tau_nu0_including_dust(level_population=level_population)
        return np.where(tau_tot_nu0!=0,1-(1-beta_nu0)*tau_nu0_onlyline/tau_tot_nu0,1)

    def B21_factor_nu0(self,level_population):
        beta_nu0 = self.beta_nu0(level_population=level_population)
        tau_tot_nu0 = self.tau_nu0_including_dust(level_population=level_population)
        K_nu0 = np.where(tau_tot_nu0!=0,self.tau_dust_nu0*self.S_dust_nu0/tau_tot_nu0,0)
        return beta_nu0*self.I_ext_nu0 + (1-beta_nu0)*K_nu0        

    #Case 2: with averaging, without line overlap treatment
    #and
    #Case 3: with averaging, with line overlap treatment
    #for these cases, using numba gets too complicated, so I don't use it for now...

    def line_iterator(self,level_population):
        for line_index,line in enumerate(self.emitting_molecule.rad_transitions):
            x1 = level_population[line.low.number]
            x2 = level_population[line.up.number]
            N1 = x1*self.N
            N2 = x2*self.N
            yield line_index,line,x1,x2,N1,N2

    def Jbar_averaged(self,level_population):
        Jbar = np.empty(self.emitting_molecule.n_rad_transitions)
        for line_index,line,x1,x2,N1,N2 in self.line_iterator(
                                              level_population=level_population):
            if self.treat_line_overlap:
                tau_func = self.emitting_molecule.get_tau_tot_nu(
                              line_index=line_index,level_population=level_population,
                              N=self.N,tau_dust=self.tau_dust)
                source_func = self.emitting_molecule.get_total_S_nu(
                               line_index=line_index,level_population=level_population,
                               N=self.N,tau_dust=self.tau_dust,S_dust=self.S_dust)
            else:
                def tau_func(nu):
                    return line.tau_nu(N1=N1,N2=N2,nu=nu) + self.tau_dust(nu)
                def source_func(nu):
                    S_line = line.source_function(x1=x1,x2=x2)
                    tau_line = line.tau_nu(N1=N1,N2=N2,nu=nu)
                    S_dust = self.S_dust(nu)
                    tau_dust = self.tau_dust(nu)
                    tau_line_dust = tau_line+tau_dust
                    return np.where(tau_line_dust!=0,
                                    (S_line*tau_line+S_dust*tau_dust)/tau_line_dust,0)
            def beta_func(nu):
                return self.geometry.beta(tau_func(nu))
            def J(nu):
                beta = beta_func(nu)
                Iext = self.ext_background(nu)
                S = source_func(nu)
                return beta*Iext + (1-beta)*S
            Jbar[line_index] = line.line_profile.average_over_phi_nu(J)
        return Jbar

    def get_tau_line_and_dust(self,line,N1,N2):
        def tau_line_and_dust(nu):
            return line.tau_nu(N1=N1,N2=N2,nu=nu) + self.tau_dust(nu)
        return tau_line_and_dust

    def A21_factor_averaged(self,level_population):
        A21_factor = np.empty(self.emitting_molecule.n_rad_transitions)
        for line_index,line,x1,x2,N1,N2 in self.line_iterator(
                                               level_population=level_population):
            if self.treat_line_overlap:
                #dust plus all lines
                tau_tot_func = self.emitting_molecule.get_tau_tot_nu(
                                 line_index=line_index,level_population=level_population,
                                 N=self.N,tau_dust=self.tau_dust)
            else:
                #only the line under consideration plus dust
                tau_tot_func = self.get_tau_line_and_dust(line=line,N1=N1,N2=N2)
            def A21_factor_func(nu):
                tau_line = line.tau_nu(N1=N1,N2=N2,nu=nu)
                tau_tot = tau_tot_func(nu)
                beta = self.geometry.beta(tau_tot)
                return np.where(tau_tot>0,1-(1-beta)*tau_line/tau_tot,1)
            A21_factor[line_index] = line.line_profile.average_over_phi_nu(A21_factor_func)
        return A21_factor

    def B21_factor_averaged(self,level_population):
        B21_factor = np.empty(self.emitting_molecule.n_rad_transitions)
        for line_index,line,x1,x2,N1,N2 in self.line_iterator(
                                               level_population=level_population):
            if self.treat_line_overlap:
                tau_tot_func = self.emitting_molecule.get_tau_tot_nu(
                                 line_index=line_index,level_population=level_population,
                                 N=self.N,tau_dust=self.tau_dust)
                K_func = self.emitting_molecule.get_K_nu(
                                line_index=line_index,level_population=level_population,
                                N=self.N,tau_dust=self.tau_dust,S_dust=self.S_dust)
            else:
                tau_tot_func = self.get_tau_line_and_dust(line=line,N1=N1,N2=N2)
                def K_func(nu):
                    tau_tot = tau_tot_func(nu)
                    return np.where(tau_tot!=0,self.tau_dust(nu)*self.S_dust(nu)/tau_tot,0)
            def B21_factor_func(nu):
                tau_tot = tau_tot_func(nu)
                beta = self.geometry.beta(tau_tot)
                Iext = self.ext_background(nu)
                return beta*Iext + (1-beta)*K_func(nu)
            B21_factor[line_index] = line.line_profile.average_over_phi_nu(B21_factor_func)
        return B21_factor

    def get_initial_level_pop(self):
        #assume optically thin emission for the first iteration
        ones = np.ones(self.emitting_molecule.n_rad_transitions)
        beta = ones
        if self.rate_equations.mode == 'ALI':
            A21_factor = ones
            B21_factor = beta*self.I_ext_nu0
            return self.rate_equations.solve(A21_factor=A21_factor,
                                             B21_factor=B21_factor,
                                             **self.Einstein_kwargs)
        elif self.rate_equations.mode == 'LI':
            Jbar = beta*self.I_ext_nu0
            return self.rate_equations.solve(Jbar=Jbar,**self.Einstein_kwargs)
        else:
            raise ValueError(f'unknown mode "{self.rate_equations.mode}"')

    def get_new_level_pop_with_average(self,old_level_pop):
        #this functions treats the cases with and without line overlap treatment
        if self.rate_equations.mode == 'ALI':
            A21_factor = self.A21_factor_averaged(
                                        level_population=old_level_pop)
            B21_factor = self.B21_factor_averaged(
                                     level_population=old_level_pop)
            return self.rate_equations.solve(
                                     A21_factor=A21_factor,B21_factor=B21_factor,
                                     **self.Einstein_kwargs)
        elif self.rate_equations.mode == 'LI':
            Jbar = self.Jbar_averaged(level_population=old_level_pop)
            return self.rate_equations.solve(Jbar=Jbar,**self.Einstein_kwargs)
        else:
            raise ValueError(f'unknown mode "{self.rate_equations.mode}"')

    def get_new_level_pop_without_average(self,old_level_pop):
        assert not self.treat_line_overlap, 'line overlap treatment requires averaging'
        if self.rate_equations.mode == 'ALI':
            A21_factor = self.A21_factor_nu0(level_population=old_level_pop)
            B21_factor = self.B21_factor_nu0(level_population=old_level_pop)
            return self.rate_equations.solve(A21_factor=A21_factor,
                                             B21_factor=B21_factor,
                                             **self.Einstein_kwargs)
        elif self.rate_equations.mode == 'LI':
            Jbar = self.Jbar_nu0(level_population=old_level_pop)
            return self.rate_equations.solve(Jbar=Jbar,**self.Einstein_kwargs)
        else:
            raise ValueError(f'unknown mode "{self.rate_equations.mode}"')

    def get_new_level_pop(self,old_level_pop):
        if self.average_over_line_profile:
            return self.get_new_level_pop_with_average(
                                            old_level_pop=old_level_pop)
        else:
            return self.get_new_level_pop_without_average(
                                               old_level_pop=old_level_pop)

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
    def compute_residual(Tex_residual,tau,min_tau_considered_for_convergence):
        #same as RADEX: consider only lines above a minimum tau for convergence
        selection = tau > min_tau_considered_for_convergence
        n_selected = selection.sum()
        if n_selected > 0:
            return np.sum(Tex_residual[selection]) / n_selected
        else:
            return 0

    def solve_radiative_transfer(self):
        """Solves the radiative transfer."""
        level_pop = self.get_initial_level_pop()
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
            tau_nu0 = self.emitting_molecule.get_tau_nu0(
                                       N=self.N,level_population=new_level_pop)
            residual = self.compute_residual(
                 Tex_residual=Tex_residual,tau=tau_nu0,
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
        self.n_iter_convergence = counter
        #tau_nu0 is the optical depth of the individual lines, not the total optical depth
        #(i.e. we do not consider the contribution of overlapping lines or dust)
        self.tau_nu0_individual_transitions = self.emitting_molecule.get_tau_nu0(
                                               N=self.N,level_population=level_pop)
        if self.warn_negative_tau:
            if np.any(self.tau_nu0_individual_transitions < 0):
                negative_tau_transition_indices = np.where(self.tau_nu0_individual_transitions < 0)[0]
                negative_tau_transitions = [self.emitting_molecule.rad_transitions[i]
                                            for i in negative_tau_transition_indices]
                warnings.warn('negative optical depth!')
                for i,trans in zip(negative_tau_transition_indices,
                                   negative_tau_transitions):
                    print(f'{trans.name}: tau_nu0 = {self.tau_nu0_individual_transitions[i]:.3g}')
        self.level_pop = level_pop
        self.Tex = self.emitting_molecule.get_Tex(level_pop)
        self.flux_calculator = flux.FluxCalculator(
                                 emitting_molecule=self.emitting_molecule,
                                 level_population=self.level_pop,
                                 geometry_name=self.geometry_name,
                                 V_LVG_sphere=self.V_LVG_sphere,
                                 compute_flux_nu=self.geometry.compute_flux_nu,
                                 tau_nu0_individual_transitions=self.tau_nu0_individual_transitions,
                                 tau_dust=self.tau_dust,S_dust=self.S_dust)

    def fluxes_of_individual_transitions(self,solid_angle,transitions):
        #TODO add documentation; in particular, these are fluxes of individual lines,
        #ignoring any overlapping lines or dust?
        return self.flux_calculator.fluxes_of_individual_transitions(
                         solid_angle=solid_angle,transitions=transitions)

    def tau_nu(self,nu):
        #TODO add documentation
        self.flux_calculator.set_nu(nu=nu)
        return self.flux_calculator.tau_nu_tot

    def spectrum(self,solid_angle,nu):
        #TODO add documentation
        self.flux_calculator.set_nu(nu=nu)
        return self.flux_calculator.spectrum(solid_angle=solid_angle)

    def model_grid(self,ext_backgrounds,N_values,Tkin_values,collider_densities_values,
                   requested_output,T_dust=None,tau_dust=None,solid_angle=None,
                   transitions=None,nu=None):
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
                are 'level_pop','Tex','tau_nu0_individual_transitions','fluxes_of_individual_transitions','tau_nu',
                and 'spectrum'
            T_dust (func): The dust temperature in [K] as a function of frequency.
                 It is assumed that the source function of the dust is a black body
                 at temperature T_dust. Defaults to None (i.e. no internal dust
                 radiation field).
            tau_dust (func): optical depth of the dust as a function of frequency.
                Defaults to None (i.e. no internal dust radiation field).
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
                Defaults to None. Compulsory if 'fluxes_of_individual_transitions' or 'spectrum'
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
                'tau_nu0': no units;'fluxes_of_individual_transitions': [W/m\ :sup:`2`]; 'tau_nu': no units;
                'spectrum': [W/m\ :sup:`2`/Hz]
        '''
        #it is expensive to update Tkin and collider densities, so those should be in
        #the outermost loops
        allowed_outputs = ('level_pop','Tex','tau_nu0_individual_transitions',
                           'fluxes_of_individual_transitions','tau_nu','spectrum')
        for request in requested_output:
            assert request in allowed_outputs,f'requested output "{request}" is invalid'
        if 'fluxes_of_individual_transitions' in requested_output\
                                      or 'spectrum' in requested_output:
            assert solid_angle is not None
        if 'tau_nu' in requested_output or 'spectrum' in requested_output:
            assert nu is not None
        if transitions is None:
            transitions = [i for i in range(self.emitting_molecule.n_rad_transitions)]
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
                               collider_densities=collider_densities,T_dust=T_dust,
                               tau_dust=tau_dust)
                        self.solve_radiative_transfer()
                        output = {'ext_background':ext_background_name,'N':N,
                                  'Tkin':Tkin,'collider_densities':collider_densities}
                        if 'level_pop' in requested_output:
                            output['level_pop'] = self.level_pop
                        if 'Tex' in requested_output:
                            output['Tex'] = self.Tex[transitions]
                        if 'tau_nu0_individual_transitions' in requested_output:
                            output['tau_nu0_individual_transitions']\
                                       = self.tau_nu0_individual_transitions[transitions]
                        if 'fluxes_of_individual_transitions' in requested_output:
                            output['fluxes_of_individual_transitions']\
                                   = self.fluxes_of_individual_transitions(
                                         solid_angle=solid_angle,transitions=transitions)
                        if 'tau_nu' in requested_output:
                            output['tau_nu'] = self.tau_nu(nu=nu)
                        if 'spectrum' in requested_output:
                            output['spectrum'] = self.spectrum(
                                                    solid_angle=solid_angle,nu=nu)
                        yield output

    def print_results(self):
        '''Prints the results from the radiative transfer computation.'''
        print('\n')
        print('  up   low      nu0 [GHz]    T_ex [K]      poplow         popup'\
              +'         tau_nu0')
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            output = f'{line.up.number:>4d} {line.low.number:>4d} '\
                     +f'{line.nu0/constants.giga:>14.6f} {self.Tex[i]:>10.2f} '\
                     +f'{self.level_pop[line.low.number]:>14g} '\
                     +f'{self.level_pop[line.up.number]:>14g} '\
                     +f'{self.tau_nu0_individual_transitions[i]:>14g}'
            print(output)
        print('\n')