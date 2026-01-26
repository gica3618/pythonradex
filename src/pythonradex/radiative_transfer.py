from scipy import constants
import numpy as np
from pythonradex import helpers,escape_probability,atomic_transition,intensity,rate_equations
from pythonradex.molecule import EmittingMolecule
import warnings
import numba as nb
import numbers


class Source():

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
        lower_level_population (numpy.ndarray): for each transition, the fractional
            population of the lower level of that transition
        upper_level_population (numpy.ndarray): for each transition, the fractional
            population of the upper level of that transition

    Note:
        The attributes tau_nu0_individual_transitions, level_pop, Tex,
        lower_level_population and upper_level_population are available
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
    geometries = {'static sphere':escape_probability.StaticSphere,
                  'static sphere RADEX':escape_probability.StaticSphereRADEX,
                  'static slab':escape_probability.StaticSlab,
                  'LVG slab':escape_probability.LVGSlab,
                  'LVG sphere':escape_probability.LVGSphere,
                  'LVG sphere RADEX':escape_probability.LVGSphereRADEX}
    line_profiles = {'Gaussian':atomic_transition.GaussianLineProfile,
                     'rectangular':atomic_transition.RectangularLineProfile}

    def __init__(self,datafilepath,geometry,line_profile_type,width_v,
                 use_Ng_acceleration=True,treat_line_overlap=False,
                 warn_negative_tau=True,verbose=False,test_mode=False):
        '''Initialises a new instance of the Source class.

        Args:
            datafilepath (:obj:`str`): filepath to the file in LAMDA format that contains
                the atomic / molecular data
            geometry (:obj:`str`): determines how the escape probability
                and flux are calculated. Available options: "static sphere", 
                "static sphere RADEX", "static slab", "LVG slab", "LVG sphere" and
                "LVG sphere RADEX". The options containing "RADEX" are meant to
                mimic the behaviour of the original RADEX code by using the same
                equations for the escape probability and the (non-background-subtracted)
                intensity as RADEX.
            line_profile_type (:obj:`str`): The type of the line profile.
                Available options: "rectangular" or "Gaussian". Note that for
                LVG geometries, only "rectangular" is allowed.
            width_v (:obj:`float`): The width of the line profile in [m/s]. For a Gaussian
                profile, this is interpreted as the FWHM. Note that the interpretation
                of this parameter depends on the adopted geometry. For the static
                geometries, width_v is the width of the intrinsic emission profile.
                On the other hand, for LVG geometries (for which the line profile
                is rectangular), width_v corresponds to the total velocity width
                of the source. So, for "LVG sphere", width_v=2*V, where V is the
                velocity at the surface of the sphere. In terms of the constant
                velocity gradient dv/dr=V/R (with R the radius of the sphere),
                we can also say that width_v=dv/dr*2*R. For "LVG slab", width_v=dv/dz*Z
                where Z is the depth of the slab and dv/dz the constant velocity
                gradient of the slab.
            use_Ng_acceleration (:obj:`bool`): Whether to use Ng acceleration. Defaults
                to True.
            treat_line_overlap (:obj:`bool`): Whether to treat the overlap of emission
                lines by averaging over the line profile. If False, all calculations
                are done at the rest frequency (same as RADEX), and overlap effects
                are neglected. If True, overlapping lines are treated correctly by
                performing averages over frequency. This slows down the calculation 
                considerably. Defaults to False. Can only be used in combination
                with static geometries (i.e. not with LVG geometries).
            warn_negative_tau (:obj:`bool`): Whether the raise a warning when negative
                optical depth is encountered. Defaults to True. Setting this to False
                may be useful when calculating a grid of models.
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
        self.treat_line_overlap = treat_line_overlap
        self.geometry_name = geometry
        if self.treat_line_overlap:
            if 'LVG' in self.geometry_name:
                #this is because in LVG, the assumption is that once the radiation
                #escapes from a local slab, it also escapes the source. With overlapping
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
        self.use_Ng_acceleration = use_Ng_acceleration
        self.warn_negative_tau = warn_negative_tau
        self.verbose = verbose
        self.Einstein_kwargs = {'B12':self.emitting_molecule.B12,
                                'B21':self.emitting_molecule.B21,
                                'A21':self.emitting_molecule.A21}

    def check_parameters(self,collider_densities,T_dust,tau_dust,ext_background):
        if collider_densities is not None:
            available_colliders = self.emitting_molecule.coll_transitions.keys()
            for collider in collider_densities.keys():
                if collider not in available_colliders:
                    raise ValueError(f'collider "{collider}" not available'
                                    +' in the LAMDA file (available'
                                    +f' colliders: {list(available_colliders)})')
        if 'LVG' in self.geometry_name:
            if not (T_dust in (None,0) and tau_dust in (None,0)):
                #this is because for LVG, it is assumed that radiation escaping
                #the local slab will escape the entire source, which is not true
                #if there is dust
                raise ValueError('including dust continuum is currently not'
                                 +' supported for LVG geometries')

    @staticmethod
    def should_be_updated(new_func_or_number,old):
        #if it is a number, I can decide whether to update or not; if it is a function,
        #I always update
        if new_func_or_number is None:
            return False
        if isinstance(new_func_or_number,numbers.Number):
            return new_func_or_number != old
        else:
            #if it's not a number and not None, then always update
            return True

    def update_parameters(self,N=None,Tkin=None,collider_densities=None,
                          ext_background=None,T_dust=None,tau_dust=None):
        r'''Set or update the parameters for a radiative transfer calculation.
            Any of the parameters can be set to None if no update of that
            parameter is wished.

        Args:
            ext_background (func, number or None): A function taking the
                frequency in Hz as input and returning the background radiation
                field in [W/m\ :sup:`2`/Hz/sr]. A single number is interpreted as
                a constant value for all frequencies. Defaults to None
                (i.e. do not update).
            N (:obj:`float` or None): The column density in [m\ :sup:`-2`].
                Defaults to None (i.e. do not update).
            Tkin (:obj:`float` or None): The kinetic temperature of the gas in [K].
                Defaults to None (i.e. do not update).
            collider_densities (dict or None): A dictionary of the number densities of
                each collider that should be considered, in units of [m\ :sup:`-3`]. The
                following keys are recognised: "H2", "para-H2", "ortho-H2", "e",
                "H", "He", "H+". Defaults to None (i.e. do not update).
            T_dust (func, number or None): The dust temperature in [K] as a
                function of frequency. It is assumed that the source function
                of the dust is a black body at temperature T_dust. A single number
                is interpreted as a constant temperature for all frequencies.
                Can only be used with static geometries (i.e. not with LVG geometries).
                Defaults to None (i.e. do not update). For a model without dust,
                put this parameter to 0.
            tau_dust (func, number or None): optical depth of the dust as a function of
                frequency. A single number is interpreted as a constant optical
                depth for all frequencies. Can only be used with static geometries
                (i.e. not with LVG geometries). Defaults to None (i.e. do not update).
                For a model without dust, put this parameter to 0.
                
        '''
        #why do I put this into a seperate method rather than __init__? The reason
        #is that the stuff in __init__ is expensive (in particular reading the
        #LAMDA file). So if I explore e.g. a grid of parameters, the stuff in __init__
        #should only be done once to save computation time. Then this method
        #can be called inside the loop over e.g. the N values
        #note that setting up the rate equations is expensive, so I only do it if
        #necessary
        #unfortunately a fairly complicated function, couldn't find any easier way...
        self.check_parameters(collider_densities=collider_densities,T_dust=T_dust,
                              tau_dust=tau_dust,ext_background=ext_background)
        if not hasattr(self,'rate_equations'):
            params = {'N':N,'Tkin':Tkin,'collider_densities':collider_densities,
                      'ext_background':ext_background,'T_dust':T_dust,
                      'tau_dust':tau_dust}
            for p_name,p in params.items():
                assert p is not None,\
                     f'for initial setup, all params need to be defined; {p_name} missing'
            self.rate_equations = rate_equations.RateEquations(
                                 molecule=self.emitting_molecule,
                                 treat_line_overlap=self.treat_line_overlap,
                                 geometry=self.geometry,**params)
        else:
            if N is not None:
                #updating N should be very cheap, so I don't care if I really need
                #to update or not...
                self.rate_equations.set_N(N)
            update_Tkin = self.should_be_updated(
                                 new_func_or_number=Tkin,old=self.rate_equations.Tkin)
            update_coll_densities = self.should_be_updated(
                                        new_func_or_number=collider_densities,
                                        old=self.rate_equations.collider_densities)
            if update_Tkin or update_coll_densities:
                Tkin_updated = Tkin if update_Tkin else self.rate_equations.Tkin
                coll_dens_updated = collider_densities if update_coll_densities else\
                                      self.rate_equations.collider_densities
                self.rate_equations.set_collision_rates(
                               Tkin=Tkin_updated,collider_densities=coll_dens_updated)
            if self.should_be_updated(new_func_or_number=ext_background,
                                      old=self.rate_equations.ext_background):
                self.rate_equations.set_ext_background(ext_background=ext_background)
            update_T_dust = self.should_be_updated(
                                     new_func_or_number=T_dust,old=self.rate_equations.T_dust)
            update_tau_dust = self.should_be_updated(
                                               new_func_or_number=tau_dust,
                                               old=self.rate_equations.tau_dust)
            if update_T_dust or update_tau_dust:
                Td = T_dust if update_T_dust else self.rate_equations.T_dust
                taud = tau_dust if update_tau_dust else self.rate_equations.tau_dust
                self.rate_equations.set_dust(T_dust=Td,tau_dust=taud)

    def get_initial_level_pop(self):
        #assume LTE for the first iteration
        Boltzmann_level_population\
             = self.emitting_molecule.Boltzmann_level_population(T=self.rate_equations.Tkin)
        return self.rate_equations.solve(level_population=Boltzmann_level_population)        

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def compute_residual(Tex_residual,tau,min_tau):
        #same as RADEX: consider only lines above a minimum tau for convergence
        selection = tau > min_tau
        n_selected = selection.sum()
        if n_selected > 0:
            return np.sum(Tex_residual[selection]) / n_selected
        else:
            return 0

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

    def solve_radiative_transfer(self):
        """Solves the radiative transfer."""
        level_pop = self.get_initial_level_pop()
        #typed list because this will be used by compiled function ng_accelerate:
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
            new_level_pop = self.rate_equations.solve(level_population=level_pop)
            Tex = self.emitting_molecule.get_Tex(new_level_pop)
            Tex_residual = helpers.relative_difference(Tex,old_Tex)
            if self.verbose:
                print(f'max relative Tex residual: {np.max(Tex_residual):.3g}')
            tau_nu0 = self.emitting_molecule.get_tau_nu0_lines(
                         N=self.rate_equations.N,level_population=new_level_pop)
            residual = self.compute_residual(
                             Tex_residual=Tex_residual,tau=tau_nu0,
                             min_tau=self.min_tau_considered_for_convergence)
            old_Tex = Tex.copy()
            old_level_pops.append(level_pop.copy())
            if len(old_level_pops) > 3:
                old_level_pops.pop(0)
            level_pop = self.underrelaxation*new_level_pop\
                                    + (1-self.underrelaxation)*level_pop
            if self.use_Ng_acceleration\
                         and counter > self.min_iter_before_ng_acceleration\
                         and counter%self.ng_acceleration_interval == 0:
                level_pop = self.ng_accelerate(level_pop=level_pop,
                                               old_level_pops=old_level_pops)
        if self.verbose:
            print(f'converged in {counter} iterations')
        self.n_iter_convergence = counter
        self.tau_nu0_individual_transitions = self.emitting_molecule.get_tau_nu0_lines(
                                               N=self.rate_equations.N,level_population=level_pop)
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
        self.intensity_calculator = intensity.IntensityCalculator(
                                 emitting_molecule=self.emitting_molecule,
                                 level_population=self.level_pop,
                                 geometry_name=self.geometry_name,
                                 V_LVG_sphere=self.V_LVG_sphere,
                                 specific_intensity=self.geometry.specific_intensity,
                                 tau_nu0_individual_transitions=self.tau_nu0_individual_transitions,
                                 tau_dust=self.rate_equations.tau_dust,
                                 S_dust=self.rate_equations.S_dust)
        lower_level_population = [self.level_pop[t.low.index] for t in
                                  self.emitting_molecule.rad_transitions]
        self.lower_level_population = np.array(lower_level_population)
        upper_level_population = [self.level_pop[t.up.index] for t in
                                  self.emitting_molecule.rad_transitions]
        self.upper_level_population = np.array(upper_level_population)

    def all_transition_indices(self):
        return np.arange(self.emitting_molecule.n_rad_transitions)

    @staticmethod
    def get_solid_angle_warning_text(output_type):
        #if output_type != "flux density" and solid_angle is not None:
        return "solid_angle specified, but not needed for"\
                      +f" output_type \"{output_type}\", will be ignored"

    def transform_transition_indices(self,indices):
        if indices is None:
            out = self.all_transition_indices()
        else:
            out = np.atleast_1d(indices)
            if out.ndim > 1:
                raise ValueError("transitions indices must be 0- or 1-dimensional")
        return out

    def frequency_integrated_emission(self,output_type,transitions=None,
                                      solid_angle=None):
        r'''Calculates the frequency-integrated emission of individual transitions.
            This calculation is only easily possible if the dust is optically thin (i.e.
            the dust does not hinder line photons from escaping the source). Thus, this
            function throws an error if the dust is not optically thin. Similarly,
            the calculation is not possible when lines are overlapping and are not
            optically thin.
            The user needs to choose an appropriate observational quantity to
            be compared to the line fluxes calculated here. In particular, for
            optically thin lines, the continuum-subtracted observation might be
            appropriate. On the other hand, for optically thick lines, the
            non-continuum-subtracted observations might be more appropriate
            (because the optically thick line blocks the continuum at the
            line enter; see e.g. Weaver et al. 2018)

        Args:
            output_type (str): Specifies the type of the output. Can either be
                "intensity" (in [W/m2/sr]) or "flux" (in [W/m2]; solid_angle
                needs to be specified)
            transitions (0- or 1-dimensional array_like (dytpe int), or None):
                The index or indices of the transitions for which to calculate the emission.
                For a single transition, an integer can be provided. For several transitions,
                a list of integers is provided. If None, then the emission of all transitions
                is calculated. Defaults to None. The indices correspond to the
                list of transitions in the LAMDA-formatted file, with the first
                transition having index 0.
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
                Needed for output_type "flux". Defaults to None.
        
        Returns:
            list: The list of intensities or fluxes corresponding to the
            input list of requested transitions. If no specific transitions
            where requested (transitions=None), then the list of fluxes corresponds
            to the transitions as listed in the LAMDA file.
        '''
        if output_type != "flux" and solid_angle is not None:
            warning_text = self.get_solid_angle_warning_text(output_type=output_type)
            warnings.warn(warning_text)
        transitions = self.transform_transition_indices(indices=transitions)
        I = self.intensity_calculator.intensities_of_individual_transitions(
                                                  transitions=transitions)
        if output_type == "intensity":
            return I
        elif output_type == "flux":
            if solid_angle is None:
                raise ValueError("solid_angle needs to be specified for"
                                 +" output_type 'flux'")
            return I*solid_angle
        else:
            raise ValueError(f"output_type '{output_type}' not understood")

    def tau_nu(self,nu):
        r''' Calculate the total optical depth (all lines plus dust) at each
        input frequencies

        Args:
            nu (numpy.ndarray): The frequencies in [Hz] for which the optical depth
                should be calculated
        
        Returns:
            np.ndarray: The total optical depth at the input frequencies.
        '''
        self.intensity_calculator.set_nu(nu=nu)
        return self.intensity_calculator.tau_nu_tot

    @staticmethod
    def transform_specific_intensity(specific_intensity,nu,solid_angle,output_type):
        if output_type == "flux density" and solid_angle is None:
            raise ValueError("For flux density, solid angle needs to be specified")
        if output_type == "flux density":
            return specific_intensity*solid_angle
        elif output_type == "Rayleigh-Jeans":
            return helpers.RJ_brightness_temperature(
                                  specific_intensity=specific_intensity,nu=nu)
        elif output_type == "Planck":
            return helpers.Planck_brightness_temperature(
                                 specific_intensity=specific_intensity,nu=nu)
        else:
            raise ValueError(f"invalid output_type \"{output_type}\"")

    def warn_if_solid_angle_not_needed(self,output_type,solid_angle):
        if output_type != "flux density" and solid_angle is not None:
            warning_text = self.get_solid_angle_warning_text(output_type=output_type)
            warnings.warn(warning_text)

    def spectrum(self,nu,output_type,solid_angle=None):
        r''' Calculate the emission (lines + dust) as a function of frequency.

        Args:
            nu (numpy.ndarray): The frequencies in [Hz] for which the spectrum
                should be calculated.
            output_type (str): Specifies the type of the output spectrum.
                Possible choices are "specific intensity" (in [W/m2/Hz/sr]),
                "flux density" (in [W/m2/Hz]; solid_angle needs to be specified),
                "Rayleigh-Jeans" (Rayleigh-Jeans brightness temperature in [K]),
                "Planck" (brightness temperature calculated using the Planck
                function, in [K]).
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
                Needed for output_type "flux density". Defaults to None.
        
        Returns:
            np.ndarray: The emission for each input frequency.
        '''
        self.warn_if_solid_angle_not_needed(output_type=output_type,
                                            solid_angle=solid_angle)
        self.intensity_calculator.set_nu(nu=nu)
        specific_intensity = self.intensity_calculator.specific_intensity_spectrum()
        if output_type == "specific intensity":
            return specific_intensity
        else:
            return self.transform_specific_intensity(
                            specific_intensity=specific_intensity,nu=nu,
                            solid_angle=solid_angle,output_type=output_type)

    def emission_at_line_center(self,output_type,transitions=None,solid_angle=None):
        r'''Calculate the emission at the line center (rest frequency) for the
        specified transitions. The output includes contributions from dust and
        overlapping lines.

        Args:
            transitions (0- or 1-dimensional array_like (dytpe int), or None):
                The index or indices of the transitions for which to calculate the emission.
                For a single transition, an integer can be provided. For several transitions,
                a list of integers is provided. If None, then the emission of all transitions
                is calculated. Defaults to None. The indices correspond to the
                list of transitions in the LAMDA-formatted file, with the first
                transition having index 0.
            output_type (str): Specifies the type of the output.
                Possible choices are "specific intensity" (in [W/m2/Hz/sr]),
                "flux density" (in [W/m2/Hz]; solid_angle needs to be specified),
                "Rayleigh-Jeans" (Rayleigh-Jeans brightness temperature in [K]),
                "Planck" (brightness temperature calculated using the Planck
                function, in [K]).
            solid_angle (:obj:`float`): The solid angle of the source in [sr].
                    Needed for output_type "flux density". Defaults to None.
        
        Returns:
            list: The list of emission values corresponding to the
            input list of requested transitions. If no specific transitions
            where requested (transitions=None), then the list corresponds
            to all transitions as listed in the LAMDA file.
        '''
        self.warn_if_solid_angle_not_needed(output_type=output_type,
                                            solid_angle=solid_angle)
        transitions = self.transform_transition_indices(indices=transitions)
        nu0s = self.emitting_molecule.nu0[transitions]
        if self.emitting_molecule.any_line_has_overlap(line_indices=transitions):
            #do it the slow way
            specific_intensity_nu0 = self.spectrum(
                                        nu=nu0s,output_type="specific intensity")
        else:
            #use fast way
            specific_intensity_nu0 = self.intensity_calculator.specific_intensity_nu0_no_overlap(
                              transitions=transitions)
        if output_type == "specific intensity":
            out = specific_intensity_nu0
        else:
            out = self.transform_specific_intensity(
                         specific_intensity=specific_intensity_nu0,nu=nu0s,
                         solid_angle=solid_angle,output_type=output_type)
        return np.squeeze(out)

    # def efficient_parameter_iterator(self,ext_backgrounds,N_values,Tkin_values,
    #                                  collider_densities_values,T_dust=0,
    #                                  tau_dust=0):
    #     r'''Iterator to update parameters in an efficient way.
    #         For all possible combinations of the input parameters
    #         ext_backgrounds, N_values, Tkin_values and collider_densities_values,
    #         the parameters are updated in each iteration.

    #     Args:
    #         ext_backgrounds (:obj:`dict`): A dictionary, one entry for
    #             each background that should be used. Each entry can be a function
    #             of frequency, or a single number (interpreted as a radiation field
    #             independent of frequency). The units are W/m\ :sup:2/Hz/sr. The keys
    #             of the dictionary are used in the output to identify which
    #             background was used.
    #         N_values (:obj:`list` or numpy.ndarray): The list of column densities to
    #             compute models for, in [m\ :sup:`-2`].
    #         Tkin_values (:obj:`list` or numpy.ndarray): The list of kinetic temperatures
    #             to compute models for, in [K].
    #         collider_densities_values (:obj:`dict`): A dictionary with one entry for each
    #             collider. Each entry is a list of densities for which models should be
    #             computed for, using a "zip" logic (i.e. calculate a model for the first
    #             entries of each list, for the second entries of each list, etc).
    #             Units are [m\ :sup:`-3`]. Example:
    #             collider_densities_values={'para-H2':[4e9,6e9],'ortho-H2':[7e8,1e10]}
    #         T_dust (func or number): The dust temperature in [K] as a function of frequency.
    #              It is assumed that the source function of the dust is a black body
    #              at temperature T_dust. A single number is interpreted as a constant value
    #              for all frequencies. Defaults to 0 (i.e. no internal dust
    #              radiation field).
    #         tau_dust (func or number): optical depth of the dust as a function of frequency.
    #             A single number is interpreted as a constant value
    #             for all frequencies. Defaults to 0 (i.e. no internal dust
    #             radiation field).

    #     Yields:
    #         dict: A dictionary with fields 'ext_background', 'N', 'Tkin' and 
    #             'collider_densities' to identify the values of the parameters
    #             that were updated.
    #     '''
    #     #it is expensive to update Tkin and collider densities, so those should be in
    #     #the outermost loops
    #     n_coll_values = np.array([len(coll_values) for coll_values in
    #                               collider_densities_values.values()])
    #     assert np.all(n_coll_values==n_coll_values[0]),\
    #            'please provide the same number of collider densities for each collider'
    #     n_coll_values = n_coll_values[0]
    #     #set T_dust and tau_dust, other values are not important because they are
    #     #going to change in the loop
    #     initial_ext_bg = list(ext_backgrounds.values())[0]
    #     initial_coll_dens = {coll_name:coll_values[0] for coll_name,coll_values
    #                          in collider_densities_values.items()}
    #     self.update_parameters(
    #            ext_background=initial_ext_bg,N=N_values[0],Tkin=Tkin_values[0],
    #            collider_densities=initial_coll_dens,T_dust=T_dust,
    #            tau_dust=tau_dust)
    #     for Tkin in Tkin_values:
    #         for i in range(n_coll_values):
    #             collider_densities = {collider:values[i] for collider,values
    #                                   in collider_densities_values.items()}
    #             #note: updating Tkin and coll dens together to avoid
    #             #unnecessary overhead
    #             self.update_parameters(Tkin=Tkin,collider_densities=collider_densities)
    #             for ext_background_name,ext_background in ext_backgrounds.items():
    #                 self.update_parameters(ext_background=ext_background)
    #                 for N in N_values:
    #                     self.update_parameters(N=N)
    #                     yield {'ext_background':ext_background_name,'N':N,
    #                            'Tkin':Tkin,'collider_densities':collider_densities}

    def print_results(self):
        '''Prints the results from the radiative transfer computation.'''
        print('\n')
        print('  up   low      nu0 [GHz]    T_ex [K]      poplow         popup'\
              +'         tau_nu0')
        for i,line in enumerate(self.emitting_molecule.rad_transitions):
            output = f'{line.up.index:>4d} {line.low.index:>4d} '\
                     +f'{line.nu0/constants.giga:>14.6f} {self.Tex[i]:>10.2f} '\
                     +f'{self.level_pop[line.low.index]:>14g} '\
                     +f'{self.level_pop[line.up.index]:>14g} '\
                     +f'{self.tau_nu0_individual_transitions[i]:>14g}'
            print(output)
        print('\n')