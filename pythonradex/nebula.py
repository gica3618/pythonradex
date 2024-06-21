from scipy import constants
import numpy as np
from pythonradex import helpers,escape_probability,atomic_transition
from pythonradex.molecule import EmittingMolecule
import warnings
import time
import numba as nb


class RateEquations():

    '''Represents the equations of statistical equilibrium for the level populations
    of a molecule'''

    def __init__(self,molecule,collider_densities,Tkin,mode='std'):
        '''molecule is an instance of the Molecule class
        collider_densities is a dict with the densities of the collision partners
        Tkin is the kinetic temperature
        mode is the method to solve the radiative transfer: either std (i.e.
        lambda iteration) or ALI (i.e. accelerated lambda iteration)'''
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
        assert mode in ('std','ALI')
        self.mode = mode
        if self.mode == 'std':
            self.rad_rates = self.rad_rates_std
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
    def rad_rates_std(Jbar_lines,A21_lines,B12_lines,B21_lines):
        '''compute the rates for the Lambda iteration method for radiative transitions,
        from the average radiation field for all lines given by Jbar_lines.'''
        uprate = B12_lines*Jbar_lines
        downrate = A21_lines+B21_lines*Jbar_lines
        return uprate,downrate

    @staticmethod
    #@nb.jit(nopython=True,cache=True)#doesn't help
    def rad_rates_ALI(beta_lines,I_ext_lines,A21_lines,B12_lines,B21_lines):
        '''compute the rates for the accelerated Lambda iteration (ALI) method
        for radiative transitions, from the escape probability for all lines (beta_lines)
        and the external intensity for all lines (I_ext_lines).'''
        #see section 7.10 of the Dullemond radiative transfer lectures (in the
        #ALI_explained.pdf document)
        uprate = B12_lines*I_ext_lines*beta_lines
        downrate = A21_lines*beta_lines + B21_lines*I_ext_lines*beta_lines
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
        '''Solve the SE equations
           for std mode, kwargs includes Jbar_lines
           for ALI mode, kwargs includes beta_lines and I_ext_lines'''
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


class Nebula():

    '''Represents an emitting gas cloud

    Important attributes:
    ---------------

    - emitting_molecule: EmittingMolecule
        An object containing atomic data and line profile information
    
    - geometry: str
        geometry of the gas cloud

    - ext_background: func
        function returning the external background radiation field for given frequency

    - Tkin: float
        kinetic temperature of colliders

    - collider_densities: dict
        densities of the collision partners

    - Ntot: float
        total column density

    - rate_equations: RateEquations
        object used to set up and solve the equations of statistical equilibrium

    - warn_negative_tau: bool
        if True, print a warning if the calculation results in negative optical depth
        
    The following attributes are available after the radiative transfer has been solved:
    
    - tau_nu0: numpy.ndarray
        optical depth of each transition at the central frequency.
    - level_pop: numpy.ndarray
        fractional population of levels.
    - Tex: numpy.ndarray
        excitation temperature of each transition.
    '''
    relative_convergence = 1e-6
    min_iter = 10
    max_iter = 1000
    underrelaxation = 0.3 #RADEX uses 0.3
    min_tau_considered_for_convergence = 1e-2
    min_iter_before_ng_acceleration = 4
    ng_acceleration_interval = 4
    geometries = {'uniform sphere':escape_probability.UniformSphere,
                  'uniform sphere RADEX':escape_probability.UniformSphereRADEX,
                  'uniform slab':escape_probability.UniformSlab,
                  'LVG slab':escape_probability.UniformLVGSlab,
                  'LVG sphere':escape_probability.UniformLVGSphere,
                  'LVG sphere RADEX':escape_probability.LVGSphereRADEX}
    line_profiles = {'Gaussian':atomic_transition.GaussianLineProfile,
                     'rectangular':atomic_transition.RectangularLineProfile}

    def __init__(self,datafilepath,geometry,line_profile,width_v,iteration_mode='ALI',
                 use_NG_acceleration=True,average_beta_over_line_profile=False,
                 warn_negative_tau=True,debug=False):
        '''
        Parameters:    
        ---------------        
        
        datafilepath: str
             path to the LAMDA data file that contains the atomic data

        geometry: str
            geometry of the gas cloud that determines how the escape probability
            and flux are calculated. Currently available are "uniform sphere", 
            "uniform sphere RADEX", "uniform slab", "LVG slab" and "LVG shere".
            Here, "uniform sphere RADEX" uses the forumla for a uniform sphere
            for the escape probability and the formula for a uniform slab to calculate
            the flux, as in the original RADEX code. LVG (large velocity gradient) slab
            and sphere are the same as in RADEX.

        line_profile: str
            type of line profile. Available are "Gaussian" and "rectangular".

        width_v: float
            width of the line in [m/s]. For Gaussian, this is the FWHM.

        iteration_mode: str
            Which method to use to solve the radiative transfer: "std" for standard
            Lambda iteration, or "ALI" for Accelerated Lambda Iteration. ALI is
            recommended, and is the default.
    
        use_NG_acceleration: bool
            Whether or not to use Ng-acceleration to speed up convergence. Default
            is to use it.

        partition_function: func
            Partition function. If None, partition function will be calculated from the
            atomic data provided by the datafile

        average_beta_over_line_profile: bool
            Whether or not to average the escape probability over the line profile,
            or just take its value at the frequency of the line center (as in RADEX).
            Averaging gives more accurate results, but slows down the calculation

        warn_negative_tau: bool
            if True, print a warning if the calculation results in negative optical depth

        debug: bool
            if True, additional information is printed out
        '''
        self.emitting_molecule = EmittingMolecule.from_LAMDA_datafile(
                                    datafilepath=datafilepath,
                                    line_profile_cls=self.line_profiles[line_profile],
                                    width_v=width_v)
        self.geometry = self.geometries[geometry]()
        self.iteration_mode = iteration_mode
        self.use_NG_acceleration = use_NG_acceleration
        if average_beta_over_line_profile:
            self.beta_alllines = self.beta_alllines_average
        else:
            self.beta_alllines = self.beta_alllines_nu0
        self.warn_negative_tau = warn_negative_tau
        self.debug = debug
        #the following attributes are needed for the compiled functions:
        rad_trans = self.emitting_molecule.rad_transitions
        self.A21_lines = np.array([line.A21 for line in rad_trans])
        self.B12_lines = np.array([line.B12 for line in rad_trans])
        self.B21_lines = np.array([line.B21 for line in rad_trans])
        self.nu_lines = np.array([line.line_profile.coarse_nu_array for line in
                                  rad_trans])
        self.nu0_lines = np.array([line.nu0 for line in rad_trans])
        self.phi_nu_lines = np.array([line.line_profile.coarse_phi_nu_array for
                                      line in rad_trans])
        self.phi_nu0_lines = np.array([line.line_profile.phi_nu0 for line in
                                       rad_trans])
        self.gup_lines = np.array([line.up.g for line in rad_trans])
        self.glow_lines = np.array([line.low.g for line in rad_trans])
        self.low_number_lines = np.array([line.low.number for line in rad_trans])
        self.up_number_lines = np.array([line.up.number for line in rad_trans])
        self.dense_nu_lines = np.array([line.line_profile.dense_nu_array for line in
                                        rad_trans])
        self.dense_phi_nu_lines = np.array([line.line_profile.dense_phi_nu_array for
                                            line in rad_trans])

    def set_cloud_parameters(self,ext_background,Ntot,Tkin,collider_densities):
        '''
        ext_background: func
            The function should take the frequency in Hz as input and return the
            background radiation field in [W/m2/Hz/sr]

        Ntot: float
            total column density in [1/m2]

        Tkin: float
            kinetic temperature of the colliders

        collider_densities: dict
            number densities of the collision partners in [1/m3]. Following keys
            are recognised: "H2", "para-H2", "ortho-H2", "e", "H", "He", "H+"
        '''
        #why do I put this into a seperate method rather than __init__? The reason
        #is that the stuff in __init__ is expensive (in particular reading the
        #LAMDA file). So if I explore e.g. a grid of Ntot, that stuff in __init__
        #should only be done once to save computation time. Then this method
        #can be called inside the loop over e.g. the Ntot values
        #note that setting up the rate equations is expensive, so I only do it if
        #necessary
        self.ext_background = ext_background
        #needed for the compiled functions:
        self.I_ext_lines = np.array([self.ext_background(line.nu0) for line in
                                     self.emitting_molecule.rad_transitions])
        self.Ntot = Ntot
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

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_averaged_beta_lines(level_populations,low_number_lines,up_number_lines,Ntot,
                                 A21_lines,phi_nu_lines,gup_lines,glow_lines,nu_lines,
                                 beta_function):
        n_lines = low_number_lines.size
        averaged_beta = np.empty(n_lines)
        for i in range(n_lines):
            N1 = Ntot * level_populations[low_number_lines[i]]
            N2 = Ntot * level_populations[up_number_lines[i]]
            tau_nu = atomic_transition.fast_tau_nu(
                          A21=A21_lines[i],phi_nu=phi_nu_lines[i,:],
                          g_up=gup_lines[i],g_low=glow_lines[i],N1=N1,N2=N2,
                          nu=nu_lines[i,:])
            beta_nu = beta_function(tau_nu)
            #phi_nu is already normalised, but since I use the coarse grid here,
            #I re-normalise
            phi_nu = phi_nu_lines[i,:]
            nu = nu_lines[i,:]
            norm = helpers.fast_trapz(y=phi_nu,x=nu)
            averaged_beta[i] = helpers.fast_trapz(y=beta_nu*phi_nu,x=nu)/norm
        return averaged_beta

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_beta_nu0_lines(level_populations,low_number_lines,up_number_lines,Ntot,
                            A21_lines,phi_nu0_lines,gup_lines,glow_lines,nu0_lines,
                            beta_function):
        n_lines = low_number_lines.size
        tau_nu0_lines = np.empty(n_lines)
        for i in range(n_lines):
            N1 = Ntot * level_populations[low_number_lines[i]]
            N2 = Ntot * level_populations[up_number_lines[i]]
            tau_nu0_lines[i] = atomic_transition.fast_tau_nu(
                                 A21=A21_lines[i],phi_nu=phi_nu0_lines[i],
                                 g_up=gup_lines[i],g_low=glow_lines[i],N1=N1,N2=N2,
                                 nu=nu0_lines[i])
        return beta_function(tau_nu0_lines)

    def beta_alllines_average(self,level_populations):
        return self.fast_averaged_beta_lines(
                        level_populations=level_populations,
                        low_number_lines=self.low_number_lines,
                        up_number_lines=self.up_number_lines,Ntot=self.Ntot,
                        A21_lines=self.A21_lines,phi_nu_lines=self.phi_nu_lines,
                        gup_lines=self.gup_lines,glow_lines=self.glow_lines,
                        nu_lines=self.nu_lines,beta_function=self.geometry.beta)

    def beta_alllines_nu0(self,level_populations):
        return self.fast_beta_nu0_lines(
                       level_populations=level_populations,
                       low_number_lines=self.low_number_lines,
                       up_number_lines=self.up_number_lines,Ntot=self.Ntot,
                       A21_lines=self.A21_lines,phi_nu0_lines=self.phi_nu0_lines,
                       gup_lines=self.gup_lines,glow_lines=self.glow_lines,
                       nu0_lines=self.nu0_lines,beta_function=self.geometry.beta)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_source_function_alllines(level_populations,low_number_lines,
                                      up_number_lines,A21_lines,B21_lines,B12_lines):
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

    def source_function_alllines(self,level_populations):
        return self.fast_source_function_alllines(
                 level_populations=level_populations,
                 low_number_lines=self.low_number_lines,
                 up_number_lines=self.up_number_lines,A21_lines=self.A21_lines,
                 B21_lines=self.B21_lines,B12_lines=self.B12_lines)

    @staticmethod
    @nb.jit(nopython=True,cache=True)
    def fast_Jbar_allines(beta_lines,I_ext_lines,source_func):
        return beta_lines*I_ext_lines + (1-beta_lines)*source_func

    def Jbar_alllines(self,level_populations,beta_lines):
        source_func = self.source_function_alllines(level_populations=level_populations)
        return self.fast_Jbar_allines(
                    beta_lines=beta_lines,I_ext_lines=self.I_ext_lines,
                    source_func=source_func)

    def get_new_level_pop(self,old_level_pop):
        assert self.rate_equations.mode in ('ALI','std')
        Einstein_kwargs = {'A21_lines':self.A21_lines,
                          'B12_lines':self.B12_lines,'B21_lines':self.B21_lines}
        ALI_kwargs = {'I_ext_lines':self.I_ext_lines} | Einstein_kwargs
        if old_level_pop is None:
            #assume optically thin emission for the first iteration
            if self.rate_equations.mode == 'ALI':
                beta_lines = np.ones(self.emitting_molecule.n_rad_transitions)
                return self.rate_equations.solve(beta_lines=beta_lines,**ALI_kwargs)
            elif self.rate_equations.mode == 'std':
                Jbar_lines = self.I_ext_lines
                return self.rate_equations.solve(Jbar_lines=Jbar_lines,
                                                 **Einstein_kwargs)
        else:
            beta_lines = self.beta_alllines(level_populations=old_level_pop)
            if self.rate_equations.mode == 'ALI':
                return self.rate_equations.solve(beta_lines=beta_lines,**ALI_kwargs)
            elif self.rate_equations.mode == 'std':
                Jbar_lines = self.Jbar_alllines(level_populations=old_level_pop,
                                                beta_lines=beta_lines)
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
        """Solves the radiative transfer by iterating and initialises
        new attributes that contain the solution.
        """
        level_pop = self.get_new_level_pop(old_level_pop=None)
        old_level_pops = nb.typed.List([])
        Tex_residual = np.ones(self.emitting_molecule.n_rad_transitions) * np.inf
        old_Tex = 0
        counter = 0
        residual = np.inf
        while residual>self.relative_convergence or counter < self.min_iter:
            counter += 1
            if counter%10 == 0 and self.debug:
                print(f'iteration {counter}')
            if counter > self.max_iter:
                raise RuntimeError('maximum number of iterations reached')
            new_level_pop = self.get_new_level_pop(old_level_pop=level_pop)
            Tex = self.emitting_molecule.get_Tex(new_level_pop)
            Tex_residual = helpers.relative_difference(Tex,old_Tex)
            if self.debug:
                print(f'max relative Tex residual: {np.max(Tex_residual):.3g}')
            tau_nu0_lines = self.emitting_molecule.get_tau_nu0(
                                       N=self.Ntot,level_population=new_level_pop)
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
        if self.debug:
            print(f'converged in {counter} iterations')
        self.tau_nu0 = self.emitting_molecule.get_tau_nu0(
                                   N=self.Ntot,level_population=level_pop)
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
    def fast_compute_line_fluxes(solid_angle,Tex,nu_lines,phi_nu_lines,level_pop,
                                  low_number_lines,up_number_lines,gup_lines,glow_lines,
                                  Ntot,A21_lines,compute_flux_nu):
        n_lines = Tex.size
        n_nu = nu_lines[0,:].size
        tau_nu_lines = np.empty((n_lines,n_nu))
        obs_line_spectra = tau_nu_lines.copy()
        obs_line_fluxes = np.empty_like(Tex)
        for i in range(n_lines):
            source_function = helpers.B_nu(T=Tex[i],nu=nu_lines[i,:])
            x1 = level_pop[low_number_lines[i]]
            x2 = level_pop[up_number_lines[i]]
            A21 = A21_lines[i]
            N1 = Ntot*x1
            N2 = Ntot*x2
            tau_nu_lines[i,:] = atomic_transition.fast_tau_nu(
                                A21=A21,phi_nu=phi_nu_lines[i],g_low=glow_lines[i],
                                g_up=gup_lines[i],N1=N1,N2=N2,nu=nu_lines[i,:])
            obs_line_spectra[i,:] = compute_flux_nu(
                                          tau_nu=tau_nu_lines[i,:],
                                          source_function=source_function,
                                          solid_angle=solid_angle)
            obs_line_fluxes[i] = helpers.fast_trapz(obs_line_spectra[i,:],nu_lines[i,:])
        return tau_nu_lines,obs_line_spectra,obs_line_fluxes

    def compute_line_fluxes(self,solid_angle):
        '''Compute the observed spectra and total fluxes for each line. This
        requires that the radiative transfer has been solved. This method
        computes the attributes obs_line_fluxes (total observed line fluxes in W/m2)
        and obs_line_spectra (observed line spectra in W/m2/Hz).
        
        Parameters:
        ---------------
        solid_angle: float
            the solid angle of the source in [rad2]
        '''
        self.tau_nu_lines,self.obs_line_spectra,self.obs_line_fluxes\
              = self.fast_compute_line_fluxes(
                      solid_angle=solid_angle,Tex=self.Tex,nu_lines=self.dense_nu_lines,
                      phi_nu_lines=self.dense_phi_nu_lines,level_pop=self.level_pop,
                      low_number_lines=self.low_number_lines,
                      up_number_lines=self.up_number_lines,gup_lines=self.gup_lines,
                      glow_lines=self.glow_lines,Ntot=self.Ntot,
                      A21_lines=self.A21_lines,compute_flux_nu=self.geometry.compute_flux_nu)

    def print_results(self):
        '''print out the results from the radiative transfer computation. Can
        only be called if the radiative transfer has been solved.'''
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