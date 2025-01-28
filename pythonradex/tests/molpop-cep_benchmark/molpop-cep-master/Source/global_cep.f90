module global_cep
use constants_cep

! ---------------------------------------------------------
! VARIABLES
! ---------------------------------------------------------
! nz : number of points in the atmosphere
! nfrq : number of frequency points in the profile
! n_freq_ud_doppler : number of points per Doppler width in the profile
! n_freq_perfil : number of Doppler widths for the line
! nang : number of angles for the angular integration
! taufac : aumento en tau en cada punto con respecto al anterior tau(i) = taufac * tau(i-1) (NOT USED)
! col_density : maximum optical depth
! sclht : escala de alturas (NOT USED)
! tempc : temperature of the isothermal atmosphere
! iter : iteration number
! output_file : output file
! ---------------------------------------------------------

	real(kind=8) :: col_density
	real(kind=8) :: tempc, radius
	integer :: nz, riter, iter, n_iters, itracc, iaccel, nintracc, nord
	integer :: parabolic, verbose
	integer :: which_scheme, read_previous_flag
	character(len=80) :: output_file, model_file, collision_type, &
		file_physical_conditions, filename_previous

! ---------------------------------------------------------
! VARIABLES WHICH VARY WITH DEPTH 
! ---------------------------------------------------------
! z : geometricl depth 
! tau : optical depth for each transition
! chil : line opacity
! kappa : continuum opacity 
! chie : electron opacity (scattering)
! Sl : line source function
! Lstar : approximate Lamdba operator for a given transition
! Jbar : mean intensity for a given transition
! ---------------------------------------------------------

	real(kind=8), allocatable :: dz(:), tau(:,:), B(:), chil(:), kappa(:), chie(:), Sl(:), Lstar(:), Jbar(:)
	real(kind=8), allocatable :: flux(:)
	real(kind=8) :: beta_file(1600), alpha_file(1600), tau_file(1600), spline_data(1600)

! ---------------------------------------------------------
! VARIABLES WHICH VARY WITH FREQUENCY
! ---------------------------------------------------------
! perfil : line profile
! In : specific intensity
! ---------------------------------------------------------

   real(kind=8), allocatable :: x_e3(:), w_e3(:)

	
! ---------------------------------------------------------
! VARIABLES WHICH VARY WITH DEPTH AND NUMBER OF TRANSITIONS
! ---------------------------------------------------------
! Jbar_total : mean intensity for all transitions
! Lstar_total : approximate lamdba operator (diagonal) for all transitions
! ---------------------------------------------------------	
		
   real(kind=8), allocatable :: Jbar_total(:), dJbardn(:,:), lstar_total(:), flux_total(:)
	real(kind=8), allocatable :: dSldn(:,:), dJbar_totaldn(:,:), dtaudn(:,:), dchildn(:,:)

! ---------------------------------------------------------
! REST OF VARIABLES
! ---------------------------------------------------------
! precision : precision
! relat_error : maximum relative error
! ---------------------------------------------------------	
   
	real(kind=8) :: precision, relat_error, cep_precision
	integer :: relat_error_level

	
! ---------------------------------------------------------
! ATOMIC MODEL
! ---------------------------------------------------------
! nl : n. levels
! ni : n. ionic states
! nt : n. transitions
! nact : n. active transitions
! nli(nlm) : ion at which the level belongs
! itran(2,nt) : upper and lower levels of the transitions
! abun : abundance
! dlevel(1,nlm) : frequency of each level
! dlevel(2,nlm) : statistical weights
! dion(1,ni) : ionization frequency
! dtran(1,nt) : radiative cross section of the transitions = fij * pi * (e**2) / (me * c)
! dtran(2,nt) : frequency of the transition
! dtran(3,nt) : collisional rate
! dtran(4,nt) : Doppler width
! vtherm : thermal velocity
! ---------------------------------------------------------

	integer :: nl, ni, nact, nt,nr 
	integer, allocatable :: nli(:), itran(:,:)
	real(kind=8), allocatable :: dlevel(:,:), dion(:,:), dtran(:,:)
	real(kind=8), allocatable :: collis_all(:,:), collis_all_original(:,:)
	real(kind=8), allocatable :: dopplerw(:,:)
	real(kind=8) :: vtherm, einstein, molmass
	
! ---------------------------------------------------------
! ATMOSPHERE MODEL
! ---------------------------------------------------------	
! temperature (nz) : temperature vs depth
! ---------------------------------------------------------
	real(kind=8), allocatable :: temperature(:)
	real(kind=8) :: hydrogen_density, vmicrot, abund
	integer :: npoints, npoints_initial
	
! ---------------------------------------------------------
! POPULATIONS
! ---------------------------------------------------------
! nh(nz) : hydrogen abundance
! popl(nz*nl) : population of each level in LTE
! pop(nz*nl) : population of each level
! abund : abundance
! ---------------------------------------------------------

	real(kind=8), allocatable :: nh(:), popl(:), pop(:), popold(:), pop_previous_regrid(:)
	real(kind=8), allocatable :: abundance(:), scratch(:,:)

	integer :: n_quadr_beta, optically_thin, start_mode, escape_prob_algorithm
	real(kind=8), allocatable :: tau_beta(:), beta_function(:), beta_spline(:), alphap_function(:), alphap_spline(:)
	
! Strategy
	real(kind=8) :: tau_threshold, r_threshold, col_threshold, factor_abundance
	integer :: kthick_strategy, global_iter, printings_per_decade, newton_maximum_iter
	integer :: cep_convergence_maximum_iter
	
! Levels for which we write output results
	integer :: ntran_output
	integer, allocatable :: upper_output(:), lower_output(:), output_transition(:)
	real(kind=8) :: mu_output
	real(kind=8), allocatable :: infoOut(:,:,:)

	integer :: n_columns_colliders
	integer, allocatable :: collider_column(:)
	
	integer :: nInitialZones
	
end module global_cep
