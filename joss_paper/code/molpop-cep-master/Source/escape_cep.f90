module escape_cep
use constants_cep
use functions_cep
use maths_cep
implicit none
contains

!---------------------------------------------------------
!---------------------------------------------------------
! CEP routines
!---------------------------------------------------------
!---------------------------------------------------------
	
!-----------------------------------------------------------------
! Pre-calculate the optical depths and beta function
!-----------------------------------------------------------------	
	subroutine precalculate_beta
	integer :: i
		if (.not. allocated(tau_beta)) allocate(tau_beta(1000))
		if (.not. allocated(beta_function)) allocate(beta_function(1000))
		if (.not. allocated(beta_spline)) allocate(beta_spline(1000))
		if (.not. allocated(alphap_function)) allocate(alphap_function(1000))
		if (.not. allocated(alphap_spline)) allocate(alphap_spline(1000))
		
		if (verbose == 3) then
			write(*,*) 'Calculating grid for beta and alpha function in the log range:',&
				log10(1.d-30), log10(1.d20)
		endif
		do i = 1, 1000
			tau_beta(i) = (i-1.d0) / 999.d0 * (log10(1.d20)-log10(1.d-30)) +&
				log10(1.d-30)
		enddo
		
		do i = 1, 1000			
			beta_function(i) = beta(10.d0**tau_beta(i))
			alphap_function(i) = alphap(10.d0**tau_beta(i))
		enddo
					
		call splin1(tau_beta,beta_function,1.d30,1.d30,beta_spline)
		
	end subroutine precalculate_beta
   
!---------------------------------------------------------
! This function calculates the beta function
!---------------------------------------------------------
	function beta(tau_in)
   real(kind=8) :: beta, tau_in, salida, tau, paso, coef
   real(kind=8) :: d, b, q, dbdx, x(1), y(1)
   integer :: i, k
   
! KROLIK & McKEE APPROXIMATION   
   	coef = 1.d0/(6.d0*sqrt(3.d0))
   	
! CAREFUL!!!!!
! Note that here we compute tau_linecenter from the input tau, which should then be the integrated optical
! depth of the line. This is a quite strange way of proceeding.
   	tau = tau_in / dsqrt(PI)
   	if (tau < 1.d-4) then 
			salida = 1.d0
		else
			if (tau >= 3.41d0) then
   			salida = 1.d0 / (dsqrt(PI)*tau) * ( dsqrt(log(tau)) + 0.25d0 / dsqrt(log(tau)) + 0.14d0)
   		else
				salida = 1.d0 - 0.415d0*tau + 0.355d0*tau*log(tau)
   			dbdx = 0.355 -0.415 + 0.355*log(tau)
   			k = 1
   			d = tau * coef
   			b = 2.d0 * d
   			q = 1.d0
          	do while (q > 1.d-3)          
            	salida = salida - d * tau
            	dbdx = dbdx - b
             	k = k + 1             	
             	d = -d * tau * sqrt((k+1.d0)/(k+2.d0))*(k-1.d0)/(k*(k+2.d0))
             	b = (k+1.d0) * d
             	q = abs(b/dbdx)
            enddo	    			
	   	endif
		endif
   	beta = salida
   	
! Calculation of the beta function by integrating the E3 exponential integral function
   	salida = 0.d0
   	if (tau_in > 1.d-5) then
   		do i = 1, n_quadr_beta
   			salida = salida + w_e3(i)*(0.5d0-expint(3,tau_in*exp(-x_e3(i)**2)/sqrt(PI))) / tau_in				
   		enddo
   	endif
   	
   	if (tau_in <= 1.d-5) then
			salida = 0.99999999d0 !1.d0
   	endif
   	beta = salida
   	   	
   end function beta
	
!---------------------------------------------------------
! This function calculates the beta function
!---------------------------------------------------------
	function alphap(tau_in)
   real(kind=8) :: alphap, tau_in, salida, tau, paso, coef, profile
   real(kind=8) :: d, b, q, dbdx, x(1), y(1)
   integer :: i, k
      	
! Calculation of the derivative of the alpha function by integrating the E2 exponential integral function
   	salida = 0.d0   	
   	do i = 1, n_quadr_beta
			profile = exp(-x_e3(i)**2)/sqrt(PI)			
   		salida = salida + w_e3(i)*profile*expint(2,tau_in*profile)
   	enddo
   
   	alphap = salida
   	   	
   end function alphap
   
!---------------------------------------------------------
! This function calculates the beta function by interpolating with a spline routine
! If the optical depth is negative, we neglect this transition in the SEE
!---------------------------------------------------------
	function beta2(tau_in)
   real(kind=8) :: beta2, tau_in, salida, tau, paso, coef
   real(kind=8) :: d, b, q, dbdx, x(1), y(1)
   integer :: i, k
      	
! KROLIK & McKEE APPROXIMATION   
   	if (escape_prob_algorithm == 0) then

   		coef = 1.d0/(6.d0*sqrt(3.d0))
   		
! CAREFUL!!!!!
! Note that here we compute tau_linecenter from the input tau, which should then be the integrated optical
! depth of the line. The Krolik & McKee expressions are obtained for line center optical depth and we are
! working outside these routines with the total optical depth of the line
   		tau = tau_in / dsqrt(PI)
		if (tau < 0.d0) then
			if (tau < -60.d0) tau = -60.d0
			beta2 = (1.d0-exp(-tau)) / tau
			beta2 = 1.d0
			return
		endif
			
   		if (tau < 1.d-4) then 
			salida = 1.d0
		else
			if (tau >= 3.41d0) then
				salida = 1.d0 / (dsqrt(PI)*tau) * ( dsqrt(log(tau)) + &
   				0.25d0 / dsqrt(log(tau)) + 0.14d0)
   			else
				salida = 1.d0 - 0.415d0*tau + 0.355d0*tau*log(tau)
   			dbdx = 0.355 -0.415 + 0.355*log(tau)
			k = 1
			d = tau * coef
			b = 2.d0 * d
			q = 1.d0
			do while (q > 1.d-3)          
				salida = salida - d * tau
				dbdx = dbdx - b
				k = k + 1             	
				d = -d * tau * sqrt((k+1.d0)/(k+2.d0))*(k-1.d0)/(k*(k+2.d0))
				b = (k+1.d0) * d
				q = abs(b/dbdx)
			enddo	    			
		endif
		endif
   		beta2 = salida

! Interpolation on a table calculated with the exact expression
		else
		
			if (tau_in == 0.d0) then
				beta2 = 1.d0
				return
   		endif   	

			if (tau_in < 0.d0) then
! 				print *, 'NEGATIVE tau in BETA ', tau_in
				if (tau_in < -30.d0) tau_in = -30.d0
				beta2 = (1.d0-exp(-tau_in)) / tau_in
				tau_in = -tau_in
	!			return
			endif


   		x(1) = log10(tau_in)
   		call spline(tau_beta,beta_function,beta_spline,x,y)
   		salida = y(1)   	   	
   		beta2 = salida
			
		endif
   	   	   	
   end function beta2 
	
!---------------------------------------------------------
! This function calculates the derivative of the alpha function by interpolating with a spline routine
! If the optical depth is negative, we neglect this transition in the SEE
!---------------------------------------------------------
	function alphap2(tau_in)
   real(kind=8) :: alphap2, tau_in, salida, tau, paso, coef
   real(kind=8) :: d, b, q, dbdx, x(1), y(1)
   integer :: i, k
      	
   	
! KROLIK & McKEE APPROXIMATION   
   	if (escape_prob_algorithm == 0) then

   		coef = 1.d0/(6.d0*sqrt(3.d0))
   		tau = tau_in / dsqrt(PI)
   		dbdx = 0.d0
			
			if (tau < 0.d0) then
				if (tau < -60.d0) tau = -60.d0
				salida = (1.d0-exp(-tau)) / tau
				dbdx = (-1.d0+(1.d0+tau)*exp(-tau)) / tau**2
				alphap2 = salida + tau * dbdx
				alphap2 = 0.d0
				return
			endif
			
   		if (tau < 1.d-4) then 
				salida = 1.d0
			else
				if (tau >= 3.41d0) then
   				salida = 1.d0 / (dsqrt(PI)*tau) * ( dsqrt(log(tau)) + &
   					0.25d0 / dsqrt(log(tau)) + 0.14d0)
   			else
					salida = 1.d0 - 0.415d0*tau + 0.355d0*tau*log(tau)
   				dbdx = 0.355 -0.415 + 0.355*log(tau)
   				k = 1
   				d = tau * coef
   				b = 2.d0 * d
   				q = 1.d0
          		do while (q > 1.d-3)          
            		salida = salida - d * tau
            		dbdx = dbdx - b
             		k = k + 1             	
             		d = -d * tau * sqrt((k+1.d0)/(k+2.d0))*(k-1.d0)/(k*(k+2.d0))
             		b = (k+1.d0) * d
             		q = abs(b/dbdx)
            	enddo	    			
	   		endif
			endif
   		alphap2 = salida + tau * dbdx
			
		else
			
			if (tau_in == 0.d0) then
				alphap2 = 1.d0
				return
			endif

			if (tau_in < 0.d0) then
! 				print *, 'NEGATIVE tau in ALPHA ', tau_in
				if (tau_in < -30.d0) tau_in = -30.d0
				alphap2 = (1.d0-exp(-tau_in)) / tau_in + &
					tau_in * (-1.d0+(1.d0+tau_in)*exp(-tau_in)) / tau_in**2
				tau_in = -tau_in	
	!			return
			endif

   		x(1) = log10(tau_in)
   		call spline(tau_beta,alphap_function,alphap_spline,x,y)
   		salida = y(1)   	   	
   		alphap2 = salida
			
		endif
		   	   	   	
   end function alphap2	              	
		
!---------------------------------------------------------
! This subroutine adapts the initial column density so that we start from the
! optically thin or optically thick limits
!---------------------------------------------------------
   subroutine adapt_col_density_strategy(x)
	real(kind=8) :: x(:), factor
	integer :: up, low, it, ip, ipl, ipt, ip1, npmip1, i
	real(kind=8) :: sig, glu, acon, chim, chip, tau0, deltaz, chilp
	
! Calculate line absorption coefficient, line source function and optical depths
		do it = 1, nr			
			up = itran(1,it)
			low = itran(2,it)
			sig = dtran(1,it)  !sig=(h*nu*Blu)/(4*pi)
			glu = dlevel(2,low) / dlevel(2,up)
			acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2

			tau0 = 0.d0
			chip = 0.d0
			chim = 0.d0
			ip1 = 1
			tau(it,0) = 0.d0
			do ip = 1, nz
				ipl = nl * (ip-1)

				chil(ip) = sig / dopplerw(it,ip) * (x(low+ipl) - glu * x(up+ipl))
				Sl(ip) = acon * glu * x(up+ipl) / (x(low+ipl) - glu * x(up+ipl))

         	if (ip == 1) then
					tau(it,ip) = chil(ip) * dz(ip)					
         	else
					tau(it,ip) = tau(it,ip-1) + chil(ip) * dz(ip)
         	endif

			enddo
			
		enddo
						
! Adapt the column density so that the optical depth of the lines fulfills the
! values depending on the chosen strategy

		factor = 1.d0

! Start from optically thin		
		if (kthick_strategy == 0) then
			factor = tau_threshold / maxval(abs(tau(:,nz)))
			
! Modify the previous approximation to start from a number in format 10^n
			factor = 10.d0**int(log10(factor)-1)
		endif
		
! Start from optically thick
		if (kthick_strategy == 1) then
			factor = tau_threshold / minval(abs(tau(:,nz)))
! Modify the previous approximation to start from a number in format 10^n
			factor = 10.d0**int(log10(factor)+1)
		endif
		
		factor_abundance = factor

						
		col_density = sum(dz * factor_abundance * abundance * nh)
		
		deltaz = col_density / (abund * hydrogen_density) / nz
		dz = deltaz
		
! If a file is present, stop the increasing/decreasing strategy when the correct
! column density is found
		if (trim(adjustl(file_physical_conditions)) /= 'none') then
			col_threshold = sum(dz * abundance * nh)					
		endif
		if (verbose == 2) then
			write(*,*) 'Col. density from : ', col_density, ' to ', col_threshold
		endif
						  
   end subroutine adapt_col_density_strategy	
   	
!---------------------------------------------------------
! This subroutine calculates Jbar for every transition using the coupled escape probability method
!---------------------------------------------------------
   subroutine calcJbar_Lstar_cep(x)
	real(kind=8) :: x(:)
	integer :: up, low, it, ip, ipl, ipt, ip1, npmip1, i
	real(kind=8) :: sig, glu, acon, chim, chip, tau0, deltaz, chilp, Jinternal
	
! Calculate line absorption coefficient, line source function and optical depths
! Calculate also their derivatives with respect to the level populations
		do it = 1, nr
			dSldn = 0.d0
			dchildn = 0.d0
			up = itran(1,it)
			low = itran(2,it)
			sig = dtran(1,it) !sig=(h*nu*Blu)/(4*pi)		  
			glu = dlevel(2,low) / dlevel(2,up)
			acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
			Jinternal = dtran(5,it)
						
			tau(it,0) = 0.d0
			dtaudn(0,:) = 0.d0
			do ip = 1, nz
				ipl = nl * (ip-1)
				
				chil(ip) = sig / dopplerw(it,ip) * (x(low+ipl) - glu * x(up+ipl))
				Sl(ip) = acon * glu * x(up+ipl) / (x(low+ipl) - glu * x(up+ipl))
								
				dSldn(ip,up+ipl) = acon*glu / (x(low+ipl) - glu * x(up+ipl))**2 * x(low+ipl)
				dSldn(ip,low+ipl) = -acon*glu / (x(low+ipl) - glu * x(up+ipl))**2 * x(up+ipl)
				
				dchildn(ip,up+ipl) = -sig / dopplerw(it,ip) * glu
				dchildn(ip,low+ipl) = sig / dopplerw(it,ip)
				
				B(ip) = acon * glu * popl(up+ipl) / (popl(low+ipl) - glu * popl(up+ipl))
																	
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! The optical depths that enter into the Krolik & McKee tabulation should be at line center
! Here we calculate the optical depths integrated on the line, and the 1/sqrt(PI) is inside the
! beta2 function
! I know this is not very elegant
         	if (ip == 1) then
					tau(it,ip) = chil(ip) * dz(ip)
					dtaudn(ip,:) = dchildn(ip,:) * dz(ip)
         	else
					tau(it,ip) = tau(it,ip-1) + chil(ip) * dz(ip)
					dtaudn(ip,:) = dtaudn(ip-1,:) + dchildn(ip,:) * dz(ip)
         	endif
				
			enddo
			
			call rt1d_cep(nz,it)
			
! Fill up the blocks of the vector Lstar_total and Jbar_total for all the transitions from the value of
! Jbar and Lstar coming from rt1d_cep
			do ip = 1, nz
				ipt = nr*(ip-1)
				if (optically_thin == 1) then
					Lstar_total(it+ipt) = Jinternal !0.d0
					Jbar_total(it+ipt) = Jinternal !0.d0
					dJbar_totaldn(it+ipt,:) = 0.d0
					flux_total(it+ipt) = 0.d0
				else
					Lstar_total(it+ipt) = lstar(ip) + Jinternal
					Jbar_total(it+ipt) = Jbar(ip) + Jinternal
					dJbar_totaldn(it+ipt,:) = dJbardn(ip,:)
					flux_total(it+ipt) = flux(ip)
					if (tau(it,nz) < 0) flux_total(it+ipt) = 0.d0
				endif
				
			enddo
			
		enddo
		  
   end subroutine calcJbar_Lstar_cep
   
!---------------------------------------------------------
! This subroutine solves the radiative transfer equation using the 
! coupled escape probability scheme developed by Elitzur & Asensio Ramos
!---------------------------------------------------------
   subroutine rt1d_cep(np,it)
	integer :: np, it
   integer :: i, j
   real(kind=8) :: tau_local, escape, escape2, der_alpha, der_beta, error
   real(kind=8) :: delta_tau(8), alpha_tau(8)
	real(kind=8) :: alphap_tau(8), Jexternal_tau0, Jexternal_tauT

   Jbar = 0.d0
	Lstar = 0.d0
	flux = 0.d0
	dJbardn = 0.d0
	alpha_tau = 0.d0
	alphap_tau = 0.d0
	delta_tau = 0.d0
	
! External radiation fields
	Jexternal_tau0 = dtran(6,it)
	Jexternal_tauT = dtran(7,it)
		
	do i = 1, np
		tau_local = tau(it,i) - tau(it,i-1)
	
		escape = beta2(tau_local)
		
!*********************
! Local radiation field
!*********************
		Jbar(i) = Sl(i) * (1.d0 - escape)
		
		der_alpha = alphap2(tau_local)
		der_beta = (der_alpha - escape) / tau_local
		
! Derivative of the source function
		dJbardn(i,:) = (1.d0 - escape) * dSldn(i,:)
! Derivative of the beta function
		dJbardn(i,:) = dJbardn(i,:) - Sl(i) * der_beta * (dtaudn(i,:)-dtaudn(i-1,:))
		
!*********************
! External radiation
!*********************
! From the tau=0 surface
		delta_tau(5) = dabs(tau(it,i))
		escape2 = beta2(delta_tau(5))
		alpha_tau(5) = delta_tau(5) * escape2
		alphap_tau(5) = alphap2(delta_tau(5))

		delta_tau(6) = dabs(tau(it,i-1))
		escape2 = beta2(delta_tau(6))
		alpha_tau(6) = delta_tau(6) * escape2
		alphap_tau(6) = alphap2(delta_tau(6))
		
! From the tau=tau_total surface
		delta_tau(7) = dabs(tau(it,nz)-tau(it,i))
		escape2 = beta2(delta_tau(7))
		alpha_tau(7) = delta_tau(7) * escape2
		alphap_tau(7) = alphap2(delta_tau(7))

		delta_tau(8) = dabs(tau(it,nz)-tau(it,i-1))
		escape2 = beta2(delta_tau(8))
		alpha_tau(8) = delta_tau(8) * escape2
		alphap_tau(8) = alphap2(delta_tau(8))
		
		Jbar(i) = Jbar(i) + 0.5d0 * Jexternal_tau0 / tau_local * &
			(alpha_tau(5) - alpha_tau(6))
		Jbar(i) = Jbar(i) + 0.5d0 * Jexternal_tauT / tau_local * &
			(-alpha_tau(7) + alpha_tau(8))
			
! Derivative of the alpha functions
		dJbardn(i,:) = dJbardn(i,:) + 0.5d0 * Jexternal_tau0 / tau_local * &
			(alphap_tau(5) * dtaudn(i,:) - &
			alphap_tau(6) * dtaudn(i-1,:))
		dJbardn(i,:) = dJbardn(i,:) + 0.5d0 * Jexternal_tauT / tau_local * &
			(alphap_tau(7) * (dtaudn(nz,:)-dtaudn(i,:)) + & 
			alphap_tau(8) * (dtaudn(nz,:)-dtaudn(i-1,:)))
			
! Derivative of the 1/tau term
		dJbardn(i,:) = dJbardn(i,:) - 0.5d0 * Jexternal_tau0 / tau_local**2 * &
			(dtaudn(i,:)-dtaudn(i-1,:)) * &
			(alpha_tau(5) - alpha_tau(6))
		dJbardn(i,:) = dJbardn(i,:) - 0.5d0 * Jexternal_tauT / tau_local**2 * &
			(dtaudn(i,:)-dtaudn(i-1,:)) * &
			(- alpha_tau(7) + alpha_tau(8))

!*********************						
! Diagonal of the Lambda operator
!*********************
		Lstar(i) = (1.d0 - escape) + &
			0.5d0 * Jexternal_tau0 / tau_local * (alpha_tau(5) - alpha_tau(6)) + &
			0.5d0 * Jexternal_tauT / tau_local * (- alpha_tau(7) + alpha_tau(8))
			
!*********************
! Non-local radiation field
!*********************
		do j = 1, np
			if (j /= i) then
				alpha_tau = 0.d0

! Alpha function and its derivative				
				delta_tau(1) = dabs(tau(it,i) - tau(it,j))
				escape = beta2(delta_tau(1))
				alpha_tau(1) = delta_tau(1) * escape
				if (tau(it,i) > tau(it,j)) then
					alphap_tau(1) = alphap2(delta_tau(1))
				else
					alphap_tau(1) = -alphap2(delta_tau(1))
				endif
												
				delta_tau(2) = dabs(tau(it,i-1) - tau(it,j))
				escape = beta2(delta_tau(2))				
				alpha_tau(2) = delta_tau(2) * escape 
				if (tau(it,i-1) > tau(it,j)) then
					alphap_tau(2) = alphap2(delta_tau(2))
				else
					alphap_tau(2) = -alphap2(delta_tau(2))
				endif
												
				delta_tau(3) = dabs(tau(it,i) - tau(it,j-1))				
				escape = beta2(delta_tau(3))
				alpha_tau(3) = delta_tau(3) * escape 
				if (tau(it,i) > tau(it,j-1)) then
					alphap_tau(3) = alphap2(delta_tau(3))
				else
					alphap_tau(3) = -alphap2(delta_tau(3))
				endif
								
				delta_tau(4) = dabs(tau(it,i-1) - tau(it,j-1))
				escape = beta2(delta_tau(4))
				alpha_tau(4) = delta_tau(4) * escape 
				if (tau(it,i-1) > tau(it,j-1)) then
					alphap_tau(4) = alphap2(delta_tau(4))
				else
					alphap_tau(4) = -alphap2(delta_tau(4))
				endif
				
												
! Mean intensity and its derivative with respect to the level populations
				Jbar(i) = Jbar(i) + 0.5d0 * Sl(j) / tau_local * &
					(alpha_tau(1) - alpha_tau(2) - alpha_tau(3) + alpha_tau(4))
				
									
!*****************
! We separate the derivative in three terms
!*****************
!------------------------
! Derivative of the source function
!------------------------
				dJbardn(i,:) = dJbardn(i,:) + 0.5d0 * dSldn(j,:) / tau_local * &
					(alpha_tau(1) - alpha_tau(2) - alpha_tau(3) + alpha_tau(4))

!------------------------
! Derivative of the alpha functions
!------------------------
				dJbardn(i,:) = dJbardn(i,:) + 0.5d0 * Sl(j) / tau_local * &
					(alphap_tau(1) * (dtaudn(i,:)-dtaudn(j,:)) - &
					 alphap_tau(2) * (dtaudn(i-1,:)-dtaudn(j,:)) - &
					 alphap_tau(3) * (dtaudn(i,:)-dtaudn(j-1,:)) + &
					 alphap_tau(4) * (dtaudn(i-1,:)-dtaudn(j-1,:)))
					 
					 
!------------------------
! Derivative of the 1/tau factor
!------------------------
				dJbardn(i,:) = dJbardn(i,:) - 0.5d0 * Sl(j) / tau_local**2 * &
					(dtaudn(i,:)-dtaudn(i-1,:)) * &
					(alpha_tau(1) - alpha_tau(2) - alpha_tau(3) + alpha_tau(4))
									
			endif
		enddo
		
! Do not take into account saturation effects		
		if (tau_local < 0.d0) then
			Jbar(i) = 0.d0
			Lstar(i) = 0.d0
			dJbardn(i,:) = 0.d0
		endif
		if (Jbar(i) < 0.d0) then
			Jbar(i) = 0.d0
			Lstar(i) = 0.d0
			dJbardn(i,:) = 0.d0
		endif

! Total flux of the line (following Eq. (45) + (130) in the notes)
		flux(np) = flux(np) + (1.d0-Jbar(i)/Sl(i)) * Sl(i) * tau_local * dopplerw(it,i)
	enddo
	
   end subroutine rt1d_cep

!-------------------------------------------------------------------
! Returns the value of the equations of the nonlinear set to be solved
!-------------------------------------------------------------------
	subroutine funcv(x,n,fvec)
	integer :: n
	real(kind=8) :: x(n), fvec(n)
	integer :: ip
	integer :: low, up, ipl, ipt, i, j, il, it
	real(kind=8) :: A(nl, nl), B(nl), populat(nl), producto(nl), cul, acon, blu, glu, djbar, poptot

		call calcJbar_Lstar_cep(x)
		
		do ip = 1, nz

! Offsets for storing
			ipl = nl*(ip-1)
			ipt = nr*(ip-1)
		
			A = 0.d0
		
! Upward collisional rates A(low,up)
			call collis(ip,A)

! Calculate the downward rates and interchange the positions
			do i = 1, nl
				do j = 1, i-1
					cul = A(i,j)
! Rate into de j-th level				
					A(j,i) = cul
! Rate into de i-th level				
					A(i,j) = popl(i+ipl) / popl(j+ipl) * cul
				enddo
			enddo
		
! Active radiative rates (preconditioning)
			do it = 1, nact
		
				up = itran(1,it)
				low = itran(2,it)
				acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
				blu = PI4H * dtran(1,it) / dtran(2,it)
				glu = dlevel(2,low) / dlevel(2,up)
			
				djbar = Jbar_total(it+ipt)
			
				A(low,up) = A(low,up) + glu * blu * (acon + djbar )
				A(up,low) = A(up,low) + blu * djbar
			
			enddo		
		
! Background radiative rates
			do it = nact + 1, nr
		
				up = itran(1,it)
				low = itran(2,it)
				acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
				blu = PI4H * dtran(1,it) / dtran(2,it)
				glu = dlevel(2,low) / dlevel(2,up)
			
				A(low,up) = A(low,up) + glu * blu * (acon + Jbar_total(it+ipt))
				A(up,low) = A(up,low) + blu * Jbar_total(it+ipt)
			
			enddo		
		
! The system is homogeneous with null determinant. We have to substitute one of the equations by a closure
! relation. The sum over one column gives the total output rate from level i
			do i = 1, nl

				do j = 1, i-1
					A(i,i) = A(i,i) - A(j,i)
				enddo
				do j = i+1, nl
					A(i,i) = A(i,i) - A(j,i)
				enddo
				
			enddo		
		
! Conservation equation
			B = 0.d0
			A(nl, 1:nl) = 1.d0
		
			poptot = factor_abundance * abundance(ip) * nh(ip)
			B(nl) = poptot
			
			do il = 1, nl
				populat(il) = pop(il+ipl)
			enddo
			
! Return the rate equations			
			producto = matmul(A,populat)-B
						
			do il = 1, nl				
				fvec(il+ipl) = producto(il)
			enddo		
		enddo			
		
	end subroutine funcv
	
!-------------------------------------------------------------------
! Returns the value of the equations of the nonlinear set to be solved
!-------------------------------------------------------------------
	subroutine funcv_analytic(x,n,fvec,fjac)
	integer :: n
	real(kind=8) :: x(n), fvec(n), fjac(n,n)
	integer :: ip
	integer :: low, up, ipl, ipt, i, j, il, it, il2, iz, ind
	real(kind=8) :: A(nl, nl), A2(nl,nl), B(nl), populat(nl), producto(nl), cul, acon, blu, glu, djbar, poptot
	real(kind=8), allocatable :: jac(:,:,:)

		allocate(jac(nl,nl,n))
		
		call calcJbar_Lstar_cep(x)
				
		do ip = 1, nz

! Offsets for storing
			ipl = nl*(ip-1)
			ipt = nr*(ip-1)
		
			A = 0.d0
			jac = 0.d0
		
! Upward collisional rates A(low,up)
			call collis(ip,A)		

! Calculate the downward rates and interchange the positions
			do i = 1, nl
				do j = 1, i-1
					cul = A(i,j)
! Rate into de j-th level				
					A(j,i) = cul
! Rate into de i-th level				
					A(i,j) = popl(i+ipl) / popl(j+ipl) * cul
				enddo
			enddo
					
! Active radiative rates (preconditioning)
			do it = 1, nact
		
				up = itran(1,it)
				low = itran(2,it)
				acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
				blu = PI4H * dtran(1,it) / dtran(2,it)
				glu = dlevel(2,low) / dlevel(2,up)
			
				djbar = Jbar_total(it+ipt)
			
				A(low,up) = A(low,up) + glu * blu * (acon + djbar )				
				A(up,low) = A(up,low) + blu * djbar
				
				jac(low,up,:) = jac(low,up,:) + glu * blu * dJbar_totaldn(it+ipt,:)
				jac(up,low,:) = jac(up,low,:) + blu * dJbar_totaldn(it+ipt,:)
			
			enddo		
		
! Background radiative rates
			do it = nact + 1, nr
		
				up = itran(1,it)
				low = itran(2,it)
				acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
				blu = PI4H * dtran(1,it) / dtran(2,it)
				glu = dlevel(2,low) / dlevel(2,up)
			
				A(low,up) = A(low,up) + glu * blu * (acon + Jbar_total(it+ipt))
				A(up,low) = A(up,low) + blu * Jbar_total(it+ipt)
				
				jac(low,up,:) = jac(low,up,:) + glu * blu * dJbar_totaldn(it+ipt,:)
				jac(up,low,:) = jac(up,low,:) + blu * dJbar_totaldn(it+ipt,:)
			
			enddo		
		
! The system is homogeneous with null determinant. We have to substitute one of the equations by a closure
! relation. The sum over one column gives the total output rate from level i
			do i = 1, nl

				do j = 1, i-1
					A(i,i) = A(i,i) - A(j,i)
					jac(i,i,:) = jac(i,i,:) - jac(j,i,:)
				enddo
				do j = i+1, nl
					A(i,i) = A(i,i) - A(j,i)
					jac(i,i,:) = jac(i,i,:) - jac(j,i,:)
				enddo
				
			enddo		
			
			A2 = A
					
! Conservation equation
			B = 0.d0
			A(nl, 1:nl) = 1.d0
			jac(nl, 1:nl, :) = 0.d0
		
			poptot = factor_abundance * abundance(ip) * nh(ip)
			B(nl) = poptot
			
			do il = 1, nl
				populat(il) = x(il+ipl)
			enddo
			
! Return the rate equations			
			producto = matmul(A,populat)-B
						
			do il = 1, nl				
				fvec(il+ipl) = producto(il)								
			enddo

! Analytical Jacobian. Since the equations are F_j = sum(A_jk * n_k, k) - b_j, the Jacobian J_ji is 
! J_ji = dF_j/dn_i = 
! sum(dA_jk/dn_i * n_k,k) + sum(A_jk * dn_k/dn_i,k) = sum(dA_jk/dn_i * n_k,k) + A_ji
			do il = 1, nl
				do il2 = 1, nl
					fjac(il+ipl,il2+ipl) = A(il,il2)
				enddo
				do il2 = 1, nl*nz					
					fjac(il+ipl,il2) = fjac(il+ipl,il2) + sum(jac(il,:,il2)*populat)
				enddo
			enddo
						
		enddo
				
		deallocate(jac)
		
	end subroutine funcv_analytic	
	
!-------------------------------------------------------------------
! Calculates the Jacobian using forward-differences
!-------------------------------------------------------------------            
	subroutine fdjac(n,x,fvec,df)
	integer :: n,np
	real(kind=8) :: df(n,n),fvec(n),x(n),EPS
	PARAMETER (EPS=1.d-6)
	integer :: i,j
	real(kind=8) :: h,temp,f(n)
	
		do j=1,n
			temp=x(j)
			h=EPS*dabs(temp)
			if(h.eq.0.d0)h=EPS
			x(j)=temp+h
			h=x(j)-temp
			call funcv(x,n,f)
			x(j)=temp
			do i=1,n 
				df(i,j)=(f(i)-fvec(i))/h
			enddo 
		enddo 
        
	end subroutine fdjac

!-------------------------------------------------------------------
! Returns the equations and the Jacobian of the nonlinear set to be solved at point x
!-------------------------------------------------------------------
	subroutine usrfun(x,n,fvec,fjac,derivatives)
	integer :: n, i, j
	logical :: derivatives
	real(kind=8) :: x(n), fvec(n), fjac(:,:)
	real(kind=8), allocatable :: fjac_num(:,:)
	
		allocate(fjac_num(n,n))
		
		fjac = 0.d0
! Analytical derivatives	
		if (derivatives) then
			call funcv_analytic(x,n,fvec,fjac)			
		else
! Numerical derivatives
			call funcv(x,n,fvec)
			call fdjac(n,x,fvec,fjac)
		endif
		
		deallocate(fjac_num)
	end subroutine usrfun
	
!-------------------------------------------------------------------
! Solves a system of nonlinear equations using the Newton method
!-------------------------------------------------------------------
	subroutine mnewt(ntrial,x,n,tolx,tolf,derivatives,error)
	integer :: n,ntrial, error
	real(kind=8) :: tolf,tolx,x(n)
	integer :: i,k
	logical :: derivatives
	logical :: calculate_jacobian
	real(kind=8) :: d,errf,errx,fvec(n),p(n), tol, amax, factor, errx_old
	real(kind=8), allocatable :: fjac(:,:)
	integer, allocatable :: indx(:)
	
		allocate(fjac(n,n))
		errx = 1.d0
		errx_old = errx
		error = 0
		factor = 1.d0
		do k=1,ntrial
					
			call usrfun(x,n,fvec,fjac,derivatives)   ! User subroutine 
			                                         ! supplies function values at x in fvec
			
! Scale each line of the Jacobian by its maximum to decrease the dynamical range
			do i = 1, n
				amax = maxval(dabs(fjac(:,i)))
				if (amax < 1.d-20) amax = 1.d-20
				fjac(:,i) = fjac(:,i) / amax
			enddo
                
			do i=1,n  !Check function convergence.
				errf=errf+dabs(fvec(i))
			enddo 
			if (errf <= tolf) then
				print *, 'Function absolute value is smaller than tolf : ', errf
				error = 1
				deallocate(fjac)
				return
				!return
			endif
			p = -fvec
			
			if (n <= 1000) then
! Use LU decomposition
				allocate(indx(n))
				tol = 1.d-10
				call ludcmp(fjac,indx,tol)
				call lubksb(fjac,indx,p)
				deallocate(indx)
				
			else
! Use BiCGStab			
				call bicgstab(fjac,p)
			
			endif
		              
			errx=0.d0  ! Check root convergence.
			!x = x + p

! This trick seems to work nicely
			x = x + (1.d0*k/ntrial) * p
						
			!do i=1,n   !Update solution.
			!	errx=errx+dabs(p(i))                    
			!enddo
			
			errx = maxval(abs(p/x))
			
			if(errx <= tolx) then				
				if (verbose == 1) then
					write(*,FMT='(A,I4,A,E13.6)') 'Iteration ',k, ' -  Relat. error: ',errx
				endif
				deallocate(fjac)
				return
			endif
			
			if (verbose == 1) then
				write(*,FMT='(A,I4,A,E13.6)') 'Iteration ',k, ' -  Relat. error: ',errx
			endif
		enddo
		
! Check if populations are positive		
		if (minval(x) < 0.d0 .or. k >= ntrial) then
			error = 1
		endif
		
		deallocate(fjac)
        
	end subroutine mnewt
	
!-------------------------------------------------------------------
! Evaluates the function to solve for the NAG Newton 
!-------------------------------------------------------------------
! 	subroutine fcn_nag(n, x, fvec, fjac, ldfjac, iflag)
! 	integer :: n, ldfjac, iflag
! 	real(kind=8) :: x(n), fvec(n), fjac(ldfjac,n)
! 	real(kind=8) :: fvec2(n), fjac2(ldfjac,n)
! 	
! 		call usrfun(x,n,fvec2,fjac2,.TRUE.)
! 		
! 		if (iflag == 1) then
! 			fvec = fvec2
! 		endif
! 		
! 		if (iflag == 2) then
! 			fjac = fjac2
! 		endif
! 	
! 	end subroutine fcn_nag	
	
!-------------------------------------------------------------------
! Solves a system of nonlinear equations using the Newton method of the NAG library
!-------------------------------------------------------------------
! 	subroutine mnewt_nag(ntrial,x,n,tolx,tolf,derivatives,error)
! 	integer :: n,ntrial, ldfjac, ifail, maxfev, mode, nprint, nfev, njev, lr, error
! 	real(kind=8) :: tolf,tolx,x(n)
! 	integer :: i,k
! 	logical :: derivatives
! 	logical :: calculate_jacobian
! 	real(kind=8) :: d,errf,errx,fvec(n),p(n), x2(n), tol, amax, factor_mult	
! 	real(kind=8) :: xtol, factor
! 	real(kind=8), allocatable :: fjac(:,:), diag(:), r(:), qtf(:), w(:,:)
! 	integer, allocatable :: indx(:)			
! 	real(kind=8) :: f06ejf, x02ajf
! 	external f06ejf, x02ajf
! 	
! 		ldfjac = n
! 		xtol = tolx
! 		ifail = 0
! 		maxfev = 100*(n+1)
! 		mode = 1
! 		lr = n*(n+1)/2
! 		error = 0
! 		
! 		allocate(fjac(ldfjac,n))
! 		allocate(diag(n))
! 		allocate(r(lr))
! 		allocate(qtf(n))
! 		allocate(w(n,4))
! 		
! 		x = 1.d0
! 		diag = 1.d0
! 		factor = 100.d0
! 		nprint = 0
! 
! #if defined(YES)
! 		call c05pcf(fcn_nag,n,x,fvec,fjac,ldfjac,xtol,maxfev,diag,&
! 			mode,factor,nprint,nfev,njev,r,lr,qtf,w,ifail)
! 		print *, 'Final 2-norm of the residuals : ', f06ejf(n,fvec,1)
! 		print *, 'Function evaluations : ', nfev
! 		print *, 'Jacobian evaluations : ', njev
! #else
! 		print *, 'CEP-NAG not supported in this version'
! 		stop
! #endif
! 		
! 		deallocate(fjac)
! 		deallocate(diag)
! 		deallocate(r)
! 		deallocate(qtf)
! 		deallocate(w)
! 		
! 		if (ifail /= 0) error = 1
! 						
! 	end subroutine mnewt_nag	
	
!-------------------------------------------------------------------
! Solves a NLTE problem
!-------------------------------------------------------------------
	subroutine CEP_solver(error)
	integer :: error, i, loop
	real(kind=8) :: maximum_relative_change_pop
	
! This is the most external loop that solves the problem using different number
! of zones until the problems converges
		maximum_relative_change_pop = 1.d0
		loop = 1

! If the convergence criterium is positive, put a maximum number of iterations
! If not, this means that we do not want to converge the grid. Do only one iteration
		if (cep_precision > 0.d0) then
			cep_convergence_maximum_iter = newton_maximum_iter
		else
			cep_convergence_maximum_iter = 1
		endif
		
		error = 0
! Start external loop
		do while (maximum_relative_change_pop > cep_precision .and. &
			loop <= cep_convergence_maximum_iter)
			
! Use the Newton method with analytical derivatives
			if (which_scheme == 2) then			
				call mnewt(newton_maximum_iter,pop,nz*nl,precision,0.d0,.TRUE.,error)
			endif	

! Use the NAG-Newton method with analytical derivatives
			if (which_scheme == 4) then
! #if defined(YES)
! 				call mnewt_nag(newton_maximum_iter,pop,nz*nl,precision,0.d0,.TRUE.,error)
! #else
				print *, 'CEP-NAG not supported in this version'
				stop
! #endif
			endif	

! Use the Newton method with numerical derivatives
			if (which_scheme == 3) then		
				call mnewt(newton_maximum_iter,pop,nz*nl,precision,0.d0,.FALSE.,error)	
			endif		

! Use the CEP ALI method
			if (which_scheme == 1) then
				relat_error = 1.d0
				iter = 1
				
! Ng acceleration
				itracc = -4
				iaccel = 0
				nord = 4
				nintracc = 5
				
				allocate(scratch(nz*nl,nord+1))
				
! Iterate until reaching the precision we want or we go above the maximum number of iterations
				do while (relat_error > precision .and. iter < n_iters)

! Calculate Jbar_total and Lstar_total solving the RT equation		
					call calcJbar_Lstar_cep(pop)
					
! Do the population correction
      			call correct_populations
      			
      			call acceleration

					iter = iter + 1
					itracc = itracc + 1

					if (verbose == 1) then
						write(*,FMT='(A,I4,A,E13.6,A,I3)') 'Iteration: ',iter-1, ' -  Relat. error: ',&
							relat_error, ' - Level with largest change: ', relat_error_level
					endif

				enddo
				
				deallocate(scratch)
				
				if (relat_error > precision) then
					error = 1
				endif
			endif
			
			loop = loop + 1
			
! Do the regridding
			maximum_relative_change_pop = maxval(abs(pop-pop_previous_regrid) / pop)
			if (verbose == 1 .and. cep_convergence_maximum_iter /= 1) then
				write(*,*) 'Grid relat. change : ', maximum_relative_change_pop
			endif
			if (maximum_relative_change_pop > cep_precision &
				.and. loop <= cep_convergence_maximum_iter) then
				
				call regrid(.FALSE.,.TRUE.)
			endif
									
		enddo
											
	end subroutine CEP_solver
	
!-----------------------------------------------------------------
! Carry out a Ng acceleration if allowed
!-----------------------------------------------------------------
	subroutine acceleration

   	if (iaccel == 1 .and. itracc > 0) then
      	if (ng(pop,nl*nz,nord,scratch)) then
         	print *, ' **** Ng acceleration'
            itracc = -nintracc
         endif
      endif
      
	end subroutine acceleration
	
! ---------------------------------------------------------
! Carry out the Ng acceleration
! ---------------------------------------------------------
	function ng(y,m,n,yy)
	logical :: ng
	integer, INTENT(IN) :: n, m
	real(kind=8), INTENT(INOUT) :: y(m), yy(m,*)
	
	real(kind=8) :: A(7,7), C(7), di, dy, sum, wt
	integer :: k, i, j
	integer, save :: ntry
	data ntry /0/
	
	ng = .FALSE.
	if (n < 1 .or. n > 5) return
	ntry = ntry + 1
	yy(:,ntry) = y
	ng = .false.
	if (ntry < n+1) return
		
	A = 0.d0 ; C = 0.d0
	
	do k = 1, m
		wt = 1.d0 / (1.d0 + dabs(y(k)))
		do i = 1, n
			dy = yy(k,ntry-1) - yy(k,ntry)
			di = wt * (dy + yy(k,ntry-i) - yy(k,ntry-i-1) )
			c(i) = c(i) + di*dy
			do j = 1, n
				A(i,j) = A(i,j) + di * (dy + yy(k,ntry-j) - yy(k,ntry-j-1) )
			enddo !j
		enddo !i
	enddo !k
	
	call llslv(A,C,n,7)
	
	ng = .true.
	do i = 1, n
		y(:) = y(:) + c(i) * ( yy(:,ntry-i) - yy(:,ntry) )
	enddo !i
	
	ntry = 0
	
	end function ng


end module escape_cep
