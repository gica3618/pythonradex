module functions_cep
use constants_cep
use global_cep
use maths_cep
use io_cep
use maths_molpop, only: Pass_Header, rdinp
implicit none

contains


! ---------------------------------------------------------
! Inicialization of all the variables
! ---------------------------------------------------------
   subroutine init

   real(kind=8) :: div, deca, deltatau
   integer :: i

		call read_intermediate_data
						
		if (verbose == 2) then
			write(*,*) 'Output file : ', trim(adjustl(output_file))
		endif				
		
! Starting value of the column density that will be adapted later
! to fulfill the initial criterium given in the input file
		col_density = col_threshold
		
		if (verbose == 3) then
			write(*,*) 'Molecular abundance : ', abund
		endif
		
		if (verbose == 3) then
			write(*,*) 'Hydrogen density [cm^-3] : ', hydrogen_density
		endif
				
		if (verbose == 3) then
			write(*,*) 'Kinetic temperature [K] : ', tempc
		endif

		if (verbose == 3) then
			write(*,*) 'Microturbulent velocity [km/s] : ', vmicrot
		endif
                        
		if (verbose == 3) then
			write(*,*) 'Escape probability (0-> KROLIK & McKEE, 1-> EXACT) : ',&
				escape_prob_algorithm
		endif
						
		radius = col_density / (abund * hydrogen_density)
		
		read_previous_flag = 0
		if (verbose == 3) then
			write(*,*) 'Read previous calculation : ', read_previous_flag
		endif
		
		filename_previous = 'caca_old'
		if (verbose == 3 .and. read_previous_flag == 1) then
			write(*,*) 'File with the previous calculation : ', filename_previous
		endif
				
		if (verbose == 3) then
			write(*,*) 'Starting solution : ', start_mode
		endif
				
		if (verbose == 3) then
			write(*,*) 'Numerical scheme : ', which_scheme
		endif
		
		close(15)
		
		if (verbose == 3) then
			write(*,*) 'Creating atmosphere...'
		endif
		
! When resetting, dz is not used. For this reason, we can pass the array
! without being still allocated
		call grid(.TRUE.)
		
! Allocate memory for all the variables depending on the size of the atmosphere
		if (allocated(B)) deallocate(B)
		allocate(B(nz))
		
		if (allocated(chil)) deallocate(chil)
		allocate(chil(nz))
		
		if (allocated(kappa)) deallocate(kappa)
		allocate(kappa(nz))
		
		if (allocated(chie)) deallocate(chie)
		allocate(chie(nz))
		
		if (allocated(Sl)) deallocate(Sl)
		allocate(Sl(nz))
		
		if (allocated(Lstar)) deallocate(Lstar)		
		allocate(Lstar(nz))
		
		if (allocated(Jbar)) deallocate(Jbar)		
		allocate(Jbar(nz))
		
		if (allocated(flux)) deallocate(flux)		
		allocate(flux(nz))
		
		if (allocated(temperature)) deallocate(temperature)
		allocate(temperature(nz))
		
		if (allocated(nh)) deallocate(nh)
		allocate(nh(nz))
		
		if (allocated(abundance)) deallocate(abundance)
		allocate(abundance(nz))
		

! Calculate the weigths for the angular and frequency integration
      call ang_freq_weights
		
		if (allocated(Jbar_total)) deallocate(Jbar_total)
		allocate(Jbar_total(nz*nact))
		
		if (allocated(dJbar_totaldn)) deallocate(dJbar_totaldn)
		allocate(dJbar_totaldn(nz*nact,nz*nl))
		
		if (allocated(dJbardn)) deallocate(dJbardn)
		allocate(dJbardn(nz,nz*nl))
		
		if (allocated(Lstar_total)) deallocate(Lstar_total)
		allocate(lstar_total(nz*nact))
		
		if (allocated(flux_total)) deallocate(flux_total)
		allocate(flux_total(nz*nact))
		
		if (allocated(pop)) deallocate(pop)
		allocate(pop(nl*nz))
		
		if (allocated(popold)) deallocate(popold)
		allocate(popold(nl*nz))
		
		if (allocated(pop_previous_regrid)) deallocate(pop_previous_regrid)
		allocate(pop_previous_regrid(nl*nz))
		
		if (allocated(popl)) deallocate(popl)
		allocate(popl(nl*nz))
		
		if (allocated(tau)) deallocate(tau)
		allocate(tau(nact,0:nz))
		
		if (allocated(dopplerw)) deallocate(dopplerw)
		allocate(dopplerw(nt,nz))
		
		if (allocated(dSldn)) deallocate(dSldn)
		allocate(dSldn(nz,nl*nz))
		
		if (allocated(dchildn)) deallocate(dchildn)
		allocate(dchildn(nz,nl*nz))
		
		if (allocated(dtaudn)) deallocate(dtaudn)
		allocate(dtaudn(0:nz,nl*nz))
		
		if (verbose == 3) then
			write(*,*) 'Defining physical properties of the atmosphere...'
		endif
		
		call atph(.TRUE.)
		
		kappa = 0.d0
		chie = 0.d0

! Initialize in LTE
		if (verbose == 3) then
			write(*,*) 'Initializing in LTE...'
		endif		
						
		if (verbose == 3) then
			write(*,*) 'End of initialization'		
		endif
					
   end subroutine init

      
! *********************************************************
! *********************************************************
! ROUTINES FOR THE ATOMIC MODEL
! *********************************************************
! *********************************************************

	
!----------------------------------------------------------
!     DEFINE ATOMIC DATA
! 
! OUTPUT: 
!         INTEGER:
!         NL=totalnumber of levels , NI=Number of ions
!         NT=Number of transitions
!         NACT=Number of active radiative transitions
!         NR=Number of active+passive radiative transitions
!         NLI(NL)=ion to which each level belongs
!         ITRAN(2,nt)=Up/down levels in each transition
!         REAL*8:
!         ABUND= Abundance 
!         DLEVEL(1,nl)=Frecuency of every level
!       
!         DLEVEL(2,nl)=g=Pesos estadisticos
!
!         DION(1,ni)=XI=Ionization frequency
!         DTRAN(1,nt)=Radiative cross section=fij*pi*(e**2)/me/c
!         DTRAN(2,nt)=frecuency of the transition
!         DTRAN(3,nt)=Collisional cross section
!         DTRAN(4,nt)=Radiation Temperature OR Doppler Width
!----------------------------------------------------------

! ---------------------------------------------------------
! Initialize the integration weights in angle and frequency
! ---------------------------------------------------------
	subroutine ang_freq_weights
	real(kind=8) :: div
	real(kind=8), allocatable :: x_quadr(:), w_quadr(:)
	integer :: i
				      
      n_quadr_beta = 80
      if (.not. allocated(x_e3)) allocate(x_e3(n_quadr_beta))
		if (.not. allocated(w_e3)) allocate(w_e3(n_quadr_beta))
      call gauleg(-7.d0,7.d0,x_e3,w_e3,n_quadr_beta)
            		
	end subroutine ang_freq_weights

		
! ---------------------------------------------------------
! Atmosphere definition
! ---------------------------------------------------------		
	subroutine atph(reset)
	logical :: reset, stat
	integer :: i, j, k, npoints_local, loop, which_largest
	real(kind=8) :: temp1, temp2, temp3, deltaz, vth
	integer :: factor
	real(kind=8), allocatable :: temperature_temp(:)
	real(kind=8), allocatable :: nh_temp(:), abund_temp(:), linewidth_temp(:)
	real(kind=8), allocatable :: dz_temp(:)
							
! If no file with the physical conditions is used, then fill the atmosphere with
! the constant temperature and hydrogen density
		if (trim(adjustl(file_physical_conditions)) == 'none') then
			do i = 1, nz
				temperature(i) = tempc
				nh(i) = hydrogen_density
				abundance(i) = abund

! Doppler widths
				do j = 1, nt
					vth = sqrt(2.d0*PK*tempc/(molmass*UMA))					
					if (vth < vmicrot) then
						vth = vmicrot
					endif
					dopplerw(j,i) = dtran(2,j) * vth / PC
				enddo
				
			enddo
		else
! Physical conditions in a file
			inquire(file=trim(adjustl(file_physical_conditions)),exist=stat)
      	if (.not.stat) then
      		print *, 'Error when opening file ', trim(adjustl(file_physical_conditions))
      		stop
      	endif
			open(unit=45,file=trim(adjustl(file_physical_conditions)),&
				action='read',status='old')
			call Pass_Header(45)
			npoints_local = rdinp(.TRUE.,45,16)
			read(45,*)
			read(45,*)			
			read(45,*)
			
			col_density = 0.d0
			radius = 0.d0
						
! DIVIDE INTO TWO ZONES EACH ZONE			
			allocate(dz_temp(npoints_local))
			allocate(nh_temp(npoints_local))
			allocate(abund_temp(npoints_local))
			allocate(linewidth_temp(npoints_local))
			allocate(temperature_temp(npoints_local))
			
! 			allocate(physical_conditions_columns(4+n_columns_colliders))
			
! LEER AQUI LAS COLUMNAS Y METERLAS EN dz_temp, etc.

			do i = 1, npoints_local
! 				read(45,*) (physical_conditions_columns(j),j=1,4+n_columns_colliders)
				read(45,*) dz_temp(i), nh_temp(i), temperature_temp(i), abund_temp(i),&
					linewidth_temp(i)
				linewidth_temp(i) = linewidth_temp(i) * 1.d5    ! km/s to  cm/s		
			enddo
			
			col_density = col_density + sum(dz_temp * abund_temp * nh_temp)
			radius = radius + sum(dz_temp)
									
! If resetting, then just allocate memory and put the temperature and nh arrays equal to the
! ones in the file
			if (reset) then
				temperature = temperature_temp
				nh = nh_temp
				abundance = abund_temp
				collis_all = collis_all_original

! Doppler widths
				do i = 1, npoints_local
					do j = 1, nt
						vth = sqrt(2.d0*PK*temperature(i)/(molmass*UMA))						
						if (vth < linewidth_temp(i)) then
							vth = linewidth_temp(i)
						endif
						dopplerw(j,i) = dtran(2,j) * vth / PC						
					enddo					
				enddo

			else
				
				factor = nz / npoints_local
								
! And then divide all zones into "factor" zones
				loop = 1
				do i = 1, npoints_local
					do j = 1, factor
						nh(loop) = nh_temp(i)
						temperature(loop) = temperature_temp(i)
						abundance(loop) = abund_temp(i)						
						collis_all(:,loop) = collis_all_original(:,i)
						
! Doppler widths
						do k = 1, nt
							vth = sqrt(2.d0*PK*temperature_temp(i)/(molmass*UMA))
							if (vth < linewidth_temp(i)) then
								vth = linewidth_temp(i)
							endif
							dopplerw(k,loop) = dtran(2,k) * vth / PC
						enddo
				
						loop = loop + 1
					enddo
				enddo
								
			endif
			
			deallocate(dz_temp)
			deallocate(nh_temp)
			deallocate(abund_temp)
			deallocate(temperature_temp)
									
			close(45)
		endif
		
	end subroutine atph

!-----------------------------------------------------------------
! Returns the number of points and sets the z axis
! tau(i) = tau(i-1) * taufac
!-----------------------------------------------------------------		
	subroutine grid(reset)
	logical :: reset, stat
	real(kind=8) :: deltaz, ztemp, tautemp, tau0
	integer :: i, j, npoints_local, loop, which_largest, factor
	real(kind=8), allocatable :: dz_temp(:), dz_old(:)
		
! Constant physical conditions
		if (trim(adjustl(file_physical_conditions)) == 'none') then
					
! First calculate the number of zones in the atmosphere
			nz = min(npoints,n_ptos)					
		
! Allocate memory for the vector of deltaz		
			if (allocated(dz)) deallocate(dz)
			allocate(dz(nz))
		
! In this case, col_density is the column density -> N(O) = n(O) * l -> l = N(O) / n(O)		
			deltaz = col_density / (abund * hydrogen_density) / nz
						
			if (verbose == 1 .and. .not.reset) then
				print *, '  Number of zones : ', nz, ' -- deltaz : ', deltaz
			endif
		
			i = 1
			do while (i <= nz)
				dz(i) = deltaz
				i = i + 1
			enddo
		else
		
! Physical conditions in a file
			inquire(file=trim(adjustl(file_physical_conditions)),exist=stat)
      	if (.not.stat) then
      		print *, 'Error when opening file ', trim(adjustl(file_physical_conditions))
      		stop
      	endif
			open(unit=45,file=trim(adjustl(file_physical_conditions)),&
				action='read',status='old')
			call Pass_Header(45)
			npoints_local = rdinp(.TRUE.,45,16)
			read(45,*)
			read(45,*)			
			read(45,*)
									
			if (reset) then				
				npoints = npoints_local
			endif
									
! ! DOUBLE THE NUMBER OF POINTS
! ! For the moment, read only the width of each zone
! 			nz = min(npoints,n_ptos)
! 			factor = nz / npoints_local
! 			allocate(dz(nz))
! 									
! 			loop = 1
! 			do i = 1, npoints_local
! 				read(45,*) deltaz
! 				do j = 1, factor			
! 					dz(loop) = deltaz / factor
! 					loop = loop + 1
! 				enddo
! 			enddo

! DIVIDE INTO TWO ZONES EACH ZONE
			allocate(dz_temp(npoints_local))
			do i = 1, npoints_local
				read(45,*) dz_temp(i)
			enddo
			
! If resetting, then just allocate memory and put the dz array equal to the
! one in the file
			if (reset) then
				nz = min(npoints,n_ptos)
				if (allocated(dz)) deallocate(dz)
				allocate(dz(nz))
				dz = dz_temp
			else

				nz = min(npoints,n_ptos)
				
				if (allocated(dz)) deallocate(dz)
				allocate(dz(nz))
				
				factor = nz / npoints_local
				
! And then divide all zones into "factor" zones
				loop = 1
				do i = 1, npoints_local
					do j = 1, factor
						dz(loop) = dz_temp(i) / factor
						loop = loop + 1
					enddo
				enddo
					
			endif
			
			deallocate(dz_temp)
				
			close(45)
		endif

	end subroutine grid
	
!-----------------------------------------------------------------
! Carry out the regridding. It doubles the number of grid points and
! generates all the structures again
!-----------------------------------------------------------------		
	subroutine regrid(reset, messagesOn)
	logical :: reset, messagesOn
	real(kind=8) :: deltaz, ztemp, tautemp, tau0
	real(kind=8), allocatable :: pop_previous_step(:), dz_before(:), pop_spline(:)
	real(kind=8), allocatable :: pop_before(:), pop_after(:), dz_after(:), dz_previous_step(:)
	integer :: i, ip, ip2, nz_previous_step
						
		nz_previous_step = nz
		
! Save the populations now to be injected later into the new grid to start
! from a better solution			
		allocate(pop_previous_step(nl*nz_previous_step))
		allocate(dz_previous_step(nz_previous_step))
		pop_previous_step = pop
		dz_previous_step = dz
				
! First deallocate all arrays depending on nz
		if (allocated(B)) deallocate(B)
		if (allocated(chil)) deallocate(chil)
		if (allocated(kappa)) deallocate(kappa)
		if (allocated(chie)) deallocate(chie)
		if (allocated(Sl)) deallocate(Sl)
		if (allocated(Lstar)) deallocate(Lstar)
		if (allocated(Jbar)) deallocate(Jbar)
		if (allocated(flux)) deallocate(flux)
		if (allocated(temperature)) deallocate(temperature)
		if (allocated(nh)) deallocate(nh)
		if (allocated(abundance)) deallocate(abundance)
		if (allocated(Jbar_total)) deallocate(Jbar_total)
		if (allocated(dJbar_totaldn)) deallocate(dJbar_totaldn)
		if (allocated(dJbardn)) deallocate(dJbardn)
		if (allocated(lstar_total)) deallocate(lstar_total)
		if (allocated(flux_total)) deallocate(flux_total)
		if (allocated(pop)) deallocate(pop)
		if (allocated(popold)) deallocate(popold)
		if (allocated(popl)) deallocate(popl)
		if (allocated(pop_previous_regrid)) deallocate(pop_previous_regrid)
		if (allocated(tau)) deallocate(tau)		
		if (allocated(dSldn)) deallocate(dSldn)
		if (allocated(dchildn)) deallocate(dchildn)
		if (allocated(dtaudn)) deallocate(dtaudn)
		if (allocated(dz)) deallocate(dz)
		if (allocated(collis_all)) deallocate(collis_all)
		if (allocated(dopplerw)) deallocate(dopplerw)
		
! Change the number of zones. If reset is set, then put the initial number of grid points
		if (reset) then
			npoints = npoints_initial
			if (messagesOn) then
				if (verbose == 1) then
					write(*,*) '*************************'
					write(*,*) ' --- RESET REGRIDDING ---'
					write(*,*) '*************************'
				endif
			endif
		else
		
! If we are using a file with the physical conditions, divide all zones
! into one more zone for the regridding.
			if (trim(adjustl(file_physical_conditions)) /= 'none') then
				npoints = nz + npoints_initial
			else
				npoints = nz + npoints_initial
			endif
			if (verbose == 1) then
				write(*,FMT='(A,I4,A,I4,A)') ' --- REGRIDDING from ', nz_previous_step, &
					' to ', npoints, ' zones ---'
			endif
		endif
				
		call grid(reset)
			
! Allocate memory for all the variables depending on the size of the atmosphere	
		allocate(B(nz))
		allocate(chil(nz))
		allocate(kappa(nz))
		allocate(chie(nz))
		allocate(Sl(nz))
		allocate(Lstar(nz))
		allocate(Jbar(nz))
		allocate(flux(nz))
		allocate(temperature(nz))
		allocate(nh(nz))
		allocate(abundance(nz))
		allocate(Jbar_total(nz*nact))
		allocate(dJbar_totaldn(nz*nact,nz*nl))
		allocate(dJbardn(nz,nz*nl))
		allocate(lstar_total(nz*nact))
		allocate(flux_total(nz*nact))		
		allocate(pop(nl*nz))
		allocate(popold(nl*nz))
		allocate(pop_previous_regrid(nl*nz))
		allocate(popl(nl*nz))
		allocate(tau(nact,0:nz))		
		allocate(dSldn(nz,nl*nz))
		allocate(dchildn(nz,nl*nz))
		allocate(dtaudn(0:nz,nl*nz))
		allocate(collis_all(nt,nz))
		allocate(dopplerw(nt,nz))
				
		call atph(reset)
		
		kappa = 0.d0
		chie = 0.d0
		
! Recalculate the LTE populations	
		call poplte
		
! Now fill the populations in the new grid by interpolating from the
! previous grid
		if (reset) then
			pop = popl
		else
			allocate(pop_after(nz))
			allocate(dz_after(nz))
			
			allocate(dz_before(nz_previous_step))
			allocate(pop_before(nz_previous_step))
			allocate(pop_spline(nz_previous_step))
		
			dz_before = 0.d0
			dz_before(1) = dz_previous_step(1)
			do i = 2, nz_previous_step
				dz_before(i) = dz_before(i-1) + dz_previous_step(i)
			enddo
		
! Generate an axis with the depth on the slab		
			dz_after = 0.d0
			dz_after(1) = dz(1)
			do i = 2, nz
				dz_after(i) = dz_after(i-1) + dz(i)
			enddo
		
! Now fill the populations in the new grid by interpolating from the
! previous grid
			do i = 1, nl			
! Generate a vector with the population of level i at each depth			
				do ip = 1, nz_previous_step
					pop_before(ip) = pop_previous_step(i+nl*(ip-1))
				enddo
						
! Interpolate it to the new grid

! Spline interpolation
! 				call splin1(dz_before,pop_before,1.d30,1.d30,pop_spline)
! 				call spline(dz_before,pop_before,pop_spline,dz_after,pop_after)
				
! Linear	interpolation
  				call lin_interpol(dz_before,pop_before,dz_after,pop_after)

! And save it into the population vector
				do ip = 1, nz
					pop(i+nl*(ip-1)) = pop_after(ip)				
				enddo
			enddo
			
			if (allocated(dz_before)) deallocate(dz_before)
			if (allocated(pop_before)) deallocate(pop_before)
			if (allocated(pop_spline)) deallocate(pop_spline)
			if (allocated(dz_after)) deallocate(dz_after)
			if (allocated(pop_after)) deallocate(pop_after)
			
		endif
		
! Save the interpolated populations of the previous grid for comparison with the new solution		
		pop_previous_regrid = pop
		
		deallocate(pop_previous_step)
		deallocate(dz_previous_step)
								
	end subroutine regrid
				
! *********************************************************
! *********************************************************
! GENERAL ROUTINES
! *********************************************************
! *********************************************************
	
! ---------------------------------------------------------
! Put the LTE populations
! ---------------------------------------------------------	
	subroutine poplte
	real(kind=8) :: u(ni), fi(ni), fl(nl), sum ,kte, tp, fac, tot
	integer :: ip, i, ipl
	
		do ip = 1, nz
			ipl = nl * (ip-1)
			tp = temperature(ip)
			kte = PHK / tp
			
			u = 0.d0

! Partition function
			do i = 1, nl
				fl(i) = dlevel(2,i) * dexp(-dlevel(1,i) * kte)  !g*exp(-h*nu/(k*T))
				u(nli(i)) = u(nli(i)) + fl(i)
			enddo
			
! If we are in a multi-ionic atom
			if (ni > 1) then
				
				do i = 1, nl
					fl(i) = fl(i) / u(nli(i))
				enddo
				
				fac = 1.d0 * ci / (tp * dsqrt(tp))
				
				do i = 1, ni-1
					fi(i) = fac * u(i) / u(i+1) * dexp(dion(1,i+1)*kte)
				enddo
				
				fi(ni) = 1.d0
				tot = fi(ni)
				
				do i = ni - 1, 1, -1
					fi(i) = fi(i) * fi(i+1)
					tot = tot + fi(i)
				enddo
				tot = factor_abundance * abundance(ip) *  nh(ip) / tot				
				do i = 1, nl
					popl(i+ipl) = fl(i) * fi(nli(i)) * tot
				enddo
			else
				tot = factor_abundance * abundance(ip) * nh(ip) / u(1)				
				do i = 1, nl
					popl(i + ipl) = fl(i) * tot
				enddo
			endif						
			
		enddo 
	end subroutine poplte

! ---------------------------------------------------------
! Read data from a previous file
! ---------------------------------------------------------
	subroutine read_previous(filename_previous)
	character(len=80) :: filename_previous
	integer :: nl_old, nz_old, ip, ipl, i
	real(kind=8), allocatable :: z_old(:), z_new(:), pop_old(:), pop1(:), pop2(:)
	real(kind=8) :: temp, r_old, deltaz_new, deltaz_old
	logical :: stat
	
		inquire(file=filename_previous,exist=stat)
      if (.not.stat) then
      	print *, 'Error when opening file ', filename_previous
      	stop
      endif
		open(unit=20,file=filename_previous,status='old',action='read')
		
		call lb(20,6)
		read(20,*) nl_old
		call lb(20,1)
		read(20,*) nz_old
		print *, 'Previous/new number of levels : ', nl_old, nl
		print *, 'Previous/new number of points : ', nz_old, nz
		
		call lb(20,7)
		read(20,*) r_old
		
		print *, 'Previous/new radius : ', r_old, radius
				
		deltaz_old = r_old / nz_old
		deltaz_new = radius / nz
		
		print *, 'Previous/new dz : ', deltaz_old, deltaz_new
		
		allocate(z_old(nz_old))
		allocate(z_new(nz))
		allocate(pop_old(nz_old*nl_old))
		allocate(pop1(nz_old))
		allocate(pop2(nz))
		
		call lb(20,3)
		call lb(20,nz_old)
		
		z_old(1) = 0.d0
		z_new(1) = 0.d0
		
		do ip = 2, nz_old
			z_old(ip) = z_old(ip-1) + deltaz_old
		enddo
		
		do ip = 2, nz
			z_new(ip) = z_new(ip-1) + deltaz_new
		enddo
		
		call lb(20,3)

! Read the old populations		
		do ip = 1, nz_old
			ipl = nl_old * (ip-1)
			read(20,*) temp, (pop_old(i+ipl), i = 1, nl_old)
		enddo

! Now do the interpolation into the new z axis		
		do i = 1, nl_old
			do ip = 1, nz_old
				ipl = nl_old*(ip-1)
				pop1(ip) = pop_old(i+ipl)
			enddo
			call lin_interpol(z_old,pop1,z_new,pop2)
			do ip = 1, nz
				ipl = nl*(ip-1)
				pop(i+ipl) = pop2(ip)
			enddo
		enddo
		
		close(20)
		
		deallocate(z_old)
		deallocate(z_new)
		deallocate(pop_old)
		deallocate(pop1)
		deallocate(pop2)
		
	end subroutine read_previous
	

! *********************************************************
! *********************************************************
! RATE EQUATIONS ROUTINES
! *********************************************************
! *********************************************************

!-----------------------------------------------------------------
! Put the collisional rates in the matrix which is passed as a parameter for the point ip in the atmosphere
! C(i,j) :  downward collisional rate (i>j)
! In the subroutine which writes the rate equations, this matrix is transposed : 
! C(low,up) goes in A(up,low)
!-----------------------------------------------------------------
	subroutine collis(ip,C)
	integer, INTENT(IN) :: ip
	real(kind=8), INTENT(INOUT) :: C(nl, nl)
	integer :: it, up, low

! Constant physical conditions
		if (trim(adjustl(file_physical_conditions)) == 'none') then
			do it = 1, nt
				up = itran(1,it)
				low = itran(2,it)			

				C(up,low) = dtran(3,it)
			enddo
		else
			do it = 1, nt
				up = itran(1,it)
				low = itran(2,it)			

				C(up,low) = collis_all(it,ip) * nh(ip)
			enddo
		endif
				
	end subroutine collis

!-----------------------------------------------------------------
! Solves the rate equations for the point ip
! It is based on the preconditioning scheme proposed by Rybicki & Hummer (1992)
!-----------------------------------------------------------------
	subroutine rate_eq(ip, dxmax, dxmax_level)
	integer, INTENT(IN) :: ip
	real(kind=8), INTENT(INOUT) :: dxmax
	integer, INTENT(INOUT) :: dxmax_level
	integer :: low, up, ipl, ipt, i, j, il, it
	real(kind=8) :: A(nl, nl), B(nl), cul, acon, blu, glu, djbar, poptot, tol, U(nl,nl), V(nl,nl), w(nl), x(nl), wmin, wmax
	integer, allocatable :: indx(:)

! Offsets for the storing
		ipl = nl*(ip-1)
		ipt = nr*(ip-1)
		
		A = 0.d0
		
! Upward collisional rates A(low,up)
		call collis(ip,A)
		
! Calculate the downward rates using detailed balance and interchange positions
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
			acon = TWOHC2*dtran(2,it)**3  !acon = (2*h*nu**3)/c**2
			blu = PI4H * dtran(1,it) / dtran(2,it)
			glu = dlevel(2,low) / dlevel(2,up)
			
			djbar = Jbar_total(it+ipt) - Lstar_total(it+ipt) * acon * pop(up+ipl) * glu /&
					 ( pop(low+ipl) - glu * pop(up+ipl) )
			
			A(low,up) = A(low,up) + glu * blu * (acon * (1.d0 - Lstar_total(it+ipt)) + djbar )
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
		
! Conservation equation. We substitute the last equation
		B = 0.d0
		A(nl, 1:nl) = 1.d0
		
		poptot = factor_abundance * abundance(ip) * nh(ip)
		B(nl) = poptot
		
! Solve the linear system				
		allocate(indx(nl))
		tol = 1.d-10
		call ludcmp(A,indx,tol)
		call lubksb(A,indx,B)
		deallocate(indx)
		
! 		call svdcmp(a,nl,nl,nl,nl,w,v)
! 		
! 		wmax = maxval(w)
! 		wmin = wmax * 1.d-6
! 		do j = 1, nl
!  			if (w(j) < wmin) w(j) = 0.d0
! 		enddo
! 		call svbksb(a,w,v,nl,nl,nl,nl,B,x)
! 		B = x
		

! Maximum relative change
		do il = 1, nl
			if (dabs( B(il) - pop(il+ipl) ) / dabs(B(il)) > dxmax) then
				dxmax = dabs( B(il) - pop(il+ipl) ) / dabs(B(il))
				dxmax_level = il
			endif
			pop(il+ipl) = B(il)			
		enddo
	
	end subroutine rate_eq
		
!-----------------------------------------------------------------
! Do the population correction for all the points in the atmosphere
!-----------------------------------------------------------------
	subroutine correct_populations
	real(kind=8) :: relat_error_p
	integer :: ip
	
		relat_error_p = relat_error
		relat_error = 0.d0
		
		do ip = 1, nz
			call rate_eq(ip,relat_error,relat_error_level)			
		enddo	
		
	end subroutine correct_populations	   
   
end module functions_cep
