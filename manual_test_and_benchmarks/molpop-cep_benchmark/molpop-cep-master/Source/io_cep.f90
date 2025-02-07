module io_cep
use global_cep
use global_molpop, only : freq_axis, fin_tr, nprint, final, itr, jtr, wl, freq, ledet
use maths_cep
use maths_molpop, only : Tbr4I, Inv_plexp
implicit none
contains

!---------------------------------------------------------
! This subroutine reads the intermediate data written by MOLPOP
!---------------------------------------------------------
	subroutine read_intermediate_data
	integer :: i, j, up, low, deg, found, n_zones_slab, t1, t2

		verbose = 1
		
		open(unit=28,file='interface.dat',action='read',status='old')
		
! Read if there is a file with the physical conditions
		read(28,FMT='(A)') file_physical_conditions
		
! Read if there is a file with the physical conditions
		read(28,FMT='(A)') output_file
				
! Hydrogen density, abundance, thermal velocity and temperature (we assume that everything is constant for the moment)
		read(28,FMT='(6(E13.5,1X))') hydrogen_density, abund, vmicrot, vtherm, tempc, molmass
		
! Read the algorithm for the calculation of the auxiliary functions
		read(28,FMT='(I1)') escape_prob_algorithm
		
! If the thermal velocity is smaller than the microturbulent velocity, use the microturbulent
		if (vtherm < vmicrot) then
			vtherm = vmicrot
		endif
		
! Read data concerning the strategy to follow
		read(28,FMT='(E13.5,2X,E13.5,2X,E13.5,2X,I1,2X,E13.5,2X,I4,2X,F6.3)') tau_threshold,&
			r_threshold, col_threshold, kthick_strategy, precision, n_iters, mu_output
		
! Start from optically thin		
		if (kthick_strategy == 0) then
			if (verbose == 2) then
				write(*,*) 'Using increasing strategy...'
				write(*,*) 'Initial optical depths smaller than ', tau_threshold
				write(*,*) 'Stopping when radius is larger than ', r_threshold
				write(*,*) 'Stopping when column density is larger than ', col_threshold
			endif
			start_mode = 1
		endif
		
! Start from optically thick
		if (kthick_strategy == 1) then
			if (verbose == 2) then
				write(*,*) 'Using decreasing strategy...'
				write(*,*) 'Initial optical depths larger than ', tau_threshold
				write(*,*) 'Stopping when radius is smaller than ', r_threshold
				write(*,*) 'Stopping when column density is smaller than ', col_threshold
			endif
			start_mode = 0
		endif
		
! Solve only one problem
		if (kthick_strategy == 2) then
			if (verbose == 2) then
				write(*,*) 'Solving for a fixed column density'
			endif
			start_mode = 0
		endif
		
! Number of zones
		read(28,*) cep_precision, printings_per_decade, newton_maximum_iter, nInitialZones
		
! Initial number of zones
		npoints = nInitialZones		

! Initial number of points for all calculations
		npoints_initial = npoints
				
! Number of levels
		read(28,*) nl
		
		if (allocated(dlevel)) deallocate(dlevel)
		allocate(dlevel(2,nl))

! Read level's information		
		do i = 1, nl
			read(28,*) dlevel(2,i), dlevel(1,i)
			dlevel(1,i) = dlevel(1,i) * PC
		enddo
		
! Number of transitions
		read(28,*) nt, nact
		nr = nact
		
		ni = 1
		
		if (allocated(nli)) deallocate(nli)
		allocate(nli(nl))
		
		if (allocated(itran)) deallocate(itran)
		allocate(itran(2,nt))
		
		if (allocated(dion)) deallocate(dion)
		allocate(dion(1,1))
		
		if (allocated(dtran)) deallocate(dtran)
		allocate(dtran(7,nt))
		
		nli = 1
		
! Read transition's information
! dtran(5,i) is the value of J_internal given by MOLPOP
! dtran(6,i) is the value of J_boundary tau0 given by MOLPOP
! dtran(7,i) is the value of J_boundary tauT given by MOLPOP
		do i = 1, nact
			read(28,*) itran(1,i), itran(2,i), dtran(2,i), dtran(1,i), dtran(3,i), &
				dtran(5,i), dtran(6,i), dtran(7,i)
			up = itran(1,i)
			low = itran(2,i)
			dtran(1,i) = dtran(1,i) / (pi8c2*dtran(2,i)**2)*(dlevel(2,up)/dlevel(2,low))
			dtran(3,i) = dtran(3,i) * hydrogen_density
			dtran(4,i) = dtran(2,i) * vtherm / PC
		enddo
		
		do i = nact+1, nt
			read(28,*) itran(1,i), itran(2,i), dtran(2,i), dtran(3,i), dtran(5,i), dtran(6,i)
			dtran(3,i) = dtran(3,i) * hydrogen_density
			dtran(4,i) = dtran(2,i) * vtherm / PC
		enddo
				
! Read transitions where to compute output and locate which transitions they are
		read(28,FMT='(I4)') ntran_output
		allocate(upper_output(ntran_output))
		allocate(lower_output(ntran_output))
		if (allocated(output_transition)) deallocate(output_transition)
		allocate(output_transition(ntran_output))
		do i = 1, ntran_output
			read(28,FMT='(I4,2X,I4)') upper_output(i), lower_output(i)
			
			found = 0
			do j = 1, nact
				up = itran(1,j)
				low = itran(2,j)
				if (up == upper_output(i) .and. low == lower_output(i)) then
					found = j
				endif
			enddo
			output_transition(i) = found
		enddo
		
		deallocate(upper_output)
		deallocate(lower_output)
			
! And finally, collisional rates for all zones in case physical conditions vary
		if (trim(adjustl(file_physical_conditions)) /= 'none') then
			read(28,*) n_zones_slab
			
			if (allocated(collis_all)) deallocate(collis_all)
			allocate(collis_all(nt,n_zones_slab))
			
			if (allocated(collis_all_original)) deallocate(collis_all_original)
			allocate(collis_all_original(nt,n_zones_slab))
			
			do i = 1, nt
				read(28,*) t1, t2, (collis_all(i,j),j=1,n_zones_slab)
			enddo
			collis_all_original = collis_all
		endif
				
		close(28)
		
! Delete the intermediate file
  		open(unit=28,file='interface.dat',action='read',status='old')
  		close(28,STATUS='delete')
		
	end subroutine read_intermediate_data
	
!---------------------------------------------------------
! This subroutine calculates the flux for every transition using the coupled escape probability method
!---------------------------------------------------------
   subroutine calcflux_cep(x, flux_out, it)
	real(kind=8) :: x(:), flux_out(:,:)
	integer :: up, low, it, ip, ipl, ipt, ip1, npmip1, i
	real(kind=8) :: sig, glu, acon, chim, chip, tau0, deltaz, chilp
	real(kind=8) :: line_profile, f
			
		flux_out = 0.d0

! Calculate line absorption coefficient, line source function and optical depths
		up = itran(1,it)
		low = itran(2,it)
		sig = dtran(1,it) !sig=(h*nu*Blu)/(4*pi)			 
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

! Calculate the flux at each wavelength (Eq. (77) in the notes)			
! 		do i = 1, size(freq_axis)
! 			do ip = 1, nz
! 				f = freq_axis(i) * maxval(dopplerw(it,:)) / dopplerw(it,ip)				
! ! 				line_profile = exp(-(freq_axis(i)/dopplerw(it,ip))**2) / sqrt(PI)
! 				line_profile = exp(-f**2) / sqrt(PI)

! ! Flux in the line
! 				flux_out(1,i) = flux_out(1,i) + 2.d0 * PI / dopplerw(it,ip)	* abs(Sl(ip)) * &
! 					(expint(3,abs(tau(it,nz)-tau(it,ip))*line_profile) - &
! 					expint(3,abs(tau(it,ip)-tau(it,0))*line_profile) + &
! 					expint(3,abs(tau(it,ip-1)-tau(it,0))*line_profile) - &
! 					expint(3,abs(tau(it,nz)-tau(it,ip-1))*line_profile))									

! ! Intensity at mu=1
! 				flux_out(2,i) = flux_out(2,i) + Sl(ip) * ( exp(-(tau(it,nz)-tau(it,ip))*line_profile / (1.d0)) - &
! 					exp(-(tau(it,nz)-tau(it,ip-1))*line_profile / (1.d0)) )
! ! Intensity at the desired mu
! 				flux_out(3,i) = flux_out(3,i) + Sl(ip) * ( exp(-(tau(it,nz)-tau(it,ip))*line_profile / (mu_output)) - &
! 					exp(-(tau(it,nz)-tau(it,ip-1))*line_profile / (mu_output)) )
! 			enddo
! 		enddo

		do ip = 1, nz
			do i = 1, size(freq_axis)			
				f = freq_axis(i) * maxval(dopplerw(it,:)) / dopplerw(it,ip)
				! f = freq_axis(i) / dopplerw(it,ip)
				line_profile = exp(-f**2) / sqrt(PI)

				flux_out(1,i) = flux_out(1,i) + 2.d0 * PI	* abs(Sl(ip)) * &
					(expint(3,abs(tau(it,nz)-tau(it,ip))*line_profile) - &
					expint(3,abs(tau(it,ip)-tau(it,0))*line_profile) + &
					expint(3,abs(tau(it,ip-1)-tau(it,0))*line_profile) - &
					expint(3,abs(tau(it,nz)-tau(it,ip-1))*line_profile))

! Intensity at mu=1
				flux_out(2,i) = flux_out(2,i) + Sl(ip) * ( exp(-(tau(it,nz)-tau(it,ip))*line_profile / (1.d0)) - &
					exp(-(tau(it,nz)-tau(it,ip-1))*line_profile / (1.d0)) ) 
! Intensity at the desired mu
				flux_out(3,i) = flux_out(3,i) + Sl(ip) * ( exp(-(tau(it,nz)-tau(it,ip))*line_profile / (mu_output)) - &
					exp(-(tau(it,nz)-tau(it,ip-1))*line_profile / (mu_output)) ) 

! Flux in the line times the Doppler width so that we can later integrate over the reduced wavelength axis
				flux_out(4,i) = flux_out(4,i) + 2.d0 * PI	* dopplerw(it,ip) * abs(Sl(ip)) * &
					(expint(3,abs(tau(it,nz)-tau(it,ip))*line_profile) - &
					expint(3,abs(tau(it,ip)-tau(it,0))*line_profile) + &
					expint(3,abs(tau(it,ip-1)-tau(it,0))*line_profile) - &
					expint(3,abs(tau(it,nz)-tau(it,ip-1))*line_profile))

! Intensity at mu=1
				flux_out(5,i) = flux_out(5,i) + Sl(ip) * ( exp(-(tau(it,nz)-tau(it,ip))*line_profile / (1.d0)) - &
					exp(-(tau(it,nz)-tau(it,ip-1))*line_profile / (1.d0)) ) * dopplerw(it,ip)
			enddo

		enddo
		
		if (tau(it,nz) < 0.d0) then
			flux_out(1,:) = 0.d0
		endif
		  
   end subroutine calcflux_cep
	
	
!-----------------------------------------------------------------	
! Print an error code using the FITS routine
!-----------------------------------------------------------------
! 	subroutine printerror_fits(status)
!    integer status
!    character(len=30) :: errtext
!    character(len=80) :: errmessage
! 
! 		if (status <= 0) return
! 
!       call ftgerr(status,errtext)
!       print *,'FITSIO Error Status =',status,': ',errtext
! 
!       call ftgmsg(errmessage)
!       do while (errmessage /= ' ')
! 			print *,errmessage
!          call ftgmsg(errmessage)
!       enddo
! 	end subroutine printerror_fits
! 	
! !-----------------------------------------------------------------	
! ! Delete a FITS file
! !-----------------------------------------------------------------
! 	subroutine deletefile_fits(filename,status)
! 	integer :: status,unit,blocksize
! 	character(len=*) :: filename
! 
! 		if (status > 0) return
! 
!       call ftgiou(unit,status)
! 
!       call ftopen(unit,filename,1,blocksize,status)
! 
!       if (status == 0) then
! 			call ftdelt(unit,status)
!       else if (status == 103) then
! 			status = 0
! 			call ftcmsg
!       else
! 			status = 0
! 			call ftcmsg
! 			call ftdelt(unit,status)
!       endif
! 
!       call ftfiou(unit, status)
! 	end subroutine deletefile_fits
! !-----------------------------------------------------------------	
! ! Write the results in a FITS file
! !-----------------------------------------------------------------
! 	subroutine write_fits
! 	integer :: status, unit, blocksize, naxis, bitpix, group, fpixel, nelements
! 	integer :: i, ip, ipl, readwrite
! 	character(len=40) :: filename		
! 	real(kind=8), allocatable :: pop_fits(:,:)
! 	integer, allocatable :: naxes(:)
! 	logical :: simple, extend
! 	
! 		status = 0
! 		filename = 'output.fits'
! 		
! 		call deletefile_fits(filename,status)
! 		
! 		call ftgiou(unit,status)
! 		
! 		blocksize = 1		
! 		call ftinit(unit,filename,blocksize,status)
! 		
! 		simple = .TRUE.
! 		bitpix = -64
! 		naxis = 2
! 		allocate(naxes(naxis))		
! 		naxes(1) = nl
! 		naxes(2) = nz
! 		extend = .TRUE.
! 		
! 		call ftphpr(unit,simple,bitpix,naxis,naxes,0,1,extend,status)
! 		
! 		group = 1
! 		fpixel = 1
! 		nelements = naxes(1)*naxes(2)
! 		
! 		allocate(pop_fits(nl,nz))
! 		do i = 1, nl
! 			do ip = 1, nz
! 				ipl = nl * (ip-1)
! 				pop_fits(i,ip) = pop(i+ipl)
! 			enddo
! 		enddo
! 		
! 		call ftpprd(unit,group,fpixel,nelements,pop_fits,status)
! 						
! 		call ftclos(unit, status)
!       call ftfiou(unit, status)
!       
!       if (status > 0) call printerror_fits(status)
!       
!       pop_fits = 0.d0
!       
!       status = 0
!       call ftgiou(unit,status)
!       readwrite = 0
!       call ftopen(unit,filename,readwrite,blocksize,status)
!       call ftgknj(unit,'NAXIS',1,2,naxes,nfound,status)
!       
!       call ftgpvd(unit,group,firstpix,nbuffer,nullval,buffer,anynull,status)
!       
!       deallocate(pop_fits)
!       
!       pause
! 
! 	end subroutine write_fits

!-----------------------------------------------------------------	
! Write the results
!-----------------------------------------------------------------
	subroutine write_results
	integer :: i, j, ip, ipl, ipt, it, up, low
	real(kind=8) :: sig, glu, acon, agnus, chilp, chilm, snlte, slte
	real(kind=8) :: sb(nz), source(nz), epsil_equiv, sum, total
	character(len=20) :: selected
	
! 		call write_fits
		
		write(16,*) '*****************************************************'
		write(16,*) '    CEP'
		write(16,*) ' GENERAL DATA'
		write(16,*) '*****************************************************'
		write(16,*) 'N. active transitions'
		write(16,*) nact
		write(16,*) 'N. levels'
		write(16,*) nl
! 		write(16,*) 'H2 density [cm^-3]'
! 		write(16,*) hydrogen_density
! 		write(16,*) 'Molecular density [cm^-3]'
! 		write(16,*) hydrogen_density * abund
! 		write(16,*) 'Microturbulent velocity [km/s]'
! 		write(16,*) vmicrot
! 		write(16,*) 'Thermal velocity [km/s]'
! 		write(16,*) vtherm
! 		write(16,*) 'Temperature [K]'
! 		write(16,*) tempc
		
! Level information
		write(16,*)
		write(16,*) '***********************'
		write(16,*) ' Level information   '
		write(16,*) ' index  g    Energy [cm^-1]  '
		write(16,*) '***********************'
		do i = 1, nl
			write(16,FMT='(I3,2X,F5.1,2X,F10.4)') i, dlevel(2,i), dlevel(1,i)/PC
		enddo
		
! Transition information
		write(16,*)
		write(16,*) '***********************'
		write(16,*) ' Transition information'
		write(16,*) ' up  low  freq [GHz]  Aul [s^-1]  detailed (y/n)'
		write(16,*) '***********************'				
		do i = 1, nact
			selected = ' no '
			do j = 1, ntran_output
				if (output_transition(j) == i) then
					selected = ' yes '
					continue
				endif
			enddo
			up = itran(1,i)
			low = itran(2,i)
			write(16,FMT='(I3,2X,I3,2X,F12.6,2X,1PE13.6,3X,A)') itran(1,i), itran(2,i), &
				dtran(2,i)/1.d9,&
				dtran(1,i)*(pi8c2*dtran(2,i)**2)/(dlevel(2,up)/dlevel(2,low)), selected							
		enddo
				
		write(16,*)		
		
! Write number of frequency points in flux file
		write(32,*) 'N. wavelength points'
		write(32,*) 100
		write(32,*)		
		
		write(34,*) '*****************************************************'
		write(34,*) ' SLAB PARTITIONING'
		write(34,*) '*****************************************************'
		write(34,*) 'N. final zones'
		write(34,*) nz
		write(34,*)
		write(34,*) '   dz [cm]        nh [cm^-3]          T [K]         Abundance  '
		do i = 1, nz
			write(34,FMT='(1PE13.6,4X,1PE13.6,4X,1PE13.6,4X,1PE13.6)') dz(i), nh(i), temperature(i),&
				abundance(i)
		enddo
				
	end subroutine write_results
	
	
!-----------------------------------------------------------------	
! Write the results
!-----------------------------------------------------------------
	subroutine write_intermediate_results(x,xlte)
	real(kind=8) :: x(:), xlte(:), column_density, total_radius
	integer :: ii, ip, ipl, i, j, up, low, it, k
	real(kind=8) :: sig, glu, acon, chim, chip, tau0, deltaz, chilp, col_up, col_low
	real(kind=8) :: sum, intensity, e1, e2, freq_max, deltaTBR, deltaTBRJ
	real(kind=8), allocatable :: flux_out(:,:), Slte(:), total_flux(:), total_intensity(:)
	
		column_density = sum(dz * factor_abundance * abundance * nh)
		total_radius = sum(dz)

! Write output flux		
		allocate(flux_out(5,100))
		allocate(total_flux(ntran_output))
		allocate(total_intensity(ntran_output))
						
		do i = 1, ntran_output			
			it = output_transition(i)			
			
! Generate a frequency axis for this transition so that it can accomodate at least
! four Doppler widths for the zone with the largest Doppler width
! 			freq_max = 4.d0 * maxval(dopplerw(it,:))
! 			do ip = 1, 100
! 				freq_axis(ip) = (ip-1.d0) / 99.d0 * 2.d0 * freq_max - freq_max
! 			enddo
			
			call calcflux_cep(pop, flux_out, it)
			
			total_flux(i) = int_tabulated(freq_axis * maxval(dopplerw(it,:)), flux_out(1,:))
			total_intensity(i) = int_tabulated(freq_axis * maxval(dopplerw(it,:)), flux_out(2,:))
			
! Output in velocity [km/s] and emergent flux
			write(32,FMT='(A,I3,A,I3)') 'Transition : ', itran(1,it), ' -> ', itran(2,it)
			write(32,*) '    v (km/s)       flux          I(mu=1)   I(selected mu)'
			do j = 1, 100
				!write(32,FMT='(5(2X,1PE12.5))') freq_axis(j) / dtran(2,it) * PC / 1.d5, (flux_out(k,j),k=1,3)
				write(32,FMT='(5(2X,1PE12.5))') freq_axis(j), (flux_out(k,j),k=1,3)
			enddo
			write(32,*)
		enddo
				
! 		deallocate(freq_axis)
		deallocate(flux_out)
 
		write(16,*)
		write(16,*) '*******************************************************************************'
		write(16,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2/kms]: ', column_density, '   -  nzones: ', nz
		write(16,*) '*******************************************************************************'
		
		write(16,FMT='(1X,A,F6.3,A)') ' up    low     tau (line center)     cooling         N(up)/N(low)    I_lc(mu=',&
			mu_output,')   delta(Tb) [K]      del(TRJ) [K]    Int(Tb dv)    % emission'
			
		write(16,FMT='(A,A)') '                                                                        [W/m2/Hz/sr]',&
			'        [K]               [K]           [K km/s]              '
						
		write(17,FMT='(1X,A,F6.3,A)') ' up    low     tau (line center)     cooling         N(up)/N(low)    I_lc(mu=',&
			mu_output,')   delta(Tb) [K]      del(TRJ) [K]    Int(Tb dv) [K km/s]   % emission'
			
		write(17,FMT='(A,A)') '                                                                        [W/m2/Hz/sr]',&
			'        [K]               [K]           [K km/s]              '

		allocate(Slte(nz))
		
! Write populations
		write(35,*) '**********************************************************'
		write(35,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2/kms]: ', column_density, '    -  nzones: ', nz 
		write(35,*) '**********************************************************'
		write(35,FMT='(A)') 'Upper  Lower   Trad (all zones)'

		
		do i = 1, ntran_output
			it = output_transition(i)
			up = itran(1,it)
			low = itran(2,it)

			sig = dtran(1,it) !sig=(h*nu*Blu)/(4*pi)
			glu = dlevel(2,low) / dlevel(2,up)
			acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2			

			tau0 = 0.d0
			chip = 0.d0
			chim = 0.d0

			tau(it,0) = 0.d0
			col_up = 0.d0
			col_low = 0.d0
			do ip = 1, nz
				ipl = nl * (ip-1)

				chil(ip) = sig / dopplerw(i,ip) * (x(low+ipl) - glu * x(up+ipl))
				col_up = col_up + x(up+ipl) * dz(ip)
				col_low = col_low + x(low+ipl) * dz(ip)
				Sl(ip) = acon * glu * x(up+ipl) / (x(low+ipl) - glu * x(up+ipl))
				Slte(ip) = acon * glu * xlte(up+ipl) / (xlte(low+ipl) - glu * xlte(up+ipl))

         	if (ip == 1) then
					tau(it,ip) = chil(ip) * dz(ip)					
         	else
					tau(it,ip) = tau(it,ip-1) + chil(ip) * dz(ip)
         	endif

			enddo
			
			intensity = 0.d0
			do ip = 1, nz
				e1 = exp(-(tau(it,nz)-tau(it,ip)) / (sqrt(PI)*mu_output))
				e2 = exp(-(tau(it,nz)-tau(it,ip-1)) / (sqrt(PI)*mu_output))
				intensity = intensity + Sl(ip) * (e1-e2)				
			enddo
						
! 			deltaTBR = Tbr_I(dtran(2,it),intensity,tau(it,nz)/sqrt(PI))
			call Tbr4I(dtran(2,it),intensity,tau(it,nz)/sqrt(PI),deltaTBR,deltaTBRJ)
			
			write(16,FMT='(I4,2X,I4,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PG13.5,5X,1PG13.5)') up, low, &
				tau(it,nz)/sqrt(PI), flux_total(it+nr*(nz-1)) * 4.d0*PI, col_up / col_low,&
				1e-3*intensity, deltaTBR, deltaTBRJ, (PC/dtran(2,i))**3 / (2.d0*PK*1d5) * total_intensity(i), &
				100.0 * total_flux(i) / sum(total_flux)
												
			write(17,FMT='(I4,2X,I4,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PE13.5,5X,1PG13.5,5X,1PG13.5)') up, low, &
				tau(it,nz)/sqrt(PI), flux_total(it+nr*(nz-1)) * 4.d0*PI, col_up / col_low,&
				1e-3*intensity, deltaTBR, deltaTBRJ, (PC/dtran(2,i))**3 / (2.d0*PK*1d5) * total_intensity(i), &
				100.0 * total_flux(i) / sum(total_flux)
				
			write(35,FMT='(I4,2X,I4,1X,(1P8E13.5))') up, low, &
				(PHK*dtran(2,i)/log(1.d0+acon/Sl(ii)), ii = 1, nz)
		enddo

! Write populations
		write(31,*) '**********************************************************'
		write(31,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2/kms]: ', column_density, '    -  nzones: ', nz 
		write(31,*) '**********************************************************'
		write(31,FMT='(A)') 'Level   n [cm^-3]        n/n*       n* [cm^-3]'
		do ip = 1, nz
			ipl = nl * (ip-1)
			write(31,FMT='(A,I4)') 'Zone ', ip			
			do ii = 1, nl
				write(31,FMT='(I4, 3(2X,1PE12.5))') ii, pop(ii+ipl), pop(ii+ipl)/popl(ii+ipl),&
					popl(ii+ipl)
			enddo
		enddo
				
		deallocate(Slte)
		deallocate(total_flux)
		
	end subroutine write_intermediate_results	

!-----------------------------------------------------------------	
! Write the results
!-----------------------------------------------------------------
	subroutine calculate_intermediate_results(x,xlte,colIndex)
	real(kind=8) :: x(:), xlte(:), column_density, total_radius
	integer :: ii, ip, ipl, i, j, up, low, it, k, colIndex
	real(kind=8) :: sig, glu, acon, chim, chip, tau0, deltaz, chilp, col_up, col_low, tmp2, tmp
	real(kind=8) :: z, intensity, e1, e2, freq_max, deltaTBR, deltaTBRJ, columnH2, column
	real(kind=8), allocatable :: flux_out(:,:), Slte(:), total_flux(:), total_intensity(:)
	
		column_density = sum(dz * factor_abundance * abundance * nh)
		total_radius = sum(dz)

! Write output flux		
		allocate(flux_out(5,100))
		allocate(total_flux(ntran_output))
		allocate(total_intensity(ntran_output))

! Write line fluxes
		write(32,*)
		write(32,*) '*******************************************************************************'
		write(32,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2]: ', column_density, '   -  nzones: ', nz
		write(32,*) '*******************************************************************************'

! Write excitation temperature
		write(35,*) '**********************************************************'
		write(35,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2]: ', column_density, '    -  nzones: ', nz 
		write(35,*) '**********************************************************'		
						
		do i = 1, ntran_output			
			it = output_transition(i)
			
! Generate a frequency axis for this transition so that it can accomodate at least
! four Doppler widths for the zone with the largest Doppler width

			! freq_max = 4.d0 * maxval(dopplerw(it,:))
			! do ip = 1, 100
				! freq_axis(ip) = (ip-1.d0) / 99.d0 * 2.d0 * freq_max - freq_max
			! enddo
			
			call calcflux_cep(pop, flux_out, it)

			! total_flux(i) = totFlux

			total_flux(i) = 1.d23 * int_tabulated(freq_axis, flux_out(4,:)) / maxval(dopplerw(it,:))
			total_intensity(i) = int_tabulated(freq_axis, flux_out(5,:)) / maxval(dopplerw(it,:))

! Now
! 1.d23 * int_tabulated(freq_axis, flux_out(1,:)) == 4.0*PI*1d23*flux_total(it+nr*(nz-1))
! where flux_out(1,:) contains dopplerw so that we can just integrate over the reduced wavelength axix
			! print *, total_flux(i), (4.0*PI*1d23*flux_total(it+nr*(nz-1))), total_flux(i) / (4.0*PI*1d23*flux_total(it+nr*(nz-1)))
			! stop

			
! Output in velocity [km/s] and emergent flux
			write(32,FMT='(A,I3,A,I3)') 'Transition : ', itran(1,it), ' -> ', itran(2,it)
			write(32,*) '      v (km/s)        flux [erg/s/cm2/Hz]    I(mu=1)      I(selected mu)  [both in erg/s/cm2/st/Hz]'
			do j = 1, 100
				write(32,FMT='(5(5X,1PE12.5))') freq_axis(j), (flux_out(k,j),k=1,3)				
			enddo
			write(32,*)
		enddo
				
! 		deallocate(freq_axis)
		deallocate(flux_out)
 	
		allocate(Slte(nz))		
		
		do i = 1, ntran_output
			it = output_transition(i)
			up = itran(1,it)
			low = itran(2,it)

			sig = dtran(1,it) !sig=(h*nu*Blu)/(4*pi)
			glu = dlevel(2,low) / dlevel(2,up)
			acon = TWOHC2 * dtran(2,it)**3  !acon = (2*h*nu**3)/c**2			

			tau0 = 0.d0
			chip = 0.d0
			chim = 0.d0

			tau(it,0) = 0.d0
			col_up = 0.d0
			col_low = 0.d0
			do ip = 1, nz
				ipl = nl * (ip-1)

				chil(ip) = sig / dopplerw(i,ip) * (x(low+ipl) - glu * x(up+ipl))
				col_up = col_up + x(up+ipl) * dz(ip)
				col_low = col_low + x(low+ipl) * dz(ip)
				Sl(ip) = acon * glu * x(up+ipl) / (x(low+ipl) - glu * x(up+ipl))
				Slte(ip) = acon * glu * xlte(up+ipl) / (xlte(low+ipl) - glu * xlte(up+ipl))

         	if (ip == 1) then
					tau(it,ip) = chil(ip) * dz(ip)					
         	else
					tau(it,ip) = tau(it,ip-1) + chil(ip) * dz(ip)
         	endif

			enddo
			
			intensity = 0.d0
			do ip = 1, nz
				e1 = exp(-(tau(it,nz)-tau(it,ip)) / (sqrt(PI)*mu_output))
				e2 = exp(-(tau(it,nz)-tau(it,ip-1)) / (sqrt(PI)*mu_output))
				intensity = intensity + Sl(ip) * (e1-e2)				
			enddo
						
! 			deltaTBR = Tbr_I(dtran(2,it),intensity,tau(it,nz)/sqrt(PI))
			call Tbr4I(dtran(2,it),intensity,tau(it,nz)/sqrt(PI),deltaTBR,deltaTBRJ)

			final(1,colIndex) = radius
			final(2,colIndex) = sum(dz * factor_abundance * nh)
			final(3,colIndex) = column_density
			final(4,colIndex) = sum(total_flux) / 1d23 / (column_density / maxval(dopplerw))
			fin_tr(i,1,colIndex) = column_density
			fin_tr(i,2,colIndex) = tau(it,nz)/sqrt(PI)
			fin_tr(i,3,colIndex) = total_flux(i) !flux_total(it+nr*(nz-1)) * 4.d0*PI * 1.d23
			fin_tr(i,4,colIndex) = PH * dtran(2,i) / (PK*Inv_plexp(PC**2 * total_intensity(i) / (2.d0*PH*dtran(2,i)**3)))!(PC/dtran(2,i))**3 / (2.d0*PK*1d5) * total_intensity(i)
			fin_tr(i,5,colIndex) = 1e-3*intensity
			fin_tr(i,6,colIndex) = deltaTBR
			fin_tr(i,7,colIndex) = deltaTBRJ
			fin_tr(i,8,colIndex) = 0.0

			nprint = colIndex

			if (1.0e-4*wl(itr(i),jtr(i)) .ge. 1) then
            write(35,'(/5x,f8.3,'' cm ('',f7.3,'' GHz) transition between levels '',&
                i2,'' and '', i2)') &
                1.0e-4*wl(itr(i),jtr(i)),1.e-9*freq(itr(i),jtr(i)),itr(i), jtr(i)
         else
           write(35,'(/5x,f8.2,'' mic ('',f8.3,'' GHz) transition between levels '',&
               i2,'' and '', i2)') &
               wl(itr(i),jtr(i)),1.e-9*freq(itr(i),jtr(i)),itr(i), jtr(i)
         end if
         write(35,"(5x,'Upper Level: ',a)")ledet(itr(i))
         write(35,"(5x,'Lower Level: ',a)")ledet(jtr(i))

         write(35,FMT='(A)') '       z          N(H2)        N(mol)        T          Texc '
         write(35,FMT='(A)') '       cm         cm-2/kms     cm-2/kms      K            K '

			z = 0.d00
			columnH2 = 0.d0
			column = 0.d0
			do j = 1, nz
				z = z + dz(j)				
				columnH2 = columnH2 + nh(j) * dz(j) * factor_abundance
				column = column + nh(j) * dz(j) * abundance(i) * factor_abundance
				write(35,FMT='(5(1P8E13.5))') z * factor_abundance, columnH2, column, temperature(i), PHK*dtran(2,i)/log(1.d0+acon/Sl(j))
			enddo

		enddo

		write(35,*)

! Write populations
		write(31,*) '**********************************************************'
		write(31,FMT='(A,1PE12.5,A,I4)') ' N(mol) [cm-2]: ', column_density, '    -  nzones: ', nz 
		write(31,*) '**********************************************************'
		write(31,FMT='(A)') 'Level   n [cm-3]        n/n*       n* [cm-3]'
		
		do ip = 1, nz
			ipl = nl * (ip-1)
			write(31,FMT='(A,I4)') 'Zone ', ip			
			do ii = 1, nl
				write(31,FMT='(I4, 3(2X,1PE12.5))') ii, pop(ii+ipl) / factor_abundance, pop(ii+ipl)/popl(ii+ipl),&
					popl(ii+ipl) / factor_abundance				
			enddo			
		enddo
				
		deallocate(Slte)
		deallocate(total_flux)
		
	end subroutine calculate_intermediate_results	

end module io_cep
