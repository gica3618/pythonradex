module cep_molpop_interface
use global_molpop
implicit none
contains

	subroutine generate_intermediate_data
	integer :: i, j, k, ntrans, nradiat
						
		open(unit=28,file='interface.dat',action='write',status='replace')
				
! File with physical conditions
		write(28,FMT='(A)') trim(adjustl(file_physical_conditions))
		
! Root for the output files
		write(28,FMT='(A)') apath
		
! Hydrogen density, abundance, thermal velocity and temperature (we assume that everything is constant for the moment)		
		write(28,FMT='(6(E13.5,1X))') nh2, xmol, v, vt, t, mol_mass
				
! Auxiliary function mode
		if (trim(adjustl(auxiliary_functions)) == 'KROLIK-MCKEE') then
			write(28,FMT='(I1)') 0
		endif
		if (trim(adjustl(auxiliary_functions)) == 'TABULATED') then
			write(28,FMT='(I1)') 1
		endif
		
! tau_threshold, r_threshold, col_threshold, kthick_strategy, precision, mu_output
		write(28,FMT='(E13.5,2X,E13.5,2X,E13.5,2X,I1,2X,E13.5,2X,I4,2X,F6.3)') taum, rm, &
			colm, kthick, acc, itmax, mu_output		
		
! Save precision in CEP, number of printings per decade and maximum number of
! iterations in the Newton method
		write(28,FMT='(E13.5,2X,I4,2X,I4,2X,I4)') cep_precision, nr, itmax, nInitialZones
		
		ntrans = 0
		nradiat = 0

! Count the number of radiative transitions		
		do i = 1, n
			do j = 1, i-1
				if (a(i,j) /= 0.d0) then
					nradiat = nradiat + 1
				endif
				ntrans = ntrans + 1
			enddo
		enddo

! Level's information		
		write(28,*) n
		do i = 1, n
			write(28,FMT='(I3,1X,E17.10)') g(i), fr(i)
		enddo
		
! Transition's information
! Radiative transitions
! Last two columns are dust radiation field and 
! the sum between any other external radiation field and the dust radiation
		write(28,*) ntrans, nradiat
		do i = 1, n
			do j = 1, i-1
				if (a(i,j) /= 0.d0) then
					write(28,FMT='(I3,1X,I3,7(1X,E17.10))') &
						i, j, &
						freq(i,j), a(i,j), c(i,j), &
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_internal(i,j), &
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_tau0(i,j), &
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_tauT(i,j)
				endif				
			enddo
		enddo

! The rest of transitions (collisional)		
		do i = 1, n
			do j = 1, i-1
				if (a(i,j) == 0.d0) then
					write(28,FMT='(I3,1X,I3,1X,E17.10,18X,4(1X,E17.10))') i, j, &
						freq(i,j), c(i,j),&
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_internal(i,j), &
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_tau0(i,j), &
						2.d0*hpl*freq(i,j)**3 / cl**2 * rad_tauT(i,j)
				endif				
			enddo
		enddo
								
! Write the transitions where to compute things
		write(28,FMT='(I4)') n_tr
		do i = 1, n_tr
			write(28,FMT='(I4,2X,I4)') itr(i), jtr(i)
		enddo
		
! And finally, collisional rates for all zones in case physical conditions vary
		if (trim(adjustl(file_physical_conditions)) /= 'none') then
			write(28,*) n_zones_slab
			do i = 1, n
				do j = 1, i-1
					if (a(i,j) /= 0.d0) then
						write(28,*) i, j, (collis_all(i,j,k),k=1,n_zones_slab)
					endif
				enddo
			enddo
			do i = 1, n
				do j = 1, i-1
					if (a(i,j) == 0.d0) then
						write(28,*) i, j, (collis_all(i,j,k),k=1,n_zones_slab)
					endif
				enddo
			enddo
		endif
		
		close(28)		
		
	end subroutine generate_intermediate_data
end module cep_molpop_interface
