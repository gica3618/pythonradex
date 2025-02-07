module solution_cep
use constants_cep
use global_cep
use functions_cep
use escape_cep
use io_cep
use io_molpop, only : finish
implicit none
contains

!-----------------------------------------------
! This routine calls the necessary routines to solve a problem with CEP
!-----------------------------------------------
	subroutine solve_with_cep(algorithm)
	integer :: algorithm, colIndex
	real(kind=8) :: factor, previous_factor_abundance, step_factor_abundance, step_log, successful
	real(kind=8) :: col_density_now
	integer :: error
	logical :: finished

		colIndex = 1
	
! CEP with Newton
		if (algorithm == 3) then
			if (verbose == 2) then
				write(*,*) 'CEP with Newton'
			endif
			which_scheme = 2
		endif
		
! CEP with NAG-Newton
		if (algorithm == 5) then
			if (verbose == 2) then
				write(*,*) 'CEP with NAG-Newton'
			endif
			which_scheme = 4
		endif		
		
! CEP with ALI
		if (algorithm == 4) then
			if (verbose == 2) then
				write(*,*) 'CEP with ALI'
			endif
			which_scheme = 1
		endif
	
		factor_abundance = 1.d0
		
! Initialization
		call init

		call poplte
		pop(1:nl*nz) = popl(1:nl*nz)
		pop(1:nl*nz) = popl(1:nl*nz)
			
	! Precalculation of the beta and alpha functions
		call precalculate_beta
			
		open (UNIT=31,FILE=trim(adjustl(output_file))//'.CEP.pop',&
			STATUS='replace',ACTION='write')
		open (UNIT=32,FILE=trim(adjustl(output_file))//'.CEP.flux',&
			STATUS='replace',ACTION='write')
		open (UNIT=34,FILE=trim(adjustl(output_file))//'.CEP.slab',&
			STATUS='replace',ACTION='write')
		open (UNIT=35,FILE=trim(adjustl(output_file))//'.CEP.texc',&
			STATUS='replace',ACTION='write')

		write(35,"(/T9,'*** SUMMARY ***')")

	! If we start from optically thin, the system is linear. Solve it once	
		if (start_mode == 1 .and. read_previous_flag /= 1) then
			if (verbose == 2) then
				write(*,*) 'Calculating optically thin solution...'
			endif
			optically_thin = 1
			call calcJbar_Lstar_cep(pop)

		call correct_populations
			optically_thin = 0			
		endif
				
! If "increasing" or "decreasing" column density, adapt it to its correct value
! 		if ((kthick_strategy == 0 .or. kthick_strategy == 1) .and. &
! 			trim(adjustl(file_physical_conditions)) == 'none') then
		if ((kthick_strategy == 0 .or. kthick_strategy == 1)) then
			
			call adapt_col_density_strategy(pop)			
			col_density_now = sum(dz * factor_abundance * abundance * nh)
			write(*,*) 'Abundance factor : ', factor_abundance
			write(*,*) 'Using column density : ', col_density_now
			write(*,*) 'deltaz : ', dz(1)
			
		endif

		call poplte
									
! Perform a first non-linear solution
		if (verbose == 2) then
			write(*,*) 'First non-linear solution...'
		endif
		call CEP_solver(error)
		if (error == 1) then
			print *, 'ERROR IN FIRST NON-LINEAR SOLUTION'
			stop
		else
			if (verbose == 1) then
				write(*,*) '                       ---CEP OK---'
			endif
		endif

! MAIN LOOP
		global_iter = 0
		finished = .FALSE.
		step_log = printings_per_decade
		step_factor_abundance = 10.d0**(1.d0/step_log)
		
		successful = 0
		
		call write_results				
		call calculate_intermediate_results(pop,popl,colIndex)

		! call write_intermediate_results(pop,popl)
				
! Do the regridding to the original
		call regrid(.TRUE.,.FALSE.)
		
! If "increasing" or "decreasing" column density, do the calculations
! 		if ((kthick_strategy == 0 .or. kthick_strategy == 1) .and. &
! 			trim(adjustl(file_physical_conditions)) == 'none') then
		if ((kthick_strategy == 0 .or. kthick_strategy == 1)) then
		
			do while (.not.finished)
	
! Try to increase/decrease the column density
				previous_factor_abundance = factor_abundance
												
				if (kthick_strategy == 0) then
					factor_abundance = factor_abundance * step_factor_abundance
					col_density_now = sum(dz * factor_abundance * abundance * nh)
					if (col_density_now >= col_threshold) then
						finished = .TRUE.
					endif
				endif
				
				if (kthick_strategy == 1) then
					factor_abundance = factor_abundance / step_factor_abundance
					col_density_now = sum(dz * factor_abundance * abundance * nh)
					if (col_density_now <= col_threshold) then
						finished = .TRUE.
					endif
				endif
				
				if (kthick_strategy <= 1) then
					write(*,*) 'Abundance factor : ', factor_abundance
					write(*,*) 'Using column density : ', col_density_now
					write(*,*) 'deltaz : ', dz(1)
						
					if (verbose == 1) then
						write(*,*) 'Abundance factor : ', factor_abundance
						write(*,*) 'Using column density : ', col_density_now
					endif
				endif
				
				call poplte
				popold = pop
				call CEP_solver(error)
							
! Are we done?
				if (error == 1) then
					if (verbose == 1) then
						write(*,*) 'ERROR---'
					endif
					successful = 0
					pop = popold
					step_log = step_log + 1.d0
					step_factor_abundance = 10.d0**(1.d0/step_log)
					if (verbose == 1) then
						write(*,*) 'Decreasing step : ', step_factor_abundance
					endif
					if (step_factor_abundance < 1.d0) then
						stop
					endif
					factor_abundance = previous_factor_abundance
				else
					if (verbose == 2) then
						write(*,*) '                       ---CEP OK---'
					endif
					successful = successful + 1
					
					! call write_intermediate_results(pop,popl)
					call calculate_intermediate_results(pop,popl,colIndex)
				endif
				
! If more than 5 successful steps, try to increase the column density step
				if (successful > 5) then
					successful = 0
					step_log = step_log - 1.d0
					if (step_log < 4) step_log = 4.d0
					step_factor_abundance = 10.d0**(1.d0/step_log)
				endif
				
				global_iter = global_iter + 1
				
! We have finished this problem. Regrid to the original grid
				call regrid(.TRUE.,.TRUE.)

				colIndex = colIndex + 1
				
			enddo
		
		endif

		call finish(0)
				
		close(31)
		close(32)
		close(33)
		close(34)
		close(35)
		
	end subroutine solve_with_cep
	
end module solution_cep
