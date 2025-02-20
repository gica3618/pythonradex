module coll_molpop
use global_molpop
use maths_molpop
implicit none
contains

!-----------------------------------------------
! Read all information about collisional partners
!-----------------------------------------------
	subroutine read_collisions(iequal)
	logical :: iequal
	integer :: i, l, k, iunit
	character*128 :: str, Method, Option, OPT
	logical :: UCASE, error, stat
	integer :: LMeth, LOpt
	character*72 TableName, TableInfo1, TableInfo2
   data UCASE/.TRUE./
   double precision aux, cross_section
	
!     Collision Info:     
		n_col = rdinp(iequal,15,16)
		
		if (n_col.lt.1 .or. n_col.gt.10) then
	   	write (str,'(I4)') n_col
	   	OPT = 'number of collision partners'
	   	error = error_message(opt,str)
			return
		end if
		
		
      do i = 1, n_col
!       For each collision partner get relative weight, method of determining
!       collision rate coefficients, data file name, tag for relevant 
!       subroutine and cross-section renormalization for surprisal and hard sphere 
        
			fr_col(i) = rdinp(iequal,15,16)
			if (kbeta > 3 .and. trim(adjustl(file_physical_conditions)) /= 'none') then
				write(16,"(4x,I1,'.  Relative weight of collision partner from column = ',I2)") i, int(fr_col(i))
			else
				write(16,"(4x,I1,'.  Relative weight of collision partner = ',f5.2)") i, fr_col(i)
				if(i_sum .eq. 1) then
					write(17,"(4x,I1,'.  Relative weight of collision partner = ',f5.2)") i, fr_col(i)
				endif
			endif

!************************************************************************        	
!ME 2013-11-28:
!   Do away with all the cross-section options. Use only Table entry
!************************************************************************        	

!        	call rdinps2(iequal,15,Method,LMeth,UCASE)
			
!        	if (Method(1:LMeth) .eq. 'TABLE') then

        	if (.true.) then
        		call rdinps(iequal,15,fn_kij(i))
        		call rdinps2(iequal,15,Option,LOpt,UCASE) 
!        		gxsec(i) = rdinp(iequal,15,16)
        		gxsec(i) = 1.0
        	
          	i_col_src(i) = 1
          	L = fsize(fn_kij(i))
          	
!       Read collision tables from file: collision_tables.dat
! 				inquire(file=trim(adjustl(path_database))//'/Coll/collision_tables.dat',&
! 					exist=stat)
!       		if (.not.stat) then
!       			print *, 'Error when opening file ',&
!       				trim(adjustl(path_database))//'/Coll/collision_tables.dat'
!       			stop
!       		endif
!       
!           	open(10, file=trim(adjustl(path_database))//'/Coll/collision_tables.dat', status='old')
! 	    		TableName = 'dummy'
! 	    		rewind 10
! 	    		
! !        First read Header lines
! 	    		call Pass_Header(10)
! 	    		do while(.not.(fn_kij(i)(1:L) .eq. TableName))
!               	k = 0
! 		    		read(10,'(a)',iostat = k) str
!               	if (k .eq. -1) then
! !                  No match:
!                	str = fn_kij(i)(1:L)
! 	             	OPT = 'table filename'
! 	             	error = error_message(opt,str)
! 						return          
! 		     		end if
! 		    		TableName = str(1:size(str))
! 		    		read(10,'(a)') str
! 		    		TableInfo1 = str(1:size(str))
! 		    		read(10,'(a)') str
! 		    		TableInfo2 = str(1:size(str))
! 		    		read(10,'(a)') str
!           	end do
! 				close(10)	
				
!         Proper match:

! 2         	write(16,"(8x,'Collision rates from data file ',a,/8x,a,/8x,a)") fn_kij(i)(1:L),&
! 					TableInfo1, TableInfo2
!           	if (i_sum .eq.1) then
!           		write(17,"(8x,'Collision rates from data file ',a,/8x,a,/8x,a)") fn_kij(i)(1:L),&
!           			TableInfo1, TableInfo2
!           	endif
          	                    	
          	if(Option(1:LOpt) .eq. 'SQRT(T)') then
            	i_col_exsub(i) = 1
          	else if(Option(1:LOpt) .eq. 'CONST') then
            	i_col_exsub(i) = 2
          	else
            	str = Option(1:LOpt)            	
      	  		OPT = 'T-extrapolation law'
      	  		error = error_message(opt,str)
					return
          	end if
          	
! Attach the directory name to the collision info file name
        		call attach2(trim(adjustl(path_database))//'/Coll/',fn_kij(i)(1:L),str)
        		L = fsize(str)
        		fn_kij(i)(1:L) = str(1:L)
        		
        		inquire(file=fn_kij(i)(1:L),exist=stat)
          	if (.not.stat) then
					print *, 'Error when opening file ', fn_kij(i)(1:L)
					stop
				endif
				
2         	write(16,"(8x,'Collision rates from data file ',a,/8x,a,/8x,a)") fn_kij(i)(1:L)
          	if (i_sum .eq.1) then
          		write(17,"(8x,'Collision rates from data file ',a,/8x,a,/8x,a)") fn_kij(i)(1:L)
          	endif
        		
			else if(Method(1:LMeth) .eq. 'SIO_ROVIB') then
				gxsec(i) = rdinp(iequal,15,16)
         	i_col_src(i) = 2
          	write(16,"(8x,'Collision rates from SiO ro-vibrational')")
          	L = fsize(fn_kij(i))          	
            i_col_exsub(i) = 1	      	
	      	fn_kij(i) = 'SiO_H2.kij'
            write(16,"(8x,'SiO-H2: theory of Bieniek & Green 1983, ApJ 265, L29',/,8x,&
            	&'Utilizes data file ',a)") 'SiO_H2.kij'
          	if(i_sum .eq. 1) then
          		write(17,"(8x,'SiO-H2: theory of Bieniek & Green 1983, ApJ 265, L29',/,8x,&
          			&'Utilizes data file ',a)") 'SiO_H2.kij'
          	endif
          	
! Attach the directory name to the collision info file name
        		call attach2(trim(adjustl(path_database))//'/Coll/',fn_kij(i)(1:L),str)
        		L = fsize(str)
        		fn_kij(i)(1:L) = str(1:L)
        		
			else if(Method(1:LMeth) .eq. 'HARD_SPHERE') then				
         	i_col_src(i) = 3          	
            i_col_exsub(i) = 3            
            cross_section = rdinp(iequal,15,16)            
            gxsec(i) = rdinp(iequal,15,16)
            
	      	gxsec(i) = gxsec(i)*cross_section
            write(16,"(8x,'Hard-sphere collisions; geometric cross section = ',1pe9.2,&
            	&' cm^-2')") gxsec(i)
          	if(i_sum .eq. 1) then
          		write(17,"(8x,'Hard-sphere collisions; geometric cross section = ',1pe9.2,&
          			&' cm^-2')") gxsec(i)
          	endif          	          	
			else
         	str = Method(1:LMeth)
	    		OPT = 'source of collision data'
	    		error = error_message(opt,str)
				return          
        	end if
        	
		end do
				
!     Normalize the relative weights to fractional abundances of h2
! Only if single-zone escape probability (SLAB or LVG)
! If using CEP with variable physical conditions, fr_col indicates the column in the file that is used as collider
		if (kbeta < 3 .or. trim(adjustl(file_physical_conditions)) == 'none') then
			aux = vsum(fr_col,n_col)
			do i = 1, n_col
				fr_col(i) = fr_col(i) / aux
			end do
		endif

	      
	end subroutine read_collisions

!-----------------------------------------------
! This routine reads the collisional rates into the C array
!-----------------------------------------------
   subroutine loadcij
   integer i,j,l

		do i=1,n
      	do j=1,i
         	c(i,j)=0.0
        	end do
      end do
		
      do l=1,n_col			        
      	select case (i_col_src(l))          
      		case(1)
      			call loadcij_norm_table(fn_kij(l),fr_col(l),i_col_exsub(l),gxsec(l))				
      		case(2)
            	call loadcij_spec_table1(fn_kij(l),fr_col(l),gxsec(l))
            case(3)
            	call hard_sphere_collisions(fr_col(l),gxsec(l))
			end select
			
      end do
      
      return
	end subroutine loadcij

!*****************************************
! ROUTINES WITH COLLISIONAL RATES
!*****************************************

!-----------------------------------------
!-----------------------------------------
   subroutine hard_sphere_collisions(weight,xsec)
!     calculates collision rates using hard sphere model   
   integer i,j
   double precision xsec,col,g_sum, weight

   	col    = xsec*vt*weight
      g_sum = 0.0
      do i = 1, n
        	g_sum = g_sum + g(i)
      end do
      do i = 2, n
        	do j = 1, i-1
          	c(i,j) =  c(i,j)+col*g(j)/g_sum
        	end do
      end do

      return
	end subroutine hard_sphere_collisions

!-----------------------------------------
!-----------------------------------------
   subroutine loadcij_norm_table(filename,fraction,i_extrap,xsec)
!     calculates the collision rate coefficients by interpolating
!     or extrapolating the data from a table   
   character*64 filename,filen
   character*120 line
   integer i,j,k,kk,i_extrap,ntemp,unit,n_comments
   double precision f1,f2,fraction,temp(100),col(100),xsec
   logical :: stat

		inquire(file=filename,exist=stat)
      if (.not.stat) then
      	print *, 'Error when opening file ', filename
      	stop
      endif
      
      open(4, err=100, file=filename, status='unknown')
      call Pass_Header(4)
      unit = -4
      ntemp=rdinp(.true.,unit,16)
!  The following reading is the only place where the input is rigid
      do i=1,3
        	read(4,'(a)') line
      end do
      read(4,*) (temp(i),i=1,ntemp)
      read(4,'(a)') line
      if(t .le. temp(1)) then
	  		write(16,"(4x,'*** T =',f6.1,' K is below the values tabulated in ',a)")&
	  			T,filename(1:fsize(filename))
        	kk=1
        	f2=0.0
        	if(i_extrap .eq. 1) then
	    		f1 = dsqrt(T/Temp(1))
				write(16,"(/,4x,'*** Rate coefficients extrapolated from T =',f6.1,&
					&' K using sqrt(T)')") Temp(1)
        	elseif(i_extrap .eq. 2) then
	    		f1=1.0
				write(16,"(4x,'*** Using rate coefficients for T =',f6.1,' K')") Temp(1)
	  		end if
      else if(t .ge. temp(ntemp)) then
	  		write(16,"(/,4x,'*** T =',f6.1,' K is above the values tabulated in ',a)")&
	  			T,filename(1:fsize(filename))
        	kk=ntemp-1
        	f1=0.0
        	if(i_extrap .eq. 1) then
	    		f2 = dsqrt(T/Temp(ntemp))
				write(16,"(4x,'*** Rate coefficients extrapolated from T =',f6.1,&
					&' K using sqrt(T)')") Temp(ntemp)
        	elseif(i_extrap .eq. 2) then
	    		f2=1.0
				write(16,"(4x,'*** Using rate coefficients for T =',f6.1,' K')") Temp(ntemp)
	  		end if
      else
        	kk=1
        	do while(.not.(t .ge. temp(kk) .and. t .le. temp(kk+1)))
          	kk=kk+1
        	end do
        	f2 = (t-temp(kk))/(temp(kk+1)-temp(kk))
        	f1 = 1.0-f2
      end if
		
      do while(.true.)
        	read (4,*, end=10) i, j, (col(k), k=1,ntemp)
! Only include transitions between levels included in the model        	
        	if(i .gt. j .and. i <= n .and. j <= n) then
          	c(i,j) = c(i,j) + fraction*(col(kk)*f1+col(kk+1)*f2)*xsec
!     *     + coll_factor*fraction*(col(kk)*f1+col(kk+1)*f2) 
        	end if			
      end do
  10  close(4)
      return
 100  write(6,'(3a)') 'File ',filen(inmin(filen):inmax(filen)),' is missing! Stop.'
      
	end subroutine loadcij_norm_table

!-----------------------------------------
!-----------------------------------------
   subroutine loadcij_spec_table1(filename,weight,xsec)
!     Vibrational-rotational collision rate coefficients for 
!     SiO-H2 collisions. Using a semi-analytic approach
!     Reference: Bieniek & Green, 1983, ApJ, 265, L29.   
	character*64 filename
	character*120 line
	integer i,j,k,l,vup,jup,vdo,jdo,kup,lup,jm1,kk,jlow
	integer mm,mk,jmaxsio, n_comments
	logical :: stat
	double precision pip,tot,tot2,cor,cor2,d(0:4,0:19,0:4,0:19),xsec,weight

	data jmaxsio/19/

		inquire(file=filename,exist=stat)
      if (.not.stat) then
      	print *, 'Error when opening file ', filename
      	stop
      endif
      
      open(4, err=100, file=filename, status='unknown') 
      call Pass_Header(4)
      
      do vup=0, 2
        	do jup=0, 19
          	do vdo=0, vup
	            if(.not.(vup .eq. 2 .and. vdo .eq. 0)) then
              		do jdo=0,19
                		if(.not.(vup .eq. vdo .and. jup .le. jdo)) then
                  		read(4,'(4(i5),1pd12.3)') i,j,k,l,d(vup,jup,vdo,jdo)
                  		if (vup .eq. vdo) then
                    			pip=(t/2.0d3)**0.3
                  		else
                    			pip=(t/2.0d3)**2.5
                  		end if
                  		d(vup,jup,vdo,jdo)=d(vup,jup,vdo,jdo)*pip
                		end if
              		end do
            	end if
          	end do
        	end do
      end do
      close(4)
      
!       
!         correct collision rates from ground state for finite number of
!         rotational levels by multiplying cross sections by a correction
!         factor found from ratio of true pump rate (from BG paper) to
!         the partial pump rate found by summing g(jup)c(vup,jup,0,0)
!         have also adjusted v=1 to 2 cross sections in similar way
!
		tot=0.0
      tot2=0.0
      do jup=0,jmaxsio
        	tot=tot+(2*jup+1)*d(1,jup,0,0)/((t/2.0d3)**2.5)
        	tot2=tot2+(2*jup+1)*d(2,jup,1,0)/((t/2.0d3)**2.5)
      end do
      cor=4.8e-12/tot
      cor2=2.9e-11/tot2
      do vup=0,2
        	do jup=0,jmaxsio
          	do vdo=0,vup
	            if(.not.(vup .eq. 2 .and. vdo .eq. 0)) then 
              		do jdo=0,jmaxsio
                		if(.not.(vup .eq. vdo .and. jup .le. jdo)) then
!
!       now correct collisional cross-sections for finite number of levels
!
                  		if (vup .eq. 1 .and. vdo .eq. 0) then
                    			d(1,jup,0,jdo)=cor*d(1,jup,0,jdo)
                  		end if
                  		if (vup .eq. 2 .and. vdo .eq. 1) then
                    			d(2,jup,1,jdo)=cor2*d(2,jup,1,jdo)
                  		end if
                  		kup=vup*(jmaxsio+1)+jup+1
                  		lup=vdo*(jmaxsio+1)+jdo+1
                  		if (kup <= n .and. lup <= n) then
                  			c(kup,lup)=c(kup,lup)+d(vup,jup,vdo,jdo)*xsec*weight
                  		endif
                		end if
              		end do
            	end if
          	end do
        	end do
      end do
!
!        now fill in other cross sections not found in Bieniek-Green
!        1st - the v=2 to v=0 transitions
!
      jm1=jmaxsio+1
      kk=2*jmaxsio+3 
      do i=kk,kk+jmaxsio+1
        	do j=1,jm1
        		if (i <= n .and. j <= n) then
          		c(i,j)=c(i,j)+0.1*c(i,j+jm1)
          	endif
        	end do
      end do
!
!                  now the v=3 to lower transitions
!
      do i=1,jm1
        	do j=1,jm1
        		if (i+3*jm1 <= n .and. j <= n) then
          		c(i+3*jm1,j)=c(i+3*jm1,j)+0.01*c(i+2*jm1,j+jm1)
          	endif
          	if (i+3*jm1 <= n .and. j+jm1 <= n) then
          		c(i+3*jm1,j+jm1)=c(i+3*jm1,j+jm1)+0.1*c(i+2*jm1,j+jm1)
          	endif
          	if (i+3*jm1 <= n .and. j+2*jm1 <= n) then
          		c(i+3*jm1,j+2*jm1)=c(i+3*jm1,j+2*jm1)+1.0*c(i+2*jm1,j+jm1)
          	endif
        	end do
      end do

!                now  for the rotational transitions of v=3

      mm=3*jm1+1
      mk=2*jm1+1
      do jup=1,jmaxsio
        	do jlow=0,jup
        		if (mm+jup <= n .and. mm+jlow <= n) then
          		c(mm+jup,mm+jlow)=c(mm+jup,mm+jlow)+c(mk+jup,mk+jlow)
          	endif
        	end do
      end do

!               now for the v=4 transitions

      do i=1,jm1
        	do j=1,jm1
        		if (i+4*jm1 <= n .and. j <= n) then
          		c(i+4*jm1,j)=c(i+4*jm1,j)+0.001*c(i+2*jm1,j+jm1)
          	endif
          	if (i+4*jm1 <= n .and. j+jm1 <= n) then
          		c(i+4*jm1,j+jm1)=c(i+4*jm1,j+jm1)+0.01*c(i+2*jm1,j+jm1)
          	endif
          	if (i+4*jm1 <= n .and. j+2*jm1 <= n) then
          		c(i+4*jm1,j+2*jm1)=c(i+4*jm1,j+2*jm1)+0.1*c(i+2*jm1,j+jm1)
          	endif
          	if (i+4*jm1 <= n .and. j+3*jm1 <= n) then
          		c(i+4*jm1,j+3*jm1)=c(i+4*jm1,j+3*jm1)+1.0*c(i+2*jm1,j+jm1)
          	endif
        	end do
      end do

      mm=4*jm1+1
      mk=3*jm1+1
      do jup=1,jmaxsio
        	do jlow=0,jup
        		if (mm+jup <= n .and. mm+jlow <= n) then
          		c(mm+jup,mm+jlow)=c(mm+jup,mm+jlow)+c(mk+jup,mk+jlow)
          	endif
        	end do
      end do
      return
 100  write(6,'(3a)') 'File ',filename(inmin(filename):inmax(filename)),' is missing! Stop.'
	end subroutine loadcij_spec_table1

end module coll_molpop