module sio_vib_routines
implicit none
real(kind=8) :: c(100,100)

contains
      SUBROUTINE Pass_Header(iunit)
!     Get past Header lines of opened file on unit # iunit
!     The end of headers is marked by a line that contains just ">"
      integer iunit
      character*128 header
      header = "XXX"
      do while(.not. (header(1:1).eq.">"))
         read(iunit, '(a)') header
      end do
      return
      end subroutine Pass_Header


subroutine loadcij_spec_table1(t, n)
!     Vibrational-rotational collision rate coefficients for 
!     SiO-H2 collisions. Using a semi-analytic approach
!     Reference: Bieniek & Green, 1983, ApJ, 265, L29.   
    character*64 filename
    character*120 line
    integer i,j,k,l,vup,jup,vdo,jdo,kup,lup,jm1,kk,jlow
    integer mm,mk,jmaxsio, n_comments, n
    logical :: stat
    double precision pip,tot,tot2,cor,cor2,d(0:4,0:19,0:4,0:19),xsec,weight, t

    data jmaxsio/19/

    xsec = 1.0
    weight = 1.0

        inquire(file='SiO_H2.kij',exist=stat)
      if (.not.stat) then
        print *, 'Error when opening file ', filename
        stop
      endif
      
      open(4, file='SiO_H2.kij', status='unknown') 
      call Pass_Header(4)

      c = 0.d0
      
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
    end subroutine loadcij_spec_table1

end module

program sio_vib
use sio_vib_routines
implicit none
integer :: i, j, k
real(kind=8) :: T(17), cc(17,100,100)

  T = (/10.0,20.0,30.0,40.0,50.0,75.0,100.0,150.0,200.0,300.,500.,750.,1000.,1500.,2000.,2500.,3000./)

  do i = 1, 17
    call loadcij_spec_table1(T(i), 100)
    cc(i,:,:) = c
  enddo

  open(unit=16, file='out', action='write', status='replace')

  write(16,*) '!     Vibrational-rotational collision rate coefficients for '
  write(16,*) '!     SiO-H2 collisions. Using a semi-analytic approach'
  write(16,*) '!     Reference: Bieniek & Green, 1983, ApJ, 265, L29.  '
  write(16,*) ''
  write(16,*) '>'
  write(16,*) ''
  write(16,*) 'Number of temperature columns = 17'
  write(16,*) ''
  write(16,*) 'I    J                        TEMPERATURE (K)'
  write(16,*) ''
  write(16,FMT='(10X, 17(F6.1,2X))') (T(i), i=1,17)
  write(16,*) ''

  do i = 2, 100
    do j = 1, i-1
      if (cc(1,i,j) /= 0.d0) then
        write(16,FMT='(I3,2X,I3,2X,17(1PE13.6))') i, j, (cc(k,i,j),k=1,17)
      endif
    enddo
  enddo

  close(16)
  print *, c(1:4,1:4)

end program