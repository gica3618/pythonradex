program molpop
use io_molpop
use sol_molpop
use global_molpop
use maths_molpop
use solution_cep, only: solve_with_cep
implicit none
!
!  ---------------   MOLECULAR LEVEL POPULATIONS SOLVER   ---------------
!
!  Main subroutines:
!
!    const  - define mathematical and physical constants
!    input - read input data
!    data  - load molecular data
!    first  - get first solution
!    solve  - scale the dimensions and solve the level equations
!
!  Main variables:
!
!        n - number of levels
!    kbeta - the escape probability used:
!            = 0 - Sobolev;
!             -1 - LVG slab, from Scoville & Solomon 1974, ApJ, 187, L67
!              1 - sphere, from van der Tak+ 2007, A&A 468, 627, eq. 19
!              2 - slab, version #2 KROLIK AND MCKEE, AP. J. SUPP. 37, P459 (1978)
!                        (slab now uses line center tau instead of mean tau)
!    kbeta = 3     Full CEP calcualtion
!        r - dimension(cm):  radius in Sobolev approximation
!            and thickness for slab
!        v - velocity in Sobolev approximation, Doppler linewidth otherwise;
!            entered in km/sec and then scaled to cm/sec
!      sat - saturation parameter
!            = 1 - maser affects level populations
!              0 - no maser effects on level populations
!     nmax - max # of iterations allowed
!        T - kinetic temperature
!     nh2  - h2 density (cm**-3)
!     nmol - molecule density (cm**-3)
!     mcol - molecular column density (cm-2/kms)
!    tcool - total cooling rate
! mol_mass - the molecular mass
!     kprt - printing index for output
!    kcool - printing empahsizes masing (0) or cooling (1)
!   ksolpr - printing index for solve
!   newtpr - printing index for newton
!    inprt - printing index for input
!   kfirst - printing index for initial guess
!    nbig  - the number of cooling lines that will be printed
!
!
!     molecular data
!
!     g     - statistical weights;
!     a     - Einstein coefficients;
!     c     - collision rates;
!     we    - weight factors
!
!     fr    -  state energy (in cm**-1)
!     ti    -  state energy (in K)
!     freq  -  transition frequencies in Hz
!     wl    -  wavelength in microns
!     tij   -  transition frequencies in K
!     boltz -  Boltzmann factor for state
!     gap   -  Boltzmann factor for transition
!     taux  -  multipliers for optical depths: tau = r*taux*delta(x)
!     ems   -  multipliers for line emissivities
!     rad   -  external radiation intensity in the lines
!
!        x - population per sublevel divided by Boltzmann factor
!           (analogous to departure coefficient);
!
!    we(i) = g(i)*exp(-e(i)/kT);
!     n(i) = nmol*x(i)*we(i) is the population of level i;
!   pop(i) = x(i)*exp(-e(i)/kT) is the (relative) population per
!            sub-level
!
!            a and c are scaled by we, so that
!                 n(i) a(i,j) = nmol x(i) (we(i) a(i,j))
!                 and similarly for n(i)*c(i,j)
!
!  The equations to solve are F(x1,....xn) = 0;
!
!        D - the jacobian matrix of the equations.
!


    logical quit, stat
    character*168 fn_inp, fn_out
    integer io1,I, Tcount, iunit, NHcount
    real(kind=8), allocatable :: x(:),f(:),d(:,:),cool(:,:)
    real(kind=8) :: NHarray
    character*60 T_tag
    character*80 ver
    
    open(13, file='version.dat',status='old')
    read(13,FMT='(A80)') ver
    close(13)
    
    ! ver = '*** MOLPOP-CEP; version of May 12, 2017 ***'


!        code uses the input method of DUSTY
!        the file mol.inp contains a list of molecule specific input files
!        each of the filenames (eg. oh) is read and then the data from oh.inp
!        is read, the calculated results are written to oh.out
!
!  Input units:   4 - Molecular data from .lev, .aij and .kij;
!                13 - Master input - molpop.inp;
!                15 - Model input;
!                11 - reads from mol_list.dat
!                10 - reads from collision_tables.dat
!  Output units - default:
!                16 - Model output;
!     optional:
!                17 - PLOT output
!      open(6, file='execution.log', status='unknown')
    write(6,'('' Started execution '')')
    call const

    inquire(file='molpop.inp',exist=stat)
    if (.not.stat) then
       print *, 'Error when opening file molpop.inp'
       stop
    endif
    
    open(13, err=998, file='molpop.inp', status='old')

!   Now work on input files one by one

    DO     
!      Get next input file name from the master input file
       call clear_string(128,apath)
       read(13, '(128a)', iostat=io1) apath
       if(io1 .lt. 0) exit        ! We're done
       IF(.not. empty(apath)) THEN
          Tcount = 1
          call attach2(apath, '.inp', fn_inp)
          inquire(file=fn_inp,exist=stat)
          if (.not.stat) then
              print *, 'Error when opening file ', fn_inp
              stop
          endif
       
          open(15, err=1000, file=fn_inp, status='old')
          iunit = -15
       
!         Setup output file name and go to work:
          call attach2(apath, '.out', fn_out)
          call attach2(apath, '.plot', fn_sum)
          open(16, file=fn_out, status='unknown')
          write(6,'('' working on '',a)') fn_out(1:fsize(fn_out))
          write(16,'(10x,a,/)') ver(1:fsize(ver))
          quit = .false.
          call input(quit)
       
          allocate(x(n))
          allocate(f(n))
          allocate(d(n,n+1))
          allocate(cool(n,n))
       
          if (quit) goto 222
          if (overlaptest) call numlin
          nprint = 0
       
!         When we use the original options of molpop, i.e., slab and LVG:
          if (kbeta /= 3 .and. kbeta /= 4) then
              CALL FIRST(X,F,D,COOL,QUIT)
              IF (QUIT) THEN
                 WRITE(16,"(/,' *** TERMINATED. FAILED ON INITIAL SOLUTION',/,&
                 '  POPULATIONS:'/,5(1PE14.6,2X))")  (X(I), I=1,N)
                 GOTO 2
              END IF
              IF (NMAX.EQ.0) GOTO 2
!             MAIN LOOP:
              DO I = 1,NMAX
                 CALL SOLVE(X,D,F,QUIT)
                 IF (QUIT) THEN
                   CALL FINISH(-1)
                   GOTO 2
                 END IF
                 CALL OUTPUT(X,COOL,KPRT)
!                Check if we're done:
                 IF (KTHICK.EQ.0) THEN
                   if (mcol.GT.COLM)  then
                      CALL FINISH(3)
                      GOTO 2
                   end if
                 ELSE !   KTHICK = 1:
                   if (mcol.LT.COLM)  then
                      CALL FINISH(3)
                      GOTO 2
                   end if
                 END IF
              END DO
!             Did not solve in max # of steps allowed:
              CALL FINISH(0)
       
2             continue
          endif
       
!         CEP-ALI, CEP-NEWTON or CEP-NAG
          if (kbeta == 3 .or. kbeta == 4 .or. kbeta == 5) then
              print *, 'Doing CEP...'
              call solve_with_cep(kbeta)
          endif
       
          close(15)
          close(16)
          if(i_sum .eq. 1) close(17)
       
222       continue
!         Deallocate all allocated memory
          if (allocated(tau)) deallocate(tau)
          if (allocated(esc)) deallocate(esc)
          if (allocated(dbdtau)) deallocate(dbdtau)
          if (allocated(a)) deallocate(a)
          if (allocated(tij)) deallocate(tij)
          if (allocated(taux)) deallocate(taux)
          if (allocated(c)) deallocate(c)
          if (allocated(rad)) deallocate(rad)
          if (allocated(we)) deallocate(we)
          if (allocated(gap)) deallocate(gap)
          if (allocated(ems)) deallocate(ems)
          if (allocated(boltz)) deallocate(boltz)
          if (allocated(rad_internal)) deallocate(rad_internal)
          if (allocated(rad_tau0)) deallocate(rad_tau0)
          if (allocated(rad_tauT)) deallocate(rad_tauT)
          if (allocated(freq)) deallocate(freq)
          if (allocated(fr)) deallocate(fr)
          if (allocated(wl)) deallocate(wl)
          if (allocated(ti)) deallocate(ti)
          if (allocated(g)) deallocate(g)
          if (allocated(pop)) deallocate(pop)
          if (allocated(coolev)) deallocate(coolev)
          if (allocated(xp)) deallocate(xp)
          if (allocated(ledet)) deallocate(ledet)
          if (allocated(imaser)) deallocate(imaser)
          if (allocated(jmaser)) deallocate(jmaser)
          if (allocated(x)) deallocate(x)
          if (allocated(f)) deallocate(f)
          if (allocated(d)) deallocate(d)
          if (allocated(cool)) deallocate(cool)
          if (allocated(final)) deallocate(final)
          if (allocated(itr)) deallocate(itr)
          if (allocated(jtr)) deallocate(jtr)
          if (allocated(a_maser)) deallocate(a_maser)
          if (allocated(in_tr)) deallocate(in_tr)
          if (allocated(f_tr)) deallocate(f_tr)
          if (allocated(fin_tr)) deallocate(fin_tr)
          if (allocated(uplin)) deallocate(uplin)
          if (allocated(lowlin)) deallocate(lowlin)
          if (allocated(qdust)) deallocate(qdust)
          if (allocated(Xd)) deallocate(Xd)

       END IF
    END DO

!   Normal ending
    close(13)
    write(6,'(/'' Done with all input files'')')
    stop

!   Problem ending
998 write(6,'(''Master input file molpop.inp is missing!'')')
    stop
1000 write(6,'(''Input file: '',a,'' is missing!'')') fn_inp
    stop
end program molpop
