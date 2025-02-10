c     radex.inc
c
c This file is part of the RADEX software package
c to calculate molecular excitation and radiative
c transfer in a homogeneous medium.
c
c Documentation for the program is posted at
c https://sron.rug.nl/~vdtak/radex/index.shtml  
c
c Although this program has been thoroughly tested, the
c authors do not claim that it is free of errors and
c gives correct results in all situations.
c
c Publications using this program should make a reference
c to our paper: A&A 468, 627 (2007).
c
c     ---------------------------------------------------------
c     
c     Customization section: user provides location of molecular data
C     files ("radat") and escape probability formula ("method")

      character*120 outfile,molfile,radat,specref,logfile
      character*20 version
      parameter(radat   = '../../tests/LAMDA_files/')
      parameter(version = '30nov2011')

c     names of output file & data file & molecule
      common/impex/outfile,molfile,specref

      parameter(logfile = './radex.log')

c     Escape probability method (uncomment your choice)
      integer method
c      parameter (method = 1)  ! uniform sphere
c      parameter (method = 2)  ! expanding sphere (LVG)
      parameter (method = 3)  ! plane parallel slab (shock)

c     No user editing needed beyond this point

c     ---------------------------------------------------------
c     
c     Physical and astronomical constants (CODATA 2002)
c
      real*8 clight,hplanck,kboltz,pi,amu

      parameter(clight  = 2.99792458d10)  ! speed of light     (cm/s)
      parameter(hplanck = 6.6260963d-27)  ! Planck constant    (erg/Hz)
      parameter(kboltz  = 1.3806505d-16)  ! Boltzmann constant (erg/K)
      parameter(pi      = 3.14159265d0)   ! pi
      parameter(amu     = 1.67262171d-24) ! atomic mass unit   (g)
c
c     ---------------------------------------------------------
c     
c     Array sizes
c
      integer maxlev,maxline,maxcoll,maxpart,maxtemp
      parameter(maxpart = 9)     ! maximum no. of collision partners (seven defined)
      parameter(maxtemp = 99)    ! maximum no. of collision temperatures
      parameter(maxlev  = 2999)  ! maximum no. of energy levels
      parameter(maxline = 99999)  ! maximum no. of radiative transitions
      parameter(maxcoll = 99999) ! maximum no. of collisional transitions

c
c     ---------------------------------------------------------
c     
c     Molecular data
c
      integer nlev,nline,ncoll,npart,ntemp,iupp(maxline),ilow(maxline)
c     nlev:  actual number of levels
c     nline: actual number of lines
c     ncoll: actual number of transitions
c     npart: actual number of partners
c     ntemp: actual number of collision temperatures

c     iupp(i): upper level of line i
c     ilow(i): lower level of line i

      real*8 amass,eterm(maxlev),gstat(maxlev),aeinst(maxline)
     $     ,eup(maxline)

c     amass:  molecular mass              (amu)
c     eterm:  energy levels               (1/cm)
c     gstat:  statistical weights
c     aeinst: Einstein A coefficients     (1/s)
c     eup:    line upper level energy     (K)
c     colld:  downward rate coefficients  (cm^3 /s)
c     xpop:   level populations

      common/imolec/nlev,nline,ncoll,npart,ntemp,iupp,ilow
      common/rmolec/amass,eterm,gstat,aeinst,eup

c
c     ---------------------------------------------------------
c     
c     Physical conditions
c
      real*8 density(maxpart),tkin,tbg,cdmol,deltav,totdens

c     density:  number densities of collision partners  (cm^-3)
c     totdens:  total number density of all partners    (cm^-3)
c     tkin:     kinetic temperature                     (K)
c     tbg:      temperature of background radiation     (K)
c     cdmol:    molecular column density                (cm^-2)
c     deltav:   FWHM line width                         (cm/s)

      common/cphys/density,tkin,tbg,cdmol,deltav,totdens

c
c     ---------------------------------------------------------
c     
c     Numerical parameters
c
      integer miniter,maxiter
      parameter(miniter=10)     ! minimum number of iterations
      parameter(maxiter=9999)   ! maximum number of iterations

      real*8 ccrit,eps,minpop,fmin,fmax
      parameter(ccrit=1.0e-6)   ! relative tolerance on solution
      parameter(eps=1.0d-30)    ! round-off error
      parameter(minpop=1.0d-20) ! minimum level population

c     fmin,fmax: minimum/maximum output frequency
      common/freq/fmin,fmax
c
c     ---------------------------------------------------------
c     
c     Radiative quantities
c
      real*8 taul(maxline),tex(maxline),backi(maxline),xnu(maxline)
      real*8 trj(maxline),totalb(maxline),spfreq(maxline)
      real*8 fk,thc,fgaus
      character*6 qnum(maxlev)
      parameter(fk    = hplanck*clight/kboltz)
      parameter(thc   = 2.d0*hplanck*clight)
      parameter(fgaus = 1.0645*8.0*pi)

      common/radi/xnu,taul,tex,backi,totalb,spfreq,trj
      common/quant/qnum

c     xnu:    line frequency (cm^-1)
c     taul:   line optical depth
c     tex:    line excitation temperature

c     trj:    background brightness (RJ)
c     backi:  background intensity [erg s-1 cm-2 Hz-1 sr-1]
c     totalb: background temperature (BB)

c     fk,thc: help to calculate intensities
c     fgaus:  accounts for Gaussian line shape

c     spfreq: spectroscopic line frequency (GHz), not used in
C             calculation but only to print output
c     qnum:   quantum numbers of levels

c
c     ---------------------------------------------------------
c     
c     Collisional quantities
c
      real*8 ctot(maxlev),crate(maxlev,maxlev),xpop(maxlev)

c     crate: collision rate matrix (density * rate coefficient)
c     ctot:  total collision rate 
c     xpop:  level populations

      common/collie/crate,ctot,xpop
c
c     ---------------------------------------------------------

c     For development / maintenance purposes:
      logical debug
      parameter(debug=.false.)

c     ---------------------------------------------------------
c     End of common definitions
