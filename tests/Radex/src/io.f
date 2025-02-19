      SUBROUTINE getinputs
      implicit none
      include 'radex.inc'

c     io.f

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
      integer length       ! subroutine to determine
      external length      ! the length of a string
      integer ipart        ! loop over collision partners
      character*10 partner ! name of collision partner
      integer id           ! ID code of collision partner

c     Set input parameters to default values
      call defaults

c     Write log file for later reference
      open(unit=13,file=logfile,status='unknown',err=97)

 20   format(A)
c     must read file names in format(A): in free format, slashes are treated as separators
 21   format(A,$)
 22   format(A,I2,A,$)
 23   format(1pe10.3)
 24   format(1pe10.3,2x,1pe10.3)
 25   format(i2)

      write(*,21) 'Molecular data file ? '
      read(*,20) molfile
      if ((molfile(1:1).ne.'/').and.(molfile(1:1).ne.'.'))
     $     molfile = radat(1:length(radat))//molfile(1:length(molfile))
      write(13,20) molfile(1:length(molfile))

      write(*,21) 'Name of output file ? '
      read(*,20) outfile
      write(13,20) outfile(1:length(outfile))

 41   write(*,21) 'Minimum and maximum output frequency [GHz] ? '
      read(*,*) fmin,fmax
c     Default values: DC / Lyman limit
      if (fmin.eq.fmax) then
         fmin = 0.d0
         fmax = 3.d7    
      endif
      if (fmin.gt.fmax) then
         fmin = fmin + fmax
         fmax = fmin - fmax
         fmin = fmin - fmax
      endif
      fmin = dmax1(0.d0,fmin)
      fmax = dmin1(3.d7,fmax)
      write(13,24) fmin,fmax

 42   write(*,21) 'Kinetic temperature [K] ?  '
      read(*,*) tkin
      if ((tkin.lt.0.1).or.(tkin.gt.1.e4)) then
         print*,'Please enter a value between 0.1 and 1e4'
         goto 42
      endif
      write(13,23) tkin

 43   write(*,21) 'Number of collision partners ?  '
      read(*,*) npart
      if ((npart.lt.1).or.(npart.gt.7)) then
         print*,'Please enter a value between 1 and 7'
         goto 43
      endif
      write(13,25) npart

      do ipart=1,npart
 44      write(*,22) 'Type of partner',ipart,' ? '
         read(*,*) partner
         id = 0
         if ((partner.eq.'h2').or.(partner.eq.'H2')) id=1
         if ((partner(1:1).eq.'p').or.(partner(1:1).eq.'P')) id=2
         if ((partner(1:1).eq.'o').or.(partner(1:1).eq.'O')) id=3
         if ((partner(1:1).eq.'e').or.(partner(1:1).eq.'E')) id=4
         if ((partner.eq.'h').or.(partner.eq.'H')) id=5
         if ((partner.eq.'he').or.(partner.eq.'He')) id=6
         if ((partner.eq.'h+').or.(partner.eq.'H+')) id=7
         if (id.eq.0) then
            print*
     $           ,'Unknown species. Choose from: H2 p-H2 o-H2 e H He H+'
            goto 44
         endif
         write(13,*) partner
 45      write(*,22) 'Density of collision partner ',ipart,' [cm^-3] ? '
         read(*,*) density(id)
         if ((density(id).lt.1.0e-3).or.(density(id).gt.1.0e13)) then
            print*,'Please enter a value between 1e-3 and 1e13'
            goto 45
         endif
         write(13,23) density(id)
      enddo

c     Add ortho and para H2 densities if applicable
      if ((density(2).gt.0.0).or.(density(3).gt.0.0))
     $     density(1)=density(2)+density(3)

 46   write(*,21) 'Background temperature [K] ?  '
      read(*,*) tbg
c     Tbg > 0 means single blackbody such as CMB
c     Tbg = 0 means average ISRF
c     Tbg < 0 means use user-supplied routine
      if ((tbg.lt.-1.e4).or.(tbg.gt.1.e4)) then
         print*,'Please enter a value between -1e4 and 1e4'
         goto 46
      endif
      write(13,23) tbg

 47   write(*,21) 'Molecular column density [cm^-2] ?  '
      read(*,*) cdmol
      if ((cdmol.lt.1.e5).or.(cdmol.gt.1.e25)) then
         print*,'Please enter a value between 1e5 and 1e25'
         goto 47
      endif
      write(13,23) cdmol

 48   write(*,21) 'Line width [km/s] ?  '
      read(*,*) deltav
      if ((deltav.lt.1.e-3).or.(deltav.gt.1.e3)) then
         print*,'Please enter a value between 1e-3 and 1e3'
         goto 48
      endif
      write(13,23) deltav
c     convert to cm/s
      deltav = deltav * 1.0e5

c     The real stuff
      write(*,*) 'Starting calculations ...'

      return
 97   write(*,*) 'Error opening log file'
      stop
      end

c     ---------------------------------------------------------

      SUBROUTINE defaults
      implicit none
      include 'radex.inc'

c     Set physical parameters to default values

      integer ipart  ! to loop over collision partners

      tkin   = 30.0
      tbg    = 2.73
      cdmol  = 1.0e13
      deltav = 1.0

      density(1) = 1.0e5
      do ipart=2,maxpart
         density(ipart) = 0.0
      enddo

      return
      end

c     ------------------------------------------------------------

      FUNCTION length(str)
c     Returns the lengths of a string
      INTEGER length,maxl,i
      PARAMETER(maxl=200)
      CHARACTER*200 str

      do i=1,maxl
         if (str(i:i).eq.' ') then
            length=i-1
            RETURN
         endif
      enddo
      STOP 'Error: File name too long'
      END

c     ------------------------------------------------------------

      SUBROUTINE output(niter)
      implicit none
      include 'radex.inc'

c     Writes results to file

      integer iline    ! to loop over lines
      integer m,n      ! upper & lower level of the line
      integer niter    ! final number of iterations

      integer length
      external length

      real*8 xt        ! frequency cubed
      real*8 hnu       ! photon energy
      real*8 bnutex    ! line source function
      real*8 ftau      ! exp(-tau)
      real*8 toti      ! background intensity
      real*8 tbl       ! black body temperature
      real*8 wh        ! Planck correction
      real*8 tback     ! background temperature
      real*8 ta        ! line antenna temperature
      real*8 tr        ! line radiation temperature
      real*8 beta,escprob ! escape probability
      external escprob
      real*8 bnu       ! Planck function
      real*8 kkms      ! line integrated intensity (K km/s)
      real*8 ergs      ! line flux (erg / s / cm^2)
      real*8 wavel     ! line wavelength (micron)

c     Begin executable statements

c     Start with summary of input parameters
      open(unit=8,file=outfile,status='unknown',err=98)
 30   format (a,f8.3)
 31   format (a,1pe10.3)
 32   format (a)

      write(8,32) '* Radex version        : '
     $      //version(1:length(version))
      if (method.eq.1)
     $write(8,32) '* Geometry             : Uniform sphere'
      if (method.eq.2)
     $write(8,32) '* Geometry             : Expanding sphere'
      if (method.eq.3) 
     $write(8,32) '* Geometry             : Plane parallel slab'
c      write(8,32) '* Molecular data file  : '//specref(1:80)
      write(8,32) '* Molecular data file  : '//molfile(1:80)
      write(8,30) '* T(kin)            [K]: ',tkin
c      write(8,31) '* Total density  [cm-3]: ',totdens
      if(density(1).gt.eps)
     $write(8,31) '* Density of H2  [cm-3]: ',density(1)
      if(density(2).gt.eps)
     $write(8,31) '* Density of pH2 [cm-3]: ',density(2)
      if(density(3).gt.eps)
     $write(8,31) '* Density of oH2 [cm-3]: ',density(3)
      if(density(4).gt.eps)
     $write(8,31) '* Density of e-  [cm-3]: ',density(4)
      if(density(5).gt.eps)
     $write(8,31) '* Density of H   [cm-3]: ',density(5)
      if(density(6).gt.eps)
     $write(8,31) '* Density of He  [cm-3]: ',density(6)
      if(density(7).gt.eps)
     $write(8,31) '* Density of H+  [cm-3]: ',density(7)
      write(8,30) '* T(background)     [K]: ',tbg
      write(8,31) '* Column density [cm-2]: ',cdmol
      write(8,30) '* Line width     [km/s]: ',deltav/1.0d5

      write(8,33) 'Calculation finished in ',niter,' iterations'
 33   format(a,i4,a)

c     Column header
      write(8,*)
     $     '     LINE         E_UP       FREQ        WAVEL     T_EX'//
     $     '      TAU        T_R       POP        POP      '//
     $     ' FLUX        FLUX'
c     Units
      write(8,*)
     $     '                  (K)        (GHz)       (um)      (K) '//
     $     '                 (K)        UP        LOW    '//
     $     '  (K*km/s) (erg/cm2/s)'

      do iline=1,nline
	m  = iupp(iline)
	n  = ilow(iline)
	xt = xnu(iline)**3.
c     Calculate source function
	hnu = fk*xnu(iline)/tex(iline)
	if(hnu.ge.160.0d0) then
	  bnutex = 0.0d0
	else
	  bnutex = thc*xt/(dexp(fk*xnu(iline)/tex(iline))-1.d0)
	endif
c     Calculate line brightness in excess of background
	ftau = 0.0d0
	if(abs(taul(iline)).le.3.d2) ftau = dexp(-taul(iline))
	toti = backi(iline)*ftau+bnutex*(1.d0-ftau)
	if(toti.eq.0.0d0) then
	  tbl = 0.0d0
	else
	  wh = thc*xt/toti+1.d0
	  if(wh.le.0.d0) then
	    tbl = toti/(thc*xnu(iline)*xnu(iline)/fk)
	  else
	    tbl = fk*xnu(iline)/dlog(wh)
	  endif
	endif
	if(backi(iline).eq.0.0d0) then
	  tback = 0.0d0
	else
	  tback = fk*xnu(iline)/dlog(thc*xt/backi(iline)+1.d0)
	endif
c     Calculate antenna temperature
	tbl = tbl-tback
        hnu = fk*xnu(iline)
        if(abs(tback/hnu).le.0.02) then
	  ta = toti
	else
          ta = toti-backi(iline)
        endif
        ta = ta/(thc*xnu(iline)*xnu(iline)/fk)
c     Calculate radiation temperature
	beta = escprob(taul(iline))
	bnu  = totalb(iline)*beta+(1.d0-beta)*bnutex
	if(bnu.eq.0.0d0) then
	  tr = totalb(iline)
	else
	  wh = thc*xt/bnu+1.0
	  if(wh.le.0.0) then
	    tr = bnu/(thc*xnu(iline)*xnu(iline)/fk)
	  else
	    tr = fk*xnu(iline)/dlog(wh)
	  endif
	endif

c     Check if line within output freq range
	if (spfreq(iline).lt.fmax.and.spfreq(iline).gt.fmin) then
          wavel = clight / spfreq(iline) / 1.0d5 ! unit =  micron
          kkms  = 1.0645*deltav*ta
          ergs  = fgaus*kboltz*deltav*ta*(xnu(iline)**3.)
c     Line flux in K*cm/s and in erg/s/cm^2 
          if (dabs((tex(iline))).lt.1000.0) then
            write(8,113) qnum(m),qnum(n),eup(iline),spfreq(iline),wavel,
     $         tex(iline),taul(iline),ta,xpop(m),xpop(n),kkms/1.0d5,ergs
          else
            write(8,114) qnum(m),qnum(n),eup(iline),spfreq(iline),wavel,
     $         tex(iline),taul(iline),ta,xpop(m),xpop(n),kkms/1.0d5,ergs
          endif
 113      format(a,' -- ',a,f8.1,2(2x,f10.4),1x,f8.3,6(1x,1pe10.3))
 114      format(a,' -- ',a,f8.1,2(2x,f10.4),7(1x,1pe10.3))
	endif
      enddo

      return
 98   write(*,*) 'error opening output file'

      end
