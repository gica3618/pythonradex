c     background.f

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

      SUBROUTINE BACKRAD

C----- This routine returns the intensity of continuum radiation that is
C----- felt by the radiating molecules.  Intensity is computed at the
C----- line frequencies only.

C----- OPTIONS:
C-----  1 - Single blackbody; default is the cosmic microwave background
C-----      at T_CMB=2.725 K.  For values different from the default, the
C-----      corresponding redshift is determined according to 
C-----      T=T_CMB(1+z)
C-----  2 - The mean Galactic background radiation field plus CMB. This 
C-----      is a slightly revised version of Black, J. H. 1994, in 
C-----      The First Symposium on the Infrared Cirrus and Diffuse 
C-----      Interstellar Clouds. ASP Conference Series, Vol. 58, 
C-----      R.M. Cutri and W.B. Latter, Eds., p.355. Details are posted
C-----      at http://www.oso.chalmers.se/~jblack/RESEARCH/isrf.html
C-----      This spectrum is NOT adjustable by a scale factor because
C-----      it consists of several components that are not expected to
C-----      scale linearly with respect to each other.
C-----  3 - A user-defined radiation field that is specified by NRAD
C-----      values of frequency [cm-1], intensity [Jy nsr-1], and dilution
C-----      factor [dimensionless]. Spline interpolation is applied to 
C-----      this table. The intensity need not be specified at all 
C-----      frequencies of the line list, but a warning message will
C-----      appear if extrapolation (rather than interpolation) is required.
C
      implicit none

      include 'radex.inc'

      character*80 bgfile,title
      integer iline,irad,nrad
      real*8 xnubr(maxline),spinbr(maxline),dilbr(maxline),hnu,tbb3
      real*8 logfreq(maxline),logflux(maxline),fpp(maxline),huge
      parameter(huge=1.0e38) ! highest allowed by f90 (fdvt 28apr06)
      real*8 aa,bb,fout
      real*8 xnumin,xnumax,cbi,xln

c     bgfile:  file with user's background
c     title:   one-liner describing user's background
c     nrad:    number of wavelength points in user's background
c     irad:    to loop over background wavelength
c     iline:   to loop over lines
c     xnubr:   frequencies of user's background [cm^-1]
c     spinbr:  intensities of user's background [Jy/nsr]
c     dilbr:   dilution factors of user's background
c     hnu:     helps to calculate intensity
c     tbb3:    cmb addition to user's background
c     cbi:     cmb intensity

c     logfreq: base 10 log of frequency
c     logflux: base 10 log of intensity
c     fpp:     helps to calculate splines
c     aa,bb:   help to calculate Planck function
c     fout:    interpolated intensity
c     xnumin,xnumax: min/max line frequencies
c     xln:     log of line frequency

c     Executable statements begin here

      if (tbg.gt.0.0) then
c     option 1: Single black body
         do iline = 1,nline
            hnu = fk*xnu(iline)/tbg
            if (debug) write(*,*) iline,hnu,xnu(iline)
            if(hnu.ge.160.0d0) then
               backi(iline) = eps
c	sb/fvdt 30nov2011 Do not set backi to zero: may propagate into T(ex) 
c	and if line is thick, Tsum = NaN and code never converges ...
            else
               backi(iline) = thc*(xnu(iline)**3.0)
     $              /(dexp(fk*xnu(iline)/tbg)-1.d0)
            endif
            trj(iline)    = tbg
            totalb(iline) = backi(iline)
         enddo


      elseif(tbg.EQ.0.0) then
C     option 2:  mean Galactic background radiation
         call galbr

      elseif(tbg.lt.0.0) then
C     option 3:  user-specified radiation field with spline interpolation
 21      format(A,$)
         write(*,21) 'File with observed background intensity? '
         read*, bgfile
         open(unit=31,file=bgfile,status='old',err=666)

         tbb3 = -1.0*tbg

         read(31,310) title
         read(31,*) nrad

c        Read and dilute the input intensity
         do irad = 1,nrad
            read(31,*) xnubr(irad),spinbr(irad),dilbr(irad)
            spinbr(irad) = spinbr(irad)*dilbr(irad)
         enddo

         close(31)

 310   format(a)

C      In most cases it is safest to interpolate within the logarithms
C      of the frequencies and intensities
       do irad = 1,nrad
        logfreq(irad) = dlog(xnubr(irad))
        logflux(irad) = dlog(spinbr(irad))
       enddo

C      Set the spline coefficients
       call splcoeff(logfreq,logflux,nrad,huge,huge,fpp)

C      Interpolate continuum brightness at frequencies in the line list
C---------------------------------------------------
C------ NOTE:  Interpolation is done in the input --
C------ spectrum after dilution factors have      --
C------ been applied but before the CMB has been  --
C------ added. Units converted from               --
C------ [Jy nsr-1] to [erg s-1 cm-2 Hz-1 sr-1]    --
C---------------------------------------------------

       xnumin = xnu(1)
       xnumax = 0.0d0

       do iline = 1,nline
          xln = dlog(xnu(iline))
          call splintrp(logfreq,logflux,fpp,NRAD,xln,fout)
          aa  = thc*(xnu(iline)**3.0d0)
          bb  = fk*xnu(iline)
c     Add CMB if applicable
          if(tbb3.gt.0.0d0) then
             cbi = aa/(dexp(bb/tbb3)-1.0d0)
          else
             cbi = 0.0d0
          endif        
          if(xnu(iline).ge.xnubr(1).and.xnu(iline).le.xnubr(nrad)) then
             backi(iline) = 1.0d-14*dexp(fout) + cbi
          else
             backi(iline) = cbi
          endif
          trj(iline) = bb/dlog(aa/backi(iline) + 1.0d0)
          if(xnu(iline).lt.xnumin) xnumin = xnu(iline)
          if(xnu(iline).gt.xnumax) xnumax = xnu(iline)
c         added 24aug2011 (thanks Simon Bruderer)
          totalb(iline) = backi(iline)
       enddo

       if ((xnumin.lt.xnubr(1)).or.(xnumax.gt.xnubr(nrad))) print*, 
     $ 'Warning: the line list requires extrapolation of the background'

C        xlmin = dlog10(xnumin)
C        xlmax = dlog10(xnumax)
C        step = (xlmax-xlmin)/float(nstep)

C        do istep = 1,nstep+1
C           xl  = 10.d0**(xlmin + float(istep-1)*step)
C           xln = dlog(xl)

C           if(xl.lt.xnubr(1).or.xl.gt.xnubr(nrad)) then
C            print*, ' Warning:  the line list requires extrapolation of t
C      ;he continuum radiation '
C            write(8,834)
C  834       format(' WARNING: the continuous background requires extrapol
C      ;ation beyond the specified frequencies',/,' for some lines in the
C      ;line list')
C           endif

C           aa = thc*(xl**3.0)
C           bb = fk*xl

C           call splintrp(logfreq,logflux,fpp,NRAD,xln,fout)

C           if(tbb3.gt.0.0d0) then
C              cbi = aa/(dexp(bb/tbb3)-1.0d0)
C           else
C              cbi = 0.0d0
C           endif        
C           toti = 1.0d-14*dexp(fout) + cbi
C           trad = bb/dlog(aa/toti + 1.0d0)
C        enddo

      endif

      return
 666  print*,'Error opening background file'
      stop
      end


      SUBROUTINE GALBR
C
C.....Computes the mean background radiation near the Sun's location
C.....in the Galaxy:  the cosmic microwave background radiation is a
C.....blackbody with T_CMB=2.725 (Fixsen & Mather 2002, ApJ, 581, 817)       
C.....The far-IR/submm component is based upon an earlier analysis of 
C.....COBE data and is described by a single-temperature fit (Wright 
C.....et al. 1991, ApJ, 381, 200).  At frequencies below 10 cm-1 
C.....(29.9 GHz), there is a background contribution from non-thermal 
C.....radiation. The UV/Visible/near-IR part of the spectrum is based 
C.....on the model of average Galactic starlight in the solar 
C.....neighborhood of Mathis, Mezger, and Panagia (1983, Astron. 
C.....Astrophys., 128, 212).
C
      implicit none

      include 'radex.inc'

      integer iline
      real*8 aa,hnuk,tcmb,cbi,cmi,cmib,yy,xla,ylg
      parameter(tcmb=2.725)

c     aa,hnuk: help to calculate Planck function
c     tcmb:    CMB temperature
c     cbi:     CMB intensity
c     cmi:     synchrotron radiation intensity
c     cmib:    dust radiation intensity
c     yy,xla,ylg: to calculate stellar radiation field

      do iline = 1,nline
       aa   = thc*(xnu(iline)**3.0d0)
       hnuk = fk*xnu(iline)/tcmb

       if(xnu(iline).le.10.0d0) then
        cbi = aa/(dexp(hnuk) - 1.0d0) 
        cmi = 0.3d0*1.767d-19/(xnu(iline)**0.75d0)     ! synchrotron component

       elseif(xnu(iline).le.104.98d0) then
        cbi  = aa/(dexp(hnuk) - 1.0d0)
        cmib = aa/(dexp(fk*xnu(iline)/23.3d0) - 1.0d0)
        cmi  = 0.3d0*5.846d-7*(xnu(iline)**1.65d0)*cmib  ! COBE single-T dust

       elseif(xnu(iline).le.1113.126d0) then
        cmi = 1.3853d-12*(xnu(iline)**(-1.8381d0))
        cbi = 0.0d0                                      

       elseif(xnu(iline).le.4461.40d0) then
        cbi = 0.0d0
        cmi = 1.0d-18*(18.213601d0 - 0.023017717d0*xnu(iline)
     ;      + 1.1029705d-5*(xnu(iline)**2.0d0) 
     ;      - 2.1887383d-9*(xnu(iline)**3.0d0)
     ;      + 1.5728533d-13*(xnu(iline)**4.0d0))   

       elseif(xnu(iline).le.8333.33d0) then
        cbi = 0.0d0
        cmi = 1.d-18*(-2.4304726d0 + 0.0020261152d0*xnu(iline)
     ;      - 2.0830715d-7*(xnu(iline)**2.0d0)
     ;      + 6.1703393d-12*(xnu(iline)**3.0d0)) 

       elseif(xnu(iline).le.14286.d0) then
        yy  = -17.092474d0 - 4.2153656d-5*xnu(iline)
        cbi = 0.0d0
        cmi = 10.d0**yy

       elseif(xnu(iline).le.40000.d0) then
        xla = 1.0d8/xnu(iline)
        ylg = -1.7506877d-14*(xla**4.d0) 
     ;      + 3.9030189d-10*(xla**3.d0)
     ;      + 3.1282174d-7*(xla*xla) 
     ;      - 3.0189024d-3*xla 
     ;      + 2.0845155d0
        cmi = 1.581d-24*xnu(iline)*ylg
        cbi = 0.0d0

       elseif(xnu(iline).le.55556.d0) then
        xla = 1.0d8/xnu(iline)
        ylg = -0.56020085d0 + 9.806303d-4*xla
        cmi = 1.581d-24*xnu(iline)*ylg
        cbi = 0.0d0

       elseif(xnu(iline).le.90909.d0) then
        xla = 1.0d8/xnu(iline)
        ylg = -21.822255d0 + 3.2072800d-2*xla - 7.3408518d-6*xla*xla
        cmi = 1.581d-25*xnu(iline)*ylg
        cbi = 0.0d0

       elseif(xnu(iline).le.109678.76d0) then
        xla = 1.0d8/xnu(iline)
        ylg = 30.955076d0 - 7.3393509d-2*xla + 4.4906433d-5*xla*xla
        cmi = 1.581d-25*xnu(iline)*ylg
        cbi = 0.0d0

       else
c       radiation field extends to Lyman limit of H
      write(*,202) xnu(iline)                                                     
  202 FORMAT(' ** XNU = ',1PE13.6,' IS OUTSIDE THE RANGE OF THE FITTING       
     ; FUNCTION AND BEYOND LYMAN LIMIT')
        cbi=0.0d0
        cmi=0.0d0

       endif

      backi(iline) = cbi+cmi                                                        
      trj(iline)   = fk*xnu(iline)/dlog(1.0d0+aa/backi(iline))    ! brightness temperature
c	  added 24aug2011 (thanks Simon Bruderer)
      totalb(iline) = backi(iline)

      enddo

      return                                                                  
      end                                                                     

C
C.....This package contains several routines for applications of 
C.....cubic-spline interpolation.  A further routine, splinteg, is
C.....available on request (John.Black@chalmers.se) and does
C.....numerical quadrature through use of the spline coefficients
C.....determined for a set of points in routine splcoeff.  These
C.....routines have been written by J. H. Black.  The basic spline
C.....interpolation routines are based upon the routines of Numerical
C.....Recipes (Ch. 3.3), but are not identical to them.
C
      SUBROUTINE SPLCOEFF(x,f,N,fp1,fpn,fpp)
      IMPLICIT NONE
      INTEGER N,NMAX
      INTEGER I,K
      PARAMETER (NMAX=2500)
      REAL*8 fp1,fpn,x(nmax),f(nmax),fpp(nmax)
      REAL*8 p,qn,sig,un,u(NMAX)
C.....N values of a function f(x) are tabulated at points x(i), in 
C.....order of increasing x(i).  fpp(x) is the evaluated second derivative
C.....of f(x).  fp1 and fpn are the values of the first derivatives at
C.....the beginning and ending points, i=1 and i=N.  For natural splines,
C.....the second derivative is set to zero at a boundary.  Set fp1 and/or
C.....fpn > 1.0D60 for natural splines.  Otherwise, the derivatives can
C.....be specified and matched to those of extrapolating functions.
C
C.....Lower boundary condition (use 1e60 for f77/g77, 1e38 for f90):
      if (fp1.gt.0.99d38) then
        fpp(1)=0.d0
        u(1)=0.d0
      else
        fpp(1)=-0.5d0
        u(1)=(3.d0/(x(2)-x(1)))*((f(2)-f(1))/(x(2)-x(1))-fp1)
      endif
C
      do i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*fpp(i-1)+2.d0
        fpp(i)=(sig-1.d0)/p
        u(i)=(6.d0*((f(i+1)-f(i))/(x(i+1)-x(i))-(f(i)-f(i-1))/
     +    (x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      enddo
C
C.....Upper boundary condition (see above):
      if (fpn.gt.0.99d38) then
        qn=0.d0
        un=0.d0
      else
        qn=0.5d0
        un=(3.d0/(x(n)-x(n-1)))*(fpn-(f(n)-f(n-1))/(x(n)-x(n-1)))
      endif
C
      fpp(n)=(un-qn*u(n-1))/(qn*fpp(n-1)+1.)
      do k=n-1,1,-1
        fpp(k)=fpp(k)*fpp(k+1)+u(k)
      enddo
C
      return
      END

 
      SUBROUTINE splintrp(xin,fin,fppin,N,x,fout)
C.....For N tabulated values xin(i) and fin(x) of a function, and
C.....the array fppin(x), which is the 2nd derivative of fin(x) delivered
C.....by SPLCOEFF above, an interpolated value of fout is delivered
C.....at x.  The routine can also, if desired, return the 
C.....values of the first and second derivatives of the fitted 
C.....function, foutp and foutpp.
      IMPLICIT NONE
      INTEGER N
      INTEGER k,khi,klo
      REAL*8 x,fout,xin(n),fppin(n),fin(n)
      REAL*8 a,b,h,foutp,foutpp
      klo=1
      khi=N
1     if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xin(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xin(khi)-xin(klo)
      if (h.eq.0.d0) pause 'Warning: bad xin input in splintrp '
      a=(xin(khi)-x)/h
      b=(x-xin(klo))/h
      fout=a*fin(klo)+b*fin(khi)+((a**3.d0-a)*fppin(klo)+
     +     (b**3.d0-b)*fppin(khi))*(h**2.d0)/6.d0
C
C.....first derivative
C
      foutp=(fin(khi)-fin(klo))/h - (3.d0*a*a-1.d0)*h*fppin(klo)/6.d0 +
     +      (3.d0*b*b-1.d0)*h*fppin(khi)/6.d0
C
C.....second derivative
C
      foutpp=a*fppin(klo)+b*fppin(khi)
C
      return
      END

      





