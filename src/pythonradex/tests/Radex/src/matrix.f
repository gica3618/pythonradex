c     matrix.f
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
      SUBROUTINE matrix(niter,conv)
      implicit none
      include 'radex.inc'

c     Set up rate matrix

      integer niter            ! iteration counter
      integer ilev,jlev,klev   ! to loop over energy levels
      integer nplus            ! to solve statistical equilibrium
      integer iline            ! to loop over lines
      integer m,n              ! line upper/lower levels
      integer nthick           ! counts optically thick lines
      integer nfat             ! counts highly optically thick lines
      integer nreduce          ! size of reduced rate matrix
      integer indx,dsign       ! needed for NumRep equation solver

      real*8 rhs(maxlev)           ! RHS of rate equation
      real*8 yrate(maxlev,maxlev)  ! rate matrix

      real*8 etr,exr               ! to calculate radiative rates
      real*8 xt                    ! frequency cubed
      real*8 hnu                   ! photon energy
      real*8 bnutex                ! line source function
      real*8 cddv                  ! N(mol) / delta V
      real*8 beta,escprob          ! escape probability
      external escprob
      real*8 bnu                   ! Planck function
      real*8 uarray(maxlev,maxlev) ! reduced rate matrix
      real*8 redcrit               ! reduction criterion
      real*8 sumx                  ! summed radiative rate
      real*8 total                 ! to normalize populations
      real*8 tsum,thistex          ! to check convergence

      logical conv                 ! are we converged?

c	keep old population for underrelaxation procedure -- sb/fvdt 30nov2011
	  real*8 xpopold(maxlev)		 

c     db
      logical reduce
      reduce=.false.

c     Executable statements begin here

      if (debug) write(*,*) 'niter = ',niter

c     Clear array of level populations.
      do ilev=1,nlev
        rhs(ilev) = 0.0
        do jlev=1,nlev
          yrate(ilev,jlev) = 0.0
        enddo
      enddo

c     Initialize rate matrix
      nplus = nlev +1
      do ilev=1,nlev
c$$$	ctot(ilev)=0.0
		do jlev=1,nlev
	  	  yrate(ilev,jlev) = -1.0d-30*totdens
		enddo
c     Add conservation equation
		yrate(nplus,ilev) = 1.0
		rhs(ilev) = 1.0e-30*totdens
		yrate(ilev,nplus) = 1.0d-30*totdens
      enddo
      rhs(nplus) = 1.0e-30*totdens

c$$$c     Total collision rate
c$$$      do ilev=1,nlev
c$$$	do jlev=1,nlev
c$$$	  ctot(ilev)=colld(ilev,jlev)+ctot(ilev)
c$$$	enddo
c$$$      enddo

c     Contribution of radiative processes to the rate matrix.

c     First iteration: use background intensity
      if (niter.eq.0) then
         do 51 iline=1,nline
            m   = iupp(iline)
            n   = ilow(iline)
            etr = fk*xnu(iline)/trj(iline)
            if(etr.ge.160.0d0) then         
               exr = 0.0d0
            else
               exr = 1.0/(dexp(etr)-1.0d0)
            endif
	yrate(m,m) = yrate(m,m) + aeinst(iline)*(1.0+exr)
	yrate(n,n) = yrate(n,n) + aeinst(iline)*gstat(m)*exr/gstat(n)
	yrate(m,n) = yrate(m,n) - aeinst(iline)*(gstat(m)/gstat(n))*exr
	yrate(n,m) = yrate(n,m) - aeinst(iline)*(1.0+exr)
 51      enddo
      else
c     Subsequent iterations: use escape probability.
         cddv = cdmol / deltav
c     Count optically thick lines
         nthick = 0
         nfat   = 0

         do 52 iline=1,nline
            xt  = xnu(iline)**3.0
            m   = iupp(iline)
            n   = ilow(iline)

c     Calculate source function
            hnu = fk * xnu(iline) / tex(iline)
            if(hnu.ge.160.0) then
               bnutex = 0.0d0
            else
            bnutex = thc*xt/(dexp(fk*xnu(iline)/tex(iline))-1.0)
            endif

c     Calculate line optical depth.
            taul(iline) = cddv*(xpop(n)*gstat(m)/gstat(n)-xpop(m))
     $           /(fgaus*xt/aeinst(iline))
            if(taul(iline).gt.1.d-2) nthick = nthick+1
            if(taul(iline).gt.1.d05) nfat   = nfat+1

c     Use escape probability approximation for internal intensity.
            beta = escprob(taul(iline))
c     Split off local contribution to radiation field  sb/fvdt 30nov2011
            bnu  = totalb(iline)*beta
cc up to 30nov2011:     bnu  = totalb(iline)*beta+(1.0d0-beta)*bnutex
            exr  = bnu/(thc*xt)

c     Radiative contribution to the rate matrix
	  		yrate(m,m) = yrate(m,m)+aeinst(iline)*(beta+exr)
cc up to 30nov2011:		yrate(m,m) = yrate(m,m)+aeinst(iline)*(1.0+exr)
          	yrate(n,n) = yrate(n,n)+aeinst(iline)*(gstat(m)*exr/gstat(n))
          	yrate(m,n) = yrate(m,n)-aeinst(iline)*(gstat(m)/gstat(n))*exr
          	yrate(n,m) = yrate(n,m)-aeinst(iline)*(beta+exr)
cc up to 30nov2011:    	yrate(n,m) = yrate(n,m)-aeinst(iline)*(1.0+exr)
 52      enddo
      endif

c     Warn user if convergence problems expected
      if ((niter.eq.1).and.(nfat.gt.0)) print*
     $     ,"*** Warning: Some lines have very high optical depth"

      if (debug) then
         print*,yrate(1,1),yrate(1,2),yrate(1,3),yrate(1,4)
         print*,yrate(2,1),yrate(2,2),yrate(2,3),yrate(2,4)
         print*,yrate(3,1),yrate(3,2),yrate(3,3),yrate(3,4)
         print*,yrate(4,1),yrate(4,2),yrate(4,3),yrate(4,4)
      endif

c     Contribution of collisional processes to the rate matrix.
      do ilev=1,nlev
	yrate(ilev,ilev) = yrate(ilev,ilev) + ctot(ilev)
	do jlev=1,nlev
           if(ilev.ne.jlev) yrate(ilev,jlev) = yrate(ilev,jlev) 
     $          - crate(jlev,ilev)
	enddo
      enddo

      if (debug) then
         print*,yrate(1,1),yrate(1,2),yrate(1,3),yrate(1,4)
         print*,yrate(2,1),yrate(2,2),yrate(2,3),yrate(2,4)
         print*,yrate(3,1),yrate(3,2),yrate(3,3),yrate(3,4)
         print*,yrate(4,1),yrate(4,2),yrate(4,3),yrate(4,4)
      endif

c     db
      if (reduce) then

C     An auxiliary array is passed to the linear equation solver after
C     renormalization. The array Y retains the original matrix elements.

      if (debug) print*,'reducing matrix...' 

      do ilev=1,nlev
         do jlev=1,nlev
            uarray(ilev,jlev) = yrate(ilev,jlev)
         enddo
      enddo

C     Now test whether the matrix should be reduced
C     to exclude the radiatively coupled levels.

      redcrit = 10.0*tkin/fk
      nreduce = 0
      do ilev=1,nlev
         if(eterm(ilev).le.redcrit) nreduce = nreduce+1
      enddo

      if (debug) print*,'nreduce=',nreduce

C     We now separate the collisionally coupled levels from those that
C     are coupled mainly by radiative processes, compute an effective
C     cascade matrix for rates of transfer from one low-lying level
C     to another and then solve this reduced system of equations
C     explicitly for the low-lying levels only.

      do jlev=1,nreduce
         do ilev=1,nreduce
            do klev=nreduce+1,nlev
               uarray(ilev,jlev) = abs(yrate(klev,jlev)*yrate(ilev,klev)
     $              /yrate(klev,klev)) + uarray(ilev,jlev) 
            enddo
         enddo
      enddo

c     Invert this reduced matrix

      if (debug) print*,'inverting reduced matrix...'

      call ludcmp(uarray,nreduce+1,maxlev,indx,dsign)
      call lubksb(uarray,nreduce+1,maxlev,indx,rhs)

      if (debug) print*,'computing cascade...'

C     Compute the populations of the highly excited states
      if(nlev.gt.nreduce) then
         do klev=nreduce+1,nlev
            sumx = 0.0
            do jlev=1,nreduce
               sumx = rhs(jlev)*yrate(klev,jlev) + sumx
            enddo
            rhs(klev) = abs(sumx/yrate(klev,klev))
         enddo
      endif

      else  !if we don't want to reduce
         if (debug) print*,'inverting non-reduced matrix...'
         call ludcmp(yrate,nplus,maxlev,indx,dsign)
         call lubksb(yrate,nplus,maxlev,indx,rhs)
      endif

C     Level populations are the normalized RHS components
      total = 0.0d0
      do ilev=1,nlev
         total = rhs(ilev)+total   
      enddo
      if (debug) print*,'total rhs=',total
      if (debug) print*,'rhs=',(rhs(ilev),ilev=1,nlev)

c     Limit population to minpop
      do ilev=1,nlev

csb301111 Store old population 
         xpopold(ilev)=xpop(ilev)

         xpop(ilev) = dmax1(minpop,rhs(ilev)/total)

csb301111 first iteration: there is no old population
          if(niter.eq.0) xpopold(ilev)=xpop(ilev)

      enddo

      if (debug) print*,'computing T_ex...'

c     Compute excitation temperatures of the lines
      tsum = 0.0
      do iline=1,nline
		m  = iupp(iline)
		n  = ilow(iline)
        xt = xnu(iline)**3.d0
        if (niter.eq.0) then
           if ((xpop(n).le.minpop).or.(xpop(m).le.minpop)) then
              tex(iline) = totalb(iline)
           else
              tex(iline) = fk*xnu(iline)/
     $             (dlog(xpop(n)*gstat(m)/(xpop(m)*gstat(n))))
           endif
        else
           if ((xpop(n).le.minpop).or.(xpop(m).le.minpop)) then
              thistex = tex(iline)
           else
              thistex = fk*xnu(iline)/
     $             (dlog(xpop(n)*gstat(m)/(xpop(m)*gstat(n))))
           endif

c     Only thick lines count for convergence
	  if(taul(iline).gt.0.01) tsum = tsum
     $          + abs((thistex-tex(iline))/thistex)

c     Update excitation temperature & optical depth
	  tex(iline)  = 0.5*(thistex+tex(iline))
	  taul(iline) = cddv*(xpop(n)*gstat(m)/gstat(n)-xpop(m))
     $         /(fgaus*xt/aeinst(iline))
        endif
      enddo

      if (debug) write(*,*) niter,(tex(iline),iline=1,nline),
     $     (xpop(ilev),ilev=1,nlev),(taul(iline),iline=1,nline)

C     Introduce a minimum number of iterations
      if(niter.ge.miniter) then
         if(nthick.eq.0) conv = .true.
         if(tsum/nthick.lt.ccrit) conv = .true.
      if (debug) print*,niter,nthick,tsum,tsum/nthick,conv
      endif

csb301111 now do the underrelaxation!
      do ilev=1,nlev
         xpop(ilev)=0.3*xpop(ilev)+0.7*xpopold(ilev)
      end do

      return
      end

c     ------------------------------------------------------------

      FUNCTION ESCPROB(TAU)
      implicit none
      include 'radex.inc'

      real*8 escprob,beta,tau
      real*8 taur  !optical radius

      taur = tau/2.0

      if (method.eq.1) then
c     Uniform sphere formula from Osterbrock (Astrophysics of
C     Gaseous Nebulae and Active Galactic Nuclei) Appendix 2
c     with power law approximations for large and small tau
         if(abs(taur).lt.0.1) then
            beta = 1.d0-0.75d0*taur+(taur**2.)/2.5d0
     $           -(taur**3.)/6.d0+(taur**4.)/17.5d0
         else if(abs(taur).gt.5.d1) then
            beta = 0.75d0/taur
         else
            beta = 0.75d0/taur*(1.d0-1.d0/(2.d0*(taur**2.))+
     %           (1.d0/taur+1.d0/(2.d0*(taur**2.)))*dexp(-2.*taur))
         endif
      else if (method.eq.2) then
c     Expanding sphere = Large Velocity Gradient (LVG) or Sobolev case.
C     Formula from De Jong, Boland and Dalgarno (1980, A&A 91, 68)
C     corrected by factor 2 in order to match ESCPROB(TAU=0)=1
        if (abs(taur).lt.0.01) then
          beta = 1.0
        else if(abs(taur).lt.7.0) then
          beta = 2.0*(1.0 - dexp(-2.34*taur))/(4.68*taur)
        else
          beta = 2.0/(taur*4.0*(sqrt(log(taur/sqrt(pi)))))
        endif
      else if (method.eq.3) then
C     Slab geometry (e.g., shocks): de Jong, Dalgarno & Chu 1975, 
C     ApJ 199, 69 (again with power law approximations)
        if(abs(3.0*tau).lt.0.1) then
          beta = 1.0 - 1.5*(tau + tau**2.)
        else if(abs(3.0*tau).gt.50.0) then
          beta = 1.0d0/(3.0*tau)
        else
          beta = (1.0d0 - dexp(-3.0*tau))/(3.0*tau)
        endif
      else 
         print*,'Error: Escape probability method undefined'
         stop
      endif
      escprob = beta
      return
      end
      
