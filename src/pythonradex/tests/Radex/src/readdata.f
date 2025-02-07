c     readdata.f
c
      SUBROUTINE readdata
      implicit none
      include 'radex.inc'

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
c     Reads molecular data files (2003 format)

      integer ilev,jlev   ! to loop over energy levels
      integer iline       ! to loop over lines
      integer ipart,jpart ! to loop over collision partners
      integer itemp       ! to loop over collision temperatures
      integer icoll       ! to loop over collisional transitions
      integer dummy       ! to skip part of the file

      integer id(maxpart)      ! to identify collision partners

c     upper/lower levels of collisional transition 
      integer lcu(maxcoll),lcl(maxcoll) 
      real*8 coll(maxpart,maxcoll,maxtemp)
      real*8 colld(maxpart,maxlev,maxlev)
      real*8 ediff
      real*8 temp(maxtemp) ! collision temperatures

      character*120 collref ! text about source of collisional data

c     to interpolate rate coeffs
      integer iup,ilo,nint
      real*8 tupp,tlow,fint

c     to verify matching of densities and rates
      logical found
      
c     to calculate thermal o/p ratio for H2
      real*8 opr

c     Executable part begins here.

      open(unit=11,file=molfile,status='old',err=99)
c     in the header, every second line is a comment
  101 format(a)
      read(11,*) 
      read(11,101) specref
      read(11,*) 
      read(11,*) amass
      read(11,*) 
      read(11,*) nlev

      if (nlev.lt.1) stop 'error: too few energy levels defined' 
      if (nlev.gt.maxlev) stop 'error: too many energy levels defined' 
      if (debug) write(*,*) 'readdata: basics'

c     Term energies and statistical weights
      read(11,*)
      do ilev=1,nlev
         read(11,*) dummy,eterm(ilev),gstat(ilev),qnum(ilev)
         if ((dummy.lt.1).or.(dummy.gt.nlev))
     $        stop 'error:illegal level number'
      enddo

      if (debug) write(*,*) 'readdata: levels'!,(eterm(ilev),ilev=1,nlev)

c     Radiative upper & lower levels and Einstein coefficients
      read(11,*) 
      read(11,*) nline
      read(11,*)

      if (nline.lt.1) stop 'error: too few spectral lines defined' 
      if (nline.gt.maxline) stop
     $     'error: too many spectral lines defined'

      do iline=1,nline
         read(11,*) dummy,iupp(iline),ilow(iline),aeinst(iline)
     $      ,spfreq(iline),eup(iline)
         if ((dummy.lt.1).or.(dummy.gt.nline))
     $        stop 'error:illegal line number'
         xnu(iline)=(eterm(iupp(iline))-eterm(ilow(iline)))
         if ((xnu(iline).lt.eps))
     $        stop 'error:illegal line frequency'
      enddo

      if (debug) write(*,*) 'readdata: lines'!,(xnu(iline),iline=1,nline)

c     Number of collision partners
      read(11,*)
      read(11,*) npart
      if (npart.lt.1) stop 'error: too few collision partners defined' 
      if (npart.gt.maxpart) stop 'error: too many collision partners'

 102  format(i1,a)
      do ipart=1,npart
         read(11,*)
         read(11,102) id(ipart),collref 
         read(11,*)
         read(11,*) ncoll
      if (ncoll.lt.1) stop 'error: too few collision rates defined' 
      if (ncoll.gt.maxcoll) stop 'error: too many collision rates'
         read(11,*)
         read(11,*) ntemp
         if (ntemp.lt.0) stop
     $        'error: too few collision temperatures defined'
         if (ntemp.gt.maxtemp) stop
     $        'error: too many collision temperatures'
         read(11,*)
         read(11,*) (temp(itemp),itemp=1,ntemp)
         read(11,*)

         if (debug) write(*,*) 'ready to read ',ncoll,' rates for '
     $        ,ntemp,' temperatures for partner ',ipart

         do icoll=1,ncoll
            read(11,*) dummy,lcu(icoll),lcl(icoll),
     $           (coll(ipart,icoll,itemp),itemp=1,ntemp)
         if ((dummy.lt.1).or.(dummy.gt.ncoll))
     $        stop 'error:illegal collision number'
         enddo

c     interpolate array coll(ncol,ntemp) to desired temperature

c     Must do this now because generally, rates with different partners
C     are calculated for a different set of temperatures

         if(ntemp.le.1) then
            do icoll=1,ncoll
               iup=lcu(icoll)
               ilo=lcl(icoll)
               colld(ipart,iup,ilo)=coll(ipart,icoll,1)
            enddo
         else
            if(tkin.gt.temp(1)) then
               if(tkin.lt.temp(ntemp)) then
C===  interpolation :
                  do itemp=1,(ntemp-1)
                     if(tkin.gt.temp(itemp).and.tkin.le.temp(itemp+1))
     $                    nint=itemp
                  enddo
                  tupp=temp(nint+1)
                  tlow=temp(nint)
                  fint=(tkin-tlow)/(tupp-tlow)
ccc     db
cc                  write(*,*) 'ipart,nint,fint: ',ipart,nint,fint
                  do icoll=1,ncoll
                     iup=lcu(icoll)
                     ilo=lcl(icoll)
                     colld(ipart,iup,ilo)=coll(ipart,icoll,nint)
     $                    +fint*(coll(ipart,icoll,nint+1)
     $                    -coll(ipart,icoll,nint))
                     if(colld(ipart,iup,ilo).lt.0.0)
     $                    colld(ipart,iup,ilo)=coll(ipart,icoll,nint)
                  enddo
               else
C===  Tkin too high :
                  if (tkin.ne.temp(ntemp)) then
                     print*,' Warning : Tkin higher than temperatures '/
     $                    /'for which rates are present.'
                  endif
                  do icoll=1,ncoll
                     iup=lcu(icoll)
                     ilo=lcl(icoll)
                     colld(ipart,iup,ilo)=coll(ipart,icoll,ntemp)
                  enddo
               endif
            else
C     Tkin too low :
               if (tkin.ne.temp(1)) then
                  print*,' Warning : Tkin lower than temperatures '//
     %                 'for which rates are present.'
               endif
               do icoll=1,ncoll
                  iup=lcu(icoll)
                  ilo=lcl(icoll)
                  colld(ipart,iup,ilo)=coll(ipart,icoll,1)
               enddo
            endif
         endif

c     Finished reading rate coefficients

      enddo

      if (debug) write(*,*) 'readdata: rate coeffs'

      close(11)

C$$$      if (debug) then
C$$$         print*,colld(1,1,1),colld(1,1,2),colld(1,1,3)
C$$$         print*,colld(1,2,1),colld(1,2,2),colld(1,2,3)
C$$$         print*,colld(1,3,1),colld(1,3,2),colld(1,3,3)
C$$$      endif

c     Combine rate coeffs of several partners, multiplying by partner density.

C$$$      if (debug) then
C$$$         print*,'id=',(id(ipart),ipart=1,npart)
C$$$         print*,'density=',(density(ipart),ipart=1,npart)
C$$$         print*,'rate(2,1)=',(colld(ipart,2,1),ipart=1,npart)
C$$$         print*,'rate(2,1)=',(colld(id(ipart),2,1),ipart=1,npart)
C$$$      endif

      do iup=1,nlev
         do ilo=1,nlev
            crate(iup,ilo)=0.0d0
         enddo
      enddo
      
      totdens = 0.0d0
      found   = .false.

c     Special case (CO, atoms): user gives total H2 but data file has o/p-H2.
c     Quite a big IF:
      if ((npart.gt.1).and.
     $   (density(1).gt.eps).and.(density(2).lt.eps)
     $     .and.(density(3).lt.eps)) then
         opr        = min(3.d0,9.0*dexp(-170.6/tkin))
         density(2) = density(1)/(opr+1.d0)
         density(3) = density(1)/(1.d0+1.d0/opr)
         print*,'*** Warning: Assuming thermal o/p ratio for H2 of ',opr
      endif
c     Note that for files like CN which have H2 and e- rates, the
c     warning is given without reason. May fix this later if needed.

      do ipart=1,maxpart
         totdens = totdens + density(ipart)
         do jpart=1,maxpart
            if ((id(jpart).eq.ipart).and.(density(ipart).gt.0.d0)) then
               found = .true.
               do iup=1,nlev
                  do ilo=1,nlev
                     crate(iup,ilo) = crate(iup,ilo) +
     $                    density(ipart)*colld(jpart,iup,ilo)
                  enddo
               enddo
            endif
         enddo
      enddo

      if (.not.found) then
         print*,'*** Warning: No rates found for any collision partner'
         stop
      endif

C$$$      if (debug) then
C$$$         print*,crate(1,1),crate(1,2),crate(1,3)
C$$$         print*,crate(2,1),crate(2,2),crate(2,3)
C$$$         print*,crate(3,1),crate(3,2),crate(3,3)
C$$$      endif

c     Calculate upward rates from detailed balance

      do iup = 1,nlev
         do ilo = 1,nlev
	  ediff = eterm(iup)-eterm(ilo)
	  if(ediff.gt.0.0d0) then
	    if((fk*ediff/tkin).ge.160.0d0) then
	      crate(ilo,iup) = 0.0d0
	    else
	      crate(ilo,iup) = gstat(iup)/gstat(ilo)
     $              *dexp(-fk*ediff/tkin)*crate(iup,ilo)
	    endif
	  endif
       enddo
c     initialize ctot array
      ctot(iup) = 0.0d0
      enddo

c     Calculate total collision rates (inverse collisional lifetime)

      do ilev=1,nlev
	do jlev=1,nlev
	  ctot(ilev)=crate(ilev,jlev)+ctot(ilev)
	enddo
      enddo

C$$$      if (debug) then
C$$$         print*,crate(1,1),crate(1,2),crate(1,3)
C$$$         print*,crate(2,1),crate(2,2),crate(2,3)
C$$$         print*,crate(3,1),crate(3,2),crate(3,3)
C$$$      endif

c      if (debug) print*,'ctot=',(ctot(ilev),ilev=1,nlev)

      return
   99 write(*,*) 'error opening data file ',molfile
      stop
      end
