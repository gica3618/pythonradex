c     main.f
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
      PROGRAM radex
      implicit none
c     Main program: controls program flow and drives subroutines

      include 'radex.inc'

      integer niter   ! iteration counter
      integer imore   ! are we running again?
      logical conv    ! are we converged?

c     Begin executable statements
      print*
      print*,'   Welcome to Radex, version of '//version
      print*

c     Get input parameters
      if (debug) print*,'calling getinputs'
 21   call getinputs

c     Read data file
      if (debug) print*,'calling readdata'
      call readdata

c     Calculate background radiation field
      if (debug) print*,'calling backrad'
      call backrad

      niter = 0
      conv  = .false.

c     Set up rate matrix, splitting it in radiative and collisional parts
c     Invert rate matrix to get `thin' starting condition
      if (debug) print*,'calling matrix'
      call matrix(niter,conv)

c     Start iterating
      do 22 niter=1,maxiter

c     Invert rate matrix using escape probability for line photons
         call matrix(niter,conv)
         if (conv) then
            print*,'Finished in ',niter,' iterations.'
            go to 23
         endif
 22   continue

      print*,'   Warning: Calculation did not converge in ',maxiter
     $     ,' iterations.'

c     Write output
      if (debug) print*,'calling output'
 23   call output(niter)

c     See if user wants more, else call it a day
 51   format(A,$)
      write(*,51) '  Another calculation [0/1] ? '
      read(*,*) imore
      write(13,52) imore
 52   format(i2)
      if (imore.eq.1) go to 21
      write(*,*) '   Have a nice day.'
c     Done! Now close log file ...
      close(13)
c     ...and output file.
      close(8)
      stop
      end
