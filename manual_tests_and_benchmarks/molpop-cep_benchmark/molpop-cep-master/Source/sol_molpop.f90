module sol_molpop
! use global_molpop
use maths_molpop, only: optdep
implicit none
contains


      SUBROUTINE FIRST(X,F,D,COOL,ERROR)
!     GET FIRST SOLUTION.
!     KTHICK = 1: OPTICALLY THICK STRATEGY -- START FROM THERMAL EQUILIBRIUM 
!                 AT LARGE R, THEN DECREASE R
!     KTHICK = 0: THE OPTICALLY THIN STRATEGY -- START FROM R = 0 SOLUTION
!                 OF LINEAR EQUATIONS, THEN INCREASE R
!
!     KFIRST CONTROLS MESSAGES - OPTIONS ARE PARALLEL TO THOSE FOR KPRT
!
      use global_molpop
      use maths_molpop
      use io_molpop
      INTEGER ITER,I,J,IR,MAX1 
      DOUBLE PRECISION X(N),F(N),D(N,N+1),COOL(N,N)
      DOUBLE PRECISION TMIN,TMAX,TEST,ADJ
      LOGICAL ERROR,LINEAR

      SUMW = VSUM(WE,N)
            
! DECREASING STRATEGY. LOOK FOR A RADIUS THAT GIVES ALL OPACITIES LARGER THAN A THRESHOLD
! ASSUME THAT THE POPULATIONS ARE IN LTE
      IF (KTHICK.EQ.1) THEN		
         DO I = 1,N
            X(I)  = 1./SUMW
            XP(I) = 1./SUMW
         END DO
!        GET DIMENSION SO THAT ALL OPTICAL DEPTHS EXCEED TAUM:
         TMIN = 1.E35
         DO I = 1, N
           DO J = 1, I
              IF (TAUX(I,J).NE.0.) THEN
                 TEST = dabs(TAUX(I,J)*(X(J)*GAP(I,J)-X(I)))
                 IF (TEST.LT.TMIN) TMIN = TEST
              END IF 
           END DO
         END DO
         IR = DLOG10(TAUM/TMIN) + .5
         ADJ = 10.
      ELSE
! INCREASING STRATEGY. LOOK FOR A RADIUS THAT GIVES ALL OPACITIES SMALLER THAN A THRESHOLD
! ASSUME THAT THE POPULATIONS ARE IN THE OPTICALLY THIN LIMIT
         R   = 0.         
         LINEAR = .TRUE.
         d = 0.d0
         f = 0.d0
         CALL SOLVEQ(X,D,F,1,LINEAR,ERROR)
         IF (ERROR) RETURN
         IF (KFIRST.GT.0) THEN
         WRITE (16,"(/,'***  INITIAL GUESS ***')")
             CALL OUTPUT(X,COOL,KFIRST)
         END IF
         IF (NMAX.EQ.0) RETURN
         DO I = 1,N
           XP(I) = X(I)
         END DO
!        NOW GET DIMENSION SO ALL OPTICAL DEPTHS ARE SMALLER THEN TAUM:
         TMAX = 0.
         DO I = 1, N
            DO J = 1, I
              IF (TAUX(I,J).NE.0.) THEN
                 TEST = dabs(TAUX(I,J)*(X(J)*GAP(I,J)-X(I)))
                 IF (TEST.GT.TMAX) TMAX = TEST
              END IF
            END DO
         END DO
         IR = DLOG10(TAUM/TMAX)
         ADJ = .1
      END IF
! ---------------------------------------------------------------
!  NOW, TRY TO GET A FIRST ACTUAL, NON-LINEAR SOLUTION:
      LINEAR = .FALSE.
      R = 10.**IR
      ITER = itmax
      MAX1 = 5
      IF (KFIRST.GT.0) WRITE (16,"('ABOUT TO CALL SOLVE. R = ',1PE10.3)") R
      DO I = 1, MAX1
         CALL SOLVEQ(X,D,F,ITER,LINEAR,ERROR)
         IF (ERROR) THEN
!           DID NOT SOLVE; TRY TO ADJUST DIMENSION UNTIL SOLUTION IS OK:
            R = R*ADJ
         ELSE           
            CALL OUTPUT(X,COOL,KPRT)            
            RETURN
         END IF
      END DO

      END SUBROUTINE FIRST
 
      SUBROUTINE SOLVEQ(X,D,F,ITER,LINEAR,ERROR)
      use global_molpop
      INTEGER ITER,I
      LOGICAL LINEAR,ERROR
      DOUBLE PRECISION X(N),F(N),D(N,N+1)

!     SOLVE THE RATE EQUATIONS AND CHECK THAT SOLUTION IS PHYSICAL
!     THE STARTING POINT IS THE LAST SUCCESSFUL SOLUTION, STORED IN XP
      DO I = 1,N
         X(I) = XP(I)      
      END DO      
      error = .false.
      CALL NEWTON(N,X,F,D,ACC,ITER,LINEAR,NEWTPR,ERROR)      
      IF (ERROR) RETURN
      DO I = 1,N
         POP(I) = X(I)*BOLTZ(I)
         IF (X(I).LE.0.) ERROR = .TRUE.
      END DO
      RETURN
      END SUBROUTINE SOLVEQ


      SUBROUTINE SOLVE(X,D,F,ERROR)
      use global_molpop
      INTEGER I,IOK,ITER
      LOGICAL LINEAR,ERROR
      DOUBLE PRECISION X(N),F(N),D(N,N+1)
      DOUBLE PRECISION RP
      save
      DATA IOK/0/
!
!     XP STORES THE LAST SUCCESSFUL SOLUTION. RP IS THE RADIUS
!     AT THAT SOLUTION.
      RP = R
      DO I = 1,N
        XP(I) = X(I)
      END DO
!     CHECK IF STEP SIZE CAN BE INCREASED.  THIS IS DONE IF THE
!     STEP SIZE IS LESS THAN THE ORIGINAL ONE AND WE HAVE 5
!     SUCCESSIVE SUCCESSFUL SOLUTIONS.  IOK COUNTS THE SUCCESSES.
      IF (NR.GT.NR0.AND.IOK.GE.5) THEN
         NR = NR - 1
         STEP = 10.**(1./NR)
         IF (KTHICK.EQ.1) STEP = 1./STEP
      END IF
!     SCALE DIMENSION AND SOLVE THE STATISTICAL RATE EQUATIONS.
!     UPON FAILURE TO SOLVE, DECREASE THE STEP SIZE AND TRY AGAIN:
      DO WHILE(.TRUE.)
         R = RP*STEP
         ITER = itmax
         CALL SOLVEQ(X,D,F,ITER,LINEAR,ERROR)
         IF (.NOT.ERROR) THEN
            IOK = IOK + 1
            RETURN
         END IF
!        OTHERWISE - DID NOT SOLVE;  FIRST, DECREASE STEP SIZE AND
!        VERIFY THAT IT IS NOT TOO SMALL:
         IOK  = 0
         NR   = NR + 1
         IF (NR.GT.NRMAX) THEN
             ERROR = .TRUE.
             RETURN
         END IF
!        NEW STEP SIZE IS OK. TRY IT:
         STEP = 10.**(1./NR)
         IF (KSOLPR.GT.0) WRITE (16,"(' *** STEP SIZE ADJUSTED TO',F9.4,' AT R =',1PE12.4,' CM')") STEP,R
         IF (KTHICK.EQ.1) STEP = 1./STEP
      END DO

      END SUBROUTINE SOLVE
 


      SUBROUTINE NEWTON(N,X,F,DJAC,ACC,ITMAX,LINEAR,NEWTPR,ERROR)
!      include 'Global.inc'
!     Solves N non-linear (or linear) equations with the Newton-Raphson method
!     Failure is indicated by setting the ERROR flag 
!     NEWTPR IS THE PRINTING PARAMETER: 0 - PRINT NOTHING
!                                       1 - ONLY FAILURE MESSAGES
!                                       2 - PRINT EVERYTHING
      use maths_molpop, only : matin1
    
      INTEGER :: N,ITMAX,ITER,K,IER,IRES,I,J,JMX,INDEX(N)
      INTEGER :: NEWTPR
      LOGICAL ERROR, LINEAR
      DOUBLE PRECISION X(N),F(N),DJAC(N,N+1),AMAX
      DOUBLE PRECISION ACC,CHECK,DET,RES,AF,DETP,AC,A,APR
      DO ITER=1,ITMAX
         CALL EQ(X,F,DJAC,LINEAR)         
         DO K = 1,N
            DJAC(K,N + 1) = F(K)
         END DO
!        SOLUTION OF LINEAR EQUATIONS - FOR FIRST GUESS IN OPTICALLY THIN CASE
         IF (LINEAR) THEN
            CALL MATIN1(DJAC,N,N,N,1,INDEX,IER,DET)
            DO K=1,N
               X(K)=DJAC(K,N + 1)
            END DO
            IF (IER.NE.0.OR.DET.EQ.0.) THEN
              WRITE(16,"('FAILED ON LINEAR EQUATIONS AT R = 0')")
              ERROR = .TRUE.
            END IF
            RETURN
         END IF
!
!  NON-LINEAR EQUATIONS  
!  Solution through Newton's method:
!  Given the populations X and the rates F(X) they define, 
!  new populations are X + dX, where dX = -D^{-1}*F and where D 
!  is the Jacobian of F; that is, dX are the 
!  solution of the linear equations
!
!           D*dX = -F
!
!  This solution is performed by MATIN1. Upon calling,
!  the matrix DJAC contains the Jacobian D in its first N 
!  columns and the vector F in its last column. Upon return 
!  it contains in its first N columns the inverse of D, which 
!  is not needed, and in its last column the vector D^{-1}*F
!
!  FIRST FIND MAXIMUM RESIDUAL
!
         IF (NEWTPR.EQ.2.AND.ITER.EQ.1) then
             WRITE(16,*)
             write(16,*) ' NEWTON PROGRESS:'
             write(16,*) ' ITER.     DET.      EQ. #     RESIDUAL    VAR. #    REL. CHANGE'
         endif
         RES = 0.
         DO K=1,N
            AF = DABS(F(K))
            IF (AF.GT.RES) THEN
               RES  = AF
               IRES = K
            END IF
         END DO
         RES = F(IRES)
!
!  SCALE EACH LINE BY ITS MAXIMUM VALUE TO DECREASE DYNAMICAL RANGE
!
         DO K=1,N
            AMAX=DABS(DJAC(K,K))
            DO J=1,N
               IF(DABS(DJAC(K,J)).GT.AMAX) AMAX=DABS(DJAC(K,J))
            END DO
            IF(AMAX.LT.1.E-20) AMAX=1.E-20
            DO J=1,N + 1
               DJAC(K,J)=DJAC(K,J)/AMAX
            END DO
         END DO  
!
!  NOW INVERT AND CHECK FOR FAILURE
!
         CHECK = 1.
         CALL MATIN1(DJAC,N,N,N,1,INDEX,IER,DET)
         IF (ITER.GT.1) CHECK = DABS(DET/DETP)
         IF (IER.NE.0.OR.DET.EQ.0.OR.CHECK.LT.1.D-5) THEN
!           FAILED:
            ERROR = .TRUE.
            IF (NEWTPR.GT.0) WRITE (16,"(' *** NEWTON: ',I2,' ITERATIONS. DETERMINANT = 0')") ITER
            RETURN
         END IF
!        SUCCESSFUL INVERSION.  CHECK CONVERGENCE - BOTH MAXIMUM RESIDUAL
!        AND RELATIVE CHANGE
         DETP = DET
!        Store the solution dX in the vector F: 
         DO J=1,N
            F(J) = DJAC(J,N + 1)
            X(J) = X(J)-F(J)
         END DO
         AC=0.
         DO J=1,N
            A=DABS(X(J))
            IF(A.GT.1.E-20) A=DABS(F(J)/X(J))
            IF(A.GT.AC) THEN
               AC=A
               JMX=J
            END IF
         END DO
         APR=F(JMX)/X(JMX)
         IF (NEWTPR.EQ.2) WRITE (16,"(3(I3,4X,1PE10.2,5X))") ITER,DET,IRES,RES,JMX,APR
         IF (AC.LT.ACC.AND.DABS(RES).LT.ACC) THEN
         IF (NEWTPR.EQ.2) WRITE (16,"(' SUCCESSFUL SOLUTION')")     
            RETURN
         END IF
      END DO
!
!  FAILED TO SOLVE IN MAXIMUM ALLOWED NUMBER OF ITERATIONS
!
      IF (NEWTPR.GT.0) WRITE (16,"(' *** NEWTON FAILURE. NO CONVERGENCE IN',I3,' ITERATIONS')") ITMAX
      ERROR = .TRUE.
      RETURN
!
      END SUBROUTINE NEWTON





      SUBROUTINE EQ(X,F,D,LINEAR)
!     Level population equations. 
!     From Astronomical Masers, eqs. 2.7.1 and 2.7.2
!     IN THE LINEAR EQUATIONS FOR 1ST GUESS, ONLY F(N) and D(I,J) ARE CALCULATED
!     If overlaptest is false, original non-overlap EQ code is executed
!     If overlaptest is true, line overlap is accounted for using the 
!     Absorption Probability Method (Lockett and Elitzur,ApJ,344,525,1989)
!  
      use global_molpop
      LOGICAL LINEAR
      INTEGER UP,LOW,ISN,I,J,K,L,im,jm,m,lin1,lin2,ip,jip
      DOUBLE PRECISION X(N),F(N),D(N,N+1)
      DOUBLE PRECISION RATE, Del, AUX,AUX2,SUM
      double precision difpopij,pabs,pap,pip,c1,c2,c3,c4,c5,rs,temp
!     p(N_OV,N_OV) are the line absorption probabilites
!     dp(N_OV,N_OV,N_OV) are the derivitives of the absorption probabilities
!     they are passed between subroutines EQ and OVERLAP
      real(kind=8), allocatable :: p(:,:),dp(:,:,:)
      
      allocate(p(num,num))
      allocate(dp(num,num,num))

	  f = 0.d0
	  d = 0.d0
      dp = 0.d0
      p = 0.d0
      
      CALL OPTDEP(X)
      if (.not. overlaptest) then
!
!  ***NO OVERLAP CALCULATION, Use Original EQ code***      
!
!     FIRST THE LOWER N-1 RATE EQUATIONS WITH K THE EQUATION NUMBER:
         DO K = 1,N-1
!        SUM THE CONTRIBUTION OF ALL LEVELS; L IS THE SUMMATION INDEX:
            DO L = 1,N
               IF (K.EQ.L) CYCLE
!              OK, THE LEVELS ARE DIFFERENT; PROCEED. FIRST DECIDE WHICH
!              IS THE HIGHER LEVEL:
               IF (K.LT.L) THEN
                  UP   = L
                  LOW   = K
                  ISN =  1
               ELSE
                  UP   = K
                  LOW   = L
                  ISN = -1
               END IF
!         
!              COLLISIONS:
!           
               IF (.NOT.LINEAR) THEN
                 RATE = ISN*C(UP,LOW)*(X(UP)-X(LOW))
                 F(K) = F(K) + RATE
               END IF
               D(K,UP) = D(K,UP)+ISN*C(UP,LOW)
               D(K,LOW) = D(K,LOW)-ISN*C(UP,LOW)
!         
!              RADIATION:
!         
               IF (A(UP,LOW).EQ.0.) CYCLE
               RATE = ISN*A(UP,LOW)*ESC(UP,LOW)
               IF (TAU(UP,LOW).LT.0.) RATE = RATE*sat
!              CALCULATE THE FUNCTIONS FOR THE NON-LINEAR EQUATIONS:
               IF (.NOT.LINEAR) THEN
                  Del = -X(UP)+X(LOW)*GAP(UP,LOW)
                  If (dustAbsorption) then
                      AUX  = Xd(UP,LOW)*ISN*A(UP,LOW)*(1. - ESC(UP,LOW))
                      IF (TAU(UP,LOW).LT.0.) AUX = AUX*sat
                      RATE = RATE + AUX
                  End If
                  AUX = X(UP)-RAD(UP,LOW)*Del
                  F(K) = F(K) + RATE*AUX
               END IF
               D(K,UP)= D(K,UP)+RATE*(1. + RAD(UP,LOW))
               D(K,LOW)= D(K,LOW)-RATE*RAD(UP,LOW)*GAP(UP,LOW)
!              DERIVATIVE WITH RESPECT TO TAU IN THE NON-LINEAR CASE:
               IF (.NOT.LINEAR) THEN
                  AUX2 = ISN*A(UP,LOW)*AUX*TAUX(UP,LOW)
                  AUX  = DBDTAU(UP,LOW)*R
                  If (dustAbsorption) AUX =  &
                     AUX*(1. - Xd(UP,LOW)) - Xd(UP,LOW)**2*(1. - ESC(UP,LOW))/qdust(UP,LOW)
                  AUX  = AUX*AUX2
                  IF (TAU(UP,LOW).LT.0.) AUX = AUX*sat
                  D(K,UP) = D(K,UP) - AUX
                  D(K,LOW) = D(K,LOW) + AUX*GAP(UP,LOW)
               END IF
            END DO
         END DO
!       
!     THE N-TH TERMS:
!       
         IF (LINEAR) THEN
            F(N) = 1.
         ELSE
            SUM = 0.
            DO J = 1,N
              SUM  = SUM+X(J)*WE(J)
            END DO
            F(N) = SUM-1.
         END IF
         DO J = 1,N
            D(N,J) = WE(J)
         END DO     
      else
!
!  ***Use OVERLAP code***
	
!
      if (.not. LINEAR) call overlap(p,dp)
            
!  FIRST THE LOWER N-1 RATE EQUATIONS WITH K THE EQUATION NUMBER:
      DO 2 K = 1,N-1
!     SUM THE CONTRIBUTION OF ALL LEVELS; L IS THE SUMMATION INDEX:
      DO 2 L = 1,N
         IF (K.EQ.L) GO TO 2
!        OK, THE LEVELS ARE DIFFERENT; PROCEED. FIRST DECIDE WHICH
!        IS THE HIGHER LEVEL:
         IF (K.LT.L) THEN
            UP   = L
            LOW   = K
            ISN =  1
         ELSE
            UP   = K
            LOW   = L
            ISN = -1
         END IF
!
!        COLLISIONS:
!
         IF (.NOT.LINEAR) THEN
           RATE = ISN*C(UP,LOW)*(X(UP)-X(LOW))
           F(K) = F(K) + RATE
         END IF
         D(K,UP) = D(K,UP)+ISN*C(UP,LOW)
         D(K,LOW) = D(K,LOW)-ISN*C(UP,LOW)
         
         
!
!        RADIATION:
!
        IF (A(UP,LOW).EQ.0.) GOTO 2
          difpopij= -X(UP)+X(LOW)*GAP(UP,LOW)
          do  im=1,num
!             Find line number of (UP,LOW) and set = lin1
              if(uplin(im).eq.UP .and. lowlin(im) .eq. LOW) lin1=im
          end do
!       Check to see if lin1 overlaps other lines
      if (p(lin1,lin1).lt.1.e-3)then
!    
!       NEGLECT OVERLAP! Perform usual non-overlap calculations
!
          RATE = ISN*A(UP,LOW)*ESC(UP,LOW)
          IF (TAU(UP,LOW).LT.0.) RATE = RATE*sat
!         CALCULATE THE FUNCTIONS FOR THE NON-LINEAR EQUATIONS:
          IF (.NOT.LINEAR) THEN
            AUX = -X(UP)+X(LOW)*GAP(UP,LOW)
            AUX2 = X(UP)-RAD(UP,LOW)*AUX
            F(K) = F(K) + RATE*AUX2
          END IF
          D(K,UP)= D(K,UP)+RATE*(1. + RAD(UP,LOW))
          D(K,LOW)= D(K,LOW)-RATE*RAD(UP,LOW)*GAP(UP,LOW)
!        DERIVATIVE WITH RESPECT TO TAU IN THE NON-LINEAR CASE:
          IF (.NOT.LINEAR) THEN
            AUX = ISN*A(UP,LOW)*AUX2*DBDTAU(UP,LOW)*R*TAUX(UP,LOW)
            IF (TAU(UP,LOW).LT.0.) AUX = AUX*sat
            D(K,UP) = D(K,UP) - AUX
            D(K,LOW) = D(K,LOW) + AUX*GAP(UP,LOW)
          END IF
          ELSE
!
!      DO OVERLAP CALCULATION
!
!       pabs = Total Absorption Probability for photon emitted in transition (UP,LOW); ie. lin1
        pabs = 0.d0
!       pap = Absorption rate in transition (UP,LOW) from photons emitted in overlapping transitions 
        pap=0.d0 
        pip = isn*a(UP,LOW)*difpopij*rad(UP,LOW)
!  Find Lines that Overlap with lin1
          do 100 ip=2,n
           do 100 jip=1,ip-1
           if (a(ip,jip) .eq. 0) go to 100
             do  im=1,num
                if(uplin(im) .eq. ip .and. lowlin(im) .eq. jip)  lin2=im
               end do 
             if (p(lin1,lin2) .gt. 0) then
!            lin2 Overlaps with lin1 
!             c1..c5 are auxilliary variables used in calculations of Jacobian
              c1=isn*a(ip,jip)*p(lin2,lin1)  
              c2=isn*a(ip,jip)*dp(lin2,lin1,lin2)*x(ip)*taux(ip,jip)*R
              c3=isn*a(UP,LOW)*dp(lin2,lin1,lin1)*x(ip)*taux(UP,LOW)*R 
              c4=dp(lin1,lin2,lin1)*taux(UP,LOW)*R
              c5=dp(lin1,lin2,lin2)*taux(ip,jip)*R
              d(K,ip)=d(K,ip)-c1+c2-c5*pip
              d(K,jip)=d(K,jip)-c2*gap(ip,jip)+c5*pip*gap(ip,jip)
              d(K,UP)=d(K,UP)+c3-c4*pip
              d(K,LOW)=d(K,LOW)-c3*gap(UP,LOW)+c4*pip*gap(UP,LOW)      
              pabs=pabs+p(lin1,lin2)
              pap=pap-c1*x(ip)    
             end if
 100      continue 
!        Note that (1-pabs) is the total escape probability for lin1
           rs=rad(UP,LOW)*(1-pabs)
           TEMP=pap+isn*a(UP,LOW)*(X(UP)-rs*difpopij)
           F(K)=F(K)+TEMP 
           d(K,UP)=d(K,UP)+isn*a(UP,LOW)*(1+rs)
           d(K,LOW)=d(K,LOW)-isn*a(UP,LOW)*rs*gap(UP,LOW) 
         end if
      2 CONTINUE
!
!  THE N-TH TERMS:
!
      IF (LINEAR) THEN
         F(N) = 1.
      ELSE
         SUM = 0.
         DO J = 1,N
           SUM  = SUM+X(J)*WE(J)
         END DO
         F(N) = SUM-1.
      END IF
      DO J = 1,N
         D(N,J) = WE(J)
  END DO
  end if
  
    deallocate(p)
     deallocate(dp)
      RETURN
      END SUBROUTINE EQ


  subroutine overlap(p,dp)
!     Find overlapping lines and calculate the absorption probabilities; p(lin1,lin2)
!     and derivatives; dp/dtau
      use global_molpop
    use maths_molpop
  double precision delta, freqdif,t1ovt2,dtau,p0,p1,p2
      integer i,j,k,l,it,lin,lin1,lin2,m,iter
  integer i2(50),j2(50)
!     p(N_OV,N_OV) are the line absorption probabilites
!     dp(N_OV,N_OV,N_OV) are the derivatives of the absorption probabilities
!     they are passed between subroutines EQ and OVERLAP
      real(kind=8) :: p(:,:),dp(:,:,:)            
    
      do 20 i=1,num
         do 20 j=1,num
  20        p(i,j)=0.
      do 100 i=2,n
    do 100 j=1,n-1
          if (a(i,j) .eq. 0.) goto 100
!      dtau is change in tau used in calculation of numerical derivitive; dp/dtau
!      magnitude of dtau may be changed to improve convergence
         dtau = 2.*tau(i,j)
!      Variable m counts the number of lines overlapping lin(i,j) = lin1
        m=1
        i2(1)=i
        j2(1)=j
!        Doppler width = delta
        delta=freq(i,j)*v/3.e10
      do 50 k=2,n
       do 50 l=1,n-1
        if (i .eq. k .and. j .eq. l) go to 50
        if ( a(k,l) .eq. 0.) go to 50
        freqdif=dabs(freq(i,j)-freq(k,l))
!           neglect overlap if lines are separated by more than 5 Dopplerwidths
        if (freqdif .gt. 5.*delta) go to 50
            m=m+1
        i2(m)=k
            j2(m)=l
   50     continue
          if (m .eq. 1) go to 100
!  Find Absorption Probabilities = p(lin1,lin2) and derivitives dp/dtau = dp(lin1,lin2,lin1) 
     do 200 it=1,m
       do 45 lin=1,num
             if (uplin(lin).eq.i.and.lowlin(lin).eq.j) lin1 = lin
             if (uplin(lin).eq.i2(it).and.lowlin(lin).eq.j2(it))lin2=lin
  45       continue        
     if (p(lin2,lin1) .eq. 0. )then
!   Call Hermite to Integrate Absorption Probabilities 
!           iter = 1 gives p(lin1,lin2)
!           iter = 2 gives p(lin1,lin2) where tau(1) increased by dtau
!           iter = 3 gives p(lin1,lin2) where tau(2) increased by dtau
       do 60 iter = 1,3
      if (iter .eq. 1) p0=hermite(it,dtau,delta,i2,j2,m,iter)/rootpi
      if (iter .eq. 2) p1=hermite(it,dtau,delta,i2,j2,m,iter)/rootpi
      if (iter .eq. 3) p2=hermite(it,dtau,delta,i2,j2,m,iter)/rootpi
            
  60       continue 
       p(lin1,lin2)=p0 
!     Use relation: tau1*p(lin1,lin2)=tau2*p(lin2,lin1)to reduce number of numerical integrations
           t1ovt2=tau(i,j)/tau(i2(it),j2(it))
           p(lin2,lin1)=p(lin1,lin2)*t1ovt2
!     Calculate dp/dtau numerically
           dp(lin1,lin2,lin1)=(p1-p0)/dtau 
           dp(lin1,lin2,lin2)=(p2-p0)/dtau 
!     Find other 2 derivitives analytically
           dp(lin2,lin1,lin1)=t1ovt2*dp(lin1,lin2,lin1)+p(lin1,lin2)/tau(i2(it),j2(it)) 
           dp(lin2,lin1,lin2)=t1ovt2*dp(lin1,lin2,lin2)-p(lin1,lin2)*t1ovt2/tau(i2(it),j2(it)) 
        end if
                
  200   continue
  100   continue        
        return
        end subroutine overlap


end module sol_molpop
