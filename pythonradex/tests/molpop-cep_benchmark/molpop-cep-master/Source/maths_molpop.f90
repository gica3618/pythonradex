! Contains all the mathematical routines and other routines that cope
! with characters
module maths_molpop
use global_molpop
implicit none
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


      function error_message(opt,str)
      character(len=128) :: opt, str
      integer :: iunit
      logical :: error_message

      do iunit = 6,16,10
            WRITE (IUNIT,"(/,'*** Terminated because of INPUT error ***',/, &
            '    Wrong option for ',a,':  ',a)") trim(adjustl(opt)), trim(adjustl(str))
      end do
      error_message = .true.
      return
      end function error_message


      SUBROUTINE OPTDEP(X)
!     CALCULATE THE OPTICAL DEPTHS AND ESCAPE PROBABILITIES
      use global_molpop
      INTEGER I,J
      DOUBLE PRECISION X(N), TAUd

      IF (R.EQ.0.) THEN
         DO I = 2, N
           DO J = 1, I - 1
              TAU(I,J) = 0.
              ESC(I,J) = 1.
           END DO
         END DO
         RETURN
      END IF
!
      DO I = 2, N
        DO J = 1, I - 1
          IF (TAUX(I,J).NE.0.) THEN
            TAU(I,J) = R*TAUX(I,J)*(X(J)*GAP(I,J)-X(I))
            if (dustAbsorption) then
               TAUd     = R*qdust(I,J)
               TAU(I,J) = TAU(I,J) + TAUd
               Xd(I,J)  = TAUd/TAU(I,J)
            end if
            IF (KBETA.EQ.2) THEN
               CALL BETA_SLAB(TAU(I,J),ESC(I,J),DBDTAU(I,J))
            ELSEIF (KBETA.EQ.1) THEN
               CALL BETA_SPHERE(TAU(I,J),ESC(I,J),DBDTAU(I,J))
            ELSEIF (KBETA.EQ.0) THEN
               CALL BETA_LVG(TAU(I,J),ESC(I,J),DBDTAU(I,J))
            ELSEIF (KBETA.EQ.-1) THEN
               CALL BETA_LVG(3.0*TAU(I,J),ESC(I,J),DBDTAU(I,J))
               DBDTAU(I,J) = 3.0 * DBDTAU(I,J)
            END IF
          END IF
        END DO
      END DO

      RETURN
      END SUBROUTINE OPTDEP

      SUBROUTINE BETA_SLAB(TAUIN,BETA,DBDX)
!     SLAB ESCAPE ESCAPE PROBABILITY AND ITS DERIVATIVE
!     WITH RESPECT TO TAU
!     BASED ON KROLIK & MCKEE 1978, ApJS 37, 459; eq 11
!     small- and large-tau expressions are joined at tau = 3.41 instead of 5
!     Since Krolik & McKee expressions are written in terms of line center optical depth
!     we do not have to modify it
!
      use global_molpop
      DOUBLE PRECISION X,TAUIN,BETA,DBDX
      DOUBLE PRECISION B,D,Q,AUX,AUX1,AUX2,COEF
      INTEGER INIT,K
      save coef

      DATA INIT/0/

      IF (INIT.EQ.0) THEN
          COEF = 1./(6.*DSQRT(3.D0))
          INIT = 1
      END IF

      X = TAUIN
!     TAU < 0 - FUDGE FOR INVERTED TRANSITIONS:
      IF (X.LT.0.) THEN
          IF (X.LT.-60.) X=-60.
          BETA = (1.-DEXP(-X))/X
          DBDX = (-1.+(1.+X)*DEXP(-X))/(X*X)
          RETURN
      END IF

!     THERMAL LINE
!
!     TAU CLOSE TO 0
!
      IF (X.LT.1.D-4) THEN
          BETA = 1. - 0.5*X
          DBDX = -0.5
          RETURN
!
!     TAU < 3.41 BUT NOT TOO SMALL
!
      ELSE IF (X.LT.3.41) THEN
          AUX=-0.415+0.355*DLOG(X)
          BETA=1.+X*AUX
          DBDX=0.355+AUX
          K=1
          D=X*COEF
          B=2.*D
          Q=1.
          DO WHILE (Q.GT.1.D-3)
             BETA=BETA-D*X
             DBDX=DBDX-B
             K=K+1
             D=-D*X*DSQRT((K+1.D0)/(K+2.D0))*(K-1.D0)/(K*(K+2.D0))
             B=(K+1.)*D
             Q=DABS(B/DBDX)
          END DO
          RETURN
!
!     TAU > 3.41
!
      ELSE
          AUX1=DLOG(X)
          AUX2=DSQRT(AUX1)
          BETA=(AUX2+0.25/AUX2+0.14)/(ROOTPI*X)
          DBDX=(-BETA+(1-0.25/AUX1)/(2.*X*ROOTPI*AUX2))/X
          RETURN
      END IF
      END SUBROUTINE BETA_SLAB


      SUBROUTINE BETA_SPHERE(TAUIN,BETA,DBDX)
!     STATIC SPHERE ESCAPE PROBABILITY AND ITS DERIVATIVE
!     WITH RESPECT TO TAU
!     From van der Tak et al 2007, A&A 468, 627; eq 19
!     Expression in terms of line center optical depth
!     across the diameter
!
      use global_molpop
      DOUBLE PRECISION X,TAUIN,BETA,DBDX

      X = TAUIN
!     TAU < 0 - FUDGE FOR INVERTED TRANSITIONS:
      IF (X.LT.0.) THEN
          IF (X.LT.-60.) X=-60.
          BETA = (1.-DEXP(-X))/X
          DBDX = (-1.+(1.+X)*DEXP(-X))/(X*X)
          RETURN
      END IF

!     TAU CLOSE TO 0
!
      IF (X.LT.1.D-4) THEN
          BETA = 1. - 0.375*X
          DBDX = -0.375
          RETURN
      END IF    

      BETA = (1.5/X)*(1 - 2./X**2)
      DBDX = -1.5/X**2 + 9./X**4
      IF (X > 50) RETURN
!     AT SMALLER TAU, ADD THE EXPONENIAL TERMS:
      BETA = BETA + (3.*DEXP(-X)/X**2)*(1. + 1./X)
      DBDX = DBDX - (3.*DEXP(-X)/X**2)*(1. + 3./X + 3./X**2)
      RETURN
      END SUBROUTINE BETA_SPHERE


      SUBROUTINE BETA_LVG(TAUIN,BETA,DBDX)
!     THIS IS THE LVG SOBOLEV APPROXIMATION:
      use global_molpop
      DOUBLE PRECISION TAUIN,DBDX,BETA
      DOUBLE PRECISION X,Y,X1
      DOUBLE PRECISION B,Q,AUX,AUX1,AUX2,COEF
      DOUBLE PRECISION E0,E4,E5,E6,F3,P1,P2,P3,P4,P7
      save E0,E4,E5,E6,F3,P1,P2,P3,P4,P7
      INTEGER INIT,I,J
      DATA INIT/0/

! The expressions for the LVG escape probability are in terms of tau_integrated and
! we are working with line center optical depth. Make the change internally in this routine
      X = TAUIN * ROOTPI
!     EPS=1
      IF (EPS.EQ.1.) THEN
           IF (DABS(X).LT.1.E-4) THEN
             BETA =  1. - .5*X
             DBDX = -.5
             RETURN
           ELSE IF (X.GT.40.) THEN
             BETA =  1./X
             DBDX = -BETA*BETA
             RETURN
           ELSE
             IF (X.LT.-50.) X = -50.
             BETA = (1.-DEXP(-X))/X
             DBDX = (-1.+(1.+X)*DEXP(-X))/X**2
             RETURN
         END IF
      END IF
!
!     EPS<1;  FIRST DEFINE SOME NUMERICAL CONSTANTS:
      IF (INIT.EQ.0) THEN
         INIT = 1
         E0=.1
         E4=DSQRT(1.-E0)
         E5=.25*DLOG((1.+E4)/(1.-E4))/E4
         E6=2.*(1+E0/2.)/3.
         P2=PI/2.
         P4=P2**2
         P3=(P4-1.)/(P4+1.)
         P1=(P4+1.)/4.
         P7=P1*(1+E0*P3)
      END IF
!
!     NOW CALCULATE;  FIRST, NEGATIVE TAU:
      IF (X.LT.0.) THEN
         Y = -X/EPS
         IF (Y.LE.4.) THEN
            BETA  =  1.+EPS*DEXP(Y)*Y**2/512.
            DBDX = -Y*(Y+2.)*DEXP(Y)/512.
            RETURN
         END IF

         IF (Y.GT.50.) Y = 50.
         BETA  =  1.+.5*DEXP(Y)*EPS/Y**2
         DBDX = -.5*DEXP(Y)*(Y-2.)/(Y*Y*Y)
         RETURN
      END IF
!
!     NOW POSITIVE TAU, THE OLD MESS:
      IF (X.LT.3.E-3) THEN
         BETA  = 1.
         GO TO 3
        ELSE IF (X.GT.1.) THEN
         BETA=2./(3.*X)
        ELSE
         X1  = DSQRT(1.-X)
         F3  = 2.*(1.-(1.+X/2.)*X1)/(3.*X)
         BETA = F3+X1
         IF (X.LT.0.4) BETA = BETA-2.*X*(.4-X)
      END IF
      IF (X.GT.3.) GO TO 15
      BETA = BETA-.0075*X**2*(3.-X)**4
      IF (X.GT.1.2) GO TO 15
      BETA = BETA-1.2*(X*(1.2-X))**2
      IF (X.GE.1.) GO TO 15
      BETA = BETA*(1.+.5*EPS*X**.3)
      GOTO 3
  15  BETA = BETA*(1.+.5*EPS)

!     THE DERIVATIVE:
   3  IF (X.GT.1.) THEN
         DBDX = -E6/X**2
         IF (X.GT.10.) RETURN
         DBDX = DBDX+.431*DEXP(2.5*(1.-X))
         IF (X.LT.1.4.OR.X.GT.4.) RETURN
         DBDX = DBDX+.005*(X-1.4)*(4.-X)**2
      ELSE
         DBDX = -E5*(1.-X)**1.35-.268989*X**1.15
         IF (X.LT..7) DBDX = DBDX+6.*(X*(.7-X))**2
      END IF
      RETURN
      END SUBROUTINE BETA_LVG

!***********************************************************
!***********************************************************

!***********************************************************
!***********************************************************

        SUBROUTINE MATIN1 (A,DIM1,N1,DIM2,N2,INDEX,NERROR,DETERM)
!     MATRIX INVERSION SUBROUTINE; ORIGIN UNKNOWN; FIXED TYPE DEFINITIONS
!
!     MATRIX INVERSION WITH ACCOMPANYING SOLUTION OF LINEAR EQUATIONS.
!     SOLVES THE MATRIX EQUATION A*X = B FOR THE VECTOR X
!     CALLING SEQUENCE-
!     CALL MATIN1(E,NDIM,N,MDIM,M,INDEX,NERROR,DETERM)
!     PARAMETERS >
!     E - A TWO DIMENSIONAL ARRAY WITH NDIM AS COLUMN SIZE,CONTAINING
!     THE MATRIX OF ORDER N IN ITS FIRST N COLUMNS. THE MATRIX OF
!     CONSTANT VECTORS , B, IS STORED IN COLUMNS N+1 THROUGH N+M OF E
!     NDIM - FIRST DIMENSION PARAMETER OF E AS DECLARED IN THE CALLING
!     PROGRAM
!     N   - THE ORDER OF THE MATRIX A
!     MDIM  - NOT USED ( USED IN EARLIER VERSIONS  )
!     M - NUMBER OF COLUMN VECTORS IN MATRIX B. IF M = 0, ONLY THE
!     INVERSE OF A AND THE COMPUTATION OF THE DETERMINANT IS CARRIED
!     OUT
!     INDEX- A ONE DIMENSIONAL ARRAY CONTAINING N LOCATIONS, USED BY
!     MATIN1 FOR BOOK KEEPING. SPACE MUST BE PROVIDED BY THE CALLING
!     ROUTINE
!     NERROR  - OUTPUT PARAMETER WHICH IS SET ON RETURN TO NON ZERO, IF
!     AT ANY ELIMINATION STEP THE CORRESPONDING COLUMN OF A CONTAINED
!     ONLY ZEROES. A PRINTED MESSAGE IS GIVEN ON FILE TAPE2
!     INDICATING THE COLUMN.
!     DETERM - A SINGLE PRECISSION VARIABLE CONTATINIG ON RETURN THE
!     DETERMINANT OF A
!     ON RETURN, E WILL, IF NERROR = 0 CONTAIN THE INVERSE OF A IN
!     ITS FIRST N COLUMNS AND IF M NOT =  0 THE SOLUTION MATRIX IN
!     THE NEXT FOLLOWING  (N+1) TO (N+M) COLUMNS.
!
      IMPLICIT NONE
      DOUBLE PRECISION A(1),DETER,DETERM,PIVOT,SWAP
      INTEGER DIM,DIM1,DIM2,EMAT,PIVCOL,PIVCL1,PIVCL2
      INTEGER INDEX(1),N,N1,N2,NERROR,NMIN1,MAIN,I,I1,I2,I3,LPIV,ICOL,JCOL
!
      DETER=1.0
      N=N1
      EMAT=N+N2
      DIM=DIM1
      NMIN1=N-1
!     THE ROUTINE DOES ITS OWN EVALUATION FOR DOUBLE SUBSCRIPTING OF
!     ARRAY A.
      PIVCOL=1-DIM
!     MAIN LOOP TO INVERT THE MATRIX
      DO 11 MAIN=1,N
      PIVOT=0.0
      PIVCOL=PIVCOL+DIM
!     SEARCH FOR NEXT PIVOT IN COLUMN MAIN.
      PIVCL1=PIVCOL+MAIN-1
      PIVCL2=PIVCOL +NMIN1
      DO 2 I=PIVCL1,PIVCL2
!       IF(DABS(A(I))-DABS(PIVOT)) 2,2,1
      IF((DABS(A(I))-DABS(PIVOT)) .gt. 0) then
    1   PIVOT=A(I)
        LPIV=I
      endif
    2 CONTINUE
!     IS PIVOT DIFFERENT FROM ZERO
!       IF(PIVOT) 3,15,3
      IF(PIVOT .eq. 0) goto 15
!     GET THE PIVOT-LINE INDICATOR AND SWAP LINES IF NECESSARY
    3 ICOL=LPIV-PIVCOL+1
      INDEX(MAIN)=ICOL
!       IF(ICOL-MAIN) 6,6,4
      IF((ICOL-MAIN) .gt. 0) then
!     COMPLEMENT THE DETERMINANT
    4   DETER=-DETER
!     POINTER TO LINE PIVOT FOUND
        ICOL=ICOL-DIM
!     POINTER TO EXACT PIVOT LINE
        I3=MAIN-DIM
        DO I=1,EMAT
          ICOL=ICOL+DIM
          I3=I3+DIM
          SWAP=A(I3)
          A(I3)=A(ICOL)
          A(ICOL)=SWAP
        END DO
      endif
!     COMPUTE DETERMINANT
    6 DETER=DETER*PIVOT
      PIVOT=1./PIVOT
!     TRANSFORM PIVOT COLUMN
      I3=PIVCOL+NMIN1
      DO I=PIVCOL,I3
        A(I)=-A(I)*PIVOT
      END DO
      A(PIVCL1)=PIVOT
!     PIVOT ELEMENT TRANSFORMED
!
!     NOW CONVERT REST OF THE MATRIX
      I1=MAIN-DIM
!     POINTER TO PIVOT LINE ELEMENTS
      ICOL=1-DIM
!     GENERAL COLUMN POINTER
      DO 10 I=1,EMAT
      ICOL=ICOL+DIM
      I1=I1+DIM
!     POINTERS MOVED
!       IF(I-MAIN) 8,10,8
      IF((I-MAIN) .ne. 0) then
!     PIVOT COLUMN EXCLUDED
    8   JCOL=ICOL+NMIN1
        SWAP=A(I1)
        I3=PIVCOL-1
        DO I2=ICOL,JCOL
          I3=I3+1
          A(I2)=A(I2)+SWAP*A(I3)
        END DO
        A(I1)=SWAP*PIVOT
      endif
   10 CONTINUE
   11 CONTINUE
!     NOW REARRANGE THE MATRIX TO GET RIGHT INVERS
      DO 14 I1=1,N
      MAIN=N+1-I1
      LPIV=INDEX(MAIN)
!       IF(LPIV-MAIN) 12,14,12
      IF((LPIV-MAIN) .ne. 0) then
   12   ICOL=(LPIV-1)*DIM+1
        JCOL=ICOL+NMIN1
        PIVCOL=(MAIN-1)*DIM+1-ICOL
        DO I2=ICOL,JCOL
          I3=I2+PIVCOL
          SWAP=A(I2)
          A(I2)=A(I3)
          A(I3)=SWAP
        END DO
      endif
   14 CONTINUE
      DETERM=DETER
      NERROR=0
      RETURN
   15 NERROR=-1
      DETERM=DETER
      RETURN
      END SUBROUTINE MATIN1


!***********************************************************
!***********************************************************

!***********************************************************
!***********************************************************
      subroutine ordera(array,n,index,jndex,nbig)
!     finds the top nbig elements in matrix;
!     array(index(k),jndex(k)) is the k-th biggest element of the array.

!      use sizes_molpop
      implicit none

      integer index(:),jndex(:),n,nbig,i,j,k,l
      double precision array(:,:),big

! Somehow, this dimensioning doesn't work:
!
!      integer n,nbig,i,j,k,l
!      integer index(nbig),jndex(nbig)
!      double precision array(n,n),big

      if(nbig .gt. n*n) nbig = n*n

!
!       the procedure used here assumes that the diagonal elements are 0:
!
      do i = 1, nbig
        index(i) = 1
        jndex(i) = 1
      end do

      do i = 2,n
        do j = 1,i-1
          do k = nbig, 1, -1
            big = array(index(k),jndex(k))
            if (array(i,j).le.big) go to 13
          end do
   13     if(k .ne. nbig) then
            k = k + 1
            do l = nbig, k + 1, -1
              index(l) = index(l-1)
              jndex(l) = jndex(l-1)
            end do
            index(k) = i
            jndex(k) = j
          end if
        end do
      end do

      return
      end subroutine ordera


      subroutine orderv(v,n,index,nbig)
!     finds the top nbig elements of the vector v;
!     v(index(k)) is the k-th biggest element of v.
      implicit none

      integer n,nbig,index(nbig),i,k,l
      double precision v(n),big

      if(nbig .gt. n) then
        write(16,'(a,i4,a,i4)')'      Cannot find', nbig,' big elements in a vector of order', n
        stop
      end if
!
!             first, order the first nbig elements of v:
!
      index(1) = 1
      do i = 2, nbig
        do k = 1, i-1
          big = v(index(k))
          if(v(i) .gt. big) go to 13
        end do
   13   if(k .ne. i) then
          do l = i, k+1, -1
            index(l) = index(l-1)
          end do
        end if
        index(k) = i
      end do
      if (nbig .eq. n) return
!
!           compare the rest of the elements with the first nbig:
!
      do i = nbig+1, n
        do k = nbig, 1, -1
          big = v(index(k))
          if(v(i) .le. big) go to 23
        end do
   23   if(k .ne. nbig) then
          k = k + 1
          do l = nbig, k+1, -1
            index(l) = index(l-1)
          end do
          index(k) = i
        end if
      end do
      return
      end subroutine orderv


      double precision function vsum(v,n)
!     sum the elements of the vector v:
      implicit none

      integer n,i
      double precision v(n)

      vsum = 0.0
      do i = 1, n
        vsum = vsum + v(i)
      end do
      return
      end function vsum


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc




      subroutine spline(n,x,y,y2)
!     calculates 2nd derivatives for splines
      implicit none
      integer nmax,N_R
      parameter (nmax=500)
      parameter (N_R=100)
      integer n,i,k
      double precision yp1,ypn,x(N_R),y(N_R),y2(N_R),p,qn,sig,un,u(nmax)

      y2(1)=-0.5
      yp1=(y(2)-y(1))/(x(2)-x(1))
      u(1)=(3.0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)

      do i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.0
        y2(i)=(sig-1.0)/p
        u(i)=(6.0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1)) /(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      end do

      qn=0.5
      ypn=(y(n)-y(n-1))/(x(n)-x(n-1))
      un=(3.0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))

      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.0)
      do k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
      end do

      return
      end subroutine spline



      subroutine splint(n,xa,ya,y2a,x,y)
!     interpolates a function with the cubic splines
      implicit none
      integer N_R
      parameter (N_R=100)
      logical i_first
      integer n,khi,klo,j,in
      double precision x,y,xa(N_R),ya(N_R),y2a(N_R),a,b,h
       save j,i_first

      data i_first/.true./



      if(i_first) then
        i_first=.false.
        j=n/2
      end if

      if(x .ge. xa(j)) then
        in=1
        if(j .eq. n-1) goto 50
      else
        in=0
        if(j .eq. 1) goto 50
      end if

      do while (.not. (x .ge. xa(j) .and. x .lt. xa(j+1)))
        if(in .eq. 1) then
          j=j+1
        else
          j=j-1
        end if
        if(j .le. 1) then
          j=1
          goto 50
        end if
        if(j .ge. n) then
          j=n-1
          goto 50
        end if
      end do

  50  continue

      klo=j
      khi=j+1

      h=xa(khi)-xa(klo)
      if(h .eq. 0.0) stop 'bad xa input in splint'

      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h

      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.0

      return
      end subroutine splint


      double precision function bessi0(x)
!     returns the modified Bessel function I_0(x) for any x

      implicit none
      double precision x,ax,p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9,y
      save p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9

      data p1,p2,p3,p4,p5,p6,p7/1.0d0,3.5156229d0,3.0899424d0,1.2067492d0,0.2659732d0,0.360768d-1,0.45813d-2/
      data q1,q2,q3,q4,q5,q6,q7,q8,q9/0.398942228d0,0.1328592d-1,0.225319d-2,-0.157565d-2,&
    0.916281d-2,-0.2057706d-1,0.2635537d-1,-0.1647633d-1,0.392377d-2/

      if(dabs(x) .lt. 3.75) then
        y=(x/3.75)**2
        bessi0=p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))))
      else
        ax=dabs(x)
        y=3.75/ax
        bessi0=(dexp(ax)/dsqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))))
      endif

      return
      end function bessi0



      double precision function bessk0(x)
!     returns the modified Bessel function K_0(x) for x>0

      implicit none
      double precision x,p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,y
      save p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7

      data p1,p2,p3,p4,p5,p6,p7/-0.57721566d0,0.42278420d0,0.23069756d0,0.3488590d-1,0.262698d-2,0.10750d-3,0.74d-5/
      data q1,q2,q3,q4,q5,q6,q7/1.25331414d0,-0.7832358d-1,0.2189568d-1,-0.1062446d-1,0.587872d-2,-0.251540d-2,&
        0.53208d-3/

      if(x .le. 2.0) then
        y=x*x/4.0
        bessk0=(-dlog(x/2.0)*bessi0(x))+p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))))
      else
        y=2.0/x
        bessk0=(dexp(-x)/dsqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))))
      endif

      return
      end function bessk0


      double precision function bessi1(x)
!     returns the modified Bessel function I_1(x) for any x

      implicit none
      double precision x,ax,p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9,y
      save p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,q8,q9

      data p1,p2,p3,p4,p5,p6,p7/0.5d0,0.87890594d0,0.51498869d0,0.15084934d0,0.2658733d-1,0.301532d-2,0.32411d-3/
      data q1,q2,q3,q4,q5,q6,q7,q8,q9/0.39894228d0,-0.3988024d-1,-0.362018d-2,0.163801d-2,-0.1031555d-1,0.2282967d-1,&
       -0.2895312d-1,0.1787654d-1,-0.420059d-2/

      if(dabs(x) .lt. 3.75) then
        y=(x/3.75)**2
        bessi1=x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))))
      else
        ax=dabs(x)
        y=3.75/ax
        bessi1=(dexp(ax)/dsqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))))
        if(x .lt. 0.0) bessi1 = -bessi1
      endif

      return
      end function bessi1



      double precision function bessk1(x)
!     returns the modified Bessel function K_1(x) for x>0

      implicit none
      double precision x,p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7,y
      save p1,p2,p3,p4,p5,p6,p7,q1,q2,q3,q4,q5,q6,q7

      data p1,p2,p3,p4,p5,p6,p7/1.0d0,0.15443144d0,-0.67278579d0,-0.18156897d0,-0.1919402d-1,-0.110404d-2,-0.4686d-4/
      data q1,q2,q3,q4,q5,q6,q7/1.25331414d0,0.23498619d0,-0.3655620d-1,0.1504268d-1,-0.780353d-2,0.325614d-2,&
       -0.68245d-3/

      if(x .le. 2.0) then
        y=x*x/4.0
        bessk1=dlog(x/2.0)*bessi1(x) + (1.0/x)*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))))
      else
        y=2.0/x
        bessk1=(dexp(-x)/dsqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))))
      endif

      return
      end function bessk1



      double precision function bessk(n,x)
!     returns the modified Bessel function K_n(x) for x>0 and n>=2

      implicit none
      integer n,j
      double precision x,bk,bkm,bkp,tox

      if(n .lt. 2) then
        print *, 'bad argument n in bessk'
        stop
      endif

      tox = 2.0/x
      bkm = bessk0(x)
      bk  = bessk1(x)

      do j=1,n-1
        bkp = bkm +j*tox*bk
        bkm = bk
        bk  = bkp
      end do

      bessk = bk

      return
      end function bessk




      integer function inmin(string)
!     returns the minimum element of the character array different from ' '
      implicit none
      character*(*) string
      do inmin=1,len(string)
        if(string(inmin:inmin) .ne. ' ') goto 1
      end do
    1 return
      end function inmin

      integer function inmax(string)
!     returns the maximum element of the character array different from ' '
      implicit none
      character*(*) string
      do inmax=len(string),1,-1
        if(string(inmax:inmax) .ne. ' ') goto 1
      end do
    1 return
      end function inmax

      integer function fsize(string)
!     returns the length of left-aligned string different from ' '
      implicit none
      character*(*) string
      fsize = inmax(string)
    1 return
      end function fsize

      subroutine clear_string(nn,string)
!     fills a character array with ' '
      implicit none
      character*(*) string
      integer i,nn
      do i = 1, nn
        string(i:i) = ' '
      end do
      return
      end subroutine clear_string


      logical function empty(line)
!     checks whether the "line" array contains '%'
      implicit none
      character*(*) line
      empty=.false.
      if(line(inmin(line):inmin(line)) .eq. '%' .or. inmin(line) .ge. inmax(line)) empty=.true.
      return
      end function empty


      subroutine attach2(s1, s2, s)
!     concatenates meaningful strings s1 and s2 into s
      character*(*) s1, s2, s
      s(1:)=s1(inmin(s1):inmax(s1))
      s(inmax(s1)-inmin(s1)+2:)=s2(inmin(s2):inmax(s2))
      end subroutine attach2



      subroutine attach3(s1, s2, s3, s)
!     concatenates meaningful strings s1, s2 and s3 into s
      character*(*) s1, s2, s3, s
      s(1:)=s1(inmin(s1):inmax(s1))
      s(inmax(s1)-inmin(s1)+2:)=s2(inmin(s2):inmax(s2))
      s(inmax(s1)-inmin(s1)+inmax(s2)-inmin(s2)+3:)= s3(inmin(s3):inmax(s3))
      end subroutine attach3


      subroutine file_name(str1,str2,str3)
!     concatenates the part of string str1 before the last symbol
!     '/' or '\' with meaningful part of str2
!     to produce the result in str3
      implicit none
      integer i,i_s1_min,i_s1_max,i_s2_min,i_s2_max,i_slash
      character*(*) str1,str2,str3

      i_s1_min = inmin(str1)
      i_s1_max = inmax(str1)

      i_s2_min = inmin(str2)
      i_s2_max = inmax(str2)

      do i_slash=i_s1_max,i_s1_min,-1
        if(str1(i_slash:i_slash) .eq. char(47)) goto 100
        if(str1(i_slash:i_slash) .eq. char(92)) goto 100
      end do

  100 continue

      if(i_slash .gt. i_s1_min) then
        do i=i_s1_min,i_slash
          str3(i-i_s1_min+1:i-i_s1_min+1) = str1(i:i)
        end do
        do i=i_s2_min,i_s2_max
          str3(i+i_slash-i_s1_min+2-i_s2_min:i+i_slash-i_s1_min+2-i_s2_min) = str2(i:i)
        end do
        return
      else
        do i=i_s2_min,i_s2_max
          str3(i-i_s2_min+1:i-i_s2_min+1) = str2(i:i)
        end do
        return
      end if

      end subroutine file_name

! ======================================
! SEVERAL SIMPLE TESTS OF STRING CONTENT
! ======================================
    function chr(card, i)
    character card*(232)
    integer i
    character chr
      chr = card(i:i)
      return
    end function chr

    function bar(cr)
    character cr
    logical bar
      bar = cr .eq. ' '
      return
    end function bar


    function digit(cr)
    character :: cr
    logical :: digit
      digit = cr .ge. '0' .and. cr .le. '9'
      return
    end function digit

    function minus(cr)
    character :: cr
    logical :: minus
      minus = cr .eq. '-'
      return
    end function minus

    function sgn(cr)
    character :: cr
    logical :: sgn
      sgn = cr .eq. '+' .or.  cr .eq. '-'
      return
    end function sgn

    function dot(cr)
    character :: cr
    logical :: dot
      dot = cr .eq. '.'
      return
    end function dot

    function ee(cr)
    character :: cr
    logical :: ee
      ee = cr .eq. 'e' .or.  cr .eq. 'E' .or. cr .eq. 'd' .or.  cr .eq. 'D'
      return
    end function ee

    function ival(cr)
    character :: cr
    integer :: ival
      ival = ichar(cr) - ichar('0')
      return
    end function ival

! =======================================================================
! SEVERAL SIMPLE FUNCTIONS THAT RETURN SOME STRING PARAMETERS
! =======================================================================

      subroutine rdinps(equal,iunit,str)
!     reads strings of symbols
      integer i,iunit,ind,first,last,next
      character card*(232),cr
      character*(*) str
      logical equal !,bar
      save card, first, last
      data first/1/, last/0/
!
!     Function statements
!
!!!      chr(i)    = card(i:i)
!!!      bar(cr)   = cr .eq. ' '
!
      if(iunit .lt. 0) then
        first = last + 1
        iunit = -iunit
      end if
!        start the search for the next string:
  1   continue
      if(first .gt. last) then
        read(iunit, '(a)', end = 99) card
        first = 1
        if(chr(card,first) .eq. '*') write(16,'(a)') trim(adjustl(card))
        last = len(card)
        ind = index(card,'%')
        if(ind .gt. 0) last = ind-1
      end if
      if(equal) then
        do while (chr(card,first) .ne. '=')
          first = first + 1
          if(first .gt. last) goto 1
        end do
      end if
      first = first + 1

      do while (bar(chr(card,first)) .and. first .le. last)
        first = first + 1
      end do
      if(first .gt. last) goto 1

      next=first+1
      do while( .not. bar(chr(card,next)) )
        next = next + 1
      end do
      str=card(first:next-1)
      return

99    write(16,'(3(1x,a,/))') ' Terminated. EOF reached by rdinp while looking for input. ',' last line read:',card
      return
      end subroutine rdinps


      subroutine rdinps2(equal,iunit,str,LENGTH,UCASE)
!     2nd version. Returns also length of meaningful string; if UCASE flag
!     is set, the returned string is in upper case to avoid keyboard entry problems
      integer i,iunit,ind,first,last,next,length
      character card*(232),cr
      character*(*) str, ch*1
      logical equal,UCASE
      save card, first, last
      data first/1/, last/0/
!
!     Function statements
!
!!!      chr(i)    = card(i:i)
!!!      bar(cr)   = cr .eq. ' '
!
      if(iunit .lt. 0) then
        first = last + 1
        iunit = -iunit
      end if
!        start the search for the next string:
  1   continue
      if(first .gt. last) then
        read(iunit, '(a)', end = 99) card
        first = 1
        if(chr(card,first) .eq. '*') write(16,'(a)') trim(adjustl(card))
        last = len(card)
        ind = index(card,'%')
        if(ind .gt. 0) last = ind-1
      end if
      if(equal) then
        do while (chr(card,first) .ne. '=')
          first = first + 1
          if(first .gt. last) goto 1
        end do
      end if
      first = first + 1

      do while (bar(chr(card,first)) .and. first .le. last)
        first = first + 1
      end do
      if(first .gt. last) goto 1

      next=first+1
      do while( .not. bar(chr(card,next)) )
        next = next + 1
      end do
      str=card(first:next-1)
  length = next - first

  if (UCASE) then
!        convert string to UPPER CASE
         DO i = 1, length
            ch = str(i:i)
            IF ((ch .GE. 'a') .AND. (ch .LE. 'z'))str(i:i) = Char(IChar(ch)+IChar('A')-IChar('a'))
         END DO
      end if
      return

99    write(16,'(3(1x,a,/))') ' Terminated. EOF reached by rdinp while looking for input. ',' last line read:',card
      return
      end subroutine rdinps2


      function to_upper(strIn) result(strOut)
!     Adapted from http://www.star.le.ac.uk/~cgp/fortran.html (25 May 2012)
      implicit none
      character(len=*), intent(in) :: strIn
      character(len=len(strIn))    :: strOut
      integer :: i,j
  
      do i = 1, len(strIn)
           j = iachar(strIn(i:i))
           if (j>= iachar("a") .and. j<=iachar("z") ) then
                strOut(i:i) = achar(iachar(strIn(i:i))-32)
           else
                strOut(i:i) = strIn(i:i)
           end if
      end do
      
      end function to_upper



!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!     This function is taken from Moshe Elitzur.           [Z.I., Nov. 1995]

      double precision function rdinp(equal, iunit, outUnit)
! =======================================================================
!     Read lines, up to 232 long, from pre-opened unit IUNIT and extract
!     all input numbers from them. When EQUAL is set, numeric input data
!     must be preceded by an equal sign.All non-numeric data and numbers
!     not preceded by = when EQUAL is on are ignored.RDINP = next number
!     encountered (after equal sign) and terminated by a nun-numeric
!     symbol (termination with blank is best). Commas and exponential
!     notation are allowed.  All text after % is ignored, as in TeX.
!     Lines with * in the first column are echoed to the output device.
!     The number is comprised of an actual VALUE, decimal part FRAC and
!     (integer) exponent PWR.  It has a SIGN, and the exponential power
!     has sign SIGNEX. Logical flag to the decimal part is set in
!     DECIMAL. The search is conducted between FIRST, which is
!     continuously increased, and LAST.  A new line is read when FIRST
!     exceeds LAST, and can be forced by calling with -IUNIT.  Actual
!     extraction of numerical value is done in separate FUNCTION VAL.
! =======================================================================
      implicit none
      integer iunit, outunit, ind, first, last
      double precision value,frac,pwr,sign,signex
      character card*(232),cr,prev,term,next
      logical, intent(in) :: equal
      logical decimal
      save card,first,last
      data first/1/, last/0/

!    F90 doesn't like in-line functions. The following have been redefined
!    as regular external functions:
!!!      digit(cr) = cr .ge. '0' .and. cr .le. '9'
!!!      minus(cr) = cr .eq. '-'
!!!      sgn(cr)   = cr .eq. '+' .or.  cr .eq. '-'
!!!      dot(cr)   = cr .eq. '.'
!!!      e(cr)     = cr .eq. 'e' .or.  cr .eq. 'E' .or. cr .eq. 'd' .or.  cr .eq. 'D'

      if(iunit .lt. 0) then
        first = last + 1
        iunit = -iunit
      end if

!     start the search for the next number:
1     rdinp  = 0.
      value  = 0.
      frac   = 0.
      pwr    = 0.
      sign   = 1.
      signex = 1.
      decimal = .false.
      if(first .gt. last) then
!       time to get a new line
        read (iunit, '(a)' , end = 99) card

        first = 1
        last = len(card)
!       find start of trailing junk:
        do while(card(last:last) .le. ' ')
           last = last - 1
           if(last .lt. first) goto 1
        end do
        if(card(first:first).eq.'*') write (outUnit,'(a)') card(1:last)
        ind = index(card,'%')
        if(ind .gt. 0) last = ind - 1
      end if

!     get past the next '=' when the equal flag is set
      if(equal) then
         do while (card(first:first).ne.'=')
            first = first + 1
            if (first.gt.last) goto 1
         end do
      end if
!     ok, start searching for the next digit:
      do while (.not.digit(card(first:first)))
         first = first + 1
         if (first.gt.last) goto 1
      end do
!     check if it is a negative or decimal number
      if (first.gt.1) then
         prev = card(first-1:first-1)
         if (minus(prev)) sign = -1.
         if (dot(prev)) then
            decimal = .true.
            if (first.gt.2 .and. minus(card(first-2:first-2))) sign = -1.
         end if
      end if
!     extract the numerical value
      if (.not.decimal) then
         value = val_dp(card,first,last,decimal,term)
!        check for a possible decimal part.  termination with '.e' or
!        '.d' is equivalent to the same ending without '.'
         if (first.lt.last.and.dot(term)) then
            first = first + 1
            next = card(first:first)
            if (digit(next)) decimal = .true.
            if (ee(next)) term = 'e'
         end if
      end if
!     extract the decimal fraction, when it exists
      if (decimal) frac = val_dp(card,first,last,decimal,term)
!     an exponential may exist if any part terminated with 'e' or 'd'
      if(first .lt. last .and. ee(term)) then
         first = first + 1
         next = card(first:first)
         if(first .lt. last .and. sgn(next))then
            first = first + 1
            if(minus(next)) signex = -1.
         end if
         decimal = .false.
         pwr = val_dp(card,first,last,decimal,term)
      end if
!     finally, put the number together
      rdinp = sign*(value + frac)*10**(signex*pwr)

      return

99    write (outUnit,'(3(1x,a,/))') &
      ' ****Terminated. EOF reached by rdinp while looking for input. ',' *** Last line read:',card

      return
      end function rdinp


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


! ***********************************************************************
! This function is taken from Moshe Elitzur            [Z.I., Nov. 1995]
! =======================================================================
      double precision function val_dp(card,first,last,decimal,term)
!     extract numerical value from card, begining at position first up
!     to the first non-digit encountered.  The terminating character is
!     returned in term, its position in first. Commas are treated as
!     simple separators.

      implicit none
      character card*(*), term, ch
      logical decimal !, digit
!       integer ival
      integer first, last, first0
      double precision pwr

! !       ival(ch)  = ichar(ch) - ichar('0')
! !       digit(ch) = ch .ge. '0' .and. ch .le. '9'

      val_dp = 0.0
      pwr = 1.0
      first0 = first
      do first = first0, last
        term = card(first:first)
        if(term .ne. ',') then
          if(.not. digit(term)) return
          if(decimal) then
            pwr = pwr*0.1
            val_dp = val_dp + pwr*ival(term)
          else
            val_dp = 10.0*val_dp + ival(term)
          end if
        end if
      end do
      term = ' '

      return
      end function val_dp


      double precision function hermite(it,dtau,delta,i2,j2,m,iter)
!     Hermite numerically integrates the absorption probabilities using 16 point Gaussian quadrature
      use global_molpop
  double precision temp,d,dtau,delta,tautot,z,f
  integer it,ip,iter,ll,i
  integer m,i2(50),j2(50)
    double precision arg(50),tauf(50,50)
!  x(i) and w(i) are the abscissas and weights from Abramowitz and Stegun
  double precision x(8),w(8)
      data (x(i),i=1,8)/.27348104613815d0,.82295144914466d0,1.38025853919888,&
       1.95178799091625d0,2.54620215784748d0,3.17699916197996d0,&
       3.86944790486012d0,4.68873893930582d0/
      data (w(i),i=1,8)/5.0792947901d-1,2.806474585285d-1,&
       8.381004139899D-2,1.288031153551D-2,9.322840086242D-4,&
       2.711860092538D-5,2.320980844865D-7,2.654807474011d-10/
! hermite = absorption probability found using eq. 2.12 of Lockett and Elitzur,ApJ,344,525,1989
       hermite=0.d0
        do 100 i=1,8
           do 90 ll=1,2
              if (ll .eq. 1) then
                f=x(i)*delta+freq(i2(1),j2(1))
              else
                f=-x(i)*delta+freq(i2(1),j2(1))
         end if
         tautot=0.d0
!          Add up taus of overlapping lines to get tautot
              do 80 ip=1,m
                 arg(ip)=((f-freq(i2(ip),j2(ip)))/delta)**2
             if (iter .eq. 1) z=tau(i2(ip),j2(ip))
                 if (iter .eq. 2 ) then
                 if (ip .eq. 1) then
                        z=tau(i2(ip),j2(ip))+dtau
                      else
                        z=tau(i2(ip),j2(ip))
                     end if
                 end if
                 if (iter .eq. 3 ) then
                if (ip .eq. it) then
                        z=tau(i2(ip),j2(ip))+dtau
                     else
                        z=tau(i2(ip),j2(ip))
                     end if
                 end if
!                 tauf(i2(ip),j2(ip))=z*dexp(-arg(ip))/rootpi
!  (11 March 05) removed division by rootpi to agree with line center tau
                  tauf(i2(ip),j2(ip))=z*dexp(-arg(ip))
             tautot=tautot+tauf(i2(ip),j2(ip))
   80          continue
          temp =1.d0-(.5d0-expint(3,tautot))/tautot
          d=temp*tauf(i2(it),j2(it))/tautot
          hermite=hermite+w(i)*d
   90     continue
  100   continue
        return
        end function hermite


       double precision function expint(n,x)
!  This routine calculates the 3rd exponential integral function
!  Press and Teukolsky, Computers in Physics, Vol 2, No.5, p88. 1988.
       implicit double precision (a-h,o-z)
     integer :: itmax, i, n, nm1, ii
     real(kind=8) :: del, d, c
       parameter (itmax=100,eps=1.d-7,tiny=1.d-30,gamma=.5772156649)
       nm1=n-1
       if (x.gt.1.d2)then
           expint=0.d0
           return
       end if
       if (x .lt.0)  x=0
       if(n.lt.0.or.x.lt.0.d0.or.(x.eq.0.d0.and.(n.eq.0.or.n.eq.1)))then
             print *, 'bad arguments', n, x
             stop
       else if (n.eq.0)then
          expint=dexp(-x)/x
       else if (x.eq.0.d0) then
          expint=1.d0/nm1
       else if (x.gt.1.d0) then
          f=1.d0/x
          d=f
          c=1.d0/tiny
          do 11 i=1,itmax
             an=i+nm1
             d=1.d0/(1.d0+an*d)
             c=1.d0+an/c
             del=d*c
             f=f*del
             d=1.d0/(x+i*d)
             c=x+i/c
             del=d*c
             f =f*del
             if(dabs(del-1.d0).lt.eps)then
               expint=f*dexp(-x)
                return
             end if
  11       continue
           stop
         else
           if(nm1.ne.0) then
             expint=1.d0/nm1
           else
           expint=-dlog(x)-gamma
           end if
         fact=1.d0
         do 13 i=1,itmax
            fact=-fact*x/i
            if(i.ne.nm1) then
               del=-fact/(i-nm1)
             else
               psi=-gamma
               do 12 ii=1,nm1
                 psi=psi+1.d0/ii
  12          continue
             del=fact*(-dlog(x)+psi)
            end if
            expint=expint+del
            if(dabs(del).lt.dabs(expint)*eps) return
  13        continue
            stop
              endif
           return
           end function expint


  subroutine numlin
!      This routine finds the number of lines (num) and labels them by upper and lower level
   use global_molpop
   integer i,j

    num=0

      do i=1,n
        do j=1,i-1
          if (a(i,j) >= 1.e-16) then
            num = num+1
          endif
        enddo
    enddo

    allocate(uplin(num))
    allocate(lowlin(num))

    num=0

      do i=1,n
        do j=1,i-1
          if (a(i,j) >= 1.e-16) then
            num = num+1
            uplin(num) = i
              lowlin(num) = j
          endif
        enddo
    enddo


1000   continue
  return
  end subroutine numlin


      double precision function part_func(tt,acon,jmax)
!     calculates the partition function for rotational transitions
      implicit none
      integer jmax,j
      double precision tt,acon,a1
      part_func=0.0
      a1=acon/tt
      do j=0, jmax
        part_func=part_func+(2*j+1)*dexp(-a1*j*(j+1))
      end do
      part_func=part_func*a1
      return
      end function part_func


      double precision function threej(j1,j2,j)
!     calculates the Wigner 3j-coefficients
      implicit none
      double precision factor,sum
      integer j1,j2,j,par,z,zmin,zmax

!     some checks for validity (let's just return zero for bogus arguments)

      if(2*(j1/2)-int(2*(j1/2.0)) .ne. 0 .or. 2*(j2/2)-int(2*(j2/2.0)) .ne. 0 .or. 2*(j/2)-int(2*(j/2.0)) .ne. 0 .or.&
        j1 .lt. 0 .or. j2 .lt. 0 .or. j .lt. 0 .or. j1+j2 .lt. j .or. abs(j1-j2) .gt. j) then
          threej= 0.0
      else
        factor = 0.0
        factor = binom(j1,(j1+j2-j)/2) / binom((j1+j2+j+2)/2,(j1+j2-j)/2)
        factor = factor * binom(j2,(j1+j2-j)/2) / binom(j1,j1/2)
        factor = factor / binom(j2,j2/2) / binom(j,j/2)
        factor = dsqrt(factor)

        zmin = max(0,(j2-j)/2,(j1-j)/2)
        zmax = min((j1+j2-j)/2,j1/2,j2/2)

        sum=0.0
        do z = zmin,zmax
          par=1
          if(2*(z/2)-int(2*(z/2.0)) .ne. 0) par=-1
          sum=sum+par*binom((j1+j2-j)/2,z)*binom((j1-j2+j)/2,j1/2-z)*binom((-j1+j2+j)/2,j2/2-z)
        end do
        threej = factor*sum/(-1)**((j1-j2)/2)/sqrt(j+1.0)
      end if

      return
      end function threej

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      double precision function binom(n,r)
!     calculates Newton binom
      implicit none
      integer n,r,i

      if(r .eq. n .or. r .eq. 0) then
        binom = 1.0
      else if(r .eq. 1) then
        binom = dble(n)
      else
        binom=1.0
        do i=1,r
          binom=binom*dble(n-r+i)/dble(i)
        end do
      end if

      return
      end function binom


      subroutine const
!     defines physical, astronomical and mathematical constants
      use global_molpop
      integer k
!     constants (all units are c.g.s.):
!
!     cl  -      speed of light
!     hpl -      Planck's constant
!     bk  -      Boltzmann's constant
!     xme -      electron mass
!     xmp -      proton mass
!
      cl  = 2.99792458d10
      hpl = 6.62606876d-27
      bk  = 1.3806503d-16
      xme = 9.10938188d-28
      xmp = 1.673e-24
!
!     astronomical constants (from Allen):
!
!     solarl -   Solar luminosity
!     solarm -   Solar mass
!     solarr -   Solar radius
!     pc     -   parsec
!
      solarl  = 3.90e33
      solarm  = 1.99e33
      solarr  = 6.96e10
      pc      = 3.086e18

      pi      = 3.141592654d0
      twopi   = 2.0*pi
      fourpi  = 4.0*pi
      eitpi   = 8.0*pi
      rootpi  = dsqrt(pi)


      return
      end subroutine const


      double precision function prlog(x)
!     calculates log10(x) with proper safeguards
      implicit none

      double precision x

      if(x .le. 0.0) then
        prlog = 0.0
      else
        prlog = dlog10(x)
      end if
      return
      end function prlog



! =================  Stuff related to Planck function  ====================


      double precision function plexp(x)
!-------------------------------------------------------
!     calculates the Planck function, modulo 2h*nu^3/c^2
!     That is:   plexp = 1/[exp(x) - 1]
!-------------------------------------------------------
         implicit none
         double precision x

         if(x .eq. 0.0) then
           write(16,'(6x,a)') 'ERROR! Function plexp called with argument zero.'
           plexp = 1.d100
         else if(x .gt. 50.0) then
           plexp = 0.0
         else if(dabs(x) .lt. 0.001) then
           plexp = 1.0/x
         else
           plexp = 1.0/(dexp(x) - 1.0)
         end if
         return
      end function plexp


      double precision function Inv_plexp(P)
!----------------------------------------------------------------
!     Finds the argument of the Planck function given its value P
!     That is, solves the equation P = 1/[exp(x) - 1]
!----------------------------------------------------------------
         implicit none
         double precision, intent(in) :: P

         if (P > 1.e3) then  ! might as well use small x (RJ) limit
             inv_plexp = 1./P
         else
             inv_plexp = DLOG(1. + 1./P)
         end if
         return
      END function Inv_plexp


  double precision function BB(nu, Temp)
!----------------------------------------------------------------
!     Planck function in mks units W/m^2/Hz/ster:
!        W = 1E7 erg/sec   m^2 = 1E4 cm^2
!        therefore 
!        W/m^2 = 1E3 erg/cm^2
!----------------------------------------------------------------

  implicit none
  double precision, intent(in) :: nu, Temp
      BB = 1.d-3*(2.*hpl*nu**3/cl**2)*plexp(hpl*nu/bk/Temp)
      
  end function BB




      Subroutine Tbr4Tx(Tl,Tx,taul,Tbr,TRJ)
!----------------------------------------------------------------
!     For a line with temperature-equivalent frequency Tl
!     enter with excitation temperature Tx and optical depth taul
!     calculate brightness temperature from
!
!        B(Tbr) = [B(Tx) - B(Tcmb)]*[1 - exp(-taul)]
!
!     All intensitie are in photon occupation number because
!     we use plexp for B(T); so B(Tbr) is simply TRJ/Tl
!     where TRJ is the Rayleigh Jeans equivalent T
!----------------------------------------------------------------
         implicit none
         double precision, intent(in)  :: Tl, Tx, taul
         double precision, intent(out) :: Tbr, TRJ
         double precision B
         integer sgn

         if (Tx == Tcmb) then
            TRJ = 0.
            Tbr = 0.
            return
         end if


         B = (plexp(Tl/Tx) - plexp(Tl/Tcmb)) * (1. - dexp(-taul))

!        negative B means Tx < Tcmb so we get absorption line; negative Tbr
         sgn = 1
         if (B < 0.d0) sgn = -1

         TRJ = Tl*B
         Tbr = sgn*Tl/Inv_plexp(dabs(B))
         return
      END Subroutine Tbr4Tx


      Subroutine Tbr4I(nu,I,taul,Tbr,TRJ)
!------------------------------------------------------------
!     For a line with frequency nu
!     enter with intensity I and optical depth taul
!     calculate brightness temperature from
!
!        B(Tbr) = I - B(Tcmb)*[1 - exp(-taul)]
!
!     All intensities are converted to photon occupation number
!     For B(T) we use plexp, I is converted with 2h*nu^3/c^2
!     Then the RJ tempearture is simply TRJ = Tl*B(Tbr)
!-------------------------------------------------------------
         implicit none
         double precision, intent(in)  :: nu, I, taul
         double precision, intent(out) :: Tbr, TRJ
         double precision B, Tl, Intensity
         integer sgn

         Tl = hPl*nu/Bk
         Intensity = I/(2*hPl*nu**3/cl**2)

         B = Intensity - plexp(Tl/Tcmb) * (1. - dexp(-taul))

         if (B == 0.d0) then
            TRJ = 0.
            Tbr = 0.
            return
         end if

!        negative B means we get absorption line; negative Tbr
         sgn = 1
         if (B < 0.d0) sgn = -1

         TRJ = Tl*B
         Tbr = sgn*Tl/Inv_plexp(dabs(B))
         return
      END Subroutine Tbr4I

!========================================================================



! ***********************************************************************
      SUBROUTINE SIMPSON(N,N1,N2,x,y,integral)
! =======================================================================
! This subroutine calculates integral I(y(x)*dx). Both y and x are
! 1D arrays, y(i), x(i) with i=1,N (declared with NN). Lower and upper
! integration limits are x(N1) and x(N2), respectively. The method used
! is Simpson (trapezoid) approximation. The resulting integral is sum of
! y(i)*wgth, i=N1,N2.                                  [Z.I., Mar. 1996]
! =======================================================================
      IMPLICIT none
      INTEGER i, N, N1, N2
      DOUBLE PRECISION x(N), y(N), wgth, integral
! -----------------------------------------------------------------------
!     set integral to 0 and accumulate result in the loop
      integral = 0.0
!     calculate weight, wgth, and integrate in the same loop
      IF (N2.GT.N1) THEN
        DO i = N1, N2
!         weigths
          IF (i.NE.N1.AND.i.NE.N2) THEN
            wgth = 0.5 * (x(i+1)-x(i-1))
          ELSE
            IF (i.eq.N1) wgth = 0.5 * (x(N1+1)-x(N1))
            IF (i.eq.N2) wgth = 0.5 * (x(N2)-x(N2-1))
          END IF
!         add contribution to the integral
          integral = integral + y(i) * wgth
        END DO
      ELSE
        integral = 0.0
      END IF
! -----------------------------------------------------------------------
      RETURN
      END subroutine simpson
! ***********************************************************************


!=======================================================================================!
!
! Read a tabulation from a file and return in array out(i,j) the tabulation
! at the wavelength positions wl(i,j)
! The file may contain many columns (the DUSTY output does) but
! only the first two count: 1st is wavelength, 2nd the tabulated quantity
! When norm is on, the tabulation is for the SED lambda*Flambda and the flux
! Flambda that is read in is re-normalized to bolometric unity
!
!=======================================================================================!
    subroutine interpolateExternalFile(fileName, wl, out, norm, error)
    implicit none
    character*(*) fileName
    real(kind=8) :: wl(:,:), out(:,:), f1, f2, scale
    real(kind=8), allocatable :: w(:), F(:)
    integer :: i, Nin, nx, ny, j, k
    logical :: norm, stat, error

      nx = size(wl,1)
      ny = size(wl,2)

      inquire(file=trim(adjustl(fileName)),exist=stat)
      if (.not.stat) then
        do k = 6,16,10
           write(k,"(/'**** Error opening file ',a)") trim(adjustl(fileName))
        end do
        error = .true.
        return
      endif

      open(10,file=trim(adjustl(fileName)), status='old')

! First count the number of data lines in the file
      call Pass_Header(10)
      Nin = 0
      do while(.true.)
        read(10,*, end=2)
        Nin = Nin + 1
      enddo
2     continue

      allocate(w(Nin))
      allocate(F(Nin))

! Now read the data:
      rewind(10)
      call Pass_Header(10)

      do i = 1, Nin
        read(10,*, end=1) w(i), F(i)
        if (w(i) <= 0. .or. F(i) <0.) then
           write(16,"(5x,'Bad entry in file ',a,/,5x, &
             'wavelength = ',ES9.2,' mic; tabulation = ',ES9.2,/,5x, &
             'wavelength must be positive and the entry non-negative')") trim(adjustl(fileName)), w(i), F(i)
           error = .true.
           return
        end if
      end do
1     close(10)

      IF (norm) then
!        we need to re-normalize the tabulated F (which is lambda*Flambda) to
!        unity bolometric flux; namely, scale by the integral of F/w:

         call Simpson(Nin,1,Nin,w,F/w,scale)
         F = F/scale
      End If

      do i = 1, nx
        do j = 1, ny
!           interpolate between the tabulation elements
            k=1
            if (wl(i,j) /= 0) then
              if (wl(i,j) < w(1)) then
                out(i,j) = F(1)
              else if (wl(i,j) > w(Nin)) then
                out(i,j) = F(Nin)
              else
                do while(.not.(wl(i,j) >= w(k) .and. wl(i,j) <= w(k+1)))
                  k = k+1
                enddo
                f2 = (wl(i,j)-w(k))/(w(k+1)-w(k))
                f1 = 1.0-f2
                out(i,j) = F(k)*f1+F(k+1)*f2
              endif
            endif
        enddo
      enddo
      deallocate(w,F)

    return
    end subroutine interpolateExternalFile


    subroutine rad_file(fname,Jbol,str,L,error)
!   add radiation from file
    use global_molpop
    integer i,j, L
    real(kind=8) :: F(N,N)
    real(kind=8) :: Jbol, factor
    character*(*) :: fname, str
    logical norm/.true./
    logical error

!   Read SED of external radiation from file such as, e.g., DUSTY output
!   1st column is wavelength (in micron)
!   2nd column is spectral shape lambda*F_lambda
!   Normalize the spectral shape to unit bolometric flux and scale it by Jbol

    call interpolateExternalFile(trim(adjustl(fname)), wl, F, norm, error)
    if (error) return

!   convert the normalized lambda*F_lambda to photon occupation number:
!   multiply by Jbol and divide by nu to get J_nu; recall lambda*F_lambda = nu*F_nu
!   divide by 2h*nu^3/c^2 to get photon occupation number
!   All in all get c^2/nu^4 = (lambda/nu)^2, which may be better for numerics

    factor = Jbol/(2.*hpl)

    do i = 2, N
      do j = 1, i-1
         F(i,j) = F(i,j)*factor*(1.e-4*wl(i,j)/freq(i,j))**2
         rad(i,j) = rad(i,j) + F(i,j)

!   Coming from the left
         if (str(1:L) == 'LEFT') then
           rad_tau0(i,j) = rad_tau0(i,j) + F(i,j)
         endif

!   Coming from the right
         if (str(1:L) == 'RIGHT') then
            rad_tauT(i,j) = rad_tauT(i,j) + F(i,j)
         endif

!   Coming from both sides
         if (str(1:L) == 'BOTH') then
            rad_tau0(i,j) = rad_tau0(i,j) + F(i,j)
            rad_tauT(i,j) = rad_tauT(i,j) + F(i,j)
         endif

!   Internal radiation
         if (str(1:L) == 'INTERNAL') then
            rad_internal(i,j) = rad_internal(i,j) + F(i,j)
         endif

      end do
    end do

    return
    end subroutine rad_file



  subroutine dust_rad(Td,tau_d,str,L)
! calculates dust radiation properties, occupation numbers
! according to eq.1 from Lockett et al, 1999, ApJ, 511, 235
! tau_d is the dust optical depth at V
!
    use global_molpop
    integer i,j,k, L
    character*(*) :: str
    double precision Td, tau_d, taulamb

    do i=2, n
      do j=1, i-1
        taulamb  = tau_d*qdust(i,j)
        rad(i,j) = rad(i,j)+(1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)

!   Coming from the left
        if (str(1:L) == 'LEFT') then
          rad_tau0(i,j) = rad_tau0(i,j) + (1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)
        endif

!   Coming from the right
        if (str(1:L) == 'RIGHT') then
          rad_tauT(i,j) = rad_tauT(i,j) + (1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)
        endif

!   Coming from both sides
        if (str(1:L) == 'BOTH') then
          rad_tau0(i,j) = rad_tau0(i,j) + (1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)
          rad_tauT(i,j) = rad_tauT(i,j) + (1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)
        endif

!   Internal radiation
        if (str(1:L) == 'INTERNAL') then
          rad_internal(i,j) = rad_internal(i,j) + (1.0-dexp(-taulamb))*plexp(tij(i,j)/Td)
        endif
      enddo
    end do

    return
    end subroutine dust_rad



end module maths_molpop
