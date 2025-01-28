module maths_cep
implicit none
contains

! *********************************************************
! *********************************************************
! MATHEMATICAL ROUTINES
! *********************************************************
! *********************************************************

! ---------------------------------------------------------
! Read some lines from a file
! ---------------------------------------------------------
	subroutine lb(unit,nlines)
   integer :: unit, nlines, i
   	do i = 1, nlines
      	read(unit,*)
      enddo
   end subroutine lb

! ---------------------------------------------------------
! Given xa(:) and ya(:) which tabulate a function, returns the interpolation using
! linear interpolation of vector x(:) in y(:)
! ---------------------------------------------------------
   subroutine lin_interpol(xa,ya,x,y)
   real*8, INTENT(IN) :: xa(:), ya(:), x(:)
   real*8, INTENT(INOUT) :: y(:)
   integer :: i, n, na
   integer :: locat(1), loc

   	n = size(x)
   	na = size(xa)

   	do i = 1, n
			loc = 1
			do while (xa(loc) < x(i))
				loc = loc + 1
			enddo
			if (loc > na) then				
				loc = na				
			endif
   		if (loc == 1) then
				y(i) = ya(loc)   		
   		else
				y(i) = (ya(loc)-ya(loc-1))/(xa(loc)-xa(loc-1)) * (x(i)-xa(loc-1)) + ya(loc-1)
   		endif
   	enddo

   end subroutine lin_interpol
	
! ---------------------------------------------------------
! LU decomposition of a matrix
!  INPUT:
!		- a is the matrix to decompose
!		
!  OUTPUT:
!		- a is the LU decomposition of a
!		- indx is a vector that records the row permutation effected by the partial pivoting
!		- d takes values +1/-1 depending on whether the number of row interchanges was odd or even
! ---------------------------------------------------------
	subroutine ludcmp(a,indx,d)
	integer, INTENT(INOUT) :: indx(:)
	real(kind=8), INTENT(INOUT) :: a(:,:), d
	real(kind=8), parameter :: TINY = 1.d-20
	integer :: i, imax, j, k, n
	real(kind=8) :: aamax, dum, sum, vv(size(a,1))
		d = 1.d0
		n = size(a,1)
		
		do i = 1, n
			aamax = 0.d0	
			aamax = maxval(dabs(a(i,:)))
			if (aamax == 0.d0) then
				print *, 'Singular matrix in LU decomposition'
				stop
			endif
			vv(i) = 1.d0 / aamax
		enddo
		
		do j = 1, n
			do i = 1, j-1
				sum = a(i,j)
				do k = 1, i-1
					sum = sum - a(i,k) * a(k,j)
				enddo
				a(i,j) = sum
			enddo
			aamax = 0.d0
			do i = j, n
				sum = a(i,j)
				do k = 1, j-1
					sum = sum - a(i,k) * a(k,j)
				enddo
				a(i,j) = sum
				dum = vv(i) * dabs(sum)
				if (dum >= aamax) then
					imax = i
					aamax = dum
				endif				
			enddo
			if (j /= imax) then
				do k = 1, n
					dum = a(imax,k)
					a(imax,k) = a(j,k)
					a(j,k) = dum
				enddo
				d = -d
				vv(imax) = vv(j)
			endif
			indx(j) = imax
			if (a(j,j) == 0.d0) a(j,j) = TINY
			if (j /= n) then
				dum = 1.d0 / a(j,j)
				do i = j+1, n
					a(i,j) = a(i,j) * dum
				enddo
			endif
		enddo
	
	end subroutine ludcmp

! ---------------------------------------------------------
! Solves the set of equations AX=b where A is the LU decomposition of a matrix
!  INPUT:
!		- a is the LU decomposition of the system matrix
!		- b is the right hand side vector of the system
! 		- indx is the vector returned by ludcmp
!  OUTPUT:
!		- b is the solution of the system
! ---------------------------------------------------------
	subroutine lubksb(a,indx,b)
	real(kind=8), INTENT(IN) :: a(:,:)
	real(kind=8), INTENT(INOUT) :: b(:)
	integer, INTENT(IN) :: indx(:)
	integer :: i, ii, n, j, ll
	real(kind=8) :: sum
		n = size(a,1)
		ii = 0
		do i = 1, n
			ll = indx(i)
			sum = b(ll)
			b(ll) = b(i)
			if (ii /= 0) then
				do j = ii, i-1
					sum = sum - a(i,j) * b(j)
				enddo
			else if (sum /= 0.d0) then
				ii = i
			endif
			b(i) = sum
		enddo
		do i = n, 1, -1
			sum = b(i)
			do j = i+1, n
				sum = sum - a(i,j) * b(j)
			enddo
			b(i) = sum / a(i,i)
		enddo
	end subroutine lubksb
	
! ---------------------------------------------------------
! Resuelve un sistema de ecuaciones
! ---------------------------------------------------------
!     SOLUTION OF A SYSTEM OF LINEAR EQUATIONS -- A*X = X
!
!     INPUT FOR LUSLV
!     A        -LEFT HAND SIDE
!     X        -RIGHT HAND SIDE
!     N        -ORDER OF THE SYSTEM
!     NR       -ACTUAL FIRST DIMENSION OF A
!
!     OUTPUT FOR LUSLV
!     A        -LU FACTORIZATION OF ORIGINAL A
!     X        -SOLUTION OF THE LINEAR EQUATIONS
!
!     ONCE LUSLV HAS BEEN CALLED, OTHER SYSTEMS WITH THE SAME LEFT HAND
!     SIDE BUT DIFFERENT RIGHT HAND SIDES MAY BE EFFICIENTLY SOLVED BY
!     USING RESLV.
!
!     INPUT FOR RESLV
!     A        -LU FACTORIZATION OF LEFT HAND SIDE, PRODUCED BY A PREVIOUS
!              -CALL TO LUSLV
!     X        -RIGHT HAND SIDE
!     N        -ORDER OF THE SYSTEM, MUST BE THE SAME AS WHEN LUSLV WAS CALLED
!     NR       -ACTUAL FIRST DIMENSION OF A
!
!     OUTPUT FOR RESLV
!     X        -SOLUTION OF THE LINEAR EQUATIONS
	SUBROUTINE LUSLV(A,X,N,NR)
	
      IMPLICIT real(kind=8)(A-H,O-Z)
		integer :: n, nr
      DIMENSION A(NR,*),X(*),D(100)
      INTEGER R,P(100), i, k, mr1, j, itemp, jp1
      COMMON /LUCOMM/ D,P
      DO 7 R = 1, N
         DO 1 K = 1, N
            D(K) = A(K,R)
    1    CONTINUE
         MR1 = R - 1
         IF(MR1.LT.1) GO TO 4
         DO 3 J = 1, MR1
            ITEMP = P(J)
            A(J,R) = D(ITEMP)
            D(ITEMP) = D(J)
            JP1 = J + 1
            DO 2 I = JP1, N
               D(I) = D(I) - A(I,J)*A(J,R)
    2       CONTINUE
    3    CONTINUE
    4    DMX = ABS(D(R))
         P(R) = R
         DO 5 I = R, N
            IF(DMX.GT.ABS(D(I))) GO TO 5
            DMX = ABS(D(I))
            P(R) = I
    5    CONTINUE
         ITEMP = P(R)
         A(R,R) = 1.0/D(ITEMP)
         D(ITEMP) = D(R)
         MR1 = R + 1
         IF(MR1.GT.N) GO TO 8
         DO 6 I = MR1, N
            A(I,R) = D(I)*A(R,R)
    6    CONTINUE
    7 CONTINUE
    8 CALL RESLV(A,X,N,NR)
      
      END subroutine luslv
		
		
      SUBROUTINE RESLV(A,X,N,NR)
		
      IMPLICIT real(kind=8)(A-H,O-Z)
		integer :: n, nr
      INTEGER P(100), i, ip1, j, k, kp1, itemp
      real(kind=8) D(100),A(NR,*),X(*)
      COMMON /LUCOMM/ D,P
      DO 9 I = 1, N
         D(I) = X(I)
    9 CONTINUE
      DO 11 I = 1, N
         ITEMP = P(I)
         X(I) = D(ITEMP)
         D(ITEMP) = D(I)
         IP1 = I + 1
         IF(IP1.GT.N) GO TO 12
         DO 10 J = IP1, N
            D(J) = D(J) - A(J,I)*X(I)
   10    CONTINUE
   11 CONTINUE
   12 K = N + 1
      DO 15 I = 1, N
         K = K - 1
         SUM = 0.0
         KP1 = K + 1
         IF(KP1.GT.N) GO TO 14
         DO 13 J = KP1, N
            SUM = SUM + A(K,J)*X(J)
   13    CONTINUE
   14    X(K) = A(K,K)*(X(K)-SUM)
   15 CONTINUE
      END subroutine reslv


      

! ---------------------------------------------------------
! Solve a linear system of equations
! ---------------------------------------------------------	
	subroutine llslv(S,X,N,NR)
	integer, INTENT(IN) :: n, nr
	real(kind=8), INTENT(INOUT) :: S(nr,nr), x(nr)
	real(kind=8) :: sum
	integer :: i, j, k
	
	do i = 1, n
		do j = 1, i
			sum = s(i,j)
			do k = 1, j-1
				sum = sum - S(i,k) * S(j,k)
			enddo !k
			if (j < i) then
				s(i,j) = sum / S(j,j)
			else
				S(i,j) = dsqrt(dabs(sum))
			endif
		enddo !j
	enddo !i
	
	entry llreslv(S,X,N,NR)
	
	do i = 1, n
		sum = x(i)
		do j = 1, i-1
			sum = sum - S(i,j)*x(j)
		enddo !j
		x(i) = sum / S(i,i)
	enddo !i
	do i = n, 1, -1
		sum = x(i)
		do j = n, i+1, -1
			sum = sum - S(j,i)*x(j)
		enddo !j
		x(i) = sum / S(i,i)
	enddo !i
	
	end subroutine llslv

! ---------------------------------------------------------
! pythag
! ---------------------------------------------------------            
      FUNCTION pythag(a,b)
      REAL(kind=8) a,b,pythag
      REAL(kind=8) absa,absb
      absa=abs(a)
      absb=abs(b)
      if(absa.gt.absb)then
        pythag=absa*sqrt(1.+(absb/absa)**2)
      else
        if(absb.eq.0.)then
          pythag=0.
        else
          pythag=absb*sqrt(1.+(absa/absb)**2)
        endif
      endif
      return
      END function pythag

! ---------------------------------------------------------
! Sigular value decomposition of a matrix
! ---------------------------------------------------------
     
      SUBROUTINE svdcmp(a,m,n,mp,np,w,v)
      INTEGER m,mp,n,np,NMAX
      REAL(kind=8) :: a(mp,np),v(np,np),w(np)
      PARAMETER (NMAX=500)
      INTEGER i,its,j,jj,k,l,nm
      REAL(kind=8) :: anorm,c,f,g,h,s,scale,x,y,z,rv1(NMAX)
      g=0.0
      scale=0.0
      anorm=0.0
      do 25 i=1,n
        l=i+1
        rv1(i)=scale*g
        g=0.0
        s=0.0
        scale=0.0
        if(i.le.m)then
          do 11 k=i,m
            scale=scale+abs(a(k,i))
11        continue
          if(scale.ne.0.0)then
            do 12 k=i,m
              a(k,i)=a(k,i)/scale
              s=s+a(k,i)*a(k,i)
12          continue
            f=a(i,i)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,i)=f-g
            do 15 j=l,n
              s=0.0
              do 13 k=i,m
                s=s+a(k,i)*a(k,j)
13            continue
              f=s/h
              do 14 k=i,m
                a(k,j)=a(k,j)+f*a(k,i)
14            continue
15          continue
            do 16 k=i,m
              a(k,i)=scale*a(k,i)
16          continue
          endif
        endif
        w(i)=scale *g
        g=0.0
        s=0.0
        scale=0.0
        if((i.le.m).and.(i.ne.n))then
          do 17 k=l,n
            scale=scale+abs(a(i,k))
17        continue
          if(scale.ne.0.0)then
            do 18 k=l,n
              a(i,k)=a(i,k)/scale
              s=s+a(i,k)*a(i,k)
18          continue
            f=a(i,l)
            g=-sign(sqrt(s),f)
            h=f*g-s
            a(i,l)=f-g
            do 19 k=l,n
              rv1(k)=a(i,k)/h
19          continue
            do 23 j=l,m
              s=0.0
              do 21 k=l,n
                s=s+a(j,k)*a(i,k)
21            continue
              do 22 k=l,n
                a(j,k)=a(j,k)+s*rv1(k)
22            continue
23          continue
            do 24 k=l,n
              a(i,k)=scale*a(i,k)
24          continue
          endif
        endif
        anorm=max(anorm,(abs(w(i))+abs(rv1(i))))
25    continue
      do 32 i=n,1,-1
        if(i.lt.n)then
          if(g.ne.0.0)then
            do 26 j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
26          continue
            do 29 j=l,n
              s=0.0
              do 27 k=l,n
                s=s+a(i,k)*v(k,j)
27            continue
              do 28 k=l,n
                v(k,j)=v(k,j)+s*v(k,i)
28            continue
29          continue
          endif
          do 31 j=l,n
            v(i,j)=0.0
            v(j,i)=0.0
31        continue
        endif
        v(i,i)=1.0
        g=rv1(i)
        l=i
32    continue
      do 39 i=min(m,n),1,-1
        l=i+1
        g=w(i)
        do 33 j=l,n
          a(i,j)=0.0
33      continue
        if(g.ne.0.0)then
          g=1.0/g
          do 36 j=l,n
            s=0.0
            do 34 k=l,m
              s=s+a(k,i)*a(k,j)
34          continue
            f=(s/a(i,i))*g
            do 35 k=i,m
              a(k,j)=a(k,j)+f*a(k,i)
35          continue
36        continue
          do 37 j=i,m
            a(j,i)=a(j,i)*g
37        continue
        else
          do 38 j= i,m
            a(j,i)=0.0
38        continue
        endif
        a(i,i)=a(i,i)+1.0
39    continue
      do 49 k=n,1,-1
        do 48 its=1,330
          do 41 l=k,1,-1
            nm=l-1
            if((abs(rv1(l))+anorm).eq.anorm)  goto 2
            if((abs(w(nm))+anorm).eq.anorm)  goto 1
41        continue
1         c=0.0
          s=1.0
          do 43 i=l,k
            f=s*rv1(i)
            rv1(i)=c*rv1(i)
            if((abs(f)+anorm).eq.anorm) goto 2
            g=w(i)
            h=pythag(f,g)
            w(i)=h
            h=1.0/h
            c= (g*h)
            s=-(f*h)
            do 42 j=1,m
              y=a(j,nm)
              z=a(j,i)
              a(j,nm)=(y*c)+(z*s)
              a(j,i)=-(y*s)+(z*c)
42          continue
43        continue
2         z=w(k)
          if(l.eq.k)then
            if(z.lt.0.0)then
              w(k)=-z
              do 44 j=1,n
                v(j,k)=-v(j,k)
44            continue
            endif
            goto 3
          endif
          if(its.eq.330) stop 'no convergence in svdcmp'
          x=w(l)
          nm=k-1
          y=w(nm)
          g=rv1(nm)
          h=rv1(k)
          f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
          g=pythag(f,1.d0)
          f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
          c=1.0
          s=1.0
          do 47 j=l,nm
            i=j+1
            g=rv1(i)
            y=w(i)
            h=s*g
            g=c*g
            z=pythag(f,h)
            rv1(j)=z
            c=f/z
            s=h/z
            f= (x*c)+(g*s)
            g=-(x*s)+(g*c)
            h=y*s
            y=y*c
            do 45 jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)= (x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
45          continue
            z=pythag(f,h)
            w(j)=z
            if(z.ne.0.0)then
              z=1.0/z
              c=f*z
              s=h*z
            endif
            f= (c*g)+(s*y)
            x=-(s*g)+(c*y)
            do 46 jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)= (y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
46          continue
47        continue
          rv1(l)=0.0
          rv1(k)=f
          w(k)=x
48      continue
3       continue
49    continue
      return
      END subroutine svdcmp
      
! ---------------------------------------------------------
! Solve a linear system using SVD
! ---------------------------------------------------------            
         SUBROUTINE svbksb(u,w,v,m,n,mp,np,b,x)
      INTEGER m,mp,n,np,NMAX
      REAL(kind=8) b(mp),u(mp,np),v(np,np),w(np),x(np)
      PARAMETER (NMAX=500)
      INTEGER i,j,jj
      REAL(kind=8) s,tmp(NMAX)
      do 12 j=1,n
        s=0.
        if(w(j).ne.0.)then
          do 11 i=1,m
            s=s+u(i,j)*b(i)
11        continue
          s=s/w(j)
        endif
        tmp(j)=s
12    continue
      do 14 j=1,n
        s=0.
        do 13 jj=1,n
          s=s+v(j,jj)*tmp(jj)
13      continue
        x(j)=s
14    continue
      return
      END subroutine svbksb
	      
! ---------------------------------------------------------
! Return the exponential integral En(x)
! ---------------------------------------------------------      
      FUNCTION expint(n,x)
      INTEGER n,MAXIT
      REAL(kind=8) expint,x,EPS,FPMIN,EULER
      PARAMETER (MAXIT=100,EPS=1.e-7,FPMIN=1.e-30,EULER=.5772156649)
      INTEGER i,ii,nm1
      REAL(kind=8) a,b,c,d,del,fact,h,psi
      nm1=n-1
      if(n.lt.0.or.x.lt.0..or.(x.eq.0..and.(n.eq.0.or.n.eq.1)))then
        print *, n, x
        print *, 'bad arguments in expint'
        stop
      else if(n.eq.0)then
        expint=exp(-x)/x
      else if(x.eq.0.)then
        expint=1./nm1
      else if(x.gt.1.)then
        b=x+n
        c=1./FPMIN
        d=1./b
        h=d
        do 11 i=1,MAXIT
          a=-i*(nm1+i)
          b=b+2.
          d=1./(a*d+b)
          c=b+a/c
          del=c*d
          h=h*del
          if(abs(del-1.).lt.EPS)then
            expint=h*exp(-x)
            return
          endif
11      continue
        print *, 'continued fraction failed in expint'
        stop
      else
        if(nm1.ne.0)then
          expint=1./nm1
        else
          expint=-log(x)-EULER
        endif
        fact=1.
        do 13 i=1,MAXIT
          fact=-fact*x/i
          if(i.ne.nm1)then
            del=-fact/(i-nm1)
          else
            psi=-EULER
            do 12 ii=1,nm1
              psi=psi+1./ii
12          continue
            del=fact*(-log(x)+psi)
          endif
          expint=expint+del
          if(abs(del).lt.abs(expint)*EPS) return
13      continue
        print *, 'series failed in expint'
        stop
      endif
      return
      END function expint


! ---------------------------------------------------------
! Given x(:) and y(:) which tabulate a function and the derivative at the boundary points
! this function returns the second derivative of the spline at each point
! ---------------------------------------------------------
		subroutine splin1(x,y,yp1,ypn,y2)
		real(kind=8), INTENT(IN) :: x(:), y(:), yp1, ypn
		real(kind=8), INTENT(INOUT) :: y2(size(x))
		integer :: n, i, k
		real(kind=8) :: p, qn, sig, un, u(size(x))

			n = size(x)
			
			if (yp1 > .99d30) then
				y2(1) = 0.d0
				u(1) = 0.d0
			else
				y2(1) = -0.5d0
				u(1) = (3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
			endif

			do i = 2, n-1
				sig = (x(i)-x(i-1))/(x(i+1)-x(i-1))				
				p = sig * y2(i-1)+2.d0
				y2(i) = (sig-1.d0)/p
				u(i) = (6.d0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/&
					(x(i+1)-x(i-1))-sig*u(i-1))/p
			enddo
			if (ypn > .99d30) then
				qn = 0.d0
				un = 0.d0
			else
				qn = 0.5d0
				un = (3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
			endif
			
			y2(n) = (un-qn*u(n-1))/(qn*y2(n-1)+1.d0)

			do k = n-1, 1, -1
				y2(k) = y2(k)*y2(k+1)+u(k)
			enddo

		end subroutine splin1

! ---------------------------------------------------------
! Given xa(:) and ya(:) which tabulate a function, returns the interpolation using
! splines of vector x(:) in y(:)
! ---------------------------------------------------------
		subroutine spline(xa,ya,y2a,x,y)
		real(kind=8), INTENT(INOUT) :: y(:)
		real(kind=8), INTENT(IN) :: xa(:), ya(:), x(:)
		real(kind=8) :: y2a(:)
		integer :: n_x, n, i, k, khi, klo
		real(kind=8) :: a, b, h, extrap
			
			n = size(xa)
			n_x = size(x)
			!call splin1(xa,ya,1.d30,1.d30,y2a)						

			do i = 1, n_x					

! Downward extrapolation 
				if (x(i) < xa(1)) then
!					y(i) = ya(1)
					y(i) = ya(1) + (ya(1)-ya(2))/(xa(1)-xa(2)) * (xa(1) - x(i))
				else 

! Upward extrapolation
				if (x(i) > xa(n)) then
!					y(i) = ya(n)
					y(i) = ya(n) + (ya(n)-ya(n-1)) / (xa(n)-xa(n-1)) * (x(i) - xa(n))
				else
! In range
						klo = 1
						khi = n
1						if(khi-klo > 1) then
							k = (khi+klo)/2
							if (xa(k) > x(i)) then
								khi = k
							else
								klo = k
							endif					
							go to 1
						endif

						h = xa(khi)-xa(klo)

						if (h == 0.d0) then
							print *, 'bad xa input in spline'
							stop
						endif
						a = (xa(khi)-x(i))/h
						b = (x(i)-xa(klo))/h

						y(i) = a*ya(klo)+b*ya(khi)+((a**3.d0-a)*y2a(klo)+&
							(b**3.d0-b)*y2a(khi))*(h**2.d0)/6.d0		
					endif
				endif
			enddo

		end subroutine spline
		
!-----------------------------------------------------------------
! Returns the weights (w) and the abscissas (x) for a Gaussian integration using the 
! Gauss-Legendre formula, using n points
!-----------------------------------------------------------------
	subroutine gauleg(x1,x2,x,w,n)
	integer, INTENT(IN) :: n
	real(kind=8), INTENT(IN) :: x1,x2
	real(kind=8), INTENT(INOUT) :: x(n),w(n)
	real(kind=8), parameter :: eps = 3.d-14
	integer :: i,j,m
	real(kind=8) :: p1,p2,p3,pp,xl,xm,z,z1
      
	m=(n+1)/2
   xm=0.5d0*(x2+x1)
   xl=0.5d0*(x2-x1)
   do i=1,m
   	z=dcos(3.141592654d0*(i-.25d0)/(n+.5d0))
1  	continue
   	p1=1.d0
   	p2=0.d0
   	do j=1,n
   		p3=p2
      	p2=p1
      	p1=((2.d0*j-1.d0)*z*p2-(j-1.d0)*p3)/j
		enddo
   	pp=n*(z*p1-p2)/(z*z-1.d0)
   	z1=z
   	z=z1-p1/pp
  		if(abs(z-z1).gt.EPS)goto 1
   	x(i)=xm-xl*z
   	x(n+1-i)=xm+xl*z
   	w(i)=2.d0*xl/((1.d0-z*z)*pp*pp)
   	w(n+1-i)=w(i)
	enddo
	
	end subroutine gauleg
	
! ---------------------------------------------------------
! This subroutine solves a linear system of equations using a BiCGStab iterative method
! ---------------------------------------------------------		  
	subroutine bicgstab(a,b)
	real(kind=8), INTENT(INOUT) :: a(:,:), b(:)
	real(kind=8) :: x(size(b)), r(size(b)), rhat(size(b)), p(size(b)), phat(size(b)), v(size(b))
	real(kind=8) :: s(size(b)), shat(size(b)), t(size(b)), delta(size(b))
	real(kind=8) :: rho, rho_ant, alpha, beta, omega, relative
	integer :: i
		
		alpha = 0.d0
		omega = 0.d0
! Initial solution
		x = 1.d0
	
		r = b - matmul(A,x)
		rhat = r
		rho = 1.d0
		relative = 1.d10
		i = 1
		do while (relative > 1.d-10)
			rho_ant = rho
			rho = sum(rhat*r)
			if (rho == 0) then 
				stop
			endif
			if (i == 1) then
				p = r
			else
				beta = (rho/rho_ant) * (alpha/omega)
				p = r + beta * (p - omega * v)
			endif
			phat = p
			v = matmul(A,phat)
			alpha = rho / sum(rhat*v)
			s = r - alpha * v

			shat = s
			t = matmul(A,shat)
			omega = sum(t*s)/sum(t*t)
			delta = alpha * p + omega * s
			relative = maxval(abs(delta) / x)
			x = x + delta		
			r = s - omega * t
			i = i + 1
		enddo	
		
		b = x
		  
	end subroutine bicgstab	
	
!----------------------------------------------------------------
! This function integrates a tabulated function
!----------------------------------------------------------------		
	function int_tabulated(x, f)
	real(kind=8) :: x(:), f(:), int_tabulated, res, error_res
	integer :: n
		n = size(x)
		call cubint (f, x, n, 1, n, res, error_res)
		int_tabulated = res
	end function int_tabulated

!-------------------------------------------------------------
! CUBINT approximates an integral using cubic interpolation of data.
!  Parameters:
!
!    Input, real FTAB(NTAB), contains the tabulated function
!    values, FTAB(I) = F(XTAB(I)).
!
!    Input, real XTAB(NTAB), contains the points at which the
!    function was tabulated.  XTAB should contain distinct
!    values, given in ascending order.
!
!    Input, integer NTAB, the number of tabulated points.
!    NTAB must be at least 4.
!
!    Input, integer IA, the entry of XTAB at which integration
!    is to begin.  IA must be no less than 1 and no greater
!    than NTAB.
!
!    Input, integer IB, the entry of XTAB at which integration
!    is to end.  IB must be no less than 1 and no greater than
!    NTAB.
!
!    Output, real RESULT, the approximate value of the
!    integral from XTAB(IA) to XTAB(IB) of the function.
!
!    Output, real ERROR, an estimate of the error in
!    integration.
!-------------------------------------------------------------
	subroutine cubint ( ftab, xtab, ntab, ia, ib, result, error )

  	integer ntab
!
  	real(kind=8) :: c, d1, d2, d3, error, ftab(ntab), h1, h2, h3, h4
  	integer :: i, ia, ib, ind, it, j, k
  	real(kind=8) r1, r2, r3, r4, result, s, term, xtab(ntab)
!
  	result = 0.0E+00
  	error = 0.0E+00
 
  	if ( ia == ib ) then
    	return
  	end if
 
  	if ( ntab < 4 ) then
    	write ( *, '(a)' ) ' '
    	write ( *, '(a)' ) 'CUBINT - Fatal error!'
    	write ( *, '(a,i6)' ) '  NTAB must be at least 4, but input NTAB = ',ntab
    	stop
  	endif
 
  	if ( ia < 1 ) then
    	write ( *, '(a)' ) ' '
    	write ( *, '(a)' ) 'CUBINT - Fatal error!'
    	write ( *, '(a,i6)' ) '  IA must be at least 1, but input IA = ',ia
    	stop
  	endif
 
  	if ( ia > ntab ) then
   	write ( *, '(a)' ) ' '
    	write ( *, '(a)' ) 'CUBINT - Fatal error!'
    	write ( *, '(a,i6)' ) '  IA must be <= NTAB, but input IA=',ia
    	stop
  	endif
 
  	if ( ib < 1 ) then
    	write ( *, '(a)' ) ' '
    	write ( *, '(a)' ) 'CUBINT - Fatal error!'
    	write ( *, '(a,i6)' ) '  IB must be at least 1, but input IB = ',ib
    	stop
  	endif
 
  	if ( ib > ntab ) then
    	write ( *, '(a)' ) ' '
    	write ( *, '(a)' ) 'CUBINT - Fatal error!'
    	write ( *, '(a,i6)' ) '  IB must be <= NTAB, but input IB=',ib
    	stop
  	endif
!
!  Temporarily switch IA and IB, and store minus sign in IND
!  so that, while integration is carried out from low X's
!  to high ones, the sense of the integral is preserved.
!
  	if ( ia > ib ) then
    	ind = -1
    	it = ib
    	ib = ia
    	ia = it
  	else
    	ind = 1
  	endif
 
  	s = 0.0E+00
  	c = 0.0E+00
  	r4 = 0.0E+00
  	j = ntab-2
  	if ( ia < ntab-1 .or. ntab == 4 ) then
    	j=max(3,ia)
  	endif

  	k = 4
  	if ( ib > 2 .or. ntab == 4 ) then
    	k=min(ntab,ib+2)-1
  	endif
 
  	do i = j, k
 
    	if ( i <= j ) then
 
      	h2 = xtab(j-1)-xtab(j-2)
      	d3 = (ftab(j-1)-ftab(j-2)) / h2
      	h3 = xtab(j)-xtab(j-1)
      	d1 = (ftab(j)-ftab(j-1)) / h3
      	h1 = h2+h3
      	d2 = (d1-d3)/h1
      	h4 = xtab(j+1)-xtab(j)
      	r1 = (ftab(j+1)-ftab(j)) / h4
      	r2 = (r1-d1) / (h4+h3)
      	h1 = h1+h4
      	r3 = (r2-d2) / h1
 
      	if ( ia <= 1 ) then
        		result = h2 * (ftab(1)+h2*(0.5*d3-h2*(d2/6.0-(h2+h3+h3)*r3/12.)))
        		s = -h2**3 * (h2*(3.0*h2+5.0*h4)+10.0*h3*h1)/60.0
      	endif
 
    	else
 
	      h4 = xtab(i+1)-xtab(i)
      	r1 = (ftab(i+1)-ftab(i))/h4
      	r4 = h4+h3
      	r2 = (r1-d1)/r4
      	r4 = r4+h2
      	r3 = (r2-d2)/r4
      	r4 = (r3-d3)/(r4+h1)
 
    	endif
 
    	if ( i > ia .and. i <= ib ) then
 
      	term = h3*((ftab(i)+ftab(i-1))*0.5-h3*h3*(d2+r2+(h2-h4)*r3) / 12.0 )
      	result = result+term
      	c = h3**3*(2.0E+00 *h3*h3+5.*(h3*(h4+h2) + 2.0 * h2 * h4 ) ) / 120.0E+00
      	error = error+(c+s)*r4
 
      	if ( i /= j ) then
        		s = c
      	else
        		s = s+c+c
      	endif
 
    	else
 
	      error = error+r4*s
 
    	endif
 
    	if ( i >= k ) then
 
      	if ( ib >= ntab ) then
	        	term = h4*(ftab(ntab) - h4*(0.5*r1+h4*(r2/6.0 +(h3+h3+h4)*r3/12.)))
        		result = result + term
        		error = error - h4**3 * r4 * &
          		( h4 * ( 3.0 * h4 + 5.0 * h2 ) &
          		+ 10.0 * h3 * ( h2 + h3 + h4 ) ) / 60.0E+00
      	endif
 
      	if ( ib >= ntab-1 ) error=error+s*r4
    	else
	      h1 = h2
      	h2 = h3
      	h3 = h4
      	d1 = r1
      	d2 = r2
      	d3 = r3
    	endif
 
  	enddo
!
!  Restore original values of IA and IB, reverse signs
!  of RESULT and ERROR, to account for integration
!  that proceeded from high X to low X.
!
  	if ( ind /= 1 ) then
    	it = ib
    	ib = ia
    	ia = it
    	result = -result
    	error = -error
  	endif
 
  	return
	end subroutine



end module maths_cep
