module constants_cep
implicit none

! ---------------------------------------------------------
! PHYSICAL CONSTANTS AND RELATIONS AMONG THEM
! ---------------------------------------------------------
! PI : pi
! PE : electron charge
! PME : electron mass
! PK : Bolztmann constant
! PH : Planck constant
! PC : light speed
! CI :
! PI8C2 : 8 * pi / c^2
! PI4H : 4 * pi / h
! PHK : h / k
! ---------------------------------------------------------

	real(kind=8), parameter :: PI = 3.141592654d0, PE = 4.8032d-10
	real(kind=8), parameter :: PME = 9.10938188d-28, UMA = 1.66053873d-24
	real(kind=8), parameter :: PK = 1.3806503d-16, PH = 6.62606876d-27
	real(kind=8), parameter :: PC = 2.99792458d10, CI = 2.0706d-16
	real(kind=8), parameter :: PI8C2 = 8.d0 * PI / (PC**2), PI4H = 4.d0 * PI / PH, PHK = PH / PK
	real(kind=8), parameter :: TWOHC2 = 2.d0 * PH / PC**2

! ---------------------------------------------------------
! ADDITIONAL CONSTANTS
! ---------------------------------------------------------
! n_ptos : maximum number of points
! n_iters : maximum number of iterations
! ---------------------------------------------------------
	
   integer, parameter :: n_ptos = 8001

end module constants_cep
