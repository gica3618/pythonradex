module global_molpop
implicit none
  
   real(kind=8) pi,twopi,fourpi,eitpi,rootpi
   real(kind=8) cl,hpl,bk,xme,xmp
   real(kind=8) solarl,solarm,solarr,pc
   real(kind=8) Tcmb

   real(kind=8) freq_axis(100)

   character*128 apath, path_database
  
   integer n

   real(kind=8) T,v,vt,r,tcool,xi, cep_precision

   real(kind=8) nh2,xmol,nmol

   real(kind=8) sat, taum, eps

   real(kind=8), allocatable :: tau(:,:)
   
   real(kind=8), allocatable :: esc(:,:),dbdtau(:,:)
   INTEGER kbeta

   INTEGER KPRT,NEWTPR,KCOOL,KFIRST,INPRT,KSOLPR

   integer prstep,nprint,nbig

   real(kind=8) colm,RM
   real(kind=8), allocatable :: final(:,:)
   integer nmax

   real(kind=8) acc,step,hcol,sumw,mcol
!     22 Aug 04 add itmax to solv; itmax is maximum number of iterations in NEWTON
   integer nr,nr0,nrmax,kthick,itmax

   character*168 fn_sum
   integer i_sum

   integer i_str

   integer i_tr, n_tr
   integer, allocatable :: itr(:), jtr(:), in_tr(:)
   character*168, allocatable :: f_tr(:)
   real(kind=8), allocatable :: fin_tr(:,:,:)
   logical, allocatable :: a_maser(:)


   logical l_mol_part,l_mol_ROT 
   character*32  mol_name,s_mol, molecular_species
   integer i_mol, N_max
   real(kind=8) mol_mass,mol_d


   integer n_col,i_col_src(10),i_col_exsub(10),n_columns_colliders
   real(kind=8) fr_col(10),gxsec(10),collider_column(10)
   character*168 fn_kij(10)


   real(kind=8), allocatable :: a(:,:),tij(:,:),taux(:,:),&
    c(:,:),rad(:,:),we(:),gap(:,:),ems(:,:),boltz(:),&
    rad_internal(:,:), rad_tau0(:,:), rad_tauT(:,:)
    
   integer :: n_zones_slab
   real(kind=8), allocatable :: collis_all(:,:,:)

   real(kind=8), allocatable :: freq(:,:),fr(:),wl(:,:),ti(:)

   integer, allocatable :: g(:)

   real(kind=8), allocatable :: pop(:),coolev(:)

   real(kind=8), allocatable ::  xp(:)

   logical ipr_lev(6),ipr_tran(6)

   character*80, allocatable :: ledet(:)

   integer, allocatable :: imaser(:),jmaser(:)
   integer :: n_mtr,nmaser

! Overlap common; added 7 Aug 04
   integer num,nulev
   integer, allocatable :: uplin(:),lowlin(:)
   logical overlaptest
  
   real(kind=8) :: mu_output

   real(kind=8) :: vmax_profile
  
   character(len=168) :: file_physical_conditions
   character(len=20) :: auxiliary_functions
   
   integer :: nInitialZones

! Dust absorption effects
   real(kind=8), allocatable :: qdust(:,:), Xd(:,:)
   logical dustAbsorption
   integer :: Idust, n_prt_cols
  
end module global_molpop
