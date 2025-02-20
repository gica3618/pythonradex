
Module File_Units
   character(80)    :: Lname, inp_file, out_file
   integer          :: inunit = 10, outunit = 11, lunit = 12
End Module File_Units

Module Mol_Info
   character(40)    :: Mol_name
   integer          :: Nlev
   real             :: weight
End Module Mol_Info

!***********************************************************************

Program Leiden

! Convert atomic data files from the Leiden database to MOLPOP input format
! Files to be converted can be listed in a file (one per line) and the program invoked with 
!   the (arbitrary) filename as an argument
! If the program is invoked without an argument, it prompts for names of files to convert
! The Leiden file names can be specified with or without their .dat extension
! A Leiden file 'fname.dat' will produce 'fname_lamda.molecule' with the energy levels 
!  and A-coefficients in MOLPOP format, and a set of 'Coll/fname_collider_lamda.kij' 
!  for the collision rate coefficients of each collider in the Leiden lamda file 

  use File_Units
  Implicit None
  integer          :: Open_Status
  character(80)    :: File_List

  print "(/,'  ****  Leiden-to-MOLPOP Conversion ****',/)"

! Now work:
  call GETARG(1, File_List)
  if (trim(File_List).eq."") then ! use keyboard entry
     call Keyboard_entry
  else                            ! work on list from the file whose name was entered as argument
     call File_list_entry(trim(File_List))
  end if   

  print*, 'Done!'
  stop
End Program Leiden

!***********************************************************************

Subroutine Keyboard_entry

! Keyboard entry of Leiden input filename , with or without the .dat extension

  IMPLICIT None
  character(80) :: str

  do  
    WRITE(*,'("Enter a Leiden database file name (q to quit): ")', ADVANCE='NO')
    !WRITE(*,'(": ")', ADVANCE='NO')   
    read *, str
    if (str == 'q') return ! done; otherwise
    call Process(str)      ! work on the file entered and get another one
  end do
  
End Subroutine Keyboard_entry

!***********************************************************************

Subroutine File_list_entry(fname)

! Entry of Leiden input filename, with or without the .dat extension
! from a list of files in 'fname' 

   use File_Units
   IMPLICIT None
   character(*), intent(in) :: fname
   character(80) :: datfile
   integer       :: Open_Status, line_status

   open(lunit, file=fname, status='Old', action='Read', IOSTAT = Open_Status)
   if (Open_Status > 0) then
       print *, '***** Error; file ',fname,' did not open properly *****' 
       return
   end if
   line_status = 0
   do
      read(lunit,'(a)',IOSTAT = line_status) datfile
      if (line_status /= 0) return          ! done, end of the file; otherwise,
      call Process(trim(datfile))           ! work on the next entry in the file list
   end do
   
End Subroutine File_list_entry

!***********************************************************************

Subroutine Process(str)
! This is the centarl control for processing the 
! Leiden database file 'Lname.dat' into MOLPOP data files 

   use File_Units
   use Mol_Info
   IMPLICIT None
   integer          :: ind, Open_Status
   character(*), intent(in) :: str
    
    ind = index(str,'.dat')
    if (ind == 0) then ! the file name was specified without the .dat extension
        Lname    = str
        inp_file = trim(Lname)//'.dat'
    else               ! the file was specified with the .dat extension
        inp_file = str
        Lname    = str(1:ind - 1)     
    end if
   
   open(inunit, file=inp_file, status='Old', action='Read', IOSTAT = Open_Status)
   if (Open_Status > 0) then
       print *, '***** Error; file ',trim(inp_file),' not opened properly *****' 
       return
   end if
   print *, '== Processing Leiden data file ',trim(inp_file)

!  First get the molecule information:
   Call Pass_Bang(inunit)
   read(inunit,'(a)') Mol_name
   Call Pass_Bang(inunit) 
   read(inunit,*) weight
   Call Pass_Bang(inunit)
   read(inunit,*) Nlev

!  Proceed with creating the MOLPOP data files
   Call Energy_Levels
   call Kij 

   print *
   return 
End Subroutine Process

!***********************************************************************

Subroutine Energy_Levels
!  Output energy levels and A-coefficients to the .levels file
   use File_Units
   use Mol_Info
   IMPLICIT None
   character(255)   :: Rest
   double precision :: rdinp, A, E, F
   integer          :: Ntran, n, i, j, k, g, first
   logical, parameter :: iequal = .true., noequal = .false.

   Call StartFile('_lamda.molecule', ' Energy levels and A-coefficients ')
   write(outunit,"('Molecular species: ',a,/)") trim(Mol_name)  
   write(outunit,"('# of levels     molecular mass',/'>',/,I5,10x,F10.2,/)") Nlev, weight  
   write(outunit,*)"   N     g      Energy in cm^{-1}     Level details"
   write(outunit,*)">"

!  Energy levels:
   Call Pass_Bang(inunit) 
   do i = 1, Nlev
      inunit = -inunit
      j  = rdinp(noequal,Rest,inunit,outunit)
      E  = rdinp(noequal,Rest,inunit,outunit)
      g  = int(rdinp(noequal,Rest,inunit,outunit))
      first = 1
      do while(Rest(first:first).eq." ")
         first = first + 1
      end do
      write(outunit,'(I5,3x, I3, 2x,1pe18.6,8x,a)'),  &
           j, g, E, "'"//Trim(Rest(first:))//"'"
   end do
 
!  A-coefficients:
   write(outunit,"(/,'Einstein A-coefficients:')") 
   write(outunit,*)"   i     j    A_ij in s^{-1}"
   write(outunit,*)">"
  
   Call Pass_Bang(inunit) 
   read(inunit,*) Ntran
   Call Pass_Bang(inunit) 
   do k = 1, Ntran
      read(inunit,*) n, i, j, A, F, E
      write(outunit,'(I5,3x, I3, 2x,1pe12.3)') i, j, A
   end do

   Call FinishFile 
   return
   
End Subroutine Energy_Levels

!***********************************************************************

Subroutine Kij
!  Output collision rate coefficients to the .kij file
   use File_Units
   use Mol_Info
   IMPLICIT None
   character(80) :: line, fname
   character(10) :: collider
   real, dimension(100) :: T, R
   integer  :: Ncoll, NTemp, Ntran, ind, ind2, m, n, i, j, k, l

   Call Pass_Bang(inunit) 
   read(inunit,*) Ncoll

   do m = 1, Ncoll
      Call Pass_Bang(inunit) 
      read(inunit,'(a)') line        
      ind  = Index(line,'-') + 1
      do while (line(ind:ind) == ' ')
         ind = ind + 1
      end do 
      ind2 = Index(trim(line(ind:)),' ') - 1
      collider = line(ind:ind + ind2)
      fname = trim(Lname)//'_'//trim(collider)//'_lamda.kij'

      out_file = 'Coll/'//trim(fname)
      open (outunit, file=out_file, status='Replace', action='Write') 
      write(outunit, "('MOLPOP collision rates generated from the Leiden database file ',a)") trim(inp_file) 
      write(outunit,'(a)') trim(line)
      
      Call Pass_Bang(inunit) 
      read(inunit,*) Ntran
      Call Pass_Bang(inunit) 
      read(inunit,*) NTemp
      write(outunit,"(/,'>',/,'Number of temperature columns =',I3,/)") Ntemp   
      write(outunit,"(/,5x,a,/)") 'I    J                        TEMPERATURE (K)'       
      Call Pass_Bang(inunit)
      read(inunit,*) (T(i), i = 1, Ntemp)
      write(outunit,'(10x,50(F12.2))') (T(i), i = 1, Ntemp)
      write(outunit,*)
      Call Pass_Bang(inunit) 
      do k = 1, Ntran
         read(inunit,*) n, i, j, (R(l), l = 1, Ntemp)
         write(outunit,'(I5,3x, I3, 2x,50(1pe12.3))') i, j, (R(l), l = 1, Ntemp)
      end do
      Call FinishFile 
   end do
   return
   
End Subroutine Kij

!***********************************************************************

Subroutine StartFile(ext, info)
   use File_Units
   IMPLICIT None
   character(*), intent(in) :: ext, info

   out_file = trim(Lname)//ext 
   open(outunit, file=out_file, status='Replace', action='Write') 
   write(outunit, "('MOLPOP',a,' file',/, 'Generated from the Leiden database file ',a)") trim(info), trim(inp_file) 
   return
End Subroutine StartFile

Subroutine FinishFile
   use File_Units
   IMPLICIT None

   write(outunit,'(/)')
   close(outunit)
   print *, '     *** Done with generating ',trim(out_file)
   return
End Subroutine FinishFile

   
!***********************************************************************

SUBROUTINE Pass_Bang(iunit)
!     Get past Leiden comment line that starts with "!"
      integer iunit
      character*128 comment
      comment = "XXX"
      do while(.not. (comment(1:1).eq."!"))
         read(iunit, '(a)') comment
      end do
      return
End SUBROUTINE Pass_Bang
!***********************************************************************


!                                                                           !
!===========================================================================!
double precision function RDINP(Equal, Rest, inUnit, outUnit)
! ==========================================================================!
!  Read lines, up to 255 long, from pre-opened unit inUnit and extract      !
!  all input numbers from them. When EQUAL is set, numeric input data       !
!  must be preceded by an equal sign. All non-numeric data and numbers      !
!  not preceded by = are ignored when EQUAL is on.                          !
!                                                                           !
!  RDINP = next number encountered (after equal sign) and terminated blank. !
!  The blank can be optionally preceded by comma.                           !
!  Input numbers can be written in any FORTRAN allowable format.            !
!  In addition, comma separation is allowed in the input numbers.           !
!                                                                           !
!  All text after % (or !) is ignored, as in TeX (or F90).                  !
!  Lines with * in the first column are echoed to output on pre-opened      !
!  unit outUnit.                                                            !
!                                                                           !
!  The search is conducted between FIRST, which is                          !
!  continuously increased, and LAST.  A new line is read when FIRST         !
!  exceeds LAST, and can be forced by calling with -iUnit.                  !
!                                                                           !
!===========================================================================!
   IMPLICIT None
   Integer :: inUnit, outUnit, ind, First = 1, Last = 0
   Logical, intent(in) :: Equal
   CHARACTER(255) Card, no_comma, Rest
   Save Card, First, Last
! -----------------------------------------------------------------------
!
   IF (inUnit.lt.0) Then                         ! force a new line
      First = Last + 1
      inUnit = -inUnit
   END IF

   DO
      If (first > last) then                     ! Time to get a new line
         READ (inUnit, '(A)' , END = 99) Card
         if (len_trim(Card) == 0) cycle          ! ignore empty lines
                                                 ! Echo to output lines that start with * 
         IF (Card(1:1) == '*') &
             WRITE (outUnit,'(A)') Card(1:len_trim(card))
         Card = trim(no_comma(Card))             ! remove commas to allow comma-separated numbers
         first = 1
         last = len_Trim(Card)
         ind = MAX(Index(Card,'%'), Index(Card,'!'))
         if (ind.gt.0) last = ind - 1            ! Everything after % and ! is ignored 
      End If

     !Get past the next '=' when the EQUAL flag is set
      If (Equal) then
        DO WHILE ((Card(first:first) /= '=') .and. (first <= last) )
          first = first + 1
        END DO
      End If
      first = first + 1
      IF (first > last) cycle

     !Find start of next number; necessary when the EQUAL flag is off
      Do While (.not. (Card(first:first).ge.'0'.AND.Card(first:first).le.'9') &
                .and. (first <= last))
          first = first + 1
      End Do
      if (first > last) cycle

     !OK; time to get the number
      READ(card(first - 1:last), *, ERR = 98) RDINP 
      
     !and move past its end
      Do While (Card(first:first) /= ' ')
          first = first + 1
      End Do
      
      Rest = Card(first:last)
      return
   End DO

98 WRITE (outUnit,'(3(1x,a,/))')                               &
   ' ****ERROR. RDINP could not read a proper number from', &
   Card(first - 1:last),																			 &
   ' ****Number should be preceded and terminated by spaces'
   RETURN
99 WRITE (outUnit,'(3(1x,a,/))')                                      &
   ' ****TERMINATED. EOF reached by RDINP while looking for input. ', &
   ' *** Last line read:', Card
   RETURN
end function RDINP
! ***********************************************************************


!***********************************************************************
!=======================================================================
function no_comma(str)
!=======================================================================
! Remove all commas from string str. This enables input to RDINP of 
! comma separated numbers
!=======================================================================
   character(len = *), intent(in) :: str
   character(255) temp, no_comma
   integer l, k
   
   ! First create a blank string:      
   do l = 1, len(temp)
      temp(l:l) = ' '  
   end do 

   ! Now fill the blank string with the characters
   ! of str, skipping the commas:
   k = 1
   do l = 1, len_trim(str)
      if (str(l:l) /= ',') then
         temp(k:k) = str(l:l)
         k = k + 1
      end if
   end do

   no_comma = trim(temp)
   return
   
end function no_comma
!***********************************************************************


