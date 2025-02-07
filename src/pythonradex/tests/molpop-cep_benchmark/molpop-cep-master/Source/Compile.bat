rem Cleanup:
del *.o
del *.mod

@echo Done with cleanup; now to compilation
@pause

gfortran -c -O3 global_molpop.f90
gfortran -c -O3 maths_molpop.f90 
gfortran -c -O3 coll_molpop.f90
gfortran -c -O3 cep_molpop_interface.f90
gfortran -c -O3 io_molpop.f90
gfortran -c -O3 maths_cep.f90	
gfortran -c -O3 constants_cep.f90
gfortran -c -O3 global_cep.f90	
gfortran -c -O3 io_cep.f90
gfortran -c -O3 functions_cep.f90
gfortran -c -O3 escape_cep.f90
gfortran -c -O3 sol_cep.f90
gfortran -c -O3 sol_molpop.f90
gfortran -c -O3 molpop.f90 

gfortran coll_molpop.o global_molpop.o io_molpop.o maths_molpop.o molpop.o sol_molpop.o cep_molpop_interface.o sol_cep.o constants_cep.o	global_cep.o maths_cep.o escape_cep.o functions_cep.o io_cep.o -o molpop.exe

@echo Done!
@pause

@del *.o
@del *.mod
