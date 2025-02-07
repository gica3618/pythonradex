# Compiler 

# The following lines are used to choose the compiler
# We have tested gfortran and ifort
# In case you want to use another compiler, use equivalent
# keywords
#  FLAGS -> is used to indicate that the preprocessor has to be invoked
#  OPTIONS -> these are general compilation flags that, for the moment, only indicate
#             that the object file is generated, without linking
# It should be easy to find the equivalent flags in your compiler

#COMPILER = ifort
COMPILER = $(FC) #gfortran

OPTIONS = -c -O2
# c -O3 
# -Waliasing -Wampersand -Wconversion -Wsurprising -Wintrinsics-std -Wintrinsic-shadow -Wline-truncation -Wreal-q-constant -Wunused

FINAL_EXECUTABLE = molpop

molpop: coll_molpop.o global_molpop.o io_molpop.o maths_molpop.o molpop.o \
	sol_molpop.o cep_molpop_interface.o sol_cep.o constants_cep.o global_cep.o \
	maths_cep.o escape_cep.o functions_cep.o io_cep.o 
	$(COMPILER) coll_molpop.o global_molpop.o io_molpop.o maths_molpop.o molpop.o\
		sol_molpop.o cep_molpop_interface.o sol_cep.o constants_cep.o\
		global_cep.o maths_cep.o escape_cep.o functions_cep.o io_cep.o\
		$(NAG_LIBRARY) -o $(FINAL_EXECUTABLE)
	cp $(FINAL_EXECUTABLE) ../
	
clean: 
	find . -maxdepth 2 -name "*.o" -delete ; find . -maxdepth 1 -name "*.mod" -delete ;
	find . -maxdepth 1 -name "*.f90~" -delete ; find . -maxdepth 2 -name "*.a" -delete ;
	find . -maxdepth 1 -name "molpop" -delete ; find . -maxdepth 1 -name "molpop_nonag" -delete;
	find ../ -maxdepth 1 -name "molpop" -delete ; find ../ -maxdepth 1 -name "molpop_nonag" -delete;
	
nag:
	cd NAG ; $(COMPILER) $(OPTIONS) *.f ; ar cr libnag.a *.o

coll_molpop.o : coll_molpop.f90 maths_molpop.o global_molpop.o
	$(COMPILER) $(OPTIONS) coll_molpop.f90
global_molpop.o : global_molpop.f90 
	$(COMPILER) $(OPTIONS) global_molpop.f90
io_molpop.o : io_molpop.f90 coll_molpop.o cep_molpop_interface.o\
	maths_molpop.o global_molpop.o
	$(COMPILER) $(OPTIONS) io_molpop.f90
maths_molpop.o : maths_molpop.f90 global_molpop.o
	$(COMPILER) $(OPTIONS) maths_molpop.f90 
molpop.o : molpop.f90 sol_cep.o maths_molpop.o global_molpop.o sol_molpop.o sol_molpop.o io_molpop.o
	$(COMPILER) $(OPTIONS) molpop.f90 
sol_molpop.o : sol_molpop.f90 maths_molpop.o global_molpop.o
	$(COMPILER) $(OPTIONS) sol_molpop.f90
cep_molpop_interface.o : cep_molpop_interface.f90 global_molpop.o
	$(COMPILER) $(OPTIONS) cep_molpop_interface.f90
sol_cep.o : sol_cep.f90 io_cep.o escape_cep.o functions_cep.o global_cep.o constants_cep.o io_molpop.o
	$(COMPILER) $(OPTIONS) sol_cep.f90
constants_cep.o : constants_cep.f90
	$(COMPILER) $(OPTIONS) constants_cep.f90
global_cep.o : global_cep.f90 constants_cep.o
	$(COMPILER) $(OPTIONS) global_cep.f90	
maths_cep.o : maths_cep.f90
	$(COMPILER) $(OPTIONS) maths_cep.f90	
escape_cep.o : escape_cep.f90 functions_cep.o maths_cep.o constants_cep.o
	$(COMPILER) $(OPTIONS) escape_cep.f90
functions_cep.o : functions_cep.f90 maths_molpop.o io_cep.o maths_cep.o global_cep.o\
	constants_cep.o
	$(COMPILER) $(OPTIONS) functions_cep.f90
io_cep.o : io_cep.f90 maths_cep.o global_cep.o maths_molpop.o
	$(COMPILER) $(OPTIONS) io_cep.f90
