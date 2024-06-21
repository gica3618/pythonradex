# Makefile for Radex

# This file is part of the RADEX software package
# to calculate molecular excitation and radiative
# transfer in a homogeneous medium.
#
# Documentation for the program is posted at
# https://sron.rug.nl/~vdtak/radex/index.shtml  
#
# Although this program has been thoroughly tested, the
# authors do not claim that it is free of errors and
# gives correct results in all situations.
#
# Publications using this program should make a reference
# to our paper: A&A 468, 627 (2007).

# Uncomment the FC/FFLAGS combination of your choice
# Option 1:
#FC      = g77
#FFLAGS += -fno-automatic -Wall -O2
# Option 2:
FC = gfortran
FFLAGS += -O2
# Option 3:
#FC      = ifort
#FFLAGS += -O2
# Option 4:
#FC = g95
#FFLAGS += -Wall -O2
# there may be other options of course ...

OBJECTS = main.o io.o readdata.o matrix.o background.o slatec.o
EXEC    = radex
BINDIR  = ../bin/

all: $(EXEC) install clean

$(EXEC): $(OBJECTS) radex.inc Makefile
	$(LINK.F) $(OBJECTS) -o $@
	strip $@

%.o: %.f radex.inc Makefile
	$(COMPILE.F) $*.f -o $@

install: $(EXEC)
	install -m 755 -p -s $(EXEC) $(BINDIR)

clean:
	- rm *.o
	- rm $(EXEC)
