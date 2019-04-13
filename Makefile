########################################################################
# Compiler options.

# make cuda=1

# LINUX:
CC=gcc
NC=nvcc
CCFLAGS = -DLINUX 
#-std=c99
LDFLAGS =
OPTFLAG =  
OPTFLAG = -O3 -ffast-math -fopenmp -lrt 

# Sun:
# CCFLAGS =
# LDFLAGS =
# OPTFLAG = -xO3

# SGI:
# CCFLAGS = -n32
# LDFLAGS = -n32
# OPTFLAG = -O3

########################################################################
# Precision.  Possible values are SINGLEPREC, MIXEDPREC, and DOUBLEPREC.

# LINUX, SGI:

#PREC = SINGLEPREC
PREC = DOUBLEPREC

# Sun:
# PREC = MIXEDPREC

########################################################################
ifeq ($(cuda),1)
    OPTFLAG += -D USE_CUDA
    CUDA_H = cuda_link.h
    CUDA_O = cuda_link.o
    CUDALIBS = -L /usr/local/cuda/lib64 -lcuda -lcudart
endif

scattering: svnversion libscat-$(PREC).a xsinc.o
	$(CC) $(LDFLAGS) -o xsinc \
	  xsinc.o libscat-$(PREC).a \
	  $(CUDALIBS)  -lm -lgomp -lrt


xsinc.o: xsinc.c dataload.h $(CUDA_H) scatintensity.h moleculardynamicsxsinc.h
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) $(USEFREQ) -c xsinc.c 

dataload.o: dataload.c
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) $(USEFREQ) -c dataload.c 

cuda_link.o: cuda_link.cu
	$(NC) -c  cuda_link.cu

scatintensity.o: scatintensity.c
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) $(USEFREQ) -c scatintensity.c

moleculardynamicsxsinc.o: moleculardynamicsxsinc.c pbc.h
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) $(USEFREQ) -c moleculardynamicsxsinc.c

periodicbc.o: periodicbc.c
	$(CC) $(CCFLAGS) -D$(PREC) $(OPTFLAG) $(USEFREQ) -c periodicbc.c

libscat-$(PREC).a: $(CUDA_O) dataload.o scatintensity.o moleculardynamicsxsinc.o periodicbc.o
	ar ruv libscat-$(PREC).a $(CUDA_O) dataload.o scatintensity.o moleculardynamicsxsinc.o periodicbc.o

svnversion:
	bash -c 'cp   Xsinc_version.h           Xsinc_version_updated.h.tmp'
	bash -c 'cat  Xsinc_version_updated.h.tmp | sed s/CURRENT_SVN_VERSION/`svnversion|cut -d: -f2`/ > Xsinc_version_updated.h'
	bash -c 'cp   Xsinc_version_updated.h   Xsinc_version_updated.h.tmp'
	bash -c 'cat  Xsinc_version_updated.h.tmp | sed s/CURRENT_COMPILE_COMMAND/$(CC)/  > Xsinc_version_updated.h'
	bash -c 'rm   Xsinc_version_updated.h.tmp'






