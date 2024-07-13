OBJS = main.o kernels.o helpers.o

EXENAME = main

CC = nvcc
CCOPTS = -c -O3 -gencode=arch=compute_86,code=sm_86 -G -lineinfo -v
LINK = nvcc
LINKOPTS = -lcurand -o

all : $(EXENAME)

$(EXENAME) : $(OBJS)
	$(LINK) $(LINKOPTS) $(EXENAME) $(OBJS)

main.o : main.cpp kernels.h helpers.h
	$(CC) $(CCOPTS) main.cpp

kernels.o : kernels.cu kernels.h kernels.h
	$(CC) $(CCOPTS) kernels.cu

helpers.o : helpers.cpp helpers.h
	$(CC) $(CCOPTS) helpers.cpp


clean :
	rm -f *.o $(EXENAME) 2>/dev/null