#For this file, I owe my soul to:
#https://devblogs.nvidia.com/parallelforall/separate-compilation-linking-cuda-device-code

CC = /usr/local/cuda-7.5/bin/nvcc
GCC = g++
objects = mmio.o main.o genresult.o

spmv: $(objects)
	$(CC) -arch=sm_30 $(objects) -o spmv

all: $(objects) 
	$(CC) -arch=sm_30 $(objects) -o spmv

mmio.o: src/mmio.c
	$(GCC) -Iinclude -w src/mmio.c -c

main.o: src/main.c
	$(CC) -x cu -arch=sm_30 -Iinclude -I. -dc src/main.c -o $@

genresult.o: src/genresult.c src/spmv_atomic.c src/spmv_segment.c src/spmv_design.c
	$(CC) -x cu -arch=sm_30 -Iinclude -I. -dc src/genresult.c -o $@

%.o: %.c
	$(CC) -x cu -arch=sm_30 -Iinclude -I. -dc $< -o $@

clean:
	rm -f *.o spmv

#Quick conveniences for submission etc.
tar:
	tar -cvf spmv_proj.tar *.c *.cuh *.pdf makefile

untar:
	tar -xvf spmv_proj.tar
