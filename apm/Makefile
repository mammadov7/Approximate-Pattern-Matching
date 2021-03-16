SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
CFLAGS=-O3 -I$(HEADER_DIR) -Wall -std=c99 -fopenmp
LDFLAGS= 


all: $(OBJ_DIR) apm_ompic

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $^

apm_ompic : $(SRC_DIR)/apm_ompic.cu $(SRC_DIR)/apm_ompic.c
	nvcc -lineinfo -c $(SRC_DIR)/$@.cu -o $@_c.o
	$(CC) $(CFLAGS) $(LDFLAGS) -c $(SRC_DIR)/$@.c -o $@.o
	$(CC) $(CFLAGS) $(LDFLAGS)  $@.o $@_c.o -L /usr/local/cuda-10.2/lib64 -lcudart -o $@
clean:
	rm -f apm apm_mpi apm_omp apm_ompic $(OBJ) ; rm -r $(OBJ_DIR)
