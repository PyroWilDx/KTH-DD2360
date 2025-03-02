CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
# CFLAGS = -O3 -g -Xcompiler "-Wall -Wshadow -march=native"
CFLAGS = -O3 -g -Xcompiler "-Wall -Wshadow -march=native" -w
LDFLAGS = -lm

all: check example1 example2 example3 example4 mnist_run

sigmoid: CFLAGS += -Dgenann_act=genann_act_sigmoid_cached
sigmoid: all

threshold: CFLAGS += -Dgenann_act=genann_act_threshold
threshold: all

linear: CFLAGS += -Dgenann_act=genann_act_linear
linear: all

test: test.o genann_cu.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check: test
	./$^

example1: example1.o genann_cu.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

example2: example2.o genann_cu.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

example3: example3.o genann_cu.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

example4: example4.o genann_cu.o
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

mnist_run: mnist_run.o mnist.o genann_cu.o
	$(NVCC) $(CFLAGS) mnist_run.o mnist.o genann_cu.o -o mnist_run

mnist.o: mnist.cu mnist.cuh genann.cuh
	$(NVCC) $(CFLAGS) -c mnist.cu -o mnist.o

mnist_run.o: mnist_run.cu mnist.cuh genann.cuh
	$(NVCC) $(CFLAGS) -c mnist_run.cu -o mnist_run.o

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o
	$(RM) test example1 example2 example3 example4 mnist_run *.exe
	$(RM) persist.txt

.PHONY: sigmoid threshold linear clean

