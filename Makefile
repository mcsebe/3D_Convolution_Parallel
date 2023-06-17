CUDA_PATH ?= /usr/local/cuda

CC := $(CUDA_PATH)/bin/nvcc
SRCS := main.cu
TARGET := prog

INCLUDES := -I$(CUDA_PATH)/include
LIBS := -L$(CUDA_PATH)/lib64 -lcufft

all:
	$(CC) $(SRCS) -o $(TARGET) $(INCLUDES) $(LIBS)
