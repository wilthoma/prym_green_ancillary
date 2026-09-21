# CUDA_ARCH=120 matches the paper's Blackwell GPUs. Override for another GPU.
CARGO ?= cargo
PYTHON ?= python3
NVCC ?= nvcc
CUDA_ARCH ?= 120
NVCCFLAGS ?= -O3 -std=c++17
# For toolkits requiring a particular host compiler: make CUDA_CXX=/path/to/g++
CUDA_CXX ?=
CUDA_HOST = $(if $(CUDA_CXX),-ccbin $(CUDA_CXX),)
CUDA_PROGRAMS = cuprym_cyclic cuprym_cyclic_kernel_vectors cuprym_deformation cuprym_deformation_kernel_vectors
CUDA_BINARIES = $(addprefix cuda/,$(CUDA_PROGRAMS))
CUDA_HEADERS = $(wildcard cuda/*.h cuda/include/*.hpp)

.PHONY: all cpu cuda test test-python
all: cpu cuda

cpu:
	$(CARGO) build --release --locked --workspace

cuda: $(CUDA_BINARIES)

cuda/%: cuda/%.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST) -gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) $< -o $@ -lzstd

# CPU correctness tests; no CUDA execution or downloaded results.
test:
	$(CARGO) test --locked --workspace
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v
