# Barenn — build e medição
#
#   make  (ou make barenn)  compila a versão atual (barenn.cpp) com -O3 e -mcpu/-march=native
#   make baseline           compila o código legado da tag $(BASELINE_REF) com -O2
#   make profile            versão atual com -g (símbolos de linha para o callgrind)
#   make callgrind          roda o callgrind sobre o build de profile (precisa de valgrind/Linux)
#   make cuda               compila a versão CUDA (barenn.cu) com nvcc -O2
#   make nsys               roda o Nsight Systems sobre a versão CUDA (precisa de GPU)
#   make clean
#
# Compilador: make CXX=clang++   (padrão: g++, que no macOS é o Apple clang)

CXX      ?= g++
STD      := -std=c++17
WARN     := -Wall -Wextra
SRC      := barenn.cpp

# -mcpu=native no ARM (Apple M4), -march=native no x86.
ifneq (,$(filter arm64 aarch64,$(shell uname -m)))
NATIVE := -mcpu=native
else
NATIVE := -march=native
endif

# OpenMP: no macOS o Apple clang usa a libomp do Homebrew (brew install libomp).
ifeq ($(shell uname -s),Darwin)
LIBOMP ?= /opt/homebrew/opt/libomp
OPENMP := -Xpreprocessor -fopenmp -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp
else
OPENMP := -fopenmp
endif

RELEASE_FLAGS  := $(STD) $(WARN) -O3 $(NATIVE) $(OPENMP)
BASELINE_FLAGS := $(STD) $(WARN) -O2
PROFILE_FLAGS  := $(RELEASE_FLAGS) -g

# O baseline é extraído do histórico do git, então "antes" e "depois" são
# compilados lado a lado com o mesmo compilador.
BASELINE_REF := v0.0-baseline
BASELINE_SRC := build/barenn_$(BASELINE_REF).cpp

CALLGRIND_OUT   ?= callgrind.out
CALLGRIND_FLAGS ?=

# CUDA: sm_70 é a V100 do Santos Dumont. GPUs mais novas usam o PTX embutido.
NVCC       ?= nvcc
CUDA_ARCH  ?= sm_70
CUDA_SRC   := barenn.cu
CUDA_FLAGS := $(STD) -O2 -arch=$(CUDA_ARCH) -lineinfo
# O nvcc usa o g++ do PATH; o g++ 8 (Santos Dumont) precisa de -lstdc++fs para std::filesystem.
ifeq ($(shell g++ -dumpversion 2>/dev/null | cut -d. -f1),8)
CUDA_LIBS := -lstdc++fs
endif

NSYS_OUT   ?= nsys
NSYS_FLAGS ?=

.PHONY: all baseline profile callgrind cuda nsys clean

all: barenn

barenn: $(SRC) Makefile
	$(CXX) $(RELEASE_FLAGS) -o $@ $<

baseline: barenn_baseline

barenn_baseline: $(BASELINE_SRC) Makefile
	$(CXX) $(BASELINE_FLAGS) -o $@ $<

$(BASELINE_SRC):
	mkdir -p build
	git show $(BASELINE_REF):barenn.cpp > $@.tmp && mv $@.tmp $@

profile: barenn_profile

barenn_profile: $(SRC) Makefile
	$(CXX) $(PROFILE_FLAGS) -o $@ $<

# Gera $(CALLGRIND_OUT) (dados brutos) e $(CALLGRIND_OUT).txt (custo próprio por
# função + anotação por linha). Ex.: CALLGRIND_FLAGS="--cache-sim=yes ..."
callgrind: barenn_profile
	valgrind --tool=callgrind $(CALLGRIND_FLAGS) --callgrind-out-file=$(CALLGRIND_OUT) ./barenn_profile
	callgrind_annotate --inclusive=no --auto=yes $(CALLGRIND_OUT) > $(CALLGRIND_OUT).txt
	head -40 $(CALLGRIND_OUT).txt

cuda: barenn_cuda

barenn_cuda: $(CUDA_SRC) Makefile
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(CUDA_LIBS)

# Gera $(NSYS_OUT).nsys-rep e $(NSYS_OUT).txt (API CUDA, kernels e cópias).
# Com milhões de kernels o relatório fica grande: NSYS_FLAGS=--duration=20 limita a captura.
nsys: barenn_cuda
	nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true $(NSYS_FLAGS) -o $(NSYS_OUT) ./barenn_cuda
	nsys stats --force-export=true --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum $(NSYS_OUT).nsys-rep > $(NSYS_OUT).txt
	rm -f $(NSYS_OUT).sqlite
	head -20 $(NSYS_OUT).txt

clean:
	rm -f barenn barenn_baseline barenn_profile barenn_cuda
	rm -rf build
