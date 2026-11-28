# Barenn — build e medição
#
#   make  (ou make barenn)  compila a versão atual (barenn.cpp) com -O3 e -mcpu/-march=native
#   make baseline           compila o código legado da tag $(BASELINE_REF) com -O2
#   make profile            versão atual com -g (símbolos de linha para o callgrind)
#   make callgrind          roda o callgrind sobre o build de profile (precisa de valgrind/Linux)
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

.PHONY: all baseline profile callgrind clean

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

clean:
	rm -f barenn barenn_baseline barenn_profile
	rm -rf build
