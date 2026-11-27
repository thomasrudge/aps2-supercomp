# Barenn — build e medição
#
#   make baseline        compila o código legado (barenn.cpp intocado) com -O2
#   make profile         mesmo build com -g (símbolos de linha para o callgrind)
#   make callgrind       roda o callgrind sobre o build de profile (precisa de valgrind/Linux)
#   make clean
#
# Compilador: make baseline CXX=clang++   (padrão: g++, que no macOS é o Apple clang)

CXX      ?= g++
STD      := -std=c++17
WARN     := -Wall -Wextra
SRC      := barenn.cpp

BASELINE_FLAGS := $(STD) $(WARN) -O2
PROFILE_FLAGS  := $(BASELINE_FLAGS) -g

CALLGRIND_OUT   ?= callgrind.out.baseline
CALLGRIND_FLAGS ?=

.PHONY: all baseline profile callgrind clean

all: baseline

baseline: barenn_baseline

barenn_baseline: $(SRC)
	$(CXX) $(BASELINE_FLAGS) -o $@ $<

profile: barenn_profile

barenn_profile: $(SRC)
	$(CXX) $(PROFILE_FLAGS) -o $@ $<

# Gera $(CALLGRIND_OUT) (dados brutos) e $(CALLGRIND_OUT).txt (custo próprio por
# função + anotação por linha). Ex.: CALLGRIND_FLAGS="--cache-sim=yes ..."
callgrind: barenn_profile
	valgrind --tool=callgrind $(CALLGRIND_FLAGS) --callgrind-out-file=$(CALLGRIND_OUT) ./barenn_profile
	callgrind_annotate --inclusive=no --auto=yes $(CALLGRIND_OUT) > $(CALLGRIND_OUT).txt
	head -40 $(CALLGRIND_OUT).txt

clean:
	rm -f barenn_baseline barenn_profile
