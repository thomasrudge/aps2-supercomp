#!/usr/bin/env bash
# Roda o binário N vezes e resume o "Tempo de treino" e o tempo total do processo
# (leitura, treino e avaliação): média, desvio-padrão amostral, mín, máx.
#
#   ./bench.sh [binario] [repeticoes]
#   ./bench.sh                       # = ./bench.sh ./barenn 5
set -euo pipefail
export LC_ALL=C

BIN=${1:-./barenn}
N=${2:-5}

if [[ ! -x $BIN ]]; then
    echo "bench: binário '$BIN' não encontrado (rode 'make')" >&2
    exit 1
fi

# Executa o binário e acrescenta à saída o tempo total, medido de fora do processo.
roda() {
    python3 -c '
import subprocess, sys, time
t = time.perf_counter()
r = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, text=True)
print(r.stdout, end="")
print("Tempo total %.4f s" % (time.perf_counter() - t))
sys.exit(r.returncode)' "$@"
}

# Aquecimento descartado: no macOS a 1a execução de um binário novo inclui a verificação do sistema.
roda "$BIN" > /dev/null

tempos=""
for ((i = 1; i <= N; i++)); do
    saida=$(roda "$BIN")
    t=$(awk '/^Tempo de treino/ { print $4 }' <<<"$saida")
    total=$(awk '/^Tempo total/ { print $3 }' <<<"$saida")
    if [[ -z $t ]]; then
        echo "bench: linha 'Tempo de treino' não encontrada na saída de $BIN" >&2
        exit 1
    fi
    printf 'execucao %d/%d: treino %s s | total %s s\n' "$i" "$N" "$t" "$total"
    tempos+="$t $total"$'\n'
done

printf '%s' "$tempos" | awk -v bin="$BIN" '
    function resumo(nome, c,    i, s, m, d, ss, mn, mx, dp) {
        for (i = 1; i <= NR; i++) s += v[i, c]
        m = s / NR; mn = mx = v[1, c]
        for (i = 1; i <= NR; i++) {
            d = v[i, c] - m; ss += d * d
            if (v[i, c] < mn) mn = v[i, c]
            if (v[i, c] > mx) mx = v[i, c]
        }
        dp = (NR > 1) ? sqrt(ss / (NR - 1)) : 0
        printf "%s: media %.4f s | desvio %.4f s (%.1f%%) | min %.4f s | max %.4f s\n",
               nome, m, dp, 100 * dp / m, mn, mx
    }
    { v[NR, 1] = $1; v[NR, 2] = $2 }
    END {
        printf "\n%s | n=%d\n", bin, NR
        resumo("Tempo de treino", 1)
        resumo("Tempo total", 2)
    }'
