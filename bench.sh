#!/usr/bin/env bash
# Roda o binário N vezes e resume o "Tempo de treino" (média, desvio-padrão amostral, mín, máx).
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

tempos=""
for ((i = 1; i <= N; i++)); do
    saida=$("$BIN")
    t=$(awk '/^Tempo de treino/ { print $4 }' <<<"$saida")
    if [[ -z $t ]]; then
        echo "bench: linha 'Tempo de treino' não encontrada na saída de $BIN" >&2
        exit 1
    fi
    printf 'execucao %d/%d: %s s\n' "$i" "$N" "$t"
    tempos+="$t"$'\n'
done

printf '%s' "$tempos" | awk -v bin="$BIN" '
    { v[NR] = $1; soma += $1 }
    END {
        n = NR; media = soma / n
        min = max = v[1]
        for (i = 1; i <= n; i++) {
            d = v[i] - media; ss += d * d
            if (v[i] < min) min = v[i]
            if (v[i] > max) max = v[i]
        }
        dp = (n > 1) ? sqrt(ss / (n - 1)) : 0
        printf "\n%s | n=%d\n", bin, n
        printf "Tempo de treino: media %.4f s | desvio %.4f s (%.1f%%) | min %.4f s | max %.4f s\n",
               media, dp, 100 * dp / media, min, max
    }'
