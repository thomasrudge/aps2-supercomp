#!/usr/bin/env bash
# Compara a acurácia e a perda média da AVALIACAO FINAL com a referência.
#
#   ./validate.sh [binario] [referencia]
#   ./validate.sh                    # = ./validate.sh ./barenn_baseline referencia.txt
#
# Tolerâncias (sobrescreva por variável de ambiente):
#   TOL_ACC   diferença máxima de acurácia, em pontos percentuais (padrão 0.5)
#   TOL_PERDA diferença máxima absoluta da perda média         (padrão 0.002)
# Só trocar compilador/flags, sem mexer no código, já muda o resultado em
# 0.1 p.p. e 0.0004 (arredondamento/FMA em ponto flutuante), por isso
# igualdade exata não serve de critério.
set -euo pipefail
export LC_ALL=C

BIN=${1:-./barenn_baseline}
REF=${2:-referencia.txt}
TOL_ACC=${TOL_ACC:-0.5}
TOL_PERDA=${TOL_PERDA:-0.002}

if [[ ! -x $BIN ]]; then
    echo "validate: binário '$BIN' não encontrado (rode 'make baseline')" >&2
    exit 1
fi
if [[ ! -r $REF ]]; then
    echo "validate: referência '$REF' não encontrada" >&2
    exit 1
fi

# Lê a saída do programa e imprime "acuracia perda" da avaliação final.
# "^Perda media" (maiúscula) só casa com a avaliação; as épocas usam "perda media".
extrai() {
    awk '/^Acuracia/    { a = $2; sub("%", "", a) }
         /^Perda media/ { p = $3 }
         END { if (a == "" || p == "") exit 1; print a, p }'
}

if ! ref=$(extrai <"$REF"); then
    echo "validate: 'Acuracia'/'Perda media' não encontrados em $REF" >&2
    exit 1
fi
if ! saida=$("$BIN"); then
    echo "validate: FALHOU — $BIN terminou com erro" >&2
    exit 1
fi
if ! obtido=$(extrai <<<"$saida"); then
    echo "validate: FALHOU — 'Acuracia'/'Perda media' não encontrados na saída de $BIN" >&2
    exit 1
fi

awk -v r="$ref" -v o="$obtido" -v ta="$TOL_ACC" -v tp="$TOL_PERDA" -v bin="$BIN" -v refarq="$REF" '
    function abs(x) { return x < 0 ? -x : x }
    BEGIN {
        split(r, R, " "); split(o, O, " ")
        da = O[1] - R[1]; dp = O[2] - R[2]
        printf "%s vs %s\n", bin, refarq
        printf "%-10s %10s %10s %10s %10s\n", "metrica", "referencia", "obtido", "diferenca", "tolerancia"
        printf "%-10s %10s %10s %+10.1f %10s\n", "acuracia%", R[1], O[1], da, ta
        printf "%-10s %10s %10s %+10.4f %10s\n", "perda", R[2], O[2], dp, tp
        ok = abs(da) <= ta + 1e-9 && abs(dp) <= tp + 1e-9
        if (!ok)                    print "FALHOU: fora da tolerancia"
        else if (da == 0 && dp == 0) print "OK: identico a referencia"
        else                        print "OK: dentro da tolerancia"
        exit ok ? 0 : 1
    }'
