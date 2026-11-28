#!/bin/bash
#SBATCH --job-name=barenn_cuda
#SBATCH --partition=merry_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=barenn_cuda_%j.out

# Compila e mede os blocos 5, 6 e 7 na GPU do nó: N execuções, validação, nsys e ncu.
# Também mede, no mesmo nó, cada versão de CPU que estiver em cpu/*.cpp.
#
#   cd aps_blocos && sbatch sd_job.sh
#
# Resultados em blocoN/resultados/ e cpu/resultados_<nome>/.
# Ambiente e resumo dos tempos em resumo_<job>.txt.
# Variáveis opcionais: N, BLOCOS, CUDA_MODULE, ARCH, TOL_ACC, TOL_PERDA, NSYS_DURACAO_BLOCO5.

set -uo pipefail
export LC_ALL=C
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

RAIZ=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
cd "$RAIZ" || exit 1
N=${N:-5}
BLOCOS=${BLOCOS:-"bloco5 bloco6 bloco7"}
TREINO=$RAIZ/optdigits.tra
TESTE=$RAIZ/optdigits.tes
REF=$RAIZ/referencia.txt
TOL_ACC=${TOL_ACC:-0.5}
TOL_PERDA=${TOL_PERDA:-0.002}
RESUMO=$RAIZ/resumo_${SLURM_JOB_ID:-local}.txt

for arquivo in "$TREINO" "$TESTE" "$REF"; do
    [[ -r $arquivo ]] || { echo "faltando $arquivo"; exit 1; }
done

module load "${CUDA_MODULE:-cuda/12.8.1}"

# Compila para a GPU do nó (ex.: sm_80). Se o driver não informar, usa native.
if [[ -z ${ARCH:-} ]]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    ARCH=${cc:+sm_$cc}
    ARCH=${ARCH:-native}
fi
NVCC_FLAGS="-std=c++17 -O2 -arch=$ARCH -lineinfo -Xptxas -v"

{
    echo "job ${SLURM_JOB_ID:-local} | $(hostname) | $(date)"
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max --format=csv
    nvcc --version | tail -2
    g++ --version | head -1
    nsys --version
    ncu --version | tail -1
    lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core"
    echo "nvcc $NVCC_FLAGS | OMP_NUM_THREADS=$OMP_NUM_THREADS | N=$N"
    echo
} > "$RESUMO" 2>&1
nvidia-smi > "$RAIZ/nvidia-smi_${SLURM_JOB_ID:-local}.txt" 2>&1

# Imprime "acuracia perda" da avaliação final de uma saída.
metricas() {
    awk '/^Acuracia/    { a = $2; sub("%", "", a) }
         /^Perda media/ { p = $3 }
         END { if (a == "" || p == "") exit 1; print a, p }' "$1"
}

# Compara a saída com a referência, com as tolerâncias do validate.sh.
valida() {
    awk -v r="$(metricas "$REF")" -v o="$(metricas "$1")" -v ta="$TOL_ACC" -v tp="$TOL_PERDA" -v f="$(basename "$1")" '
        function abs(x) { return x < 0 ? -x : x }
        BEGIN {
            if (o == "") { printf "FALHOU %s: sem avaliacao na saida\n", f; exit 1 }
            split(r, R, " "); split(o, O, " ")
            da = O[1] - R[1]; dp = O[2] - R[2]
            ok = abs(da) <= ta + 1e-9 && abs(dp) <= tp + 1e-9
            printf "%s %s: acuracia %s (%+.1f p.p.) | perda %s (%+.4f)\n",
                   ok ? "OK" : "FALHOU", f, O[1], da, O[2], dp
            exit !ok
        }'
}

# Média, desvio-padrão amostral, mínimo e máximo do "Tempo de treino".
estatistica() {
    awk '/^Tempo de treino/ { v[++n] = $4; s += $4 }
         END {
             if (!n) { print "sem tempos"; exit 1 }
             m = s / n; mn = mx = v[1]
             for (i = 1; i <= n; i++) {
                 d = v[i] - m; ss += d * d
                 if (v[i] < mn) mn = v[i]
                 if (v[i] > mx) mx = v[i]
             }
             dp = n > 1 ? sqrt(ss / (n - 1)) : 0
             printf "n=%d | media %.4f s | desvio %.4f s (%.1f%%) | min %.4f s | max %.4f s\n",
                    n, m, dp, 100 * dp / m, mn, mx
         }' "$@"
}

# Roda o binário N vezes, valida cada execução e confere se as saídas são iguais.
mede() {
    local bin=$1 dir=$2 nome=$3 ok=0 iguais=sim
    for ((i = 1; i <= N; i++)); do
        "$bin" "$TREINO" "$TESTE" > "$dir/execucao_$i.txt" 2>&1
        valida "$dir/execucao_$i.txt" >> "$dir/validacao.txt" && ok=$((ok + 1))
        cmp -s <(grep -v '^Tempo de treino' "$dir/execucao_1.txt") \
               <(grep -v '^Tempo de treino' "$dir/execucao_$i.txt") || iguais=nao
    done
    estatistica "$dir"/execucao_*.txt > "$dir/tempos.txt"
    printf '%-22s %s | validacao %d/%d | saidas iguais: %s\n' \
           "$nome" "$(cat "$dir/tempos.txt")" "$ok" "$N" "$iguais" | tee -a "$RESUMO"
}

# Referências de CPU no mesmo nó (ex.: cpu/barenn_v0.0-baseline.cpp).
for src in "$RAIZ"/cpu/*.cpp; do
    [[ -e $src ]] || continue
    nome=$(basename "$src" .cpp)
    R=$RAIZ/cpu/resultados_$nome
    rm -rf "$R" && mkdir -p "$R"
    if ! g++ -std=c++17 -O2 -fopenmp -o "$RAIZ/cpu/$nome" "$src" > "$R/compilacao.txt" 2>&1 &&
       ! g++ -std=c++17 -O2 -fopenmp -o "$RAIZ/cpu/$nome" "$src" -lstdc++fs >> "$R/compilacao.txt" 2>&1; then
        echo "$nome: falha na compilacao ($R/compilacao.txt)" | tee -a "$RESUMO"
        continue
    fi
    mede "$RAIZ/cpu/$nome" "$R" "$nome"
done

for b in $BLOCOS; do
    D=$RAIZ/$b
    R=$D/resultados
    rm -rf "$R" && mkdir -p "$R"

    # -lstdc++fs só é necessário com g++ 8 (std::filesystem).
    if ! nvcc $NVCC_FLAGS -o "$D/barenn" "$D/barenn.cu" > "$R/compilacao.txt" 2>&1 &&
       ! nvcc $NVCC_FLAGS -o "$D/barenn" "$D/barenn.cu" -lstdc++fs >> "$R/compilacao.txt" 2>&1; then
        echo "$b: falha na compilacao ($R/compilacao.txt)" | tee -a "$RESUMO"
        continue
    fi
    mede "$D/barenn" "$R" "$b"

    # Nsight Systems: API CUDA, kernels e cópias. O bloco 5 gera milhões de eventos,
    # então só os primeiros segundos são capturados.
    if command -v nsys > /dev/null; then
        duracao=()
        [[ $b == bloco5 ]] && duracao=(--duration="${NSYS_DURACAO_BLOCO5:-20}")
        nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
            ${duracao[@]+"${duracao[@]}"} -o "$R/nsys" "$D/barenn" "$TREINO" "$TESTE" > "$R/nsys_execucao.txt" 2>&1
        nsys stats --force-export=true \
            --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
            "$R/nsys.nsys-rep" > "$R/nsys_stats.txt" 2>&1 ||
        nsys stats --force-export=true --report cudaapisum,gpukernsum,gpumemtimesum,gpumemsizesum \
            "$R/nsys.nsys-rep" > "$R/nsys_stats.txt" 2>&1
        rm -f "$R/nsys.sqlite"
    fi

    # Nsight Compute: pula 100 passos de SGD e perfila todos os kernels de um passo.
    # No bloco 7 são só dois lançamentos: treino e avaliação.
    if command -v ncu > /dev/null; then
        case $b in
            bloco5) filtro=(--launch-skip 1100 --launch-count 11) ;;
            bloco6) filtro=(--launch-skip 1200 --launch-count 12) ;;
            *)      filtro=(--launch-count 2) ;;
        esac
        ncu --set full --import-source yes "${filtro[@]}" -f -o "$R/ncu" \
            "$D/barenn" "$TREINO" "$TESTE" > "$R/ncu_execucao.txt" 2>&1
        if grep -q ERR_NVGPUCTRPERM "$R/ncu_execucao.txt"; then
            echo "$b: ncu sem permissao para os contadores (ERR_NVGPUCTRPERM)" | tee -a "$RESUMO"
        else
            ncu --import "$R/ncu.ncu-rep" --page details > "$R/ncu_detalhes.txt" 2>&1
            grep -E "^  [A-Za-z_:]+\(|Duration|Registers Per Thread|Static Shared Memory Per Block|Theoretical Occupancy|Achieved Occupancy|Compute \(SM\) Throughput|Memory Throughput|Block Size|Grid Size|Waves Per SM" \
                "$R/ncu_detalhes.txt" > "$R/ncu_resumo.txt"
        fi
    fi
done

echo
cat "$RESUMO"
