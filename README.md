# Barenn: resumo dos resultados

Rede 64→32→16→10 treinada por SGD (uma amostra por passo) no optdigits. Otimizei em blocos, cada um com Issue, branch, PR e tag anotada. O enunciado original continua abaixo da linha.

## Tabela consolidada

Tempo de treino de 51 épocas, média ± desvio de 5 execuções. Speedup = tempo da v0.0 na mesma máquina / tempo da versão.

| Versão | M4 Pro: treino | Speedup | Santos Dumont: treino | Speedup | Acurácia / perda |
|:--|--:|--:|--:|--:|:--|
| v0.0 baseline legado | 0.3414 s ± 0.0012 | 1.00× | 1.3039 s ± 0.0006 | 1.00× | 95.5% / 0.0432 |
| v0.1 memória | 0.1664 s ± 0.0025 | 2.05× | 0.8220 s ± 0.0011 | 1.59× | 95.5% / 0.0432 |
| v0.2 SIMD | 0.0828 s ± 0.0002 | 4.12× | 0.7416 s ± 0.0009 | 1.76× | 95.4% / 0.0447 |
| v0.3 OpenMP | 0.0812 s ± 0.0005 | 4.20× | — | — | 95.4% / 0.0447 |
| v0.4 CUDA inicial | — | — | 20.5284 s ± 0.1106 | 0.06× | 95.6% / 0.0438 |
| v0.5 CUDA residente | — | — | 5.2691 s ± 0.0244 | 0.25× | 95.6% / 0.0438 |
| v0.6 CUDA shared | — | — | 0.4601 s ± 0.0161 | 2.83× | 95.6% / 0.0438 |

M4 Pro: Apple M4 Pro (8P+4E, 24 GB), macOS 27.2.1, Apple clang 21 com as flags do Makefile de cada tag (v0.0 e v0.1 `-O2`; v0.2 `-O3 -mcpu=native -fopenmp-simd`; v0.3 com OpenMP, libomp 23.1.6, 12 threads). Não tem GPU NVIDIA.

Santos Dumont: nó sdumont8051 (job 11761660), 2× Xeon Gold 6252 e Tesla V100-SXM2-32GB. CPU com g++ 8.5 `-O2 -fopenmp` e 4 threads, GPU com CUDA 12.6 e `nvcc -O2 -arch=sm_70`. CPU e GPU rodaram no mesmo nó e no mesmo job, então a comparação entre elas vale: a v0.6 treina 1.6× mais rápido que a v0.2 (0.7416 s → 0.4601 s).

A v0.3 não muda o treino, porque o SGD por amostra é serial. O ganho dela é no tempo total do processo (leitura paralela): 0.1143 s → 0.0916 s no M4. Por isso ela ficou fora do job do Santos Dumont. Colunas diferentes não se comparam: no M4 a CPU com `-O3` nativo treina em 0.08 s, mais rápido que a V100, que usa 1 bloco em 1 dos 80 SMs ([#13](https://github.com/thomasrudge/aps2-supercomp/issues/13)).

## Índice

| Bloco | Issue (diagnóstico, profiling e análise) | PR | Tag |
|:-:|:--|:-:|:--|
| 1 | [#1 Baseline: medição e profiling do código legado](https://github.com/thomasrudge/aps2-supercomp/issues/1) | [#2](https://github.com/thomasrudge/aps2-supercomp/pull/2) | [v0.0-baseline](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.0-baseline) |
| 2 | [#3 Memória: pesos contíguos, sem cópias e sem grad_in descartado](https://github.com/thomasrudge/aps2-supercomp/issues/3) | [#4](https://github.com/thomasrudge/aps2-supercomp/pull/4) | [v0.1-memoria](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.1-memoria) |
| 3 | [#5 SIMD: -O3 native, grad_in por linha e Forward vetorizado](https://github.com/thomasrudge/aps2-supercomp/issues/5) | [#6](https://github.com/thomasrudge/aps2-supercomp/pull/6) | [v0.2-simd](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.2-simd) |
| 4 | [#7 OpenMP: avaliação e leitura paralelas (SGD serial, Amdahl)](https://github.com/thomasrudge/aps2-supercomp/issues/7) | [#8](https://github.com/thomasrudge/aps2-supercomp/pull/8) | [v0.3-openmp](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.3-openmp) |
| 5 | [#9 CUDA inicial: porte dos laços para kernels](https://github.com/thomasrudge/aps2-supercomp/issues/9) | [#10](https://github.com/thomasrudge/aps2-supercomp/pull/10) | [v0.4-cuda-inicial](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.4-cuda-inicial) |
| 6 | [#11 CUDA residente: dados na GPU e sem cópias no laço](https://github.com/thomasrudge/aps2-supercomp/issues/11) | [#12](https://github.com/thomasrudge/aps2-supercomp/pull/12) | [v0.5-cuda-residente](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.5-cuda-residente) |
| 7 | [#13 CUDA shared: kernel único com pesos em shared memory](https://github.com/thomasrudge/aps2-supercomp/issues/13) | [#14](https://github.com/thomasrudge/aps2-supercomp/pull/14) | [v0.6-cuda-shared](https://github.com/thomasrudge/aps2-supercomp/releases/tag/v0.6-cuda-shared) |

Os perfis brutos (callgrind e nsys) estão em [profiling/](profiling/).

## Como reproduzir

```bash
# CPU. No macOS precisa da libomp (brew install libomp).
make && ./bench.sh && ./validate.sh             # versão atual de CPU (v0.3)
make baseline && ./bench.sh ./barenn_baseline   # código legado, extraído da tag v0.0-baseline
make callgrind                                  # perfil de CPU (precisa de valgrind/Linux)

# GPU, num nó com NVIDIA (compila para sm_70, a V100).
make cuda && ./bench.sh ./barenn_cuda && ./validate.sh ./barenn_cuda
make nsys                                       # perfil de GPU (Nsight Systems)
```

Para uma versão antiga: `git checkout <tag>` e os mesmos comandos (na v0.0 só existe `make baseline`).

No Santos Dumont, o `sd_job.sh` mede todas as versões no mesmo nó (5 execuções, validação e nsys). Ele lê o fonte de cada tag de uma pasta:

```bash
mkdir -p sd/cpu sd/bloco5 sd/bloco6 sd/bloco7
cp sd_job.sh optdigits.tra optdigits.tes referencia.txt sd/
for t in v0.0-baseline v0.1-memoria v0.2-simd v0.3-openmp; do git show $t:barenn.cpp > sd/cpu/barenn_$t.cpp; done
git show v0.4-cuda-inicial:barenn.cu   > sd/bloco5/barenn.cu
git show v0.5-cuda-residente:barenn.cu > sd/bloco6/barenn.cu
git show v0.6-cuda-shared:barenn.cu    > sd/bloco7/barenn.cu
cd sd && sbatch sd_job.sh                       # tempos em sd/resumo_<job>.txt
```

---

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
