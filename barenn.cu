#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
constexpr int threads_por_bloco = 256;
// Pesos e vieses das camadas 64 -> 32 -> 16 -> 10, em sequência.
// Cada linha de pesos tem um float de padding.
constexpr int parametros_total = (entradas + 2) * 32 + (32 + 2) * 16 + (16 + 2) * classes;
// last_z, out, delta e grad_in de cada camada.
constexpr int buffers_total = (3 * 32 + entradas) + (3 * 16 + 32) + (3 * classes + 16);

// Transforma uma falha da API CUDA em exceção.
void checa(cudaError_t erro) {
    if (erro != cudaSuccess) {
        throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro));
    }
}

// Camada executada por todas as threads de um bloco. Os ponteiros apontam
// para a shared memory.
struct DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // weights[i * (in_features + 1) + j] liga a entrada j ao neurônio de saída i.
    // O padding evita que as threads de um warp caiam no mesmo banco da shared memory.
    float* weights;
    float* bias;
    // Dados do último forward usados para calcular os gradientes.
    const float* last_input = nullptr;
    float* last_z;
    // Buffers reutilizados a cada amostra.
    float* out;
    float* delta;
    float* grad_in;

    // Os parâmetros começam em p e os buffers em b. Os dois avançam.
    __device__ DenseLayer(int input_size, int output_size, bool is_final, float*& p, float*& b)
        : in_features(input_size), out_features(output_size), final(is_final),
          weights(p), bias(p + (input_size + 1) * output_size), last_z(b),
          out(b + output_size), delta(b + 2 * output_size), grad_in(b + 3 * output_size) {
        p += (input_size + 2) * output_size;
        b += 3 * output_size + input_size;
    }

    __device__ const float* Forward(const float* input) {
        last_input = input;
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        // A thread t calcula os neurônios t, t + blockDim.x, ...
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            float sum = bias[i];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * weights[i * (in_features + 1) + j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : fmaxf(0.0f, sum);
        }
        __syncthreads();
        return out;
    }

    __device__ const float* Backward(const float* grad_out, float learning_rate,
                                     bool compute_grad_in = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();

        // O gradiente da entrada usa os pesos anteriores à atualização.
        if (compute_grad_in) {
            for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
                float sum = 0;
                for (int j = 0; j < out_features; ++j) {
                    sum += delta[j] * weights[j * (in_features + 1) + i];
                }
                grad_in[i] = sum;
            }
            __syncthreads();
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta.
        for (int k = threadIdx.x; k < out_features * in_features; k += blockDim.x) {
            int i = k / in_features;
            int j = k % in_features;
            weights[i * (in_features + 1) + j] -= learning_rate * (delta[i] * last_input[j]);
        }
        for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
            bias[i] -= learning_rate * delta[i];
        }
        __syncthreads();
        return grad_in;
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
__host__ __device__ float perda(const float* escores, int rotulo,
                               float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

// Copia as imagens e os rótulos para a GPU, em arrays contíguos.
void envia(const vector<Amostra>& dados, float** d_imagens, int** d_rotulos) {
    vector<float> imagens;
    vector<int> rotulos;
    for (const auto& amostra : dados) {
        imagens.insert(imagens.end(), amostra.imagem.begin(), amostra.imagem.end());
        rotulos.push_back(amostra.rotulo);
    }
    checa(cudaMalloc(d_imagens, imagens.size() * sizeof(float)));
    checa(cudaMalloc(d_rotulos, rotulos.size() * sizeof(int)));
    checa(cudaMemcpy(*d_imagens, imagens.data(), imagens.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
    checa(cudaMemcpy(*d_rotulos, rotulos.data(), rotulos.size() * sizeof(int),
                     cudaMemcpyHostToDevice));
}

// Pesos iniciais de uma camada, na mesma ordem do construtor da CPU,
// seguidos do padding de cada linha e dos vieses.
void inicializa(vector<float>& parametros, int input_size, int output_size) {
    // A semente fixa torna a inicialização de cada camada reproduzível.
    mt19937 gen(1337);
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            parametros.push_back(distribution(gen));
        }
        parametros.push_back(0.0f);
    }
    parametros.insert(parametros.end(), output_size, 0.0f);
}

// Um único bloco treina todas as épocas, porque cada passo de SGD depende do
// anterior. Pesos, vieses e ativações ficam na shared memory do bloco.
__global__ void treino_kernel(float* parametros, const float* imagens, const int* rotulos,
                              int n, int epocas, float taxa, double* perdas) {
    __shared__ float memoria[parametros_total + buffers_total];
    __shared__ float imagem[entradas];
    __shared__ float gradiente[classes];
    for (int k = threadIdx.x; k < parametros_total; k += blockDim.x) {
        memoria[k] = parametros[k];
    }
    float* p = memoria;
    float* b = memoria + parametros_total;
    DenseLayer l1(entradas, 32, false, p, b);
    DenseLayer l2(32, 16, false, p, b);
    DenseLayer l3(16, classes, true, p, b);

    for (int epoca = 0; epoca < epocas; ++epoca) {
        double perda_total = 0;
        // Uma época percorre todo o conjunto na ordem do arquivo.
        for (int k = 0; k < n; ++k) {
            int rotulo = rotulos[k];
            for (int j = threadIdx.x; j < entradas; j += blockDim.x) {
                imagem[j] = imagens[k * entradas + j];
            }
            __syncthreads();
            const float* escores = l3.Forward(l2.Forward(l1.Forward(imagem)));
            if (threadIdx.x == 0) {
                perda_total += perda(escores, rotulo, gradiente);
            }
            __syncthreads();

            // Propaga o gradiente da última camada até a primeira.
            // Cada Backward também atualiza os parâmetros da sua camada.
            const float* grad3 = l3.Backward(gradiente, taxa);
            const float* grad2 = l2.Backward(grad3, taxa);
            // A primeira camada não precisa do gradiente da entrada.
            l1.Backward(grad2, taxa, false);
        }
        if (threadIdx.x == 0) {
            perdas[epoca] = perda_total / n;
        }
    }

    // Devolve os parâmetros treinados para a memória global.
    for (int k = threadIdx.x; k < parametros_total; k += blockDim.x) {
        parametros[k] = memoria[k];
    }
}

// Só forward. Cada bloco avalia as imagens blockIdx.x, blockIdx.x + gridDim.x, ...
__global__ void avaliacao_kernel(const float* parametros, const float* imagens, int n,
                                 float* escores) {
    __shared__ float memoria[parametros_total + buffers_total];
    __shared__ float imagem[entradas];
    for (int k = threadIdx.x; k < parametros_total; k += blockDim.x) {
        memoria[k] = parametros[k];
    }
    float* p = memoria;
    float* b = memoria + parametros_total;
    DenseLayer l1(entradas, 32, false, p, b);
    DenseLayer l2(32, 16, false, p, b);
    DenseLayer l3(16, classes, true, p, b);

    for (int k = blockIdx.x; k < n; k += gridDim.x) {
        for (int j = threadIdx.x; j < entradas; j += blockDim.x) {
            imagem[j] = imagens[k * entradas + j];
        }
        __syncthreads();
        const float* saida = l3.Forward(l2.Forward(l1.Forward(imagem)));
        for (int i = threadIdx.x; i < classes; i += blockDim.x) {
            escores[k * classes + i] = saida[i];
        }
    }
}

void avaliar(const vector<float>& todos_escores, const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Os escores vieram do forward na GPU. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        vector<float> escores(todos_escores.begin() + i * classes,
                              todos_escores.begin() + (i + 1) * classes);
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores.data(), amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        vector<float> parametros;
        Barenn::inicializa(parametros, Barenn::entradas, 32);
        Barenn::inicializa(parametros, 32, 16);
        Barenn::inicializa(parametros, 16, Barenn::classes);
        float* d_parametros;
        double* d_perdas;
        Barenn::checa(cudaMalloc(&d_parametros, parametros.size() * sizeof(float)));
        Barenn::checa(cudaMemcpy(d_parametros, parametros.data(),
                                 parametros.size() * sizeof(float), cudaMemcpyHostToDevice));
        Barenn::checa(cudaMalloc(&d_perdas, epocas * sizeof(double)));
        array<double, epocas> perdas{};
        float* d_imagens;
        int* d_rotulos;

        // A medição inclui a cópia do treino para a GPU, o kernel de treino
        // e a cópia das perdas de volta.
        const auto inicio = chrono::steady_clock::now();
        Barenn::envia(treino, &d_imagens, &d_rotulos);
        Barenn::treino_kernel<<<1, Barenn::threads_por_bloco>>>(
            d_parametros, d_imagens, d_rotulos, static_cast<int>(treino.size()), epocas, taxa,
            d_perdas);
        // A cópia espera o kernel terminar.
        Barenn::checa(cudaMemcpy(perdas.data(), d_perdas, epocas * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        // Falhas no lançamento dos kernels aparecem aqui.
        Barenn::checa(cudaGetLastError());

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        float* d_imagens_teste;
        int* d_rotulos_teste;
        float* d_escores;
        Barenn::envia(teste, &d_imagens_teste, &d_rotulos_teste);
        Barenn::checa(cudaMalloc(&d_escores, teste.size() * Barenn::classes * sizeof(float)));
        // Um bloco por imagem de teste.
        Barenn::avaliacao_kernel<<<static_cast<int>(teste.size()), Barenn::threads_por_bloco>>>(
            d_parametros, d_imagens_teste, static_cast<int>(teste.size()), d_escores);
        vector<float> escores(teste.size() * Barenn::classes);
        Barenn::checa(cudaMemcpy(escores.data(), d_escores, escores.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        Barenn::checa(cudaGetLastError());
        Barenn::avaliar(escores, teste);

        cudaFree(d_parametros);
        cudaFree(d_perdas);
        cudaFree(d_imagens);
        cudaFree(d_rotulos);
        cudaFree(d_imagens_teste);
        cudaFree(d_rotulos_teste);
        cudaFree(d_escores);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
