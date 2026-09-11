# Mini Projeto Avaliativo — Visão Computacional
## Machine Learning e Visão Computacional

Mini-projeto desenvolvido como atividade avaliativa do Módulo 2 — Semana 07, utilizando técnicas de Visão Computacional clássica com OpenCV e Aprendizado Profundo com TensorFlow/Keras.

---
## 📹 Vídeo de Apresentação
* **Link da gravação:** [https://drive.google.com/file/d/1YXvKie4PPN5v_bL7pCEpF-k1qHJKJefu/view?usp=sharing](https://drive.google.com/file/d/1YXvKie4PPN5v_bL7pCEpF-k1qHJKJefu/view?usp=sharing)
---

## 1. Objetivo
O projeto tem como objetivo desenvolver um pipeline de inspeção visual de peças metálicas, combinando técnicas tradicionais de processamento de imagens com uma Rede Neural Convolucional (CNN).
O projeto é dividido em duas etapas principais:

1. **Análise exploratória com OpenCV:** processamento de uma amostra de imagens para visualizar características dos componentes e destacar possíveis defeitos.
2. **Classificação automatizada com TensorFlow/Keras:** utilização do conjunto completo de imagens para treinar uma CNN capaz de classificar as peças em duas categorias: OK ou Defeituosa.

---

## 2. Dataset
Foi utilizado o dataset público *Casting Product Image Data for Quality Inspection*, composto por imagens de peças de fundição metálica com e sem defeitos.
O dataset é organizado em diretórios correspondentes às classes utilizadas durante o treinamento do modelo.
Para o desenvolvimento no Google Colab, o dataset foi armazenado no Google Drive e carregado diretamente pelo notebook.

### Divisão dos dados
O carregamento foi realizado utilizando `image_dataset_from_directory`, com:
* 80% dos dados para treinamento;
* 20% dos dados para validação;
* tamanho das imagens redimensionado para 128 × 128 pixels;
* tamanho do lote (*batch_size*) de 32 imagens.

A base utilizada possui 1.300 imagens, sendo:
* 1.040 imagens para treinamento;
* 260 imagens para validação.

---

## 3. Tecnologias Utilizadas
* Python
* Google Colab
* OpenCV
* NumPy
* Matplotlib
* TensorFlow
* Keras
* Git & GitHub

---

## 4. Desenvolvimento por Sprints

### Sprint 1 — Configuração e Versionamento
Nesta etapa foi preparado o ambiente de desenvolvimento e realizado o versionamento do projeto.
* Configuração do ambiente Google Colab;
* Montagem do Google Drive;
* Configuração do Git;
* Conexão com o repositório GitHub;
* Organização do projeto em branches e commits;
* Preparação do acesso ao dataset.

O desenvolvimento foi versionado no GitHub de forma incremental, mantendo commits distintos para as funcionalidades desenvolvidas em cada Sprint.

### Sprint 2 — Análise Exploratória com OpenCV
Foi selecionada uma imagem de amostra do conjunto de dados para realizar a análise exploratória utilizando OpenCV.
Foram aplicadas duas etapas principais de pré-processamento:

* **Conversão para escala de cinza:** A imagem original foi convertida do formato BGR para Grayscale, reduzindo a informação de cor e facilitando a análise das características estruturais da peça.
* **Suavização:** Foi aplicado o filtro *GaussianBlur*, utilizando um kernel de 5 × 5, com o objetivo de reduzir ruídos e pequenas variações da imagem antes das etapas seguintes de processamento.

### Sprint 3 — Destaque de Características
Após a conversão para escala de cinza e suavização, foram aplicadas técnicas para destacar características da imagem.

* **Thresholding:** Foi aplicada a técnica de limiarização binária, separando os pixels da imagem de acordo com um valor de limiar.
* **Detecção de bordas:** Foi utilizado o algoritmo Canny para identificar bordas e alterações de intensidade na imagem. Essa etapa permite visualizar melhor contornos e características estruturais que podem estar relacionadas aos defeitos da peça.
* **Morfologia:** Foi aplicada a operação morfológica de Dilatação, utilizando um kernel de 3 × 3. A dilatação permite reforçar e conectar determinadas regiões identificadas durante a detecção de bordas.

**Pipeline OpenCV:**
`Imagem original` → `Grayscale` → `GaussianBlur` → `Threshold` → `Canny` → `Dilatação`

### Sprint 4 — Ingestão de Dados e Data Augmentation
Para o treinamento da rede neural, o dataset completo foi carregado automaticamente utilizando `image_dataset_from_directory`. A divisão entre treinamento e validação foi realizada utilizando `validation_split`, mantendo a mesma semente (*seed*) para garantir uma divisão reprodutível.

**Data Augmentation:**
Foi implementado um pipeline dinâmico de aumento de dados utilizando Keras com as seguintes transformações:
* `RandomFlip`
* `RandomRotation`
* `RandomZoom`
* `RandomBrightness`

As transformações geométricas simulam pequenas variações de posicionamento e orientação das peças, enquanto a alteração de brilho busca representar variações de iluminação durante uma possível inspeção industrial.

### Sprint 5 — Arquitetura da CNN e Treinamento
Foi construída uma Rede Neural Convolucional utilizando a API `Sequential` do Keras.
A arquitetura utilizada foi composta por:
* Data Augmentation;
* Normalização dos pixels com `Rescaling`;
* `Conv2D` com 32 filtros + `MaxPooling2D`;
* `Conv2D` com 64 filtros + `MaxPooling2D`;
* `Conv2D` com 128 filtros + `MaxPooling2D`;
* `Flatten`;
* `Dense` com 64 neurônios + `Dropout(0.5)`;
* `Dense` final com 1 neurônio e ativação `sigmoid`.

**Compilação:**
* Otimizador: `Adam`
* Função de perda: `Binary Crossentropy`
* Métrica: `Accuracy`

Como o problema possui duas classes, a camada final utiliza uma função sigmoid, adequada para classificação binária. O treinamento foi realizado durante 15 épocas.

### Sprint 6 — Auditoria dos Resultados
Ao final do treinamento foi gerado um relatório gráfico utilizando Matplotlib analisando as curvas de Acurácia e Perda (*Loss*).

**Resultado Final:**

| Métrica | Treinamento | Validação |
| :--- | :---: | :---: |
| **Accuracy** | 72,8% | 78,1% |
| **Loss** | 0,551 | 0,479 |

**Análise:**
As curvas não apresentam o comportamento clássico de *overfitting*, no qual a perda de treinamento continua diminuindo enquanto a perda de validação aumenta de forma consistente.
No resultado obtido, a validação acompanhou o comportamento do treinamento e terminou com uma *Loss* inferior à do treinamento. Esse comportamento está diretamente relacionado à utilização de *Data Augmentation* e *Dropout*, que tornam o processo de treinamento mais desafiador para o modelo durante a fase de treino.

---

## 5. Pipeline Completo

```text
                 DATASET
                    │
          ┌─────────┴─────────┐
          │                   │
       OpenCV             TensorFlow/Keras
          │                   │
     Amostra              Dataset completo
          │                   │
      Grayscale              │
          ↓                   │
    GaussianBlur              │
          ↓                   │
      Threshold               │
          ↓                   │
        Canny                 │
          ↓                   │
      Dilatação               │
          │                   ↓
          │             Data Augmentation
          │                   ↓
          │                 CNN
          │                   ↓
          │              Classificação
          │                   │
          └───────────┬───────┘
                      ↓
               Análise dos resultados
