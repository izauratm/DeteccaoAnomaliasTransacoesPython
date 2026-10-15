<div align="center">
  
# 💳 Detecção de Fraudes em Cartões de Crédito com Machine Learning

</div>

Este repositório contém a solução do desafio de detecção de fraudes em transações de cartão de crédito. O foco do projeto foi construir um pipeline ponta a ponta em Python para lidar com **dados extremamente desbalanceados**, otimizar o **Recall da classe de fraude** e fornecer **explicabilidade do modelo** (XAI) usando SHAP.

O projeto nasceu a partir do curso do Bootcamp **Accenture - Python para Análise e Automação de Dados**, oferecido pela plataforma [Dio.me](https://web.dio.me/). Dentro dessa formação, recebi o desafio de desenvolver o projeto **"Detecção de Anomalias em Transações em Python"**,  para treinar modelos que detectam fraudes em transações reais de cartão de crédito, um problema em que a fraude é rara e a acurácia engana.

---

## 🎯 1. O Problema e o Desafio do Desbalanceamento

A detecção de fraudes financeiras lida com um cenário em que as transações legítimas representam a esmagadora maioria dos dados, enquanto as fraudes são eventos raros.

- **Proporção da Base:** 99.8% das transações são normais (`Class 0`) e apenas **0.17%** são fraudulentas (`Class 1`).
- **Por que a Acurácia Engana?** Um modelo ingênuo que simplesmente classifique todas as transações como "normais" alcançará uma acurácia de **99.83%**. No entanto, ele falhará em capturar **100% das fraudes**, gerando um prejuízo financeiro massivo para a instituição.

### Métricas Decisivas
Neste cenário, a avaliação é redirecionada para a classe minoritária (`Class 1`):
* **Recall (Sensibilidade):** Métrica principal. Indica qual porcentagem das fraudes reais o modelo conseguiu capturar.
* **Precisão:** Mede quantas das transações marcadas como fraude eram de fato fraudulentas (evitando bloqueios indevidos a clientes legítimos).
* **F1-Score:** Média harmônica entre Precisão e Recall, garantindo um equilíbrio razoável entre ambas.

---

## 🛠️ 2. Preparação dos Dados (Data Preparation)

Para estruturar a base sem vazar informações do conjunto de teste para o de treino (*data leakage*), o seguinte fluxo foi aplicado:

1. **Feature Engineering (`Log_Amount`):** A variável `Amount` possui grande dispersão e assimetria. Foi aplicado a transformação logarítmica ($\ln(x+1)$) via `np.log1p` para reduzir o impacto de valores extremos.
2. **Seleção de Atributos:** As colunas `Time` e a variável `Amount` original foram removidas, mantendo `Log_Amount` e as variáveis PCA (`V1` a `V28`).
3. **Divisão Estratificada (Train/Test Split):** A divisão foi feita na proporção **80/20** utilizando o parâmetro `stratify=y`. Isso garantiu que tanto o conjunto de treino quanto o de teste mantivessem exatamente **0.17%** de casos de fraude.
4. **Padronização:** Aplicação do `StandardScaler` ajustado apenas nos dados de treino para evitar vazamento de dados.

---

## 📊 3. Comparação de Modelos e Métricas

Foram testados e comparados diferentes algoritmos de classificação. Os resultados abaixo refletem o desempenho focado exclusivamente na **classe de fraude (`Class 1`)**:

| Modelo | Limiar (Threshold) | Recall (Fraude) | Precisão (Fraude) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Regressão Logística** (Baseline) | $0.50$ | ~92% | ~6% | ~0.11 |
| **Regressão Logística** (Otimizado) | Limiar Ajustado | ~88% | ~75% | ~0.81 |
| **Random Forest** | Limiar Ajustado | ~82% | ~88% | ~0.85 |
| **XGBoost** | Padrão | ~80% | ~89% | ~0.84 |

> **Nota:** A escolha do modelo final depende do custo financeiro de um Falso Negativo (deixar passar uma fraude) versus um Falso Positivo (bloquear um cliente legítimo).

---

## 🎯 4. Ajuste de Limiar e Explicabilidade com SHAP

### Limiar de Decisão Escolhido (Threshold Tuning)
Por padrão, os classificadores usam um limiar de probabilidade de $0.5$. Em dados desbalanceados, esse limiar costuma gerar muitos falsos alarmes ou perder fraudes. Através da **Curva Precision-Recall**, ajustamos o limiar de decisão para maximizar a captura de fraudes mantendo um nível razoável de precisão.

### Explicabilidade com SHAP (SHapley Additive exPlanations)
Utilizando o `shap.TreeExplainer` no modelo XGBoost, identificamos o impacto de cada variável nas decisões de fraude:

- **Principais Variáveis Identificadas:** Variáveis como `V14`, `V10`, `V12` e `V4` apresentaram o maior impacto na predição.
- **Comportamento das Variáveis:** Valore extremamente baixos em `V14` e `V12` aumentam significativamente a probabilidade de a transação ser classificada como fraude, enquanto valores altos em `V4` elevam o risco financeiro.

---

## 🚀 5. Mudanças e Evoluções Implementadas

Além das etapas solicitadas no desafio base, as seguintes melhorias foram incorporadas ao pipeline:

- [x] **Testes de Reamostragem:** Comparação do comportamento dos modelos utilizando técnicas de **Undersampling** e **Oversampling (SMOTE)** para equilibrar a base de treino.
- [x] **Random Forest no Pipeline:** Treinamento do Random Forest com ajuste de pesos de classe (`class_weight='balanced'`) e otimização de limiar.
- [x] **Pipeline de Avaliação com XGBoost:** Implementação do XGBoost ajustado com `scale_pos_weight` para alimentar a análise explicável do SHAP.
- [x] **Visualizações Gráficas Salvas:** Inclusão direta das curvas **ROC-AUC** e **Precision-Recall** executadas e salvas no próprio notebook para evidência de funcionamento no repositório.

---

## 🛠️ Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)

---

📎 Link do curso: [DIO.me](https://web.dio.me/home) 

---
## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

<img alt="Static Badge" src="https://img.shields.io/badge/license-MIT-green">

   
