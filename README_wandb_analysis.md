# Análise de Métricas do WandB - Beat This

Este conjunto de scripts extrai e analisa métricas de teste do projeto "beatthis" no Weights & Biases (wandb).

## 📋 Pré-requisitos

1. **Conta no WandB**: Você precisa ter acesso ao projeto `juliohsu/beatthis`
2. **Python 3.7+**: Com as dependências básicas instaladas
3. **Login no WandB**: Execute `wandb login` e cole sua API key

## 🚀 Uso Rápido

### Opção 1: Script Completo (Recomendado)
```bash
cd beat_this
python run_complete_analysis.py
```

### Opção 2: Scripts Individuais
```bash
# 1. Extrair métricas do wandb
python extract_wandb_metrics.py

# 2. Calcular médias dos folds
python calculate_fold_averages.py
```

## 📁 Estrutura dos Resultados

Após a execução, você terá:

```
beat_this/
├── wandb_metrics/           # Métricas individuais por fold
│   ├── vqt_tr3/
│   │   ├── fold_0_metrics.txt
│   │   ├── fold_0_metrics.json
│   │   ├── fold_1_metrics.txt
│   │   ├── fold_1_metrics.json
│   │   └── ... (até fold_7)
│   ├── cqt_tr1/
│   │   └── ... (mesma estrutura)
│   └── mel_tr4/
│       └── ... (mesma estrutura)
└── fold_averages/           # Médias e estatísticas
    ├── vqt_tr3_averages.txt
    ├── vqt_tr3_averages.json
    ├── cqt_tr1_averages.txt
    ├── cqt_tr1_averages.json
    ├── mel_tr4_averages.txt
    ├── mel_tr4_averages.json
    ├── comparison_summary.txt  # 📊 ARQUIVO PRINCIPAL
    └── comparison_summary.json
```

## 📊 Arquivos Principais

### `comparison_summary.txt`
Contém uma tabela comparativa entre os três tipos de modelo (vqt_tr3, cqt_tr1, mel_tr4) com:
- Médias de cada métrica
- Desvios padrão
- Valores mínimos e máximos
- Ranking do melhor modelo por métrica

### Arquivos `*_averages.txt`
Para cada tipo de modelo, contém:
- Estatísticas detalhadas de cada métrica
- Lista dos folds válidos processados
- Valores individuais de cada fold

## 🔧 Configuração

### Tipos de Run Processados
- `vqt_tr3`: Variable Q-Transform com 3 transformações
- `cqt_tr1`: Constant Q-Transform com 1 transformação  
- `mel_tr4`: Mel-spectrogram com 4 transformações

### Padrão de Nomes das Runs
```
{tipo}Fold{id}_frankothers
```
Exemplos:
- `vqt_tr3Fold0_frankothers`
- `cqt_tr1Fold3_frankothers`
- `mel_tr4Fold7_frankothers`

### Métricas Extraídas
Todas as métricas que começam com `test_` são automaticamente extraídas.

## 🛠️ Scripts Individuais

### `extract_wandb_metrics.py`
- Conecta ao wandb projeto `juliohsu/beatthis`
- Busca runs com padrão específico
- Extrai métricas de teste
- Salva em formato TXT e JSON

### `calculate_fold_averages.py`
- Lê os arquivos JSON gerados pelo script anterior
- Calcula médias, desvios padrão, min/max
- Gera resumo comparativo entre modelos
- Identifica melhor modelo por métrica

### `run_complete_analysis.py`
- Executa ambos os scripts em sequência
- Verifica dependências automaticamente
- Instala wandb se necessário
- Valida login do wandb

## ⚠️ Solução de Problemas

### "Run não encontrada"
- Verifique se o nome da run está correto no wandb
- Confirme se você tem acesso ao projeto `juliohsu/beatthis`

### "Você não está logado no wandb"
```bash
wandb login
# Cole sua API key quando solicitado
```

### "Nenhuma métrica de teste encontrada"
- Verifique se as runs têm métricas que começam com `test_`
- Confirme se as runs foram executadas completamente

## 📈 Interpretação dos Resultados

### Métricas Típicas
As métricas extraídas podem incluir:
- `test_accuracy`: Acurácia no conjunto de teste
- `test_f1`: F1-score no conjunto de teste
- `test_precision`: Precisão no conjunto de teste
- `test_recall`: Recall no conjunto de teste

### Comparação de Modelos
No arquivo `comparison_summary.txt`, procure por:
- **Maiores médias**: Geralmente indicam melhor performance
- **Menores desvios**: Indicam maior consistência
- **Seção "MELHOR MODELO POR MÉTRICA"**: Ranking automático

## 🔄 Atualizações

Para processar novas runs ou métricas atualizadas:
1. Delete as pastas `wandb_metrics` e `fold_averages`
2. Execute novamente o script principal

## 📞 Suporte

Se encontrar problemas:
1. Verifique se todas as runs existem no wandb
2. Confirme se você tem as permissões necessárias
3. Verifique se o wandb está atualizado: `pip install --upgrade wandb` 