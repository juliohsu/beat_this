import os
import random

def criar_arquivo_split(pasta_arquivos, arquivo_saida="split.txt", proporcao_val=0.2, seed=None, fold_index=0, total_folds=1):
    """
    Cria um arquivo de split (train/val) para arquivos em uma pasta.

    Args:
        pasta_arquivos (str): Caminho para a pasta contendo os arquivos.
        arquivo_saida (str, opcional): Nome do arquivo txt de saída. Padrão é "split.txt".
        proporcao_val (float, opcional): Proporção de arquivos para o conjunto de validação (0.0 a 1.0). Padrão é 0.2 (20%).
        seed (int, opcional): Semente para o gerador de números aleatórios. Se None, não usa semente fixa.
        fold_index (int, opcional): Índice do fold atual (0 a total_folds-1). Padrão é 0.
        total_folds (int, opcional): Número total de folds. Padrão é 1.
    """

    arquivos = []
    try:
        arquivos = [f.split('.')[0] for f in os.listdir(pasta_arquivos) if os.path.isfile(os.path.join(pasta_arquivos, f))]
    except FileNotFoundError:
        print(f"Erro: Pasta '{pasta_arquivos}' não encontrada.")
        return

    if not arquivos:
        print(f"Aviso: Pasta '{pasta_arquivos}' está vazia.")
        return

    if seed is not None:
        random.seed(seed)
    
    random.shuffle(arquivos)  # Embaralha os arquivos para garantir uma divisão aleatória

    # Calcula o tamanho de cada fold
    fold_size = len(arquivos) // total_folds
    
    # Determina o intervalo do fold atual para validação
    start_idx = fold_index * fold_size
    end_idx = start_idx + fold_size if fold_index < total_folds - 1 else len(arquivos)
    
    # Seleciona os arquivos de validação do fold atual
    arquivos_val = arquivos[start_idx:end_idx]
    
    # Os arquivos de treino são todos os outros
    arquivos_train = [f for f in arquivos if f not in arquivos_val]

    # Ajusta o nome do arquivo de saída para incluir o número do fold
    if total_folds > 1:
        nome_base, extensao = os.path.splitext(arquivo_saida)
        arquivo_saida = f"{nome_base}_fold{fold_index}{extensao}"

    with open(arquivo_saida, 'w') as arquivo_txt:
        for nome_arquivo in arquivos_train:
            arquivo_txt.write(f"{nome_arquivo}\ttrain\n")
        for nome_arquivo in arquivos_val:
            arquivo_txt.write(f"{nome_arquivo}\tval\n")

    print(f"Arquivo de split '{arquivo_saida}' criado com sucesso.")
    print(f"Conjunto de treino: {len(arquivos_train)} arquivos")
    print(f"Conjunto de validação: {len(arquivos_val)} arquivos")


def criar_arquivo_8folds(pasta_arquivos, arquivo_saida="8-folds.split", seed=0, total_folds=8):
    """
    Cria um arquivo de split 8-folds no formato esperado pelo dataset.py.
    
    Args:
        pasta_arquivos (str): Caminho para a pasta contendo os arquivos.
        arquivo_saida (str, opcional): Nome do arquivo txt de saída. Padrão é "8-folds.split".
        seed (int, opcional): Semente para o gerador de números aleatórios. Padrão é 0.
        total_folds (int, opcional): Número total de folds. Padrão é 8.
    """
    # Verifica se o arquivo já existe
    if os.path.exists(arquivo_saida):
        print(f"Arquivo '{arquivo_saida}' já existe. Pulando criação.")
        return True
    
    arquivos = []
    try:
        arquivos = [f.split('.')[0] for f in os.listdir(pasta_arquivos) if os.path.isfile(os.path.join(pasta_arquivos, f))]
    except FileNotFoundError:
        print(f"Erro: Pasta '{pasta_arquivos}' não encontrada.")
        return False

    if not arquivos:
        print(f"Aviso: Pasta '{pasta_arquivos}' está vazia.")
        return False

    # Definir semente para reprodutibilidade
    random.seed(seed)
    
    # Embaralha os arquivos para garantir uma divisão aleatória
    random.shuffle(arquivos)
    
    # Distribuir os arquivos em 8 folds
    fold_size = len(arquivos) // total_folds
    assigned_folds = {}
    
    for fold in range(total_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < total_folds - 1 else len(arquivos)
        
        for i in range(start_idx, end_idx):
            assigned_folds[arquivos[i]] = fold
    
    # Criar o arquivo de saída no formato piece\tpart
    with open(arquivo_saida, 'w') as arquivo_txt:
        for arquivo, fold in assigned_folds.items():
            arquivo_txt.write(f"{arquivo}\t{fold}\n")
    
    print(f"Arquivo de split '{arquivo_saida}' criado com sucesso no formato 8-folds.")
    print(f"Total de arquivos: {len(arquivos)}")
    
    # Imprimir distribuição de arquivos por fold
    fold_counts = {i: 0 for i in range(total_folds)}
    for fold_num in assigned_folds.values():
        fold_counts[fold_num] += 1
    
    for fold, count in fold_counts.items():
        print(f"Fold {fold}: {count} arquivos")
    
    return True


def criar_arquivo_single_split(pasta_arquivos, arquivo_saida="single.split", proporcao_val=0.2, seed=0):
    """
    Cria um arquivo de split (train/val) para arquivos em uma pasta no formato single.split.
    
    Args:
        pasta_arquivos (str): Caminho para a pasta contendo os arquivos.
        arquivo_saida (str, opcional): Nome do arquivo txt de saída. Padrão é "single.split".
        proporcao_val (float, opcional): Proporção de arquivos para o conjunto de validação (0.0 a 1.0). Padrão é 0.2 (20%).
        seed (int, opcional): Semente para o gerador de números aleatórios. Padrão é 0.
    """
    # Verifica se o arquivo já existe
    if os.path.exists(arquivo_saida):
        print(f"Arquivo '{arquivo_saida}' já existe. Pulando criação.")
        return True
    
    arquivos = []
    try:
        arquivos = [f.split('.')[0] for f in os.listdir(pasta_arquivos) if os.path.isfile(os.path.join(pasta_arquivos, f))]
    except FileNotFoundError:
        print(f"Erro: Pasta '{pasta_arquivos}' não encontrada.")
        return False

    if not arquivos:
        print(f"Aviso: Pasta '{pasta_arquivos}' está vazia.")
        return False

    # Definir semente para reprodutibilidade
    random.seed(seed)
    
    # Embaralha os arquivos para garantir uma divisão aleatória
    random.shuffle(arquivos)
    
    # Dividir em treino e validação
    num_arquivos_val = int(len(arquivos) * proporcao_val)
    arquivos_val = arquivos[:num_arquivos_val]
    arquivos_train = arquivos[num_arquivos_val:]
    
    # Criar o arquivo de saída no formato piece\tpart (train/val)
    with open(arquivo_saida, 'w') as arquivo_txt:
        for nome_arquivo in arquivos_train:
            arquivo_txt.write(f"{nome_arquivo}\ttrain\n")
        for nome_arquivo in arquivos_val:
            arquivo_txt.write(f"{nome_arquivo}\tval\n")
    
    print(f"Arquivo de split '{arquivo_saida}' criado com sucesso (single.split).")
    print(f"Conjunto de treino: {len(arquivos_train)} arquivos")
    print(f"Conjunto de validação: {len(arquivos_val)} arquivos")
    
    return True


def processar_pasta_anotacoes(pasta_principal, arquivo_saida_base, total_folds=8, seed=0):
    """
    Processa uma pasta de anotações e gera múltiplos splits.
    
    Args:
        pasta_principal (str): Caminho para a pasta principal contendo as anotações.
        arquivo_saida_base (str): Caminho base para os arquivos de saída.
        total_folds (int, opcional): Número total de folds a gerar. Padrão é 8.
        seed (int, opcional): Semente para o gerador de números aleatórios. Padrão é 0.
    """
    print(f"Processando pasta: {pasta_principal}")
    print(f"Gerando {total_folds} splits com seed {seed}")
    
    for fold in range(total_folds):
        criar_arquivo_split(
            pasta_arquivos=pasta_principal,
            arquivo_saida=arquivo_saida_base,
            seed=seed,
            fold_index=fold,
            total_folds=total_folds
        )
    
    print(f"Todos os {total_folds} splits foram gerados com sucesso.")


def processar_datasets_anotacoes(pasta_base_anotacoes):
    """
    Processa todos os datasets de anotações e cria arquivos 8-folds.split e single.split.
    
    Args:
        pasta_base_anotacoes (str): Caminho base para a pasta contendo os datasets de anotações.
    """
    # Verifica se a pasta base existe
    if not os.path.exists(pasta_base_anotacoes):
        print(f"Erro: Pasta base '{pasta_base_anotacoes}' não encontrada.")
        return
    
    # Lista todos os diretórios na pasta base (cada um é um dataset)
    datasets = [d for d in os.listdir(pasta_base_anotacoes) 
               if os.path.isdir(os.path.join(pasta_base_anotacoes, d))]
    
    if not datasets:
        print(f"Aviso: Pasta base '{pasta_base_anotacoes}' não contém datasets.")
        return
    
    print(f"Encontrados {len(datasets)} datasets: {', '.join(datasets)}")
    
    # Para cada dataset, processa a pasta de anotações de beats
    for dataset in datasets:
        pasta_dataset = os.path.join(pasta_base_anotacoes, dataset)
        pasta_beats = os.path.join(pasta_dataset, "annotations", "beats")
        
        if not os.path.exists(pasta_beats):
            print(f"Aviso: Dataset '{dataset}' não possui pasta de anotações de beats.")
            continue
        
        arquivo_8folds = os.path.join(pasta_dataset, "8-folds.split")
        arquivo_single = os.path.join(pasta_dataset, "single.split")
        
        print(f"\nProcessando dataset: {dataset}")
        print("Criando arquivo 8-folds.split...")
        if criar_arquivo_8folds(pasta_beats, arquivo_8folds):
            print(f"Arquivo 8-folds.split para dataset '{dataset}' criado com sucesso.")
        
        print("\nCriando arquivo single.split...")
        if criar_arquivo_single_split(pasta_beats, arquivo_single):
            print(f"Arquivo single.split para dataset '{dataset}' criado com sucesso.")


if __name__ == "__main__":
    # Pasta base com todos os datasets de anotações
    pasta_base = '/home/julio.hsu/beat_this/beat_this/data/annotations'
    
    # Processa todos os datasets
    processar_datasets_anotacoes(pasta_base)
    
    # Como exemplo alternativo, para processar apenas um dataset específico:
    # pasta_beats = '/home/julio.hsu/beat_this/beat_this/data/annotations/groovemidi/annotations/beats'
    # arquivo_saida = "/home/julio.hsu/beat_this/beat_this/data/annotations/groovemidi/8-folds.split"
    # criar_arquivo_8folds(pasta_beats, arquivo_saida)