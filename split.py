import os
import random

def criar_arquivo_split(pasta_arquivos, arquivo_saida="split.txt", proporcao_val=0.2):
    """
    Cria um arquivo de split (train/val) para arquivos em uma pasta.

    Args:
        pasta_arquivos (str): Caminho para a pasta contendo os arquivos.
        arquivo_saida (str, opcional): Nome do arquivo txt de saída. Padrão é "split.txt".
        proporcao_val (float, opcional): Proporção de arquivos para o conjunto de validação (0.0 a 1.0). Padrão é 0.2 (20%).
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

    random.shuffle(arquivos)  # Embaralha os arquivos para garantir uma divisão aleatória

    num_arquivos_val = int(len(arquivos) * proporcao_val)
    arquivos_val = arquivos[:num_arquivos_val]
    arquivos_train = arquivos[num_arquivos_val:]

    with open(arquivo_saida, 'w') as arquivo_txt:
        for nome_arquivo in arquivos_train:
            arquivo_txt.write(f"{nome_arquivo}\ttrain\n")
        for nome_arquivo in arquivos_val:
            arquivo_txt.write(f"{nome_arquivo}\tval\n")

    print(f"Arquivo de split '{arquivo_saida}' criado com sucesso.")
    print(f"Conjunto de treino: {len(arquivos_train)} arquivos")
    print(f"Conjunto de validação: {len(arquivos_val)} arquivos")


if __name__ == "__main__":
    pasta = '/home/julio.hsu/beat_this/beat_this/data/annotations/gtzan/annotations/beats'
    arquivo_saida =  "/home/julio.hsu/beat_this/beat_this/data/annotations/gtzan/single.split"
    proporcao_val = 0.2

    criar_arquivo_split(pasta, arquivo_saida, proporcao_val)