#!/usr/bin/env python3
"""
Script principal para executar análise completa das métricas do WandB
1. Extrai métricas de teste do wandb
2. Calcula médias e estatísticas dos folds
"""

import subprocess
import sys
from pathlib import Path

def check_wandb_installation():
    """Verifica se wandb está instalado"""
    try:
        import wandb
        return True
    except ImportError:
        return False

def install_wandb():
    """Instala wandb"""
    print("📦 Instalando wandb...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    print("✅ wandb instalado com sucesso!")

def check_wandb_login():
    """Verifica se usuário está logado no wandb"""
    try:
        result = subprocess.run(["wandb", "whoami"], capture_output=True, text=True)
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Usuário logado no wandb: {username}")
            return True
        else:
            return False
    except:
        return False

def main():
    """Função principal"""
    print("🚀 Iniciando análise completa das métricas do WandB")
    print("=" * 60)
    
    # Verificar instalação do wandb
    if not check_wandb_installation():
        install_wandb()
    
    # Verificar login
    if not check_wandb_login():
        print("❌ Você não está logado no wandb")
        print("Por favor, execute: wandb login")
        print("E cole sua API key do wandb")
        return 1
    
    # Executar extração de métricas
    print("\n🔄 ETAPA 1: Extraindo métricas do wandb...")
    print("-" * 40)
    
    try:
        from extract_wandb_metrics import extract_wandb_metrics
        extract_wandb_metrics()
        print("✅ Extração de métricas concluída!")
    except Exception as e:
        print(f"❌ Erro na extração de métricas: {str(e)}")
        return 1
    
    # Verificar se há dados para processar
    metrics_dir = Path("wandb_metrics")
    if not metrics_dir.exists() or not any(metrics_dir.iterdir()):
        print("❌ Nenhuma métrica foi extraída. Verifique se as runs existem no wandb.")
        return 1
    
    # Executar cálculo de médias
    print("\n🔄 ETAPA 2: Calculando médias dos folds...")
    print("-" * 40)
    
    try:
        from calculate_fold_averages import calculate_fold_averages
        calculate_fold_averages()
        print("✅ Cálculo de médias concluído!")
    except Exception as e:
        print(f"❌ Erro no cálculo de médias: {str(e)}")
        return 1
    
    # Resumo final
    print("\n🎉 ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 60)
    print("📁 Resultados disponíveis em:")
    print("   📂 wandb_metrics/     - Métricas individuais por fold")
    print("   📂 fold_averages/     - Médias e estatísticas")
    print("\n📊 Arquivos principais:")
    print("   📄 fold_averages/comparison_summary.txt - Comparação entre modelos")
    
    # Listar arquivos gerados
    averages_dir = Path("fold_averages")
    if averages_dir.exists():
        print("\n📋 Arquivos de médias gerados:")
        for file in sorted(averages_dir.glob("*.txt")):
            print(f"   📄 {file}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 