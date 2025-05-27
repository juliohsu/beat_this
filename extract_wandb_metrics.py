#!/usr/bin/env python3
"""
Script para extrair métricas de teste do Weights & Biases (wandb)
Entity: juliohsu
Projeto: beatthis
"""

import wandb
import os
import json
from pathlib import Path

def extract_wandb_metrics():
    """
    Extrai métricas de teste do wandb para o projeto beatthis
    """
    # Configurações
    entity = "juliohsu"
    project = "beat_this"
    
    # Tipos de runs e número de folds
    run_types = ["vqt_tr3", "cqt_tr1", "mel_tr4"]
    num_folds = 8
    
    # Criar pasta para salvar os resultados
    output_dir = Path("wandb_metrics")
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar API do wandb
    api = wandb.Api()
    
    print(f"Conectando ao projeto {entity}/{project}...")
    
    for run_type in run_types:
        print(f"\nProcessando tipo de run: {run_type}")
        
        # Criar pasta para este tipo de run
        type_dir = output_dir / run_type
        type_dir.mkdir(exist_ok=True)
        
        for fold_id in range(num_folds):
            # Construir nome da run
            run_name = f"{run_type}Fold{fold_id}_frankothers"
            print(f"  Buscando run: {run_name}")
            
            try:
                # Buscar a run específica
                runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
                runs_list = list(runs)
                
                if not runs_list:
                    print(f"    ⚠️  Run não encontrada: {run_name}")
                    continue
                
                if len(runs_list) > 1:
                    print(f"    ⚠️  Múltiplas runs encontradas para {run_name}, usando a primeira")
                
                run = runs_list[0]
                
                # Extrair métricas que começam com "test_"
                test_metrics = {}
                for key, value in run.summary.items():
                    if key.startswith("test_"):
                        test_metrics[key] = value
                
                if not test_metrics:
                    print(f"    ⚠️  Nenhuma métrica de teste encontrada para {run_name}")
                    continue
                
                # Salvar métricas em arquivo txt
                output_file = type_dir / f"fold_{fold_id}_metrics.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Run: {run_name}\n")
                    f.write(f"Run ID: {run.id}\n")
                    f.write(f"URL: {run.url}\n")
                    f.write("=" * 50 + "\n")
                    f.write("TEST METRICS:\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for metric_name, metric_value in sorted(test_metrics.items()):
                        f.write(f"{metric_name}: {metric_value}\n")
                
                print(f"    ✅ Métricas salvas em: {output_file}")
                print(f"    📊 {len(test_metrics)} métricas encontradas")
                
                # Também salvar em formato JSON para facilitar processamento posterior
                json_file = type_dir / f"fold_{fold_id}_metrics.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "run_name": run_name,
                        "run_id": run.id,
                        "run_url": run.url,
                        "metrics": test_metrics
                    }, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"    ❌ Erro ao processar {run_name}: {str(e)}")
                continue
    
    print(f"\n🎉 Processamento concluído! Resultados salvos em: {output_dir}")
    
    # Criar um resumo
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RESUMO DA EXTRAÇÃO DE MÉTRICAS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Entity: {entity}\n")
        f.write(f"Projeto: {project}\n")
        f.write(f"Tipos de run: {', '.join(run_types)}\n")
        f.write(f"Número de folds: {num_folds}\n\n")
        
        f.write("ESTRUTURA DE ARQUIVOS:\n")
        f.write("-" * 20 + "\n")
        for run_type in run_types:
            f.write(f"{run_type}/\n")
            for fold_id in range(num_folds):
                f.write(f"  ├── fold_{fold_id}_metrics.txt\n")
                f.write(f"  └── fold_{fold_id}_metrics.json\n")
    
    print(f"📋 Resumo salvo em: {summary_file}")

if __name__ == "__main__":
    try:
        extract_wandb_metrics()
    except KeyboardInterrupt:
        print("\n⏹️  Processo interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro geral: {str(e)}")
        print("Verifique se você está logado no wandb (wandb login)") 