#!/usr/bin/env python3
"""
Script para calcular a média das métricas de teste entre os folds
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import statistics

def calculate_fold_averages():
    """
    Calcula a média das métricas entre os folds para cada tipo de run
    """
    # Configurações
    run_types = ["vqt_tr3", "cqt_tr1", "mel_tr4"]
    num_folds = 8
    
    # Pasta onde estão os resultados
    metrics_dir = Path("wandb_metrics")
    
    if not metrics_dir.exists():
        print("❌ Pasta wandb_metrics não encontrada!")
        print("Execute primeiro o script extract_wandb_metrics.py")
        return
    
    # Criar pasta para os resultados das médias
    averages_dir = Path("fold_averages")
    averages_dir.mkdir(exist_ok=True)
    
    print("📊 Calculando médias dos folds...")
    
    all_results = {}
    
    for run_type in run_types:
        print(f"\nProcessando tipo: {run_type}")
        
        type_dir = metrics_dir / run_type
        if not type_dir.exists():
            print(f"  ⚠️  Pasta não encontrada: {type_dir}")
            continue
        
        # Coletar métricas de todos os folds
        fold_metrics = {}
        valid_folds = []
        
        for fold_id in range(num_folds):
            json_file = type_dir / f"fold_{fold_id}_metrics.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metrics = data.get('metrics', {})
                    if metrics:
                        fold_metrics[fold_id] = metrics
                        valid_folds.append(fold_id)
                        print(f"  ✅ Fold {fold_id}: {len(metrics)} métricas")
                    else:
                        print(f"  ⚠️  Fold {fold_id}: sem métricas")
                        
                except Exception as e:
                    print(f"  ❌ Erro ao ler fold {fold_id}: {str(e)}")
            else:
                print(f"  ⚠️  Arquivo não encontrado: {json_file}")
        
        if not fold_metrics:
            print(f"  ❌ Nenhuma métrica encontrada para {run_type}")
            continue
        
        # Calcular médias e desvios padrão
        print(f"  📈 Calculando estatísticas para {len(valid_folds)} folds válidos...")
        
        # Coletar todas as métricas únicas
        all_metric_names = set()
        for metrics in fold_metrics.values():
            all_metric_names.update(metrics.keys())
        
        # Calcular estatísticas para cada métrica
        metric_stats = {}
        for metric_name in sorted(all_metric_names):
            values = []
            for fold_id in valid_folds:
                if metric_name in fold_metrics[fold_id]:
                    value = fold_metrics[fold_id][metric_name]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
            
            if values:
                metric_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'values': values
                }
        
        # Salvar resultados em arquivo texto
        txt_file = averages_dir / f"{run_type}_averages.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"MÉDIAS DAS MÉTRICAS - {run_type.upper()}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Folds válidos: {valid_folds}\n")
            f.write(f"Total de folds: {len(valid_folds)}/{num_folds}\n\n")
            
            f.write("ESTATÍSTICAS DAS MÉTRICAS:\n")
            f.write("-" * 40 + "\n\n")
            
            for metric_name, stats in sorted(metric_stats.items()):
                f.write(f"{metric_name}:\n")
                f.write(f"  Média:   {stats['mean']:.6f}\n")
                f.write(f"  Desvio:  {stats['std']:.6f}\n")
                f.write(f"  Mínimo:  {stats['min']:.6f}\n")
                f.write(f"  Máximo:  {stats['max']:.6f}\n")
                f.write(f"  Amostras: {stats['count']}\n")
                f.write(f"  Valores: {[f'{v:.6f}' for v in stats['values']]}\n")
                f.write("\n")
        
        # Salvar resultados em JSON
        json_file = averages_dir / f"{run_type}_averages.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'run_type': run_type,
                'valid_folds': valid_folds,
                'total_folds': len(valid_folds),
                'metrics': metric_stats
            }, f, indent=2, ensure_ascii=False)
        
        all_results[run_type] = metric_stats
        
        print(f"  ✅ Resultados salvos:")
        print(f"     📄 {txt_file}")
        print(f"     📄 {json_file}")
        print(f"  📊 {len(metric_stats)} métricas processadas")
    
    # Criar resumo comparativo
    create_comparison_summary(all_results, averages_dir)
    
    print(f"\n🎉 Cálculo de médias concluído!")
    print(f"📁 Resultados salvos em: {averages_dir}")

def create_comparison_summary(all_results, output_dir):
    """
    Cria um resumo comparativo entre todos os tipos de run
    """
    if not all_results:
        return
    
    # Encontrar métricas comuns
    common_metrics = None
    for run_type, metrics in all_results.items():
        metric_names = set(metrics.keys())
        if common_metrics is None:
            common_metrics = metric_names
        else:
            common_metrics = common_metrics.intersection(metric_names)
    
    if not common_metrics:
        print("  ⚠️  Nenhuma métrica comum encontrada entre os tipos de run")
        return
    
    # Criar tabela comparativa
    comparison_file = output_dir / "comparison_summary.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("RESUMO COMPARATIVO ENTRE TIPOS DE RUN\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MÉTRICAS COMUNS:\n")
        f.write("-" * 20 + "\n")
        for metric in sorted(common_metrics):
            f.write(f"\n{metric}:\n")
            f.write("  Tipo        | Média      | Desvio     | Min        | Max\n")
            f.write("  " + "-" * 55 + "\n")
            
            for run_type in ["vqt_tr3", "cqt_tr1", "mel_tr4"]:
                if run_type in all_results and metric in all_results[run_type]:
                    stats = all_results[run_type][metric]
                    f.write(f"  {run_type:<10} | {stats['mean']:>10.6f} | {stats['std']:>10.6f} | {stats['min']:>10.6f} | {stats['max']:>10.6f}\n")
            f.write("\n")
        
        # Encontrar melhor modelo para cada métrica
        f.write("\nMELHOR MODELO POR MÉTRICA:\n")
        f.write("-" * 30 + "\n")
        for metric in sorted(common_metrics):
            best_run = None
            best_value = None
            
            # Assumir que métricas maiores são melhores (ajustar se necessário)
            for run_type in all_results:
                if metric in all_results[run_type]:
                    value = all_results[run_type][metric]['mean']
                    if best_value is None or value > best_value:
                        best_value = value
                        best_run = run_type
            
            if best_run:
                f.write(f"{metric}: {best_run} ({best_value:.6f})\n")
    
    # Salvar comparação em JSON
    comparison_json = output_dir / "comparison_summary.json"
    with open(comparison_json, 'w', encoding='utf-8') as f:
        json.dump({
            'common_metrics': list(common_metrics),
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  📊 Resumo comparativo salvo:")
    print(f"     📄 {comparison_file}")
    print(f"     📄 {comparison_json}")

if __name__ == "__main__":
    try:
        calculate_fold_averages()
    except KeyboardInterrupt:
        print("\n⏹️  Processo interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro geral: {str(e)}")
        import traceback
        traceback.print_exc() 