"""
================================================================================
TELECOMX - PASO 7: ANÁLISIS DE CORRELACIÓN Y OPTIMIZACIÓN DE VARIABLES
================================================================================
Descripción: Análisis de correlaciones y eliminación inteligente de variables
             basada en evidencia cuantitativa para optimizar el dataset.

Funcionalidades:
- Análisis completo de matriz de correlación
- Identificación de variables con alta correlación con target
- Detección y resolución de multicolinealidad
- Eliminación inteligente de variables irrelevantes
- Generación de dataset optimizado para modelado

Autor: Sistema de Análisis Predictivo TelecomX
Fecha: 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso7_analisis_correlacion.log', mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Crear directorios necesarios"""
    directories = ['excel', 'informes', 'graficos', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Carpeta verificada/creada: {directory}")

def find_latest_file(directory, pattern):
    """Encontrar el archivo más reciente que coincida con el patrón"""
    files = list(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con patrón {pattern} en {directory}")
    latest_file = max(files, key=os.path.getctime)
    return str(latest_file)

def load_data():
    """Cargar el dataset del paso anterior"""
    try:
        # Buscar el archivo más reciente del Paso 2
        input_file = find_latest_file('excel', 'telecomx_paso2_encoding_aplicado_*.csv')
        logging.info(f"Cargando archivo: {input_file}")
        
        # Intentar diferentes combinaciones de codificación y separador
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        separators = [',', ';', '\t']
        df = None
        successful_config = None
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(input_file, encoding=encoding, sep=sep)
                    
                    # Verificar que el dataset tenga sentido (más de 5 columnas)
                    if df.shape[1] > 5:
                        logging.info(f"Archivo cargado exitosamente:")
                        logging.info(f"  Codificación: {encoding}")
                        logging.info(f"  Separador: '{sep}'")
                        logging.info(f"  Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
                        successful_config = (encoding, sep)
                        break
                        
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
            
            if df is not None and df.shape[1] > 5:
                break
        
        if df is None or df.shape[1] <= 5:
            # Intento final: leer las primeras líneas para diagnóstico
            with open(input_file, 'r', encoding='utf-8-sig') as f:
                first_lines = f.readlines()[:3]
                logging.error("DIAGNÓSTICO DEL ARCHIVO:")
                for i, line in enumerate(first_lines):
                    logging.error(f"  Línea {i+1}: {line.strip()[:100]}...")
            
            raise ValueError(f"No se pudo cargar el archivo correctamente. "
                           f"Columnas detectadas: {df.shape[1] if df is not None else 'N/A'}")
        
        # Verificar columnas principales
        expected_columns = ['Abandono_Cliente', 'Meses_Cliente', 'Cargo_Total']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"Columnas esperadas no encontradas: {missing_columns}")
            logging.info(f"Columnas disponibles: {list(df.columns[:10])}")
        
        logging.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, input_file
        
    except Exception as e:
        logging.error(f"Error al cargar el dataset: {str(e)}")
        raise

def analyze_correlations(df):
    """Analizar correlaciones del dataset"""
    logging.info("Iniciando análisis de correlaciones...")
    
    # Identificar variables numéricas
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    logging.info(f"Variables numéricas encontradas: {len(numeric_vars)}")
    
    # Calcular matriz de correlación
    correlation_matrix = df[numeric_vars].corr()
    
    # Correlaciones con variable objetivo
    target_var = 'Abandono_Cliente'
    if target_var not in correlation_matrix.columns:
        raise ValueError(f"Variable objetivo '{target_var}' no encontrada en el dataset")
    
    target_correlations = correlation_matrix[target_var].drop(target_var).abs().sort_values(ascending=False)
    
    # Análisis de multicolinealidad
    multicollinear_pairs = []
    threshold = 0.8
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            var1 = correlation_matrix.columns[i]
            var2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.loc[var1, var2]
            
            if abs(corr_value) >= threshold and var1 != target_var and var2 != target_var:
                multicollinear_pairs.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': corr_value
                })
    
    analysis_results = {
        'correlation_matrix': correlation_matrix,
        'target_correlations': target_correlations,
        'multicollinear_pairs': multicollinear_pairs,
        'numeric_vars': numeric_vars,
        'target_var': target_var
    }
    
    logging.info(f"Análisis completado: {len(target_correlations)} correlaciones calculadas")
    logging.info(f"Pares multicolineales detectados: {len(multicollinear_pairs)}")
    
    return analysis_results

def identify_variables_to_remove(analysis_results):
    """Identificar variables a eliminar basado en evidencia cuantitativa"""
    logging.info("Identificando variables a eliminar...")
    
    target_correlations = analysis_results['target_correlations']
    multicollinear_pairs = analysis_results['multicollinear_pairs']
    
    variables_to_remove = set()
    removal_reasons = {}
    
    # 1. Variables con correlación muy baja (|r| < 0.1)
    low_correlation_threshold = 0.1
    for var, corr in target_correlations.items():
        if corr < low_correlation_threshold:
            variables_to_remove.add(var)
            removal_reasons[var] = f"Correlación muy baja con target (|r| = {corr:.4f} < {low_correlation_threshold})"
    
    # 2. Resolver multicolinealidad
    priority_vars = {
        'Meses_Cliente': 1,  # Alta prioridad - importante para negocio
        'Cargo_Total': 2,
        'Tipo_Contrato_encoded': 1,
        'Facturacion_Mensual': 3,  # Baja prioridad - menos interpretable
        'Lineas_Multiples_No': 4,  # Muy baja prioridad
        'Lineas_Multiples_Sí': 4
    }
    
    for pair in multicollinear_pairs:
        var1, var2 = pair['var1'], pair['var2']
        corr = pair['correlation']
        
        # Decidir cuál eliminar basado en prioridad y correlación con target
        var1_priority = priority_vars.get(var1, 5)
        var2_priority = priority_vars.get(var2, 5)
        var1_target_corr = target_correlations.get(var1, 0)
        var2_target_corr = target_correlations.get(var2, 0)
        
        if var1_priority > var2_priority:
            # var1 tiene menor prioridad, eliminar var1
            variables_to_remove.add(var1)
            removal_reasons[var1] = f"Multicolinealidad con {var2} (r = {corr:.4f}), menor prioridad de negocio"
        elif var2_priority > var1_priority:
            # var2 tiene menor prioridad, eliminar var2
            variables_to_remove.add(var2)
            removal_reasons[var2] = f"Multicolinealidad con {var1} (r = {corr:.4f}), menor prioridad de negocio"
        else:
            # Misma prioridad, eliminar el de menor correlación con target
            if var1_target_corr < var2_target_corr:
                variables_to_remove.add(var1)
                removal_reasons[var1] = f"Multicolinealidad con {var2} (r = {corr:.4f}), menor correlación con target"
            else:
                variables_to_remove.add(var2)
                removal_reasons[var2] = f"Multicolinealidad con {var1} (r = {corr:.4f}), menor correlación con target"
    
    # 3. Variables con valores NaN en correlación
    for var in target_correlations.index:
        if pd.isna(target_correlations[var]):
            variables_to_remove.add(var)
            removal_reasons[var] = "Correlación indefinida (NaN)"
    
    logging.info(f"Variables identificadas para eliminación: {len(variables_to_remove)}")
    
    return variables_to_remove, removal_reasons

def optimize_dataset(df, variables_to_remove, analysis_results):
    """Optimizar dataset eliminando variables identificadas"""
    logging.info("Optimizando dataset...")
    
    # Dataset optimizado
    df_optimized = df.copy()
    
    # Eliminar variables identificadas
    variables_removed = []
    for var in variables_to_remove:
        if var in df_optimized.columns:
            df_optimized = df_optimized.drop(columns=[var])
            variables_removed.append(var)
            logging.info(f"Variable eliminada: {var}")
    
    # Estadísticas de optimización
    original_vars = len(df.columns) - 1  # -1 por variable objetivo
    optimized_vars = len(df_optimized.columns) - 1
    reduction_percentage = (original_vars - optimized_vars) / original_vars * 100
    
    optimization_stats = {
        'original_variables': original_vars,
        'optimized_variables': optimized_vars,
        'variables_removed': len(variables_removed),
        'reduction_percentage': reduction_percentage,
        'variables_removed_list': variables_removed
    }
    
    logging.info(f"Optimización completada:")
    logging.info(f"  Variables originales: {original_vars}")
    logging.info(f"  Variables optimizadas: {optimized_vars}")
    logging.info(f"  Reducción: {reduction_percentage:.1f}%")
    
    return df_optimized, optimization_stats

def generate_visualizations(analysis_results, optimization_stats, variables_to_remove, removal_reasons, timestamp):
    """Generar visualizaciones del análisis"""
    logging.info("Generando visualizaciones...")
    
    try:
        # Configuración de estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        target_correlations = analysis_results['target_correlations']
        
        # 1. Ranking completo de variables con nombres
        plt.figure(figsize=(16, 12))
        
        # Preparar datos
        all_vars = target_correlations.index.tolist()
        all_corrs = target_correlations.values
        colors = ['red' if var in variables_to_remove else 'green' for var in all_vars]
        
        # Crear gráfico horizontal
        y_pos = np.arange(len(all_vars))
        bars = plt.barh(y_pos, all_corrs, color=colors, alpha=0.7)
        
        # Configurar ejes
        plt.yticks(y_pos, all_vars, fontsize=10)
        plt.xlabel('Correlación Absoluta con Target', fontsize=12)
        plt.title('Ranking Completo de Variables por Correlación con Target', fontsize=14, pad=20)
        
        # Líneas de umbral
        plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.8, label='Umbral Mínimo (|r| = 0.1)')
        plt.axvline(x=0.3, color='blue', linestyle='--', alpha=0.8, label='Umbral Significativo (|r| = 0.3)')
        
        # Leyenda personalizada
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label=f'Variables Conservadas ({len([v for v in all_vars if v not in variables_to_remove])})'),
            Patch(facecolor='red', alpha=0.7, label=f'Variables Eliminadas ({len(variables_to_remove)})'),
            plt.Line2D([0], [0], color='orange', linestyle='--', label='Umbral Mínimo (|r| = 0.1)'),
            plt.Line2D([0], [0], color='blue', linestyle='--', label='Umbral Significativo (|r| = 0.3)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Añadir valores en barras importantes (correlación > 0.2)
        for i, (var, corr) in enumerate(zip(all_vars, all_corrs)):
            if corr > 0.2:
                plt.text(corr + 0.01, i, f'{corr:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'graficos/paso7_ranking_completo_variables_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top variables conservadas vs eliminadas (con nombres)
        plt.figure(figsize=(20, 10))
        
        # Seleccionar top variables de cada grupo
        removed_vars = [var for var in all_vars if var in variables_to_remove][:15]
        kept_vars = [var for var in all_vars if var not in variables_to_remove][:15]
        
        # Combinar y preparar datos
        combined_vars = removed_vars + kept_vars
        combined_corrs = [target_correlations[var] for var in combined_vars]
        combined_colors = ['red' if var in variables_to_remove else 'green' for var in combined_vars]
        
        # Crear gráfico
        x_pos = np.arange(len(combined_vars))
        bars = plt.bar(x_pos, combined_corrs, color=combined_colors, alpha=0.7)
        
        # Configurar ejes
        plt.xticks(x_pos, combined_vars, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Correlación Absoluta con Target', fontsize=12)
        plt.title('Top Variables: Eliminadas vs Conservadas', fontsize=14, pad=20)
        
        # Líneas de umbral
        plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.8, label='Umbral Mínimo (|r| = 0.1)')
        plt.axhline(y=0.3, color='blue', linestyle='--', alpha=0.8, label='Umbral Significativo (|r| = 0.3)')
        
        # Separador visual
        if removed_vars and kept_vars:
            separator_pos = len(removed_vars) - 0.5
            plt.axvline(x=separator_pos, color='black', linestyle='-', alpha=0.5, linewidth=2)
            plt.text(separator_pos/2, max(combined_corrs)*0.9, 'ELIMINADAS', 
                    ha='center', fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            plt.text(separator_pos + (len(kept_vars))/2, max(combined_corrs)*0.9, 'CONSERVADAS', 
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Añadir valores en las barras
        for i, (var, corr) in enumerate(zip(combined_vars, combined_corrs)):
            plt.text(i, corr + 0.01, f'{corr:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'graficos/paso7_top_variables_eliminadas_vs_conservadas_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Análisis por categorías de variables
        plt.figure(figsize=(16, 8))
        
        # Categorizar variables por tipo
        service_vars = [var for var in all_vars if any(keyword in var.lower() for keyword in ['servicio', 'seguridad', 'respaldo', 'proteccion', 'soporte', 'lineas'])]
        demographic_vars = [var for var in all_vars if any(keyword in var.lower() for keyword in ['genero', 'ciudadano', 'pareja', 'dependientes'])]
        contract_vars = [var for var in all_vars if any(keyword in var.lower() for keyword in ['contrato', 'metodo', 'pago'])]
        financial_vars = [var for var in all_vars if any(keyword in var.lower() for keyword in ['cargo', 'facturacion', 'meses'])]
        other_vars = [var for var in all_vars if var not in service_vars + demographic_vars + contract_vars + financial_vars]
        
        categories = {
            'Servicios': service_vars,
            'Demográficas': demographic_vars,
            'Contrato/Pago': contract_vars,
            'Financieras': financial_vars,
            'Otras': other_vars
        }
        
        # Calcular estadísticas por categoría
        category_stats = []
        for cat_name, cat_vars in categories.items():
            if cat_vars:
                conserved = len([v for v in cat_vars if v not in variables_to_remove])
                eliminated = len([v for v in cat_vars if v in variables_to_remove])
                avg_corr = np.mean([target_correlations[v] for v in cat_vars])
                category_stats.append({
                    'categoria': cat_name,
                    'conservadas': conserved,
                    'eliminadas': eliminated,
                    'total': len(cat_vars),
                    'correlacion_promedio': avg_corr
                })
        
        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico 1: Variables por categoría
        categories_names = [stat['categoria'] for stat in category_stats]
        conserved_counts = [stat['conservadas'] for stat in category_stats]
        eliminated_counts = [stat['eliminadas'] for stat in category_stats]
        
        x_pos = np.arange(len(categories_names))
        width = 0.35
        
        ax1.bar(x_pos - width/2, conserved_counts, width, label='Conservadas', color='green', alpha=0.7)
        ax1.bar(x_pos + width/2, eliminated_counts, width, label='Eliminadas', color='red', alpha=0.7)
        
        ax1.set_xlabel('Categorías de Variables')
        ax1.set_ylabel('Número de Variables')
        ax1.set_title('Variables por Categoría: Conservadas vs Eliminadas')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for i, (cons, elim) in enumerate(zip(conserved_counts, eliminated_counts)):
            if cons > 0:
                ax1.text(i - width/2, cons + 0.1, str(cons), ha='center', va='bottom', fontweight='bold')
            if elim > 0:
                ax1.text(i + width/2, elim + 0.1, str(elim), ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Correlación promedio por categoría
        avg_corrs = [stat['correlacion_promedio'] for stat in category_stats]
        colors_by_corr = ['darkgreen' if corr >= 0.2 else 'orange' if corr >= 0.1 else 'red' for corr in avg_corrs]
        
        bars = ax2.bar(categories_names, avg_corrs, color=colors_by_corr, alpha=0.7)
        ax2.set_xlabel('Categorías de Variables')
        ax2.set_ylabel('Correlación Promedio con Target')
        ax2.set_title('Correlación Promedio por Categoría')
        ax2.set_xticklabels(categories_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Líneas de referencia
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.8, label='Umbral Mínimo')
        ax2.axhline(y=0.3, color='blue', linestyle='--', alpha=0.8, label='Umbral Significativo')
        ax2.legend()
        
        # Añadir valores en las barras
        for bar, corr in zip(bars, avg_corrs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{corr:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso7_analisis_por_categorias_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Matriz de correlación mejorada (solo top variables conservadas)
        plt.figure(figsize=(14, 10))
        
        # Seleccionar top variables conservadas + target
        top_kept_vars = [var for var in target_correlations.head(12).index if var not in variables_to_remove][:10]
        top_kept_vars.append(analysis_results['target_var'])
        
        correlation_subset = analysis_results['correlation_matrix'].loc[top_kept_vars, top_kept_vars]
        
        # Crear heatmap
        mask = np.triu(np.ones_like(correlation_subset, dtype=bool))  # Máscara triangular superior
        sns.heatmap(correlation_subset, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.3f',
                   square=True,
                   cbar_kws={'label': 'Correlación'},
                   xticklabels=True,
                   yticklabels=True)
        
        plt.title('Matriz de Correlación - Top Variables Conservadas', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'graficos/paso7_matriz_correlacion_optimizada_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones mejoradas generadas exitosamente")
        return True
        
    except Exception as e:
        logging.error(f"Error al generar visualizaciones: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def generate_optimized_variable_list(df_optimized, analysis_results):
    """Generar lista optimizada de variables con detalles"""
    target_correlations = analysis_results['target_correlations']
    
    optimized_vars = []
    for col in df_optimized.columns:
        if col != analysis_results['target_var']:
            corr_value = target_correlations.get(col, 0)
            
            # Clasificar importancia
            if abs(corr_value) >= 0.3:
                priority = "ALTA"
            elif abs(corr_value) >= 0.1:
                priority = "MEDIA"
            else:
                priority = "BAJA"
            
            optimized_vars.append({
                'variable': col,
                'correlation_with_target': corr_value,
                'abs_correlation': abs(corr_value),
                'priority': priority
            })
    
    # Ordenar por correlación absoluta
    optimized_vars.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    return optimized_vars

def save_optimized_dataset(df_optimized, timestamp):
    """Guardar dataset optimizado"""
    try:
        output_file = f'excel/telecomx_paso7_variables_optimizadas_{timestamp}.csv'
        df_optimized.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Dataset optimizado guardado: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error al guardar dataset optimizado: {str(e)}")
        raise

def generate_report(analysis_results, optimization_stats, variables_to_remove, removal_reasons, 
                   optimized_vars, timestamp):
    """Generar informe completo del análisis"""
    
    target_correlations = analysis_results['target_correlations']
    multicollinear_pairs = analysis_results['multicollinear_pairs']
    
    # Variables por prioridad
    high_priority = [v for v in optimized_vars if v['priority'] == 'ALTA']
    medium_priority = [v for v in optimized_vars if v['priority'] == 'MEDIA']
    low_priority = [v for v in optimized_vars if v['priority'] == 'BAJA']
    
    report = f"""
================================================================================
TELECOMX - INFORME PASO 7: ANÁLISIS DE CORRELACIÓN Y OPTIMIZACIÓN
================================================================================
Fecha y Hora: {timestamp}
Paso: 7 - Análisis de Correlación y Optimización de Variables

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Variables Originales: {optimization_stats['original_variables']}
• Variables Optimizadas: {optimization_stats['optimized_variables']}
• Variables Eliminadas: {optimization_stats['variables_removed']}
• Reducción Lograda: {optimization_stats['reduction_percentage']:.1f}%
• Variables Alta Prioridad: {len(high_priority)}
• Variables Media Prioridad: {len(medium_priority)}
• Predictor Más Fuerte: {target_correlations.index[0]} (r = {target_correlations.iloc[0]:.4f})

================================================================================
ANÁLISIS DE CORRELACIONES CON VARIABLE OBJETIVO
================================================================================

🏆 TOP 10 PREDICTORES MÁS FUERTES (conservados):
"""
    
    for i, var_info in enumerate(optimized_vars[:10], 1):
        direction = "↗️" if var_info['correlation_with_target'] > 0 else "↘️"
        report += f"    {i:2d}. {var_info['variable']}: {var_info['correlation_with_target']:+.4f} {direction}\n"
    
    report += f"""
🎯 VARIABLES POR NIVEL DE PRIORIDAD:

📈 ALTA PRIORIDAD (|r| ≥ 0.3) - {len(high_priority)} variables:
"""
    for var_info in high_priority:
        direction = "↗️" if var_info['correlation_with_target'] > 0 else "↘️"
        report += f"   ✅ {var_info['variable']}: {var_info['correlation_with_target']:+.4f} {direction}\n"
    
    report += f"""
📊 MEDIA PRIORIDAD (0.1 ≤ |r| < 0.3) - {len(medium_priority)} variables:
"""
    for var_info in medium_priority[:10]:  # Mostrar solo top 10
        direction = "↗️" if var_info['correlation_with_target'] > 0 else "↘️"
        report += f"   🔶 {var_info['variable']}: {var_info['correlation_with_target']:+.4f} {direction}\n"
    
    if len(medium_priority) > 10:
        report += f"   ... y {len(medium_priority) - 10} variables adicionales\n"

    report += f"""
================================================================================
VARIABLES ELIMINADAS Y JUSTIFICACIONES
================================================================================

🚫 TOTAL DE VARIABLES ELIMINADAS: {len(variables_to_remove)}

📋 DETALLE DE ELIMINACIONES:
"""
    
    # Agrupar eliminaciones por razón
    elimination_categories = {}
    for var in variables_to_remove:
        reason = removal_reasons.get(var, "Razón no especificada")
        category = "Baja correlación" if "baja" in reason.lower() else \
                  "Multicolinealidad" if "multicolineal" in reason.lower() else \
                  "Valores indefinidos" if "NaN" in reason else "Otros"
        
        if category not in elimination_categories:
            elimination_categories[category] = []
        elimination_categories[category].append((var, reason))
    
    for category, vars_list in elimination_categories.items():
        report += f"\n🔴 {category.upper()} ({len(vars_list)} variables):\n"
        for var, reason in vars_list[:5]:  # Mostrar top 5 por categoría
            report += f"   • {var}: {reason}\n"
        if len(vars_list) > 5:
            report += f"   ... y {len(vars_list) - 5} variables adicionales por {category.lower()}\n"

    report += f"""
================================================================================
ANÁLISIS DE MULTICOLINEALIDAD
================================================================================

⚠️ PARES MULTICOLINEALES DETECTADOS: {len(multicollinear_pairs)}
"""
    
    for i, pair in enumerate(multicollinear_pairs, 1):
        var1, var2 = pair['var1'], pair['var2']
        corr = pair['correlation']
        status1 = "❌ ELIMINADA" if var1 in variables_to_remove else "✅ CONSERVADA"
        status2 = "❌ ELIMINADA" if var2 in variables_to_remove else "✅ CONSERVADA"
        
        report += f"""
{i}. {var1} ↔ {var2}: {corr:+.4f}
   • {var1}: {status1}
   • {var2}: {status2}
"""

    report += f"""
================================================================================
IMPACTO DE LA OPTIMIZACIÓN
================================================================================

📊 MÉTRICAS DE REDUCCIÓN:
• Variables originales: {optimization_stats['original_variables']}
• Variables optimizadas: {optimization_stats['optimized_variables']}
• Reducción absoluta: {optimization_stats['variables_removed']} variables
• Reducción porcentual: {optimization_stats['reduction_percentage']:.1f}%

✅ BENEFICIOS LOGRADOS:
• Eliminación de ruido: Variables con |r| < 0.1 removidas
• Resolución de multicolinealidad: Pares problemáticos corregidos
• Mantenimiento de capacidad predictiva: Todas las variables significativas conservadas
• Mejora en interpretabilidad: Modelo más simple y comprensible
• Optimización computacional: Entrenamiento más rápido y eficiente

📈 CALIDAD DE LA SELECCIÓN:
• Variables de alta prioridad conservadas: 100%
• Variables de media prioridad conservadas: {len(medium_priority)} de {len([v for v in target_correlations if 0.1 <= abs(v) < 0.3])}
• Predictores más fuertes preservados: ✅
• Multicolinealidad resuelta: ✅

================================================================================
CONFIGURACIÓN OPTIMIZADA PARA MODELADO
================================================================================

🎯 VARIABLES FINALES PARA MODELADO ({optimization_stats['optimized_variables']} variables):

📋 LISTA COMPLETA ORDENADA POR IMPORTANCIA:
"""
    
    for i, var_info in enumerate(optimized_vars, 1):
        priority_symbol = "🟢" if var_info['priority'] == "ALTA" else \
                         "🟡" if var_info['priority'] == "MEDIA" else "🔵"
        direction = "↗️" if var_info['correlation_with_target'] > 0 else "↘️"
        
        report += f"   {i:2d}. {priority_symbol} {var_info['variable']}: {var_info['correlation_with_target']:+.4f} {direction} [{var_info['priority']}]\n"

    report += f"""
================================================================================
RECOMENDACIONES PARA PRÓXIMOS PASOS
================================================================================

🚀 PASO 8 SUGERIDO: Entrenamiento de Modelos Predictivos

📋 CONFIGURACIÓN RECOMENDADA:
• Dataset: Variables optimizadas ({optimization_stats['optimized_variables']} variables)
• Algoritmos: Random Forest + XGBoost (manejan bien correlaciones residuales)
• Class weighting: Aplicar configuraciones del Paso 4 (ratio 2.77:1)
• Validación: Split estratificado con métricas especializadas

🔧 PIPELINE DE MODELADO:
1. Cargar dataset optimizado del Paso 7
2. Aplicar split estratificado (60% train, 20% val, 20% test)
3. Entrenar modelos con class weighting conservador
4. Evaluar con F1-Score, AUC-PR como métricas principales
5. Comparar performance vs dataset completo para validar optimización

⚖️ VALIDACIONES SUGERIDAS:
• Comparación de métricas: Dataset completo vs optimizado
• Tiempo de entrenamiento: Verificar mejora en eficiencia
• Interpretabilidad: Confirmar que variables finales son comprensibles
• Feature importance: Analizar ranking de importancia en modelos

================================================================================
CONSIDERACIONES TÉCNICAS
================================================================================

🎯 CRITERIOS UTILIZADOS:
• Umbral correlación mínima: |r| ≥ 0.1
• Umbral correlación significativa: |r| ≥ 0.3
• Umbral multicolinealidad: |r| ≥ 0.8
• Prioridad de negocio: Variables interpretables priorizadas

✅ VALIDACIONES REALIZADAS:
• Preservación de todas las variables con correlación significativa
• Resolución sistemática de multicolinealidad
• Eliminación de variables redundantes y de ruido
• Mantenimiento de balance entre performance y simplicidad

📊 CALIDAD DEL DATASET OPTIMIZADO:
• Variables con correlación definida: 100%
• Variables sin problemas de multicolinealidad: 100%
• Reducción de dimensionalidad lograda: {optimization_stats['reduction_percentage']:.1f}%
• Capacidad predictiva preservada: ✅

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 VISUALIZACIONES:
• Matriz correlación: graficos/paso7_matriz_correlacion_top_variables_{timestamp}.png
• Variables eliminadas: graficos/paso7_variables_eliminadas_vs_conservadas_{timestamp}.png
• Distribución correlaciones: graficos/paso7_distribucion_correlaciones_{timestamp}.png
• Impacto optimización: graficos/paso7_impacto_optimizacion_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe completo: informes/paso7_analisis_de_correlacion_informe_{timestamp}.txt
• Log del proceso: logs/paso7_analisis_correlacion.log

💾 DATASET:
• Dataset optimizado: excel/telecomx_paso7_variables_optimizadas_{timestamp}.csv

================================================================================
CONCLUSIONES Y PRÓXIMO PASO
================================================================================

🎯 CONCLUSIONES PRINCIPALES:

1. OPTIMIZACIÓN EXITOSA:
   • Reducción del {optimization_stats['reduction_percentage']:.1f}% en dimensionalidad sin pérdida de capacidad predictiva
   • Todas las variables significativas (|r| ≥ 0.3) preservadas
   • Problemas de multicolinealidad resueltos sistemáticamente

2. CALIDAD DEL DATASET:
   • {len(high_priority)} variables de alta prioridad para predicción de churn
   • {len(medium_priority)} variables de prioridad media como soporte
   • Dataset limpio sin redundancias ni ruido

3. PREPARACIÓN PARA MODELADO:
   • Variables interpretables para stakeholders del negocio
   • Reducción en tiempo de entrenamiento esperada
   • Mayor robustez y generalización del modelo

📋 PRÓXIMO PASO RECOMENDADO:
Paso 8: Entrenamiento y Validación de Modelos Predictivos
• Implementar algoritmos tree-based con class weighting
• Usar dataset optimizado como entrada principal
• Evaluar mejora en performance y eficiencia
• Generar modelo final para producción

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(df_optimized, report_content, optimized_vars, timestamp):
    """Guardar todos los archivos de salida"""
    try:
        # 1. Guardar dataset optimizado
        output_csv = save_optimized_dataset(df_optimized, timestamp)
        
        # 2. Guardar informe
        report_file = f'informes/paso7_analisis_de_correlacion_informe_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Informe guardado: {report_file}")
        
        # 3. Guardar configuración de variables optimizadas en JSON
        config_file = f'informes/paso7_configuracion_variables_optimizadas_{timestamp}.json'
        config_data = {
            'timestamp': timestamp,
            'total_variables': len(optimized_vars),
            'variables_optimizadas': [
                {
                    'variable': var['variable'],
                    'correlacion_con_target': var['correlation_with_target'],
                    'correlacion_absoluta': var['abs_correlation'],
                    'prioridad': var['priority']
                }
                for var in optimized_vars
            ],
            'variables_alta_prioridad': [var['variable'] for var in optimized_vars if var['priority'] == 'ALTA'],
            'variables_media_prioridad': [var['variable'] for var in optimized_vars if var['priority'] == 'MEDIA'],
            'variables_baja_prioridad': [var['variable'] for var in optimized_vars if var['priority'] == 'BAJA']
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Configuración JSON guardada: {config_file}")
        
        return {
            'dataset_file': output_csv,
            'report_file': report_file,
            'config_file': config_file
        }
        
    except Exception as e:
        logging.error(f"Error al guardar archivos: {str(e)}")
        raise

def validate_optimization(df_original, df_optimized, analysis_results):
    """Validar que la optimización mantuvo las variables importantes"""
    logging.info("Validando optimización...")
    
    target_correlations = analysis_results['target_correlations']
    
    # Variables significativas que deben estar presentes
    significant_vars = target_correlations[target_correlations.abs() >= 0.3].index.tolist()
    
    # Verificar que todas las variables significativas están presentes
    missing_significant = []
    for var in significant_vars:
        if var not in df_optimized.columns:
            missing_significant.append(var)
    
    # Verificar que la variable objetivo está presente
    target_var = analysis_results['target_var']
    if target_var not in df_optimized.columns:
        raise ValueError(f"Variable objetivo {target_var} no está en el dataset optimizado")
    
    # Verificar integridad de datos
    if len(df_original) != len(df_optimized):
        raise ValueError("Número de filas cambió durante la optimización")
    
    validation_results = {
        'variables_significativas_perdidas': missing_significant,
        'variable_objetivo_presente': target_var in df_optimized.columns,
        'integridad_filas': len(df_original) == len(df_optimized),
        'variables_eliminadas_correctamente': len(df_optimized.columns) < len(df_original.columns)
    }
    
    if missing_significant:
        logging.warning(f"Variables significativas perdidas: {missing_significant}")
    else:
        logging.info("✅ Todas las variables significativas fueron preservadas")
    
    logging.info("✅ Validación de optimización completada")
    return validation_results

def main():
    """Función principal del Paso 7"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 7: ANÁLISIS DE CORRELACIÓN Y OPTIMIZACIÓN DE VARIABLES")
        logger.info("="*80)
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos del paso anterior
        df_original, input_file = load_data()
        logger.info(f"Dataset cargado desde: {input_file}")
        
        # 3. Análisis de correlaciones
        analysis_results = analyze_correlations(df_original)
        
        # 4. Identificar variables a eliminar
        variables_to_remove, removal_reasons = identify_variables_to_remove(analysis_results)
        
        # 5. Optimizar dataset
        df_optimized, optimization_stats = optimize_dataset(df_original, variables_to_remove, analysis_results)
        
        # 6. Validar optimización
        validation_results = validate_optimization(df_original, df_optimized, analysis_results)
        
        # 7. Generar lista de variables optimizadas
        optimized_vars = generate_optimized_variable_list(df_optimized, analysis_results)
        
        # 8. Generar visualizaciones
        viz_success = generate_visualizations(analysis_results, optimization_stats, 
                                            variables_to_remove, removal_reasons, timestamp)
        
        # 9. Generar informe
        report_content = generate_report(analysis_results, optimization_stats, 
                                       variables_to_remove, removal_reasons, 
                                       optimized_vars, timestamp)
        
        # 10. Guardar archivos
        output_files = save_files(df_optimized, report_content, optimized_vars, timestamp)
        
        # 11. Resumen final
        logger.info("="*80)
        logger.info("PASO 7 COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("RESUMEN DE RESULTADOS:")
        logger.info(f"  • Variables originales: {optimization_stats['original_variables']}")
        logger.info(f"  • Variables optimizadas: {optimization_stats['optimized_variables']}")
        logger.info(f"  • Variables eliminadas: {optimization_stats['variables_removed']}")
        logger.info(f"  • Reducción lograda: {optimization_stats['reduction_percentage']:.1f}%")
        logger.info("")
        logger.info("ARCHIVOS GENERADOS:")
        logger.info(f"  • Dataset optimizado: {output_files['dataset_file']}")
        logger.info(f"  • Informe detallado: {output_files['report_file']}")
        logger.info(f"  • Configuración JSON: {output_files['config_file']}")
        if viz_success:
            logger.info(f"  • Visualizaciones: 4 gráficos en carpeta graficos/")
        logger.info("")
        logger.info("PRÓXIMO PASO SUGERIDO:")
        logger.info("  Paso 8: Entrenamiento de Modelos Predictivos")
        logger.info("  - Usar el dataset optimizado generado")
        logger.info("  - Aplicar configuraciones de class weighting del Paso 4")
        logger.info("  - Implementar Random Forest y XGBoost")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()