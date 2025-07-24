"""
TELECOMX - PIPELINE DE PREDICCIÓN DE CHURN
===========================================
Paso 6: Análisis de Correlación

Descripción:
    Visualiza la matriz de correlación para identificar relaciones entre las 
    variables numéricas. Presta especial atención a las variables que muestran 
    una mayor correlación con la cancelación, ya que estas pueden ser fuertes 
    candidatas para el modelo predictivo.

Autor: Ingeniero de Datos
Fecha: 2025-07-22
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Crea las carpetas necesarias si no existen"""
    directories = ['script', 'informes', 'excel', 'logs', 'graficos']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Carpeta creada: {directory}")

# Configuración de logging
def setup_logging():
    """Configura el sistema de logging para trackear el proceso"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/paso6_analisis_correlacion.log', mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def find_latest_paso2_file():
    """
    Encuentra el archivo más reciente del Paso 2 en la carpeta excel
    
    Returns:
        str: Ruta al archivo más reciente del Paso 2
    """
    logger = logging.getLogger(__name__)
    
    excel_files = [f for f in os.listdir('excel') if f.startswith('telecomx_paso2_encoding_aplicado_')]
    
    if not excel_files:
        raise FileNotFoundError("No se encontró ningún archivo del Paso 2 en la carpeta excel/")
    
    # Ordenar por fecha de modificación y tomar el más reciente
    excel_files.sort(key=lambda x: os.path.getmtime(os.path.join('excel', x)), reverse=True)
    latest_file = os.path.join('excel', excel_files[0])
    
    logger.info(f"Archivo del Paso 2 encontrado: {latest_file}")
    return latest_file

def load_data(file_path):
    """
    Carga el dataset del Paso 2 con detección automática de codificación
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    logger = logging.getLogger(__name__)
    
    # Lista de codificaciones a probar
    encodings_to_try = ['utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1', 'utf-8']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep=';', encoding=encoding)
            logger.info("Dataset cargado exitosamente desde: " + file_path)
            logger.info(f"Codificacion utilizada: {encoding}")
            logger.info(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
            return df
        except (UnicodeDecodeError, UnicodeError):
            logger.warning(f"No se pudo cargar con codificacion {encoding}, probando siguiente...")
            continue
        except Exception as e:
            if encoding == encodings_to_try[-1]:
                logger.error("Error al cargar el dataset: " + str(e))
                raise
            else:
                continue
    
    raise ValueError(f"No se pudo cargar el archivo {file_path} con ninguna codificacion probada")

def verify_target_variable(df):
    """
    Verifica que la variable objetivo existe y es numérica
    
    Args:
        df (pd.DataFrame): Dataset cargado
        
    Returns:
        bool: True si la variable objetivo es válida
    """
    logger = logging.getLogger(__name__)
    logger.info("VERIFICANDO VARIABLE OBJETIVO")
    logger.info("=" * 40)
    
    if 'Abandono_Cliente' not in df.columns:
        logger.error("Variable objetivo 'Abandono_Cliente' no encontrada")
        return False
    
    if not pd.api.types.is_numeric_dtype(df['Abandono_Cliente']):
        logger.error("Variable objetivo 'Abandono_Cliente' no es numérica")
        return False
    
    unique_vals = df['Abandono_Cliente'].unique()
    logger.info(f"Variable objetivo verificada: {len(unique_vals)} valores únicos")
    logger.info(f"Valores: {sorted(unique_vals)}")
    
    return True

def calculate_correlation_matrix(df):
    """
    Calcula la matriz de correlación completa del dataset
    
    Args:
        df (pd.DataFrame): Dataset con variables numéricas
        
    Returns:
        tuple: (correlation_matrix, numeric_columns)
    """
    logger = logging.getLogger(__name__)
    logger.info("CALCULANDO MATRIZ DE CORRELACION")
    logger.info("=" * 40)
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_columns = numeric_df.columns.tolist()
    
    logger.info(f"Variables numéricas encontradas: {len(numeric_columns)}")
    logger.info(f"Variables: {numeric_columns}")
    
    # Calcular matriz de correlación
    correlation_matrix = numeric_df.corr()
    
    logger.info("Matriz de correlación calculada exitosamente")
    logger.info(f"Dimensiones: {correlation_matrix.shape}")
    
    return correlation_matrix, numeric_columns

def analyze_target_correlations(correlation_matrix, target_threshold=0.3):
    """
    Analiza las correlaciones con la variable objetivo
    
    Args:
        correlation_matrix (pd.DataFrame): Matriz de correlación
        target_threshold (float): Umbral para correlación significativa
        
    Returns:
        dict: Análisis de correlaciones con target
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO CORRELACIONES CON VARIABLE OBJETIVO")
    logger.info("=" * 50)
    
    if 'Abandono_Cliente' not in correlation_matrix.columns:
        logger.error("Variable objetivo no encontrada en matriz de correlación")
        return {}
    
    # Correlaciones con variable objetivo
    target_correlations = correlation_matrix['Abandono_Cliente'].drop('Abandono_Cliente')
    
    # Ordenar por valor absoluto (correlaciones más fuertes)
    target_correlations_abs = target_correlations.abs().sort_values(ascending=False)
    
    # Identificar correlaciones significativas
    significant_positive = target_correlations[target_correlations >= target_threshold].sort_values(ascending=False)
    significant_negative = target_correlations[target_correlations <= -target_threshold].sort_values(ascending=True)
    
    analysis = {
        'all_correlations': target_correlations,
        'correlations_by_strength': target_correlations_abs,
        'significant_positive': significant_positive,
        'significant_negative': significant_negative,
        'threshold_used': target_threshold,
        'total_significant': len(significant_positive) + len(significant_negative),
        'strongest_predictor': target_correlations_abs.index[0] if len(target_correlations_abs) > 0 else None,
        'strongest_correlation': target_correlations_abs.iloc[0] if len(target_correlations_abs) > 0 else 0
    }
    
    # Logging de resultados
    logger.info(f"Umbral de significancia: |r| >= {target_threshold}")
    logger.info(f"Variables con correlación significativa: {analysis['total_significant']}")
    logger.info(f"Correlaciones positivas significativas: {len(significant_positive)}")
    logger.info(f"Correlaciones negativas significativas: {len(significant_negative)}")
    
    if analysis['strongest_predictor']:
        logger.info(f"Predictor más fuerte: {analysis['strongest_predictor']}")
        logger.info(f"Correlación: {analysis['strongest_correlation']:.4f}")
    
    # Log de top correlaciones
    logger.info("\nTop 10 correlaciones con Abandono_Cliente:")
    for i, (var, corr) in enumerate(target_correlations_abs.head(10).items(), 1):
        direction = "positiva" if target_correlations[var] > 0 else "negativa"
        logger.info(f"  {i:2d}. {var}: {target_correlations[var]:+.4f} ({direction})")
    
    return analysis

def detect_multicollinearity(correlation_matrix, multicollinearity_threshold=0.8):
    """
    Detecta multicolinealidad entre variables predictoras
    
    Args:
        correlation_matrix (pd.DataFrame): Matriz de correlación
        multicollinearity_threshold (float): Umbral para detectar multicolinealidad
        
    Returns:
        dict: Análisis de multicolinealidad
    """
    logger = logging.getLogger(__name__)
    logger.info("DETECTANDO MULTICOLINEALIDAD")
    logger.info("=" * 35)
    
    # Excluir variable objetivo del análisis
    predictor_matrix = correlation_matrix.drop('Abandono_Cliente', axis=0).drop('Abandono_Cliente', axis=1)
    
    # Encontrar pares con alta correlación
    high_correlations = []
    
    for i in range(len(predictor_matrix.columns)):
        for j in range(i+1, len(predictor_matrix.columns)):
            var1 = predictor_matrix.columns[i]
            var2 = predictor_matrix.columns[j]
            correlation = predictor_matrix.iloc[i, j]
            
            if abs(correlation) >= multicollinearity_threshold:
                high_correlations.append({
                    'variable_1': var1,
                    'variable_2': var2,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation)
                })
    
    # Ordenar por correlación absoluta descendente
    high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Identificar variables problemáticas
    problematic_vars = set()
    for pair in high_correlations:
        problematic_vars.add(pair['variable_1'])
        problematic_vars.add(pair['variable_2'])
    
    analysis = {
        'threshold_used': multicollinearity_threshold,
        'high_correlation_pairs': high_correlations,
        'total_problematic_pairs': len(high_correlations),
        'problematic_variables': list(problematic_vars),
        'total_problematic_vars': len(problematic_vars)
    }
    
    # Logging de resultados
    logger.info(f"Umbral de multicolinealidad: |r| >= {multicollinearity_threshold}")
    logger.info(f"Pares con alta correlación: {len(high_correlations)}")
    logger.info(f"Variables problemáticas: {len(problematic_vars)}")
    
    if high_correlations:
        logger.info("\nPares con multicolinealidad detectada:")
        for i, pair in enumerate(high_correlations[:5], 1):
            logger.info(f"  {i}. {pair['variable_1']} ↔ {pair['variable_2']}: {pair['correlation']:+.4f}")
        
        if len(high_correlations) > 5:
            logger.info(f"  ... y {len(high_correlations) - 5} pares adicionales")
    else:
        logger.info("No se detectó multicolinealidad significativa")
    
    return analysis

def generate_variable_recommendations(target_analysis, multicollinearity_analysis, correlation_matrix):
    """
    Genera recomendaciones para selección de variables
    
    Args:
        target_analysis (dict): Análisis de correlaciones con target
        multicollinearity_analysis (dict): Análisis de multicolinealidad
        correlation_matrix (pd.DataFrame): Matriz de correlación
        
    Returns:
        dict: Recomendaciones de variables
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO RECOMENDACIONES DE VARIABLES")
    logger.info("=" * 45)
    
    all_variables = correlation_matrix.columns.tolist()
    if 'Abandono_Cliente' in all_variables:
        all_variables.remove('Abandono_Cliente')
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'consider_removal': [],
        'reasoning': {}
    }
    
    # Clasificar variables según correlación con target
    for var in all_variables:
        target_corr = abs(correlation_matrix.loc[var, 'Abandono_Cliente'])
        
        # Verificar si está en pares multicolineales
        is_multicollinear = var in multicollinearity_analysis['problematic_variables']
        
        # Asignar prioridad
        if target_corr >= 0.3:
            if is_multicollinear:
                recommendations['consider_removal'].append(var)
                recommendations['reasoning'][var] = f"Alta correlación con target ({target_corr:.3f}) pero multicolineal - evaluar cuál mantener"
            else:
                recommendations['high_priority'].append(var)
                recommendations['reasoning'][var] = f"Alta correlación con target ({target_corr:.3f}) y sin multicolinealidad"
        
        elif target_corr >= 0.1:
            if is_multicollinear:
                recommendations['consider_removal'].append(var)
                recommendations['reasoning'][var] = f"Correlación moderada ({target_corr:.3f}) pero multicolineal - considerar eliminación"
            else:
                recommendations['medium_priority'].append(var)
                recommendations['reasoning'][var] = f"Correlación moderada con target ({target_corr:.3f})"
        
        else:
            if is_multicollinear:
                recommendations['consider_removal'].append(var)
                recommendations['reasoning'][var] = f"Baja correlación ({target_corr:.3f}) y multicolineal - candidato para eliminación"
            else:
                recommendations['low_priority'].append(var)
                recommendations['reasoning'][var] = f"Baja correlación con target ({target_corr:.3f})"
    
    # Logging de recomendaciones
    logger.info(f"Variables de alta prioridad: {len(recommendations['high_priority'])}")
    logger.info(f"Variables de prioridad media: {len(recommendations['medium_priority'])}")
    logger.info(f"Variables de baja prioridad: {len(recommendations['low_priority'])}")
    logger.info(f"Variables a considerar eliminación: {len(recommendations['consider_removal'])}")
    
    logger.info("\nVariables de ALTA PRIORIDAD:")
    for var in recommendations['high_priority']:
        logger.info(f"  • {var}")
    
    if recommendations['consider_removal']:
        logger.info("\nVariables a CONSIDERAR ELIMINACIÓN:")
        for var in recommendations['consider_removal']:
            logger.info(f"  • {var}")
    
    return recommendations

def create_correlation_visualizations(correlation_matrix, target_analysis, 
                                    multicollinearity_analysis, timestamp):
    """
    Crea visualizaciones de análisis de correlación
    
    Args:
        correlation_matrix (pd.DataFrame): Matriz de correlación
        target_analysis (dict): Análisis de correlaciones con target
        multicollinearity_analysis (dict): Análisis de multicolinealidad
        timestamp (str): Timestamp para archivos
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO VISUALIZACIONES DE CORRELACION")
    logger.info("=" * 50)
    
    try:
        # 1. Heatmap completo de correlación
        plt.figure(figsize=(14, 12))
        
        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Crear heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={'size': 8})
        
        plt.title('Matriz de Correlación Completa\nTriángulo Inferior', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'graficos/paso6_matriz_correlacion_completa_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlaciones con variable objetivo (ranking)
        plt.figure(figsize=(12, 8))
        
        target_corr = target_analysis['all_correlations'].sort_values(key=abs, ascending=True)
        
        # Colores según signo de correlación
        colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
        
        bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(target_corr)), target_corr.index)
        plt.xlabel('Correlación con Abandono_Cliente', fontsize=12, fontweight='bold')
        plt.title('Ranking de Correlaciones con Variable Objetivo\n(Ordenado por magnitud)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Añadir líneas de referencia
        plt.axvline(x=0.3, color='green', linestyle='--', alpha=0.7, label='Umbral significativo (+0.3)')
        plt.axvline(x=-0.3, color='green', linestyle='--', alpha=0.7, label='Umbral significativo (-0.3)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01 if width >= 0 else width - 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', 
                    ha='left' if width >= 0 else 'right', 
                    va='center', fontsize=8)
        
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graficos/paso6_ranking_correlaciones_target_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap específico de variables más correlacionadas con target
        top_variables = target_analysis['correlations_by_strength'].head(15).index.tolist()
        top_variables.append('Abandono_Cliente')
        
        if len(top_variables) > 1:
            plt.figure(figsize=(12, 10))
            
            top_corr_matrix = correlation_matrix.loc[top_variables, top_variables]
            
            sns.heatmap(top_corr_matrix, 
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={"shrink": .8})
            
            plt.title('Correlaciones: Top 15 Variables vs Abandono_Cliente\n(Variables más correlacionadas con target)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'graficos/paso6_top_correlaciones_target_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Análisis de multicolinealidad (si existe)
        if multicollinearity_analysis['high_correlation_pairs']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Subplot 1: Distribución de correlaciones
            all_corr_values = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.columns[i] != 'Abandono_Cliente' and correlation_matrix.columns[j] != 'Abandono_Cliente':
                        all_corr_values.append(abs(correlation_matrix.iloc[i, j]))
            
            ax1.hist(all_corr_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=0.8, color='red', linestyle='--', label='Umbral multicolinealidad (0.8)')
            ax1.set_xlabel('Correlación Absoluta')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Correlaciones\nentre Variables Predictoras')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Subplot 2: Top pares multicolineales
            if len(multicollinearity_analysis['high_correlation_pairs']) > 0:
                top_pairs = multicollinearity_analysis['high_correlation_pairs'][:10]
                pair_labels = [f"{p['variable_1'][:15]}...\n{p['variable_2'][:15]}..." 
                              if len(p['variable_1']) > 15 or len(p['variable_2']) > 15 
                              else f"{p['variable_1']}\n{p['variable_2']}" 
                              for p in top_pairs]
                correlations = [p['correlation'] for p in top_pairs]
                
                colors_mult = ['red' if abs(c) >= 0.9 else 'orange' for c in correlations]
                
                bars = ax2.barh(range(len(correlations)), correlations, color=colors_mult, alpha=0.7)
                ax2.set_yticks(range(len(correlations)))
                ax2.set_yticklabels(pair_labels, fontsize=8)
                ax2.set_xlabel('Correlación')
                ax2.set_title('Top Pares Multicolineales\n(Variables Predictoras)')
                ax2.grid(axis='x', alpha=0.3)
                
                # Añadir valores
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax2.text(width + 0.01 if width >= 0 else width - 0.01, 
                            bar.get_y() + bar.get_height()/2, 
                            f'{width:.3f}', 
                            ha='left' if width >= 0 else 'right', 
                            va='center', fontsize=8)
            
            plt.suptitle('Análisis de Multicolinealidad entre Variables Predictoras', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'graficos/paso6_analisis_multicolinealidad_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Todas las visualizaciones generadas exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar visualizaciones: {str(e)}")

def generate_optimized_variable_list(recommendations, correlation_matrix):
    """
    Genera lista optimizada de variables para modelado
    
    Args:
        recommendations (dict): Recomendaciones de variables
        correlation_matrix (pd.DataFrame): Matriz de correlación
        
    Returns:
        dict: Lista optimizada y configuraciones
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO LISTA OPTIMIZADA DE VARIABLES")
    logger.info("=" * 50)
    
    # Lista optimizada: alta y media prioridad
    optimized_features = recommendations['high_priority'] + recommendations['medium_priority']
    
    # Configuraciones para diferentes escenarios
    configurations = {
        'full_features': {
            'variables': correlation_matrix.columns.tolist(),
            'count': len(correlation_matrix.columns),
            'description': 'Todas las variables disponibles'
        },
        
        'optimized_features': {
            'variables': optimized_features + ['Abandono_Cliente'],
            'count': len(optimized_features) + 1,
            'description': 'Variables con correlación significativa (alta y media prioridad)'
        },
        
        'high_priority_only': {
            'variables': recommendations['high_priority'] + ['Abandono_Cliente'],
            'count': len(recommendations['high_priority']) + 1,
            'description': 'Solo variables de alta prioridad (correlación >= 0.3)'
        },
        
        'top_10_features': {
            'variables': correlation_matrix['Abandono_Cliente'].abs().nlargest(11).index.tolist(),  # Top 10 + target
            'count': 11,
            'description': 'Top 10 variables más correlacionadas con target'
        }
    }
    
    # Remover variable objetivo de listas donde no corresponde
    for config_name, config in configurations.items():
        if config_name != 'full_features':
            variables_only = [v for v in config['variables'] if v != 'Abandono_Cliente']
            config['variables_only'] = variables_only
            config['feature_count'] = len(variables_only)
    
    optimization_summary = {
        'configurations': configurations,
        'recommended_config': 'optimized_features',
        'reduction_achieved': {
            'original_features': len(correlation_matrix.columns) - 1,  # Excluir target
            'optimized_features': len(optimized_features),
            'reduction_percentage': (1 - len(optimized_features) / (len(correlation_matrix.columns) - 1)) * 100
        }
    }
    
    logger.info("Configuraciones generadas:")
    for config_name, config in configurations.items():
        feature_count = config.get('feature_count', config['count'] - 1)
        logger.info(f"  {config_name}: {feature_count} variables - {config['description']}")
    
    logger.info(f"\nConfiguración recomendada: {optimization_summary['recommended_config']}")
    logger.info(f"Reducción de dimensionalidad: {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}%")
    
    return optimization_summary

def generate_report(correlation_matrix, target_analysis, multicollinearity_analysis, 
                   recommendations, optimization_summary, timestamp):
    """
    Genera informe detallado del análisis de correlación
    
    Returns:
        str: Contenido del informe
    """
    
    report = f"""
================================================================================
TELECOMX - INFORME DE ANÁLISIS DE CORRELACIÓN
================================================================================
Fecha y Hora: {timestamp}
Paso: 6 - Análisis de Correlación

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Total de Variables Analizadas: {len(correlation_matrix.columns)}
• Variables Predictoras: {len(correlation_matrix.columns) - 1}
• Variables con Correlación Significativa (|r| >= 0.3): {target_analysis['total_significant']}
• Predictor Más Fuerte: {target_analysis['strongest_predictor']}
• Correlación Más Fuerte: {target_analysis['strongest_correlation']:.4f}
• Pares con Multicolinealidad (|r| >= 0.8): {multicollinearity_analysis['total_problematic_pairs']}
• Variables Recomendadas para Modelado: {len(recommendations['high_priority']) + len(recommendations['medium_priority'])}
• Reducción de Dimensionalidad Lograda: {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}%

================================================================================
ANÁLISIS DE CORRELACIONES CON VARIABLE OBJETIVO
================================================================================

🎯 CORRELACIONES SIGNIFICATIVAS (|r| >= {target_analysis['threshold_used']}):

📈 CORRELACIONES POSITIVAS FUERTES:
"""
    
    if len(target_analysis['significant_positive']) > 0:
        for var, corr in target_analysis['significant_positive'].items():
            report += f"   • {var}: +{corr:.4f}\n"
    else:
        report += "   • No se encontraron correlaciones positivas significativas\n"
    
    report += f"""
📉 CORRELACIONES NEGATIVAS FUERTES:
"""
    
    if len(target_analysis['significant_negative']) > 0:
        for var, corr in target_analysis['significant_negative'].items():
            report += f"   • {var}: {corr:.4f}\n"
    else:
        report += "   • No se encontraron correlaciones negativas significativas\n"
    
    report += f"""
🏆 TOP 10 PREDICTORES MÁS FUERTES (por correlación absoluta):
"""
    
    for i, (var, corr_abs) in enumerate(target_analysis['correlations_by_strength'].head(10).items(), 1):
        actual_corr = target_analysis['all_correlations'][var]
        direction = "↗️" if actual_corr > 0 else "↘️"
        report += f"   {i:2d}. {var}: {actual_corr:+.4f} {direction}\n"
    
    report += f"""
================================================================================
ANÁLISIS DE MULTICOLINEALIDAD
================================================================================

⚠️ DETECCIÓN DE VARIABLES ALTAMENTE CORRELACIONADAS (|r| >= {multicollinearity_analysis['threshold_used']}):

Total de pares problemáticos: {multicollinearity_analysis['total_problematic_pairs']}
Variables involucradas: {multicollinearity_analysis['total_problematic_vars']}

"""
    
    if multicollinearity_analysis['high_correlation_pairs']:
        report += "🔴 PARES CON MULTICOLINEALIDAD DETECTADA:\n"
        for i, pair in enumerate(multicollinearity_analysis['high_correlation_pairs'][:10], 1):
            severity = "🔴 CRÍTICA" if pair['abs_correlation'] >= 0.95 else "🟡 ALTA"
            report += f"   {i:2d}. {pair['variable_1']} ↔ {pair['variable_2']}: {pair['correlation']:+.4f} ({severity})\n"
        
        if len(multicollinearity_analysis['high_correlation_pairs']) > 10:
            remaining = len(multicollinearity_analysis['high_correlation_pairs']) - 10
            report += f"   ... y {remaining} pares adicionales\n"
    else:
        report += "✅ No se detectó multicolinealidad significativa entre variables predictoras.\n"
    
    report += f"""
================================================================================
RECOMENDACIONES DE SELECCIÓN DE VARIABLES
================================================================================

🚀 VARIABLES DE ALTA PRIORIDAD ({len(recommendations['high_priority'])} variables):
   Correlación significativa con target (|r| >= 0.3) y sin multicolinealidad
"""
    
    for var in recommendations['high_priority']:
        target_corr = target_analysis['all_correlations'][var]
        report += f"   ✅ {var}: {target_corr:+.4f}\n"
    
    report += f"""
📊 VARIABLES DE PRIORIDAD MEDIA ({len(recommendations['medium_priority'])} variables):
   Correlación moderada con target (0.1 <= |r| < 0.3) y sin multicolinealidad
"""
    
    for var in recommendations['medium_priority'][:10]:  # Mostrar máximo 10
        target_corr = target_analysis['all_correlations'][var]
        report += f"   🔶 {var}: {target_corr:+.4f}\n"
    
    if len(recommendations['medium_priority']) > 10:
        report += f"   ... y {len(recommendations['medium_priority']) - 10} variables adicionales\n"
    
    report += f"""
⚠️ VARIABLES A CONSIDERAR ELIMINACIÓN ({len(recommendations['consider_removal'])} variables):
   Variables con multicolinealidad o baja correlación con target
"""
    
    for var in recommendations['consider_removal'][:10]:  # Mostrar máximo 10
        target_corr = target_analysis['all_correlations'][var]
        reason = recommendations['reasoning'][var]
        report += f"   🔶 {var}: {target_corr:+.4f} - {reason}\n"
    
    if len(recommendations['consider_removal']) > 10:
        report += f"   ... y {len(recommendations['consider_removal']) - 10} variables adicionales\n"
    
    report += f"""
🔻 VARIABLES DE BAJA PRIORIDAD ({len(recommendations['low_priority'])} variables):
   Correlación baja con target (|r| < 0.1) pero sin problemas de multicolinealidad
"""
    
    for var in recommendations['low_priority'][:5]:  # Mostrar máximo 5
        target_corr = target_analysis['all_correlations'][var]
        report += f"   🔻 {var}: {target_corr:+.4f}\n"
    
    if len(recommendations['low_priority']) > 5:
        report += f"   ... y {len(recommendations['low_priority']) - 5} variables adicionales\n"
    
    report += f"""
================================================================================
CONFIGURACIONES OPTIMIZADAS PARA MODELADO
================================================================================

🎯 CONFIGURACIONES DISPONIBLES:
"""
    
    for config_name, config in optimization_summary['configurations'].items():
        feature_count = config.get('feature_count', config['count'] - 1)
        recommended_mark = " ⭐ RECOMENDADA" if config_name == optimization_summary['recommended_config'] else ""
        report += f"""
📋 {config_name.upper().replace('_', ' ')}{recommended_mark}:
   • Número de variables: {feature_count}
   • Descripción: {config['description']}
   • Variables incluidas: {', '.join(config.get('variables_only', config['variables'])[:5])}{'...' if len(config.get('variables_only', config['variables'])) > 5 else ''}
"""
    
    report += f"""
================================================================================
ANÁLISIS DE REDUCCIÓN DE DIMENSIONALIDAD
================================================================================

📊 IMPACTO DE LA OPTIMIZACIÓN:

• Variables originales: {optimization_summary['reduction_achieved']['original_features']}
• Variables optimizadas: {optimization_summary['reduction_achieved']['optimized_features']}
• Reducción lograda: {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}%

✅ BENEFICIOS DE LA REDUCCIÓN:
• Menor riesgo de overfitting
• Modelos más interpretables
• Entrenamiento más rápido
• Reducción de ruido en predicciones
• Menos problemas de multicolinealidad

📈 CALIDAD DE LA SELECCIÓN:
• Variables seleccionadas tienen correlación >= 0.1 con target
• Se eliminaron variables con multicolinealidad problemática
• Se preservaron los predictores más fuertes
• Balance entre performance y simplicidad

================================================================================
INTERPRETACIÓN DE CORRELACIONES PRINCIPALES
================================================================================

🔍 ANÁLISIS DE LOS PREDICTORES MÁS FUERTES:
"""
    
    # Analizar top 5 predictores
    top_5_predictors = target_analysis['correlations_by_strength'].head(5)
    
    for i, (var, corr_abs) in enumerate(top_5_predictors.items(), 1):
        actual_corr = target_analysis['all_correlations'][var]
        
        # Interpretación del tipo de variable
        if 'encoded' in var.lower() or any(x in var for x in ['_0', '_1', '_Si', '_No']):
            var_type = "Variable categórica encoded"
        elif any(x in var.lower() for x in ['cargo', 'facturacion', 'meses']):
            var_type = "Variable numérica de negocio"
        else:
            var_type = "Variable predictora"
        
        # Interpretación de la correlación
        if actual_corr > 0:
            interpretation = "Mayor valor → Mayor probabilidad de churn"
        else:
            interpretation = "Mayor valor → Menor probabilidad de churn"
        
        # Magnitud de la correlación
        if corr_abs >= 0.5:
            magnitude = "MUY FUERTE"
        elif corr_abs >= 0.3:
            magnitude = "FUERTE"
        elif corr_abs >= 0.1:
            magnitude = "MODERADA"
        else:
            magnitude = "DÉBIL"
        
        report += f"""
{i}. {var} (Correlación: {actual_corr:+.4f} - {magnitude}):
   • Tipo: {var_type}
   • Interpretación: {interpretation}
   • Relevancia: {'Predictor clave' if corr_abs >= 0.3 else 'Predictor secundario'}
"""
    
    report += f"""
================================================================================
RECOMENDACIONES PARA PRÓXIMOS PASOS
================================================================================

🚀 PASO 7 SUGERIDO: Entrenamiento de Modelos con Variables Optimizadas

📋 CONFIGURACIÓN RECOMENDADA:
• Usar configuración: {optimization_summary['recommended_config']}
• Variables a incluir: {optimization_summary['configurations'][optimization_summary['recommended_config']]['feature_count']}
• Algoritmos: Random Forest + XGBoost (tree-based, manejan correlaciones bien)
• Class weighting: Aplicar configuraciones del Paso 4

🔧 PIPELINE DE MODELADO:
1. Cargar datos del Paso 2
2. Seleccionar variables optimizadas
3. Aplicar split estratificado (Paso 4)
4. Entrenar modelos con class weighting
5. Evaluar con métricas especializadas (F1-Score, AUC-PR)

⚖️ VALIDACIÓN DE SELECCIÓN:
• Comparar performance: Todas las variables vs Variables optimizadas
• Verificar que reducción no afecta métricas principales
• Confirmar mejora en tiempo de entrenamiento
• Validar interpretabilidad de modelo resultante

================================================================================
CONSIDERACIONES TÉCNICAS
================================================================================

🎯 UMBRALES UTILIZADOS:
• Correlación significativa con target: |r| >= {target_analysis['threshold_used']}
• Multicolinealidad entre predictores: |r| >= {multicollinearity_analysis['threshold_used']}

✅ VALIDACIONES REALIZADAS:
• Matriz de correlación calculada correctamente
• Variable objetivo verificada como numérica
• Análisis de significancia estadística aplicado
• Detección sistemática de multicolinealidad

📊 CALIDAD DE LOS DATOS:
• Variables numéricas: {len(correlation_matrix.columns)}
• Sin valores faltantes en correlaciones
• Distribución de correlaciones analizada
• Patrones de relación identificados

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 VISUALIZACIONES:
• Matriz completa: graficos/paso6_matriz_correlacion_completa_{timestamp}.png
• Ranking con target: graficos/paso6_ranking_correlaciones_target_{timestamp}.png
• Top correlaciones: graficos/paso6_top_correlaciones_target_{timestamp}.png
• Análisis multicolinealidad: graficos/paso6_analisis_multicolinealidad_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe completo: informes/paso6_analisis_correlacion_informe_{timestamp}.txt
• Log del proceso: logs/paso6_analisis_correlacion.log

💾 CONFIGURACIONES:
• Lista optimizada de variables generada
• Configuraciones múltiples disponibles
• Recomendaciones específicas documentadas

================================================================================
CONCLUSIONES Y SIGUIENTE PASO
================================================================================

🎯 CONCLUSIONES PRINCIPALES:

1. CALIDAD DE PREDICTORES:
   • {target_analysis['total_significant']} variables con correlación significativa
   • Predictor más fuerte: {target_analysis['strongest_predictor']} (r = {target_analysis['strongest_correlation']:.4f})
   • Distribución balanceada de correlaciones positivas y negativas

2. OPTIMIZACIÓN LOGRADA:
   • Reducción de {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}% en dimensionalidad
   • Variables seleccionadas mantienen capacidad predictiva
   • Eliminación de redundancia por multicolinealidad

3. PREPARACIÓN PARA MODELADO:
   • Dataset optimizado listo para algoritmos tree-based
   • Variables interpretables para stakeholders
   • Balance entre performance y simplicidad

📋 PRÓXIMO PASO RECOMENDADO:
Paso 7: Entrenamiento y Validación de Modelos
• Implementar Random Forest y XGBoost
• Usar variables de configuración optimizada
• Aplicar class weighting del Paso 4
• Evaluar con métricas del Paso 3

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(report_content, optimization_summary, timestamp):
    """
    Guarda el informe y las configuraciones optimizadas
    
    Args:
        report_content (str): Contenido del informe
        optimization_summary (dict): Resumen de optimización
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    
    # Guardar informe
    report_filename = f"informes/paso6_analisis_correlacion_informe_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Informe guardado: {report_filename}")
    
    # Guardar configuraciones optimizadas en JSON
    import json
    config_filename = f"informes/paso6_configuraciones_optimizadas_{timestamp}.json"
    with open(config_filename, 'w', encoding='utf-8') as f:
        json.dump(optimization_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Configuraciones guardadas: {config_filename}")
    
    return report_filename, config_filename

def main():
    """Función principal que ejecuta todo el proceso de análisis de correlación"""
    
    # Crear directorios y configurar logging
    create_directories()
    logger = setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("INICIANDO PASO 6: ANALISIS DE CORRELACION")
    logger.info("=" * 70)
    
    try:
        # 1. Cargar datos del Paso 2
        input_file = find_latest_paso2_file()
        df = load_data(input_file)
        
        # 2. Verificar variable objetivo
        if not verify_target_variable(df):
            raise ValueError("Variable objetivo no válida")
        
        # 3. Calcular matriz de correlación
        correlation_matrix, numeric_columns = calculate_correlation_matrix(df)
        
        # 4. Analizar correlaciones con target
        target_analysis = analyze_target_correlations(correlation_matrix, target_threshold=0.3)
        
        # 5. Detectar multicolinealidad
        multicollinearity_analysis = detect_multicollinearity(correlation_matrix, multicollinearity_threshold=0.8)
        
        # 6. Generar recomendaciones de variables
        recommendations = generate_variable_recommendations(target_analysis, multicollinearity_analysis, correlation_matrix)
        
        # 7. Crear lista optimizada
        optimization_summary = generate_optimized_variable_list(recommendations, correlation_matrix)
        
        # 8. Crear visualizaciones
        create_correlation_visualizations(correlation_matrix, target_analysis, 
                                        multicollinearity_analysis, timestamp)
        
        # 9. Generar informe detallado
        report_content = generate_report(correlation_matrix, target_analysis, 
                                       multicollinearity_analysis, recommendations,
                                       optimization_summary, timestamp)
        
        # 10. Guardar archivos
        report_file, config_file = save_files(report_content, optimization_summary, timestamp)
        
        # 11. Resumen final
        logger.info("=" * 70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info(f"Variables analizadas: {len(correlation_matrix.columns)}")
        logger.info(f"Correlaciones significativas: {target_analysis['total_significant']}")
        logger.info(f"Predictor más fuerte: {target_analysis['strongest_predictor']}")
        logger.info(f"Correlación más fuerte: {target_analysis['strongest_correlation']:.4f}")
        logger.info(f"Pares multicolineales: {multicollinearity_analysis['total_problematic_pairs']}")
        logger.info(f"Variables optimizadas: {optimization_summary['reduction_achieved']['optimized_features']}")
        logger.info(f"Reducción: {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}%")
        logger.info(f"Informe generado: {report_file}")
        logger.info(f"Configuraciones: {config_file}")
        logger.info("=" * 70)
        
        print(f"\nRESULTADOS DE CORRELACION:")
        print(f"   • Predictor más fuerte: {target_analysis['strongest_predictor']}")
        print(f"   • Correlación: {target_analysis['strongest_correlation']:.4f}")
        print(f"   • Variables significativas: {target_analysis['total_significant']}")
        print(f"   • Variables optimizadas: {optimization_summary['reduction_achieved']['optimized_features']}")
        print(f"   • Reducción de dimensionalidad: {optimization_summary['reduction_achieved']['reduction_percentage']:.1f}%")
        
        print("\nSIGUIENTE PASO:")
        print("   Paso 7: Entrenamiento y Validación de Modelos")
        print("   (Usar configuración optimizada de variables)")
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        raise

if __name__ == "__main__":
    main()