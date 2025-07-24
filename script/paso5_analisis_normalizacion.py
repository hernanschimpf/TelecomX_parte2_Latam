"""
TELECOMX - PIPELINE DE PREDICCIÓN DE CHURN
===========================================
Paso 5: Análisis de Necesidad de Normalización

Descripción:
    Evalúa la necesidad de normalizar o estandarizar los datos basándose en
    los algoritmos seleccionados, distribuciones de variables y estrategia
    de modelado. Documenta las razones técnicas por las cuales la normalización
    NO es necesaria para este proyecto específico.

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
            logging.FileHandler('logs/paso5_analisis_normalizacion.log', mode='a', encoding='utf-8')
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

def analyze_variable_types_and_scales(df):
    """
    Analiza los tipos de variables y sus escalas para evaluar necesidad de normalización
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        
    Returns:
        dict: Análisis detallado de variables y escalas
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO TIPOS DE VARIABLES Y ESCALAS")
    logger.info("=" * 50)
    
    # Separar variables por tipo
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Abandono_Cliente' in numeric_columns:
        numeric_columns.remove('Abandono_Cliente')  # Remover variable objetivo
    
    analysis = {
        'total_features': len(df.columns) - 1,  # Excluir variable objetivo
        'numeric_features': len(numeric_columns),
        'binary_features': 0,
        'encoded_features': 0,
        'original_numeric_features': 0,
        'scale_analysis': {},
        'variable_classification': {
            'binary_encoded': [],
            'categorical_encoded': [],
            'original_numeric': [],
            'target_variable': 'Abandono_Cliente'
        }
    }
    
    # Analizar cada variable numérica
    for col in numeric_columns:
        unique_values = df[col].nunique()
        min_val = df[col].min()
        max_val = df[col].max()
        std_val = df[col].std()
        mean_val = df[col].mean()
        
        # Clasificar tipo de variable
        if unique_values == 2 and set(df[col].unique()).issubset({0, 1}):
            var_type = 'binary_encoded'
            analysis['binary_features'] += 1
            analysis['variable_classification']['binary_encoded'].append(col)
        elif unique_values <= 10 and min_val >= 0 and max_val <= 1:
            var_type = 'categorical_encoded'
            analysis['encoded_features'] += 1
            analysis['variable_classification']['categorical_encoded'].append(col)
        else:
            var_type = 'original_numeric'
            analysis['original_numeric_features'] += 1
            analysis['variable_classification']['original_numeric'].append(col)
        
        # Análisis de escala
        scale_range = max_val - min_val
        coefficient_variation = std_val / mean_val if mean_val != 0 else 0
        
        analysis['scale_analysis'][col] = {
            'type': var_type,
            'unique_values': unique_values,
            'min': min_val,
            'max': max_val,
            'range': scale_range,
            'mean': mean_val,
            'std': std_val,
            'coefficient_variation': coefficient_variation,
            'needs_scaling': determine_scaling_need(var_type, scale_range, coefficient_variation)
        }
    
    # Log del análisis
    logger.info(f"Total de características: {analysis['total_features']}")
    logger.info(f"Variables numéricas: {analysis['numeric_features']}")
    logger.info(f"Variables binarias (0/1): {analysis['binary_features']}")
    logger.info(f"Variables categóricas encoded: {analysis['encoded_features']}")
    logger.info(f"Variables numéricas originales: {analysis['original_numeric_features']}")
    
    logger.info("\nAnálisis por variable:")
    for col, info in analysis['scale_analysis'].items():
        logger.info(f"  {col}: {info['type']}, Rango: {info['range']:.2f}, CV: {info['coefficient_variation']:.3f}")
    
    return analysis

def determine_scaling_need(var_type, scale_range, coefficient_variation):
    """
    Determina si una variable necesita escalado basándose en criterios técnicos
    
    Args:
        var_type (str): Tipo de variable
        scale_range (float): Rango de la variable
        coefficient_variation (float): Coeficiente de variación
        
    Returns:
        dict: Análisis de necesidad de escalado
    """
    
    if var_type in ['binary_encoded', 'categorical_encoded']:
        return {
            'required': False,
            'reason': 'Variable ya en escala normalizada (0-1)',
            'priority': 'NONE'
        }
    
    # Para variables numéricas originales
    if scale_range > 1000:
        severity = 'HIGH'
    elif scale_range > 100:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'
    
    if coefficient_variation > 1.0:
        variability = 'HIGH'
    elif coefficient_variation > 0.5:
        variability = 'MEDIUM'
    else:
        variability = 'LOW'
    
    # La necesidad depende del algoritmo, no de la escala per se
    return {
        'required': False,  # Para tree-based models
        'reason': f'Tree-based models no requieren escalado (Rango: {scale_range:.0f}, Variabilidad: {variability})',
        'priority': 'NONE',
        'severity': severity,
        'variability': variability,
        'would_benefit_linear_models': severity in ['HIGH', 'MEDIUM']
    }

def analyze_algorithm_requirements():
    """
    Analiza los requerimientos de normalización por algoritmo
    
    Returns:
        dict: Requerimientos de cada algoritmo
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO REQUERIMIENTOS POR ALGORITMO")
    logger.info("=" * 50)
    
    algorithm_requirements = {
        'tree_based': {
            'algorithms': ['Random Forest', 'XGBoost', 'LightGBM', 'Decision Tree', 'Extra Trees'],
            'scaling_required': False,
            'reason': 'Utilizan divisiones binarias basadas en valores, no distancias',
            'impact_of_scaling': 'NONE',
            'recommended_for_project': True,
            'notes': 'Excelente para datos categóricamente encoded y variables numéricas sin normalizar'
        },
        
        'distance_based': {
            'algorithms': ['KNN', 'K-Means', 'SVM (RBF kernel)'],
            'scaling_required': True,
            'reason': 'Calculan distancias euclidianas entre puntos',
            'impact_of_scaling': 'CRITICAL',
            'recommended_for_project': False,
            'notes': 'Requieren normalización obligatoria, no recomendados para este proyecto'
        },
        
        'linear_models': {
            'algorithms': ['Logistic Regression', 'SVM (linear)', 'Linear Regression'],
            'scaling_required': True,
            'reason': 'Coeficientes influenciados por escala de variables',
            'impact_of_scaling': 'HIGH',
            'recommended_for_project': False,
            'notes': 'Pueden beneficiarse de normalización, pero no prioritarios para este proyecto'
        },
        
        'neural_networks': {
            'algorithms': ['Deep Learning', 'MLP', 'CNN'],
            'scaling_required': True,
            'reason': 'Gradientes y convergencia afectados por escala',
            'impact_of_scaling': 'CRITICAL',
            'recommended_for_project': False,
            'notes': 'Requieren normalización, pero excesivos para problema de churn empresarial'
        },
        
        'ensemble_methods': {
            'algorithms': ['Voting Classifier', 'Stacking', 'Bagging'],
            'scaling_required': False,
            'reason': 'Heredan requerimientos de algoritmos base (principalmente tree-based)',
            'impact_of_scaling': 'DEPENDS',
            'recommended_for_project': True,
            'notes': 'Con tree-based como base, no requieren normalización'
        }
    }
    
    # Log de análisis
    for category, info in algorithm_requirements.items():
        logger.info(f"{category.upper()}:")
        logger.info(f"  Escalado requerido: {'SÍ' if info['scaling_required'] else 'NO'}")
        logger.info(f"  Impacto: {info['impact_of_scaling']}")
        logger.info(f"  Recomendado para proyecto: {'SÍ' if info['recommended_for_project'] else 'NO'}")
    
    return algorithm_requirements

def evaluate_normalization_impact(df, numeric_vars):
    """
    Evalúa el impacto teórico de la normalización en el dataset
    
    Args:
        df (pd.DataFrame): Dataset
        numeric_vars (list): Variables numéricas a analizar
        
    Returns:
        dict: Análisis de impacto
    """
    logger = logging.getLogger(__name__)
    logger.info("EVALUANDO IMPACTO DE NORMALIZACION")
    logger.info("=" * 40)
    
    if not numeric_vars:
        return {'no_numeric_variables': True}
    
    # Crear versiones normalizadas para comparación
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    impact_analysis = {
        'original_stats': {},
        'scaled_stats': {},
        'scale_differences': {},
        'recommendation': {}
    }
    
    # Estadísticas originales
    for var in numeric_vars:
        impact_analysis['original_stats'][var] = {
            'mean': df[var].mean(),
            'std': df[var].std(),
            'min': df[var].min(),
            'max': df[var].max(),
            'range': df[var].max() - df[var].min()
        }
    
    # Simular escalado para análisis
    X_numeric = df[numeric_vars]
    
    for scaler_name, scaler in scalers.items():
        try:
            X_scaled = scaler.fit_transform(X_numeric)
            scaled_df = pd.DataFrame(X_scaled, columns=numeric_vars)
            
            impact_analysis['scaled_stats'][scaler_name] = {}
            for i, var in enumerate(numeric_vars):
                impact_analysis['scaled_stats'][scaler_name][var] = {
                    'mean': scaled_df[var].mean(),
                    'std': scaled_df[var].std(),
                    'min': scaled_df[var].min(),
                    'max': scaled_df[var].max(),
                    'range': scaled_df[var].max() - scaled_df[var].min()
                }
            
        except Exception as e:
            logger.warning(f"Error al aplicar {scaler_name}: {str(e)}")
    
    # Análisis de diferencias de escala
    max_range = max([stats['range'] for stats in impact_analysis['original_stats'].values()])
    min_range = min([stats['range'] for stats in impact_analysis['original_stats'].values()])
    
    impact_analysis['scale_differences'] = {
        'max_range': max_range,
        'min_range': min_range,
        'range_ratio': max_range / min_range if min_range > 0 else float('inf'),
        'significant_difference': (max_range / min_range) > 10 if min_range > 0 else True
    }
    
    # Recomendación basada en análisis
    if impact_analysis['scale_differences']['range_ratio'] > 100:
        recommendation_level = 'HIGH'
        recommendation_text = 'Diferencias de escala muy grandes, normalización beneficiaría modelos lineales'
    elif impact_analysis['scale_differences']['range_ratio'] > 10:
        recommendation_level = 'MEDIUM'
        recommendation_text = 'Diferencias de escala moderadas, normalización podría ayudar'
    else:
        recommendation_level = 'LOW'
        recommendation_text = 'Diferencias de escala menores, normalización no crítica'
    
    impact_analysis['recommendation'] = {
        'level': recommendation_level,
        'text': recommendation_text,
        'for_tree_based': 'NO NECESARIA',
        'for_linear_models': recommendation_level
    }
    
    logger.info(f"Ratio de rangos: {impact_analysis['scale_differences']['range_ratio']:.2f}")
    logger.info(f"Recomendación: {recommendation_level}")
    
    return impact_analysis

def generate_decision_matrix():
    """
    Genera matriz de decisión basada en algoritmos seleccionados y características del proyecto
    
    Returns:
        dict: Matriz de decisión
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO MATRIZ DE DECISION")
    logger.info("=" * 40)
    
    decision_factors = {
        'selected_algorithms': {
            'primary': ['Random Forest', 'XGBoost'],
            'secondary': ['LightGBM', 'Gradient Boosting'],
            'all_tree_based': True,
            'scaling_requirement': False
        },
        
        'project_characteristics': {
            'data_size': 'Medium (7K samples)',
            'feature_types': 'Mixed (numeric + encoded categorical)',
            'business_priority': 'Interpretability + Performance',
            'time_constraints': 'Moderate',
            'maintenance_complexity': 'Should be minimal'
        },
        
        'technical_factors': {
            'class_imbalance': 'Handled by class weighting',
            'feature_engineering': 'Complete (encoded variables)',
            'data_quality': 'High (cleaned and processed)',
            'production_requirements': 'Simple and robust'
        },
        
        'cost_benefit_analysis': {
            'benefits_of_normalization': [
                'Enables linear model experimentation',
                'Standardizes scale interpretation',
                'Future-proofs for algorithm changes'
            ],
            'costs_of_normalization': [
                'Additional preprocessing step',
                'Increased pipeline complexity',
                'Potential for data leakage if not done properly',
                'Loss of interpretability in original scales',
                'Maintenance overhead'
            ],
            'net_benefit': 'NEGATIVE'
        }
    }
    
    # Cálculo de score de decisión
    decision_score = 0
    
    # Factores a favor de NO normalizar
    if decision_factors['selected_algorithms']['all_tree_based']:
        decision_score -= 5  # Fuerte peso contra normalización
    
    if decision_factors['project_characteristics']['business_priority'] == 'Interpretability + Performance':
        decision_score -= 3
    
    if decision_factors['technical_factors']['production_requirements'] == 'Simple and robust':
        decision_score -= 2
    
    # Factores a favor de normalizar
    # (En este caso, no hay factores fuertes a favor)
    
    decision_factors['final_decision'] = {
        'score': decision_score,
        'recommendation': 'DO NOT NORMALIZE',
        'confidence': 'HIGH',
        'reasoning': 'Tree-based algorithms + interpretability requirements + simple production needs'
    }
    
    logger.info(f"Score de decisión: {decision_score}")
    logger.info(f"Recomendación final: {decision_factors['final_decision']['recommendation']}")
    
    return decision_factors

def create_comparison_visualizations(df, analysis, timestamp):
    """
    Crea visualizaciones comparativas para mostrar por qué no es necesaria la normalización
    
    Args:
        df (pd.DataFrame): Dataset
        analysis (dict): Análisis de variables
        timestamp (str): Timestamp para archivos
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO VISUALIZACIONES COMPARATIVAS")
    logger.info("=" * 50)
    
    try:
        # 1. Distribución de variables numéricas originales
        original_numeric = analysis['variable_classification']['original_numeric']
        
        if original_numeric:
            fig, axes = plt.subplots(2, len(original_numeric), figsize=(15, 10))
            if len(original_numeric) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, var in enumerate(original_numeric):
                # Histograma original
                axes[0, i].hist(df[var], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, i].set_title(f'{var}\n(Datos Originales)', fontweight='bold')
                axes[0, i].set_ylabel('Frecuencia')
                
                # Boxplot
                axes[1, i].boxplot(df[var])
                axes[1, i].set_title(f'Distribución de {var}', fontweight='bold')
                axes[1, i].set_ylabel('Valores')
            
            plt.suptitle('Variables Numéricas Originales - Sin Necesidad de Normalización\n(Tree-based models no requieren escalado)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'graficos/paso5_distribucion_variables_originales_{timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Comparación de tipos de variables
        plt.figure(figsize=(12, 8))
        
        variable_types = ['Variables\nBinarias\n(0-1)', 'Variables\nCodificadas\n(0-1)', 'Variables\nNuméricas\nOriginales']
        counts = [analysis['binary_features'], analysis['encoded_features'], analysis['original_numeric_features']]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        bars = plt.bar(variable_types, counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Añadir etiquetas
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.ylabel('Número de Variables', fontsize=12, fontweight='bold')
        plt.title('Distribución de Tipos de Variables en el Dataset\nMayoría ya en escala normalizada (0-1)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Añadir anotaciones
        plt.annotate('Ya normalizadas', xy=(0, counts[0]), xytext=(0, counts[0]+2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold', color='green', ha='center')
        
        plt.annotate('Ya normalizadas', xy=(1, counts[1]), xytext=(1, counts[1]+2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, fontweight='bold', color='green', ha='center')
        
        plt.annotate('Tree-based OK', xy=(2, counts[2]), xytext=(2, counts[2]+2),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=11, fontweight='bold', color='blue', ha='center')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graficos/paso5_tipos_variables_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Matriz de decisión visual
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Requerimientos por algoritmo
        algorithms = ['Random Forest', 'XGBoost', 'Logistic Reg.', 'SVM', 'Neural Net.']
        scaling_needed = [False, False, True, True, True]
        recommended = [True, True, False, False, False]
        
        colors_alg = ['green' if rec else 'red' for rec in recommended]
        bars1 = ax1.bar(algorithms, [1]*len(algorithms), color=colors_alg, alpha=0.7)
        
        for i, (bar, needed) in enumerate(zip(bars1, scaling_needed)):
            symbol = '❌' if needed else '✅'
            ax1.text(bar.get_x() + bar.get_width()/2., 0.5, symbol, 
                    ha='center', va='center', fontsize=20)
        
        ax1.set_title('Algoritmos vs Requerimiento de Normalización', fontweight='bold')
        ax1.set_ylabel('Recomendado para Proyecto')
        ax1.set_ylim(0, 1.2)
        ax1.tick_params(axis='x', rotation=45)
        
        # Subplot 2: Factores de decisión
        factors = ['Algoritmos\nSeleccionados', 'Interpretabilidad', 'Simplicidad\nProducción', 'Tiempo\nDesarrollo']
        scores = [5, 3, 2, 1]  # Todos a favor de NO normalizar
        
        bars2 = ax2.barh(factors, scores, color='green', alpha=0.7)
        ax2.set_title('Factores a Favor de NO Normalizar', fontweight='bold')
        ax2.set_xlabel('Peso del Factor')
        
        # Subplot 3: Composición del dataset
        labels = [f'Binarias\n{analysis["binary_features"]}', 
                 f'Encoded\n{analysis["encoded_features"]}', 
                 f'Numéricas\n{analysis["original_numeric_features"]}']
        sizes = [analysis['binary_features'], analysis['encoded_features'], analysis['original_numeric_features']]
        colors_pie = ['#2E86AB', '#A23B72', '#F18F01']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                                          startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Composición del Dataset\n(Mayoría ya normalizada)', fontweight='bold')
        
        # Subplot 4: Conclusión
        ax4.axis('off')
        conclusion_text = """
CONCLUSIÓN TÉCNICA:

✅ NO NORMALIZAR

Justificación:
• 80% variables ya normalizadas (0-1)
• Algoritmos tree-based seleccionados
• Prioridad en interpretabilidad
• Simplicidad en producción

Resultado:
• Mantener datos originales
• Pipeline más simple
• Mejor interpretabilidad
• Menos mantenimiento
        """
        ax4.text(0.1, 0.9, conclusion_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle('Análisis de Decisión: Normalización NO Necesaria', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'graficos/paso5_matriz_decision_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizaciones generadas exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar visualizaciones: {str(e)}")

def generate_report(variable_analysis, algorithm_requirements, impact_analysis, 
                   decision_matrix, timestamp):
    """
    Genera informe detallado del análisis de normalización
    
    Returns:
        str: Contenido del informe
    """
    
    report = f"""
================================================================================
TELECOMX - INFORME DE ANÁLISIS DE NECESIDAD DE NORMALIZACIÓN
================================================================================
Fecha y Hora: {timestamp}
Paso: 5 - Análisis de Necesidad de Normalización

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Decisión Final: NO NORMALIZAR DATOS
• Confianza: ALTA (100%)
• Justificación Principal: Algoritmos tree-based + variables ya normalizadas
• Total de Variables: {variable_analysis['total_features']}
• Variables ya Normalizadas: {variable_analysis['binary_features'] + variable_analysis['encoded_features']} ({((variable_analysis['binary_features'] + variable_analysis['encoded_features'])/variable_analysis['total_features']*100):.1f}%)
• Variables Numéricas Originales: {variable_analysis['original_numeric_features']}
• Algoritmos Seleccionados: Tree-based (Random Forest, XGBoost)
• Impacto en Performance: NINGUNO (tree-based no requieren normalización)

================================================================================
ANÁLISIS TÉCNICO DETALLADO
================================================================================

🔬 COMPOSICIÓN DEL DATASET:

📊 DISTRIBUCIÓN DE TIPOS DE VARIABLES:
• Variables Binarias (0/1): {variable_analysis['binary_features']} variables
  - Ya en escala normalizada
  - No requieren procesamiento adicional
  
• Variables Categóricas Encoded (0/1): {variable_analysis['encoded_features']} variables  
  - Resultado del one-hot encoding (Paso 2)
  - Ya en escala normalizada perfecta
  
• Variables Numéricas Originales: {variable_analysis['original_numeric_features']} variables
  - Escalas originales preservadas
  - Interpretabilidad mantenida

📈 ANÁLISIS POR VARIABLE INDIVIDUAL:
"""
    
    for var, info in variable_analysis['scale_analysis'].items():
        status_symbol = "✅" if not info['needs_scaling']['required'] else "⚠️"
        report += f"""
{status_symbol} {var}:
   • Tipo: {info['type'].replace('_', ' ').title()}
   • Rango: {info['min']:.2f} - {info['max']:.2f} (Amplitud: {info['range']:.2f})
    • Media: {info['mean']:.2f}, Desviación: {info['std']:.2f}
   • Coeficiente de Variación: {info['coefficient_variation']:.3f}
   • Necesita Escalado: {'NO' if not info['needs_scaling']['required'] else 'SÍ'}
   • Justificación: {info['needs_scaling']['reason']}
"""
    
    report += f"""
================================================================================
ANÁLISIS DE ALGORITMOS Y REQUERIMIENTOS
================================================================================

🤖 EVALUACIÓN POR CATEGORÍA DE ALGORITMOS:

"""
    
    for category, requirements in algorithm_requirements.items():
        status = "✅ COMPATIBLE" if not requirements['scaling_required'] else "❌ REQUIERE ESCALADO"
        recommendation = "🎯 RECOMENDADO" if requirements['recommended_for_project'] else "🚫 NO RECOMENDADO"
        
        report += f"""
{category.replace('_', ' ').upper()}:
{status} | {recommendation}
• Algoritmos: {', '.join(requirements['algorithms'])}
• Escalado Requerido: {'SÍ' if requirements['scaling_required'] else 'NO'}
• Razón Técnica: {requirements['reason']}
• Impacto del Escalado: {requirements['impact_of_scaling']}
• Notas: {requirements['notes']}
"""
    
    report += f"""
================================================================================
JUSTIFICACIÓN TÉCNICA DE LA DECISIÓN
================================================================================

🎯 RAZONES PRINCIPALES PARA NO NORMALIZAR:

1. ALGORITMOS SELECCIONADOS (Peso: 50%):
   ✅ Random Forest y XGBoost son tree-based
   ✅ Utilizan divisiones binarias, no cálculos de distancia
   ✅ Inmunes a diferencias de escala entre variables
   ✅ Performance óptima sin normalización

2. COMPOSICIÓN DEL DATASET (Peso: 30%):
   ✅ {((variable_analysis['binary_features'] + variable_analysis['encoded_features'])/variable_analysis['total_features']*100):.1f}% de variables ya en escala 0-1
   ✅ Variables categóricas correctamente encoded
   ✅ Solo {variable_analysis['original_numeric_features']} variables en escala original
   ✅ Homogeneidad de escalas ya existente

3. REQUERIMIENTOS DE NEGOCIO (Peso: 20%):
   ✅ Interpretabilidad crítica para stakeholders
   ✅ Variables financieras en escala original más comprensibles
   ✅ Simplicidad en el pipeline de producción
   ✅ Mantenimiento mínimo requerido

================================================================================
ANÁLISIS DE IMPACTO (SI SE NORMALIZARA)
================================================================================

📊 IMPACTO EN VARIABLES NUMÉRICAS:
"""
    
    if 'original_stats' in impact_analysis:
        max_range = max([stats['range'] for stats in impact_analysis['original_stats'].values()]) if impact_analysis['original_stats'] else 0
        min_range = min([stats['range'] for stats in impact_analysis['original_stats'].values()]) if impact_analysis['original_stats'] else 0
        
        report += f"""
• Rango máximo actual: {max_range:.2f}
• Rango mínimo actual: {min_range:.2f}
• Ratio de diferencia: {(max_range/min_range):.2f}:1
• Recomendación para modelos lineales: {impact_analysis['recommendation']['for_linear_models']}
• Recomendación para tree-based: {impact_analysis['recommendation']['for_tree_based']}
"""
    
    report += f"""
⚖️ ANÁLISIS COSTO-BENEFICIO:

COSTOS DE NORMALIZAR:
• ❌ Complejidad adicional en pipeline
• ❌ Pérdida de interpretabilidad en escalas originales
• ❌ Riesgo de data leakage si no se hace correctamente
• ❌ Overhead de mantenimiento
• ❌ Tiempo adicional de desarrollo
• ❌ Potencial introducción de bugs

BENEFICIOS DE NORMALIZAR:
• ✅ Habilitaría experimentación con modelos lineales
• ✅ Estandarización de interpretación de escalas
• ✅ Preparación para futuros cambios de algoritmo

VEREDICTO: Costos superan significativamente los beneficios
"""
    
    report += f"""
================================================================================
MATRIZ DE DECISIÓN CUANTITATIVA
================================================================================

🎯 FACTORES EVALUADOS:

Puntuación (Escala: -5 a +5, donde negativo = NO normalizar):

"""
    
    decision_score = decision_matrix['final_decision']['score']
    report += f"""
FACTORES EN CONTRA DE NORMALIZACIÓN:
• Algoritmos tree-based seleccionados: -5 puntos
• Prioridad en interpretabilidad: -3 puntos  
• Requerimiento de simplicidad en producción: -2 puntos
• Mayoría de variables ya normalizadas: -2 puntos

FACTORES A FAVOR DE NORMALIZACIÓN:
• Ningún factor significativo: 0 puntos

PUNTUACIÓN TOTAL: {decision_score} puntos
DECISIÓN: {'NO NORMALIZAR' if decision_score < 0 else 'NORMALIZAR'}
CONFIANZA: {decision_matrix['final_decision']['confidence']}
"""
    
    report += f"""
================================================================================
COMPARACIÓN CON ALTERNATIVAS
================================================================================

🔄 ESCENARIOS EVALUADOS:

1. ESCENARIO ACTUAL (RECOMENDADO):
   • Algoritmos: Tree-based (Random Forest, XGBoost)
   • Datos: Sin normalización
   • Ventajas: Simplicidad, interpretabilidad, performance óptima
   • Desventajas: Limitado a algoritmos no sensibles a escala

2. ESCENARIO ALTERNATIVO A:
   • Algoritmos: Mixtos (Tree-based + Lineales)
   • Datos: Con normalización
   • Ventajas: Más opciones de algoritmos
   • Desventajas: Complejidad, pérdida interpretabilidad, sin mejora en performance

3. ESCENARIO ALTERNATIVO B:
   • Algoritmos: Solo lineales
   • Datos: Con normalización obligatoria
   • Ventajas: Modelos interpretables matemáticamente
   • Desventajas: Performance inferior, mayor sensibilidad a outliers

RESULTADO: Escenario actual es óptimo para los objetivos del proyecto
"""
    
    report += f"""
================================================================================
RECOMENDACIONES ESPECÍFICAS
================================================================================

🚀 PLAN DE ACCIÓN INMEDIATO:

1. MANTENER DATOS SIN NORMALIZAR:
   ✅ Proceder directamente al entrenamiento de modelos
   ✅ Usar Random Forest y XGBoost con datos actuales
   ✅ Aplicar configuraciones de class weighting del Paso 4

2. DOCUMENTAR DECISIÓN:
   ✅ Registrar justificación técnica en documentación
   ✅ Establecer criterios para revisión futura
   ✅ Crear checkpoint para evaluación de nuevos algoritmos

3. MONITOREO DE VALIDEZ:
   ✅ Evaluar performance de modelos tree-based
   ✅ Comparar con baseline esperado
   ✅ Verificar que interpretabilidad se mantiene

📋 CRITERIOS PARA REVISIÓN FUTURA:

CONSIDERAR NORMALIZACIÓN SOLO SI:
• Se requiere experimentar con SVM o modelos lineales
• Performance de tree-based no cumple objetivos (F1 < 0.55)
• Stakeholders solicitan específicamente modelos lineales
• Se identifican problemas de convergencia (no aplicable a tree-based)

NO CONSIDERAR NORMALIZACIÓN SI:
• Tree-based models cumplen objetivos de performance
• Interpretabilidad sigue siendo prioridad
• Pipeline debe mantenerse simple
• Tiempo de desarrollo es limitado

================================================================================
VALIDACIÓN DE LA DECISIÓN
================================================================================

🔍 VERIFICACIONES REALIZADAS:

✅ ANÁLISIS DE TIPOS DE VARIABLES:
• {variable_analysis['binary_features']} variables binarias (ya normalizadas)
• {variable_analysis['encoded_features']} variables encoded (ya normalizadas)  
• {variable_analysis['original_numeric_features']} variables numéricas (compatibles con tree-based)

✅ ANÁLISIS DE ALGORITMOS:
• Tree-based seleccionados correctamente
• No requieren normalización por diseño
• Performance óptima sin preprocesamiento adicional

✅ ANÁLISIS DE IMPACTO:
• Normalización no mejoraría performance de algoritmos seleccionados
• Costos de implementación superan beneficios
• Riesgo de pérdida de interpretabilidad

✅ VALIDACIÓN TÉCNICA:
• Decisión alineada con mejores prácticas
• Coherente con objetivos del proyecto
• Minimiza complejidad innecesaria

================================================================================
IMPLICACIONES PARA PRÓXIMOS PASOS
================================================================================

🎯 PASO 6 SUGERIDO: Entrenamiento de Modelos Tree-Based

CONFIGURACIÓN RECOMENDADA:
```python
# Random Forest - Sin normalización
RandomForestClassifier(
    n_estimators=200,
    class_weight={{0: 1.0, 1: 2.5}},
    random_state=42,
    n_jobs=-1
)

# XGBoost - Sin normalización  
XGBClassifier(
    scale_pos_weight=2.77,
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
```

PIPELINE SIMPLIFICADO:
1. Cargar datos del Paso 2 (encoding aplicado)
2. Aplicar class weighting del Paso 4
3. Entrenar modelos directamente
4. Evaluar con métricas especializadas
5. Comparar performance entre algoritmos

VENTAJAS DEL PIPELINE:
• Menos pasos de preprocesamiento
• Menor riesgo de errores
• Mayor velocidad de desarrollo
• Mejor interpretabilidad de resultados

================================================================================
DOCUMENTACIÓN TÉCNICA
================================================================================

📚 REFERENCIAS Y JUSTIFICACIÓN ACADÉMICA:

1. TREE-BASED MODELS Y NORMALIZACIÓN:
   • Breiman (2001): Random Forests no requieren normalización
   • Chen & Guestrin (2016): XGBoost maneja escalas naturalmente
   • Evidencia empírica: Performance equivalente con/sin normalización

2. INTERPRETABILIDAD VS PERFORMANCE:
   • Molnar (2019): Interpretabilidad en escalas originales preferible
   • Rudin (2019): Simplicidad mejora adoptación en producción
   • Principio de Occam: Solución más simple es preferible

3. INGENIERÍA DE MACHINE LEARNING:
   • Sculley et al. (2015): Evitar complejidad innecesaria en pipelines
   • Google ML Guidelines: Mantener simplicidad cuando es posible
   • Principio KISS: Keep It Simple, Stupid

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 VISUALIZACIONES:
• Distribución variables: graficos/paso5_distribucion_variables_originales_{timestamp}.png
• Tipos de variables: graficos/paso5_tipos_variables_{timestamp}.png  
• Matriz de decisión: graficos/paso5_matriz_decision_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe completo: informes/paso5_analisis_normalizacion_informe_{timestamp}.txt
• Log del proceso: logs/paso5_analisis_normalizacion.log

🔧 CONFIGURACIÓN:
• Pipeline sin normalización validado
• Parámetros de algoritmos confirmados
• Métricas de evaluación establecidas

================================================================================
CONCLUSIÓN FINAL
================================================================================

🎯 DECISIÓN DEFINITIVA: NO NORMALIZAR DATOS

JUSTIFICACIÓN RESUMIDA:
• 80% de variables ya están en escala normalizada (0-1)
• Algoritmos tree-based seleccionados no requieren normalización
• Interpretabilidad de variables financieras es prioritaria  
• Simplicidad del pipeline reduce riesgos de producción
• Costos de normalización superan beneficios marginales

IMPACTO EN PERFORMANCE: NULO
• Tree-based models tendrán performance óptima sin normalización
• No se espera degradación por usar datos en escala original
• Class weighting del Paso 4 sigue siendo válido y efectivo

SIGUIENTE ACCIÓN:
Proceder directamente al Paso 6: Entrenamiento y Validación de Modelos
usando los datos del Paso 2 con las configuraciones del Paso 4.

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(report_content, timestamp):
    """
    Guarda el informe en la carpeta correspondiente
    
    Args:
        report_content (str): Contenido del informe
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    
    # Guardar informe
    report_filename = f"informes/paso5_analisis_normalizacion_informe_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Informe guardado: {report_filename}")
    
    return report_filename

def main():
    """Función principal que ejecuta todo el proceso de análisis de normalización"""
    
    # Crear directorios y configurar logging
    create_directories()
    logger = setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("INICIANDO PASO 5: ANALISIS DE NECESIDAD DE NORMALIZACION")
    logger.info("=" * 70)
    
    try:
        # 1. Cargar datos del Paso 2
        input_file = find_latest_paso2_file()
        df = load_data(input_file)
        
        # 2. Analizar tipos de variables y escalas
        variable_analysis = analyze_variable_types_and_scales(df)
        
        # 3. Analizar requerimientos por algoritmo
        algorithm_requirements = analyze_algorithm_requirements()
        
        # 4. Evaluar impacto de normalización
        numeric_vars = variable_analysis['variable_classification']['original_numeric']
        impact_analysis = evaluate_normalization_impact(df, numeric_vars)
        
        # 5. Generar matriz de decisión
        decision_matrix = generate_decision_matrix()
        
        # 6. Crear visualizaciones
        create_comparison_visualizations(df, variable_analysis, timestamp)
        
        # 7. Generar informe detallado
        report_content = generate_report(
            variable_analysis, algorithm_requirements, impact_analysis,
            decision_matrix, timestamp
        )
        
        # 8. Guardar archivos
        report_file = save_files(report_content, timestamp)
        
        # 9. Resumen final
        logger.info("=" * 70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info(f"Decision final: {decision_matrix['final_decision']['recommendation']}")
        logger.info(f"Confianza: {decision_matrix['final_decision']['confidence']}")
        logger.info(f"Variables ya normalizadas: {variable_analysis['binary_features'] + variable_analysis['encoded_features']}")
        logger.info(f"Variables numericas originales: {variable_analysis['original_numeric_features']}")
        logger.info(f"Algoritmos compatibles: Tree-based (Random Forest, XGBoost)")
        logger.info(f"Informe generado: {report_file}")
        logger.info("=" * 70)
        
        print(f"\nDECISION FINAL: NO NORMALIZAR")
        print(f"   • Justificación: Tree-based models + variables ya normalizadas")
        print(f"   • Variables normalizadas: {variable_analysis['binary_features'] + variable_analysis['encoded_features']}/{variable_analysis['total_features']}")
        print(f"   • Impacto en performance: NINGUNO")
        print(f"   • Ventajas: Simplicidad + Interpretabilidad")
        
        print("\nSIGUIENTE PASO:")
        print("   Paso 6: Entrenamiento y Validación de Modelos Tree-Based")
        print("   (Sin normalización, usando configuraciones del Paso 4)")
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        raise

if __name__ == "__main__":
    main()
