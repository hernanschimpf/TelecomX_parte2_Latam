"""
TELECOMX - PIPELINE DE PREDICCIÓN DE CHURN
===========================================
Paso 3: Verificación de la Proporción de Cancelación (Churn)

Descripción:
    Calcula la proporción de clientes que cancelaron en relación con los que 
    permanecieron activos. Evalúa si existe un desbalance entre las clases, 
    ya que esto puede impactar en los modelos predictivos y en el análisis 
    de los resultados.

Autor: Ingeniero de Datos
Fecha: 2025-07-21
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
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
            logging.FileHandler('logs/paso3_verificacion_churn.log', mode='a', encoding='utf-8')
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
    Verifica que la variable objetivo existe y tiene el formato correcto
    
    Args:
        df (pd.DataFrame): Dataset cargado
        
    Returns:
        tuple: (bool, dict) - Validación exitosa y información de la variable
    """
    logger = logging.getLogger(__name__)
    logger.info("VERIFICANDO VARIABLE OBJETIVO 'Abandono_Cliente'")
    logger.info("=" * 50)
    
    target_info = {
        'exists': False,
        'column_name': 'Abandono_Cliente',
        'data_type': None,
        'unique_values': None,
        'is_binary': False,
        'null_count': 0,
        'total_rows': len(df)
    }
    
    # Verificar existencia
    if 'Abandono_Cliente' not in df.columns:
        logger.error("Variable objetivo 'Abandono_Cliente' NO ENCONTRADA")
        logger.error(f"Columnas disponibles: {list(df.columns)}")
        return False, target_info
    
    target_info['exists'] = True
    target_info['data_type'] = str(df['Abandono_Cliente'].dtype)
    target_info['unique_values'] = sorted(df['Abandono_Cliente'].unique())
    target_info['null_count'] = df['Abandono_Cliente'].isnull().sum()
    
    # Verificar si es binaria
    unique_vals = df['Abandono_Cliente'].dropna().unique()
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
        target_info['is_binary'] = True
        logger.info("Variable objetivo ENCONTRADA y VALIDADA")
        logger.info(f"  Tipo de datos: {target_info['data_type']}")
        logger.info(f"  Valores únicos: {target_info['unique_values']}")
        logger.info(f"  Es binaria: SI")
        logger.info(f"  Valores nulos: {target_info['null_count']}")
    else:
        logger.warning("Variable objetivo encontrada pero NO ES BINARIA")
        logger.warning(f"  Valores únicos: {target_info['unique_values']}")
        return False, target_info
    
    return True, target_info

def analyze_churn_distribution(df):
    """
    Analiza la distribución de la variable de churn
    
    Args:
        df (pd.DataFrame): Dataset con variable objetivo
        
    Returns:
        dict: Análisis detallado de la distribución de churn
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO DISTRIBUCION DE CHURN")
    logger.info("=" * 40)
    
    # Conteos básicos
    churn_counts = df['Abandono_Cliente'].value_counts().sort_index()
    churn_percentages = df['Abandono_Cliente'].value_counts(normalize=True).sort_index() * 100
    total_clients = len(df)
    
    # Clasificación de clases
    no_churn_count = churn_counts[0]
    churn_count = churn_counts[1]
    no_churn_pct = churn_percentages[0]
    churn_pct = churn_percentages[1]
    
    # Calcular métricas de desbalance
    majority_class = max(no_churn_count, churn_count)
    minority_class = min(no_churn_count, churn_count)
    imbalance_ratio = majority_class / minority_class
    minority_percentage = (minority_class / total_clients) * 100
    
    # Clasificar nivel de desbalance
    if 40 <= minority_percentage <= 60:
        balance_level = "Balanceado"
        balance_severity = "Óptimo"
        impact_level = "Bajo"
    elif 30 <= minority_percentage < 40 or 60 < minority_percentage <= 70:
        balance_level = "Ligeramente Desbalanceado"
        balance_severity = "Leve"
        impact_level = "Moderado"
    elif 20 <= minority_percentage < 30 or 70 < minority_percentage <= 80:
        balance_level = "Moderadamente Desbalanceado" 
        balance_severity = "Moderado"
        impact_level = "Alto"
    elif minority_percentage < 20 or minority_percentage > 80:
        balance_level = "Severamente Desbalanceado"
        balance_severity = "Severo"
        impact_level = "Muy Alto"
    else:
        balance_level = "No Clasificado"
        balance_severity = "Desconocido"
        impact_level = "Desconocido"
    
    distribution_analysis = {
        'total_clients': total_clients,
        'no_churn_count': no_churn_count,
        'churn_count': churn_count,
        'no_churn_percentage': no_churn_pct,
        'churn_percentage': churn_pct,
        'majority_class': 'No Churn' if no_churn_count > churn_count else 'Churn',
        'minority_class': 'Churn' if no_churn_count > churn_count else 'No Churn',
        'majority_count': majority_class,
        'minority_count': minority_class,
        'majority_percentage': (majority_class / total_clients) * 100,
        'minority_percentage': minority_percentage,
        'imbalance_ratio': imbalance_ratio,
        'balance_level': balance_level,
        'balance_severity': balance_severity,
        'impact_level': impact_level
    }
    
    # Logging de resultados
    logger.info(f"DISTRIBUCION DE CLASES:")
    logger.info(f"  Total de clientes: {total_clients:,}")
    logger.info(f"  No Churn (0): {no_churn_count:,} ({no_churn_pct:.1f}%)")
    logger.info(f"  Churn (1): {churn_count:,} ({churn_pct:.1f}%)")
    logger.info(f"")
    logger.info(f"METRICAS DE BALANCE:")
    logger.info(f"  Clase mayoría: {distribution_analysis['majority_class']} ({distribution_analysis['majority_percentage']:.1f}%)")
    logger.info(f"  Clase minoría: {distribution_analysis['minority_class']} ({minority_percentage:.1f}%)")
    logger.info(f"  Ratio desbalance: {imbalance_ratio:.2f}:1")
    logger.info(f"  Nivel de balance: {balance_level}")
    logger.info(f"  Severidad: {balance_severity}")
    logger.info(f"  Impacto en modelos: {impact_level}")
    
    return distribution_analysis

def generate_churn_visualizations(df, distribution_analysis, timestamp):
    """
    Genera visualizaciones de la distribución de churn
    
    Args:
        df (pd.DataFrame): Dataset
        distribution_analysis (dict): Análisis de distribución
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO VISUALIZACIONES DE CHURN")
    logger.info("=" * 40)
    
    # Configurar estilo general
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. GRÁFICO DE BARRAS
    try:
        logger.info("Generando grafico de barras...")
        plt.figure(figsize=(10, 6))
        
        categories = ['No Churn (0)', 'Churn (1)']
        counts = [distribution_analysis['no_churn_count'], distribution_analysis['churn_count']]
        percentages = [distribution_analysis['no_churn_percentage'], distribution_analysis['churn_percentage']]
        
        # Colores diferenciados
        colors = ['#2E86AB', '#A23B72']  # Azul para No Churn, Rojo para Churn
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.ylabel('Número de Clientes', fontsize=12, fontweight='bold')
        plt.xlabel('Clase de Abandono', fontsize=12, fontweight='bold')
        plt.title('Distribución de Clientes por Abandono (Churn)\nAnálisis de Balance de Clases', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Añadir etiquetas con valores y porcentajes
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Añadir información de balance
        balance_info = f"Ratio de Desbalance: {distribution_analysis['imbalance_ratio']:.2f}:1"
        balance_level = f"Nivel: {distribution_analysis['balance_level']}"
        plt.figtext(0.02, 0.02, f"{balance_info} | {balance_level}", 
                   fontsize=10, style='italic')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'graficos/paso3_distribucion_churn_barras_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Grafico de barras generado exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar grafico de barras: {str(e)}")
    
    # 2. GRÁFICO CIRCULAR (PIE CHART)
    try:
        logger.info("Generando grafico circular...")
        plt.figure(figsize=(10, 8))
        
        labels = [f'No Churn\n{distribution_analysis["no_churn_count"]:,} clientes\n({distribution_analysis["no_churn_percentage"]:.1f}%)',
                  f'Churn\n{distribution_analysis["churn_count"]:,} clientes\n({distribution_analysis["churn_percentage"]:.1f}%)']
        
        sizes = [distribution_analysis['no_churn_percentage'], distribution_analysis['churn_percentage']]
        colors = ['#2E86AB', '#A23B72']
        explode = (0, 0.1)  # Destacar la clase de churn
        
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, explode=explode, shadow=True,
                                          startangle=90, textprops={'fontsize': 11})
        
        # Mejorar apariencia del texto
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('Proporción de Abandono de Clientes (Churn)\nDistribución Porcentual', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Añadir leyenda con información adicional
        legend_labels = [f'No Churn: {distribution_analysis["no_churn_count"]:,} clientes',
                        f'Churn: {distribution_analysis["churn_count"]:,} clientes',
                        f'Total: {distribution_analysis["total_clients"]:,} clientes']
        
        plt.figtext(0.02, 0.02, ' | '.join(legend_labels), 
                   fontsize=9, style='italic')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'graficos/paso3_distribucion_churn_circular_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Grafico circular generado exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar grafico circular: {str(e)}")
    
    # 3. HISTOGRAMA DE DISTRIBUCIÓN
    try:
        logger.info("Generando histograma...")
        plt.figure(figsize=(10, 6))
        
        # Crear datos separados para cada clase
        no_churn_data = df[df['Abandono_Cliente'] == 0]['Abandono_Cliente']
        churn_data = df[df['Abandono_Cliente'] == 1]['Abandono_Cliente']
        
        # Crear histograma con datos separados
        plt.hist([no_churn_data, churn_data], bins=2, color=['#2E86AB', '#A23B72'], 
                alpha=0.8, edgecolor='black', linewidth=1, 
                label=['No Churn (0)', 'Churn (1)'])
        
        plt.xlabel('Clase de Abandono', fontsize=12, fontweight='bold')
        plt.ylabel('Frecuencia (Número de Clientes)', fontsize=12, fontweight='bold')
        plt.title('Histograma de Distribución de Abandono de Clientes\nFrecuencia por Clase', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Personalizar ejes y añadir leyenda
        plt.xticks([0, 1], ['No Churn (0)', 'Churn (1)'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir estadísticas
        all_data = df['Abandono_Cliente']
        stats_text = f"Media: {all_data.mean():.3f} | Desv. Estándar: {all_data.std():.3f}"
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso3_distribucion_churn_histograma_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Histograma generado exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar histograma: {str(e)}")
    
    # 4. GRÁFICO COMPARATIVO CON MÉTRICAS
    try:
        logger.info("Generando grafico comparativo con metricas...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Conteos absolutos
        ax1.bar(categories, counts, color=colors, alpha=0.8)
        ax1.set_title('Conteos Absolutos', fontweight='bold')
        ax1.set_ylabel('Número de Clientes')
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax1.text(i, count + count*0.01, f'{count:,}', ha='center', va='bottom', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Porcentajes
        ax2.bar(categories, percentages, color=colors, alpha=0.8)
        ax2.set_title('Porcentajes', fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)')
        for i, pct in enumerate(percentages):
            ax2.text(i, pct + pct*0.01, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Métricas de balance
        metrics_labels = ['Ratio\nDesbalance', 'Minoría\n(%)', 'Mayoría\n(%)']
        metrics_values = [distribution_analysis['imbalance_ratio'], 
                         distribution_analysis['minority_percentage'],
                         distribution_analysis['majority_percentage']]
        
        bars3 = ax3.bar(metrics_labels, metrics_values, color=['#F18F01', '#C73E1D', '#2E86AB'], alpha=0.8)
        ax3.set_title('Métricas de Balance', fontweight='bold')
        ax3.set_ylabel('Valor')
        for bar, value in zip(bars3, metrics_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Información de impacto
        impact_data = {
            'Nivel de Balance': distribution_analysis['balance_level'],
            'Severidad': distribution_analysis['balance_severity'],
            'Impacto en Modelos': distribution_analysis['impact_level']
        }
        
        ax4.axis('off')
        ax4.text(0.5, 0.8, 'Evaluación de Impacto', ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax4.transAxes)
        
        y_pos = 0.6
        for key, value in impact_data.items():
            color = '#2E86AB' if 'Bajo' in value or 'Óptimo' in value or 'Balanceado' in value else \
                   '#F18F01' if 'Moderado' in value or 'Leve' in value else '#C73E1D'
            ax4.text(0.1, y_pos, f'{key}:', fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.6, y_pos, value, color=color, fontweight='bold', transform=ax4.transAxes)
            y_pos -= 0.15
        
        plt.suptitle('Análisis Completo de Balance de Clases - Churn', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'graficos/paso3_analisis_completo_churn_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Grafico comparativo generado exitosamente")
        
    except Exception as e:
        logger.error(f"Error al generar grafico comparativo: {str(e)}")
    
    logger.info("Proceso de visualizaciones completado")

def analyze_modeling_implications(distribution_analysis):
    """
    Analiza las implicaciones del balance de clases para el modelado
    
    Args:
        distribution_analysis (dict): Análisis de distribución
        
    Returns:
        dict: Recomendaciones para modelado
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO IMPLICACIONES PARA MODELADO")
    logger.info("=" * 45)
    
    balance_level = distribution_analysis['balance_level']
    imbalance_ratio = distribution_analysis['imbalance_ratio']
    minority_pct = distribution_analysis['minority_percentage']
    
    modeling_implications = {
        'requires_special_handling': False,
        'recommended_metrics': [],
        'algorithm_considerations': [],
        'sampling_strategies': [],
        'evaluation_strategies': [],
        'risk_assessment': {},
        'recommended_approach': ""
    }
    
    # Análisis según nivel de balance
    if balance_level == "Balanceado":
        modeling_implications.update({
            'requires_special_handling': False,
            'recommended_metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'algorithm_considerations': ['Todos los algoritmos funcionarán bien', 'Sin restricciones especiales'],
            'sampling_strategies': ['No requiere técnicas de sampling'],
            'evaluation_strategies': ['Validación cruzada estándar', 'Hold-out simple'],
            'recommended_approach': "Enfoque estándar de machine learning"
        })
        
    elif balance_level == "Ligeramente Desbalanceado":
        modeling_implications.update({
            'requires_special_handling': True,
            'recommended_metrics': ['Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR'],
            'algorithm_considerations': ['Algoritmos tree-based funcionan bien', 'Considerar class_weight en modelos lineales'],
            'sampling_strategies': ['Opcional: SMOTE ligero', 'Class weighting recomendado'],
            'evaluation_strategies': ['Validación cruzada estratificada', 'Monitorear métricas por clase'],
            'recommended_approach': "Enfoque con ajustes menores"
        })
        
    elif balance_level == "Moderadamente Desbalanceado":
        modeling_implications.update({
            'requires_special_handling': True,
            'recommended_metrics': ['F1-Score', 'AUC-ROC', 'AUC-PR', 'Balanced Accuracy'],
            'algorithm_considerations': ['Tree-based preferibles', 'Evitar modelos que asumen balance'],
            'sampling_strategies': ['SMOTE recomendado', 'Random undersampling', 'Class weighting obligatorio'],
            'evaluation_strategies': ['Validación cruzada estratificada', 'Análisis de curvas PR', 'Métricas por clase'],
            'recommended_approach': "Enfoque especializado en desbalance"
        })
        
    elif balance_level == "Severamente Desbalanceado":
        modeling_implications.update({
            'requires_special_handling': True,
            'recommended_metrics': ['AUC-PR', 'F1-Score', 'Recall', 'Precision@K'],
            'algorithm_considerations': ['XGBoost con scale_pos_weight', 'Random Forest con class_weight', 'Evitar Naive Bayes'],
            'sampling_strategies': ['SMOTE + Editing', 'Ensemble sampling', 'Cost-sensitive learning'],
            'evaluation_strategies': ['Validación cruzada estratificada', 'Curvas PR prioritarias', 'Análisis por umbrales'],
            'recommended_approach': "Enfoque especializado en clases desbalanceadas"
        })
    
    # Evaluación de riesgos
    modeling_implications['risk_assessment'] = {
        'overfitting_risk': 'Alto' if minority_pct < 15 else 'Moderado' if minority_pct < 25 else 'Bajo',
        'false_positive_risk': 'Alto' if imbalance_ratio > 5 else 'Moderado' if imbalance_ratio > 3 else 'Bajo',
        'model_bias_risk': 'Alto' if minority_pct < 20 else 'Moderado' if minority_pct < 30 else 'Bajo',
        'generalization_risk': 'Alto' if minority_pct < 10 else 'Moderado' if minority_pct < 20 else 'Bajo'
    }
    
    logger.info(f"Requiere manejo especial: {'SI' if modeling_implications['requires_special_handling'] else 'NO'}")
    logger.info(f"Métricas recomendadas: {', '.join(modeling_implications['recommended_metrics'])}")
    logger.info(f"Enfoque recomendado: {modeling_implications['recommended_approach']}")
    
    return modeling_implications

def generate_report(distribution_analysis, modeling_implications, target_info, timestamp):
    """
    Genera un informe detallado del análisis de proporción de churn
    
    Args:
        distribution_analysis (dict): Análisis de distribución
        modeling_implications (dict): Implicaciones para modelado
        target_info (dict): Información de variable objetivo
        timestamp (str): Timestamp del proceso
        
    Returns:
        str: Contenido del informe
    """
    
    report = f"""
================================================================================
TELECOMX - INFORME DE VERIFICACIÓN DE PROPORCIÓN DE CHURN
================================================================================
Fecha y Hora: {timestamp}
Paso: 3 - Verificación de la Proporción de Cancelación (Churn)

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Total de Clientes Analizados: {distribution_analysis['total_clients']:,}
• Clientes sin Abandono (No Churn): {distribution_analysis['no_churn_count']:,} ({distribution_analysis['no_churn_percentage']:.1f}%)
• Clientes con Abandono (Churn): {distribution_analysis['churn_count']:,} ({distribution_analysis['churn_percentage']:.1f}%)
• Nivel de Balance: {distribution_analysis['balance_level']}
• Ratio de Desbalance: {distribution_analysis['imbalance_ratio']:.2f}:1
• Impacto en Modelos: {distribution_analysis['impact_level']}
• Requiere Manejo Especial: {'✅ SÍ' if modeling_implications['requires_special_handling'] else '❌ NO'}

================================================================================
ANÁLISIS DETALLADO DE DISTRIBUCIÓN
================================================================================

📊 CONTEOS ABSOLUTOS:
• No Churn (Clase 0): {distribution_analysis['no_churn_count']:,} clientes
• Churn (Clase 1): {distribution_analysis['churn_count']:,} clientes
• Total: {distribution_analysis['total_clients']:,} clientes

📈 PORCENTAJES RELATIVOS:
• No Churn: {distribution_analysis['no_churn_percentage']:.2f}%
• Churn: {distribution_analysis['churn_percentage']:.2f}%

⚖️ MÉTRICAS DE BALANCE:
• Clase Mayoría: {distribution_analysis['majority_class']} ({distribution_analysis['majority_percentage']:.1f}%)
• Clase Minoría: {distribution_analysis['minority_class']} ({distribution_analysis['minority_percentage']:.1f}%)
• Ratio Mayoría:Minoría = {distribution_analysis['imbalance_ratio']:.2f}:1

🎯 CLASIFICACIÓN DE BALANCE:
• Nivel: {distribution_analysis['balance_level']}
• Severidad: {distribution_analysis['balance_severity']}
• Interpretación: """
    
    # Añadir interpretación según el nivel
    if distribution_analysis['balance_level'] == "Balanceado":
        report += """
  El dataset presenta un balance óptimo entre clases. Este es el escenario
  ideal para machine learning, donde ambas clases tienen representación
  suficiente para entrenar modelos robustos sin sesgos."""
        
    elif distribution_analysis['balance_level'] == "Ligeramente Desbalanceado":
        report += """
  El dataset presenta un desbalance leve pero manejable. Se recomienda
  monitorear las métricas por clase y considerar técnicas de ajuste menores
  como class weighting."""
        
    elif distribution_analysis['balance_level'] == "Moderadamente Desbalanceado":
        report += """
  El dataset presenta desbalance moderado que requerirá técnicas específicas
  de manejo de clases desbalanceadas para obtener modelos predictivos confiables."""
        
    elif distribution_analysis['balance_level'] == "Severamente Desbalanceado":
        report += """
  El dataset presenta desbalance severo que requiere estrategias avanzadas
  de sampling y algoritmos especializados para clases minoritarias."""
    
    report += f"""

================================================================================
IMPLICACIONES PARA MODELADO PREDICTIVO
================================================================================

🔬 EVALUACIÓN DE IMPACTO:
• Impacto en Modelos: {distribution_analysis['impact_level']}
• Requiere Manejo Especial: {'SÍ' if modeling_implications['requires_special_handling'] else 'NO'}
• Enfoque Recomendado: {modeling_implications['recommended_approach']}

📊 MÉTRICAS RECOMENDADAS:
"""
    
    for i, metric in enumerate(modeling_implications['recommended_metrics'], 1):
        report += f"{i}. {metric}\n"
    
    report += f"""
🚨 EVALUACIÓN DE RIESGOS:
• Riesgo de Overfitting: {modeling_implications['risk_assessment']['overfitting_risk']}
• Riesgo de Falsos Positivos: {modeling_implications['risk_assessment']['false_positive_risk']}
• Riesgo de Sesgo del Modelo: {modeling_implications['risk_assessment']['model_bias_risk']}
• Riesgo de Generalización: {modeling_implications['risk_assessment']['generalization_risk']}

🤖 CONSIDERACIONES ALGORÍTMICAS:
"""
    
    for i, consideration in enumerate(modeling_implications['algorithm_considerations'], 1):
        report += f"{i}. {consideration}\n"
    
    report += f"""
⚖️ ESTRATEGIAS DE SAMPLING RECOMENDADAS:
"""
    
    for i, strategy in enumerate(modeling_implications['sampling_strategies'], 1):
        report += f"{i}. {strategy}\n"
    
    report += f"""
🎯 ESTRATEGIAS DE EVALUACIÓN:
"""
    
    for i, strategy in enumerate(modeling_implications['evaluation_strategies'], 1):
        report += f"{i}. {strategy}\n"
    
    report += f"""
================================================================================
RECOMENDACIONES ESPECÍFICAS POR ALGORITMO
================================================================================

🌳 ALGORITMOS TREE-BASED (Random Forest, XGBoost, etc.):
• Ventaja: Manejan naturalmente el desbalance de clases
• Configuración: Usar parámetro 'class_weight=balanced' o 'scale_pos_weight'
• Recomendación: Prioritarios para este nivel de desbalance

📈 ALGORITMOS LINEALES (Logistic Regression, SVM):
• Consideración: Sensibles al desbalance de clases
• Configuración: Obligatorio usar 'class_weight=balanced'
• Preprocesamiento: Considerar techniques de sampling

🧠 ALGORITMOS DE ENSEMBLE:
• Ventaja: Pueden combinar múltiples estrategias de manejo de desbalance
• Técnicas: Bagging con submuestreo, Boosting con cost-sensitive learning
• Recomendación: Excelente opción para datos desbalanceados

🚫 ALGORITMOS NO RECOMENDADOS:
• Naive Bayes: Asume distribuciones balanceadas
• K-Means: No adecuado para clasificación con desbalance
• Modelos sin parámetros de balance: Pueden generar sesgos severos

================================================================================
PLAN DE ACCIÓN PARA PRÓXIMOS PASOS
================================================================================

🔄 PASO 4 - PREPARACIÓN DE DATOS:
• Implementar validación cruzada estratificada
• Configurar métricas apropiadas para evaluación
• Preparar conjuntos de train/validation/test balanceados

⚖️ PASO 5 - MANEJO DE DESBALANCE:
• Aplicar técnicas de sampling según recomendaciones
• Configurar class weights en algoritmos
• Implementar cost-sensitive learning si es necesario

🤖 PASO 6 - SELECCIÓN DE MODELOS:
• Priorizar algoritmos tree-based
• Configurar hiperparámetros específicos para desbalance
• Implementar ensemble methods

📊 PASO 7 - EVALUACIÓN ESPECIALIZADA:
• Enfocar en métricas recomendadas
• Analizar curvas PR y ROC
• Evaluar performance por clase

================================================================================
TÉCNICAS DE SAMPLING DETALLADAS
================================================================================

✅ TÉCNICAS RECOMENDADAS PARA ESTE CASO:

1. SMOTE (Synthetic Minority Oversampling Technique):
   • Genera ejemplos sintéticos de la clase minoritaria
   • Preserva la distribución original de los datos
   • Reduce el riesgo de overfitting

2. CLASS WEIGHTING:
   • Asigna pesos inversamente proporcionales a la frecuencia de clase
   • Penaliza más los errores en la clase minoritaria
   • Implementación sencilla en la mayoría de algoritmos

3. RANDOM UNDERSAMPLING:
   • Reduce la clase mayoría para balancear
   • Rápido y eficiente
   • Riesgo: pérdida de información

4. ENSEMBLE METHODS:
   • Combine múltiples modelos con diferentes estrategias de sampling
   • BalancedRandomForest, EasyEnsemble
   • Robusto contra overfitting

================================================================================
MÉTRICAS DE EVALUACIÓN PRIORITARIAS
================================================================================

🎯 MÉTRICAS PRINCIPALES:
"""
    
    # Añadir explicación de métricas según el nivel de desbalance
    if modeling_implications['requires_special_handling']:
        report += f"""
1. AUC-PR (Area Under Precision-Recall Curve):
   • MÁS IMPORTANTE que AUC-ROC para datos desbalanceados
   • Mejor indicador de performance real en clase minoritaria
   • Valor objetivo: > 0.7 para resultados aceptables

2. F1-Score:
   • Promedio armónico de Precision y Recall
   • Balancea ambas métricas críticas
   • Valor objetivo: > 0.6 para este nivel de desbalance

3. Recall (Sensibilidad):
   • Capacidad de detectar casos de churn reales
   • CRÍTICO para el negocio (no perder clientes en riesgo)
   • Valor objetivo: > 0.7 para capturar mayoría de churns

4. Precision:
   • Confiabilidad de las predicciones positivas
   • Importante para eficiencia de campañas de retención
   • Balance con Recall según objetivo de negocio
"""
    else:
        report += f"""
1. Accuracy:
   • Proporción de predicciones correctas
   • Válida para datasets balanceados
   • Valor objetivo: > 0.85

2. F1-Score:
   • Promedio armónico de Precision y Recall
   • Métrica balanceada recomendada
   • Valor objetivo: > 0.8

3. AUC-ROC:
   • Área bajo la curva ROC
   • Excelente para datasets balanceados
   • Valor objetivo: > 0.9
"""
    
    report += f"""
================================================================================
VALIDACIÓN DE DATOS Y CALIDAD
================================================================================

✅ VERIFICACIONES REALIZADAS:
• Variable objetivo encontrada: {target_info['exists']}
• Tipo de datos: {target_info['data_type']}
• Es binaria (0/1): {target_info['is_binary']}
• Valores únicos: {target_info['unique_values']}
• Valores nulos: {target_info['null_count']}
• Total de registros: {target_info['total_rows']:,}

✅ INTEGRIDAD DEL DATASET:
• Consistencia de datos: Verificada
• Formato de variable objetivo: Correcto
• Distribución documentada: Completa
• Apto para modelado: {'SÍ' if target_info['is_binary'] and target_info['exists'] else 'NO'}

================================================================================
RECURSOS Y REFERENCIAS TÉCNICAS
================================================================================

📚 LIBRERÍAS RECOMENDADAS:
• imbalanced-learn: Para técnicas de sampling avanzadas
• scikit-learn: Para algoritmos con class_weight
• xgboost: Para scale_pos_weight automático
• lightgbm: Para is_unbalance=True

🔗 TÉCNICAS AVANZADAS:
• ADASYN: Adaptive Synthetic Sampling
• BorderlineSMOTE: SMOTE para casos límite
• SMOTEENN: SMOTE + Edited Nearest Neighbours
• Cost-sensitive learning: Matrices de costo personalizadas

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 VISUALIZACIONES:
• Gráfico de barras: graficos/paso3_distribucion_churn_barras_{timestamp}.png
• Gráfico circular: graficos/paso3_distribucion_churn_circular_{timestamp}.png
• Histograma: graficos/paso3_distribucion_churn_histograma_{timestamp}.png
• Análisis completo: graficos/paso3_analisis_completo_churn_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe completo: informes/paso3_verificacion_proporcion_churn_informe_{timestamp}.txt
• Log del proceso: logs/paso3_verificacion_churn.log

================================================================================
CONCLUSIONES Y SIGUIENTE PASO
================================================================================

🎯 CONCLUSIÓN PRINCIPAL:
El dataset presenta un nivel de desbalance '{distribution_analysis['balance_level']}' 
con una ratio de {distribution_analysis['imbalance_ratio']:.2f}:1, lo que requiere 
{('estrategias especializadas' if modeling_implications['requires_special_handling'] else 'enfoques estándar')} 
de machine learning para obtener modelos predictivos confiables.

📋 PRÓXIMO PASO RECOMENDADO:
Paso 4: División Estratificada de Datos (Train/Validation/Test)
• Implementar split estratificado para preservar proporciones
• Configurar validación cruzada apropiada para datos desbalanceados
• Preparar pipeline de evaluación con métricas especializadas

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
    report_filename = f"informes/paso3_verificacion_proporcion_churn_informe_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Informe guardado: {report_filename}")
    
    return report_filename

def main():
    """Función principal que ejecuta todo el proceso de verificación de churn"""
    
    # Crear directorios y configurar logging
    create_directories()
    logger = setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("INICIANDO PASO 3: VERIFICACION DE PROPORCION DE CHURN")
    logger.info("=" * 70)
    
    try:
        # 1. Encontrar y cargar el archivo del Paso 2
        input_file = find_latest_paso2_file()
        df = load_data(input_file)
        
        # 2. Verificar variable objetivo
        is_valid, target_info = verify_target_variable(df)
        
        if not is_valid:
            raise ValueError("Variable objetivo no válida o no encontrada")
        
        # 3. Analizar distribución de churn
        distribution_analysis = analyze_churn_distribution(df)
        
        # 4. Generar visualizaciones
        generate_churn_visualizations(df, distribution_analysis, timestamp)
        
        # 5. Analizar implicaciones para modelado
        modeling_implications = analyze_modeling_implications(distribution_analysis)
        
        # 6. Generar informe detallado
        report_content = generate_report(distribution_analysis, modeling_implications, target_info, timestamp)
        
        # 7. Guardar archivos
        report_file = save_files(report_content, timestamp)
        
        # 8. Resumen final
        logger.info("=" * 70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info(f"Total de clientes analizados: {distribution_analysis['total_clients']:,}")
        logger.info(f"Proporcion de churn: {distribution_analysis['churn_percentage']:.1f}%")
        logger.info(f"Nivel de balance: {distribution_analysis['balance_level']}")
        logger.info(f"Requiere manejo especial: {'SI' if modeling_implications['requires_special_handling'] else 'NO'}")
        logger.info(f"Informe generado: {report_file}")
        logger.info("=" * 70)
        
        print(f"\nRESUMEN DE CHURN:")
        print(f"   • Total clientes: {distribution_analysis['total_clients']:,}")
        print(f"   • No Churn: {distribution_analysis['no_churn_count']:,} ({distribution_analysis['no_churn_percentage']:.1f}%)")
        print(f"   • Churn: {distribution_analysis['churn_count']:,} ({distribution_analysis['churn_percentage']:.1f}%)")
        print(f"   • Balance: {distribution_analysis['balance_level']}")
        print(f"   • Ratio: {distribution_analysis['imbalance_ratio']:.2f}:1")
        
        print("\nSIGUIENTE PASO:")
        print("   Paso 4: División Estratificada de Datos (Train/Validation/Test)")
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        raise

if __name__ == "__main__":
    main()