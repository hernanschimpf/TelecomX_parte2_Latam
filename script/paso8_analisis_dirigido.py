"""
================================================================================
TELECOMX - PASO 8: ANÁLISIS DIRIGIDO DE VARIABLES CLAVE (VERSIÓN BÁSICA)
================================================================================
Descripción: Análisis simplificado de las dos relaciones más importantes:
             Tiempo de Contrato × Cancelación y Gasto Total × Cancelación

Funcionalidades:
- Análisis Tiempo de Contrato × Cancelación
- Análisis Gasto Total × Cancelación  
- Visualizaciones claras y simples
- Recomendaciones básicas para retención

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
from datetime import datetime
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso8_analisis_dirigido.log', mode='a', encoding='utf-8'),
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
    """Cargar el dataset optimizado del paso anterior"""
    try:
        # Buscar el archivo más reciente del Paso 7
        input_file = find_latest_file('excel', 'telecomx_paso7_variables_optimizadas_*.csv')
        logging.info(f"Cargando archivo: {input_file}")
        
        # Intentar diferentes combinaciones de codificación y separador
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        separators = [',', ';', '\t']
        df = None
        
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(input_file, encoding=encoding, sep=sep)
                    if df.shape[1] > 5:
                        logging.info(f"Archivo cargado exitosamente:")
                        logging.info(f"  Codificación: {encoding}")
                        logging.info(f"  Separador: '{sep}'")
                        logging.info(f"  Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
                        break
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
            if df is not None and df.shape[1] > 5:
                break
        
        if df is None or df.shape[1] <= 5:
            raise ValueError(f"No se pudo cargar el archivo correctamente.")
        
        # Verificar variables clave
        required_vars = ['Abandono_Cliente', 'Meses_Cliente', 'Cargo_Total']
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            logging.error(f"Variables requeridas no encontradas: {missing_vars}")
            raise ValueError(f"Variables faltantes: {missing_vars}")
        
        logging.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df, input_file
        
    except Exception as e:
        logging.error(f"Error al cargar el dataset: {str(e)}")
        raise

def analyze_time_vs_churn(df):
    """Analizar relación Tiempo de Contrato × Cancelación"""
    logging.info("Analizando Tiempo de Contrato × Cancelación...")
    
    time_var = 'Meses_Cliente'
    target_var = 'Abandono_Cliente'
    
    # Estadísticas descriptivas básicas
    no_churn_stats = df[df[target_var] == 0][time_var].describe()
    churn_stats = df[df[target_var] == 1][time_var].describe()
    
    # Test estadístico simple
    no_churn_data = df[df[target_var] == 0][time_var]
    churn_data = df[df[target_var] == 1][time_var]
    
    # Test de Mann-Whitney U
    statistic, p_value = stats.mannwhitneyu(no_churn_data, churn_data, alternative='two-sided')
    
    # Segmentación simple (3 grupos)
    df['Segmento_Tiempo'] = pd.cut(df[time_var], 
                                  bins=[0, 12, 36, float('inf')],
                                  labels=['Nuevos (0-12m)', 'Intermedios (12-36m)', 'Veteranos (36m+)'])
    
    # Tasa de churn por segmento
    churn_by_segment = df.groupby('Segmento_Tiempo')[target_var].agg(['count', 'sum', 'mean']).round(3)
    churn_by_segment.columns = ['Total_Clientes', 'Churns', 'Tasa_Churn']
    
    analysis_results = {
        'variable': time_var,
        'no_churn_stats': no_churn_stats,
        'churn_stats': churn_stats,
        'statistical_test': {
            'test_name': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'segment_analysis': churn_by_segment
    }
    
    logging.info(f"Análisis de tiempo completado. Diferencia significativa: {p_value < 0.05}")
    return analysis_results

def analyze_spending_vs_churn(df):
    """Analizar relación Gasto Total × Cancelación"""
    logging.info("Analizando Gasto Total × Cancelación...")
    
    spending_var = 'Cargo_Total'
    target_var = 'Abandono_Cliente'
    
    # Estadísticas descriptivas básicas
    no_churn_stats = df[df[target_var] == 0][spending_var].describe()
    churn_stats = df[df[target_var] == 1][spending_var].describe()
    
    # Test estadístico simple
    no_churn_data = df[df[target_var] == 0][spending_var]
    churn_data = df[df[target_var] == 1][spending_var]
    
    # Test de Mann-Whitney U
    statistic, p_value = stats.mannwhitneyu(no_churn_data, churn_data, alternative='two-sided')
    
    # Segmentación simple (3 grupos por terciles)
    terciles = df[spending_var].quantile([0, 0.33, 0.66, 1.0])
    df['Segmento_Gasto'] = pd.cut(df[spending_var], 
                                  bins=terciles,
                                  labels=['Bajo Gasto', 'Gasto Medio', 'Alto Gasto'],
                                  include_lowest=True)
    
    # Tasa de churn por segmento
    churn_by_spending = df.groupby('Segmento_Gasto')[target_var].agg(['count', 'sum', 'mean']).round(3)
    churn_by_spending.columns = ['Total_Clientes', 'Churns', 'Tasa_Churn']
    
    analysis_results = {
        'variable': spending_var,
        'no_churn_stats': no_churn_stats,
        'churn_stats': churn_stats,
        'statistical_test': {
            'test_name': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        },
        'segment_analysis': churn_by_spending
    }
    
    logging.info(f"Análisis de gasto completado. Diferencia significativa: {p_value < 0.05}")
    return analysis_results

def generate_simple_visualizations(df, time_analysis, spending_analysis, timestamp):
    """Generar visualizaciones simples y claras"""
    logging.info("Generando visualizaciones simples...")
    
    try:
        plt.style.use('default')
        
        # 1. Análisis de Tiempo - Gráfico simple
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot para tiempo
        no_churn_time = df[df['Abandono_Cliente'] == 0]['Meses_Cliente']
        churn_time = df[df['Abandono_Cliente'] == 1]['Meses_Cliente']
        
        ax1.boxplot([no_churn_time, churn_time], labels=['No Churn', 'Churn'])
        ax1.set_title('Distribución de Meses como Cliente', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Meses como Cliente')
        ax1.grid(True, alpha=0.3)
        
        # Agregar estadísticas en el gráfico
        ax1.text(0.02, 0.98, f'No Churn:\nPromedio: {no_churn_time.mean():.1f} meses\nMediana: {no_churn_time.median():.1f} meses', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax1.text(0.02, 0.70, f'Churn:\nPromedio: {churn_time.mean():.1f} meses\nMediana: {churn_time.median():.1f} meses', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Tasa de churn por segmento de tiempo
        if 'Segmento_Tiempo' in df.columns:
            segment_data = time_analysis['segment_analysis']
            bars = ax2.bar(range(len(segment_data)), segment_data['Tasa_Churn'], 
                          color=['red' if x > df['Abandono_Cliente'].mean() else 'green' 
                                for x in segment_data['Tasa_Churn']], alpha=0.7)
            ax2.set_title('Tasa de Churn por Segmento de Tiempo', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Tasa de Churn')
            ax2.set_xticks(range(len(segment_data)))
            ax2.set_xticklabels(segment_data.index, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Línea de promedio general
            avg_churn = df['Abandono_Cliente'].mean()
            ax2.axhline(y=avg_churn, color='blue', linestyle='--', 
                       label=f'Promedio General: {avg_churn:.3f}')
            ax2.legend()
            
            # Valores en las barras
            for i, (bar, value) in enumerate(zip(bars, segment_data['Tasa_Churn'])):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso8_analisis_tiempo_simple_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Análisis de Gasto - Gráfico simple
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot para gasto
        no_churn_spending = df[df['Abandono_Cliente'] == 0]['Cargo_Total']
        churn_spending = df[df['Abandono_Cliente'] == 1]['Cargo_Total']
        
        ax1.boxplot([no_churn_spending, churn_spending], labels=['No Churn', 'Churn'])
        ax1.set_title('Distribución de Gasto Total', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cargo Total ($)')
        ax1.grid(True, alpha=0.3)
        
        # Agregar estadísticas en el gráfico
        ax1.text(0.02, 0.98, f'No Churn:\nPromedio: ${no_churn_spending.mean():.2f}\nMediana: ${no_churn_spending.median():.2f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax1.text(0.02, 0.70, f'Churn:\nPromedio: ${churn_spending.mean():.2f}\nMediana: ${churn_spending.median():.2f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Tasa de churn por segmento de gasto
        if 'Segmento_Gasto' in df.columns:
            segment_data = spending_analysis['segment_analysis']
            bars = ax2.bar(range(len(segment_data)), segment_data['Tasa_Churn'], 
                          color=['red' if x > df['Abandono_Cliente'].mean() else 'green' 
                                for x in segment_data['Tasa_Churn']], alpha=0.7)
            ax2.set_title('Tasa de Churn por Segmento de Gasto', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Tasa de Churn')
            ax2.set_xticks(range(len(segment_data)))
            ax2.set_xticklabels(segment_data.index, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Línea de promedio general
            avg_churn = df['Abandono_Cliente'].mean()
            ax2.axhline(y=avg_churn, color='blue', linestyle='--', 
                       label=f'Promedio General: {avg_churn:.3f}')
            ax2.legend()
            
            # Valores en las barras
            for i, (bar, value) in enumerate(zip(bars, segment_data['Tasa_Churn'])):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso8_analisis_gasto_simple_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot de relación tiempo vs gasto
        plt.figure(figsize=(12, 8))
        
        # Separar datos por churn
        no_churn_df = df[df['Abandono_Cliente'] == 0]
        churn_df = df[df['Abandono_Cliente'] == 1]
        
        # Scatter plot
        plt.scatter(no_churn_df['Meses_Cliente'], no_churn_df['Cargo_Total'], 
                   alpha=0.6, c='green', label='No Churn', s=20)
        plt.scatter(churn_df['Meses_Cliente'], churn_df['Cargo_Total'], 
                   alpha=0.6, c='red', label='Churn', s=20)
        
        plt.title('Relación entre Tiempo como Cliente y Gasto Total', fontsize=16, fontweight='bold')
        plt.xlabel('Meses como Cliente', fontsize=12)
        plt.ylabel('Cargo Total ($)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Agregar líneas de promedio
        plt.axvline(df['Meses_Cliente'].mean(), color='blue', linestyle='--', alpha=0.7,
                   label=f'Promedio Tiempo: {df["Meses_Cliente"].mean():.1f} meses')
        plt.axhline(df['Cargo_Total'].mean(), color='orange', linestyle='--', alpha=0.7,
                   label=f'Promedio Gasto: ${df["Cargo_Total"].mean():.2f}')
        
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'graficos/paso8_relacion_tiempo_gasto_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones simples generadas exitosamente")
        return True
        
    except Exception as e:
        logging.error(f"Error al generar visualizaciones: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def generate_simple_recommendations(time_analysis, spending_analysis):
    """Generar recomendaciones básicas y directas"""
    logging.info("Generando recomendaciones básicas...")
    
    recommendations = {
        'time_based': [],
        'spending_based': [],
        'general': []
    }
    
    # Analizar resultados de tiempo
    time_segments = time_analysis['segment_analysis']
    avg_churn = time_segments['Tasa_Churn'].mean()
    
    for segment, data in time_segments.iterrows():
        if data['Tasa_Churn'] > avg_churn * 1.2:  # 20% más que el promedio
            if 'Nuevos' in segment:
                recommendations['time_based'].append({
                    'segment': segment,
                    'issue': f'Alta tasa de churn: {data["Tasa_Churn"]:.1%}',
                    'recommendation': 'Programa de onboarding intensivo',
                    'action': 'Seguimiento personalizado en primeros 90 días'
                })
            else:
                recommendations['time_based'].append({
                    'segment': segment,
                    'issue': f'Tasa de churn elevada: {data["Tasa_Churn"]:.1%}',
                    'recommendation': 'Programa de fidelización',
                    'action': 'Ofertas especiales y beneficios adicionales'
                })
    
    # Analizar resultados de gasto
    spending_segments = spending_analysis['segment_analysis']
    
    for segment, data in spending_segments.iterrows():
        if data['Tasa_Churn'] > avg_churn * 1.2:
            if 'Bajo' in segment:
                recommendations['spending_based'].append({
                    'segment': segment,
                    'issue': f'Alta tasa de churn: {data["Tasa_Churn"]:.1%}',
                    'recommendation': 'Planes de valor con servicios adicionales',
                    'action': 'Ofertas de upgrade con descuentos especiales'
                })
            else:
                recommendations['spending_based'].append({
                    'segment': segment,
                    'issue': f'Pérdida de clientes valiosos: {data["Tasa_Churn"]:.1%}',
                    'recommendation': 'Atención personalizada VIP',
                    'action': 'Gestor de cuenta dedicado y soporte premium'
                })
    
    # Recomendaciones generales
    recommendations['general'] = [
        {
            'area': 'Retención Temprana',
            'recommendation': 'Enfocar esfuerzos en los primeros 12 meses',
            'justification': 'Período crítico identificado en análisis'
        },
        {
            'area': 'Segmentación',
            'recommendation': 'Estrategias diferenciadas por nivel de gasto',
            'justification': 'Patrones distintos entre segmentos de gasto'
        },
        {
            'area': 'Monitoreo',
            'recommendation': 'Sistema de alertas basado en tiempo y gasto',
            'justification': 'Variables clave para predicción de churn'
        }
    ]
    
    logging.info("Recomendaciones básicas generadas exitosamente")
    return recommendations

def generate_simple_report(time_analysis, spending_analysis, recommendations, timestamp):
    """Generar informe simplificado"""
    
    report = f"""
================================================================================
TELECOMX - INFORME PASO 8: ANÁLISIS DIRIGIDO BÁSICO
================================================================================
Fecha y Hora: {timestamp}
Paso: 8 - Análisis Dirigido (Versión Básica)

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Análisis realizados: 2 (Tiempo de Contrato y Gasto Total)
• Variables analizadas: Meses_Cliente y Cargo_Total
• Recomendaciones generadas: {len(recommendations['time_based']) + len(recommendations['spending_based']) + len(recommendations['general'])}
• Enfoque: Identificación de patrones simples y claros

================================================================================
ANÁLISIS 1: TIEMPO DE CONTRATO × CANCELACIÓN
================================================================================

🎯 VARIABLE: Meses_Cliente

📊 ESTADÍSTICAS COMPARATIVAS:
"""
    
    # Estadísticas de tiempo
    no_churn_stats = time_analysis['no_churn_stats']
    churn_stats = time_analysis['churn_stats']
    
    report += f"""
NO CHURN:
   • Promedio: {no_churn_stats['mean']:.1f} meses
   • Mediana: {no_churn_stats['50%']:.1f} meses
   • Rango: {no_churn_stats['min']:.0f} - {no_churn_stats['max']:.0f} meses

CHURN:
   • Promedio: {churn_stats['mean']:.1f} meses
   • Mediana: {churn_stats['50%']:.1f} meses
   • Rango: {churn_stats['min']:.0f} - {churn_stats['max']:.0f} meses

📈 ANÁLISIS ESTADÍSTICO:
• Test: {time_analysis['statistical_test']['test_name']}
• P-valor: {time_analysis['statistical_test']['p_value']:.6f}
• Resultado: {"SIGNIFICATIVO" if time_analysis['statistical_test']['significant'] else "NO SIGNIFICATIVO"}
• Conclusión: {"El tiempo como cliente SÍ influye en el churn" if time_analysis['statistical_test']['significant'] else "No hay evidencia de influencia del tiempo"}

🔍 SEGMENTACIÓN POR TIEMPO:
"""
    
    for segment, data in time_analysis['segment_analysis'].iterrows():
        risk_level = "🔴 ALTO RIESGO" if data['Tasa_Churn'] > 0.3 else "🟡 RIESGO MEDIO" if data['Tasa_Churn'] > 0.2 else "🟢 BAJO RIESGO"
        report += f"""
{segment}: {risk_level}
   • Total Clientes: {data['Total_Clientes']:,}
   • Tasa de Churn: {data['Tasa_Churn']:.1%}
   • Clientes perdidos: {data['Churns']:,}"""

    report += f"""

================================================================================
ANÁLISIS 2: GASTO TOTAL × CANCELACIÓN
================================================================================

🎯 VARIABLE: Cargo_Total

📊 ESTADÍSTICAS COMPARATIVAS:
"""
    
    # Estadísticas de gasto
    no_churn_spending = spending_analysis['no_churn_stats']
    churn_spending = spending_analysis['churn_stats']
    
    report += f"""
NO CHURN:
   • Promedio: ${no_churn_spending['mean']:.2f}
   • Mediana: ${no_churn_spending['50%']:.2f}
   • Rango: ${no_churn_spending['min']:.2f} - ${no_churn_spending['max']:.2f}

CHURN:
   • Promedio: ${churn_spending['mean']:.2f}
   • Mediana: ${churn_spending['50%']:.2f}
   • Rango: ${churn_spending['min']:.2f} - ${churn_spending['max']:.2f}

📈 ANÁLISIS ESTADÍSTICO:
• Test: {spending_analysis['statistical_test']['test_name']}
• P-valor: {spending_analysis['statistical_test']['p_value']:.6f}
• Resultado: {"SIGNIFICATIVO" if spending_analysis['statistical_test']['significant'] else "NO SIGNIFICATIVO"}
• Conclusión: {"El gasto total SÍ influye en el churn" if spending_analysis['statistical_test']['significant'] else "No hay evidencia de influencia del gasto"}

🔍 SEGMENTACIÓN POR GASTO:
"""
    
    for segment, data in spending_analysis['segment_analysis'].iterrows():
        risk_level = "🔴 ALTO RIESGO" if data['Tasa_Churn'] > 0.3 else "🟡 RIESGO MEDIO" if data['Tasa_Churn'] > 0.2 else "🟢 BAJO RIESGO"
        report += f"""
{segment}: {risk_level}
   • Total Clientes: {data['Total_Clientes']:,}
   • Tasa de Churn: {data['Tasa_Churn']:.1%}
   • Clientes perdidos: {data['Churns']:,}"""

    report += f"""

================================================================================
RECOMENDACIONES PARA RETENCIÓN
================================================================================

🕐 ESTRATEGIAS BASADAS EN TIEMPO:
"""
    
    if recommendations['time_based']:
        for rec in recommendations['time_based']:
            report += f"""
• {rec['segment']}:
  - Problema: {rec['issue']}
  - Recomendación: {rec['recommendation']}
  - Acción: {rec['action']}"""
    else:
        report += "\n• No se identificaron segmentos de tiempo con riesgo elevado"

    report += f"""

💰 ESTRATEGIAS BASADAS EN GASTO:
"""
    
    if recommendations['spending_based']:
        for rec in recommendations['spending_based']:
            report += f"""
• {rec['segment']}:
  - Problema: {rec['issue']}
  - Recomendación: {rec['recommendation']}
  - Acción: {rec['action']}"""
    else:
        report += "\n• No se identificaron segmentos de gasto con riesgo elevado"

    report += f"""

🎯 RECOMENDACIONES GENERALES:
"""
    
    for rec in recommendations['general']:
        report += f"""
• {rec['area']}: {rec['recommendation']}
  - Justificación: {rec['justification']}"""

    report += f"""

================================================================================
CONCLUSIONES PRINCIPALES
================================================================================

🔍 INSIGHTS CLAVE:
• Tiempo como cliente: {"Factor significativo" if time_analysis['statistical_test']['significant'] else "No es factor determinante"} en el churn
• Gasto total: {"Factor significativo" if spending_analysis['statistical_test']['significant'] else "No es factor determinante"} en el churn
• Segmento más crítico por tiempo: {time_analysis['segment_analysis']['Tasa_Churn'].idxmax()}
• Segmento más crítico por gasto: {spending_analysis['segment_analysis']['Tasa_Churn'].idxmax()}

📊 MÉTRICAS DE IMPACTO:
• Tasa de churn más alta por tiempo: {time_analysis['segment_analysis']['Tasa_Churn'].max():.1%}
• Tasa de churn más alta por gasto: {spending_analysis['segment_analysis']['Tasa_Churn'].max():.1%}
• Diferencia entre mejor y peor segmento (tiempo): {(time_analysis['segment_analysis']['Tasa_Churn'].max() - time_analysis['segment_analysis']['Tasa_Churn'].min()):.1%}
• Diferencia entre mejor y peor segmento (gasto): {(spending_analysis['segment_analysis']['Tasa_Churn'].max() - spending_analysis['segment_analysis']['Tasa_Churn'].min()):.1%}

================================================================================
PRÓXIMO PASO RECOMENDADO
================================================================================

Paso 9: Entrenamiento de Modelos Predictivos
• Usar variables Meses_Cliente y Cargo_Total como predictores clave
• Aplicar segmentaciones identificadas para mejorar predicciones
• Validar modelos contra patrones encontrados en este análisis

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 VISUALIZACIONES:
• Análisis tiempo: graficos/paso8_analisis_tiempo_simple_{timestamp}.png
• Análisis gasto: graficos/paso8_analisis_gasto_simple_{timestamp}.png
• Relación tiempo-gasto: graficos/paso8_relacion_tiempo_gasto_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe: informes/paso8_analisis_dirigido_informe_{timestamp}.txt
• Log del proceso: logs/paso8_analisis_dirigido.log

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(report_content, recommendations, timestamp):
    """Guardar archivos de salida"""
    try:
        # Guardar informe
        report_file = f'informes/paso8_analisis_dirigido_informe_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Informe guardado: {report_file}")
        
        # Guardar recomendaciones en formato simple
        recommendations_file = f'informes/paso8_recomendaciones_basicas_{timestamp}.txt'
        with open(recommendations_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TELECOMX - RECOMENDACIONES BÁSICAS DE RETENCIÓN\n")
            f.write("="*80 + "\n\n")
            
            f.write("ESTRATEGIAS BASADAS EN TIEMPO:\n")
            f.write("-" * 40 + "\n")
            for rec in recommendations['time_based']:
                f.write(f"• {rec['segment']}: {rec['recommendation']}\n")
                f.write(f"  Acción: {rec['action']}\n\n")
            
            f.write("ESTRATEGIAS BASADAS EN GASTO:\n")
            f.write("-" * 40 + "\n")
            for rec in recommendations['spending_based']:
                f.write(f"• {rec['segment']}: {rec['recommendation']}\n")
                f.write(f"  Acción: {rec['action']}\n\n")
            
            f.write("RECOMENDACIONES GENERALES:\n")
            f.write("-" * 40 + "\n")
            for rec in recommendations['general']:
                f.write(f"• {rec['area']}: {rec['recommendation']}\n")
                f.write(f"  Justificación: {rec['justification']}\n\n")
        
        logging.info(f"Recomendaciones guardadas: {recommendations_file}")
        
        return {
            'report_file': report_file,
            'recommendations_file': recommendations_file
        }
        
    except Exception as e:
        logging.error(f"Error al guardar archivos: {str(e)}")
        raise

def main():
    """Función principal del Paso 8 - Versión Básica"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 8: ANÁLISIS DIRIGIDO BÁSICO")
        logger.info("="*80)
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos del paso anterior
        df, input_file = load_data()
        logger.info(f"Dataset cargado desde: {input_file}")
        
        # 3. Análisis dirigidos básicos
        logger.info("Realizando análisis dirigidos básicos...")
        
        time_analysis = analyze_time_vs_churn(df)
        spending_analysis = analyze_spending_vs_churn(df)
        
        # 4. Generar visualizaciones simples
        viz_success = generate_simple_visualizations(df, time_analysis, spending_analysis, timestamp)
        
        # 5. Generar recomendaciones básicas
        recommendations = generate_simple_recommendations(time_analysis, spending_analysis)
        
        # 6. Generar informe simplificado
        report_content = generate_simple_report(time_analysis, spending_analysis, recommendations, timestamp)
        
        # 7. Guardar archivos
        output_files = save_files(report_content, recommendations, timestamp)
        
        # 8. Resumen final
        logger.info("="*80)
        logger.info("PASO 8 BÁSICO COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("RESUMEN DE RESULTADOS:")
        
        # Mostrar insights principales
        time_significant = "SÍ" if time_analysis['statistical_test']['significant'] else "NO"
        spending_significant = "SÍ" if spending_analysis['statistical_test']['significant'] else "NO"
        
        logger.info(f"  • Tiempo influye en churn: {time_significant}")
        logger.info(f"  • Gasto influye en churn: {spending_significant}")
        logger.info(f"  • Segmentos de tiempo analizados: {len(time_analysis['segment_analysis'])}")
        logger.info(f"  • Segmentos de gasto analizados: {len(spending_analysis['segment_analysis'])}")
        logger.info(f"  • Recomendaciones generadas: {len(recommendations['time_based']) + len(recommendations['spending_based']) + len(recommendations['general'])}")
        logger.info("")
        
        # Mostrar segmentos más críticos
        critical_time_segment = time_analysis['segment_analysis']['Tasa_Churn'].idxmax()
        critical_spending_segment = spending_analysis['segment_analysis']['Tasa_Churn'].idxmax()
        max_time_churn = time_analysis['segment_analysis']['Tasa_Churn'].max()
        max_spending_churn = spending_analysis['segment_analysis']['Tasa_Churn'].max()
        
        logger.info("SEGMENTOS CRÍTICOS IDENTIFICADOS:")
        logger.info(f"  • Tiempo: {critical_time_segment} ({max_time_churn:.1%} churn)")
        logger.info(f"  • Gasto: {critical_spending_segment} ({max_spending_churn:.1%} churn)")
        logger.info("")
        
        logger.info("ARCHIVOS GENERADOS:")
        logger.info(f"  • Informe detallado: {output_files['report_file']}")
        logger.info(f"  • Recomendaciones: {output_files['recommendations_file']}")
        if viz_success:
            logger.info(f"  • Visualizaciones: 3 gráficos simples en carpeta graficos/")
        logger.info("")
        
        logger.info("VISUALIZACIONES GENERADAS:")
        logger.info("  • Análisis de tiempo: Boxplot + tasas por segmento")
        logger.info("  • Análisis de gasto: Boxplot + tasas por segmento") 
        logger.info("  • Relación tiempo-gasto: Scatter plot con patrones de churn")
        logger.info("")
        
        logger.info("PRÓXIMO PASO SUGERIDO:")
        logger.info("  Paso 9: Entrenamiento de Modelos Predictivos")
        logger.info("  - Usar Meses_Cliente y Cargo_Total como predictores principales")
        logger.info("  - Aplicar insights de segmentación en feature engineering")
        logger.info("  - Validar modelos contra patrones identificados")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()