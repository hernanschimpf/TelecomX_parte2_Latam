"""
================================================================================
TELECOMX - PASO 13B: SEGMENTACIÓN ESTRATÉGICA BASADA EN RIESGO DE CHURN
================================================================================
Descripción: Segmentación de clientes basada en probabilidades de churn del
             modelo predictivo. Define estrategias específicas por segmento
             de riesgo y métricas de seguimiento.

Inputs: 
- Factores críticos del Paso 13A
- Dataset con variables optimizadas (Paso 7)
- Modelo entrenado (Regresión Logística)

Outputs:
- Segmentación de clientes por riesgo (Bajo, Medio, Alto)
- Estrategias específicas por segmento
- Dashboard ejecutivo en Excel
- Visualizaciones estratégicas

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
import pickle
from datetime import datetime
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso13b_segmentacion_estrategica.log', mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Crear directorios necesarios"""
    directories = ['excel', 'informes', 'graficos', 'logs', 'datos']
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

def load_critical_factors():
    """Cargar factores críticos del Paso 13A"""
    try:
        logging.info("Cargando factores críticos del Paso 13A...")
        
        # Buscar el archivo más reciente del paso 13a
        json_file = find_latest_file('datos', 'paso13a_analisis_consolidado_v2_*.json')
        
        with open(json_file, 'r', encoding='utf-8') as f:
            paso13a_data = json.load(f)
        
        critical_factors = paso13a_data['critical_factors']
        category_analysis = paso13a_data['category_analysis']
        model_info = paso13a_data['model_info']
        
        logging.info(f"Factores críticos cargados: {len(critical_factors)}")
        logging.info(f"Categorías analizadas: {len(category_analysis)}")
        logging.info(f"Mejor modelo: {model_info['best_model']}")
        
        return {
            'critical_factors': critical_factors,
            'category_analysis': category_analysis,
            'model_info': model_info,
            'paso13a_file': json_file
        }
        
    except Exception as e:
        logging.error(f"Error cargando factores críticos: {str(e)}")
        raise

def load_dataset_and_model():
    """Cargar dataset optimizado y modelo entrenado"""
    try:
        logging.info("Cargando dataset optimizado y modelo entrenado...")
        
        # 1. Cargar dataset optimizado
        dataset_file = find_latest_file('excel', 'telecomx_paso7_variables_optimizadas_*.csv')
        
        # Probar diferentes encodings
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(dataset_file, encoding=encoding)
                logging.info(f"Dataset cargado con encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("No se pudo cargar el dataset con ningún encoding")
        
        logging.info(f"Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")
        logging.info(f"Tasa de churn: {df['Abandono_Cliente'].mean():.1%}")
        
        # 2. Cargar modelo entrenado
        model_file = find_latest_file('modelos', 'logistic_regression_pipeline_*.pkl')
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        logging.info(f"Modelo cargado: {model_file}")
        
        # 3. Preparar datos para predicción
        X = df.drop(columns=['Abandono_Cliente'])
        y = df['Abandono_Cliente']
        
        # Verificar que las columnas coincidan con las del modelo
        expected_features = list(X.columns)
        logging.info(f"Features disponibles: {len(expected_features)}")
        
        return {
            'dataset': df,
            'X': X,
            'y': y,
            'model': model,
            'dataset_file': dataset_file,
            'model_file': model_file
        }
        
    except Exception as e:
        logging.error(f"Error cargando dataset y modelo: {str(e)}")
        raise

def generate_churn_probabilities(data_model):
    """Generar probabilidades de churn para cada cliente"""
    try:
        logging.info("Generando probabilidades de churn...")
        
        df = data_model['dataset']
        X = data_model['X']
        model = data_model['model']
        
        # Generar probabilidades
        probabilities = model.predict_proba(X)[:, 1]  # Probabilidad de churn (clase 1)
        predictions = model.predict(X)
        
        # Crear DataFrame con resultados
        results_df = df.copy()
        results_df['Probabilidad_Churn'] = probabilities
        results_df['Prediccion_Churn'] = predictions
        results_df['Cliente_ID'] = range(1, len(results_df) + 1)
        
        # Estadísticas de las probabilidades
        prob_stats = {
            'min': float(probabilities.min()),
            'max': float(probabilities.max()),
            'mean': float(probabilities.mean()),
            'median': float(np.median(probabilities)),
            'std': float(probabilities.std()),
            'percentile_25': float(np.percentile(probabilities, 25)),
            'percentile_75': float(np.percentile(probabilities, 75))
        }
        
        logging.info(f"Probabilidades generadas para {len(results_df):,} clientes")
        logging.info(f"Probabilidad promedio: {prob_stats['mean']:.1%}")
        logging.info(f"Percentil 25: {prob_stats['percentile_25']:.1%}")
        logging.info(f"Percentil 75: {prob_stats['percentile_75']:.1%}")
        
        return {
            'results_df': results_df,
            'probabilities': probabilities,
            'predictions': predictions,
            'prob_stats': prob_stats
        }
        
    except Exception as e:
        logging.error(f"Error generando probabilidades: {str(e)}")
        raise

def create_risk_segments(prob_data):
    """Crear segmentos de riesgo basados en percentiles"""
    try:
        logging.info("Creando segmentos de riesgo...")
        
        results_df = prob_data['results_df']
        prob_stats = prob_data['prob_stats']
        
        # Definir umbrales basados en percentiles 25% y 75%
        threshold_low = prob_stats['percentile_25']
        threshold_high = prob_stats['percentile_75']
        
        # Crear segmentos
        def assign_risk_segment(prob):
            if prob <= threshold_low:
                return 'Bajo_Riesgo'
            elif prob <= threshold_high:
                return 'Medio_Riesgo'
            else:
                return 'Alto_Riesgo'
        
        results_df['Segmento_Riesgo'] = results_df['Probabilidad_Churn'].apply(assign_risk_segment)
        
        # Análisis por segmento
        segment_analysis = {}
        
        for segment in ['Bajo_Riesgo', 'Medio_Riesgo', 'Alto_Riesgo']:
            segment_data = results_df[results_df['Segmento_Riesgo'] == segment]
            
            if len(segment_data) > 0:
                segment_analysis[segment] = {
                    'count': len(segment_data),
                    'percentage': len(segment_data) / len(results_df) * 100,
                    'avg_probability': segment_data['Probabilidad_Churn'].mean(),
                    'min_probability': segment_data['Probabilidad_Churn'].min(),
                    'max_probability': segment_data['Probabilidad_Churn'].max(),
                    'actual_churn_rate': segment_data['Abandono_Cliente'].mean(),
                    'predicted_churn_rate': segment_data['Prediccion_Churn'].mean(),
                    'avg_months_tenure': segment_data['Meses_Cliente'].mean(),
                    'avg_total_charges': segment_data['Cargo_Total'].mean()
                }
        
        # Validación de segmentación
        total_clients = sum([seg['count'] for seg in segment_analysis.values()])
        
        logging.info("SEGMENTACIÓN COMPLETADA:")
        for segment, data in segment_analysis.items():
            logging.info(f"  {segment}: {data['count']:,} clientes ({data['percentage']:.1f}%)")
            logging.info(f"    Prob. promedio: {data['avg_probability']:.1%}")
            logging.info(f"    Churn real: {data['actual_churn_rate']:.1%}")
        
        return {
            'segmented_df': results_df,
            'segment_analysis': segment_analysis,
            'thresholds': {
                'low': threshold_low,
                'high': threshold_high
            },
            'total_clients': total_clients
        }
        
    except Exception as e:
        logging.error(f"Error creando segmentos: {str(e)}")
        raise

def define_segment_strategies(segment_data, critical_factors):
    """Definir estrategias específicas por segmento de riesgo"""
    try:
        logging.info("Definiendo estrategias por segmento...")
        
        segment_analysis = segment_data['segment_analysis']
        
        # Estrategias por segmento
        strategies = {
            'Bajo_Riesgo': {
                'objetivo': 'Mantener satisfacción y fidelizar',
                'prioridad': 'Baja',
                'enfoque': 'Preventivo',
                'frecuencia_contacto': 'Trimestral',
                'canal_preferido': 'Digital/Automated',
                'inversion_recomendada': 'Baja',
                'acciones_principales': [
                    'Programas de fidelización y recompensas',
                    'Comunicación proactiva sobre nuevos servicios',
                    'Encuestas de satisfacción automatizadas',
                    'Ofertas de upgrade personalizadas',
                    'Newsletter mensual con tips y novedades'
                ],
                'metricas_kpi': [
                    'NPS (Net Promoter Score)',
                    'CSAT (Customer Satisfaction)',
                    'Tasa de adopción de nuevos servicios',
                    'Lifetime Value progression',
                    'Engagement rate con comunicaciones'
                ],
                'budget_asignado': '10-15%',
                'roi_esperado': '300-500%',
                'tiempo_implementacion': '1-2 meses'
            },
            'Medio_Riesgo': {
                'objetivo': 'Intervención proactiva y retención',
                'prioridad': 'Media-Alta',
                'enfoque': 'Proactivo',
                'frecuencia_contacto': 'Mensual',
                'canal_preferido': 'Mixto (Digital + Humano)',
                'inversion_recomendada': 'Media',
                'acciones_principales': [
                    'Outreach proactivo del equipo de retención',
                    'Análisis personalizado de uso y necesidades',
                    'Ofertas de ajuste de plan o descuentos',
                    'Mejoras en servicios identificados como problemáticos',
                    'Programas de educación sobre beneficios del servicio'
                ],
                'metricas_kpi': [
                    'Tasa de retención post-intervención',
                    'Tiempo promedio para resolución de issues',
                    'Satisfacción post-contacto',
                    'Conversion rate de ofertas de retención',
                    'Reducción en scoring de riesgo'
                ],
                'budget_asignado': '40-50%',
                'roi_esperado': '200-400%',
                'tiempo_implementacion': '2-3 meses'
            },
            'Alto_Riesgo': {
                'objetivo': 'Retención inmediata y recuperación',
                'prioridad': 'Crítica',
                'enfoque': 'Reactivo/Intensivo',
                'frecuencia_contacto': 'Semanal/Bi-semanal',
                'canal_preferido': 'Humano (Phone + Face-to-face)',
                'inversion_recomendada': 'Alta',
                'acciones_principales': [
                    'Intervención inmediata del senior retention team',
                    'Ofertas especiales y descuentos significativos',
                    'Resolución expedita de problemas técnicos',
                    'Migración a planes más adecuados',
                    'Follow-up intensivo durante 90 días',
                    'Escalación a management para casos críticos'
                ],
                'metricas_kpi': [
                    'Tasa de "save" (clientes retenidos)',
                    'Tiempo de respuesta a escalaciones',
                    'Customer effort score',
                    'Retention cost per customer',
                    'Post-save satisfaction score'
                ],
                'budget_asignado': '35-40%',
                'roi_esperado': '150-300%',
                'tiempo_implementacion': '1 mes (urgente)'
            }
        }
        
        # Enriquecer estrategias con insights de factores críticos
        for segment, strategy in strategies.items():
            # Agregar factores críticos relevantes
            relevant_factors = []
            for factor in critical_factors['critical_factors'][:5]:
                if factor['actionability_score'] >= 6:  # Factores accionables
                    relevant_factors.append({
                        'factor': factor['variable'],
                        'importancia': factor['avg_importance'],
                        'accionabilidad': factor['actionability_score'],
                        'tiempo_impacto': factor['time_to_impact']
                    })
            
            strategy['factores_criticos_accionables'] = relevant_factors
            
            # Calcular métricas financieras esperadas
            segment_info = segment_analysis.get(segment, {})
            if segment_info:
                cliente_count = segment_info['count']
                avg_revenue = segment_info.get('avg_total_charges', 0)
                
                strategy['metricas_financieras'] = {
                    'clientes_objetivo': cliente_count,
                    'revenue_promedio_cliente': avg_revenue,
                    'revenue_total_segmento': cliente_count * avg_revenue,
                    'churn_rate_actual': segment_info.get('actual_churn_rate', 0),
                    'probabilidad_promedio': segment_info.get('avg_probability', 0)
                }
        
        logging.info("Estrategias definidas para todos los segmentos")
        return strategies
        
    except Exception as e:
        logging.error(f"Error definiendo estrategias: {str(e)}")
        raise

def create_segment_visualization(segment_data, strategies, timestamp):
    """Crear visualizaciones de segmentación estratégica"""
    try:
        logging.info("Creando visualizaciones de segmentación...")
        
        # Debug: verificar datos de entrada
        segment_analysis = segment_data['segment_analysis']
        logging.info(f"Segmentos disponibles: {list(segment_analysis.keys())}")
        
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.93, bottom=0.07, left=0.07, right=0.95)
        
        # 1. Distribución de clientes por segmento
        segments = list(segment_analysis.keys())
        counts = [segment_analysis[seg]['count'] for seg in segments]
        percentages = [segment_analysis[seg]['percentage'] for seg in segments]
        
        colors_segments = {'Bajo_Riesgo': '#2E8B57', 'Medio_Riesgo': '#FFD700', 'Alto_Riesgo': '#DC143C'}
        colors = [colors_segments[seg] for seg in segments]
        
        wedges, texts, autotexts = ax1.pie(counts, labels=segments, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title('Distribución de Clientes por Segmento de Riesgo', fontweight='bold', fontsize=14, pad=20)
        
        # Agregar total en el centro
        ax1.text(0, 0, f'Total:\n{sum(counts):,}\nclientes', ha='center', va='center', 
                fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Probabilidades promedio por segmento
        avg_probs = [segment_analysis[seg]['avg_probability'] * 100 for seg in segments]
        
        bars = ax2.bar(segments, avg_probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_ylabel('Probabilidad Promedio de Churn (%)', fontweight='bold', fontsize=12)
        ax2.set_title('Probabilidad Promedio de Churn por Segmento', fontweight='bold', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, prob in zip(bars, avg_probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 3. Churn real vs Probabilidad promedio por segmento
        actual_churn = [segment_analysis[seg]['actual_churn_rate'] * 100 for seg in segments]
        avg_probability = [segment_analysis[seg]['avg_probability'] * 100 for seg in segments]
        
        x = np.arange(len(segments))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, actual_churn, width, label='Churn Real', color='#FF6347', alpha=0.8)
        bars2 = ax3.bar(x + width/2, avg_probability, width, label='Probabilidad Promedio', color='#4682B4', alpha=0.8)
        
        ax3.set_ylabel('Porcentaje (%)', fontweight='bold', fontsize=12)
        ax3.set_title('Churn Real vs Probabilidad Promedio por Segmento', fontweight='bold', fontsize=14, pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(segments)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Inversión recomendada por segmento
        budget_percentages = []
        segment_names_short = []
        
        for segment in segments:
            strategy = strategies[segment]
            budget_range = strategy['budget_asignado']
            # Tomar el promedio del rango
            if '-' in budget_range:
                low, high = budget_range.replace('%', '').split('-')
                budget_avg = (float(low) + float(high)) / 2
            else:
                budget_avg = float(budget_range.replace('%', ''))
            
            budget_percentages.append(budget_avg)
            segment_names_short.append(segment.replace('_', '\n'))
        
        bars = ax4.bar(segment_names_short, budget_percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylabel('Asignación de Budget (%)', fontweight='bold', fontsize=12)
        ax4.set_title('Asignación Recomendada de Budget por Segmento', fontweight='bold', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, budget in zip(bars, budget_percentages):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{budget:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Título principal
        fig.suptitle('TelecomX - Análisis de Segmentación Estratégica por Riesgo de Churn', 
                    fontsize=18, fontweight='bold', y=0.97)
        
        # Asegurar que el directorio existe
        os.makedirs('graficos', exist_ok=True)
        
        # Guardar visualización
        viz_file = f'graficos/paso13b_segmentacion_estrategica_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        # Verificar que el archivo se creó
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Visualización guardada exitosamente: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear el archivo de visualización: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando visualización: {str(e)}")
        return None

def create_excel_dashboard(segment_data, strategies, critical_factors, timestamp):
    """Crear dashboard ejecutivo en Excel"""
    try:
        logging.info("Creando dashboard ejecutivo en Excel...")
        
        excel_file = f'excel/paso13b_dashboard_segmentacion_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # 1. Hoja: Resumen Ejecutivo
            summary_data = []
            segment_analysis = segment_data['segment_analysis']
            
            for segment, data in segment_analysis.items():
                strategy = strategies[segment]
                summary_data.append({
                    'Segmento': segment,
                    'Clientes': data['count'],
                    'Porcentaje': f"{data['percentage']:.1f}%",
                    'Prob_Churn_Promedio': f"{data['avg_probability']:.1%}",
                    'Churn_Real': f"{data['actual_churn_rate']:.1%}",
                    'Prioridad': strategy['prioridad'],
                    'Budget_Asignado': strategy['budget_asignado'],
                    'ROI_Esperado': strategy['roi_esperado'],
                    'Tiempo_Implementacion': strategy['tiempo_implementacion']
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen Ejecutivo', index=False)
            
            # 2. Hoja: Segmentación Detallada
            segmented_df = segment_data['segmented_df'][['Cliente_ID', 'Probabilidad_Churn', 'Prediccion_Churn', 
                                                        'Segmento_Riesgo', 'Abandono_Cliente', 'Meses_Cliente', 
                                                        'Cargo_Total']]
            segmented_df.to_excel(writer, sheet_name='Segmentación Clientes', index=False)
            
            # 3. Hoja: Estrategias por Segmento
            strategies_data = []
            for segment, strategy in strategies.items():
                strategies_data.append({
                    'Segmento': segment,
                    'Objetivo': strategy['objetivo'],
                    'Enfoque': strategy['enfoque'],
                    'Frecuencia_Contacto': strategy['frecuencia_contacto'],
                    'Canal_Preferido': strategy['canal_preferido'],
                    'Acción_Principal_1': strategy['acciones_principales'][0] if strategy['acciones_principales'] else '',
                    'Acción_Principal_2': strategy['acciones_principales'][1] if len(strategy['acciones_principales']) > 1 else '',
                    'KPI_Principal_1': strategy['metricas_kpi'][0] if strategy['metricas_kpi'] else '',
                    'KPI_Principal_2': strategy['metricas_kpi'][1] if len(strategy['metricas_kpi']) > 1 else ''
                })
            
            strategies_df = pd.DataFrame(strategies_data)
            strategies_df.to_excel(writer, sheet_name='Estrategias', index=False)
            
            # 4. Hoja: Factores Críticos
            factors_data = []
            for factor in critical_factors['critical_factors'][:10]:
                factors_data.append({
                    'Variable': factor['variable'],
                    'Importancia': f"{factor['avg_importance']:.1f}%",
                    'Accionabilidad': f"{factor['actionability_score']}/10",
                    'Categoría': factor['category'],
                    'Tiempo_Impacto': f"{factor['time_to_impact']} meses",
                    'Complejidad': factor['implementation_complexity']
                })
            
            factors_df = pd.DataFrame(factors_data)
            factors_df.to_excel(writer, sheet_name='Factores Críticos', index=False)
            
            # 5. Hoja: Métricas por Segmento
            metrics_data = []
            for segment, data in segment_analysis.items():
                metrics_data.append({
                    'Segmento': segment,
                    'Total_Clientes': data['count'],
                    'Prob_Min': f"{data['min_probability']:.1%}",
                    'Prob_Max': f"{data['max_probability']:.1%}",
                    'Prob_Promedio': f"{data['avg_probability']:.1%}",
                    'Meses_Promedio': f"{data['avg_months_tenure']:.1f}",
                    'Cargo_Promedio': f"${data['avg_total_charges']:.2f}",
                    'Revenue_Total_Segmento': f"${data['count'] * data['avg_total_charges']:,.2f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Métricas Detalladas', index=False)
        
        logging.info(f"Dashboard Excel creado: {excel_file}")
        return excel_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard Excel: {str(e)}")
        return None

def save_segmentation_results(segment_data, strategies, critical_factors, prob_data, timestamp):
    """Guardar resultados de segmentación"""
    try:
        logging.info("Guardando resultados de segmentación...")
        
        # 1. CSV con segmentación de clientes
        csv_file = f'excel/paso13b_segmentacion_clientes_{timestamp}.csv'
        segmented_df = segment_data['segmented_df']
        
        # Seleccionar columnas relevantes para el CSV
        csv_columns = ['Cliente_ID', 'Probabilidad_Churn', 'Prediccion_Churn', 'Segmento_Riesgo', 
                      'Abandono_Cliente', 'Meses_Cliente', 'Cargo_Total', 'Tipo_Contrato_encoded',
                      'Servicio_Internet_Fibra Óptica', 'Soporte_Tecnico_No']
        
        csv_df = segmented_df[csv_columns].copy()
        csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 2. JSON con datos completos para scripts posteriores
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'script': 'paso13b_Segmentación_Estratégica',
                'version': '1.0',
                'total_clients': segment_data['total_clients'],
                'segmentation_method': 'Risk-based (percentiles 25-75)'
            },
            'segmentation_summary': segment_data['segment_analysis'],
            'thresholds': segment_data['thresholds'],
            'strategies': strategies,
            'probability_stats': prob_data['prob_stats'],
            'critical_factors_used': critical_factors['critical_factors'][:10],
            'model_info': critical_factors['model_info']
        }
        
        json_file = f'informes/paso13b_segmentacion_estrategica_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 3. TXT con informe ejecutivo
        txt_file = f'informes/paso13b_segmentacion_estrategica_{timestamp}.txt'
        report_content = generate_executive_report(segment_data, strategies, critical_factors, prob_data, timestamp)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"CSV segmentación guardado: {csv_file}")
        logging.info(f"JSON datos completos guardado: {json_file}")
        logging.info(f"Informe ejecutivo guardado: {txt_file}")
        
        return {
            'csv_file': csv_file,
            'json_file': json_file,
            'txt_file': txt_file
        }
        
    except Exception as e:
        logging.error(f"Error guardando resultados: {str(e)}")
        raise

def generate_executive_report(segment_data, strategies, critical_factors, prob_data, timestamp):
    """Generar informe ejecutivo de segmentación"""
    
    segment_analysis = segment_data['segment_analysis']
    prob_stats = prob_data['prob_stats']
    thresholds = segment_data['thresholds']
    
    report = f"""
================================================================================
TELECOMX - PASO 13B: SEGMENTACIÓN ESTRATÉGICA BASADA EN RIESGO - INFORME EJECUTIVO
================================================================================
Fecha: {timestamp}
Script: paso13b_Segmentación_Estratégica.py

================================================================================
RESUMEN EJECUTIVO
================================================================================

🎯 OBJETIVO CUMPLIDO:
• Segmentación de {segment_data['total_clients']:,} clientes en 3 grupos de riesgo
• Estrategias específicas definidas por segmento
• Métricas y KPIs establecidos para seguimiento

📊 MODELO UTILIZADO:
• Tipo: {critical_factors['model_info']['best_model']}
• Performance: {critical_factors['model_info']['model_score']:.4f}
• Método de segmentación: Percentiles (25% - 75%)

================================================================================
DISTRIBUCIÓN DE SEGMENTOS
================================================================================

🟢 SEGMENTO BAJO RIESGO:
• Clientes: {segment_analysis.get('Bajo_Riesgo', {}).get('count', 0):,} ({segment_analysis.get('Bajo_Riesgo', {}).get('percentage', 0):.1f}%)
• Probabilidad promedio: {segment_analysis.get('Bajo_Riesgo', {}).get('avg_probability', 0):.1%}
• Rango probabilidad: 0% - {thresholds['low']:.1%}
• Churn real: {segment_analysis.get('Bajo_Riesgo', {}).get('actual_churn_rate', 0):.1%}
• Revenue promedio: ${segment_analysis.get('Bajo_Riesgo', {}).get('avg_total_charges', 0):,.2f}

🟡 SEGMENTO MEDIO RIESGO:
• Clientes: {segment_analysis.get('Medio_Riesgo', {}).get('count', 0):,} ({segment_analysis.get('Medio_Riesgo', {}).get('percentage', 0):.1f}%)
• Probabilidad promedio: {segment_analysis.get('Medio_Riesgo', {}).get('avg_probability', 0):.1%}
• Rango probabilidad: {thresholds['low']:.1%} - {thresholds['high']:.1%}
• Churn real: {segment_analysis.get('Medio_Riesgo', {}).get('actual_churn_rate', 0):.1%}
• Revenue promedio: ${segment_analysis.get('Medio_Riesgo', {}).get('avg_total_charges', 0):,.2f}

🔴 SEGMENTO ALTO RIESGO:
• Clientes: {segment_analysis.get('Alto_Riesgo', {}).get('count', 0):,} ({segment_analysis.get('Alto_Riesgo', {}).get('percentage', 0):.1f}%)
• Probabilidad promedio: {segment_analysis.get('Alto_Riesgo', {}).get('avg_probability', 0):.1%}
• Rango probabilidad: {thresholds['high']:.1%} - 100%
• Churn real: {segment_analysis.get('Alto_Riesgo', {}).get('actual_churn_rate', 0):.1%}
• Revenue promedio: ${segment_analysis.get('Alto_Riesgo', {}).get('avg_total_charges', 0):,.2f}

================================================================================
ESTRATEGIAS POR SEGMENTO
================================================================================
"""

    # Agregar estrategias detalladas por segmento
    segment_icons = {'Bajo_Riesgo': '🟢', 'Medio_Riesgo': '🟡', 'Alto_Riesgo': '🔴'}
    
    for segment, strategy in strategies.items():
        icon = segment_icons.get(segment, '•')
        segment_info = segment_analysis.get(segment, {})
        
        report += f"""
{icon} ESTRATEGIA {segment.replace('_', ' ').upper()}:

📋 CARACTERÍSTICAS:
• Objetivo: {strategy['objetivo']}
• Prioridad: {strategy['prioridad']}
• Enfoque: {strategy['enfoque']}
• Frecuencia de contacto: {strategy['frecuencia_contacto']}
• Canal preferido: {strategy['canal_preferido']}
• Inversión recomendada: {strategy['inversion_recomendada']}

💰 MÉTRICAS FINANCIERAS:
• Budget asignado: {strategy['budget_asignado']} del total
• ROI esperado: {strategy['roi_esperado']}
• Tiempo implementación: {strategy['tiempo_implementacion']}
• Revenue total segmento: ${segment_info.get('count', 0) * segment_info.get('avg_total_charges', 0):,.2f}

🎯 ACCIONES PRINCIPALES:
"""
        for i, accion in enumerate(strategy['acciones_principales'], 1):
            report += f"   {i}. {accion}\n"
        
        report += f"""
📊 KPIs DE SEGUIMIENTO:
"""
        for i, kpi in enumerate(strategy['metricas_kpi'], 1):
            report += f"   {i}. {kpi}\n"
        
        report += f"""
⚡ FACTORES CRÍTICOS ACCIONABLES:
"""
        for i, factor in enumerate(strategy.get('factores_criticos_accionables', [])[:3], 1):
            report += f"   {i}. {factor['factor']} (Importancia: {factor['importancia']:.1f}%, Accionabilidad: {factor['accionabilidad']}/10)\n"
        
        report += "\n"

    report += f"""
================================================================================
ANÁLISIS DE ESTADÍSTICAS DE PROBABILIDADES
================================================================================

📊 DISTRIBUCIÓN GENERAL:
• Probabilidad mínima: {prob_stats['min']:.1%}
• Probabilidad máxima: {prob_stats['max']:.1%}
• Probabilidad promedio: {prob_stats['mean']:.1%}
• Mediana: {prob_stats['median']:.1%}
• Desviación estándar: {prob_stats['std']:.1%}

🎯 UMBRALES DE SEGMENTACIÓN:
• Threshold Bajo-Medio: {thresholds['low']:.1%} (Percentil 25)
• Threshold Medio-Alto: {thresholds['high']:.1%} (Percentil 75)

================================================================================
FACTORES CRÍTICOS PARA RETENCIÓN
================================================================================

🔥 TOP 5 FACTORES MÁS IMPORTANTES:
"""
    
    for i, factor in enumerate(critical_factors['critical_factors'][:5], 1):
        actionability_icon = "🟢" if factor['actionability_score'] >= 7 else "🟡" if factor['actionability_score'] >= 5 else "🔴"
        report += f"""
{i}. {factor['variable'].upper()}:
   📈 Importancia: {factor['avg_importance']:.2f}%
   ⚡ Accionabilidad: {actionability_icon} {factor['actionability_score']}/10
   🏢 Categoría: {factor['category']}
   ⏱️ Tiempo impacto: {factor['time_to_impact']} meses
   📋 Complejidad: {factor['implementation_complexity']}
"""

    report += f"""

================================================================================
RECOMENDACIONES INMEDIATAS
================================================================================

🚀 ACCIONES PRIORITARIAS (PRÓXIMOS 30 DÍAS):

1. SEGMENTO ALTO RIESGO - INTERVENCIÓN INMEDIATA:
   • Activar equipo de retención senior para {segment_analysis.get('Alto_Riesgo', {}).get('count', 0):,} clientes
   • Implementar outreach telefónico personalizado
   • Preparar ofertas especiales y descuentos
   • Establecer seguimiento semanal durante 90 días

2. SEGMENTO MEDIO RIESGO - PREVENCIÓN PROACTIVA:
   • Desarrollar campañas de outreach proactivo
   • Analizar patrones de uso y necesidades no cubiertas
   • Implementar programa de educación sobre beneficios
   • Configurar alertas automáticas para escalación

3. SEGMENTO BAJO RIESGO - FIDELIZACIÓN:
   • Implementar programa de recompensas y fidelización
   • Establecer comunicación trimestral automatizada
   • Identificar oportunidades de upselling y cross-selling
   • Monitorear NPS y satisfacción

================================================================================
PLAN DE IMPLEMENTACIÓN
================================================================================

📅 FASE 1 (MES 1): IMPLEMENTACIÓN CRÍTICA
• Activación inmediata para segmento Alto Riesgo
• Configuración de dashboards y alertas
• Entrenamiento de equipos de retención
• Setup de métricas y KPIs

📅 FASE 2 (MES 2-3): EXPANSIÓN PROACTIVA
• Implementación completa para segmento Medio Riesgo
• Optimización de procesos basada en resultados iniciales
• Desarrollo de contenido y materiales de comunicación
• Análisis de efectividad y ajustes

📅 FASE 3 (MES 4-6): OPTIMIZACIÓN Y SCALING
• Implementación para segmento Bajo Riesgo
• Automatización de procesos repetitivos
• Análisis ROI y refinamiento de estrategias
• Preparación para siguiente ciclo de optimización

================================================================================
MÉTRICAS DE ÉXITO ESPERADAS
================================================================================

🎯 OBJETIVOS A 6 MESES:

SEGMENTO ALTO RIESGO:
• Reducir churn rate en 40-60%
• Aumentar "save rate" a 25-35%
• ROI target: 150-300%

SEGMENTO MEDIO RIESGO:
• Reducir progression a Alto Riesgo en 50%
• Aumentar satisfacción en 20-30%
• ROI target: 200-400%

SEGMENTO BAJO RIESGO:
• Mantener churn rate < 5%
• Aumentar NPS en 15-25%
• ROI target: 300-500%

================================================================================
PRÓXIMOS PASOS
================================================================================

📋 SIGUIENTES SCRIPTS DEL PASO 13:

1. paso13c_Business_Case_Completo.py:
   • Calculará ROI específico por segmento y estrategia
   • Estimará impacto financiero detallado de cada intervención
   • Desarrollará business case para aprobación de inversiones

2. paso13d_Roadmap_Detallado.py:
   • Creará cronograma detallado de implementación
   • Definirá milestones y checkpoints críticos
   • Establecerá plan de contingencia y manejo de riesgos

3. paso13e_Outputs_Ejecutivos.py:
   • Generará dashboards ejecutivos finales
   • Creará presentaciones para stakeholders
   • Desarrollará KPI tracking automático

📊 ARCHIVOS GENERADOS:
• Segmentación clientes CSV: excel/paso13b_segmentacion_clientes_{timestamp}.csv
• Datos completos JSON: informes/paso13b_segmentacion_estrategica_{timestamp}.json
• Dashboard Excel: excel/paso13b_dashboard_segmentacion_{timestamp}.xlsx
• Visualizaciones: graficos/paso13b_segmentacion_estrategica_{timestamp}.png
• Este informe: informes/paso13b_segmentacion_estrategica_{timestamp}.txt

================================================================================
CONCLUSIÓN
================================================================================

✅ SEGMENTACIÓN ESTRATÉGICA COMPLETADA EXITOSAMENTE:

• {segment_data['total_clients']:,} clientes segmentados en 3 grupos de riesgo bien diferenciados
• Estrategias específicas y accionables definidas para cada segmento
• Métricas y KPIs establecidos para seguimiento efectivo
• Plan de implementación por fases con objetivos claros
• Base sólida para desarrollar business case detallado

🎯 SEGMENTO PRIORITARIO: Alto Riesgo ({segment_analysis.get('Alto_Riesgo', {}).get('count', 0):,} clientes)
⚡ OPORTUNIDAD PRINCIPAL: Prevención proactiva en Medio Riesgo
💰 REVENUE EN RIESGO: ${sum([seg.get('count', 0) * seg.get('avg_total_charges', 0) for seg in segment_analysis.values()]):,.2f}

La segmentación estratégica está lista para generar business case detallado
e implementación efectiva con alto potencial de ROI.

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def main():
    """Función principal del Paso 13B - Segmentación Estratégica"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13B: SEGMENTACIÓN ESTRATÉGICA BASADA EN RIESGO")
        logger.info("="*80)
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar factores críticos del Paso 13A
        logger.info("="*50)
        logger.info("CARGANDO FACTORES CRÍTICOS DEL PASO 13A")
        critical_factors = load_critical_factors()
        
        # 3. Cargar dataset y modelo
        logger.info("="*50)
        logger.info("CARGANDO DATASET Y MODELO ENTRENADO")
        data_model = load_dataset_and_model()
        
        # 4. Generar probabilidades de churn
        logger.info("="*50)
        logger.info("GENERANDO PROBABILIDADES DE CHURN")
        prob_data = generate_churn_probabilities(data_model)
        
        # 5. Crear segmentos de riesgo
        logger.info("="*50)
        logger.info("CREANDO SEGMENTOS DE RIESGO")
        segment_data = create_risk_segments(prob_data)
        
        # 6. Definir estrategias por segmento
        logger.info("="*50)
        logger.info("DEFINIENDO ESTRATEGIAS POR SEGMENTO")
        strategies = define_segment_strategies(segment_data, critical_factors)
        
        # 7. Crear visualizaciones
        logger.info("="*50)
        logger.info("CREANDO VISUALIZACIONES")
        viz_file = create_segment_visualization(segment_data, strategies, timestamp)
        
        # 8. Crear dashboard Excel
        logger.info("="*50)
        logger.info("CREANDO DASHBOARD EXCEL")
        excel_file = create_excel_dashboard(segment_data, strategies, critical_factors, timestamp)
        
        # 9. Guardar resultados
        logger.info("="*50)
        logger.info("GUARDANDO RESULTADOS")
        output_files = save_segmentation_results(segment_data, strategies, critical_factors, prob_data, timestamp)
        
        # 10. Resumen final
        logger.info("="*80)
        logger.info("PASO 13B COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar resultados principales
        segment_analysis = segment_data['segment_analysis']
        
        logger.info("🎯 SEGMENTACIÓN COMPLETADA:")
        for segment, data in segment_analysis.items():
            icon = "🟢" if segment == "Bajo_Riesgo" else "🟡" if segment == "Medio_Riesgo" else "🔴"
            logger.info(f"  {icon} {segment}: {data['count']:,} clientes ({data['percentage']:.1f}%) - Prob: {data['avg_probability']:.1%}")
        logger.info("")
        
        logger.info("💰 REVENUE POR SEGMENTO:")
        for segment, data in segment_analysis.items():
            total_revenue = data['count'] * data['avg_total_charges']
            logger.info(f"  • {segment}: ${total_revenue:,.2f}")
        logger.info("")
        
        logger.info("⚡ ESTRATEGIAS DEFINIDAS:")
        for segment, strategy in strategies.items():
            logger.info(f"  • {segment}: {strategy['prioridad']} prioridad - {strategy['budget_asignado']} budget")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • CSV segmentación: {output_files['csv_file']}")
        logger.info(f"  • JSON datos completos: {output_files['json_file']}")
        logger.info(f"  • Informe ejecutivo: {output_files['txt_file']}")
        if excel_file:
            logger.info(f"  • Dashboard Excel: {excel_file}")
        if viz_file:
            logger.info(f"  • Visualizaciones: {viz_file}")
        logger.info("")
        
        logger.info("📋 LISTO PARA PRÓXIMOS SCRIPTS:")
        logger.info("  • paso13c: Business case con ROI detallado")
        logger.info("  • paso13d: Roadmap de implementación")
        logger.info("  • paso13e: Outputs ejecutivos finales")
        logger.info("="*80)
        
        return {
            'segment_data': segment_data,
            'strategies': strategies,
            'output_files': output_files,
            'excel_file': excel_file,
            'viz_file': viz_file
        }
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()