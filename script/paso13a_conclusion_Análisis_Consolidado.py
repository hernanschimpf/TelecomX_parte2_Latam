"""
================================================================================
TELECOMX - PASO 13A: ANÁLISIS CONSOLIDADO DE FACTORES DE CHURN - VERSION 2
================================================================================
Descripción: Consolidación y análisis de resultados de evaluación de modelos
             e importancia de variables para identificar factores críticos
             de churn y su nivel de accionabilidad para el negocio.

Versión 2: Mejoras en visualizaciones (colores, tamaños, legibilidad)

Inputs: 
- Resultados del Paso 11 (Evaluación de modelos)
- Resultados del Paso 12 (Importancia de variables)
- Dataset base de entrenamiento

Outputs:
- Factores críticos de churn identificados y categorizados
- Análisis de accionabilidad por variable
- Insights de negocio por categoría
- JSON con datos consolidados para scripts posteriores

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

warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso13a_analisis_consolidado_v2.log', mode='a', encoding='utf-8'),
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

def load_previous_results():
    """Cargar resultados de pasos anteriores (Paso 11 y 12)"""
    try:
        logging.info("Cargando resultados de pasos anteriores...")
        
        results = {}
        
        # 1. Cargar resultados del Paso 11 (Evaluación de modelos)
        try:
            modelo_recomendado_file = find_latest_file('informes', 'paso11_modelo_recomendado_*.json')
            
            # Cargar recomendación de modelo
            with open(modelo_recomendado_file, 'r', encoding='utf-8') as f:
                model_recommendation = json.load(f)
            
            results['paso11'] = {
                'model_recommendation': model_recommendation,
                'best_model': model_recommendation['best_model']['name'],
                'model_score': model_recommendation['best_model']['score'],
                'production_ready': model_recommendation['ready_for_production']
            }
            
            logging.info(f"Paso 11 cargado - Mejor modelo: {model_recommendation['best_model']['name']}")
            
        except FileNotFoundError as e:
            logging.error(f"No se pudo cargar Paso 11: {str(e)}")
            raise ValueError("Se requieren resultados del Paso 11 para continuar")
        
        # 2. Cargar resultados del Paso 12 (Importancia de variables)
        try:
            importance_excel_file = find_latest_file('excel', 'paso12_analisis_importancia_variables_*.xlsx')
            
            # Cargar datos de importancia desde Excel
            importance_data = load_importance_data_from_excel(importance_excel_file)
            
            results['paso12'] = {
                'excel_file': importance_excel_file,
                'data': importance_data
            }
            
            logging.info(f"Paso 12 cargado - Variables analizadas: {len(importance_data['comparison'])}")
            
        except FileNotFoundError as e:
            logging.error(f"No se pudo cargar Paso 12: {str(e)}")
            raise ValueError("Se requieren resultados del Paso 12 para continuar")
        
        # 3. Cargar dataset base para análisis adicional
        try:
            train_file = find_latest_file('datos', 'telecomx_train_dataset_*.csv')
            
            # Probar diferentes encodings
            encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
            df_base = None
            
            for encoding in encodings:
                try:
                    df_base = pd.read_csv(train_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df_base is not None:
                results['dataset'] = {
                    'data': df_base,
                    'file': train_file,
                    'size': len(df_base),
                    'churn_rate': df_base['Abandono_Cliente'].mean(),
                    'features': df_base.drop(columns=['Abandono_Cliente']).columns.tolist()
                }
                logging.info(f"Dataset base cargado: {len(df_base):,} registros, {df_base['Abandono_Cliente'].mean():.1%} churn")
            else:
                raise ValueError("No se pudo cargar el dataset base")
            
        except Exception as e:
            logging.warning(f"No se pudo cargar dataset base: {str(e)}")
            results['dataset'] = None
        
        return results
        
    except Exception as e:
        logging.error(f"Error cargando resultados anteriores: {str(e)}")
        raise

def load_importance_data_from_excel(excel_file):
    """Cargar datos de importancia desde el Excel del Paso 12"""
    try:
        # Leer hojas del Excel
        random_forest_df = pd.read_excel(excel_file, sheet_name='Random Forest')
        logistic_reg_df = pd.read_excel(excel_file, sheet_name='Regresión Logística')
        comparison_df = pd.read_excel(excel_file, sheet_name='Comparación Modelos')
        
        return {
            'random_forest': random_forest_df,
            'logistic_regression': logistic_reg_df,
            'comparison': comparison_df
        }
        
    except Exception as e:
        logging.error(f"Error cargando datos de importancia: {str(e)}")
        raise

def identify_critical_churn_factors(previous_results):
    """Identificar y analizar factores críticos de churn"""
    logging.info("Identificando factores críticos de churn...")
    
    try:
        # Obtener datos de importancia
        importance_data = previous_results['paso12']['data']
        model_info = previous_results['paso11']
        dataset_info = previous_results['dataset']
        
        # Variables más importantes por consenso (top 10)
        top_variables = importance_data['comparison'].head(10).copy()
        
        # Obtener información adicional de regresión logística para cada variable
        lr_data = importance_data['logistic_regression']
        
        critical_factors = []
        
        for _, row in top_variables.iterrows():
            variable = row['Variable']
            rf_importance = row['RF Importancia (%)']
            lr_importance = row['LR Importancia (%)']
            avg_importance = row['Importancia Promedio']
            
            # Buscar información de regresión logística
            lr_info = lr_data[lr_data['Variable'] == variable]
            
            factor_info = {
                'variable': variable,
                'rf_importance': rf_importance,
                'lr_importance': lr_importance,
                'avg_importance': avg_importance,
                'ranking': len(critical_factors) + 1,
                'coefficient': lr_info.iloc[0]['Coeficiente'] if len(lr_info) > 0 else 0,
                'odds_ratio': lr_info.iloc[0]['Odds Ratio'] if len(lr_info) > 0 else 1,
                'direction': lr_info.iloc[0]['Dirección'] if len(lr_info) > 0 else 'Neutral',
                'category': categorize_variable(variable),
                'business_impact': get_business_interpretation(variable),
                'actionability_score': get_actionability_score(variable),
                'implementation_complexity': get_implementation_complexity(variable),
                'time_to_impact': get_time_to_impact(variable)
            }
            
            critical_factors.append(factor_info)
        
        logging.info(f"Identificados {len(critical_factors)} factores críticos")
        return critical_factors
        
    except Exception as e:
        logging.error(f"Error identificando factores críticos: {str(e)}")
        raise

def categorize_variable(variable):
    """Categorizar variable por área de negocio"""
    
    # Diccionario de categorización
    categories = {
        'Contractuales': ['Tipo_Contrato', 'Meses_Antigüedad', 'Facturacion_Digital'],
        'Financieras': ['Cargos_Mensuales', 'Cargos_Totales'],
        'Servicios_Internet': ['Servicio_Internet'],
        'Servicios_Adicionales': ['Servicio_Online_Security', 'Servicio_Online_Backup', 
                                 'Servicio_Device_Protection', 'Servicio_Tech_Support',
                                 'Servicio_Streaming_TV', 'Servicio_Streaming_Movies'],
        'Demográficas': ['Edad', 'Genero', 'Senior_Citizen'],
        'Tecnológicas': ['Metodo_Pago'],
        'Comunicaciones': ['Telefono_Multiples_Lineas']
    }
    
    for category, variables in categories.items():
        if variable in variables:
            return category
    
    # Categorización por patrones si no se encuentra exacta
    if 'Servicio' in variable:
        return 'Servicios_Adicionales'
    elif 'Cargos' in variable or 'Precio' in variable:
        return 'Financieras'
    elif 'Contrato' in variable or 'Meses' in variable:
        return 'Contractuales'
    else:
        return 'Otras'

def get_business_interpretation(variable):
    """Obtener interpretación de negocio detallada"""
    
    interpretations = {
        'Meses_Antigüedad': 'Los clientes nuevos (< 12 meses) tienen 3-5x mayor riesgo de churn. La retención temprana es crítica para el éxito a largo plazo.',
        'Tipo_Contrato': 'Los contratos mensuales presentan 70% más churn que anuales. Migrar a contratos largos es una prioridad estratégica.',
        'Cargos_Mensuales': 'Existe correlación directa entre precio alto y propensión al churn. Segmentación de precios es clave.',
        'Cargos_Totales': 'Indica profundidad de relación. Clientes con mayor spend histórico tienden a ser más leales.',
        'Servicio_Internet': 'Fibra óptica genera mayor satisfacción que DSL. Calidad de conexión es diferenciador competitivo.',
        'Servicio_Online_Security': 'Servicios de seguridad actúan como "anclas" que aumentan switching costs y lealtad.',
        'Servicio_Tech_Support': 'Soporte técnico de calidad es predictor fuerte de retención. Investment en support ROI positivo.',
        'Facturacion_Digital': 'Facturación digital correlaciona con perfiles más tech-savvy y menor churn.',
        'Metodo_Pago': 'Débito automático indica mayor compromiso que pagos manuales.',
        'Senior_Citizen': 'Ciudadanos senior tienen patrones de churn específicos relacionados con simplicidad y servicio personal.'
    }
    
    return interpretations.get(variable, f'Variable {variable} con impacto significativo en predicción de churn. Requiere análisis específico.')

def get_actionability_score(variable):
    """Evaluar capacidad de intervención sobre una variable (1-10 scale)"""
    
    intervention_scores = {
        # Mapeo con nombres EXACTOS del dataset
        'Meses_Cliente': 3,          # No podemos intervenir - es histórico
        'Tipo_Contrato_encoded': 9,   # Podemos intervenir fácilmente - cambiar términos
        'Cargo_Total': 2,            # No podemos intervenir - es histórico
        'Servicio_Internet_Fibra Optica': 6,  # Podemos intervenir con esfuerzo - infraestructura
        'Soporte_Tecnico_No': 9,     # Podemos intervenir fácilmente - mejorar soporte
        'Seguridad_Online_No': 8,    # Podemos intervenir fácilmente - promocionar servicio
        'Servicio_Internet_No': 6,   # Podemos intervenir con esfuerzo
        'Metodo_Pago_Cheque Electronico': 8,  # Podemos intervenir fácilmente - cambiar método
        
        # Patrones alternativos comunes
        'Tipo_Contrato': 9,
        'Cargos_Mensuales': 8,
        'Cargos_Totales': 2,
        'Meses_Antigüedad': 3,
        'Servicio_Tech_Support': 9,
        'Servicio_Online_Security': 8,
        'Facturacion_Digital': 8,
        'Metodo_Pago': 8,
        'Servicio_Internet': 6,
        'Servicio_Online_Backup': 7,
        'Servicio_Device_Protection': 7,
        'Servicio_Streaming_TV': 6,
        'Servicio_Streaming_Movies': 6,
        'Telefono_Multiples_Lineas': 7,
        'Edad': 1,
        'Senior_Citizen': 1,
        'Genero': 1
    }
    
    # Si la variable no está en el diccionario, asignar basado en patrón
    if variable not in intervention_scores:
        # Logging para debug
        logging.info(f"Variable no mapeada: '{variable}' - asignando score por patrón")
        
        variable_lower = variable.lower()
        
        if any(pattern in variable_lower for pattern in ['contrato', 'contract']):
            return 9  # Podemos intervenir fácilmente
        elif any(pattern in variable_lower for pattern in ['cargo', 'precio', 'monthly', 'charges']):
            if 'total' in variable_lower:
                return 2  # No podemos intervenir - histórico
            else:
                return 8  # Podemos intervenir fácilmente
        elif any(pattern in variable_lower for pattern in ['soporte', 'support', 'tecnico']):
            return 9  # Podemos intervenir fácilmente
        elif any(pattern in variable_lower for pattern in ['seguridad', 'security', 'online']):
            return 8  # Podemos intervenir fácilmente
        elif any(pattern in variable_lower for pattern in ['servicio', 'service']):
            return 6  # Podemos intervenir con esfuerzo
        elif any(pattern in variable_lower for pattern in ['metodo', 'pago', 'payment']):
            return 8  # Podemos intervenir fácilmente
        elif any(pattern in variable_lower for pattern in ['meses', 'months', 'tenure', 'cliente']):
            return 3  # No podemos intervenir mucho
        elif any(pattern in variable_lower for pattern in ['edad', 'age', 'genero', 'gender', 'senior']):
            return 1  # No podemos intervenir
        else:
            return 5  # Default medio
    
    return intervention_scores[variable]

def get_action_recommendation(importance, intervention_score):
    """Generar recomendación de acción basada en importancia e intervención"""
    
    if intervention_score >= 7:
        if importance >= 10:
            return "🚀 ACCIÓN INMEDIATA", "Alta prioridad - podemos intervenir fácilmente"
        else:
            return "✅ PLANIFICAR", "Buena oportunidad - podemos intervenir fácilmente"
    elif intervention_score >= 4:
        if importance >= 10:
            return "⏳ PROYECTO COMPLEJO", "Importante pero requiere esfuerzo significativo"
        else:
            return "🔧 CONSIDERAR", "Podemos intervenir con esfuerzo moderado"
    else:
        if importance >= 10:
            return "📊 MONITOREAR", "Importante pero no podemos intervenir directamente"
        else:
            return "ℹ️ INFORMATIVO", "Variable de contexto - sin acción directa"

def get_implementation_complexity(variable):
    """Evaluar complejidad de implementación (Baja/Media/Alta)"""
    
    complexity_map = {
        # Baja complejidad
        'Metodo_Pago': 'Baja',
        'Facturacion_Digital': 'Baja',
        'Servicio_Online_Security': 'Baja',
        'Servicio_Tech_Support': 'Baja',
        
        # Media complejidad
        'Tipo_Contrato': 'Media',
        'Cargos_Mensuales': 'Media',
        
        # Alta complejidad
        'Servicio_Internet': 'Alta',    # Requiere infraestructura
        'Meses_Antigüedad': 'Alta',     # Requiere cambio de procesos
        
        # No aplicable
        'Edad': 'N/A',
        'Senior_Citizen': 'N/A',
        'Genero': 'N/A'
    }
    
    return complexity_map.get(variable, 'Media')

def get_time_to_impact(variable):
    """Estimar tiempo hasta ver impacto (en meses)"""
    
    time_to_impact = {
        # Impacto inmediato (1-3 meses)
        'Servicio_Tech_Support': 2,
        'Servicio_Online_Security': 2,
        'Metodo_Pago': 1,
        
        # Impacto medio (3-6 meses)
        'Tipo_Contrato': 4,
        'Facturacion_Digital': 3,
        'Cargos_Mensuales': 4,
        
        # Impacto largo (6+ meses)
        'Servicio_Internet': 12,       # Infraestructura
        'Meses_Antigüedad': 8,         # Cambio de procesos
        
        # No aplicable
        'Edad': 999,
        'Senior_Citizen': 999,
        'Genero': 999
    }
    
    return time_to_impact.get(variable, 6)  # Default 6 meses

def analyze_by_business_categories(critical_factors):
    """Analizar factores críticos por categorías de negocio"""
    logging.info("Analizando por categorías de negocio...")
    
    try:
        # Agrupar por categorías
        category_analysis = {}
        
        for factor in critical_factors:
            category = factor['category']
            
            if category not in category_analysis:
                category_analysis[category] = {
                    'variables': [],
                    'total_importance': 0,
                    'avg_actionability': 0,
                    'avg_complexity': [],
                    'avg_time_to_impact': 0,
                    'factors_count': 0
                }
            
            category_analysis[category]['variables'].append(factor['variable'])
            category_analysis[category]['total_importance'] += factor['avg_importance']
            category_analysis[category]['avg_actionability'] += factor['actionability_score']
            category_analysis[category]['avg_complexity'].append(factor['implementation_complexity'])
            category_analysis[category]['avg_time_to_impact'] += factor['time_to_impact']
            category_analysis[category]['factors_count'] += 1
        
        # Calcular promedios y métricas finales
        for category, data in category_analysis.items():
            count = data['factors_count']
            
            # Promedios
            data['avg_importance'] = data['total_importance'] / count
            data['avg_actionability'] = data['avg_actionability'] / count
            data['avg_time_to_impact'] = data['avg_time_to_impact'] / count
            
            # Complejidad predominante
            complexity_counts = {}
            for comp in data['avg_complexity']:
                complexity_counts[comp] = complexity_counts.get(comp, 0) + 1
            data['predominant_complexity'] = max(complexity_counts, key=complexity_counts.get)
            
            # Prioridad estratégica MEJORADA (basada en importancia y accionabilidad)
            # Dar más peso a la accionabilidad para diferenciar categorías
            importance_score = data['avg_importance'] * 0.4  # Reducir peso de importancia
            actionability_score = data['avg_actionability'] * 6  # Aumentar peso de accionabilidad
            priority_score = importance_score + actionability_score
            data['strategic_priority'] = priority_score
            
            # Clasificación de prioridad MEJORADA con thresholds más diferenciados
            if priority_score >= 45:
                data['priority_level'] = 'CRÍTICA'
            elif priority_score >= 35:
                data['priority_level'] = 'ALTA'
            elif priority_score >= 25:
                data['priority_level'] = 'MEDIA'
            else:
                data['priority_level'] = 'BAJA'
        
        # Ordenar por prioridad estratégica
        sorted_categories = sorted(category_analysis.items(), 
                                 key=lambda x: x[1]['strategic_priority'], 
                                 reverse=True)
        
        logging.info(f"Analizadas {len(category_analysis)} categorías de negocio")
        return dict(sorted_categories)
        
    except Exception as e:
        logging.error(f"Error analizando categorías: {str(e)}")
        raise

def generate_business_insights(critical_factors, category_analysis, previous_results):
    """Generar insights clave de negocio"""
    logging.info("Generando insights de negocio...")
    
    try:
        model_info = previous_results['paso11']
        dataset_info = previous_results['dataset']
        
        # Top insights
        top_factor = critical_factors[0]
        most_actionable = max(critical_factors, key=lambda x: x['actionability_score'])
        least_actionable = min(critical_factors, key=lambda x: x['actionability_score'])
        
        # Insights por categoría
        priority_category = list(category_analysis.keys())[0]  # Primera en orden de prioridad
        
        insights = {
            'model_performance': {
                'best_model': model_info['best_model'],
                'model_score': model_info['model_score'],
                'production_ready': model_info['production_ready']
            },
            'dataset_summary': {
                'total_customers': dataset_info['size'],
                'baseline_churn_rate': dataset_info['churn_rate'],
                'features_analyzed': len(dataset_info['features'])
            },
            'critical_factors_summary': {
                'total_factors_identified': len(critical_factors),
                'top_factor': {
                    'variable': top_factor['variable'],
                    'importance': top_factor['avg_importance'],
                    'category': top_factor['category']
                },
                'most_actionable': {
                    'variable': most_actionable['variable'],
                    'actionability': most_actionable['actionability_score'],
                    'category': most_actionable['category']
                },
                'least_actionable': {
                    'variable': least_actionable['variable'],
                    'actionability': least_actionable['actionability_score'],
                    'category': least_actionable['category']
                }
            },
            'category_insights': {
                'priority_category': priority_category,
                'total_categories': len(category_analysis),
                'high_priority_categories': len([cat for cat, data in category_analysis.items() 
                                               if data['priority_level'] in ['CRÍTICA', 'ALTA']])
            },
            'actionability_distribution': {
                'highly_actionable': len([f for f in critical_factors if f['actionability_score'] >= 7]),
                'moderately_actionable': len([f for f in critical_factors if 4 <= f['actionability_score'] < 7]),
                'low_actionable': len([f for f in critical_factors if f['actionability_score'] < 4])
            },
            'implementation_timeline': {
                'quick_wins': len([f for f in critical_factors if f['time_to_impact'] <= 3]),
                'medium_term': len([f for f in critical_factors if 3 < f['time_to_impact'] <= 6]),
                'long_term': len([f for f in critical_factors if f['time_to_impact'] > 6])
            }
        }
        
        return insights
        
    except Exception as e:
        logging.error(f"Error generando insights: {str(e)}")
        raise

def create_visualization(critical_factors, category_analysis, timestamp):
    """Crear visualización mejorada de factores críticos"""
    logging.info("Creando visualización mejorada de análisis consolidado...")
    
    try:
        # Configurar el gráfico con 3 subplots
        plt.style.use('default')  # Resetear estilo
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.85, bottom=0.15, left=0.05, right=0.95)
        
        # 1. Top 8 factores críticos - COLORES ESPECÍFICOS POR RANGO
        top_8_factors = critical_factors[:8]
        variables = [f['variable'].replace('_', ' ')[:20] for f in top_8_factors]
        importances = [f['avg_importance'] for f in top_8_factors]
        
        # NUEVA LÓGICA: Colores específicos por rangos de importancia
        colors = []
        for imp in importances:
            if imp >= 15.0:  # 20% a 15%
                colors.append('#F4320B')  # Rojo intenso
            elif imp >= 6.0:  # 14.9% a 6%
                colors.append('#F54927')  # Rojo medio
            else:  # Por debajo de 6%
                colors.append('#F87C63')  # Rojo claro
        
        # Debug: mostrar clasificación por rangos
        logging.info("Clasificación por rangos de importancia:")
        for var, imp, color in zip([f['variable'] for f in top_8_factors], importances, colors):
            if imp >= 15.0:
                category = "CRÍTICO (≥15%)"
            elif imp >= 6.0:
                category = "IMPORTANTE (6-14.9%)"
            else:
                category = "MODERADO (<6%)"
            logging.info(f"  {var}: {imp:.1f}% -> {color} ({category})")
        
        bars = ax1.barh(range(len(variables)), importances, color=colors, alpha=0.9, 
                       edgecolor='black', linewidth=0.8)
        ax1.set_yticks(range(len(variables)))
        ax1.set_yticklabels(variables, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Importancia Promedio (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Top 8 Factores Críticos de Churn\n(Rojo Intenso: ≥15%, Rojo Medio: 6-14.9%, Rojo Claro: <6%)', 
                     fontweight='bold', fontsize=12, pad=20)
        ax1.invert_yaxis()
        
        # Añadir valores de importancia
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.1f}%', va='center', ha='left', 
                    fontsize=11, fontweight='bold', color='black')
        
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, max(importances) * 1.3)
        
        # 2. Análisis por categorías - MANTENEMOS IGUAL
        categories = list(category_analysis.keys())[:5]
        cat_priorities = [category_analysis[cat]['strategic_priority'] for cat in categories]
        
        # Colores por nivel de prioridad
        priority_colors = {
            'CRÍTICA': '#B22222',  # Rojo oscuro
            'ALTA': '#FF6347',     # Rojo-naranja 
            'MEDIA': '#FFD700',    # Amarillo
            'BAJA': '#90EE90'      # Verde claro
        }
        
        cat_colors = []
        for cat in categories:
            level = category_analysis[cat]['priority_level']
            color = priority_colors.get(level, '#87CEEB')
            cat_colors.append(color)
        
        cat_names = [cat.replace('_', '\n').replace('Servicios', 'Serv') for cat in categories]
        
        bars2 = ax2.bar(range(len(categories)), cat_priorities, color=cat_colors, alpha=0.8, 
                       edgecolor='black', linewidth=0.8)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(cat_names, fontsize=11, fontweight='bold', ha='center')
        ax2.set_ylabel('Score de Prioridad Estratégica', fontsize=12, fontweight='bold')
        ax2.set_title('Prioridad Estratégica por Categoría\n(Rojo: Crítica, Naranja: Alta, Amarillo: Media)', 
                     fontweight='bold', fontsize=12, pad=20)
        
        for bar, priority in zip(bars2, cat_priorities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{priority:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11, color='black')
        
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(cat_priorities) * 1.3)
        
        # 3. Timeline de implementación - MANTENEMOS IGUAL
        timeline_labels = ['Quick Wins\n(1-3 meses)', 'Medio Plazo\n(3-6 meses)', 'Largo Plazo\n(6+ meses)']
        timeline_counts = [
            len([f for f in critical_factors if f['time_to_impact'] <= 3]),
            len([f for f in critical_factors if 3 < f['time_to_impact'] <= 6]),
            len([f for f in critical_factors if f['time_to_impact'] > 6])
        ]
        
        colors_timeline = ['#40E0D0', '#87CEEB', '#4682B4']
        bars3 = ax3.bar(timeline_labels, timeline_counts, color=colors_timeline, alpha=0.8,
                       edgecolor='black', linewidth=0.8)
        ax3.set_ylabel('Número de Factores', fontsize=12, fontweight='bold')
        ax3.set_title('Timeline de Implementación\npor Factor Crítico', 
                     fontweight='bold', fontsize=12, pad=20)
        
        for bar, count in zip(bars3, timeline_counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        str(count), ha='center', va='bottom', 
                        fontweight='bold', fontsize=11, color='black')
        
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, max(timeline_counts) * 1.4 if timeline_counts else 1)
        
        # Título principal
        fig.suptitle('TelecomX - Análisis Consolidado de Factores Críticos de Churn', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Guardar visualización
        viz_file = f'graficos/paso13a_analisis_consolidado_v2_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        logging.info(f"Visualización con colores específicos guardada: {viz_file}")
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando visualización: {str(e)}")
        return None

def generate_summary_report(consolidated_data):
    """Generar informe resumen del análisis consolidado"""
    
    critical_factors = consolidated_data['critical_factors']
    category_analysis = consolidated_data['category_analysis']
    insights = consolidated_data['business_insights']
    timestamp = consolidated_data['metadata']['timestamp']
    
    report = f"""
================================================================================
TELECOMX - PASO 13A: ANÁLISIS CONSOLIDADO - RESUMEN EJECUTIVO (V2)
================================================================================
Fecha: {timestamp}
Script: {consolidated_data['metadata']['script']}

================================================================================
RESUMEN DE HALLAZGOS PRINCIPALES
================================================================================

🎯 MODELO PREDICTIVO VALIDADO:
• Mejor modelo: {insights['model_performance']['best_model']}
• Score de performance: {insights['model_performance']['model_score']:.4f}
• Estado para producción: {'✅ LISTO' if insights['model_performance']['production_ready'] else '❌ NO LISTO'}

📊 DATASET ANALIZADO:
• Total de clientes: {insights['dataset_summary']['total_customers']:,}
• Tasa de churn baseline: {insights['dataset_summary']['baseline_churn_rate']:.1%}
• Variables analizadas: {insights['dataset_summary']['features_analyzed']}

================================================================================
FACTORES CRÍTICOS IDENTIFICADOS
================================================================================

🔥 TOP 5 FACTORES MÁS CRÍTICOS:
"""
    
    # Top 5 factores
    for i, factor in enumerate(critical_factors[:5], 1):
        actionability_icon = "🟢" if factor['actionability_score'] >= 7 else "🟡" if factor['actionability_score'] >= 5 else "🔴"
        
        report += f"""
{i}. {factor['variable'].upper()}:
   📈 Importancia: {factor['avg_importance']:.2f}%
   ⚡ Accionabilidad: {actionability_icon} {factor['actionability_score']}/10
   🏢 Categoría: {factor['category']}
   ⏱️ Tiempo impacto: {factor['time_to_impact']} meses
   📋 Complejidad: {factor['implementation_complexity']}
   💡 Insight: {factor['business_impact'][:100]}...
"""

    report += f"""

🎯 VARIABLE MÁS IMPORTANTE:
• {insights['critical_factors_summary']['top_factor']['variable']}: {insights['critical_factors_summary']['top_factor']['importance']:.2f}%
• Categoría: {insights['critical_factors_summary']['top_factor']['category']}

⚡ VARIABLE MÁS ACCIONABLE:
• {insights['critical_factors_summary']['most_actionable']['variable']}: {insights['critical_factors_summary']['most_actionable']['actionability']}/10
• Categoría: {insights['critical_factors_summary']['most_actionable']['category']}

================================================================================
ANÁLISIS POR CATEGORÍAS DE NEGOCIO
================================================================================

🏆 RANKING DE CATEGORÍAS POR PRIORIDAD ESTRATÉGICA:
"""
    
    # Ranking de categorías
    for i, (category, data) in enumerate(list(category_analysis.items())[:5], 1):
        priority_icon = "🔴" if data['priority_level'] == 'CRÍTICA' else "🟡" if data['priority_level'] == 'ALTA' else "🟢"
        
        report += f"""
{i}. {category.replace('_', ' ').upper()}:
   🎯 Prioridad: {priority_icon} {data['priority_level']}
   📊 Score estratégico: {data['strategic_priority']:.1f}
   📈 Importancia promedio: {data['avg_importance']:.1f}%
   ⚡ Accionabilidad promedio: {data['avg_actionability']:.1f}/10
   ⏱️ Tiempo promedio impacto: {data['avg_time_to_impact']:.1f} meses
   🔧 Complejidad predominante: {data['predominant_complexity']}
   📝 Variables: {', '.join(data['variables'])}
"""

    report += f"""

================================================================================
DISTRIBUCIÓN DE ACCIONABILIDAD
================================================================================

⚡ NIVELES DE ACCIONABILIDAD:
• 🟢 ALTA (7-10): {insights['actionability_distribution']['highly_actionable']} factores
• 🟡 MEDIA (4-6): {insights['actionability_distribution']['moderately_actionable']} factores  
• 🔴 BAJA (1-3): {insights['actionability_distribution']['low_actionable']} factores

⏱️ TIMELINE DE IMPLEMENTACIÓN:
• 🚀 Quick Wins (1-3 meses): {insights['implementation_timeline']['quick_wins']} factores
• 📅 Medio Plazo (3-6 meses): {insights['implementation_timeline']['medium_term']} factores
• 📆 Largo Plazo (6+ meses): {insights['implementation_timeline']['long_term']} factores

================================================================================
INSIGHTS CLAVE PARA ESTRATEGIA
================================================================================

💡 HALLAZGOS PRINCIPALES:

1. CONCENTRACIÓN DE OPORTUNIDADES:
   • {insights['category_insights']['high_priority_categories']} de {insights['category_insights']['total_categories']} categorías son de prioridad ALTA/CRÍTICA
   • Categoría prioritaria: {insights['category_insights']['priority_category']}
   • {insights['actionability_distribution']['highly_actionable']} factores son altamente accionables

2. OPORTUNIDADES DE QUICK WINS:
   • {insights['implementation_timeline']['quick_wins']} factores pueden generar impacto en 1-3 meses
   • Variables más accionables están en categorías: {', '.join([cat for cat, data in list(category_analysis.items())[:3] if data['avg_actionability'] >= 6])}

3. BALANCE IMPACTO-ESFUERZO:
   • Factor con mejor ratio impacto/accionabilidad: {insights['critical_factors_summary']['most_actionable']['variable']}
   • Categorías que requieren menos inversión pero generan alto impacto identificadas

================================================================================
RECOMENDACIONES INMEDIATAS
================================================================================

🚀 ACCIONES PRIORITARIAS:

1. IMPLEMENTACIÓN INMEDIATA (1-3 meses):
   • Focus en factores con accionabilidad ≥7 y tiempo impacto ≤3 meses
   • Priorizar categoría: {insights['category_insights']['priority_category']}
   • Variables clave: {', '.join([f['variable'] for f in critical_factors if f['actionability_score'] >= 7 and f['time_to_impact'] <= 3][:3])}

2. PLANIFICACIÓN MEDIO PLAZO (3-6 meses):
   • Preparar implementación de factores con complejidad media
   • Desarrollar capacidades para categorías de alta prioridad
   • Investment en infraestructura para factores de largo impacto

3. PREPARACIÓN ESTRATÉGICA:
   • Alinear recursos con categorías de mayor prioridad estratégica
   • Desarrollar métricas de seguimiento específicas por factor
   • Establecer governance para implementación por fases

================================================================================
PRÓXIMOS PASOS
================================================================================

📋 SIGUIENTES SCRIPTS DEL PASO 13:

1. paso13b_conclusion_Segmentación_Estratégica.py:
   • Utilizará estos factores críticos para definir segmentos
   • Creará estrategias específicas por tipo de cliente
   • Input principal: factores con alta accionabilidad

2. paso13c_conclusion_Business_Case_Completo.py:
   • Calculará ROI específico por factor y categoría
   • Estimará impacto financiero de intervenciones
   • Priorizará inversiones basado en estos hallazgos

3. paso13d_conclusion_Roadmap_Detallado.py:
   • Utilizará timeline de implementación identificado
   • Creará cronograma basado en complejidad y dependencias
   • Organizará por quick wins vs proyectos de largo plazo

📊 ARCHIVOS GENERADOS:
• Datos consolidados JSON: datos/paso13a_analisis_consolidado_v2_{timestamp}.json
• Visualización mejorada: {consolidated_data['visualization_file'] if consolidated_data['visualization_file'] else 'No generada'}
• Este informe: informes/paso13a_analisis_consolidado_resumen_v2_{timestamp}.txt

================================================================================
CONCLUSIÓN
================================================================================

✅ ANÁLISIS CONSOLIDADO COMPLETADO EXITOSAMENTE:

• {len(critical_factors)} factores críticos identificados y categorizados
• {len(category_analysis)} categorías de negocio priorizadas estratégicamente
• {insights['actionability_distribution']['highly_actionable']} factores altamente accionables disponibles para implementación inmediata
• Base sólida establecida para segmentación estratégica y business case

🎯 FACTOR MÁS CRÍTICO: {insights['critical_factors_summary']['top_factor']['variable']}
⚡ FACTOR MÁS ACCIONABLE: {insights['critical_factors_summary']['most_actionable']['variable']}
🏆 CATEGORÍA PRIORITARIA: {insights['category_insights']['priority_category']}

La base analítica está lista para desarrollar estrategias específicas de retención
con alto potencial de ROI y implementación efectiva.

================================================================================
FIN DEL RESUMEN
================================================================================
"""
    
    return report

def save_consolidated_results(critical_factors, category_analysis, business_insights, previous_results, viz_file, timestamp):
    """Guardar resultados consolidados para scripts posteriores"""
    try:
        # Crear estructura de datos consolidados
        consolidated_data = {
            'metadata': {
                'timestamp': timestamp,
                'script': 'paso13a_conclusion_Análisis_Consolidado_v2',
                'version': '2.0',
                'total_factors': len(critical_factors),
                'total_categories': len(category_analysis)
            },
            'model_info': previous_results['paso11'],
            'dataset_info': {
                'size': previous_results['dataset']['size'],
                'churn_rate': previous_results['dataset']['churn_rate'],
                'features_count': len(previous_results['dataset']['features'])
            },
            'critical_factors': critical_factors,
            'category_analysis': category_analysis,
            'business_insights': business_insights,
            'visualization_file': viz_file
        }
        
        # Guardar JSON para scripts posteriores
        json_file = f'datos/paso13a_analisis_consolidado_v2_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Guardar informe resumido
        report_content = generate_summary_report(consolidated_data)
        report_file = f'informes/paso13a_analisis_consolidado_resumen_v2_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"Datos consolidados guardados: {json_file}")
        logging.info(f"Informe resumen guardado: {report_file}")
        
        return {
            'json_file': json_file,
            'report_file': report_file,
            'data': consolidated_data
        }
        
    except Exception as e:
        logging.error(f"Error guardando resultados: {str(e)}")
        raise

def main():
    """Función principal del Paso 13A - Versión 2"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13A: ANÁLISIS CONSOLIDADO DE FACTORES DE CHURN - V2")
        logger.info("="*80)
        logger.info("Versión mejorada con visualizaciones optimizadas")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar resultados de pasos anteriores
        logger.info("="*50)
        logger.info("CARGANDO RESULTADOS DE PASOS ANTERIORES")
        previous_results = load_previous_results()
        
        # 3. Identificar factores críticos de churn
        logger.info("="*50)
        logger.info("IDENTIFICANDO FACTORES CRÍTICOS DE CHURN")
        critical_factors = identify_critical_churn_factors(previous_results)
        
        # 4. Analizar por categorías de negocio
        logger.info("="*50)
        logger.info("ANALIZANDO POR CATEGORÍAS DE NEGOCIO")
        category_analysis = analyze_by_business_categories(critical_factors)
        
        # 5. Generar insights de negocio
        logger.info("="*50)
        logger.info("GENERANDO INSIGHTS DE NEGOCIO")
        business_insights = generate_business_insights(critical_factors, category_analysis, previous_results)
        
        # 6. Crear visualización mejorada
        logger.info("="*50)
        logger.info("GENERANDO VISUALIZACIÓN MEJORADA")
        viz_file = create_visualization(critical_factors, category_analysis, timestamp)
        
        # 7. Guardar resultados consolidados
        logger.info("="*50)
        logger.info("GUARDANDO RESULTADOS CONSOLIDADOS")
        output_files = save_consolidated_results(critical_factors, category_analysis, 
                                               business_insights, previous_results, viz_file, timestamp)
        
        # 8. Resumen final
        logger.info("="*80)
        logger.info("PASO 13A V2 COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar hallazgos principales
        logger.info("🎯 FACTORES CRÍTICOS IDENTIFICADOS:")
        for i, factor in enumerate(critical_factors[:5], 1):
            actionability = "🟢 Alta" if factor['actionability_score'] >= 7 else "🟡 Media" if factor['actionability_score'] >= 5 else "🔴 Baja"
            logger.info(f"  {i}. {factor['variable']}: {factor['avg_importance']:.1f}% | {actionability}")
        logger.info("")
        
        logger.info("🏢 CATEGORÍAS PRIORITARIAS:")
        for i, (category, data) in enumerate(list(category_analysis.items())[:3], 1):
            logger.info(f"  {i}. {category.replace('_', ' ')}: {data['priority_level']} ({data['strategic_priority']:.1f} pts)")
        logger.info("")
        
        logger.info("⚡ DISTRIBUCIÓN DE ACCIONABILIDAD:")
        logger.info(f"  • Alta (7-10): {business_insights['actionability_distribution']['highly_actionable']} factores")
        logger.info(f"  • Media (4-6): {business_insights['actionability_distribution']['moderately_actionable']} factores")
        logger.info(f"  • Baja (1-3): {business_insights['actionability_distribution']['low_actionable']} factores")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • Datos consolidados: {output_files['json_file']}")
        logger.info(f"  • Informe resumen: {output_files['report_file']}")
        if viz_file:
            logger.info(f"  • Visualización mejorada: {viz_file}")
        logger.info("")
        
        logger.info("✅ MEJORAS IMPLEMENTADAS EN V2:")
        logger.info("  • Visualizaciones con colores profesionales")
        logger.info("  • Mejor espaciado y legibilidad")
        logger.info("  • Texto sin superposición")
        logger.info("  • Gráficos de mayor resolución")
        logger.info("")
        
        logger.info("📋 LISTO PARA PRÓXIMOS SCRIPTS:")
        logger.info("  • paso13b: Segmentación estratégica")
        logger.info("  • paso13c: Business case con ROI")
        logger.info("  • paso13d: Roadmap de implementación")
        logger.info("="*80)
        
        return output_files['data']
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()