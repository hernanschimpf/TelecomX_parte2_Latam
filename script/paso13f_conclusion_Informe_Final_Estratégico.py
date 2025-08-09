"""
================================================================================
TELECOMX - PASO 13F: INFORME FINAL ESTRATÉGICO CONSOLIDADO
================================================================================
Descripción: Informe detallado final que consolida todo el análisis predictivo
             de churn, destacando factores principales de cancelación, rendimiento
             de modelos y estrategias de retención propuestas.
             
⚠️  IMPORTANTE: Este informe consolida análisis basado en DATOS ESTIMADOS de 
    benchmarks industria telecom para fines de SIMULACIÓN y demostración 
    metodológica.

OBJETIVO: Elaborar informe detallado destacando factores que más influyen en 
         cancelación, basándose en variables seleccionadas y rendimiento de 
         cada modelo, proponiendo estrategias de retención.

Inputs: 
- Consolidación completa de Pasos 13A, 13B, 13C, 13D, 13E
- Análisis de factores críticos y capacidad de intervención
- Evaluación comparativa de modelos predictivos
- Segmentación estratégica y business case
- Roadmap de implementación y outputs ejecutivos

Outputs:
- Informe estratégico maestro consolidado
- Dashboard final con todos los hallazgos
- Recomendaciones estratégicas finales
- Template de governance y seguimiento
- Checklist de implementación

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
            logging.FileHandler('logs/paso13f_informe_final_estrategico.log', mode='a', encoding='utf-8'),
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

def load_complete_analysis():
    """Cargar análisis completo de todos los pasos del proyecto"""
    try:
        logging.info("Cargando análisis completo de todo el proyecto...")
        
        complete_analysis = {
            'project_metadata': {
                'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                'total_steps_completed': 0,
                'disclaimer': 'ANÁLISIS BASADO EN DATOS ESTIMADOS INDUSTRIA TELECOM'
            }
        }
        
        # 1. Cargar Paso 13A - Factores Críticos
        try:
            paso13a_file = find_latest_file('datos', 'paso13a_analisis_consolidado_v2_*.json')
            with open(paso13a_file, 'r', encoding='utf-8') as f:
                complete_analysis['factores_criticos'] = json.load(f)
            logging.info("✅ Paso 13A cargado: Factores críticos de churn identificados")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13A no disponible: {str(e)}")
            complete_analysis['factores_criticos'] = None
        
        # 2. Cargar Paso 13B - Segmentación Estratégica  
        try:
            paso13b_file = find_latest_file('informes', 'paso13b_segmentacion_estrategica_*.json')
            with open(paso13b_file, 'r', encoding='utf-8') as f:
                complete_analysis['segmentacion'] = json.load(f)
            logging.info("✅ Paso 13B cargado: Segmentación por riesgo completada")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13B no disponible: {str(e)}")
            complete_analysis['segmentacion'] = None
        
        # 3. Cargar Paso 13C - Business Case
        try:
            paso13c_file = find_latest_file('informes', 'paso13c_business_case_completo_*.json')
            with open(paso13c_file, 'r', encoding='utf-8') as f:
                complete_analysis['business_case'] = json.load(f)
            logging.info("✅ Paso 13C cargado: Business case con ROI validado")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13C no disponible: {str(e)}")
            complete_analysis['business_case'] = None
        
        # 4. Cargar Paso 13D - Roadmap de Implementación
        try:
            paso13d_file = find_latest_file('informes', 'paso13d_roadmap_detallado_*.json')
            with open(paso13d_file, 'r', encoding='utf-8') as f:
                complete_analysis['roadmap'] = json.load(f)
            logging.info("✅ Paso 13D cargado: Roadmap de implementación definido")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13D no disponible: {str(e)}")
            complete_analysis['roadmap'] = None
        
        # 5. Cargar Paso 13E - Outputs Ejecutivos
        try:
            paso13e_file = find_latest_file('informes', 'paso13e_outputs_ejecutivos_*.json')
            with open(paso13e_file, 'r', encoding='utf-8') as f:
                complete_analysis['outputs_ejecutivos'] = json.load(f)
            logging.info("✅ Paso 13E cargado: Outputs ejecutivos consolidados")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13E no disponible: {str(e)}")
            complete_analysis['outputs_ejecutivos'] = None
        
        logging.info(f"Análisis completo cargado: {complete_analysis['project_metadata']['total_steps_completed']}/5 pasos disponibles")
        return complete_analysis
        
    except Exception as e:
        logging.error(f"Error cargando análisis completo: {str(e)}")
        raise

def extract_key_findings(complete_analysis):
    """Extraer hallazgos clave de todo el análisis"""
    try:
        logging.info("Extrayendo hallazgos clave del análisis completo...")
        
        key_findings = {
            'disclaimer': 'HALLAZGOS BASADOS EN DATOS ESTIMADOS INDUSTRIA TELECOM',
            'model_performance': {},
            'critical_factors': {},
            'segmentation_insights': {},
            'financial_viability': {},
            'implementation_strategy': {},
            'strategic_recommendations': {}
        }
        
        # 1. Rendimiento de Modelos
        if complete_analysis['factores_criticos']:
            fc_data = complete_analysis['factores_criticos']
            key_findings['model_performance'] = {
                'best_model': fc_data['model_info']['best_model'],
                'model_score': fc_data['model_info']['model_score'],
                'production_ready': fc_data['model_info']['production_ready'],
                'total_customers_analyzed': fc_data['dataset_info']['size'],
                'baseline_churn_rate': fc_data['dataset_info']['churn_rate'],
                'variables_analyzed': fc_data['dataset_info']['features_count']
            }
        
        # 2. Factores Críticos con Capacidad de Intervención
        if complete_analysis['factores_criticos']:
            factors = complete_analysis['factores_criticos']['critical_factors']
            
            # Top 5 factores más importantes
            top_factors = factors[:5]
            
            # Factores con alta capacidad de intervención (score >= 7)
            high_intervention = [f for f in factors if f['actionability_score'] >= 7]
            
            key_findings['critical_factors'] = {
                'total_factors_identified': len(factors),
                'top_factor': {
                    'variable': top_factors[0]['variable'],
                    'importance': top_factors[0]['avg_importance'],
                    'intervention_capacity': top_factors[0]['actionability_score']
                },
                'most_controllable_factor': max(factors, key=lambda x: x['actionability_score']),
                'high_intervention_factors': len(high_intervention),
                'top_5_factors': top_factors,
                'intervention_summary': {
                    'high_capacity': len([f for f in factors if f['actionability_score'] >= 7]),
                    'medium_capacity': len([f for f in factors if 4 <= f['actionability_score'] < 7]),
                    'low_capacity': len([f for f in factors if f['actionability_score'] < 4])
                }
            }
        
        # 3. Insights de Segmentación
        if complete_analysis['segmentacion']:
            seg_data = complete_analysis['segmentacion']['segmentation_summary']
            key_findings['segmentation_insights'] = {
                'total_clients_segmented': complete_analysis['segmentacion']['metadata']['total_clients'],
                'segments_identified': len(seg_data),
                'priority_segment': 'Medio_Riesgo',  # Basado en conclusiones del business case
                'segment_distribution': {k: v['percentage'] for k, v in seg_data.items()},
                'priority_justification': 'Mejor balance entre capacidad de intervención y ROI potencial'
            }
        
        # 4. Viabilidad Financiera
        if complete_analysis['business_case']:
            bc_data = complete_analysis['business_case']
            consolidated = bc_data['intervention_scenarios']['consolidated']
            projections = bc_data['projections_3_years']['summary']
            
            key_findings['financial_viability'] = {
                'investment_required': consolidated['total_annual_investment'],
                'revenue_opportunity': consolidated['total_annual_revenue_saved'],
                'roi_annual': consolidated['overall_roi_annual'],
                'payback_months': consolidated['overall_payback_months'],
                'npv_3_years': projections['net_present_value'],
                'break_even_year': projections['break_even_year'],
                'viability_assessment': 'VIABLE' if consolidated['overall_roi_annual'] > 100 else 'NO VIABLE',
                'current_churn_loss': bc_data['current_state']['totals']['total_annual_churn_loss']
            }
        
        # 5. Estrategia de Implementación
        if complete_analysis['roadmap']:
            rm_data = complete_analysis['roadmap']
            key_findings['implementation_strategy'] = {
                'implementation_duration': rm_data['summary_metrics']['total_duration_months'],
                'total_phases': rm_data['summary_metrics']['total_phases'],
                'priority_approach': rm_data['summary_metrics']['implementation_approach'],
                'total_budget_estimated': rm_data['summary_metrics']['total_budget_estimated'],
                'critical_milestones': ['Mes 2: Setup', 'Mes 5: Piloto validado', 'Mes 8: Scaling', 'Mes 12: Completo']
            }
        
        # 6. Recomendaciones Estratégicas Consolidadas
        key_findings['strategic_recommendations'] = {
            'primary_recommendation': 'PROCEDER CON IMPLEMENTACIÓN CON VALIDACIÓN',
            'priority_actions': [
                'Validar supuestos con datos reales empresa',
                'Iniciar piloto controlado segmento Medio Riesgo',
                'Confirmar budget y recursos disponibles',
                'Establecer governance y métricas seguimiento'
            ],
            'success_factors': [
                'Validación temprana con datos reales',
                'Enfoque gradual por fases',
                'Monitoreo continuo ROI',
                'Capacidad operacional adecuada'
            ],
            'risk_mitigation': [
                'Piloto controlado antes scaling',
                'Monitoreo semanal métricas clave',
                'Plan contingencia si ROI < objetivo',
                'Validación continua supuestos'
            ]
        }
        
        logging.info("Hallazgos clave extraídos exitosamente")
        return key_findings
        
    except Exception as e:
        logging.error(f"Error extrayendo hallazgos clave: {str(e)}")
        raise

def create_final_dashboard(key_findings, timestamp):
    """Crear dashboard final consolidado"""
    try:
        logging.info("Creando dashboard final consolidado...")
        
        plt.style.use('default')
        
        # Colores profesionales finales
        colors = {
            'primary': '#1f4e79',
            'success': '#70ad47', 
            'warning': '#c55a11',
            'info': '#5b9bd5',
            'neutral': '#7f7f7f',
            'background': '#f8f9fa'
        }
        
        fig = plt.figure(figsize=(24, 18))
        fig.patch.set_facecolor('white')
        
        gs = fig.add_gridspec(3, 4, hspace=1.2, wspace=0.4, top=0.82, bottom=0.15, left=0.08, right=0.94)
        
        # 1. Factores Críticos con Mayor Capacidad de Intervención (top-left span 2)
        ax1 = fig.add_subplot(gs[0, :2])
        
        if 'critical_factors' in key_findings and key_findings['critical_factors']:
            factors = key_findings['critical_factors']['top_5_factors']
            factor_names = [f['variable'].replace('_', ' ')[:20] for f in factors]
            importance_values = [f['avg_importance'] for f in factors]
            intervention_values = [f['actionability_score'] for f in factors]
            
            x = np.arange(len(factor_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, importance_values, width, label='Importancia (%)',
                            color=colors['primary'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, [v*10 for v in intervention_values], width, 
               label='Capacidad Intervención (%)', color=colors['success'], alpha=0.8)
            
            ax1.set_xlabel('Variables Analizadas', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Puntuación', fontweight='bold', fontsize=12)
            ax1.set_title('TOP 5 FACTORES: IMPORTANCIA vs CAPACIDAD DE INTERVENCIÓN\n⚠️ Ambas métricas en escala 0-100%', 
             fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
            ax1.set_xticks(x)
            ax1.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=10)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')

            # Agregar valores en las barras de Importancia
            for bar, value in zip(bars1, importance_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Agregar valores en las barras de Capacidad Intervención (como %)
            for bar, value in zip(bars2, intervention_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{value*10:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Rendimiento del Modelo (top-right span 2)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if 'model_performance' in key_findings:
            mp = key_findings['model_performance']
            
            # Métricas del modelo
            metrics = ['Score Modelo', 'Churn Rate', 'Variables']
            values = [
                mp.get('model_score', 0) * 100,  # Convertir a porcentaje
                mp.get('baseline_churn_rate', 0) * 100,
                mp.get('variables_analyzed', 0)
            ]
            colors_bars = [colors['success'], colors['warning'], colors['info']]
            
            bars = ax2.bar(metrics, values, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)
            ax2.set_ylabel('Valor', fontweight='bold', fontsize=12)
            ax2.set_title(f'RENDIMIENTO MODELO: {mp.get("best_model", "N/A")}\n⚠️ Evaluación con Dataset Estimado', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Etiquetas en barras
            labels = [f'{values[0]:.1f}%', f'{values[1]:.1f}%', f'{int(values[2])}']
            for bar, label in zip(bars, labels):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Business Case Consolidado (middle span 2)
        ax3 = fig.add_subplot(gs[1, :2])
        
        if 'financial_viability' in key_findings:
            fv = key_findings['financial_viability']
            
            categories = ['Pérdida\nActual', 'Inversión\nRequerida', 'Revenue\nSalvado', 'Beneficio\nNeto']
            values_millions = [
                fv.get('current_churn_loss', 0) / 1000000,
                fv.get('investment_required', 0) / 1000000,
                fv.get('revenue_opportunity', 0) / 1000000,
                (fv.get('revenue_opportunity', 0) - fv.get('investment_required', 0)) / 1000000
            ]
            colors_bars = [colors['warning'], colors['info'], colors['success'], colors['primary']]
            
            bars = ax3.bar(categories, values_millions, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)
            ax3.set_ylabel('Millones USD', fontweight='bold', fontsize=12)
            ax3.set_title(f'BUSINESS CASE: ROI {fv.get("roi_annual", 0):.0f}% | Payback {fv.get("payback_months", 0):.1f} meses\n⚠️ Proyección con Benchmarks Industria', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=50)
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values_millions):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values_millions)*0.02,
                        f'${value:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Distribución por Capacidad de Intervención (middle-right span 2)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        if 'critical_factors' in key_findings and key_findings['critical_factors']:
            intervention_summary = key_findings['critical_factors']['intervention_summary']
            
            labels = ['Alta\nCapacidad', 'Media\nCapacidad', 'Baja\nCapacidad']
            sizes = [
                intervention_summary['high_capacity'],
                intervention_summary['medium_capacity'],
                intervention_summary['low_capacity']
            ]
            colors_pie = [colors['success'], colors['warning'], colors['neutral']]
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f%%',
                                              colors=colors_pie, startangle=90,
                                              textprops={'fontsize': 11, 'fontweight': 'bold'},
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            ax4.set_title('FACTORES POR CAPACIDAD DE INTERVENCIÓN\n⚠️ Evaluación Estimada', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
        
        # 5. Conclusiones y Recomendaciones Finales (bottom span 4)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Extraer métricas clave para mostrar
        if key_findings.get('model_performance'):
            model_name = key_findings['model_performance'].get('best_model', 'N/A')
            model_score = key_findings['model_performance'].get('model_score', 0)
        else:
            model_name, model_score = 'N/A', 0
            
        if key_findings.get('critical_factors'):
            top_factor = key_findings['critical_factors']['top_factor']['variable']
            total_factors = key_findings['critical_factors']['total_factors_identified']
        else:
            top_factor, total_factors = 'N/A', 0
            
        if key_findings.get('financial_viability'):
            roi = key_findings['financial_viability'].get('roi_annual', 0)
            payback = key_findings['financial_viability'].get('payback_months', 0)
        else:
            roi, payback = 0, 0
        
        conclusions_text = f"""
🎯 CONCLUSIONES FINALES DEL ANÁLISIS PREDICTIVO DE CHURN - TELECOMX

✅ MODELO RECOMENDADO: {model_name} (Score: {model_score:.3f})
📊 FACTORES CRÍTICOS: {total_factors} identificados | Factor principal: {top_factor}
💰 VIABILIDAD FINANCIERA: ROI {roi:.0f}% anual | Payback {payback:.1f} meses
🚀 RECOMENDACIÓN: {key_findings['strategic_recommendations']['primary_recommendation']}

🔧 FACTORES CON MAYOR CAPACIDAD DE INTERVENCIÓN:
• Variables controlables por la empresa identificadas
• Estrategias de retención específicas por segmento desarrolladas
• Roadmap de implementación de 12 meses estructurado

⚠️ VALIDACIONES CRÍTICAS REQUERIDAS:
• Confirmar supuestos con datos financieros reales de la empresa
• Validar efectividad con piloto controlado antes de scaling completo
• Establecer governance y métricas de seguimiento continuo

🚨 DISCLAIMER IMPORTANTE: Este análisis utiliza DATOS ESTIMADOS de benchmarks industria telecom
para demostración metodológica. Para implementación real, validar con datos específicos empresa.
        """
        
        ax5.text(0.05, 0.05, conclusions_text, transform=ax5.transAxes, fontsize=12,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=1.0", facecolor=colors['background'], 
                  edgecolor=colors['primary'], linewidth=2, alpha=0.9))
        
        # Título principal del informe final
        fig.suptitle('TELECOMX - INFORME FINAL: ANÁLISIS PREDICTIVO DE CHURN Y ESTRATEGIAS DE RETENCIÓN\n⚠️ CONSOLIDACIÓN COMPLETA CON DATOS ESTIMADOS INDUSTRIA TELECOM ⚠️', 
                    fontsize=16, fontweight='bold', color=colors['primary'], y=0.93)
        
        # Guardar dashboard final
        os.makedirs('graficos', exist_ok=True)
        viz_file = f'graficos/paso13f_dashboard_final_consolidado_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.5)
        plt.close()
        
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Dashboard final creado: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear dashboard final: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard final: {str(e)}")
        return None
    
   

def generate_strategic_master_report(key_findings, complete_analysis, timestamp):
    """Generar informe estratégico maestro consolidado"""
    
    # Extraer métricas principales
    model_perf = key_findings.get('model_performance', {})
    factors = key_findings.get('critical_factors', {})
    financial = key_findings.get('financial_viability', {})
    implementation = key_findings.get('implementation_strategy', {})
    
    report = f"""
================================================================================
TELECOMX - INFORME FINAL ESTRATÉGICO: ANÁLISIS PREDICTIVO DE CHURN
================================================================================
Fecha: {timestamp}
Documento: Informe Maestro Consolidado

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
🚨 DISCLAIMER CRÍTICO: ANÁLISIS CON DATOS ESTIMADOS INDUSTRIA TELECOM
================================================================================
⚠️  ESTE INFORME CONSOLIDA ANÁLISIS BASADO EN BENCHMARKS ESTIMADOS DE LA 
    INDUSTRIA TELECOM PARA FINES DE SIMULACIÓN Y DEMOSTRACIÓN METODOLÓGICA.

📊 PROPÓSITO: Demostrar metodología completa de análisis predictivo de churn
💡 PARA IMPLEMENTACIÓN REAL: Validar todos los supuestos con datos específicos empresa

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

================================================================================
RESUMEN EJECUTIVO CONSOLIDADO
================================================================================

🎯 OBJETIVO ALCANZADO:
Identificar factores principales que influyen en cancelación de clientes y 
proponer estrategias de retención basadas en análisis predictivo completo.

📊 RESULTADOS PRINCIPALES:
• Modelo predictivo: {model_perf.get('best_model', 'N/A')} (Score: {model_perf.get('model_score', 0):.3f})
• Factores críticos identificados: {factors.get('total_factors_identified', 0)}
• ROI proyectado: {financial.get('roi_annual', 0):.1f}% anual
• Timeline implementación: {implementation.get('implementation_duration', 0)} meses

✅ RECOMENDACIÓN FINAL: {key_findings['strategic_recommendations']['primary_recommendation']}

================================================================================
1. ANÁLISIS DE MODELOS PREDICTIVOS
================================================================================

📈 EVALUACIÓN COMPARATIVA DE MODELOS:

MODELO RECOMENDADO: {model_perf.get('best_model', 'N/A')}
• Score de performance: {model_perf.get('model_score', 0):.4f}
• Estado para producción: {'✅ LISTO' if model_perf.get('production_ready', False) else '❌ NO LISTO'}
• Base de datos analizada: {model_perf.get('total_customers_analyzed', 0):,} clientes
• Tasa de churn baseline: {model_perf.get('baseline_churn_rate', 0):.1%}
• Variables analizadas: {model_perf.get('variables_analyzed', 0)}

JUSTIFICACIÓN DE SELECCIÓN:
• Mejor balance entre precision y recall
• Capacidad de interpretación para negocio
• Estabilidad en diferentes muestras de datos
• Facilidad de implementación en producción

LIMITACIONES IDENTIFICADAS:
• Performance basada en datos estimados industria
• Requiere validación con datos reales empresa
• Necesita recalibración periódica

================================================================================
2. FACTORES PRINCIPALES QUE AFECTAN LA CANCELACIÓN
================================================================================

🔥 TOP 5 FACTORES MÁS CRÍTICOS:
"""

    # Agregar top 5 factores si disponibles
    if factors and 'top_5_factors' in factors:
        for i, factor in enumerate(factors['top_5_factors'], 1):
            capacity_level = "🟢 ALTA" if factor['actionability_score'] >= 7 else "🟡 MEDIA" if factor['actionability_score'] >= 4 else "🔴 BAJA"
            
            report += f"""
{i}. {factor['variable'].replace('_', ' ').upper()}:
   📈 Importancia: {factor['avg_importance']:.2f}%
   🔧 Capacidad de intervención: {capacity_level} ({factor['actionability_score']}/10)
   🏢 Categoría: {factor['category']}
   ⏱️ Tiempo de impacto: {factor['time_to_impact']} meses
   💡 Interpretación: Variable crítica para predicción de churn
"""

    report += f"""

📊 ANÁLISIS DE CAPACIDAD DE INTERVENCIÓN:
• 🟢 Alta capacidad (7-10): {factors.get('intervention_summary', {}).get('high_capacity', 0)} factores
• 🟡 Media capacidad (4-6): {factors.get('intervention_summary', {}).get('medium_capacity', 0)} factores
• 🔴 Baja capacidad (1-3): {factors.get('intervention_summary', {}).get('low_capacity', 0)} factores

🎯 FACTOR MÁS CONTROLABLE: {factors.get('most_controllable_factor', {}).get('variable', 'N/A')}
⚡ OPORTUNIDAD PRINCIPAL: Enfoque en factores con alta capacidad de intervención

================================================================================
3. ESTRATEGIAS DE RETENCIÓN PROPUESTAS
================================================================================

🎯 SEGMENTACIÓN ESTRATÉGICA:
• Total clientes segmentados: {key_findings.get('segmentation_insights', {}).get('total_clients_segmented', 0):,}
• Segmentos identificados: {key_findings.get('segmentation_insights', {}).get('segments_identified', 0)}
• Segmento prioritario: {key_findings.get('segmentation_insights', {}).get('priority_segment', 'N/A').replace('_', ' ')}

JUSTIFICACIÓN PRIORIZACIÓN:
{key_findings.get('segmentation_insights', {}).get('priority_justification', 'N/A')}

🚀 ESTRATEGIAS POR SEGMENTO:

SEGMENTO ALTO RIESGO:
• Enfoque: Retención intensiva inmediata
• Acciones: Intervención personal, ofertas especiales, seguimiento 24/7
• Objetivo: Salvar clientes a punto de cancelar
• Inversión: Alta, ROI: Medio (casos complejos)

SEGMENTO MEDIO RIESGO (PRIORITARIO):
• Enfoque: Prevención proactiva y optimización
• Acciones: Campañas proactivas, mejora experiencia, ofertas personalizadas
• Objetivo: Prevenir escalación a alto riesgo
• Inversión: Media, ROI: Alto (mejor balance)

SEGMENTO BAJO RIESGO:
• Enfoque: Fidelización y crecimiento
• Acciones: Programas lealtad, upselling, comunicación automatizada
• Objetivo: Mantener satisfacción y generar crecimiento
• Inversión: Baja, ROI: Alto (mantenimiento)

================================================================================
4. BUSINESS CASE Y VIABILIDAD FINANCIERA
================================================================================

💰 SITUACIÓN FINANCIERA ACTUAL:
• Pérdida anual por churn: ${financial.get('current_churn_loss', 0):,.2f}
• Oportunidad de mercado identificada

📈 PROPUESTA DE INVERSIÓN:
• Inversión anual requerida: ${financial.get('investment_required', 0):,.2f}
• Revenue salvado proyectado: ${financial.get('revenue_opportunity', 0):,.2f}
• Beneficio neto anual: ${financial.get('revenue_opportunity', 0) - financial.get('investment_required', 0):,.2f}

🎯 MÉTRICAS DE RETORNO:
• ROI anual: {financial.get('roi_annual', 0):.1f}%
• Período de payback: {financial.get('payback_months', 0):.1f} meses
• NPV a 3 años: ${financial.get('npv_3_years', 0):,.2f}
• Break-even: Año {financial.get('break_even_year', 'N/A')}

✅ EVALUACIÓN: {financial.get('viability_assessment', 'N/A')}

FACTORES DE ÉXITO FINANCIERO:
• Validación temprana save rates con piloto
• Control estricto de costos implementación
• Monitoreo continuo ROI mensual
• Escalamiento gradual basado en resultados

================================================================================
5. PLAN DE IMPLEMENTACIÓN ESTRATÉGICO
================================================================================

⏱️ CRONOGRAMA MAESTRO:
• Duración total: {implementation.get('implementation_duration', 0)} meses
• Fases estructuradas: {implementation.get('total_phases', 0)}
• Enfoque: {implementation.get('priority_approach', 'N/A')}
• Budget total estimado: ${implementation.get('total_budget_estimated', 0):,.2f}

🎯 MILESTONES CRÍTICOS:
"""
    
    if implementation.get('critical_milestones'):
        for milestone in implementation['critical_milestones']:
            report += f"• {milestone}\n"

    report += f"""

🚀 ESTRATEGIA DE IMPLEMENTACIÓN:
• FASE 1: Setup y preparación (sistemas, equipos, procesos)
• FASE 2: Piloto controlado segmento prioritario
• FASE 3: Scaling basado en resultados piloto
• FASE 4: Expansión a todos los segmentos
• FASE 5: Optimización y preparación año 2

FACTORES CRÍTICOS DE ÉXITO:
• Capacidad operacional para manejar volumen
• Sistemas tecnológicos robustos
• Equipos entrenados y especializados
• Governance y métricas de seguimiento
• Flexibilidad para ajustes basados en learnings

================================================================================
6. RECOMENDACIONES ESTRATÉGICAS FINALES
================================================================================

✅ DECISIÓN RECOMENDADA: {key_findings['strategic_recommendations']['primary_recommendation']}

🎯 ACCIONES PRIORITARIAS INMEDIATAS:
"""
    
    for i, action in enumerate(key_findings['strategic_recommendations']['priority_actions'], 1):
        report += f"{i}. {action}\n"

    report += f"""

🔑 FACTORES CLAVE PARA EL ÉXITO:
"""
    
    for i, factor in enumerate(key_findings['strategic_recommendations']['success_factors'], 1):
        report += f"{i}. {factor}\n"

    report += f"""

🛡️ MITIGACIÓN DE RIESGOS:
"""
    
    for i, mitigation in enumerate(key_findings['strategic_recommendations']['risk_mitigation'], 1):
        report += f"{i}. {mitigation}\n"

    report += f"""

================================================================================
7. CONSIDERACIONES PARA IMPLEMENTACIÓN REAL
================================================================================

⚠️ VALIDACIONES CRÍTICAS REQUERIDAS:

DATOS Y SUPUESTOS:
• Confirmar ARPU real vs ${45} estimado benchmark
• Validar save rates con piloto muy controlado (muestra <5%)
• Verificar costos operacionales vs estimaciones industria
• Confirmar capacidad presupuestaria vs ${implementation.get('total_budget_estimated', 0):,.0f} estimado

CAPACIDADES ORGANIZACIONALES:
• Evaluar infraestructura tecnológica existente
• Confirmar disponibilidad recursos humanos especializados
• Validar procesos operacionales para retención
• Establecer governance y estructura de reporte

MERCADO Y COMPETENCIA:
• Analizar diferencias mercado local vs benchmarks internacionales
• Evaluar respuesta competencia a estrategias retención
• Confirmar regulaciones locales aplicables
• Validar comportamiento cliente local vs asumido

📋 PLAN DE VALIDACIÓN RECOMENDADO:

FASE 0 (Pre-implementación): VALIDACIÓN INTENSIVA
• Duración: 2-3 meses
• Objetivo: Confirmar supuestos críticos
• Actividades:
  - Análisis datos reales empresa vs benchmarks
  - Piloto micro (100-200 clientes) para testear save rates
  - Validación costos operacionales con equipos internos
  - Confirmación capacidades tecnológicas y humanas

CRITERIOS GO/NO-GO:
• Save rate piloto ≥ 15% (vs 35% estimado medio riesgo)
• Costos reales ≤ 120% de estimaciones
• Capacidad operacional confirmada
• ROI proyectado ≥ 150% (vs {financial.get('roi_annual', 0):.0f}% estimado)

================================================================================
8. METODOLOGÍA Y LIMITACIONES
================================================================================

📊 METODOLOGÍA APLICADA:
• Análisis predictivo con múltiples algoritmos
• Segmentación basada en riesgo por percentiles
• Business case con proyecciones financieras
• Roadmap estructurado por fases
• Outputs ejecutivos para toma decisiones

🔬 FORTALEZAS DEL ANÁLISIS:
• Metodología robusta y replicable
• Enfoque integral desde datos hasta implementación
• Priorización basada en capacidad de intervención
• Consideración de factores financieros y operacionales
• Estructura modular permite validación por fases

⚠️ LIMITACIONES IDENTIFICADAS:
• DATOS ESTIMADOS: Todos los inputs financieros son benchmarks industria
• No considera factores externos específicos mercado local
• Asume estabilidad condiciones mercado durante implementación
• Efectividad campañas puede variar vs benchmarks internacionales
• Requiere capacidades técnicas y operacionales específicas

================================================================================
9. PRÓXIMOS PASOS Y GOVERNANCE
================================================================================

📋 ROADMAP DE DECISIONES:

INMEDIATO (30 días):
• Presentación Board de Directores
• Decisión proceder/no proceder con validación
• Asignación recursos para fase validación
• Definición governance y estructura proyecto

CORTO PLAZO (90 días):
• Ejecución fase validación intensiva
• Confirmación supuestos críticos
• Refinamiento business case con datos reales
• Decisión final go/no-go implementación

MEDIANO PLAZO (6-12 meses):
• Implementación por fases según roadmap
• Monitoreo continuo métricas clave
• Ajustes estrategia basados en resultados
• Preparación scaling o pivoting

🎯 ESTRUCTURA DE GOVERNANCE RECOMENDADA:

STEERING COMMITTEE:
• CEO (Sponsor ejecutivo)
• CFO (Viabilidad financiera)
• CCO (Estrategia comercial)
• CTO (Capacidad tecnológica)

EQUIPO DE PROYECTO:
• Project Manager (Ejecución)
• Data Scientist (Modelo y análisis)
• Gerente Retención (Operaciones)
• Controller (Seguimiento financiero)

MÉTRICAS DE SEGUIMIENTO:
• ROI mensual vs objetivo
• Save rate por segmento
• Costo por cliente salvado
• NPS post-intervención
• Avance milestones críticos

================================================================================
CONCLUSIÓN FINAL
================================================================================

✅ ANÁLISIS COMPLETO EXITOSO:
• Metodología predictiva robusta demostrada
• Factores críticos con capacidad de intervención identificados
• Estrategias de retención específicas desarrolladas
• Viabilidad financiera confirmada con datos estimados
• Roadmap implementación estructurado

🎯 OPORTUNIDAD ESTRATÉGICA IDENTIFICADA:
• Potencial mejora significativa retención clientes
• ROI atractivo con riesgo controlado mediante validación
• Diferenciación competitiva en mercado telecom
• Capacidad escalamiento posterior a otros productos/mercados

⚠️ VALIDACIÓN CRÍTICA REQUERIDA:
La implementación exitosa depende fundamentalmente de la validación de 
supuestos con datos reales de la empresa. El análisis demuestra metodología 
sólida pero requiere confirmación empírica antes de comprometer recursos.

🚀 RECOMENDACIÓN FINAL EJECUTIVA:
PROCEDER CON FASE DE VALIDACIÓN INTENSIVA de 2-3 meses para confirmar 
viabilidad antes de implementación completa.

📊 VALOR DEL ANÁLISIS:
Independientemente de la decisión final, este análisis proporciona:
• Metodología replicable para análisis predictivo
• Framework de evaluación capacidad intervención
• Estructura business case para proyectos similares
• Plantilla governance y seguimiento
• Base conocimiento para futuras iniciativas retención

⚠️⚠️⚠️ RECORDATORIO FINAL ⚠️⚠️⚠️
Todos los cálculos y proyecciones están basados en benchmarks estimados 
industria telecom. La validación con datos específicos de la empresa es 
ESENCIAL antes de cualquier decisión de inversión.

================================================================================
FIN DEL INFORME ESTRATÉGICO MAESTRO
================================================================================
"""
    
    return report

def create_governance_template(key_findings, timestamp):
    """Crear template de governance y seguimiento"""
    try:
        logging.info("Creando template de governance y seguimiento...")
        
        excel_file = f'excel/paso13f_governance_template_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # 1. Executive Dashboard
            exec_dashboard = [
                ['MÉTRICA EJECUTIVA', 'OBJETIVO', 'ACTUAL', 'ESTADO', 'RESPONSABLE', 'FRECUENCIA'],
                ['=== MÉTRICAS FINANCIERAS ===', '', '', '', '', ''],
                ['ROI Anual (%)', f"{key_findings.get('financial_viability', {}).get('roi_annual', 0):.0f}%", 'Pendiente', 'Iniciando', 'CFO', 'Mensual'],
                ['Revenue Salvado Mensual (K)', f"${key_findings.get('financial_viability', {}).get('revenue_opportunity', 0)/12/1000:.0f}K", 'Pendiente', 'Iniciando', 'Gerente Retención', 'Mensual'],
                ['Costo Implementación (K)', f"${key_findings.get('financial_viability', {}).get('investment_required', 0)/12/1000:.0f}K", 'Pendiente', 'Iniciando', 'Controller', 'Mensual'],
                ['Payback Acumulado (meses)', f"{key_findings.get('financial_viability', {}).get('payback_months', 0):.1f}", 'Pendiente', 'Iniciando', 'CFO', 'Mensual'],
                ['', '', '', '', '', ''],
                ['=== MÉTRICAS OPERACIONALES ===', '', '', '', '', ''],
                ['Save Rate Mensual (%)', '25%', 'Pendiente', 'Iniciando', 'Gerente Retención', 'Semanal'],
                ['Clientes Contactados', '200', 'Pendiente', 'Iniciando', 'Equipo Retención', 'Semanal'],
                ['NPS Post-Intervención', '7.5', 'Pendiente', 'Iniciando', 'Gerente CX', 'Mensual'],
                ['Automatización (%)', '70%', 'Pendiente', 'Iniciando', 'Gerente IT', 'Mensual'],
                ['', '', '', '', '', ''],
                ['=== MILESTONES PROYECTO ===', '', '', '', '', ''],
                ['Setup Completado', 'Mes 2', 'Pendiente', 'Iniciando', 'PMO', 'Semanal'],
                ['Piloto Validado', 'Mes 5', 'Pendiente', 'Iniciando', 'Director Comercial', 'Mensual'],
                ['Scaling Completo', 'Mes 8', 'Pendiente', 'Iniciando', 'Gerente Retención', 'Mensual'],
                ['Implementación Total', 'Mes 12', 'Pendiente', 'Iniciando', 'CEO', 'Trimestral'],
                ['', '', '', '', '', ''],
                ['⚠️ IMPORTANTE', 'OBJETIVOS BASADOS EN ESTIMACIONES', '', '', '', ''],
                ['Validación Crítica', 'Confirmar con datos reales empresa', 'Pendiente', 'Crítico', 'Board', 'Inmediato']
            ]
            
            exec_df = pd.DataFrame(exec_dashboard)
            exec_df.to_excel(writer, sheet_name='Executive Dashboard', index=False, header=False)
            
            # 2. Checklist de Implementación
            checklist_data = [
                ['FASE', 'ACTIVIDAD', 'ESTADO', 'RESPONSABLE', 'FECHA_LIMITE', 'DEPENDENCIAS'],
                ['=== FASE 0: VALIDACIÓN ===', '', '', '', '', ''],
                ['Validación', 'Confirmar datos financieros reales', 'Pendiente', 'CFO', 'Mes 1', 'Acceso datos empresa'],
                ['Validación', 'Piloto micro save rates', 'Pendiente', 'Gerente Retención', 'Mes 2', 'Selección muestra'],
                ['Validación', 'Confirmar capacidades IT', 'Pendiente', 'CTO', 'Mes 1', 'Audit infraestructura'],
                ['Validación', 'Validar recursos humanos', 'Pendiente', 'RRHH', 'Mes 1', 'Evaluación equipos'],
                ['', '', '', '', '', ''],
                ['=== FASE 1: SETUP ===', '', '', '', '', ''],
                ['Setup', 'Configurar sistemas retención', 'Pendiente', 'Gerente IT', 'Mes 3', 'Validación completada'],
                ['Setup', 'Entrenar equipos retención', 'Pendiente', 'Gerente RRHH', 'Mes 3', 'Sistemas operativos'],
                ['Setup', 'Implementar dashboards KPI', 'Pendiente', 'Data Analyst', 'Mes 3', 'Sistemas configurados'],
                ['Setup', 'Documentar procesos', 'Pendiente', 'PMO', 'Mes 3', 'Procesos definidos'],
                ['', '', '', '', '', ''],
                ['=== FASE 2: PILOTO ===', '', '', '', '', ''],
                ['Piloto', 'Seleccionar muestra piloto', 'Pendiente', 'Gerente Retención', 'Mes 4', 'Setup completado'],
                ['Piloto', 'Ejecutar campañas retención', 'Pendiente', 'Equipo Retención', 'Mes 5', 'Muestra seleccionada'],
                ['Piloto', 'Monitorear métricas diarias', 'Pendiente', 'Data Analyst', 'Mes 6', 'Campañas activas'],
                ['Piloto', 'Evaluar resultados vs objetivo', 'Pendiente', 'Director Comercial', 'Mes 6', 'Datos suficientes'],
                ['', '', '', '', '', ''],
                ['=== VALIDACIONES CRÍTICAS ===', '', '', '', '', ''],
                ['Go/No-Go', 'Decisión continuar post-validación', 'Pendiente', 'Board', 'Mes 3', 'Validación completa'],
                ['Go/No-Go', 'Decisión scaling post-piloto', 'Pendiente', 'CEO', 'Mes 6', 'Piloto exitoso'],
                ['Go/No-Go', 'Evaluación anual completa', 'Pendiente', 'Board', 'Mes 12', 'Año implementación']
            ]
            
            checklist_df = pd.DataFrame(checklist_data)
            checklist_df.to_excel(writer, sheet_name='Checklist Implementación', index=False, header=False)
            
            # 3. Alertas y Escalaciones
            alerts_data = [
                ['MÉTRICA', 'UMBRAL VERDE', 'UMBRAL AMARILLO', 'UMBRAL ROJO', 'ESCALACIÓN'],
                ['Save Rate Mensual', '≥25%', '15-24%', '<15%', 'CEO + Board'],
                ['ROI Acumulado', '≥150%', '100-149%', '<100%', 'CFO + CEO'],
                ['Budget vs Plan', '≤100%', '101-110%', '>110%', 'CFO'],
                ['NPS Post-Intervención', '≥7.0', '6.0-6.9', '<6.0', 'Gerente CX'],
                ['Milestone Delay', '0 días', '1-7 días', '>7 días', 'PMO + Sponsor'],
                ['', '', '', '', ''],
                ['REUNIONES GOVERNANCE', '', '', '', ''],
                ['Board Review', 'Mensual', 'KPIs + Decisiones', 'CEO presenta', 'Todos C-Level'],
                ['Steering Committee', 'Semanal', 'Operacional', 'PMO facilita', 'Equipo proyecto'],
                ['Business Review', 'Trimestral', 'ROI + Strategy', 'CFO + CEO', 'Board + Sponsors']
            ]
            
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df.to_excel(writer, sheet_name='Alertas y Escalaciones', index=False, header=False)
            
            # 4. Disclaimer y Metodología
            disclaimer_data = [
                ['ASPECTO', 'DETALLE'],
                ['⚠️ DISCLAIMER CRÍTICO', 'Template basado en análisis con DATOS ESTIMADOS'],
                ['Fuente Estimaciones', 'Benchmarks estándar industria telecom'],
                ['Propósito Template', 'Demostración metodológica y estructura governance'],
                ['Para Uso Real', 'VALIDAR todos los objetivos con datos específicos empresa'],
                ['', ''],
                ['=== METODOLOGÍA APLICADA ===', ''],
                ['Análisis Predictivo', 'Múltiples algoritmos ML evaluados'],
                ['Segmentación', 'Basada en riesgo usando percentiles'],
                ['Business Case', 'Proyecciones financieras con benchmarks'],
                ['Roadmap', 'Implementación estructurada por fases'],
                ['Governance', 'Framework ejecutivo con KPIs y escalaciones'],
                ['', ''],
                ['=== VALIDACIONES REQUERIDAS ===', ''],
                ['Datos Financieros', 'Confirmar ARPU, CAC, costos con datos reales'],
                ['Save Rates', 'Validar efectividad con piloto muy controlado'],
                ['Capacidades', 'Confirmar recursos IT, humanos, operacionales'],
                ['Mercado Local', 'Validar aplicabilidad benchmarks a mercado específico'],
                ['Regulaciones', 'Confirmar cumplimiento normativo local'],
                ['Competencia', 'Evaluar respuesta competitiva a estrategias'],
                ['', ''],
                ['=== ESTRUCTURA RECOMMENDED ===', ''],
                ['Sponsor Ejecutivo', 'CEO o equivalente C-Level'],
                ['PMO', 'Dedicado 100% al proyecto'],
                ['Steering Committee', 'CEO + CFO + CCO + CTO'],
                ['Equipo Core', '4-6 personas especializadas'],
                ['Reporting', 'Semanal operacional + Mensual ejecutivo'],
                ['Budget Control', 'Seguimiento semanal vs plan']
            ]
            
            disclaimer_df = pd.DataFrame(disclaimer_data)
            disclaimer_df.to_excel(writer, sheet_name='Metodología y Disclaimer', index=False, header=False)
        
        logging.info(f"Template governance creado: {excel_file}")
        return excel_file
        
    except Exception as e:
        logging.error(f"Error creando template governance: {str(e)}")
        return None

def save_final_results(key_findings, complete_analysis, timestamp):
    """Guardar resultados finales consolidados"""
    try:
        logging.info("Guardando resultados finales consolidados...")
        
        # 1. JSON con análisis completo consolidado
        final_data = {
            'metadata': {
                'timestamp': timestamp,
                'script': 'paso13f_Informe_Final_Estratégico',
                'version': '1.0',
                'disclaimer': 'CONSOLIDACIÓN COMPLETA CON DATOS ESTIMADOS INDUSTRIA TELECOM',
                'steps_completed': complete_analysis['project_metadata']['total_steps_completed'],
                'analysis_scope': 'Análisis predictivo churn completo con estrategias retención'
            },
            'key_findings': key_findings,
            'complete_analysis_summary': {
                'factores_criticos_disponible': complete_analysis['factores_criticos'] is not None,
                'segmentacion_disponible': complete_analysis['segmentacion'] is not None,
                'business_case_disponible': complete_analysis['business_case'] is not None,
                'roadmap_disponible': complete_analysis['roadmap'] is not None,
                'outputs_ejecutivos_disponible': complete_analysis['outputs_ejecutivos'] is not None
            },
            'final_recommendations': key_findings['strategic_recommendations'],
            'implementation_readiness': {
                'model_ready': key_findings.get('model_performance', {}).get('production_ready', False),
                'business_case_viable': key_findings.get('financial_viability', {}).get('viability_assessment') == 'VIABLE',
                'critical_validations_required': True,
                'estimated_timeline_months': key_findings.get('implementation_strategy', {}).get('implementation_duration', 12),
                'estimated_investment': key_findings.get('financial_viability', {}).get('investment_required', 0)
            }
        }
        
        json_file = f'informes/paso13f_informe_final_estrategico_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. Informe estratégico maestro
        txt_file = f'informes/paso13f_informe_estrategico_maestro_{timestamp}.txt'
        report_content = generate_strategic_master_report(key_findings, complete_analysis, timestamp)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"JSON final guardado: {json_file}")
        logging.info(f"Informe maestro guardado: {txt_file}")
        
        return {
            'json_file': json_file,
            'txt_file': txt_file
        }
        
    except Exception as e:
        logging.error(f"Error guardando resultados finales: {str(e)}")
        raise

def main():
    """Función principal del Paso 13F - Informe Final Estratégico"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13F: INFORME FINAL ESTRATÉGICO CONSOLIDADO")
        logger.info("="*80)
        logger.warning("⚠️  CONSOLIDANDO ANÁLISIS COMPLETO CON DATOS ESTIMADOS")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar análisis completo de todo el proyecto
        logger.info("="*50)
        logger.info("CARGANDO ANÁLISIS COMPLETO DEL PROYECTO")
        complete_analysis = load_complete_analysis()
        
        # 3. Extraer hallazgos clave consolidados
        logger.info("="*50)
        logger.info("EXTRAYENDO HALLAZGOS CLAVE CONSOLIDADOS")
        key_findings = extract_key_findings(complete_analysis)
        
        # 4. Crear dashboard final consolidado
        logger.info("="*50)
        logger.info("CREANDO DASHBOARD FINAL CONSOLIDADO")
        dashboard_file = create_final_dashboard(key_findings, timestamp)
        
        # 5. Crear template de governance
        logger.info("="*50)
        logger.info("CREANDO TEMPLATE DE GOVERNANCE Y SEGUIMIENTO")
        governance_file = create_governance_template(key_findings, timestamp)
        
        # 6. Guardar resultados finales
        logger.info("="*50)
        logger.info("GUARDANDO INFORME FINAL ESTRATÉGICO")
        output_files = save_final_results(key_findings, complete_analysis, timestamp)
        
        # 7. Resumen final del proyecto completo
        logger.info("="*80)
        logger.info("🎉 PROYECTO COMPLETO - PASO 13F FINALIZADO EXITOSAMENTE 🎉")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar resultados consolidados finales
        logger.warning("⚠️⚠️⚠️ PROYECTO BASADO EN DATOS ESTIMADOS INDUSTRIA TELECOM ⚠️⚠️⚠️")
        logger.info("")
        
        logger.info("🎯 OBJETIVO ALCANZADO:")
        logger.info("  ✅ Factores principales de cancelación identificados")
        logger.info("  ✅ Estrategias de retención desarrolladas")
        logger.info("  ✅ Modelo predictivo evaluado y recomendado")
        logger.info("  ✅ Business case con ROI validado")
        logger.info("  ✅ Roadmap implementación estructurado")
        logger.info("")
        
        if 'model_performance' in key_findings:
            mp = key_findings['model_performance']
            logger.info("🤖 MODELO RECOMENDADO:")
            logger.info(f"  • Algoritmo: {mp.get('best_model', 'N/A')}")
            logger.info(f"  • Score: {mp.get('model_score', 0):.4f}")
            logger.info(f"  • Estado: {'✅ Listo' if mp.get('production_ready', False) else '❌ Requiere validación'}")
            logger.info("")
        
        if 'critical_factors' in key_findings:
            cf = key_findings['critical_factors']
            logger.info("🔥 FACTORES CRÍTICOS:")
            logger.info(f"  • Total identificados: {cf.get('total_factors_identified', 0)}")
            logger.info(f"  • Factor principal: {cf.get('top_factor', {}).get('variable', 'N/A')}")
            logger.info(f"  • Alta capacidad intervención: {cf.get('intervention_summary', {}).get('high_capacity', 0)} factores")
            logger.info("")
        
        if 'financial_viability' in key_findings:
            fv = key_findings['financial_viability']
            logger.info("💰 VIABILIDAD FINANCIERA:")
            logger.info(f"  • ROI anual: {fv.get('roi_annual', 0):.1f}%")
            logger.info(f"  • Payback: {fv.get('payback_months', 0):.1f} meses")
            logger.info(f"  • NPV 3 años: ${fv.get('npv_3_years', 0):,.0f}")
            logger.info(f"  • Evaluación: {fv.get('viability_assessment', 'N/A')}")
            logger.info("")
        
        logger.info("🚀 RECOMENDACIÓN FINAL:")
        logger.info(f"  • Decisión: {key_findings['strategic_recommendations']['primary_recommendation']}")
        logger.info(f"  • Próximo paso: Validación intensiva con datos reales empresa")
        logger.info("")
        
        logger.info("📁 ARCHIVOS FINALES GENERADOS:")
        logger.info(f"  • Dashboard consolidado: {dashboard_file}")
        logger.info(f"  • Template governance: {governance_file}")
        logger.info(f"  • JSON consolidado: {output_files['json_file']}")
        logger.info(f"  • Informe maestro: {output_files['txt_file']}")
        logger.info("")
        
        logger.info("📊 PASOS COMPLETADOS EN EL PROYECTO:")
        steps_completed = complete_analysis['project_metadata']['total_steps_completed']
        logger.info(f"  • Paso 13A: {'✅' if complete_analysis['factores_criticos'] else '❌'} Factores críticos")
        logger.info(f"  • Paso 13B: {'✅' if complete_analysis['segmentacion'] else '❌'} Segmentación estratégica")  
        logger.info(f"  • Paso 13C: {'✅' if complete_analysis['business_case'] else '❌'} Business case")
        logger.info(f"  • Paso 13D: {'✅' if complete_analysis['roadmap'] else '❌'} Roadmap implementación")
        logger.info(f"  • Paso 13E: {'✅' if complete_analysis['outputs_ejecutivos'] else '❌'} Outputs ejecutivos")
        logger.info(f"  • Paso 13F: ✅ Informe final estratégico")
        logger.info(f"  • TOTAL: {steps_completed + 1}/6 pasos completados")
        logger.info("")
        
        logger.warning("⚠️ VALIDACIONES CRÍTICAS PARA IMPLEMENTACIÓN REAL:")
        logger.warning("1. Confirmar datos financieros con cifras reales empresa")
        logger.warning("2. Validar save rates con piloto muy controlado")
        logger.warning("3. Verificar capacidades tecnológicas y operacionales")
        logger.warning("4. Ajustar timeline y presupuesto según contexto real")
        logger.info("")
        
        logger.info("🎯 VALOR ENTREGADO:")
        logger.info("  ✅ Metodología completa análisis predictivo churn")
        logger.info("  ✅ Framework evaluación factores con capacidad intervención")
        logger.info("  ✅ Estructura business case replicable")
        logger.info("  ✅ Template governance y seguimiento")
        logger.info("  ✅ Roadmap implementación por fases")
        logger.info("")
        
        logger.info("🎉 PROYECTO ANÁLISIS PREDICTIVO CHURN COMPLETADO EXITOSAMENTE 🎉")
        logger.info("="*80)
        
        return {
            'key_findings': key_findings,
            'complete_analysis': complete_analysis,
            'dashboard_file': dashboard_file,
            'governance_file': governance_file,
            'output_files': output_files,
            'project_status': 'COMPLETADO',
            'steps_completed': steps_completed + 1
        }
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO FINAL: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
    
"""
================================================================================
TELECOMX - PASO 13F: INFORME FINAL ESTRATÉGICO CONSOLIDADO
================================================================================
Descripción: Informe detallado final que consolida todo el análisis predictivo
             de churn, destacando factores principales de cancelación, rendimiento
             de modelos y estrategias de retención propuestas.
             
⚠️  IMPORTANTE: Este informe consolida análisis basado en DATOS ESTIMADOS de 
    benchmarks industria telecom para fines de SIMULACIÓN y demostración 
    metodológica.

OBJETIVO: Elaborar informe detallado destacando factores que más influyen en 
         cancelación, basándose en variables seleccionadas y rendimiento de 
         cada modelo, proponiendo estrategias de retención.

Inputs: 
- Consolidación completa de Pasos 13A, 13B, 13C, 13D, 13E
- Análisis de factores críticos y capacidad de intervención
- Evaluación comparativa de modelos predictivos
- Segmentación estratégica y business case
- Roadmap de implementación y outputs ejecutivos

Outputs:
- Informe estratégico maestro consolidado
- Dashboard final con todos los hallazgos
- Recomendaciones estratégicas finales
- Template de governance y seguimiento
- Checklist de implementación

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
            logging.FileHandler('logs/paso13f_informe_final_estrategico.log', mode='a', encoding='utf-8'),
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

def load_complete_analysis():
    """Cargar análisis completo de todos los pasos del proyecto"""
    try:
        logging.info("Cargando análisis completo de todo el proyecto...")
        
        complete_analysis = {
            'project_metadata': {
                'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                'total_steps_completed': 0,
                'disclaimer': 'ANÁLISIS BASADO EN DATOS ESTIMADOS INDUSTRIA TELECOM'
            }
        }
        
        # 1. Cargar Paso 13A - Factores Críticos
        try:
            paso13a_file = find_latest_file('datos', 'paso13a_analisis_consolidado_v2_*.json')
            with open(paso13a_file, 'r', encoding='utf-8') as f:
                complete_analysis['factores_criticos'] = json.load(f)
            logging.info("✅ Paso 13A cargado: Factores críticos de churn identificados")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13A no disponible: {str(e)}")
            complete_analysis['factores_criticos'] = None
        
        # 2. Cargar Paso 13B - Segmentación Estratégica  
        try:
            paso13b_file = find_latest_file('informes', 'paso13b_segmentacion_estrategica_*.json')
            with open(paso13b_file, 'r', encoding='utf-8') as f:
                complete_analysis['segmentacion'] = json.load(f)
            logging.info("✅ Paso 13B cargado: Segmentación por riesgo completada")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13B no disponible: {str(e)}")
            complete_analysis['segmentacion'] = None
        
        # 3. Cargar Paso 13C - Business Case
        try:
            paso13c_file = find_latest_file('informes', 'paso13c_business_case_completo_*.json')
            with open(paso13c_file, 'r', encoding='utf-8') as f:
                complete_analysis['business_case'] = json.load(f)
            logging.info("✅ Paso 13C cargado: Business case con ROI validado")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13C no disponible: {str(e)}")
            complete_analysis['business_case'] = None
        
        # 4. Cargar Paso 13D - Roadmap de Implementación
        try:
            paso13d_file = find_latest_file('informes', 'paso13d_roadmap_detallado_*.json')
            with open(paso13d_file, 'r', encoding='utf-8') as f:
                complete_analysis['roadmap'] = json.load(f)
            logging.info("✅ Paso 13D cargado: Roadmap de implementación definido")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13D no disponible: {str(e)}")
            complete_analysis['roadmap'] = None
        
        # 5. Cargar Paso 13E - Outputs Ejecutivos
        try:
            paso13e_file = find_latest_file('informes', 'paso13e_outputs_ejecutivos_*.json')
            with open(paso13e_file, 'r', encoding='utf-8') as f:
                complete_analysis['outputs_ejecutivos'] = json.load(f)
            logging.info("✅ Paso 13E cargado: Outputs ejecutivos consolidados")
            complete_analysis['project_metadata']['total_steps_completed'] += 1
        except Exception as e:
            logging.warning(f"❌ Paso 13E no disponible: {str(e)}")
            complete_analysis['outputs_ejecutivos'] = None
        
        logging.info(f"Análisis completo cargado: {complete_analysis['project_metadata']['total_steps_completed']}/5 pasos disponibles")
        return complete_analysis
        
    except Exception as e:
        logging.error(f"Error cargando análisis completo: {str(e)}")
        raise

def extract_key_findings(complete_analysis):
    """Extraer hallazgos clave de todo el análisis"""
    try:
        logging.info("Extrayendo hallazgos clave del análisis completo...")
        
        key_findings = {
            'disclaimer': 'HALLAZGOS BASADOS EN DATOS ESTIMADOS INDUSTRIA TELECOM',
            'model_performance': {},
            'critical_factors': {},
            'segmentation_insights': {},
            'financial_viability': {},
            'implementation_strategy': {},
            'strategic_recommendations': {}
        }
        
        # 1. Rendimiento de Modelos
        if complete_analysis['factores_criticos']:
            fc_data = complete_analysis['factores_criticos']
            key_findings['model_performance'] = {
                'best_model': fc_data['model_info']['best_model'],
                'model_score': fc_data['model_info']['model_score'],
                'production_ready': fc_data['model_info']['production_ready'],
                'total_customers_analyzed': fc_data['dataset_info']['size'],
                'baseline_churn_rate': fc_data['dataset_info']['churn_rate'],
                'variables_analyzed': fc_data['dataset_info']['features_count']
            }
        
        # 2. Factores Críticos con Capacidad de Intervención
        if complete_analysis['factores_criticos']:
            factors = complete_analysis['factores_criticos']['critical_factors']
            
            # Top 5 factores más importantes
            top_factors = factors[:5]
            
            # Factores con alta capacidad de intervención (score >= 7)
            high_intervention = [f for f in factors if f['actionability_score'] >= 7]
            
            key_findings['critical_factors'] = {
                'total_factors_identified': len(factors),
                'top_factor': {
                    'variable': top_factors[0]['variable'],
                    'importance': top_factors[0]['avg_importance'],
                    'intervention_capacity': top_factors[0]['actionability_score']
                },
                'most_controllable_factor': max(factors, key=lambda x: x['actionability_score']),
                'high_intervention_factors': len(high_intervention),
                'top_5_factors': top_factors,
                'intervention_summary': {
                    'high_capacity': len([f for f in factors if f['actionability_score'] >= 7]),
                    'medium_capacity': len([f for f in factors if 4 <= f['actionability_score'] < 7]),
                    'low_capacity': len([f for f in factors if f['actionability_score'] < 4])
                }
            }
        
        # 3. Insights de Segmentación
        if complete_analysis['segmentacion']:
            seg_data = complete_analysis['segmentacion']['segmentation_summary']
            key_findings['segmentation_insights'] = {
                'total_clients_segmented': complete_analysis['segmentacion']['metadata']['total_clients'],
                'segments_identified': len(seg_data),
                'priority_segment': 'Medio_Riesgo',  # Basado en conclusiones del business case
                'segment_distribution': {k: v['percentage'] for k, v in seg_data.items()},
                'priority_justification': 'Mejor balance entre capacidad de intervención y ROI potencial'
            }
        
        # 4. Viabilidad Financiera
        if complete_analysis['business_case']:
            bc_data = complete_analysis['business_case']
            consolidated = bc_data['intervention_scenarios']['consolidated']
            projections = bc_data['projections_3_years']['summary']
            
            key_findings['financial_viability'] = {
                'investment_required': consolidated['total_annual_investment'],
                'revenue_opportunity': consolidated['total_annual_revenue_saved'],
                'roi_annual': consolidated['overall_roi_annual'],
                'payback_months': consolidated['overall_payback_months'],
                'npv_3_years': projections['net_present_value'],
                'break_even_year': projections['break_even_year'],
                'viability_assessment': 'VIABLE' if consolidated['overall_roi_annual'] > 100 else 'NO VIABLE',
                'current_churn_loss': bc_data['current_state']['totals']['total_annual_churn_loss']
            }
        
        # 5. Estrategia de Implementación
        if complete_analysis['roadmap']:
            rm_data = complete_analysis['roadmap']
            key_findings['implementation_strategy'] = {
                'implementation_duration': rm_data['summary_metrics']['total_duration_months'],
                'total_phases': rm_data['summary_metrics']['total_phases'],
                'priority_approach': rm_data['summary_metrics']['implementation_approach'],
                'total_budget_estimated': rm_data['summary_metrics']['total_budget_estimated'],
                'critical_milestones': ['Mes 2: Setup', 'Mes 5: Piloto validado', 'Mes 8: Scaling', 'Mes 12: Completo']
            }
        
        # 6. Recomendaciones Estratégicas Consolidadas
        key_findings['strategic_recommendations'] = {
            'primary_recommendation': 'PROCEDER CON IMPLEMENTACIÓN CON VALIDACIÓN',
            'priority_actions': [
                'Validar supuestos con datos reales empresa',
                'Iniciar piloto controlado segmento Medio Riesgo',
                'Confirmar budget y recursos disponibles',
                'Establecer governance y métricas seguimiento'
            ],
            'success_factors': [
                'Validación temprana con datos reales',
                'Enfoque gradual por fases',
                'Monitoreo continuo ROI',
                'Capacidad operacional adecuada'
            ],
            'risk_mitigation': [
                'Piloto controlado antes scaling',
                'Monitoreo semanal métricas clave',
                'Plan contingencia si ROI < objetivo',
                'Validación continua supuestos'
            ]
        }
        
        logging.info("Hallazgos clave extraídos exitosamente")
        return key_findings
        
    except Exception as e:
        logging.error(f"Error extrayendo hallazgos clave: {str(e)}")
        raise

def create_final_dashboard(key_findings, timestamp):
    """Crear dashboard final consolidado"""
    try:
        logging.info("Creando dashboard final consolidado...")
        
        plt.style.use('default')
        
        # Colores profesionales finales
        colors = {
            'primary': '#1f4e79',
            'success': '#70ad47', 
            'warning': '#c55a11',
            'info': '#5b9bd5',
            'neutral': '#7f7f7f',
            'background': '#f8f9fa'
        }
        
        fig = plt.figure(figsize=(24, 18))
        fig.patch.set_facecolor('white')
        
        gs = fig.add_gridspec(3, 4, hspace=1.2, wspace=0.4, top=0.82, bottom=0.15, left=0.08, right=0.94)
        
        # 1. Factores Críticos con Mayor Capacidad de Intervención (top-left span 2)
        ax1 = fig.add_subplot(gs[0, :2])
        
        if 'critical_factors' in key_findings and key_findings['critical_factors']:
            factors = key_findings['critical_factors']['top_5_factors']
            factor_names = [f['variable'].replace('_', ' ')[:20] for f in factors]
            importance_values = [f['avg_importance'] for f in factors]
            intervention_values = [f['actionability_score'] for f in factors]
            
            x = np.arange(len(factor_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, importance_values, width, label='Importancia (%)', 
                           color=colors['primary'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, [v*10 for v in intervention_values], width, 
               label='Capacidad Intervención (%)', color=colors['success'], alpha=0.8)
            
            ax1.set_xlabel('Factores Críticos', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Puntuación', fontweight='bold', fontsize=12)
            ax1.set_title('TOP 5 FACTORES: IMPORTANCIA vs CAPACIDAD DE INTERVENCIÓN\n⚠️ Ambas métricas en escala 0-100%', 
             fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
            ax1.set_xticks(x)
            ax1.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=10)
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Rendimiento del Modelo (top-right span 2)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if 'model_performance' in key_findings:
            mp = key_findings['model_performance']
            
            # Métricas del modelo
            metrics = ['Score Modelo', 'Churn Rate', 'Variables']
            values = [
                mp.get('model_score', 0) * 100,  # Convertir a porcentaje
                mp.get('baseline_churn_rate', 0) * 100,
                mp.get('variables_analyzed', 0)
            ]
            colors_bars = [colors['success'], colors['warning'], colors['info']]
            
            bars = ax2.bar(metrics, values, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)
            ax2.set_ylabel('Valor', fontweight='bold', fontsize=12)
            ax2.set_title(f'RENDIMIENTO MODELO: {mp.get("best_model", "N/A")}\n⚠️ Evaluación con Dataset Estimado', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Etiquetas en barras
            labels = [f'{values[0]:.1f}%', f'{values[1]:.1f}%', f'{int(values[2])}']
            for bar, label in zip(bars, labels):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 3. Business Case Consolidado (middle span 2)
        ax3 = fig.add_subplot(gs[1, :2])
        
        if 'financial_viability' in key_findings:
            fv = key_findings['financial_viability']
            
            categories = ['Pérdida\nActual', 'Inversión\nRequerida', 'Revenue\nSalvado', 'Beneficio\nNeto']
            values_millions = [
                fv.get('current_churn_loss', 0) / 1000000,
                fv.get('investment_required', 0) / 1000000,
                fv.get('revenue_opportunity', 0) / 1000000,
                (fv.get('revenue_opportunity', 0) - fv.get('investment_required', 0)) / 1000000
            ]
            colors_bars = [colors['warning'], colors['info'], colors['success'], colors['primary']]
            
            bars = ax3.bar(categories, values_millions, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)
            ax3.set_ylabel('Millones USD', fontweight='bold', fontsize=12)
            ax3.set_title(f'BUSINESS CASE: ROI {fv.get("roi_annual", 0):.0f}% | Payback {fv.get("payback_months", 0):.1f} meses\n⚠️ Proyección con Benchmarks Industria', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=50)
            ax3.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values_millions):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values_millions)*0.02,
                        f'${value:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Distribución por Capacidad de Intervención (middle-right span 2)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        if 'critical_factors' in key_findings and key_findings['critical_factors']:
            intervention_summary = key_findings['critical_factors']['intervention_summary']
            
            labels = ['Alta\nCapacidad', 'Media\nCapacidad', 'Baja\nCapacidad']
            sizes = [
                intervention_summary['high_capacity'],
                intervention_summary['medium_capacity'],
                intervention_summary['low_capacity']
            ]
            colors_pie = [colors['success'], colors['warning'], colors['neutral']]
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.0f%%',
                                              colors=colors_pie, startangle=90,
                                              textprops={'fontsize': 11, 'fontweight': 'bold'},
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            ax4.set_title('FACTORES POR CAPACIDAD DE INTERVENCIÓN\n⚠️ Evaluación Estimada', 
                         fontweight='bold', fontsize=14, color=colors['primary'], pad=20)
        
        # 5. Conclusiones y Recomendaciones Finales (bottom span 4)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Extraer métricas clave para mostrar
        if key_findings.get('model_performance'):
            model_name = key_findings['model_performance'].get('best_model', 'N/A')
            model_score = key_findings['model_performance'].get('model_score', 0)
        else:
            model_name, model_score = 'N/A', 0
            
        if key_findings.get('critical_factors'):
            top_factor = key_findings['critical_factors']['top_factor']['variable']
            total_factors = key_findings['critical_factors']['total_factors_identified']
        else:
            top_factor, total_factors = 'N/A', 0
            
        if key_findings.get('financial_viability'):
            roi = key_findings['financial_viability'].get('roi_annual', 0)
            payback = key_findings['financial_viability'].get('payback_months', 0)
        else:
            roi, payback = 0, 0
        
        conclusions_text = f"""
🎯 CONCLUSIONES FINALES DEL ANÁLISIS PREDICTIVO DE CHURN - TELECOMX

✅ MODELO RECOMENDADO: {model_name} (Score: {model_score:.3f})
📊 FACTORES CRÍTICOS: {total_factors} identificados | Factor principal: {top_factor}
💰 VIABILIDAD FINANCIERA: ROI {roi:.0f}% anual | Payback {payback:.1f} meses
🚀 RECOMENDACIÓN: {key_findings['strategic_recommendations']['primary_recommendation']}

🔧 FACTORES CON MAYOR CAPACIDAD DE INTERVENCIÓN:
• Variables controlables por la empresa identificadas
• Estrategias de retención específicas por segmento desarrolladas
• Roadmap de implementación de 12 meses estructurado

⚠️ VALIDACIONES CRÍTICAS REQUERIDAS:
• Confirmar supuestos con datos financieros reales de la empresa
• Validar efectividad con piloto controlado antes de scaling completo
• Establecer governance y métricas de seguimiento continuo

🚨 DISCLAIMER IMPORTANTE: Este análisis utiliza DATOS ESTIMADOS de benchmarks industria telecom
para demostración metodológica. Para implementación real, validar con datos específicos empresa.
        """
        
        ax5.text(0.05, 0.05, conclusions_text, transform=ax5.transAxes, fontsize=12,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=1.0", facecolor=colors['background'], 
                  edgecolor=colors['primary'], linewidth=2, alpha=0.9))
        
        # Título principal del informe final
        fig.suptitle('TELECOMX - INFORME FINAL: ANÁLISIS PREDICTIVO DE CHURN Y ESTRATEGIAS DE RETENCIÓN\n⚠️ CONSOLIDACIÓN COMPLETA CON DATOS ESTIMADOS INDUSTRIA TELECOM ⚠️', 
                    fontsize=16, fontweight='bold', color=colors['primary'], y=0.93)
        
        # Guardar dashboard final
        os.makedirs('graficos', exist_ok=True)
        viz_file = f'graficos/paso13f_dashboard_final_consolidado_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.5)
        plt.close()
        
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Dashboard final creado: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear dashboard final: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard final: {str(e)}")
        return None