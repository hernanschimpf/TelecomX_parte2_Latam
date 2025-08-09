"""
================================================================================
TELECOMX - PASO 13E: OUTPUTS EJECUTIVOS PARA CEO Y BOARD DE DIRECTORES
================================================================================
Descripción: Creación de dashboards ejecutivos y outputs de alto nivel para
             presentación a CEO y Board de Directores. Consolida resultados
             de análisis predictivo, business case y roadmap de implementación.
             
⚠️  IMPORTANTE: Outputs basados en análisis que utiliza DATOS ESTIMADOS de 
    benchmarks industria telecom para fines de SIMULACIÓN.

Inputs: 
- Análisis consolidado del Paso 13A
- Segmentación estratégica del Paso 13B
- Business case completo del Paso 13C
- Roadmap detallado del Paso 13D

Outputs:
- Dashboard ejecutivo consolidado
- Resumen ejecutivo de 1 página
- Visualizaciones corporativas para Board
- Template de seguimiento de KPIs
- Presentación ejecutiva lista para Board

Audiencia: CEO, CFO, Board de Directores
Estilo: Profesional, corporativo, high-level

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
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso13e_outputs_ejecutivos.log', mode='a', encoding='utf-8'),
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

def load_consolidated_results():
    """Cargar resultados consolidados de todos los pasos anteriores"""
    try:
        logging.info("Cargando resultados consolidados de pasos 13A-13D...")
        
        results = {}
        
        # 1. Cargar Paso 13A - Análisis Consolidado
        try:
            paso13a_file = find_latest_file('datos', 'paso13a_analisis_consolidado_v2_*.json')
            with open(paso13a_file, 'r', encoding='utf-8') as f:
                results['paso13a'] = json.load(f)
            logging.info("Paso 13A cargado: Factores críticos identificados")
        except Exception as e:
            logging.warning(f"No se pudo cargar Paso 13A: {str(e)}")
            results['paso13a'] = None
        
        # 2. Cargar Paso 13B - Segmentación Estratégica
        try:
            paso13b_file = find_latest_file('informes', 'paso13b_segmentacion_estrategica_*.json')
            with open(paso13b_file, 'r', encoding='utf-8') as f:
                results['paso13b'] = json.load(f)
            logging.info("Paso 13B cargado: Segmentación por riesgo completada")
        except Exception as e:
            logging.warning(f"No se pudo cargar Paso 13B: {str(e)}")
            results['paso13b'] = None
        
        # 3. Cargar Paso 13C - Business Case
        try:
            paso13c_file = find_latest_file('informes', 'paso13c_business_case_completo_*.json')
            with open(paso13c_file, 'r', encoding='utf-8') as f:
                results['paso13c'] = json.load(f)
            logging.info("Paso 13C cargado: Business case con ROI validado")
        except Exception as e:
            logging.warning(f"No se pudo cargar Paso 13C: {str(e)}")
            results['paso13c'] = None
        
        # 4. Cargar Paso 13D - Roadmap
        try:
            paso13d_file = find_latest_file('informes', 'paso13d_roadmap_detallado_*.json')
            with open(paso13d_file, 'r', encoding='utf-8') as f:
                results['paso13d'] = json.load(f)
            logging.info("Paso 13D cargado: Roadmap de implementación definido")
        except Exception as e:
            logging.warning(f"No se pudo cargar Paso 13D: {str(e)}")
            results['paso13d'] = None
        
        # Verificar datos cargados
        loaded_steps = sum([1 for step in results.values() if step is not None])
        logging.info(f"Resultados consolidados: {loaded_steps}/4 pasos cargados exitosamente")
        
        return results
        
    except Exception as e:
        logging.error(f"Error cargando resultados consolidados: {str(e)}")
        raise

def extract_executive_metrics(consolidated_results):
    """Extraer métricas ejecutivas clave de todos los análisis"""
    try:
        logging.info("Extrayendo métricas ejecutivas clave...")
        
        executive_metrics = {
            'disclaimer': '⚠️ ANÁLISIS BASADO EN DATOS ESTIMADOS INDUSTRIA TELECOM',
            'project_overview': {},
            'financial_impact': {},
            'implementation': {},
            'risk_assessment': {},
            'recommendations': {}
        }
        
        # 1. Métricas del Análisis Consolidado (13A)
        if consolidated_results['paso13a']:
            data_13a = consolidated_results['paso13a']
            executive_metrics['project_overview'] = {
                'total_customers': data_13a['dataset_info']['size'],
                'baseline_churn_rate': data_13a['dataset_info']['churn_rate'],
                'model_performance': data_13a['model_info']['model_score'],
                'critical_factors_identified': len(data_13a['critical_factors']),
                'top_factor': data_13a['critical_factors'][0]['variable'],
                'most_actionable_factor': max(data_13a['critical_factors'], key=lambda x: x['actionability_score'])['variable']
            }
        
        # 2. Métricas Financieras del Business Case (13C)
        if consolidated_results['paso13c']:
            data_13c = consolidated_results['paso13c']
            intervention = data_13c['intervention_scenarios']['consolidated']
            projections = data_13c['projections_3_years']
            
            executive_metrics['financial_impact'] = {
                'annual_investment_required': intervention['total_annual_investment'],
                'annual_revenue_saved': intervention['total_annual_revenue_saved'],
                'roi_annual': intervention['overall_roi_annual'],
                'payback_months': intervention['overall_payback_months'],
                'npv_3_years': projections['summary']['net_present_value'],
                'break_even_year': projections['summary']['break_even_year'],
                'current_churn_loss': data_13c['current_state']['totals']['total_annual_churn_loss']
            }
        
        # 3. Métricas de Implementación del Roadmap (13D)
        if consolidated_results['paso13d']:
            data_13d = consolidated_results['paso13d']
            executive_metrics['implementation'] = {
                'implementation_duration': data_13d['summary_metrics']['total_duration_months'],
                'total_phases': data_13d['summary_metrics']['total_phases'],
                'priority_segment': data_13d['summary_metrics']['priority_segment'],
                'total_budget_estimated': data_13d['summary_metrics']['total_budget_estimated'],
                'critical_milestone_month': 5  # Mes crítico de validación piloto
            }
        
        # 4. Segmentación del Paso 13B
        if consolidated_results['paso13b']:
            data_13b = consolidated_results['paso13b']
            executive_metrics['segmentation'] = {
                'total_clients_segmented': data_13b['metadata']['total_clients'],
                'segmentation_method': data_13b['metadata']['segmentation_method'],
                'segments': data_13b['segmentation_summary']
            }
        
        # 5. Evaluación de Riesgos
        executive_metrics['risk_assessment'] = {
            'primary_risk': 'Save rates reales menores a estimados benchmark',
            'risk_level': 'MEDIO',
            'mitigation': 'Validación temprana con piloto controlado',
            'data_dependency': 'ALTA - Análisis basado en estimaciones industria'
        }
        
        # 6. Recomendaciones Ejecutivas
        executive_metrics['recommendations'] = {
            'primary_recommendation': 'PROCEDER CON IMPLEMENTACIÓN',
            'priority_action': 'Iniciar con piloto Medio Riesgo',
            'critical_validation': 'Confirmar supuestos con datos reales empresa',
            'success_probability': 'ALTA con validación adecuada'
        }
        
        logging.info("Métricas ejecutivas extraídas exitosamente")
        return executive_metrics
        
    except Exception as e:
        logging.error(f"Error extrayendo métricas ejecutivas: {str(e)}")
        raise

def create_executive_dashboard(executive_metrics, timestamp):
    """Crear dashboard ejecutivo consolidado para CEO/Board"""
    try:
        logging.info("Creando dashboard ejecutivo para CEO/Board...")
        
        # Configurar estilo corporativo
        plt.style.use('default')
        
        # Colores corporativos profesionales
        colors = {
            'primary': '#1f4e79',      # Azul corporativo
            'secondary': '#70ad47',    # Verde profesional  
            'accent': '#c55a11',       # Naranja ejecutivo
            'neutral': '#7f7f7f',      # Gris corporativo
            'success': '#548235',      # Verde éxito
            'warning': '#d99694',      # Rojo suave
            'background': '#f8f9fa'    # Fondo claro
        }
        
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('white')
        
        # Layout ejecutivo optimizado
        gs = fig.add_gridspec(3, 4, hspace=0.6, wspace=0.4, top=0.85, bottom=0.10, left=0.06, right=0.96)
        
        # 1. RESUMEN FINANCIERO EJECUTIVO (Top priority - span 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        
        if 'financial_impact' in executive_metrics:
            fin = executive_metrics['financial_impact']
            
            metrics_labels = ['Costo\nAnual', 'Beneficio\nAnual', 'ROI\nAnual', 'Valor\n3 Años']
            metrics_values = [
                fin['annual_investment_required'] / 1000,     # Miles
                fin['annual_revenue_saved'] / 1000,          # Miles  
                fin['roi_annual'],
                fin['npv_3_years'] / 1000                    # Miles
            ]
            metrics_units = ['$K', '$K', '%', '$K']
            
            x_pos = np.arange(len(metrics_labels))
            bars = ax1.bar(x_pos, metrics_values, 
                          color=[colors['accent'], colors['success'], colors['primary'], colors['secondary']], 
                          alpha=0.8, edgecolor='white', linewidth=2)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(metrics_labels, fontsize=12, fontweight='bold')
            ax1.set_title('MÉTRICAS FINANCIERAS CLAVE\n⚠️ Basado en Estimaciones Industria Telecom', 
                         fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Agregar valores en las barras
            for bar, value, unit in zip(bars, metrics_values, metrics_units):
                height = bar.get_height()
                if unit == '%':
                    label = f'{value:.0f}{unit}'
                else:
                    label = f'${value:.0f}K'
                ax1.text(bar.get_x() + bar.get_width()/2, height + max(metrics_values)*0.02,
                        label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. ESTADO ACTUAL vs OPORTUNIDAD (top-right span 2 cols)
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if 'project_overview' in executive_metrics and 'financial_impact' in executive_metrics:
            current_loss = executive_metrics['financial_impact']['current_churn_loss'] / 1000000
            potential_save = executive_metrics['financial_impact']['annual_revenue_saved'] / 1000000
            
            categories = ['Pérdida Actual\npor Churn', 'Oportunidad\nde Retención']
            values = [current_loss, potential_save]
            colors_bars = [colors['warning'], colors['success']]
            
            bars = ax2.bar(categories, values, color=colors_bars, alpha=0.8, edgecolor='white', linewidth=2)
            ax2.set_ylabel('Millones USD', fontsize=12, fontweight='bold')
            ax2.set_title('IMPACTO FINANCIERO: SITUACIÓN ACTUAL vs OPORTUNIDAD\n⚠️ Proyección con Datos Estimados', 
                         fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        f'${value:.1f}M', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 3. TIMELINE EJECUTIVO (middle-left span 2 cols)
        ax3 = fig.add_subplot(gs[1, :2])
        
        if 'implementation' in executive_metrics:
            impl = executive_metrics['implementation']
            
            # Timeline simplificado para ejecutivos
            phases = ['Setup\n(2m)', 'Piloto\n(3m)', 'Scaling\n(2m)', 'Expansion\n(5m)']
            phase_colors = [colors['neutral'], colors['primary'], colors['success'], colors['secondary']]
            
            # Crear timeline visual
            y_pos = 0.5
            phase_widths = [2, 3, 2, 5]  # Duración en meses
            x_start = 0
            
            for i, (phase, width, color) in enumerate(zip(phases, phase_widths, phase_colors)):
                rect = plt.Rectangle((x_start, y_pos-0.15), width, 0.3, 
                                   facecolor=color, alpha=0.7, edgecolor='white', linewidth=2)
                ax3.add_patch(rect)
                
                # Agregar texto en el centro de cada fase
                ax3.text(x_start + width/2, y_pos, phase, ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
                
                x_start += width
            
            ax3.set_xlim(0, 12)
            ax3.set_ylim(0, 1)
            ax3.set_xlabel('Meses', fontsize=12, fontweight='bold')
            ax3.set_title('CRONOGRAMA EJECUTIVO DE IMPLEMENTACIÓN\n⚠️ Timeline Estimado 12 Meses', 
                         fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
            ax3.set_xticks(range(0, 13, 2))
            ax3.set_yticks([])
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. SEGMENTACIÓN Y PRIORIZACIÓN (middle-right span 2 cols)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        if 'segmentation' in executive_metrics:
            seg = executive_metrics['segmentation']['segments']
            
            segments = list(seg.keys())
            segment_sizes = [seg[s]['count'] for s in segments]
            segment_colors = [colors['success'], colors['primary'], colors['accent']]  # Verde, Azul, Naranja
            
            # Gráfico de pie ejecutivo
            wedges, texts, autotexts = ax4.pie(segment_sizes, labels=segments, autopct='%1.1f%%',
                                              colors=segment_colors, startangle=90,
                                              textprops={'fontsize': 11, 'fontweight': 'bold'},
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            ax4.set_title('DISTRIBUCIÓN DE CLIENTES POR RIESGO\n⚠️ Segmentación Predictiva', 
                         fontsize=14, fontweight='bold', color=colors['primary'], pad=20)
        
        # 5. RECOMENDACIONES EJECUTIVAS (bottom span 4 cols)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        recommendations_text = f"""
🎯 RECOMENDACIONES EJECUTIVAS PARA EL BOARD

✅ DECISIÓN RECOMENDADA: {executive_metrics['recommendations']['primary_recommendation']}
📊 JUSTIFICACIÓN FINANCIERA: ROI {executive_metrics['financial_impact']['roi_annual']:.0f}% anual, Payback {executive_metrics['financial_impact']['payback_months']:.1f} meses
🚀 ACCIÓN PRIORITARIA: {executive_metrics['recommendations']['priority_action']}

💰 INVERSIÓN REQUERIDA: ${executive_metrics['financial_impact']['annual_investment_required']:,.0f} anual
📈 RETORNO ESPERADO: ${executive_metrics['financial_impact']['annual_revenue_saved']:,.0f} anual
🎯 BREAK-EVEN: Año {executive_metrics['financial_impact']['break_even_year'] or 'N/A'}

⚠️  VALIDACIÓN CRÍTICA REQUERIDA: {executive_metrics['recommendations']['critical_validation']}
🎲 RIESGO PRINCIPAL: {executive_metrics['risk_assessment']['primary_risk']}
🛡️  MITIGACIÓN: {executive_metrics['risk_assessment']['mitigation']}

🚨 DISCLAIMER: {executive_metrics['disclaimer']}
        """
        
        ax5.text(0.05, 0.95, recommendations_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle="round,pad=0.8", facecolor=colors['background'], 
                         edgecolor=colors['primary'], linewidth=2, alpha=0.9))
        
        # Título principal ejecutivo
        fig.suptitle('TELECOMX - DASHBOARD EJECUTIVO: ANÁLISIS PREDICTIVO DE CHURN\n⚠️ PRESENTACIÓN PARA CEO Y BOARD DE DIRECTORES - DATOS ESTIMADOS INDUSTRIA ⚠️', 
                    fontsize=16, fontweight='bold', color=colors['primary'], y=0.96)
        
        # Guardar dashboard ejecutivo
        os.makedirs('graficos', exist_ok=True)
        viz_file = f'graficos/paso13e_dashboard_ejecutivo_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.5)
        plt.close()
        
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Dashboard ejecutivo creado: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear dashboard ejecutivo: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard ejecutivo: {str(e)}")
        return None

def create_executive_summary_report(executive_metrics, timestamp):
    """Crear resumen ejecutivo de 1 página para CEO/Board"""
    
    fin = executive_metrics.get('financial_impact', {})
    impl = executive_metrics.get('implementation', {})
    overview = executive_metrics.get('project_overview', {})
    
    report = f"""
================================================================================
TELECOMX - RESUMEN EJECUTIVO PARA CEO Y BOARD DE DIRECTORES
================================================================================
Fecha: {timestamp}
Audiencia: CEO, CFO, Board de Directores

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
🚨 DISCLAIMER CRÍTICO: ANÁLISIS BASADO EN DATOS ESTIMADOS INDUSTRIA TELECOM
⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

================================================================================
SITUACIÓN ACTUAL Y OPORTUNIDAD
================================================================================

📊 ESTADO ACTUAL:
• Base de clientes: {overview.get('total_customers', 'N/A'):,}
• Tasa de churn: {overview.get('baseline_churn_rate', 0):.1%} anual
• Pérdida anual por churn: ${fin.get('current_churn_loss', 0):,.0f}

🎯 OPORTUNIDAD IDENTIFICADA:
• Revenue en riesgo anual: ${fin.get('current_churn_loss', 0):,.0f}
• Potencial de retención: ${fin.get('annual_revenue_saved', 0):,.0f}
• Factores críticos identificados: {overview.get('critical_factors_identified', 'N/A')}

================================================================================
PROPUESTA DE INVERSIÓN
================================================================================

💰 REQUERIMIENTOS FINANCIEROS:
• Inversión anual requerida: ${fin.get('annual_investment_required', 0):,.0f}
• Retorno anual esperado: ${fin.get('annual_revenue_saved', 0):,.0f}
• ROI anual proyectado: {fin.get('roi_annual', 0):.1f}%
• Período de payback: {fin.get('payback_months', 0):.1f} meses

📈 PROYECCIÓN 3 AÑOS:
• NPV (valor presente neto): ${fin.get('npv_3_years', 0):,.0f}
• Break-even proyectado: Año {fin.get('break_even_year') or 'N/A'}

================================================================================
PLAN DE IMPLEMENTACIÓN
================================================================================

⏱️ CRONOGRAMA:
• Duración total: {impl.get('implementation_duration', 'N/A')} meses
• Fases planificadas: {impl.get('total_phases', 'N/A')}
• Segmento prioritario: {impl.get('priority_segment', 'N/A').replace('_', ' ')}
• Milestone crítico: Mes {impl.get('critical_milestone_month', 'N/A')} (validación piloto)

🎯 ESTRATEGIA:
• Enfoque por fases con validación continua
• Priorización segmento más rentable identificado
• Implementación gradual con learnings aplicados

================================================================================
ANÁLISIS DE RIESGOS
================================================================================

⚠️ RIESGO PRINCIPAL: {executive_metrics['risk_assessment']['primary_risk']}
📊 Nivel de riesgo: {executive_metrics['risk_assessment']['risk_level']}
🛡️ Mitigación: {executive_metrics['risk_assessment']['mitigation']}

🚨 DEPENDENCIA CRÍTICA: {executive_metrics['risk_assessment']['data_dependency']}

================================================================================
RECOMENDACIÓN EJECUTIVA
================================================================================

✅ DECISIÓN RECOMENDADA: {executive_metrics['recommendations']['primary_recommendation']}

🎯 JUSTIFICACIÓN:
• ROI atractivo: {fin.get('roi_annual', 0):.1f}% anual
• Payback razonable: {fin.get('payback_months', 0):.1f} meses
• NPV positivo: ${fin.get('npv_3_years', 0):,.0f}
• Oportunidad significativa: ${fin.get('current_churn_loss', 0):,.0f} en riesgo

🚀 PRÓXIMOS PASOS INMEDIATOS:
1. {executive_metrics['recommendations']['critical_validation']}
2. Aprobación de presupuesto: ${fin.get('annual_investment_required', 0):,.0f}
3. {executive_metrics['recommendations']['priority_action']}
4. Establecer governance y métricas de seguimiento

================================================================================
CONSIDERACIONES PARA EL BOARD
================================================================================

💡 OPORTUNIDAD ESTRATÉGICA:
• Potencial de mejora significativa en retención de clientes
• ROI atractivo con riesgo controlado
• Diferenciación competitiva en mercado telecom

⚠️ VALIDACIONES REQUERIDAS:
• Confirmar supuestos financieros con datos reales empresa
• Validar capacidad operacional para implementación
• Testear efectividad con piloto controlado antes de scaling

🎲 ALTERNATIVAS:
• OPCIÓN A: Proceder con implementación completa
• OPCIÓN B: Iniciar con piloto extenso (6 meses) para validación
• OPCIÓN C: No proceder (mantener status quo con pérdida actual)

================================================================================
CONCLUSIÓN EJECUTIVA
================================================================================

📊 ANÁLISIS DEMUESTRA VIABILIDAD con datos estimados industria
💰 OPORTUNIDAD FINANCIERA significativa identificada
🎯 RIESGO CONTROLADO con validación adecuada
⚡ ACCIÓN REQUERIDA: Decisión del Board sobre implementación

RECOMENDACIÓN FINAL: PROCEDER CON VALIDACIÓN Y PILOTO

⚠️ RECORDATORIO CRÍTICO: Todos los cálculos financieros basados en benchmarks
estimados industria telecom. Validación con datos reales empresa es esencial
antes de comprometer recursos completos.

================================================================================
FIN DEL RESUMEN EJECUTIVO
================================================================================
"""
    
    return report

def create_kpi_tracking_template(executive_metrics, timestamp):
    """Crear template de seguimiento de KPIs para monitoreo ejecutivo"""
    try:
        logging.info("Creando template de seguimiento de KPIs...")
        
        excel_file = f'excel/paso13e_kpi_tracking_template_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # 1. Dashboard KPIs Ejecutivos
            kpi_data = [
                ['KPI EJECUTIVO', 'OBJETIVO', 'ACTUAL', 'ESTADO', 'RESPONSABLE'],
                ['=== MÉTRICAS FINANCIERAS ===', '', '', '', ''],
                ['ROI Anual (%)', f"{executive_metrics['financial_impact']['roi_annual']:.0f}%", 'A medir', 'Pendiente', 'CFO'],
                ['Revenue Salvado Mensual', f"${executive_metrics['financial_impact']['annual_revenue_saved']/12:,.0f}", 'A medir', 'Pendiente', 'Gerente Retención'],
                ['Costo Mensual Implementación', f"${executive_metrics['financial_impact']['annual_investment_required']/12:,.0f}", 'A medir', 'Pendiente', 'Controller'],
                ['Payback Acumulado (meses)', f"{executive_metrics['financial_impact']['payback_months']:.1f}", 'A medir', 'Pendiente', 'CFO'],
                ['', '', '', '', ''],
                ['=== MÉTRICAS OPERACIONALES ===', '', '', '', ''],
                ['Save Rate Mensual (%)', '25%', 'A medir', 'Pendiente', 'Gerente Retención'],
                ['Clientes Contactados', '200', 'A medir', 'Pendiente', 'Gerente Retención'],
                ['NPS Post-Intervención', '7.5', 'A medir', 'Pendiente', 'Gerente CX'],
                ['Automatización Implementada (%)', '70%', 'A medir', 'Pendiente', 'Gerente Tecnología'],
                ['', '', '', '', ''],
                ['=== MILESTONES CRÍTICOS ===', '', '', '', ''],
                ['Setup Completado (%)', '100%', 'A medir', 'Pendiente', 'PMO'],
                ['Piloto Validado', 'Mes 5', 'A medir', 'Pendiente', 'Director Comercial'],
                ['Scaling Medio Riesgo (%)', '100%', 'A medir', 'Pendiente', 'Gerente Retención'],
                ['Implementación Completa', 'Mes 12', 'A medir', 'Pendiente', 'CEO'],
                ['', '', '', '', ''],
                ['⚠️ NOTA', 'OBJETIVOS BASADOS EN ESTIMACIONES', '', '', ''],
                ['Validación Requerida', 'Confirmar con datos reales empresa', '', '', 'Board']
            ]
            
            kpi_df = pd.DataFrame(kpi_data)
            kpi_df.to_excel(writer, sheet_name='KPIs Ejecutivos', index=False, header=False)
            
            # 2. Seguimiento Financiero Mensual
            monthly_tracking = []
            for month in range(1, 13):
                monthly_tracking.append({
                    'Mes': month,
                    'Inversión_Acumulada': 'A medir',
                    'Revenue_Salvado_Mes': 'A medir',
                    'Revenue_Salvado_Acumulado': 'A medir',
                    'ROI_Acumulado': 'A medir',
                    'Clientes_Salvados_Mes': 'A medir',
                    'Save_Rate_Mes': 'A medir',
                    'Estado_Fase': 'A medir',
                    'Comentarios': 'A medir'
                })
            
            monthly_df = pd.DataFrame(monthly_tracking)
            monthly_df.to_excel(writer, sheet_name='Tracking Mensual', index=False)
            
            # 3. Alertas y Escalaciones
            alerts_data = [
                ['MÉTRICA', 'UMBRAL CRÍTICO', 'ACCIÓN REQUERIDA', 'ESCALACIÓN'],
                ['Save Rate < 20%', 'CRÍTICO', 'Revisión estrategia inmediata', 'CEO + Board'],
                ['ROI < 50%', 'ALTO', 'Análisis causas + plan acción', 'CFO + CEO'],
                ['Budget > 110% plan', 'MEDIO', 'Control gastos + justificación', 'CFO'],
                ['NPS < 6.0', 'MEDIO', 'Mejora experiencia cliente', 'Gerente CX'],
                ['Atraso > 2 semanas', 'ALTO', 'Aceleración + recursos', 'PMO + CEO'],
                ['', '', '', ''],
                ['REUNIONES DE SEGUIMIENTO', '', '', ''],
                ['Board Review', 'Mensual', 'Presentación KPIs', 'CEO'],
                ['Steering Committee', 'Semanal', 'Revisión operacional', 'PMO'],
                ['Business Review', 'Trimestral', 'Análisis ROI + ajustes', 'CFO + CEO']
            ]
            
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df.to_excel(writer, sheet_name='Alertas y Escalaciones', index=False, header=False)
        
        logging.info(f"Template de KPIs creado: {excel_file}")
        return excel_file
        
    except Exception as e:
        logging.error(f"Error creando template KPIs: {str(e)}")
        return None

def save_executive_outputs(executive_metrics, timestamp):
    """Guardar outputs ejecutivos consolidados"""
    try:
        logging.info("Guardando outputs ejecutivos...")
        
        # 1. JSON con métricas ejecutivas
        json_file = f'informes/paso13e_outputs_ejecutivos_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(executive_metrics, f, indent=2, ensure_ascii=False, default=str)
        
        # 2. Resumen ejecutivo de 1 página
        txt_file = f'informes/paso13e_resumen_ejecutivo_{timestamp}.txt'
        report_content = create_executive_summary_report(executive_metrics, timestamp)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"JSON outputs ejecutivos guardado: {json_file}")
        logging.info(f"Resumen ejecutivo guardado: {txt_file}")
        
        return {
            'json_file': json_file,
            'txt_file': txt_file
        }
        
    except Exception as e:
        logging.error(f"Error guardando outputs ejecutivos: {str(e)}")
        raise

def main():
    """Función principal del Paso 13E - Outputs Ejecutivos"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13E: OUTPUTS EJECUTIVOS PARA CEO Y BOARD")
        logger.info("="*80)
        logger.warning("⚠️  CONSOLIDANDO ANÁLISIS BASADO EN DATOS ESTIMADOS")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar resultados consolidados de pasos anteriores
        logger.info("="*50)
        logger.info("CARGANDO RESULTADOS CONSOLIDADOS (PASOS 13A-13D)")
        consolidated_results = load_consolidated_results()
        
        # 3. Extraer métricas ejecutivas clave
        logger.info("="*50)
        logger.info("EXTRAYENDO MÉTRICAS EJECUTIVAS CLAVE")
        executive_metrics = extract_executive_metrics(consolidated_results)
        
        # 4. Crear dashboard ejecutivo
        logger.info("="*50)
        logger.info("CREANDO DASHBOARD EJECUTIVO PARA CEO/BOARD")
        dashboard_file = create_executive_dashboard(executive_metrics, timestamp)
        
        # 5. Crear template de seguimiento KPIs
        logger.info("="*50)
        logger.info("CREANDO TEMPLATE DE SEGUIMIENTO DE KPIS")
        kpi_template_file = create_kpi_tracking_template(executive_metrics, timestamp)
        
        # 6. Guardar outputs ejecutivos
        logger.info("="*50)
        logger.info("GUARDANDO OUTPUTS EJECUTIVOS")
        output_files = save_executive_outputs(executive_metrics, timestamp)
        
        # 7. Resumen final
        logger.info("="*80)
        logger.info("PASO 13E COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar resultados principales
        if 'financial_impact' in executive_metrics:
            fin = executive_metrics['financial_impact']
            
            logger.warning("⚠️⚠️⚠️ OUTPUTS BASADOS EN DATOS ESTIMADOS ⚠️⚠️⚠️")
            logger.info("")
            
            logger.info("💰 MÉTRICAS FINANCIERAS EJECUTIVAS:")
            logger.info(f"  • ROI anual proyectado: {fin['roi_annual']:.1f}%")
            logger.info(f"  • Inversión requerida: ${fin['annual_investment_required']:,.0f}")
            logger.info(f"  • Revenue salvado: ${fin['annual_revenue_saved']:,.0f}")
            logger.info(f"  • Payback period: {fin['payback_months']:.1f} meses")
            logger.info(f"  • NPV 3 años: ${fin['npv_3_years']:,.0f}")
            logger.info("")
        
        logger.info("🎯 RECOMENDACIÓN EJECUTIVA:")
        logger.info(f"  • Decisión: {executive_metrics['recommendations']['primary_recommendation']}")
        logger.info(f"  • Acción prioritaria: {executive_metrics['recommendations']['priority_action']}")
        logger.info(f"  • Validación crítica: {executive_metrics['recommendations']['critical_validation']}")
        logger.info("")
        
        logger.info("📊 AUDIENCIA Y PROPÓSITO:")
        logger.info("  • Target: CEO, CFO, Board de Directores")
        logger.info("  • Estilo: Corporativo, profesional, high-level")
        logger.info("  • Enfoque: Métricas clave y decisiones estratégicas")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • Dashboard ejecutivo: {dashboard_file}")
        logger.info(f"  • Template KPIs: {kpi_template_file}")
        logger.info(f"  • JSON métricas: {output_files['json_file']}")
        logger.info(f"  • Resumen ejecutivo: {output_files['txt_file']}")
        logger.info("")
        
        logger.warning("⚠️ DISCLAIMER CRÍTICO:")
        logger.warning("Todos los outputs están basados en benchmarks estimados")
        logger.warning("industria telecom para demostración metodológica")
        logger.warning("VALIDACIÓN CON DATOS REALES es esencial antes de decisiones")
        logger.info("")
        
        logger.info("📋 PRÓXIMO Y ÚLTIMO SCRIPT:")
        logger.info("  • paso13f: Informe final estratégico consolidado")
        logger.info("  • Consolidación completa de todo el proyecto")
        logger.info("="*80)
        
        return {
            'executive_metrics': executive_metrics,
            'dashboard_file': dashboard_file,
            'kpi_template_file': kpi_template_file,
            'output_files': output_files
        }
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()