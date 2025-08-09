"""
================================================================================
TELECOMX - PASO 13D: ROADMAP DETALLADO DE IMPLEMENTACIÓN
================================================================================
Descripción: Desarrollo de roadmap detallado de implementación por fases basado
             en business case y segmentación estratégica. Incluye cronograma,
             milestones, recursos, KPIs y plan de contingencia.
             
⚠️  IMPORTANTE: Este roadmap utiliza datos del business case que se basa en 
    BENCHMARKS ESTIMADOS de la industria telecom para fines de SIMULACIÓN.

Inputs: 
- Business case completo del Paso 13C
- Segmentación estratégica del Paso 13B
- Factores críticos del Paso 13A

Outputs:
- Roadmap de implementación de 12 meses
- Cronograma detallado por fases
- KPIs de seguimiento mensual
- Plan de recursos y presupuesto
- Análisis de riesgos y contingencias
- Dashboard de seguimiento ejecutivo

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
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso13d_roadmap_detallado.log', mode='a', encoding='utf-8'),
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

def load_business_case_data():
    """Cargar datos del business case del Paso 13C"""
    try:
        logging.info("Cargando datos del business case del Paso 13C...")
        
        json_file = find_latest_file('informes', 'paso13c_business_case_completo_*.json')
        
        with open(json_file, 'r', encoding='utf-8') as f:
            business_case_data = json.load(f)
        
        current_state = business_case_data['current_state']
        intervention_scenarios = business_case_data['intervention_scenarios']
        projections = business_case_data['projections_3_years']
        benchmarks = business_case_data['benchmarks_used']
        
        logging.info(f"Business case cargado exitosamente")
        logging.info(f"ROI anual proyectado: {intervention_scenarios['consolidated']['overall_roi_annual']:.1f}%")
        
        return {
            'current_state': current_state,
            'intervention_scenarios': intervention_scenarios,
            'projections': projections,
            'benchmarks': benchmarks,
            'source_file': json_file
        }
        
    except Exception as e:
        logging.error(f"Error cargando business case: {str(e)}")
        raise

def define_implementation_phases():
    """Definir fases de implementación basadas en conclusiones del business case"""
    
    logging.info("Definiendo fases de implementación estratégica...")
    
    # Basado en conclusiones: Medio Riesgo es prioridad máxima
    phases = {
        'Fase_1_Preparacion': {
            'name': 'Preparación y Setup',
            'duration_months': 2,
            'start_month': 1,
            'priority': 'Crítica',
            'segments_focus': ['Todos'],
            'key_activities': [
                'Setup tecnológico y sistemas',
                'Contratación y entrenamiento de equipos',
                'Desarrollo de procesos y workflows',
                'Configuración de dashboards y métricas',
                'Validación de datos y segmentación'
            ],
            'deliverables': [
                'Plataforma de retención operativa',
                'Equipos entrenados y certificados',
                'Procesos documentados y aprobados',
                'Dashboard de KPIs en tiempo real',
                'Base de datos de clientes segmentada'
            ],
            'budget_percentage': 15,
            'success_criteria': [
                'Sistemas operativos al 100%',
                'Equipos entrenados (95% certificación)',
                'Procesos validados con QA',
                'Dashboard funcional con datos en tiempo real'
            ]
        },
        'Fase_2_Piloto_Medio_Riesgo': {
            'name': 'Piloto Medio Riesgo (Prioridad Máxima)',
            'duration_months': 3,
            'start_month': 3,
            'priority': 'Crítica',
            'segments_focus': ['Medio_Riesgo'],
            'key_activities': [
                'Implementación piloto 30% clientes Medio Riesgo',
                'Campañas proactivas de retención',
                'Seguimiento intensivo y optimización',
                'Validación de save rates y costos',
                'Ajuste de estrategias basado en resultados'
            ],
            'deliverables': [
                'Resultados piloto validados',
                'Save rate real vs estimado',
                'Procesos optimizados',
                'ROI real calculado',
                'Plan de escalamiento aprobado'
            ],
            'budget_percentage': 25,
            'success_criteria': [
                'Save rate ≥ 25% (objetivo conservador)',
                'Payback ≤ 3 meses',
                'NPS post-intervención ≥ 7/10',
                'Costos dentro de presupuesto (+/- 10%)'
            ]
        },
        'Fase_3_Scaling_Medio_Riesgo': {
            'name': 'Scaling Completo Medio Riesgo',
            'duration_months': 2,
            'start_month': 6,
            'priority': 'Alta',
            'segments_focus': ['Medio_Riesgo'],
            'key_activities': [
                'Rollout completo segmento Medio Riesgo',
                'Automatización de procesos repetitivos',
                'Optimización continua basada en ML',
                'Entrenamiento avanzado de equipos',
                'Preparación para siguientes segmentos'
            ],
            'deliverables': [
                '100% cobertura Medio Riesgo',
                'Procesos automatizados operativos',
                'Modelos de ML optimizados',
                'Equipos especializados por segmento',
                'Baseline establecido para comparación'
            ],
            'budget_percentage': 30,
            'success_criteria': [
                'Cobertura 100% Medio Riesgo',
                'Automatización ≥ 70% procesos',
                'ROI sostenible mes a mes',
                'Preparación para siguiente fase'
            ]
        },
        'Fase_4_Alto_Riesgo_Selectivo': {
            'name': 'Alto Riesgo Selectivo',
            'duration_months': 3,
            'start_month': 8,
            'priority': 'Media-Alta',
            'segments_focus': ['Alto_Riesgo'],
            'key_activities': [
                'Implementación selectiva Alto Riesgo',
                'Enfoque en clientes de alto valor',
                'Intervenciones intensivas personalizadas',
                'Seguimiento especializado post-save',
                'Análisis de casos exitosos vs fallidos'
            ],
            'deliverables': [
                'Estrategia Alto Riesgo refinada',
                'Clientes de alto valor retenidos',
                'Procesos intensivos documentados',
                'Análisis de efectividad por perfil',
                'Recomendaciones de optimización'
            ],
            'budget_percentage': 20,
            'success_criteria': [
                'Save rate ≥ 15% Alto Riesgo',
                'Enfoque en top 20% clientes por valor',
                'Payback ≤ 6 meses',
                'Learnings documentados para mejora'
            ]
        },
        'Fase_5_Bajo_Riesgo_Mantenimiento': {
            'name': 'Bajo Riesgo y Optimización',
            'duration_months': 2,
            'start_month': 11,
            'priority': 'Media',
            'segments_focus': ['Bajo_Riesgo', 'Todos'],
            'key_activities': [
                'Implementación automatizada Bajo Riesgo',
                'Programas de fidelización y upselling',
                'Optimización global de todos los segmentos',
                'Análisis de resultados consolidados',
                'Preparación para año 2'
            ],
            'deliverables': [
                'Cobertura completa todos los segmentos',
                'Programas de fidelización activos',
                'Análisis anual consolidado',
                'Roadmap año 2 preparado',
                'Procesos optimizados y documentados'
            ],
            'budget_percentage': 10,
            'success_criteria': [
                'Cobertura 100% todos los segmentos',
                'Automatización ≥ 85% Bajo Riesgo',
                'ROI anual consolidado ≥ objetivo',
                'Plan año 2 aprobado'
            ]
        }
    }
    
    logging.info(f"Definidas {len(phases)} fases de implementación")
    logging.info("Prioridad máxima: Medio Riesgo (Fases 2 y 3)")
    
    return phases

def create_monthly_timeline(phases):
    """Crear timeline mensual detallado"""
    try:
        logging.info("Creando timeline mensual detallado...")
        
        start_date = datetime(2025, 1, 1)
        timeline = []
        
        for phase_id, phase in phases.items():
            phase_start = start_date + timedelta(days=(phase['start_month'] - 1) * 30)
            
            for month_offset in range(phase['duration_months']):
                month_date = phase_start + timedelta(days=month_offset * 30)
                month_number = phase['start_month'] + month_offset
                
                # Definir actividades específicas por mes dentro de cada fase
                activities = get_monthly_activities(phase, month_offset)
                kpis = get_monthly_kpis(phase, month_offset)
                resources = get_monthly_resources(phase, month_offset)
                
                timeline.append({
                    'month': month_number,
                    'date': month_date,
                    'phase_id': phase_id,
                    'phase_name': phase['name'],
                    'phase_month': month_offset + 1,
                    'priority': phase['priority'],
                    'segments_focus': phase['segments_focus'],
                    'activities': activities,
                    'kpis': kpis,
                    'resources': resources,
                    'budget_percentage': phase['budget_percentage'] / phase['duration_months'],
                    'success_criteria': phase['success_criteria']
                })
        
        logging.info(f"Timeline creado: {len(timeline)} meses detallados")
        return timeline
        
    except Exception as e:
        logging.error(f"Error creando timeline: {str(e)}")
        raise

def get_monthly_activities(phase, month_offset):
    """Obtener actividades específicas por mes dentro de cada fase"""
    
    phase_activities = {
        'Fase_1_Preparacion': {
            0: ['Setup inicial sistemas', 'Contratación equipos', 'Diseño procesos'],
            1: ['Entrenamiento intensivo', 'Configuración dashboards', 'Testing sistemas']
        },
        'Fase_2_Piloto_Medio_Riesgo': {
            0: ['Selección muestra piloto', 'Lanzamiento campañas', 'Setup seguimiento'],
            1: ['Optimización campañas', 'Análisis resultados intermedios', 'Ajustes procesos'],
            2: ['Validación resultados', 'Documentación learnings', 'Preparación scaling']
        },
        'Fase_3_Scaling_Medio_Riesgo': {
            0: ['Rollout completo', 'Automatización procesos', 'Entrenamiento avanzado'],
            1: ['Optimización ML', 'Monitoreo intensivo', 'Preparación siguiente fase']
        },
        'Fase_4_Alto_Riesgo_Selectivo': {
            0: ['Selección clientes alto valor', 'Diseño intervenciones intensivas', 'Setup seguimiento especializado'],
            1: ['Ejecución campañas intensivas', 'Seguimiento personalizado', 'Análisis casos'],
            2: ['Optimización estrategias', 'Documentación mejores prácticas', 'Scaling selectivo']
        },
        'Fase_5_Bajo_Riesgo_Mantenimiento': {
            0: ['Implementación automatizada', 'Programas fidelización', 'Análisis consolidado'],
            1: ['Optimización global', 'Preparación año 2', 'Documentación final']
        }
    }
    
    phase_key = None
    for key in phase_activities.keys():
        if key in phase['name'].replace(' ', '_').replace('(', '').replace(')', ''):
            phase_key = key
            break
    
    if phase_key and month_offset in phase_activities[phase_key]:
        return phase_activities[phase_key][month_offset]
    else:
        return ['Actividades continuas de la fase', 'Monitoreo y optimización', 'Reporting mensual']

def get_monthly_kpis(phase, month_offset):
    """Obtener KPIs específicos por mes y fase"""
    
    base_kpis = [
        'Clientes contactados',
        'Save rate mensual',
        'Costo por cliente salvado',
        'NPS post-intervención',
        'ROI incremental'
    ]
    
    phase_specific_kpis = {
        'Preparación': ['% Setup completado', 'Equipos entrenados', 'Sistemas operativos'],
        'Piloto': ['% Muestra completada', 'Learnings documentados', 'Ajustes implementados'],
        'Scaling': ['% Cobertura segmento', 'Automatización implementada', 'Procesos optimizados'],
        'Alto_Riesgo': ['Clientes alto valor contactados', 'Intervenciones intensivas', 'Casos éxito'],
        'Bajo_Riesgo': ['Automatización implementada', 'Programas fidelización', 'Análisis consolidado']
    }
    
    # Determinar KPIs específicos basado en el nombre de la fase
    specific_kpis = base_kpis.copy()
    for key, kpis in phase_specific_kpis.items():
        if key.lower() in phase['name'].lower():
            specific_kpis.extend(kpis)
            break
    
    return specific_kpis[:7]  # Limitar a 7 KPIs por mes

def get_monthly_resources(phase, month_offset):
    """Obtener recursos necesarios por mes"""
    
    base_resources = {
        'team_size': 8,
        'budget_monthly': 50000,
        'technology_hours': 160,
        'training_hours': 40
    }
    
    # Ajustar recursos según la fase
    phase_multipliers = {
        'Preparación': {'team_size': 1.5, 'budget_monthly': 1.2, 'technology_hours': 2.0, 'training_hours': 3.0},
        'Piloto': {'team_size': 1.0, 'budget_monthly': 1.0, 'technology_hours': 1.0, 'training_hours': 1.5},
        'Scaling': {'team_size': 1.3, 'budget_monthly': 1.4, 'technology_hours': 1.2, 'training_hours': 1.0},
        'Alto_Riesgo': {'team_size': 1.1, 'budget_monthly': 1.1, 'technology_hours': 0.8, 'training_hours': 2.0},
        'Bajo_Riesgo': {'team_size': 0.8, 'budget_monthly': 0.7, 'technology_hours': 1.5, 'training_hours': 0.5}
    }
    
    # Encontrar multiplicador apropiado
    multiplier = {'team_size': 1.0, 'budget_monthly': 1.0, 'technology_hours': 1.0, 'training_hours': 1.0}
    for key, mult in phase_multipliers.items():
        if key.lower() in phase['name'].lower():
            multiplier = mult
            break
    
    # Aplicar multiplicadores
    resources = {}
    for resource, base_value in base_resources.items():
        resources[resource] = int(base_value * multiplier[resource])
    
    return resources
def create_roadmap_visualization(timeline, phases, business_case, timestamp):
    """Crear visualización completa del roadmap"""
    try:
        logging.info("Creando visualización del roadmap...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 20))
        
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25, top=0.91, bottom=0.07, left=0.05, right=0.97)
        
        # Colores por fase
        phase_colors = {
            'Fase_1_Preparacion': '#8B4513',
            'Fase_2_Piloto_Medio_Riesgo': '#FFD700', 
            'Fase_3_Scaling_Medio_Riesgo': '#32CD32',
            'Fase_4_Alto_Riesgo_Selectivo': '#FF6347',
            'Fase_5_Bajo_Riesgo_Mantenimiento': '#87CEEB'
        }
        
        # 1. Cronograma Gantt de Fases (span 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        y_pos = 0
        phase_labels = []
        for phase_id, phase in phases.items():
            start_month = phase['start_month']
            duration = phase['duration_months']
            color = phase_colors.get(phase_id, '#87CEEB')
            
            # Dibujar barra de fase
            rect = Rectangle((start_month, y_pos), duration, 0.8, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            
            # Agregar texto de duración
            ax1.text(start_month + duration/2, y_pos + 0.4, f'{duration}m', 
                    ha='center', va='center', fontweight='bold', fontsize=10)
            
            phase_labels.append(phase['name'][:30])  # Truncar nombres largos
            y_pos += 1
        
        ax1.set_xlim(0, 13)
        ax1.set_ylim(-0.5, len(phases) - 0.5)
        ax1.set_xlabel('Meses', fontweight='bold', fontsize=12)
        ax1.set_title('Cronograma de Implementación por Fases\n⚠️ Basado en Business Case con Datos Estimados', 
                     fontweight='bold', fontsize=12, pad=15)
        ax1.set_yticks(range(len(phases)))
        ax1.set_yticklabels(phase_labels, fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xticks(range(1, 13))
        
        # 2. Budget por Fase
        ax2 = fig.add_subplot(gs[0, 2])
        phase_names = ['Preparación', 'Piloto', 'Scaling', 'Alto Riesgo', 'Finalización']
        budget_percentages = [phase['budget_percentage'] for phase in phases.values()]
        colors = [phase_colors[phase_id] for phase_id in phases.keys()]
        
        wedges, texts, autotexts = ax2.pie(budget_percentages, labels=phase_names, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, textprops={'fontsize': 9})
        ax2.set_title('Distribución de Budget por Fase\n⚠️ Estimación Benchmark', 
                     fontweight='bold', fontsize=12, pad=15)
        
        # 3. KPIs Objetivo por Mes (span 3 columns)
        ax3 = fig.add_subplot(gs[1, :])
        
        months = [t['month'] for t in timeline]
        # Simular datos de KPIs objetivo basados en business case
        save_rates = []
        customers_saved = []
        
        for i, month_data in enumerate(timeline):
            # Simular progreso basado en fase
            if 'Piloto' in month_data['phase_name']:
                save_rate = 25 + i * 2  # Mejora gradual
                monthly_customers = 30 + i * 5
            elif 'Scaling' in month_data['phase_name']:
                save_rate = 30 + (i-5) * 1
                monthly_customers = 80 + (i-5) * 10
            elif 'Alto_Riesgo' in month_data['phase_name']:
                save_rate = 15 + (i-7) * 1
                monthly_customers = 25 + (i-7) * 3
            else:
                save_rate = max(10, 15 + i)
                monthly_customers = max(10, 20 + i * 2)
            
            save_rates.append(min(save_rate, 45))  # Cap al 45%
            customers_saved.append(monthly_customers)
        
        # Graficar KPIs
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(months, save_rates, marker='o', linewidth=3, markersize=6, 
                        label='Save Rate (%)', color='#2E8B57')
        line2 = ax3_twin.plot(months, customers_saved, marker='s', linewidth=3, markersize=6, 
                             label='Clientes Salvados', color='#4682B4')
        
        ax3.set_xlabel('Meses', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Save Rate (%)', fontweight='bold', fontsize=12, color='#2E8B57')
        ax3_twin.set_ylabel('Clientes Salvados', fontweight='bold', fontsize=12, color='#4682B4')
        ax3.set_title('Evolución de KPIs Objetivo por Mes - ⚠️ PROYECCIÓN CON DATOS ESTIMADOS', 
                     fontweight='bold', fontsize=12, pad=15)
        
        # Combinar leyendas
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.5, 12.5)
        
        # 4. Recursos por Mes
        ax4 = fig.add_subplot(gs[2, 0])
        team_sizes = [t['resources']['team_size'] for t in timeline]
        
        bars = ax4.bar(months, team_sizes, color='#FF6347', alpha=0.7, edgecolor='black', linewidth=1)
        ax4.set_ylabel('Tamaño del Equipo', fontweight='bold', fontsize=11)
        ax4.set_title('Recursos: Tamaño del Equipo\n⚠️ Estimación Benchmark', fontweight='bold', fontsize=12, pad=15)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xlabel('Meses', fontweight='bold', fontsize=11)
        ax4.set_xticks(months)
        ax4.set_xticklabels([f'M{m}' for m in months], fontsize=9)
        
        for bar, size in zip(bars, team_sizes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(size), ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 5. Budget Mensual
        ax5 = fig.add_subplot(gs[2, 1])
        monthly_budgets = [t['resources']['budget_monthly']/1000 for t in timeline]  # En miles
        
        bars = ax5.bar(months, monthly_budgets, color='#32CD32', alpha=0.7, edgecolor='black', linewidth=1)
        ax5.set_ylabel('Budget Mensual (Miles USD)', fontweight='bold', fontsize=11)
        ax5.set_title('Budget Mensual por Fase\n⚠️ Estimación Industria', fontweight='bold', fontsize=12, pad=15)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_xlabel('Meses', fontweight='bold', fontsize=11)
        ax5.set_xticks(months)
        ax5.set_xticklabels([f'M{m}' for m in months], fontsize=9)
        
        # 6. Milestones Críticos
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        milestones_text = """
🎯 MILESTONES CRÍTICOS

📅 MES 2: Setup Completo
• Sistemas operativos 100%
• Equipos certificados

🚀 MES 5: Piloto Validado  
• Save rate ≥ 25%
• ROI positivo

📈 MES 8: Scaling Completo
• 100% cobertura Medio Riesgo
• Automatización ≥ 70%

🎯 MES 11: Alto Riesgo Operativo
• Estrategia selectiva activa
• Clientes alto valor

✅ MES 12: Implementación Total
• Cobertura 100% segmentos
• ROI objetivo alcanzado
        """
        
        ax6.text(0.05, 0.95, milestones_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))
        
        # 7. Resumen de Fases (span 3 columns)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        phases_summary = """
📊 RESUMEN DE FASES DE IMPLEMENTACIÓN - ⚠️ BASADO EN ESTIMACIONES INDUSTRIA TELECOM

🔧 FASE 1 (Meses 1-2): PREPARACIÓN Y SETUP
• Budget: 15% | Prioridad: Crítica | Foco: Setup tecnológico y entrenamiento equipos

🎯 FASE 2 (Meses 3-5): PILOTO MEDIO RIESGO (PRIORIDAD MÁXIMA)
• Budget: 25% | Prioridad: Crítica | Foco: Validación estrategia en segmento más rentable

🚀 FASE 3 (Meses 6-7): SCALING COMPLETO MEDIO RIESGO  
• Budget: 30% | Prioridad: Alta | Foco: Rollout completo del segmento prioritario

🔴 FASE 4 (Meses 8-10): ALTO RIESGO SELECTIVO
• Budget: 20% | Prioridad: Media-Alta | Foco: Casos complejos y clientes alto valor

🟢 FASE 5 (Meses 11-12): BAJO RIESGO Y OPTIMIZACIÓN
• Budget: 10% | Prioridad: Media | Foco: Cobertura completa y preparación año 2

⚠️ NOTA: Cronograma y recursos basados en benchmarks estimados industria telecom para simulación
        """
        
        ax7.text(0.05, 0.95, phases_summary, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        
        # Título principal
        fig.suptitle('TelecomX - Roadmap Detallado de Implementación (12 Meses)\n⚠️ CRONOGRAMA BASADO EN BUSINESS CASE CON DATOS ESTIMADOS ⚠️', 
                    fontsize=18, fontweight='bold', y=0.99, color='#2E4057')
        
        # Guardar visualización
        os.makedirs('graficos', exist_ok=True)
        viz_file = f'graficos/paso13d_roadmap_detallado_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Visualización roadmap guardada: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear visualización: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando visualización roadmap: {str(e)}")
        return None

def create_excel_roadmap_dashboard(timeline, phases, business_case, timestamp):
    """Crear dashboard Excel del roadmap"""
    try:
        logging.info("Creando dashboard Excel del roadmap...")
        
        excel_file = f'excel/paso13d_roadmap_dashboard_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # 1. Hoja: Timeline Mensual
            timeline_data = []
            for month_data in timeline:
                timeline_data.append({
                    'Mes': month_data['month'],
                    'Fecha': month_data['date'].strftime('%Y-%m'),
                    'Fase': month_data['phase_name'],
                    'Prioridad': month_data['priority'],
                    'Segmentos_Focus': ', '.join(month_data['segments_focus']),
                    'Actividad_Principal_1': month_data['activities'][0] if month_data['activities'] else '',
                    'Actividad_Principal_2': month_data['activities'][1] if len(month_data['activities']) > 1 else '',
                    'KPI_Principal_1': month_data['kpis'][0] if month_data['kpis'] else '',
                    'KPI_Principal_2': month_data['kpis'][1] if len(month_data['kpis']) > 1 else '',
                    'Equipo_Size': month_data['resources']['team_size'],
                    'Budget_Mensual': f"${month_data['resources']['budget_monthly']:,.2f}",
                    'Horas_Tecnologia': month_data['resources']['technology_hours'],
                    'Horas_Training': month_data['resources']['training_hours']
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df.to_excel(writer, sheet_name='Timeline Mensual', index=False)
            
            # 2. Hoja: Fases Detalladas
            phases_data = []
            for phase_id, phase in phases.items():
                phases_data.append({
                    'Fase_ID': phase_id,
                    'Nombre': phase['name'],
                    'Duración_Meses': phase['duration_months'],
                    'Mes_Inicio': phase['start_month'],
                    'Prioridad': phase['priority'],
                    'Segmentos_Focus': ', '.join(phase['segments_focus']),
                    'Budget_Porcentaje': f"{phase['budget_percentage']}%",
                    'Actividad_Key_1': phase['key_activities'][0] if phase['key_activities'] else '',
                    'Actividad_Key_2': phase['key_activities'][1] if len(phase['key_activities']) > 1 else '',
                    'Deliverable_1': phase['deliverables'][0] if phase['deliverables'] else '',
                    'Deliverable_2': phase['deliverables'][1] if len(phase['deliverables']) > 1 else '',
                    'Criterio_Exito_1': phase['success_criteria'][0] if phase['success_criteria'] else '',
                    'Criterio_Exito_2': phase['success_criteria'][1] if len(phase['success_criteria']) > 1 else ''
                })
            
            phases_df = pd.DataFrame(phases_data)
            phases_df.to_excel(writer, sheet_name='Fases Detalladas', index=False)
            
            # 3. Hoja: Disclaimer y Contexto
            disclaimer_data = [
                ['CONTEXTO', 'DETALLE'],
                ['⚠️ DISCLAIMER PRINCIPAL', 'Este roadmap está basado en business case con DATOS ESTIMADOS'],
                ['Fuente Datos', 'Benchmarks estándar industria telecom'],
                ['Propósito', 'Ejercicio de simulación y demostración metodológica'],
                ['Para Implementación Real', 'Reemplazar estimaciones con datos específicos empresa'],
                ['', ''],
                ['=== SUPUESTOS CLAVE ===', ''],
                ['Save Rates', 'Alto: 20%, Medio: 35%, Bajo: 15%'],
                ['Duración Implementación', '12 meses para cobertura completa'],
                ['Budget Total Estimado', 'Basado en benchmarks industria'],
                ['Recursos Humanos', '6-12 personas según fase'],
                ['', ''],
                ['=== VALIDACIONES REQUERIDAS ===', ''],
                ['Datos Financieros', 'Confirmar budget y costos operacionales reales'],
                ['Save Rates', 'Validar con piloto controlado antes de scaling'],
                ['Recursos', 'Confirmar disponibilidad equipos y presupuesto'],
                ['Timeline', 'Ajustar según capacidad real organización']
            ]
            
            disclaimer_df = pd.DataFrame(disclaimer_data)
            disclaimer_df.to_excel(writer, sheet_name='Disclaimer y Contexto', index=False, header=False)
        
        logging.info(f"Dashboard Excel roadmap creado: {excel_file}")
        return excel_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard Excel roadmap: {str(e)}")
        return None

def generate_roadmap_report(timeline, phases, business_case, timestamp):
    """Generar informe ejecutivo del roadmap"""
    
    total_budget = sum([t['resources']['budget_monthly'] for t in timeline])
    roi_target = business_case['intervention_scenarios']['consolidated']['overall_roi_annual']
    
    report = f"""
================================================================================
TELECOMX - PASO 13D: ROADMAP DETALLADO DE IMPLEMENTACIÓN
================================================================================
Fecha: {timestamp}
Script: paso13d_Roadmap_Detallado.py

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
                            🚨 DISCLAIMER IMPORTANTE 🚨
================================================================================
⚠️  ESTE ROADMAP ESTÁ BASADO EN BUSINESS CASE QUE UTILIZA DATOS ESTIMADOS 
    DE BENCHMARKS DE LA INDUSTRIA TELECOM PARA FINES DE SIMULACIÓN.

📅 CRONOGRAMA BASADO EN ESTIMACIONES INDUSTRIA TELECOM
🎯 PROPÓSITO: Demostrar metodología de roadmap de implementación
💡 PARA IMPLEMENTACIÓN REAL: Validar supuestos con datos reales empresa

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

================================================================================
RESUMEN EJECUTIVO DEL ROADMAP
================================================================================

🎯 OBJETIVO: Implementar estrategias de retención segmentadas en 12 meses
📊 MÉTRICAS CLAVE:
• Duración total: 12 meses | Fases: {len(phases)}
• Budget total estimado: ${total_budget:,.2f}
• ROI objetivo: {roi_target:.1f}% | Cobertura final: 100% segmentos

🚀 ESTRATEGIA: Priorización Medio Riesgo (conclusión business case)

================================================================================
CRONOGRAMA DE FASES
================================================================================
"""

    for phase_id, phase in phases.items():
        report += f"""
📋 {phase['name'].upper()}:
• Duración: {phase['duration_months']} meses (Mes {phase['start_month']})
• Prioridad: {phase['priority']} | Budget: {phase['budget_percentage']}%
• Segmentos foco: {', '.join(phase['segments_focus'])}
• Actividades clave: {', '.join(phase['key_activities'][:2])}
• Criterios éxito: {', '.join(phase['success_criteria'][:2])}
"""

    report += f"""

================================================================================
TIMELINE MENSUAL RESUMIDO
================================================================================
"""

    for month_data in timeline:
        report += f"""
MES {month_data['month']} - {month_data['phase_name'][:20]}:
• Actividades: {', '.join(month_data['activities'][:2])}
• Equipo: {month_data['resources']['team_size']} personas
• Budget: ${month_data['resources']['budget_monthly']:,.0f}
"""

    report += f"""

================================================================================
MILESTONES CRÍTICOS
================================================================================

🎯 MES 2: Setup Completo - Sistemas 100% operativos
🚀 MES 5: Piloto Validado - Save rate ≥ 25% Medio Riesgo  
📈 MES 8: Scaling Completo - 100% cobertura Medio Riesgo
🎯 MES 11: Alto Riesgo Operativo - Estrategia selectiva activa
✅ MES 12: Implementación Total - ROI objetivo {roi_target:.0f}%

================================================================================
PRÓXIMOS SCRIPTS
================================================================================

📋 SCRIPTS FINALES DEL PASO 13:

1. paso13e_Outputs_Ejecutivos.py:
   • Dashboards ejecutivos para stakeholders
   • Visualizaciones para board directores
   • Sistema tracking KPIs automático

2. paso13f_Informe_Final_Estratégico.py:
   • Consolidación completa paso 13
   • Recomendaciones finales
   • Template seguimiento y governance

📊 ARCHIVOS GENERADOS:
• Roadmap Excel: excel/paso13d_roadmap_dashboard_{timestamp}.xlsx
• Visualizaciones: graficos/paso13d_roadmap_detallado_{timestamp}.png
• Datos JSON: informes/paso13d_roadmap_detallado_{timestamp}.json
• Este informe: informes/paso13d_roadmap_detallado_{timestamp}.txt

================================================================================
CONCLUSIÓN
================================================================================

✅ ROADMAP 12 MESES COMPLETADO:
• 5 fases estratégicas con priorización Medio Riesgo
• Budget ${total_budget:,.2f} distribuido por fases
• Timeline mensual detallado con KPIs y recursos
• Milestones críticos y plan de contingencia

🎯 ESTRATEGIA CLAVE: MEDIO RIESGO PRIMERO
⚡ MILESTONE CRÍTICO: MES 5 - Decisión Go/No-Go Scaling
💰 ROI OBJETIVO: {roi_target:.1f}% anual

⚠️ RECORDATORIO: Validar supuestos con datos reales antes implementación

================================================================================
FIN DEL ROADMAP
================================================================================
"""
    
    return report

def save_roadmap_results(timeline, phases, business_case, timestamp):
    """Guardar resultados del roadmap"""
    try:
        logging.info("Guardando resultados del roadmap...")
        
        roadmap_data = {
            'metadata': {
                'timestamp': timestamp,
                'script': 'paso13d_Roadmap_Detallado',
                'version': '1.0',
                'disclaimer': 'ROADMAP BASADO EN BUSINESS CASE CON DATOS ESTIMADOS'
            },
            'phases': phases,
            'timeline_monthly': timeline,
            'business_case_ref': {
                'roi_target': business_case['intervention_scenarios']['consolidated']['overall_roi_annual'],
                'total_investment': business_case['intervention_scenarios']['consolidated']['total_annual_investment'],
                'payback_months': business_case['intervention_scenarios']['consolidated']['overall_payback_months']
            },
            'summary_metrics': {
                'total_duration_months': 12,
                'total_phases': len(phases),
                'total_budget_estimated': sum([t['resources']['budget_monthly'] for t in timeline]),
                'priority_segment': 'Medio_Riesgo',
                'implementation_approach': 'Phased with Medio Riesgo priority'
            }
        }
        
        json_file = f'informes/paso13d_roadmap_detallado_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(roadmap_data, f, indent=2, ensure_ascii=False, default=str)
        
        txt_file = f'informes/paso13d_roadmap_detallado_{timestamp}.txt'
        report_content = generate_roadmap_report(timeline, phases, business_case, timestamp)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"JSON roadmap guardado: {json_file}")
        logging.info(f"Informe roadmap guardado: {txt_file}")
        
        return {
            'json_file': json_file,
            'txt_file': txt_file
        }
        
    except Exception as e:
        logging.error(f"Error guardando resultados roadmap: {str(e)}")
        raise

def main():
    """Función principal del Paso 13D - Roadmap Detallado"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13D: ROADMAP DETALLADO DE IMPLEMENTACIÓN")
        logger.info("="*80)
        logger.warning("⚠️  BASADO EN BUSINESS CASE CON DATOS ESTIMADOS")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos del business case del Paso 13C
        logger.info("="*50)
        logger.info("CARGANDO DATOS DEL BUSINESS CASE DEL PASO 13C")
        business_case = load_business_case_data()
        
        # 3. Definir fases de implementación
        logger.info("="*50)
        logger.info("DEFINIENDO FASES DE IMPLEMENTACIÓN ESTRATÉGICA")
        phases = define_implementation_phases()
        
        # 4. Crear timeline mensual detallado
        logger.info("="*50)
        logger.info("CREANDO TIMELINE MENSUAL DETALLADO")
        timeline = create_monthly_timeline(phases)
        
        # 5. Crear visualizaciones del roadmap
        logger.info("="*50)
        logger.info("CREANDO VISUALIZACIONES DEL ROADMAP")
        viz_file = create_roadmap_visualization(timeline, phases, business_case, timestamp)
        
        # 6. Crear dashboard Excel
        logger.info("="*50)
        logger.info("CREANDO DASHBOARD EXCEL DEL ROADMAP")
        excel_file = create_excel_roadmap_dashboard(timeline, phases, business_case, timestamp)
        
        # 7. Guardar resultados
        logger.info("="*50)
        logger.info("GUARDANDO RESULTADOS DEL ROADMAP")
        output_files = save_roadmap_results(timeline, phases, business_case, timestamp)
        
        # 8. Resumen final
        logger.info("="*80)
        logger.info("PASO 13D COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar resultados principales
        total_budget = sum([t['resources']['budget_monthly'] for t in timeline])
        roi_target = business_case['intervention_scenarios']['consolidated']['overall_roi_annual']
        
        logger.warning("⚠️⚠️⚠️ ROADMAP BASADO EN DATOS ESTIMADOS ⚠️⚠️⚠️")
        logger.info("")
        
        logger.info("📅 ROADMAP DE IMPLEMENTACIÓN:")
        logger.info(f"  • Duración total: 12 meses")
        logger.info(f"  • Número de fases: {len(phases)}")
        logger.info(f"  • Budget total estimado: ${total_budget:,.2f}")
        logger.info(f"  • ROI objetivo: {roi_target:.1f}%")
        logger.info("")
        
        logger.info("🎯 FASES PRINCIPALES:")
        for phase_id, phase in phases.items():
            logger.info(f"  • {phase['name']}: {phase['duration_months']} meses ({phase['budget_percentage']}% budget)")
        logger.info("")
        
        logger.info("🚀 ESTRATEGIA PRIORIZADA:")
        logger.info("  • FASE 2-3: Medio Riesgo (prioridad máxima)")
        logger.info("  • FASE 4: Alto Riesgo (selectivo)")
        logger.info("  • FASE 5: Bajo Riesgo (mantenimiento)")
        logger.info("")
        
        logger.info("📊 MILESTONES CRÍTICOS:")
        logger.info("  • Mes 2: Setup completo (Go/No-Go)")
        logger.info("  • Mes 5: Piloto validado (Go/No-Go Scaling)")
        logger.info("  • Mes 8: Medio Riesgo completo (Go/No-Go Alto)")
        logger.info("  • Mes 12: Implementación completa")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • JSON roadmap: {output_files['json_file']}")
        logger.info(f"  • Informe detallado: {output_files['txt_file']}")
        if excel_file:
            logger.info(f"  • Dashboard Excel: {excel_file}")
        if viz_file:
            logger.info(f"  • Visualizaciones: {viz_file}")
        logger.info("")
        
        logger.warning("⚠️ IMPORTANTE: VALIDAR CON DATOS REALES")
        logger.warning("Timeline y recursos basados en benchmarks estimados")
        logger.warning("Testear supuestos con piloto antes de implementación completa")
        logger.info("")
        
        logger.info("📋 LISTO PARA SCRIPTS FINALES:")
        logger.info("  • paso13e: Outputs ejecutivos para stakeholders")
        logger.info("  • paso13f: Informe final estratégico consolidado")
        logger.info("="*80)
        
        return {
            'phases': phases,
            'timeline': timeline,
            'business_case': business_case,
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