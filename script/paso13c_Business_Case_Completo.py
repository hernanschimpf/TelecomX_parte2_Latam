"""
================================================================================
TELECOMX - PASO 13C: BUSINESS CASE COMPLETO CON ROI Y PROYECCIONES FINANCIERAS
================================================================================
Descripción: Desarrollo de business case completo basado en segmentación estratégica
             con cálculos de ROI, payback period y proyecciones financieras a 3 años.
             
⚠️  IMPORTANTE: Este análisis utiliza DATOS ESTIMADOS de benchmarks de la industria 
    telecom para realizar un EJERCICIO DE SIMULACIÓN, ya que no se cuenta con 
    datos financieros reales de la empresa.

Inputs: 
- Segmentación estratégica del Paso 13B
- Factores críticos del Paso 13A
- Benchmarks estándar de la industria telecom

Outputs:
- Business case completo con ROI por segmento
- Proyecciones financieras a 3 años
- Análisis de payback period
- Dashboard financiero ejecutivo
- Recomendaciones de inversión

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
            logging.FileHandler('logs/paso13c_business_case_completo.log', mode='a', encoding='utf-8'),
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

def load_segmentation_data():
    """Cargar datos de segmentación del Paso 13B"""
    try:
        logging.info("Cargando datos de segmentación del Paso 13B...")
        
        json_file = find_latest_file('informes', 'paso13b_segmentacion_estrategica_*.json')
        
        with open(json_file, 'r', encoding='utf-8') as f:
            paso13b_data = json.load(f)
        
        segmentation_summary = paso13b_data['segmentation_summary']
        strategies = paso13b_data['strategies']
        probability_stats = paso13b_data['probability_stats']
        critical_factors = paso13b_data['critical_factors_used']
        
        logging.info(f"Datos de segmentación cargados: {len(segmentation_summary)} segmentos")
        logging.info(f"Total clientes: {paso13b_data['metadata']['total_clients']:,}")
        
        return {
            'segmentation_summary': segmentation_summary,
            'strategies': strategies,
            'probability_stats': probability_stats,
            'critical_factors': critical_factors,
            'metadata': paso13b_data['metadata'],
            'source_file': json_file
        }
        
    except Exception as e:
        logging.error(f"Error cargando datos de segmentación: {str(e)}")
        raise

def define_industry_benchmarks():
    """Definir benchmarks estándar de la industria telecom para simulación"""
    
    logging.info("⚠️  DEFINIENDO BENCHMARKS DE INDUSTRIA TELECOM (DATOS ESTIMADOS)")
    
    benchmarks = {
        'financial_metrics': {
            'average_monthly_revenue_per_user': 45.0,
            'customer_acquisition_cost': 150.0,
            'annual_churn_rate_baseline': 0.265,
            'customer_lifetime_months': 24.0,
            'gross_margin_percentage': 0.75,
            'operational_cost_per_customer_monthly': 12.0
        },
        'retention_effectiveness': {
            'save_rate_alto_riesgo': 0.20,
            'save_rate_medio_riesgo': 0.35,
            'save_rate_bajo_riesgo': 0.15,
            'campaign_effectiveness_decline': 0.15,
            'time_to_impact_months': 3
        },
        'investment_costs': {
            'cost_per_retention_attempt_alto': 45.0,
            'cost_per_retention_attempt_medio': 35.0,
            'cost_per_retention_attempt_bajo': 25.0,
            'technology_setup_cost': 50000.0,
            'training_cost_per_agent': 1500.0,
            'monthly_operational_overhead': 8000.0
        },
        'market_assumptions': {
            'annual_inflation_rate': 0.03,
            'revenue_growth_retained_customers': 0.05,
            'discount_rate_npv': 0.10,
            'market_saturation_factor': 0.95
        }
    }
    
    logging.warning("="*80)
    logging.warning("⚠️  ADVERTENCIA: DATOS FINANCIEROS ESTIMADOS")
    logging.warning("="*80)
    logging.warning("Los siguientes datos son ESTIMACIONES basadas en benchmarks")
    logging.warning("de la industria telecom para fines de SIMULACIÓN:")
    logging.warning(f"• ARPU mensual: ${benchmarks['financial_metrics']['average_monthly_revenue_per_user']}")
    logging.warning(f"• CAC: ${benchmarks['financial_metrics']['customer_acquisition_cost']}")
    logging.warning(f"• Save rates: Alto {benchmarks['retention_effectiveness']['save_rate_alto_riesgo']:.0%}, "
                   f"Medio {benchmarks['retention_effectiveness']['save_rate_medio_riesgo']:.0%}, "
                   f"Bajo {benchmarks['retention_effectiveness']['save_rate_bajo_riesgo']:.0%}")
    logging.warning("="*80)
    
    return benchmarks

def calculate_current_state_financials(segmentation_data, benchmarks):
    """Calcular estado financiero actual (baseline) antes de intervenciones"""
    try:
        logging.info("Calculando estado financiero actual (baseline)...")
        
        segmentation_summary = segmentation_data['segmentation_summary']
        financial_metrics = benchmarks['financial_metrics']
        
        current_state = {}
        total_revenue = 0
        total_churn_loss = 0
        total_customers = 0
        
        for segment, data in segmentation_summary.items():
            customers = data['count']
            monthly_revenue_per_customer = financial_metrics['average_monthly_revenue_per_user']
            annual_churn_rate = data['actual_churn_rate']
            
            monthly_revenue_segment = customers * monthly_revenue_per_customer
            annual_revenue_segment = monthly_revenue_segment * 12
            annual_churned_customers = customers * annual_churn_rate
            annual_churn_revenue_loss = annual_churned_customers * monthly_revenue_per_customer * 12
            
            current_state[segment] = {
                'customers': customers,
                'monthly_revenue': monthly_revenue_segment,
                'annual_revenue': annual_revenue_segment,
                'annual_churn_rate': annual_churn_rate,
                'annual_churned_customers': annual_churned_customers,
                'annual_churn_revenue_loss': annual_churn_revenue_loss,
                'customer_lifetime_value': monthly_revenue_per_customer * financial_metrics['customer_lifetime_months'],
                'segment_ltv_total': customers * monthly_revenue_per_customer * financial_metrics['customer_lifetime_months']
            }
            
            total_revenue += annual_revenue_segment
            total_churn_loss += annual_churn_revenue_loss
            total_customers += customers
        
        current_state['totals'] = {
            'total_customers': total_customers,
            'total_annual_revenue': total_revenue,
            'total_annual_churn_loss': total_churn_loss,
            'overall_churn_rate': total_churn_loss / total_revenue if total_revenue > 0 else 0,
            'churn_impact_percentage': (total_churn_loss / total_revenue * 100) if total_revenue > 0 else 0,
            'average_revenue_per_customer': total_revenue / total_customers if total_customers > 0 else 0
        }
        
        logging.info(f"Estado actual calculado:")
        logging.info(f"  • Total customers: {total_customers:,}")
        logging.info(f"  • Revenue anual: ${total_revenue:,.2f}")
        logging.info(f"  • Pérdida por churn: ${total_churn_loss:,.2f} ({current_state['totals']['churn_impact_percentage']:.1f}%)")
        
        return current_state
        
    except Exception as e:
        logging.error(f"Error calculando estado actual: {str(e)}")
        raise

def calculate_intervention_scenarios(segmentation_data, current_state, benchmarks):
    """Calcular escenarios de intervención y ROI por segmento"""
    try:
        logging.info("Calculando escenarios de intervención y ROI...")
        
        segmentation_summary = segmentation_data['segmentation_summary']
        retention_effectiveness = benchmarks['retention_effectiveness']
        investment_costs = benchmarks['investment_costs']
        financial_metrics = benchmarks['financial_metrics']
        
        intervention_scenarios = {}
        
        cost_mapping = {
            'Alto_Riesgo': investment_costs['cost_per_retention_attempt_alto'],
            'Medio_Riesgo': investment_costs['cost_per_retention_attempt_medio'],
            'Bajo_Riesgo': investment_costs['cost_per_retention_attempt_bajo']
        }
        
        save_rate_mapping = {
            'Alto_Riesgo': retention_effectiveness['save_rate_alto_riesgo'],
            'Medio_Riesgo': retention_effectiveness['save_rate_medio_riesgo'],
            'Bajo_Riesgo': retention_effectiveness['save_rate_bajo_riesgo']
        }
        
        total_investment = 0
        total_revenue_saved = 0
        
        for segment, data in segmentation_summary.items():
            if segment in current_state:
                current_segment = current_state[segment]
                customers_at_risk = current_segment['annual_churned_customers']
                cost_per_attempt = cost_mapping.get(segment, 15.0)
                save_rate = save_rate_mapping.get(segment, 0.35)
                
                customers_to_save = customers_at_risk * save_rate
                investment_required = customers_at_risk * cost_per_attempt
                revenue_saved_annual = customers_to_save * financial_metrics['average_monthly_revenue_per_user'] * 12
                
                net_benefit = revenue_saved_annual - investment_required
                roi_percentage = (net_benefit / investment_required * 100) if investment_required > 0 else 0
                payback_months = (investment_required / (revenue_saved_annual / 12)) if revenue_saved_annual > 0 else 999
                
                ltv_customers_saved = customers_to_save * financial_metrics['average_monthly_revenue_per_user'] * 24
                
                intervention_scenarios[segment] = {
                    'baseline_customers_at_risk': customers_at_risk,
                    'save_rate_assumed': save_rate,
                    'customers_saved': customers_to_save,
                    'investment_required_annual': investment_required,
                    'revenue_saved_annual': revenue_saved_annual,
                    'ltv_customers_saved': ltv_customers_saved,
                    'net_benefit_annual': net_benefit,
                    'roi_percentage': roi_percentage,
                    'payback_months': payback_months,
                    'cost_per_attempt': cost_per_attempt,
                    'cost_per_customer_saved': investment_required / customers_to_save if customers_to_save > 0 else 0,
                    'revenue_per_customer_saved': revenue_saved_annual / customers_to_save if customers_to_save > 0 else 0
                }
                
                total_investment += investment_required
                total_revenue_saved += revenue_saved_annual
        
        setup_costs = {
            'technology_setup': investment_costs['technology_setup_cost'],
            'training_cost': investment_costs['training_cost_per_agent'] * 10,
            'total_setup': investment_costs['technology_setup_cost'] + (investment_costs['training_cost_per_agent'] * 10)
        }
        
        intervention_scenarios['consolidated'] = {
            'total_annual_investment': total_investment,
            'total_annual_revenue_saved': total_revenue_saved,
            'setup_costs': setup_costs,
            'total_first_year_cost': total_investment + setup_costs['total_setup'],
            'net_benefit_annual': total_revenue_saved - total_investment,
            'overall_roi_annual': ((total_revenue_saved - total_investment) / total_investment * 100) if total_investment > 0 else 0,
            'overall_payback_months': (total_investment / (total_revenue_saved / 12)) if total_revenue_saved > 0 else 999,
            'monthly_operational_overhead': investment_costs['monthly_operational_overhead']
        }
        
        logging.info(f"Escenarios de intervención calculados:")
        logging.info(f"  • Inversión anual total: ${total_investment:,.2f}")
        logging.info(f"  • Revenue salvado anual: ${total_revenue_saved:,.2f}")
        logging.info(f"  • ROI anual estimado: {intervention_scenarios['consolidated']['overall_roi_annual']:.1f}%")
        
        return intervention_scenarios
        
    except Exception as e:
        logging.error(f"Error calculando escenarios de intervención: {str(e)}")
        raise

def create_3year_projections(current_state, intervention_scenarios, benchmarks):
    """Crear proyecciones financieras a 3 años"""
    try:
        logging.info("Creando proyecciones financieras a 3 años...")
        
        market_assumptions = benchmarks['market_assumptions']
        retention_effectiveness = benchmarks['retention_effectiveness']
        
        projections = {
            'years': [2025, 2026, 2027],
            'without_intervention': [],
            'with_intervention': [],
            'incremental_benefit': [],
            'cumulative_investment': [],
            'cumulative_benefit': [],
            'cumulative_roi': []
        }
        
        baseline_revenue = current_state['totals']['total_annual_revenue']
        baseline_churn_loss = current_state['totals']['total_annual_churn_loss']
        
        intervention_investment = intervention_scenarios['consolidated']['total_first_year_cost']
        intervention_revenue_saved = intervention_scenarios['consolidated']['total_annual_revenue_saved']
        
        cumulative_investment = 0
        cumulative_benefit = 0
        
        for year_idx, year in enumerate(projections['years']):
            inflation_factor = (1 + market_assumptions['annual_inflation_rate']) ** year_idx
            effectiveness_decline = (1 - market_assumptions['annual_inflation_rate']) ** year_idx
            revenue_growth = (1 + market_assumptions['revenue_growth_retained_customers']) ** year_idx
            
            without_intervention = baseline_revenue * inflation_factor - (baseline_churn_loss * inflation_factor * (1 + 0.05 * year_idx))
            
            if year_idx == 0:
                annual_investment = intervention_investment
                annual_benefit = intervention_revenue_saved * revenue_growth
            else:
                annual_investment = (intervention_scenarios['consolidated']['total_annual_investment'] * 0.7 + 
                                   intervention_scenarios['consolidated']['monthly_operational_overhead'] * 12) * inflation_factor
                annual_benefit = (intervention_revenue_saved * effectiveness_decline * revenue_growth * 
                                (1 + 0.1 * year_idx))
            
            with_intervention = without_intervention + annual_benefit - annual_investment
            incremental_benefit = annual_benefit - annual_investment
            
            cumulative_investment += annual_investment
            cumulative_benefit += annual_benefit
            cumulative_roi = ((cumulative_benefit - cumulative_investment) / cumulative_investment * 100) if cumulative_investment > 0 else 0
            
            projections['without_intervention'].append(without_intervention)
            projections['with_intervention'].append(with_intervention)
            projections['incremental_benefit'].append(incremental_benefit)
            projections['cumulative_investment'].append(cumulative_investment)
            projections['cumulative_benefit'].append(cumulative_benefit)
            projections['cumulative_roi'].append(cumulative_roi)
        
        discount_rate = market_assumptions['discount_rate_npv']
        npv_benefits = sum([benefit / (1 + discount_rate) ** (i + 1) for i, benefit in enumerate(projections['incremental_benefit'])])
        
        projections['summary'] = {
            'total_3year_investment': cumulative_investment,
            'total_3year_benefit': sum(projections['incremental_benefit']),
            'net_present_value': npv_benefits,
            'final_cumulative_roi': projections['cumulative_roi'][-1],
            'break_even_year': next((i + 1 for i, roi in enumerate(projections['cumulative_roi']) if roi > 0), None)
        }
        
        logging.info(f"Proyecciones 3 años calculadas:")
        logging.info(f"  • NPV: ${npv_benefits:,.2f}")
        logging.info(f"  • ROI acumulado 3 años: {projections['summary']['final_cumulative_roi']:.1f}%")
        logging.info(f"  • Break-even año: {projections['summary']['break_even_year']}")
        
        return projections
        
    except Exception as e:
        logging.error(f"Error creando proyecciones: {str(e)}")
        raise

def create_business_case_visualization(current_state, intervention_scenarios, projections, timestamp):
    """Crear visualizaciones del business case"""
    try:
        logging.info("Creando visualizaciones del business case...")
        
        plt.style.use('default')
        fig = plt.figure(figsize=(24, 18))
        
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.25, top=0.91, bottom=0.07, left=0.05, right=0.99)
        
        colors_segments = {'Alto_Riesgo': '#DC143C', 'Medio_Riesgo': '#FFD700', 'Bajo_Riesgo': '#2E8B57'}
        
        # 1. ROI por Segmento
        ax1 = fig.add_subplot(gs[0, 0])
        segments = [seg for seg in intervention_scenarios.keys() if seg != 'consolidated']
        customers_saved_values = [intervention_scenarios[seg]['customers_saved'] for seg in segments]
        colors = [colors_segments.get(seg, '#87CEEB') for seg in segments]

        bars = ax1.bar(segments, customers_saved_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Clientes Salvados', fontweight='bold', fontsize=11)
        ax1.set_title('Clientes Salvados por Segmento (Anual)\n⚠️ Estimación Benchmark - Resultados Año 1', fontweight='bold', fontsize=12, pad=15)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, customers in zip(bars, customers_saved_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{customers:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Inversión vs Revenue Salvado
        ax2 = fig.add_subplot(gs[0, 1])
        investment_values = [intervention_scenarios[seg]['investment_required_annual']/1000 for seg in segments]
        revenue_values = [intervention_scenarios[seg]['revenue_saved_annual']/1000 for seg in segments]
        
        x = np.arange(len(segments))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, investment_values, width, label='Inversión', color='#FF6347', alpha=0.8)
        bars2 = ax2.bar(x + width/2, revenue_values, width, label='Revenue Salvado', color='#4682B4', alpha=0.8)
        
        ax2.set_ylabel('Miles USD', fontweight='bold', fontsize=11)
        ax2.set_title('Inversión vs Revenue Salvado (Anual)\n⚠️ Estimaciones Benchmark', fontweight='bold', fontsize=12, pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels([seg.replace('_', '\n') for seg in segments], fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Payback Period
        ax3 = fig.add_subplot(gs[0, 2])
        payback_values = [min(intervention_scenarios[seg]['payback_months'], 36) for seg in segments]
        
        bars = ax3.bar(segments, payback_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_ylabel('Meses', fontweight='bold', fontsize=11)
        ax3.set_title('Período de Payback\n⚠️ Simulación Telecom', fontweight='bold', fontsize=12, pad=15)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, months in zip(bars, payback_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{months:.1f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Distribución de Inversión Total
        ax4 = fig.add_subplot(gs[0, 3])
        total_investment = intervention_scenarios['consolidated']['total_annual_investment']
        segment_investments = [intervention_scenarios[seg]['investment_required_annual'] for seg in segments]
        segment_percentages = [(inv/total_investment)*100 for inv in segment_investments]
        
        wedges, texts, autotexts = ax4.pie(segment_percentages, labels=[seg.replace('_', '\n') for seg in segments], 
                                          autopct='%1.1f%%', colors=colors, startangle=90, 
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax4.set_title('Distribución de Inversión\n⚠️ Datos Estimados', fontweight='bold', fontsize=12, pad=15)
        
        # 5. Proyección 3 años - Revenue
        ax5 = fig.add_subplot(gs[1, :2])
        years = projections['years']
        without_int = [val/1000000 for val in projections['without_intervention']]
        with_int = [val/1000000 for val in projections['with_intervention']]
        
        ax5.plot(years, without_int, marker='o', linewidth=3, markersize=8, 
                label='Sin Intervención', color='#FF6347', linestyle='--')
        ax5.plot(years, with_int, marker='s', linewidth=3, markersize=8, 
                label='Con Intervención', color='#2E8B57')
        
        ax5.set_ylabel('Revenue (Millones USD)', fontweight='bold', fontsize=11)
        ax5.set_title('Proyección Revenue 3 Años - ⚠️ SIMULACIÓN CON BENCHMARKS INDUSTRIA', 
                     fontweight='bold', fontsize=12, pad=15)
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        for i, (year, without, with_val) in enumerate(zip(years, without_int, with_int)):
            ax5.annotate(f'${without:.1f}M', (year, without), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
            ax5.annotate(f'${with_val:.1f}M', (year, with_val), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
        # 6. ROI Acumulado
        ax6 = fig.add_subplot(gs[1, 2:])
        cumulative_roi = projections['cumulative_roi']
        
        bars = ax6.bar(years, cumulative_roi, color=['#FF6347' if roi < 0 else '#2E8B57' for roi in cumulative_roi], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax6.set_ylabel('ROI Acumulado (%)', fontweight='bold', fontsize=11)
        ax6.set_title('ROI Acumulado 3 Años - ⚠️ ESTIMACIÓN BENCHMARK', fontweight='bold', fontsize=12, pad=15)
        ax6.grid(True, alpha=0.3, axis='y')
        
        for bar, roi in zip(bars, cumulative_roi):
            color = 'red' if roi < 0 else 'green'
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (5 if roi > 0 else -15),
                    f'{roi:.1f}%', ha='center', va='bottom' if roi > 0 else 'top', 
                    fontweight='bold', fontsize=11, color=color)
        
        # 7. Beneficio Incremental por Año
        ax7 = fig.add_subplot(gs[2, :2])
        incremental_benefits = [val/1000000 for val in projections['incremental_benefit']]
        
        bars = ax7.bar(years, incremental_benefits, 
                      color=['#FF6347' if benefit < 0 else '#2E8B57' for benefit in incremental_benefits], 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.set_ylabel('Beneficio Incremental (Millones USD)', fontweight='bold', fontsize=11)
        ax7.set_title('Beneficio Incremental Anual - ⚠️ DATOS SIMULACIÓN TELECOM', 
                     fontweight='bold', fontsize=12, pad=15)
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar, benefit in zip(bars, incremental_benefits):
            color = 'red' if benefit < 0 else 'green'
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if benefit > 0 else -0.2),
                    f'${benefit:.1f}M', ha='center', va='bottom' if benefit > 0 else 'top', 
                    fontweight='bold', fontsize=11, color=color)
        
        # 8. Resumen Financiero Key Metrics
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('off')
        
        summary = projections['summary']
        metrics_text = f"""
📊 RESUMEN FINANCIERO - BUSINESS CASE
⚠️  SIMULACIÓN CON DATOS ESTIMADOS INDUSTRIA TELECOM

💰 INVERSIÓN Y RETORNOS:
• Inversión Total 3 años: ${summary['total_3year_investment']:,.0f}
• Beneficio Total 3 años: ${summary['total_3year_benefit']:,.0f}
• NPV (10% descuento): ${summary['net_present_value']:,.0f}

📈 MÉTRICAS DE PERFORMANCE:
• ROI Acumulado Final: {summary['final_cumulative_roi']:.1f}%
• Break-even: Año {summary['break_even_year'] or 'N/A'}
• Payback Promedio: {intervention_scenarios['consolidated']['overall_payback_months']:.1f} meses

🎯 SEGMENTO MÁS RENTABLE: Medio Riesgo
💡 RECOMENDACIÓN: PROCEDER CON IMPLEMENTACIÓN

⚠️  NOTA IMPORTANTE:
Estos cálculos utilizan benchmarks estándar de la 
industria telecom para fines de simulación.
Para implementación real, reemplazar con datos 
financieros específicos de la empresa.
        """
        
        ax8.text(0.05, 0.95, metrics_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
        
        fig.suptitle('TelecomX - Business Case Completo con ROI y Proyecciones\n⚠️ ANÁLISIS DE SIMULACIÓN CON BENCHMARKS INDUSTRIA TELECOM ⚠️', 
                    fontsize=18, fontweight='bold', y=0.99, color='#2E4057')
        
        os.makedirs('graficos', exist_ok=True)
        
        viz_file = f'graficos/paso13c_business_case_completo_{timestamp}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.3)
        plt.close()
        
        if os.path.exists(viz_file):
            file_size = os.path.getsize(viz_file)
            logging.info(f"Visualización business case guardada: {viz_file} ({file_size:,} bytes)")
        else:
            logging.error(f"ERROR: No se pudo crear visualización: {viz_file}")
            return None
        
        return viz_file
        
    except Exception as e:
        logging.error(f"Error creando visualización business case: {str(e)}")
        return None

def create_excel_business_case_dashboard(current_state, intervention_scenarios, projections, benchmarks, timestamp):
    """Crear dashboard ejecutivo Excel del business case"""
    try:
        logging.info("Creando dashboard Excel del business case...")
        
        excel_file = f'excel/paso13c_business_case_dashboard_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            
            # 1. Hoja: Executive Summary
            exec_summary_data = []
            consolidated = intervention_scenarios['consolidated']
            summary = projections['summary']
            
            exec_summary_data.append(['MÉTRICA', 'VALOR', 'NOTA'])
            exec_summary_data.append(['=== INVERSIÓN Y RETORNOS ===', '', ''])
            exec_summary_data.append(['Inversión Anual', f"${consolidated['total_annual_investment']:,.2f}", 'Costo operacional anual'])
            exec_summary_data.append(['Revenue Salvado Anual', f"${consolidated['total_annual_revenue_saved']:,.2f}", 'Revenue retenido por intervención'])
            exec_summary_data.append(['ROI Anual', f"{consolidated['overall_roi_annual']:.1f}%", 'Retorno sobre inversión año 1'])
            exec_summary_data.append(['Payback Period', f"{consolidated['overall_payback_months']:.1f} meses", 'Tiempo para recuperar inversión'])
            exec_summary_data.append(['', '', ''])
            exec_summary_data.append(['=== PROYECCIÓN 3 AÑOS ===', '', ''])
            exec_summary_data.append(['NPV (10% descuento)', f"${summary['net_present_value']:,.2f}", 'Valor presente neto'])
            exec_summary_data.append(['ROI Acumulado 3 años', f"{summary['final_cumulative_roi']:.1f}%", 'ROI acumulado final'])
            exec_summary_data.append(['Break-even', f"Año {summary['break_even_year'] or 'N/A'}", 'Año de equilibrio'])
            exec_summary_data.append(['', '', ''])
            exec_summary_data.append(['⚠️ IMPORTANTE', 'DATOS ESTIMADOS', 'Benchmarks industria telecom'])
            
            exec_summary_df = pd.DataFrame(exec_summary_data)
            exec_summary_df.to_excel(writer, sheet_name='Executive Summary', index=False, header=False)
            
            # 2. Hoja: ROI por Segmento
            roi_data = []
            for segment, data in intervention_scenarios.items():
                if segment != 'consolidated':
                    roi_data.append({
                        'Segmento': segment,
                        'Clientes_en_Riesgo': int(data['baseline_customers_at_risk']),
                        'Save_Rate_Estimado': f"{data['save_rate_assumed']:.1%}",
                        'Clientes_Salvados': int(data['customers_saved']),
                        'Inversión_Requerida': f"${data['investment_required_annual']:,.2f}",
                        'Revenue_Salvado': f"${data['revenue_saved_annual']:,.2f}",
                        'ROI_Porcentaje': f"{data['roi_percentage']:.1f}%",
                        'Payback_Meses': f"{data['payback_months']:.1f}",
                        'Costo_por_Cliente_Salvado': f"${data['cost_per_customer_saved']:,.2f}"
                    })
            
            roi_df = pd.DataFrame(roi_data)
            roi_df.to_excel(writer, sheet_name='ROI por Segmento', index=False)
            
            # 3. Hoja: Proyecciones 3 Años
            projections_data = []
            for i, year in enumerate(projections['years']):
                projections_data.append({
                    'Año': year,
                    'Revenue_Sin_Intervención': f"${projections['without_intervention'][i]:,.2f}",
                    'Revenue_Con_Intervención': f"${projections['with_intervention'][i]:,.2f}",
                    'Beneficio_Incremental': f"${projections['incremental_benefit'][i]:,.2f}",
                    'Inversión_Acumulada': f"${projections['cumulative_investment'][i]:,.2f}",
                    'Beneficio_Acumulado': f"${projections['cumulative_benefit'][i]:,.2f}",
                    'ROI_Acumulado': f"{projections['cumulative_roi'][i]:.1f}%"
                })
            
            projections_df = pd.DataFrame(projections_data)
            projections_df.to_excel(writer, sheet_name='Proyecciones 3 Años', index=False)
            
            # 4. Hoja: Estado Actual (Baseline)
            baseline_data = []
            for segment, data in current_state.items():
                if segment != 'totals':
                    baseline_data.append({
                        'Segmento': segment,
                        'Clientes': int(data['customers']),
                        'Revenue_Mensual': f"${data['monthly_revenue']:,.2f}",
                        'Revenue_Anual': f"${data['annual_revenue']:,.2f}",
                        'Churn_Rate': f"{data['annual_churn_rate']:.1%}",
                        'Clientes_Perdidos_Anual': int(data['annual_churned_customers']),
                        'Revenue_Perdido_Churn': f"${data['annual_churn_revenue_loss']:,.2f}",
                        'LTV_por_Cliente': f"${data['customer_lifetime_value']:,.2f}"
                    })
            
            baseline_df = pd.DataFrame(baseline_data)
            baseline_df.to_excel(writer, sheet_name='Estado Actual', index=False)
            
            # 5. Hoja: Benchmarks Utilizados
            benchmarks_data = []
            
            financial = benchmarks['financial_metrics']
            benchmarks_data.append(['MÉTRICA', 'VALOR', 'CATEGORÍA'])
            benchmarks_data.append(['=== MÉTRICAS FINANCIERAS ===', '', ''])
            benchmarks_data.append(['ARPU Mensual', f"${financial['average_monthly_revenue_per_user']}", 'Financial'])
            benchmarks_data.append(['CAC', f"${financial['customer_acquisition_cost']}", 'Financial'])
            benchmarks_data.append(['Churn Rate Baseline', f"{financial['annual_churn_rate_baseline']:.1%}", 'Financial'])
            benchmarks_data.append(['Customer Lifetime (meses)', f"{financial['customer_lifetime_months']}", 'Financial'])
            benchmarks_data.append(['Margen Bruto', f"{financial['gross_margin_percentage']:.1%}", 'Financial'])
            
            retention = benchmarks['retention_effectiveness']
            benchmarks_data.append(['', '', ''])
            benchmarks_data.append(['=== EFECTIVIDAD RETENCIÓN ===', '', ''])
            benchmarks_data.append(['Save Rate Alto Riesgo', f"{retention['save_rate_alto_riesgo']:.1%}", 'Retention'])
            benchmarks_data.append(['Save Rate Medio Riesgo', f"{retention['save_rate_medio_riesgo']:.1%}", 'Retention'])
            benchmarks_data.append(['Save Rate Bajo Riesgo', f"{retention['save_rate_bajo_riesgo']:.1%}", 'Retention'])
            
            costs = benchmarks['investment_costs']
            benchmarks_data.append(['', '', ''])
            benchmarks_data.append(['=== COSTOS DE INVERSIÓN ===', '', ''])
            benchmarks_data.append(['Costo Retención Alto Riesgo', f"${costs['cost_per_retention_attempt_alto']}", 'Investment'])
            benchmarks_data.append(['Costo Retención Medio Riesgo', f"${costs['cost_per_retention_attempt_medio']}", 'Investment'])
            benchmarks_data.append(['Costo Retención Bajo Riesgo', f"${costs['cost_per_retention_attempt_bajo']}", 'Investment'])
            benchmarks_data.append(['Setup Tecnología', f"${costs['technology_setup_cost']:,.0f}", 'Investment'])
            
            benchmarks_data.append(['', '', ''])
            benchmarks_data.append(['⚠️ IMPORTANTE', 'ESTOS SON DATOS ESTIMADOS', 'Disclaimer'])
            benchmarks_data.append(['Fuente', 'Benchmarks industria telecom', 'Disclaimer'])
            benchmarks_data.append(['Propósito', 'Ejercicio de simulación', 'Disclaimer'])
            
            benchmarks_df = pd.DataFrame(benchmarks_data)
            benchmarks_df.to_excel(writer, sheet_name='Benchmarks Utilizados', index=False, header=False)
        
        logging.info(f"Dashboard Excel business case creado: {excel_file}")
        return excel_file
        
    except Exception as e:
        logging.error(f"Error creando dashboard Excel: {str(e)}")
        return None
    
def generate_business_case_report(current_state, intervention_scenarios, projections, benchmarks, timestamp):
    """Generar informe ejecutivo del business case"""
    
    consolidated = intervention_scenarios['consolidated']
    summary = projections['summary']
    totals = current_state['totals']
    
    report = f"""
================================================================================
TELECOMX - PASO 13C: BUSINESS CASE COMPLETO CON ROI Y PROYECCIONES FINANCIERAS
================================================================================
Fecha: {timestamp}
Script: paso13c_Business_Case_Completo.py

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
                            🚨 DISCLAIMER IMPORTANTE 🚨
================================================================================
⚠️  ESTE ANÁLISIS UTILIZA DATOS ESTIMADOS DE BENCHMARKS DE LA INDUSTRIA 
    TELECOM PARA REALIZAR UN EJERCICIO DE SIMULACIÓN, YA QUE NO SE CUENTA 
    CON DATOS FINANCIEROS REALES DE LA EMPRESA.

📊 DATOS UTILIZADOS (ESTIMACIONES INDUSTRIA):
• ARPU mensual: ${benchmarks['financial_metrics']['average_monthly_revenue_per_user']}
• CAC: ${benchmarks['financial_metrics']['customer_acquisition_cost']}  
• Save rates: Alto {benchmarks['retention_effectiveness']['save_rate_alto_riesgo']:.0%}, Medio {benchmarks['retention_effectiveness']['save_rate_medio_riesgo']:.0%}, Bajo {benchmarks['retention_effectiveness']['save_rate_bajo_riesgo']:.0%}
• Costos retención: ${benchmarks['investment_costs']['cost_per_retention_attempt_bajo']}-${benchmarks['investment_costs']['cost_per_retention_attempt_alto']} por intento

🎯 PROPÓSITO: Demostrar metodología de business case para retención de clientes
💡 PARA IMPLEMENTACIÓN REAL: Reemplazar con datos financieros específicos de la empresa

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

================================================================================
RESUMEN EJECUTIVO
================================================================================

🎯 OBJETIVO DEL BUSINESS CASE:
Evaluar la viabilidad financiera de implementar estrategias de retención 
segmentadas para reducir churn y maximizar ROI.

📊 ESTADO ACTUAL (BASELINE):
• Total clientes: {totals['total_customers']:,}
• Revenue anual: ${totals['total_annual_revenue']:,.2f}
• Pérdida anual por churn: ${totals['total_annual_churn_loss']:,.2f}
• Impacto churn en revenue: {totals['churn_impact_percentage']:.1f}%
• Churn rate promedio: {totals['overall_churn_rate']:.1%}

💰 PROPUESTA DE INVERSIÓN:
• Inversión anual requerida: ${consolidated['total_annual_investment']:,.2f}
• Revenue salvado estimado: ${consolidated['total_annual_revenue_saved']:,.2f}
• ROI anual proyectado: {consolidated['overall_roi_annual']:.1f}%
• Período de payback: {consolidated['overall_payback_months']:.1f} meses

================================================================================
ANÁLISIS POR SEGMENTO (ESTIMACIONES BASADAS EN BENCHMARKS)
================================================================================
"""

    segments_ordered = ['Alto_Riesgo', 'Medio_Riesgo', 'Bajo_Riesgo']
    segment_icons = {'Alto_Riesgo': '🔴', 'Medio_Riesgo': '🟡', 'Bajo_Riesgo': '🟢'}
    
    for segment in segments_ordered:
        if segment in intervention_scenarios:
            data = intervention_scenarios[segment]
            icon = segment_icons.get(segment, '•')
            
            report += f"""
{icon} SEGMENTO {segment.replace('_', ' ').upper()}:

📋 SITUACIÓN ACTUAL:
• Clientes en riesgo anual: {data['baseline_customers_at_risk']:.0f}
• Save rate estimado (benchmark): {data['save_rate_assumed']:.1%}
• Clientes que se pueden salvar: {data['customers_saved']:.0f}

💰 ANÁLISIS FINANCIERO:
• Inversión requerida anual: ${data['investment_required_annual']:,.2f}
• Revenue salvado anual: ${data['revenue_saved_annual']:,.2f}
• Beneficio neto anual: ${data['net_benefit_annual']:,.2f}
• ROI estimado: {data['roi_percentage']:.1f}%
• Payback period: {data['payback_months']:.1f} meses

📊 MÉTRICAS CLAVE:
• Costo por intento retención: ${data['cost_per_attempt']:.2f}
• Costo por cliente salvado: ${data['cost_per_customer_saved']:,.2f}
• Revenue por cliente salvado: ${data['revenue_per_customer_saved']:,.2f}
• LTV clientes salvados: ${data['ltv_customers_saved']:,.2f}
"""

    report += f"""

================================================================================
PROYECCIONES FINANCIERAS 3 AÑOS (SIMULACIÓN CON DATOS ESTIMADOS)
================================================================================

📈 EVOLUCIÓN PROYECTADA:

AÑO 2025 (Implementación):
• Revenue sin intervención: ${projections['without_intervention'][0]:,.2f}
• Revenue con intervención: ${projections['with_intervention'][0]:,.2f}
• Beneficio incremental: ${projections['incremental_benefit'][0]:,.2f}
• ROI acumulado: {projections['cumulative_roi'][0]:.1f}%

AÑO 2026 (Optimización):
• Revenue sin intervención: ${projections['without_intervention'][1]:,.2f}
• Revenue con intervención: ${projections['with_intervention'][1]:,.2f}
• Beneficio incremental: ${projections['incremental_benefit'][1]:,.2f}
• ROI acumulado: {projections['cumulative_roi'][1]:.1f}%

AÑO 2027 (Madurez):
• Revenue sin intervención: ${projections['without_intervention'][2]:,.2f}
• Revenue con intervención: ${projections['with_intervention'][2]:,.2f}
• Beneficio incremental: ${projections['incremental_benefit'][2]:,.2f}
• ROI acumulado: {projections['cumulative_roi'][2]:.1f}%

🎯 RESUMEN 3 AÑOS:
• Inversión total acumulada: ${summary['total_3year_investment']:,.2f}
• Beneficio total acumulado: ${summary['total_3year_benefit']:,.2f}
• NPV (10% descuento): ${summary['net_present_value']:,.2f}
• ROI final acumulado: {summary['final_cumulative_roi']:.1f}%
• Break-even proyectado: Año {summary['break_even_year'] or 'N/A'}

================================================================================
RECOMENDACIONES ESTRATÉGICAS
================================================================================

🚀 RECOMENDACIÓN PRINCIPAL: PROCEDER CON IMPLEMENTACIÓN

✅ JUSTIFICACIÓN:
• ROI proyectado atractivo: {consolidated['overall_roi_annual']:.1f}% anual
• Payback period razonable: {consolidated['overall_payback_months']:.1f} meses
• NPV positivo: ${summary['net_present_value']:,.2f}
• Break-even temprano: Año {summary['break_even_year'] or 'N/A'}

📋 PLAN DE IMPLEMENTACIÓN RECOMENDADO:

FASE 1 (Meses 1-3): PILOTO CONTROLADO
• Implementar solo segmento Medio Riesgo (mejor ROI)
• 20% de la base de clientes como grupo de prueba
• Validar assumptions de save rate y costos
• Ajustar modelos y estrategias basado en resultados

FASE 2 (Meses 4-6): EXPANSIÓN GRADUAL  
• Rollout a segmentos Alto y Bajo Riesgo
• Implementación completa en 50% de la base
• Optimización de procesos y automatización
• Entrenamiento completo de equipos

FASE 3 (Meses 7-12): IMPLEMENTACIÓN COMPLETA
• Rollout a 100% de la base de clientes
• Monitoreo continuo y optimización
• Análisis de ROI real vs proyectado
• Preparación para año 2

================================================================================
PRÓXIMOS PASOS
================================================================================

📋 SIGUIENTES SCRIPTS DEL PASO 13:

1. paso13d_Roadmap_Detallado.py:
   • Creará cronograma detallado de implementación por fases
   • Definirá milestones, recursos y dependencias críticas
   • Establecerá plan de contingencia y manejo de riesgos

2. paso13e_Outputs_Ejecutivos.py:
   • Generará dashboards ejecutivos para presentación a stakeholders
   • Creará visualizaciones de alto impacto para board de directores
   • Desarrollará sistema de tracking de KPIs automático

3. paso13f_Informe_Final_Estratégico.py:
   • Consolidará todos los resultados en informe ejecutivo completo
   • Generará recomendaciones finales y plan de acción
   • Creará template de seguimiento y governance

📊 ARCHIVOS GENERADOS (CON DISCLAIMER DE DATOS ESTIMADOS):
• Business case Excel: excel/paso13c_business_case_dashboard_{timestamp}.xlsx
• Visualizaciones: graficos/paso13c_business_case_completo_{timestamp}.png
• Datos JSON: informes/paso13c_business_case_completo_{timestamp}.json
• Este informe: informes/paso13c_business_case_completo_{timestamp}.txt

================================================================================
CONCLUSIÓN
================================================================================

✅ BUSINESS CASE VIABLE CON DATOS ESTIMADOS:

• ROI atractivo proyectado: {consolidated['overall_roi_annual']:.1f}% anual
• Payback period razonable: {consolidated['overall_payback_months']:.1f} meses  
• NPV positivo a 3 años: ${summary['net_present_value']:,.2f}
• Beneficio incremental sostenible en el tiempo

🎯 SEGMENTO MÁS RENTABLE: Medio Riesgo (ROI estimado más alto)
⚡ OPORTUNIDAD PRINCIPAL: ${totals['total_annual_churn_loss']:,.2f} en revenue en riesgo anualmente
💰 INVERSIÓN TOTAL REQUERIDA: ${consolidated['total_annual_investment']:,.2f} anualmente

⚠️⚠️⚠️ RECORDATORIO FINAL ⚠️⚠️⚠️
Este business case utiliza DATOS ESTIMADOS de benchmarks de la industria telecom
para demostrar la metodología. Para implementación real:

1. REEMPLAZAR benchmarks con datos financieros reales de la empresa
2. VALIDAR save rates con pruebas piloto controladas  
3. CONFIRMAR costos operacionales con equipos internos
4. AJUSTAR proyecciones basado en contexto específico del negocio

La metodología demostrada es sólida y replicable con datos reales.

================================================================================
FIN DEL BUSINESS CASE
================================================================================
"""
    
    return report

def save_business_case_results(current_state, intervention_scenarios, projections, benchmarks, timestamp):
    """Guardar resultados completos del business case"""
    try:
        logging.info("Guardando resultados del business case...")
        
        business_case_data = {
            'metadata': {
                'timestamp': timestamp,
                'script': 'paso13c_Business_Case_Completo',
                'version': '1.0',
                'disclaimer': 'DATOS ESTIMADOS - BENCHMARKS INDUSTRIA TELECOM - EJERCICIO DE SIMULACIÓN'
            },
            'current_state': current_state,
            'intervention_scenarios': intervention_scenarios,
            'projections_3_years': projections,
            'benchmarks_used': benchmarks,
            'key_recommendations': {
                'proceed_with_implementation': True,
                'recommended_roi': intervention_scenarios['consolidated']['overall_roi_annual'],
                'payback_months': intervention_scenarios['consolidated']['overall_payback_months'],
                'most_profitable_segment': 'Medio_Riesgo',
                'implementation_approach': 'Phased rollout starting with pilot'
            }
        }
        
        json_file = f'informes/paso13c_business_case_completo_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(business_case_data, f, indent=2, ensure_ascii=False, default=str)
        
        txt_file = f'informes/paso13c_business_case_completo_{timestamp}.txt'
        report_content = generate_business_case_report(current_state, intervention_scenarios, 
                                                     projections, benchmarks, timestamp)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"JSON business case guardado: {json_file}")
        logging.info(f"Informe ejecutivo guardado: {txt_file}")
        
        return {
            'json_file': json_file,
            'txt_file': txt_file
        }
        
    except Exception as e:
        logging.error(f"Error guardando resultados business case: {str(e)}")
        raise

def main():
    """Función principal del Paso 13C - Business Case Completo"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 13C: BUSINESS CASE COMPLETO CON ROI Y PROYECCIONES")
        logger.info("="*80)
        logger.warning("⚠️  UTILIZANDO BENCHMARKS ESTIMADOS DE INDUSTRIA TELECOM")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos de segmentación del Paso 13B
        logger.info("="*50)
        logger.info("CARGANDO DATOS DE SEGMENTACIÓN DEL PASO 13B")
        segmentation_data = load_segmentation_data()
        
        # 3. Definir benchmarks de la industria
        logger.info("="*50)
        logger.info("DEFINIENDO BENCHMARKS DE LA INDUSTRIA TELECOM")
        benchmarks = define_industry_benchmarks()
        
        # 4. Calcular estado financiero actual (baseline)
        logger.info("="*50)
        logger.info("CALCULANDO ESTADO FINANCIERO ACTUAL")
        current_state = calculate_current_state_financials(segmentation_data, benchmarks)
        
        # 5. Calcular escenarios de intervención y ROI
        logger.info("="*50)
        logger.info("CALCULANDO ESCENARIOS DE INTERVENCIÓN Y ROI")
        intervention_scenarios = calculate_intervention_scenarios(segmentation_data, current_state, benchmarks)
        
        # 6. Crear proyecciones a 3 años
        logger.info("="*50)
        logger.info("CREANDO PROYECCIONES FINANCIERAS A 3 AÑOS")
        projections = create_3year_projections(current_state, intervention_scenarios, benchmarks)
        
        # 7. Crear visualizaciones del business case
        logger.info("="*50)
        logger.info("CREANDO VISUALIZACIONES DEL BUSINESS CASE")
        viz_file = create_business_case_visualization(current_state, intervention_scenarios, projections, timestamp)
        
        # 8. Crear dashboard Excel
        logger.info("="*50)
        logger.info("CREANDO DASHBOARD EXCEL")
        excel_file = create_excel_business_case_dashboard(current_state, intervention_scenarios, 
                                                        projections, benchmarks, timestamp)
        
        # 9. Guardar resultados
        logger.info("="*50)
        logger.info("GUARDANDO RESULTADOS DEL BUSINESS CASE")
        output_files = save_business_case_results(current_state, intervention_scenarios, 
                                                projections, benchmarks, timestamp)
        
        # 10. Resumen final
        logger.info("="*80)
        logger.info("PASO 13C COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("")
        
        # Mostrar resultados principales con disclaimer
        consolidated = intervention_scenarios['consolidated']
        summary = projections['summary']
        
        logger.warning("⚠️⚠️⚠️ RESULTADOS BASADOS EN DATOS ESTIMADOS ⚠️⚠️⚠️")
        logger.info("")
        
        logger.info("💰 BUSINESS CASE FINANCIERO:")
        logger.info(f"  • Inversión anual: ${consolidated['total_annual_investment']:,.2f}")
        logger.info(f"  • Revenue salvado: ${consolidated['total_annual_revenue_saved']:,.2f}")
        logger.info(f"  • ROI anual: {consolidated['overall_roi_annual']:.1f}%")
        logger.info(f"  • Payback: {consolidated['overall_payback_months']:.1f} meses")
        logger.info("")
        
        logger.info("📈 PROYECCIONES 3 AÑOS:")
        logger.info(f"  • NPV: ${summary['net_present_value']:,.2f}")
        logger.info(f"  • ROI acumulado: {summary['final_cumulative_roi']:.1f}%")
        logger.info(f"  • Break-even: Año {summary['break_even_year'] or 'N/A'}")
        logger.info("")
        
        logger.info("🎯 ROI POR SEGMENTO (ESTIMADO):")
        for segment, data in intervention_scenarios.items():
            if segment != 'consolidated':
                logger.info(f"  • {segment}: {data['roi_percentage']:.1f}% ROI")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • JSON business case: {output_files['json_file']}")
        logger.info(f"  • Informe ejecutivo: {output_files['txt_file']}")
        if excel_file:
            logger.info(f"  • Dashboard Excel: {excel_file}")
        if viz_file:
            logger.info(f"  • Visualizaciones: {viz_file}")
        logger.info("")
        
        logger.warning("⚠️ IMPORTANTE: VALIDAR CON DATOS REALES DE LA EMPRESA")
        logger.warning("Los benchmarks utilizados son estimaciones de la industria telecom")
        logger.warning("Para implementación real, reemplazar con datos financieros específicos")
        logger.info("")
        
        logger.info("📋 LISTO PARA PRÓXIMOS SCRIPTS:")
        logger.info("  • paso13d: Roadmap detallado de implementación")
        logger.info("  • paso13e: Outputs ejecutivos para stakeholders")
        logger.info("  • paso13f: Informe final estratégico consolidado")
        logger.info("="*80)
        
        return {
            'current_state': current_state,
            'intervention_scenarios': intervention_scenarios,
            'projections': projections,
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