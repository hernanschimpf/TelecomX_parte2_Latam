"""
================================================================================
TELECOMX - PASO 9: SEPARACIÓN DE DATOS PARA MODELADO
================================================================================
Descripción: División estratificada del dataset en conjuntos de entrenamiento,
             validación y prueba con validaciones exhaustivas y análisis de
             representatividad.

Funcionalidades:
- División estratificada Train (60%) / Validation (20%) / Test (20%)
- Validaciones de balance de clases y distribuciones
- Análisis por segmentos del Paso 8
- Generación de datasets separados
- Verificaciones de integridad y representatividad

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
from sklearn.model_selection import train_test_split
from scipy import stats
import json
warnings.filterwarnings('ignore')

# Configuración global
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def setup_logging():
    """Configurar sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/paso9_separacion_datos.log', mode='a', encoding='utf-8'),
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

def load_data():
    """Cargar el dataset optimizado del Paso 7"""
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
        
        # Verificar variable objetivo
        target_var = 'Abandono_Cliente'
        if df[target_var].nunique() != 2:
            raise ValueError(f"Variable objetivo debe ser binaria. Valores únicos: {df[target_var].unique()}")
        
        logging.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        logging.info(f"Balance de clases original: {df[target_var].value_counts().to_dict()}")
        
        return df, input_file
        
    except Exception as e:
        logging.error(f"Error al cargar el dataset: {str(e)}")
        raise

def create_stratified_split(df, target_col='Abandono_Cliente', random_state=RANDOM_SEED):
    """Crear división estratificada en Train/Validation/Test"""
    logging.info("Realizando división estratificada de datos...")
    
    # Preparar datos
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Primera división: 80% (temp) / 20% (test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    # Segunda división: 60% (train) / 20% (val) del total
    # 60/80 = 0.75 del conjunto temporal
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=random_state
    )
    
    # Crear datasets completos con variable objetivo
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Agregar identificador de conjunto
    train_df['Conjunto'] = 'Train'
    val_df['Conjunto'] = 'Validation'
    test_df['Conjunto'] = 'Test'
    
    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'total_size': len(df),
        'train_pct': len(train_df) / len(df) * 100,
        'val_pct': len(val_df) / len(df) * 100,
        'test_pct': len(test_df) / len(df) * 100
    }
    
    logging.info("División estratificada completada:")
    logging.info(f"  Train: {split_info['train_size']:,} ({split_info['train_pct']:.1f}%)")
    logging.info(f"  Validation: {split_info['val_size']:,} ({split_info['val_pct']:.1f}%)")
    logging.info(f"  Test: {split_info['test_size']:,} ({split_info['test_pct']:.1f}%)")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'split_info': split_info
    }

def validate_stratification(datasets, target_col='Abandono_Cliente'):
    """Validar que la estratificación mantenga el balance de clases"""
    logging.info("Validando estratificación de clases...")
    
    validation_results = {}
    original_distribution = None
    
    for name, df in datasets.items():
        if name == 'split_info':
            continue
            
        # Calcular distribución de clases
        class_counts = df[target_col].value_counts().sort_index()
        class_percentages = (class_counts / len(df) * 100).round(2)
        
        validation_results[name] = {
            'counts': class_counts.to_dict(),
            'percentages': class_percentages.to_dict(),
            'total_samples': len(df)
        }
        
        # Guardar distribución original para comparación
        if original_distribution is None:
            original_distribution = class_percentages
        
        logging.info(f"  {name.title()}:")
        logging.info(f"    Total: {len(df):,} muestras")
        for class_val, count in class_counts.items():
            status = "No Churn" if class_val == 0 else "Churn"
            pct = class_percentages[class_val]
            logging.info(f"    {status}: {count:,} ({pct:.2f}%)")
    
    # Verificar diferencias máximas entre conjuntos
    max_diff_0 = max([abs(validation_results[name]['percentages'][0] - original_distribution[0]) 
                      for name in validation_results.keys()])
    max_diff_1 = max([abs(validation_results[name]['percentages'][1] - original_distribution[1]) 
                      for name in validation_results.keys()])
    
    stratification_quality = "EXCELENTE" if max(max_diff_0, max_diff_1) < 1.0 else \
                            "BUENA" if max(max_diff_0, max_diff_1) < 2.0 else \
                            "ACEPTABLE" if max(max_diff_0, max_diff_1) < 3.0 else "PROBLEMATICA"
    
    logging.info(f"Calidad de estratificación: {stratification_quality}")
    logging.info(f"Diferencia máxima en distribución: {max(max_diff_0, max_diff_1):.2f}%")
    
    return {
        'validation_results': validation_results,
        'max_difference': max(max_diff_0, max_diff_1),
        'quality': stratification_quality
    }

def validate_segment_distributions(datasets, target_col='Abandono_Cliente'):
    """Validar distribuciones por segmentos del Paso 8"""
    logging.info("Validando distribuciones por segmentos...")
    
    segment_validation = {}
    
    for name, df in datasets.items():
        if name == 'split_info':
            continue
        
        # Recrear segmentaciones del Paso 8
        # Segmentación por tiempo
        df_copy = df.copy()
        df_copy['Segmento_Tiempo'] = pd.cut(df_copy['Meses_Cliente'], 
                                          bins=[0, 12, 36, float('inf')],
                                          labels=['Nuevos (0-12m)', 'Intermedios (12-36m)', 'Veteranos (36m+)'])
        
        # Segmentación por gasto
        terciles = df_copy['Cargo_Total'].quantile([0, 0.33, 0.66, 1.0])
        df_copy['Segmento_Gasto'] = pd.cut(df_copy['Cargo_Total'], 
                                         bins=terciles,
                                         labels=['Bajo Gasto', 'Gasto Medio', 'Alto Gasto'],
                                         include_lowest=True)
        
        # Analizar distribuciones por segmento de tiempo
        time_segments = df_copy.groupby('Segmento_Tiempo')[target_col].agg(['count', 'mean']).round(3)
        time_segments.columns = ['Count', 'Churn_Rate']
        
        # Analizar distribuciones por segmento de gasto  
        spending_segments = df_copy.groupby('Segmento_Gasto')[target_col].agg(['count', 'mean']).round(3)
        spending_segments.columns = ['Count', 'Churn_Rate']
        
        segment_validation[name] = {
            'time_segments': time_segments.to_dict(),
            'spending_segments': spending_segments.to_dict()
        }
    
    # Comparar consistencia entre conjuntos
    consistency_check = {}
    
    # Para segmentos de tiempo
    for segment in ['Nuevos (0-12m)', 'Intermedios (12-36m)', 'Veteranos (36m+)']:
        rates = []
        for dataset_name in ['train', 'validation', 'test']:
            if dataset_name in segment_validation:
                try:
                    rate = segment_validation[dataset_name]['time_segments']['Churn_Rate'][segment]
                    rates.append(rate)
                except KeyError:
                    continue
        
        if rates:
            consistency_check[f'time_{segment}'] = {
                'rates': rates,
                'std': np.std(rates),
                'range': max(rates) - min(rates) if len(rates) > 1 else 0
            }
    
    # Para segmentos de gasto
    for segment in ['Bajo Gasto', 'Gasto Medio', 'Alto Gasto']:
        rates = []
        for dataset_name in ['train', 'validation', 'test']:
            if dataset_name in segment_validation:
                try:
                    rate = segment_validation[dataset_name]['spending_segments']['Churn_Rate'][segment]
                    rates.append(rate)
                except KeyError:
                    continue
        
        if rates:
            consistency_check[f'spending_{segment}'] = {
                'rates': rates,
                'std': np.std(rates),
                'range': max(rates) - min(rates) if len(rates) > 1 else 0
            }
    
    logging.info("Consistencia de segmentos entre conjuntos:")
    avg_std = np.mean([check['std'] for check in consistency_check.values()])
    max_range = max([check['range'] for check in consistency_check.values()])
    
    segment_quality = "EXCELENTE" if avg_std < 0.02 and max_range < 0.05 else \
                     "BUENA" if avg_std < 0.05 and max_range < 0.10 else \
                     "ACEPTABLE"
    
    logging.info(f"  Calidad de representatividad: {segment_quality}")
    logging.info(f"  Desviación estándar promedio: {avg_std:.4f}")
    logging.info(f"  Rango máximo entre conjuntos: {max_range:.4f}")
    
    return {
        'segment_validation': segment_validation,
        'consistency_check': consistency_check,
        'quality': segment_quality,
        'avg_std': avg_std,
        'max_range': max_range
    }

def perform_statistical_tests(datasets, target_col='Abandono_Cliente'):
    """Realizar tests estadísticos para validar representatividad"""
    logging.info("Realizando tests estadísticos de representatividad...")
    
    statistical_results = {}
    
    # Preparar datos para comparación
    train_df = datasets['train']
    val_df = datasets['validation'] 
    test_df = datasets['test']
    
    # Variables numéricas para testear
    numeric_vars = ['Meses_Cliente', 'Cargo_Total']
    
    for var in numeric_vars:
        # Test de Kolmogorov-Smirnov entre conjuntos
        train_data = train_df[var]
        val_data = val_df[var]
        test_data = test_df[var]
        
        # KS test entre train y validation
        ks_stat_tv, p_value_tv = stats.ks_2samp(train_data, val_data)
        
        # KS test entre train y test
        ks_stat_tt, p_value_tt = stats.ks_2samp(train_data, test_data)
        
        # KS test entre validation y test
        ks_stat_vt, p_value_vt = stats.ks_2samp(val_data, test_data)
        
        statistical_results[var] = {
            'train_vs_val': {
                'ks_statistic': ks_stat_tv,
                'p_value': p_value_tv,
                'similar': p_value_tv > 0.05
            },
            'train_vs_test': {
                'ks_statistic': ks_stat_tt,
                'p_value': p_value_tt,
                'similar': p_value_tt > 0.05
            },
            'val_vs_test': {
                'ks_statistic': ks_stat_vt,
                'p_value': p_value_vt,
                'similar': p_value_vt > 0.05
            }
        }
        
        # Log resultados
        similar_tv = "similares" if p_value_tv > 0.05 else "diferentes"
        similar_tt = "similares" if p_value_tt > 0.05 else "diferentes"
        similar_vt = "similares" if p_value_vt > 0.05 else "diferentes"
        
        logging.info(f"  {var}:")
        logging.info(f"    Train vs Validation: {similar_tv} (p={p_value_tv:.4f})")
        logging.info(f"    Train vs Test: {similar_tt} (p={p_value_tt:.4f})")
        logging.info(f"    Validation vs Test: {similar_vt} (p={p_value_vt:.4f})")
    
    # Test Chi-cuadrado para variable objetivo entre conjuntos
    train_churn = train_df[target_col].value_counts().sort_index()
    val_churn = val_df[target_col].value_counts().sort_index()
    test_churn = test_df[target_col].value_counts().sort_index()
    
    # Crear tabla de contingencia
    contingency_table = np.array([
        train_churn.values,
        val_churn.values, 
        test_churn.values
    ])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    statistical_results['target_distribution'] = {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'similar': p_value > 0.05
    }
    
    similar_target = "similares" if p_value > 0.05 else "diferentes"
    logging.info(f"  Distribución de variable objetivo: {similar_target} (p={p_value:.4f})")
    
    return statistical_results

def generate_visualizations(datasets, timestamp):
    """Generar visualizaciones de la separación de datos"""
    logging.info("Generando visualizaciones...")
    
    try:
        plt.style.use('default')
        
        # 1. Distribución de clases por conjunto
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico de barras de distribución general
        sizes = [datasets['split_info']['train_size'], 
                datasets['split_info']['val_size'], 
                datasets['split_info']['test_size']]
        labels = ['Train (60%)', 'Validation (20%)', 'Test (20%)']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = ax1.bar(labels, sizes, color=colors, alpha=0.8)
        ax1.set_title('Distribución de Muestras por Conjunto', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Número de Muestras')
        
        # Agregar valores en las barras
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                    f'{size:,}', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de distribución de churn por conjunto
        churn_data = {}
        for name, df in datasets.items():
            if name != 'split_info':
                churn_counts = df['Abandono_Cliente'].value_counts().sort_index()
                churn_data[name.title()] = churn_counts
        
        churn_df = pd.DataFrame(churn_data)
        churn_df.index = ['No Churn', 'Churn']
        
        churn_df.plot(kind='bar', ax=ax2, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax2.set_title('Distribución de Clases por Conjunto', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Número de Muestras')
        ax2.legend(title='Conjunto')
        ax2.tick_params(axis='x', rotation=0)
        
        # Gráfico de porcentajes de churn por conjunto
        churn_pcts = {}
        for name, df in datasets.items():
            if name != 'split_info':
                churn_pct = df['Abandono_Cliente'].mean() * 100
                churn_pcts[name.title()] = churn_pct
        
        sets = list(churn_pcts.keys())
        pcts = list(churn_pcts.values())
        
        bars = ax3.bar(sets, pcts, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        ax3.set_title('Porcentaje de Churn por Conjunto', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Porcentaje de Churn (%)')
        
        # Línea de promedio general
        overall_avg = np.mean(pcts)
        ax3.axhline(y=overall_avg, color='red', linestyle='--', 
                   label=f'Promedio: {overall_avg:.2f}%')
        ax3.legend()
        
        # Valores en las barras
        for bar, pct in zip(bars, pcts):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                    f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de distribución de variables clave
        # Boxplot de Meses_Cliente por conjunto
        data_to_plot = []
        labels_plot = []
        for name, df in datasets.items():
            if name != 'split_info':
                data_to_plot.append(df['Meses_Cliente'])
                labels_plot.append(name.title())
        
        ax4.boxplot(data_to_plot, labels=labels_plot)
        ax4.set_title('Distribución de Meses Cliente por Conjunto', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Meses como Cliente')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso9_distribucion_conjuntos_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Análisis de representatividad por segmentos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Preparar datos por segmentos
        segment_data = {}
        for name, df in datasets.items():
            if name != 'split_info':
                df_copy = df.copy()
                
                # Segmentos de tiempo
                df_copy['Segmento_Tiempo'] = pd.cut(df_copy['Meses_Cliente'], 
                                                  bins=[0, 12, 36, float('inf')],
                                                  labels=['Nuevos', 'Intermedios', 'Veteranos'])
                
                time_churn = df_copy.groupby('Segmento_Tiempo')['Abandono_Cliente'].mean()
                segment_data[f'{name}_tiempo'] = time_churn
        
        # Gráfico de tasas de churn por segmento de tiempo
        time_segments = ['Nuevos', 'Intermedios', 'Veteranos']
        train_time_rates = [segment_data.get('train_tiempo', {}).get(seg, 0) for seg in time_segments]
        val_time_rates = [segment_data.get('validation_tiempo', {}).get(seg, 0) for seg in time_segments]
        test_time_rates = [segment_data.get('test_tiempo', {}).get(seg, 0) for seg in time_segments]
        
        x = np.arange(len(time_segments))
        width = 0.25
        
        ax1.bar(x - width, train_time_rates, width, label='Train', color='skyblue', alpha=0.8)
        ax1.bar(x, val_time_rates, width, label='Validation', color='lightgreen', alpha=0.8)
        ax1.bar(x + width, test_time_rates, width, label='Test', color='lightcoral', alpha=0.8)
        
        ax1.set_title('Tasa de Churn por Segmento de Tiempo', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Segmento de Tiempo')
        ax1.set_ylabel('Tasa de Churn')
        ax1.set_xticks(x)
        ax1.set_xticklabels(time_segments)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribución de Cargo_Total por conjunto
        ax2.hist([datasets['train']['Cargo_Total'], 
                 datasets['validation']['Cargo_Total'], 
                 datasets['test']['Cargo_Total']], 
                bins=30, alpha=0.7, label=['Train', 'Validation', 'Test'],
                color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Distribución de Cargo Total por Conjunto', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Cargo Total ($)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Scatter plot de Meses vs Cargo por conjunto (MEJORADO)
        # Usar muestreo para evitar overplotting
        sample_size = 800  # Muestra por conjunto para visualización clara
        
        ax3.set_title('Relación Meses vs Cargo por Conjunto (Muestra)', fontsize=14, fontweight='bold')
        
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        alphas = [0.6, 0.6, 0.6]
        
        for i, (name, df) in enumerate([(n, d) for n, d in datasets.items() if n != 'split_info']):
            # Tomar muestra estratificada para visualización
            if len(df) > sample_size:
                # Muestra estratificada por churn
                df_sample = df.groupby('Abandono_Cliente').apply(
                    lambda x: x.sample(min(len(x), sample_size//2), random_state=42)
                ).reset_index(drop=True)
            else:
                df_sample = df
            
            # Separar por churn para mejor visualización
            no_churn = df_sample[df_sample['Abandono_Cliente'] == 0]
            churn = df_sample[df_sample['Abandono_Cliente'] == 1]
            
            # Plot no churn
            ax3.scatter(no_churn['Meses_Cliente'], no_churn['Cargo_Total'], 
                       alpha=alphas[i], c=colors[i], s=15, 
                       marker='o', label=f'{name.title()} - No Churn')
            
            # Plot churn con marcador diferente
            ax3.scatter(churn['Meses_Cliente'], churn['Cargo_Total'], 
                       alpha=alphas[i], c=colors[i], s=15, 
                       marker='^', label=f'{name.title()} - Churn')
        
        ax3.set_xlabel('Meses como Cliente')
        ax3.set_ylabel('Cargo Total ($)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Agregar líneas de referencia
        ax3.axvline(df['Meses_Cliente'].mean(), color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax3.axhline(df['Cargo_Total'].mean(), color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Agregar texto explicativo
        ax3.text(0.02, 0.98, f'Muestra: ~{sample_size} por conjunto\n○ No Churn  △ Churn', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
        
        # Matriz de correlación por conjunto
        correlations = {}
        for name, df in datasets.items():
            if name != 'split_info':
                corr = df[['Meses_Cliente', 'Cargo_Total', 'Abandono_Cliente']].corr()
                correlations[name] = corr.loc['Abandono_Cliente', ['Meses_Cliente', 'Cargo_Total']]
        
        corr_df = pd.DataFrame(correlations).T
        
        # Heatmap de correlaciones
        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, ax=ax4,
                   cbar_kws={'label': 'Correlación con Churn'})
        ax4.set_title('Correlaciones con Churn por Conjunto', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Variables')
        ax4.set_ylabel('Conjuntos')
        
        plt.tight_layout()
        plt.savefig(f'graficos/paso9_analisis_representatividad_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Visualizaciones generadas exitosamente")
        return True
        
    except Exception as e:
        logging.error(f"Error al generar visualizaciones: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def save_datasets(datasets, timestamp):
    """Guardar datasets separados en archivos individuales"""
    logging.info("Guardando datasets separados...")
    
    try:
        saved_files = {}
        
        for name, df in datasets.items():
            if name != 'split_info':
                # Eliminar columna de conjunto ya que cada archivo es independiente
                df_to_save = df.drop(columns=['Conjunto']) if 'Conjunto' in df.columns else df
                
                filename = f'datos/telecomx_{name}_dataset_{timestamp}.csv'
                df_to_save.to_csv(filename, index=False, encoding='utf-8-sig')
                saved_files[name] = filename
                
                logging.info(f"  {name.title()}: {filename} ({len(df_to_save):,} filas)")
        
        # Guardar también un archivo combinado con identificador de conjunto
        combined_df = pd.concat([datasets['train'], datasets['validation'], datasets['test']], 
                               ignore_index=True)
        combined_filename = f'datos/telecomx_datasets_combinados_{timestamp}.csv'
        combined_df.to_csv(combined_filename, index=False, encoding='utf-8-sig')
        saved_files['combined'] = combined_filename
        
        logging.info(f"  Combinado: {combined_filename} ({len(combined_df):,} filas)")
        logging.info("Datasets guardados exitosamente")
        
        return saved_files
        
    except Exception as e:
        logging.error(f"Error al guardar datasets: {str(e)}")
        raise

def generate_comprehensive_report(datasets, stratification_validation, segment_validation, 
                                 statistical_results, saved_files, timestamp):
    """Generar informe completo de la separación de datos"""
    
    report = f"""
================================================================================
TELECOMX - INFORME PASO 9: SEPARACIÓN DE DATOS PARA MODELADO
================================================================================
Fecha y Hora: {timestamp}
Paso: 9 - Separación de Datos
Semilla Aleatoria: {RANDOM_SEED}

================================================================================
RESUMEN EJECUTIVO
================================================================================
• División Realizada: Train (60%) / Validation (20%) / Test (20%)
• Total de Registros: {datasets['split_info']['total_size']:,}
• Estratificación: {'EXITOSA' if stratification_validation['quality'] in ['EXCELENTE', 'BUENA'] else 'PROBLEMATICA'}
• Representatividad por Segmentos: {segment_validation['quality']}
• Archivos Generados: {len(saved_files)}

================================================================================
DISTRIBUCIÓN DE CONJUNTOS
================================================================================

📊 TAMAÑOS DE CONJUNTOS:
• Train: {datasets['split_info']['train_size']:,} registros ({datasets['split_info']['train_pct']:.1f}%)
• Validation: {datasets['split_info']['val_size']:,} registros ({datasets['split_info']['val_pct']:.1f}%)
• Test: {datasets['split_info']['test_size']:,} registros ({datasets['split_info']['test_pct']:.1f}%)
• Total: {datasets['split_info']['total_size']:,} registros

⚖️ CONFIGURACIÓN UTILIZADA:
• Método: train_test_split con estratificación
• Semilla aleatoria: {RANDOM_SEED}
• Variable de estratificación: Abandono_Cliente
• División: 60% train, 20% validation, 20% test

================================================================================
VALIDACIÓN DE ESTRATIFICACIÓN
================================================================================

🎯 CALIDAD DE ESTRATIFICACIÓN: {stratification_validation['quality']}
• Diferencia máxima entre conjuntos: {stratification_validation['max_difference']:.3f}%

📊 DISTRIBUCIÓN DE CLASES POR CONJUNTO:
"""
    
    # Detalles de distribución por conjunto
    for name, result in stratification_validation['validation_results'].items():
        no_churn_pct = result['percentages'][0]
        churn_pct = result['percentages'][1]
        total = result['total_samples']
        
        report += f"""
{name.upper()}:
   • Total: {total:,} registros
   • No Churn: {result['counts'][0]:,} ({no_churn_pct:.2f}%)
   • Churn: {result['counts'][1]:,} ({churn_pct:.2f}%)"""

    report += f"""

✅ EVALUACIÓN DE BALANCE:
• Estratificación: {'Correcta' if stratification_validation['quality'] != 'PROBLEMATICA' else 'Requiere atención'}
• Variación entre conjuntos: {'Mínima' if stratification_validation['max_difference'] < 1.0 else 'Aceptable' if stratification_validation['max_difference'] < 2.0 else 'Alta'}
• Recomendación: {'Proceder con modelado' if stratification_validation['quality'] in ['EXCELENTE', 'BUENA'] else 'Revisar división'}

================================================================================
VALIDACIÓN POR SEGMENTOS DEL PASO 8
================================================================================

🎯 CALIDAD DE REPRESENTATIVIDAD: {segment_validation['quality']}
• Desviación estándar promedio: {segment_validation['avg_std']:.4f}
• Rango máximo entre conjuntos: {segment_validation['max_range']:.4f}

📊 ANÁLISIS POR SEGMENTOS DE TIEMPO:
"""
    
    # Mostrar consistencia por segmentos de tiempo
    time_segments = ['Nuevos (0-12m)', 'Intermedios (12-36m)', 'Veteranos (36m+)']
    for segment in time_segments:
        segment_key = f'time_{segment}'
        if segment_key in segment_validation['consistency_check']:
            rates = segment_validation['consistency_check'][segment_key]['rates']
            std = segment_validation['consistency_check'][segment_key]['std']
            range_val = segment_validation['consistency_check'][segment_key]['range']
            
            report += f"""
{segment}:
   • Tasas de churn: {' | '.join([f'{r:.3f}' for r in rates])} (T|V|Te)
   • Desviación estándar: {std:.4f}
   • Rango: {range_val:.4f}"""

    report += f"""

📊 ANÁLISIS POR SEGMENTOS DE GASTO:
"""
    
    # Mostrar consistencia por segmentos de gasto
    spending_segments = ['Bajo Gasto', 'Gasto Medio', 'Alto Gasto']
    for segment in spending_segments:
        segment_key = f'spending_{segment}'
        if segment_key in segment_validation['consistency_check']:
            rates = segment_validation['consistency_check'][segment_key]['rates']
            std = segment_validation['consistency_check'][segment_key]['std']
            range_val = segment_validation['consistency_check'][segment_key]['range']
            
            report += f"""
{segment}:
   • Tasas de churn: {' | '.join([f'{r:.3f}' for r in rates])} (T|V|Te)
   • Desviación estándar: {std:.4f}
   • Rango: {range_val:.4f}"""

    report += f"""

================================================================================
TESTS ESTADÍSTICOS DE REPRESENTATIVIDAD
================================================================================

🔬 TESTS DE KOLMOGOROV-SMIRNOV:
"""
    
    # Tests estadísticos para variables numéricas
    for var, results in statistical_results.items():
        if var != 'target_distribution':
            report += f"""
{var}:
   • Train vs Validation: {'Similares' if results['train_vs_val']['similar'] else 'Diferentes'} (p={results['train_vs_val']['p_value']:.4f})
   • Train vs Test: {'Similares' if results['train_vs_test']['similar'] else 'Diferentes'} (p={results['train_vs_test']['p_value']:.4f})
   • Validation vs Test: {'Similares' if results['val_vs_test']['similar'] else 'Diferentes'} (p={results['val_vs_test']['p_value']:.4f})"""

    # Test para distribución de variable objetivo
    target_similar = 'Similares' if statistical_results['target_distribution']['similar'] else 'Diferentes'
    target_p = statistical_results['target_distribution']['p_value']
    
    report += f"""

🎯 TEST CHI-CUADRADO PARA VARIABLE OBJETIVO:
• Distribución entre conjuntos: {target_similar} (p={target_p:.4f})
• Interpretación: {'Las proporciones de churn son consistentes entre conjuntos' if statistical_results['target_distribution']['similar'] else 'Hay diferencias significativas en las proporciones'}

================================================================================
ANÁLISIS DE VARIABLES CLAVE
================================================================================

📊 ESTADÍSTICAS DESCRIPTIVAS POR CONJUNTO:

MESES_CLIENTE:
"""
    
    # Estadísticas por conjunto para Meses_Cliente
    for name, df in datasets.items():
        if name != 'split_info':
            stats = df['Meses_Cliente'].describe()
            report += f"""
{name.upper()}:
   • Promedio: {stats['mean']:.1f} meses
   • Mediana: {stats['50%']:.1f} meses
   • Desviación: {stats['std']:.1f} meses
   • Rango: {stats['min']:.0f} - {stats['max']:.0f} meses"""

    report += f"""

CARGO_TOTAL:
"""
    
    # Estadísticas por conjunto para Cargo_Total
    for name, df in datasets.items():
        if name != 'split_info':
            stats = df['Cargo_Total'].describe()
            report += f"""
{name.upper()}:
   • Promedio: ${stats['mean']:.2f}
   • Mediana: ${stats['50%']:.2f}
   • Desviación: ${stats['std']:.2f}
   • Rango: ${stats['min']:.2f} - ${stats['max']:.2f}"""

    report += f"""

📈 CORRELACIONES CON VARIABLE OBJETIVO POR CONJUNTO:
"""
    
    # Correlaciones por conjunto
    for name, df in datasets.items():
        if name != 'split_info':
            corr_meses = df['Meses_Cliente'].corr(df['Abandono_Cliente'])
            corr_cargo = df['Cargo_Total'].corr(df['Abandono_Cliente'])
            
            report += f"""
{name.upper()}:
   • Meses_Cliente: {corr_meses:+.4f}
   • Cargo_Total: {corr_cargo:+.4f}"""

    report += f"""

================================================================================
EVALUACIÓN GENERAL DE LA DIVISIÓN
================================================================================

✅ CRITERIOS DE CALIDAD EVALUADOS:

1. ESTRATIFICACIÓN DE CLASES:
   • Estado: {'✅ EXITOSA' if stratification_validation['quality'] != 'PROBLEMATICA' else '❌ PROBLEMÁTICA'}
   • Diferencia máxima: {stratification_validation['max_difference']:.3f}% (objetivo: <2.0%)
   • Calidad: {stratification_validation['quality']}

2. REPRESENTATIVIDAD POR SEGMENTOS:
   • Estado: {'✅ EXITOSA' if segment_validation['quality'] != 'PROBLEMÁTICA' else '❌ PROBLEMÁTICA'}
   • Desviación promedio: {segment_validation['avg_std']:.4f} (objetivo: <0.05)
   • Calidad: {segment_validation['quality']}

3. SIMILARIDAD ESTADÍSTICA:
   • Variables numéricas: {'✅ SIMILARES' if all([all([r['train_vs_val']['similar'], r['train_vs_test']['similar'], r['val_vs_test']['similar']]) for k, r in statistical_results.items() if k != 'target_distribution']) else '⚠️ ALGUNAS DIFERENCIAS'}
   • Variable objetivo: {'✅ SIMILAR' if statistical_results['target_distribution']['similar'] else '⚠️ DIFERENCIAS'}

🏆 EVALUACIÓN GENERAL:
"""
    
    # Evaluación general
    criteria_passed = 0
    total_criteria = 3
    
    if stratification_validation['quality'] != 'PROBLEMÁTICA':
        criteria_passed += 1
    if segment_validation['quality'] != 'PROBLEMÁTICA':
        criteria_passed += 1
    if statistical_results['target_distribution']['similar']:
        criteria_passed += 1
    
    overall_quality = "EXCELENTE" if criteria_passed == 3 else \
                     "BUENA" if criteria_passed == 2 else \
                     "ACEPTABLE" if criteria_passed == 1 else "PROBLEMÁTICA"
    
    report += f"""
• Criterios cumplidos: {criteria_passed}/{total_criteria}
• Calidad general: {overall_quality}
• Recomendación: {'Proceder con entrenamiento de modelos' if criteria_passed >= 2 else 'Considerar re-división de datos'}

================================================================================
ARCHIVOS GENERADOS
================================================================================

📊 DATASETS INDIVIDUALES:
"""
    
    # Listar archivos generados
    for name, filepath in saved_files.items():
        if name != 'combined':
            dataset_size = len(datasets[name]) if name in datasets else 'N/A'
            report += f"""
• {name.title()}: {filepath}
  - Registros: {dataset_size:,}
  - Uso: {'Entrenamiento de modelos' if name == 'train' else 'Validación de hiperparámetros' if name == 'validation' else 'Evaluación final'}"""

    report += f"""

📄 ARCHIVO COMBINADO:
• Combinado: {saved_files.get('combined', 'N/A')}
  - Registros: {sum([len(df) for name, df in datasets.items() if name != 'split_info']):,}
  - Uso: Análisis conjunto con columna identificadora

📊 VISUALIZACIONES:
• Distribución conjuntos: graficos/paso9_distribucion_conjuntos_{timestamp}.png
• Análisis representatividad: graficos/paso9_analisis_representatividad_{timestamp}.png

📄 DOCUMENTACIÓN:
• Informe completo: informes/paso9_separacion_datos_informe_{timestamp}.txt
• Log del proceso: logs/paso9_separacion_datos.log

================================================================================
RECOMENDACIONES PARA MODELADO
================================================================================

🎯 ESTRATEGIA DE MODELADO SUGERIDA:

1. ENTRENAMIENTO:
   • Usar dataset: datos/telecomx_train_dataset_{timestamp}.csv
   • Registros: {datasets['train'].shape[0]:,}
   • Aplicar class weighting del Paso 4

2. VALIDACIÓN:
   • Usar dataset: datos/telecomx_validation_dataset_{timestamp}.csv
   • Registros: {datasets['validation'].shape[0]:,}
   • Para ajuste de hiperparámetros y early stopping

3. EVALUACIÓN FINAL:
   • Usar dataset: datos/telecomx_test_dataset_{timestamp}.csv
   • Registros: {datasets['test'].shape[0]:,}
   • Solo para evaluación final del modelo seleccionado

⚖️ MÉTRICAS RECOMENDADAS:
• Primarias: F1-Score, AUC-PR (por desbalance de clases)
• Secundarias: Precision, Recall, AUC-ROC
• Evitar: Accuracy como métrica principal

🔄 VALIDACIÓN CRUZADA:
• Aplicar sobre conjunto de entrenamiento únicamente
• Usar estratificación: StratifiedKFold
• Folds recomendados: 5

================================================================================
PRÓXIMO PASO RECOMENDADO
================================================================================

Paso 10: Entrenamiento de Modelos Predictivos
• Cargar datasets desde carpeta datos/
• Implementar algoritmos: Random Forest, XGBoost
• Aplicar configuraciones de class weighting del Paso 4
• Usar insights del Paso 8 para feature engineering adicional
• Evaluar con métricas especializadas para datos desbalanceados

================================================================================
CONSIDERACIONES TÉCNICAS
================================================================================

🔧 CONFIGURACIÓN REPRODUCIBLE:
• Semilla aleatoria fija: {RANDOM_SEED}
• Estratificación aplicada en todas las divisiones
• Tests estadísticos con nivel de confianza 95%

✅ VALIDACIONES APLICADAS:
• Kolmogorov-Smirnov para variables numéricas
• Chi-cuadrado para variable objetivo
• Análisis de consistencia por segmentos
• Verificación de balance de clases

📊 CONTROL DE CALIDAD:
• {'✅' if overall_quality in ['EXCELENTE', 'BUENA'] else '⚠️'} División cumple estándares de calidad
• {'✅' if stratification_validation['max_difference'] < 2.0 else '⚠️'} Estratificación dentro de tolerancias
• {'✅' if segment_validation['avg_std'] < 0.05 else '⚠️'} Representatividad por segmentos adecuada

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(datasets, report_content, timestamp):
    """Guardar informe y archivos de configuración"""
    try:
        # Guardar informe
        report_file = f'informes/paso9_separacion_datos_informe_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Informe guardado: {report_file}")
        
        # Guardar configuración de la división
        split_config = {
            'timestamp': timestamp,
            'random_seed': RANDOM_SEED,
            'split_ratios': {
                'train': 0.6,
                'validation': 0.2,
                'test': 0.2
            },
            'stratification_variable': 'Abandono_Cliente',
            'total_samples': datasets['split_info']['total_size'],
            'split_sizes': {
                'train': datasets['split_info']['train_size'],
                'validation': datasets['split_info']['val_size'],
                'test': datasets['split_info']['test_size']
            }
        }
        
        config_file = f'informes/paso9_configuracion_division_{timestamp}.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(split_config, f, indent=2, ensure_ascii=False)
        logging.info(f"Configuración guardada: {config_file}")
        
        return {
            'report_file': report_file,
            'config_file': config_file
        }
        
    except Exception as e:
        logging.error(f"Error al guardar archivos: {str(e)}")
        raise

def main():
    """Función principal del Paso 9"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 9: SEPARACIÓN DE DATOS PARA MODELADO")
        logger.info("="*80)
        logger.info(f"Configuración: Train (60%) / Validation (20%) / Test (20%)")
        logger.info(f"Semilla aleatoria: {RANDOM_SEED}")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos del Paso 7
        df, input_file = load_data()
        logger.info(f"Dataset cargado desde: {input_file}")
        
        # 3. Realizar división estratificada
        datasets = create_stratified_split(df)
        
        # 4. Validar estratificación
        stratification_validation = validate_stratification(datasets)
        
        # 5. Validar representatividad por segmentos
        segment_validation = validate_segment_distributions(datasets)
        
        # 6. Realizar tests estadísticos
        statistical_results = perform_statistical_tests(datasets)
        
        # 7. Generar visualizaciones
        viz_success = generate_visualizations(datasets, timestamp)
        
        # 8. Guardar datasets separados
        saved_files = save_datasets(datasets, timestamp)
        
        # 9. Generar informe completo
        report_content = generate_comprehensive_report(
            datasets, stratification_validation, segment_validation,
            statistical_results, saved_files, timestamp
        )
        
        # 10. Guardar archivos finales
        output_files = save_files(datasets, report_content, timestamp)
        
        # 11. Resumen final
        logger.info("="*80)
        logger.info("PASO 9 COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("RESUMEN DE RESULTADOS:")
        logger.info(f"  • Total de registros procesados: {datasets['split_info']['total_size']:,}")
        logger.info(f"  • Train: {datasets['split_info']['train_size']:,} ({datasets['split_info']['train_pct']:.1f}%)")
        logger.info(f"  • Validation: {datasets['split_info']['val_size']:,} ({datasets['split_info']['val_pct']:.1f}%)")
        logger.info(f"  • Test: {datasets['split_info']['test_size']:,} ({datasets['split_info']['test_pct']:.1f}%)")
        logger.info("")
        
        # Evaluación de calidad
        strat_quality = stratification_validation['quality']
        segment_quality = segment_validation['quality']
        target_similar = statistical_results['target_distribution']['similar']
        
        logger.info("VALIDACIÓN DE CALIDAD:")
        logger.info(f"  • Estratificación: {strat_quality}")
        logger.info(f"  • Representatividad por segmentos: {segment_quality}")
        logger.info(f"  • Distribución objetivo consistente: {'SÍ' if target_similar else 'NO'}")
        logger.info("")
        
        logger.info("ARCHIVOS GENERADOS:")
        logger.info(f"  • Datasets individuales: {len([f for f in saved_files.keys() if f != 'combined'])}")
        logger.info(f"  • Dataset combinado: {saved_files.get('combined', 'N/A')}")
        logger.info(f"  • Informe detallado: {output_files['report_file']}")
        logger.info(f"  • Configuración: {output_files['config_file']}")
        if viz_success:
            logger.info(f"  • Visualizaciones: 2 gráficos en carpeta graficos/")
        logger.info("")
        
        logger.info("DATASETS LISTOS PARA MODELADO:")
        logger.info(f"  • Train: datos/telecomx_train_dataset_{timestamp}.csv")
        logger.info(f"  • Validation: datos/telecomx_validation_dataset_{timestamp}.csv")
        logger.info(f"  • Test: datos/telecomx_test_dataset_{timestamp}.csv")
        logger.info("")
        
        logger.info("PRÓXIMO PASO SUGERIDO:")
        logger.info("  Paso 10: Entrenamiento de Modelos Predictivos")
        logger.info("  - Usar datasets de la carpeta datos/")
        logger.info("  - Implementar Random Forest y XGBoost")
        logger.info("  - Aplicar class weighting del Paso 4")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()