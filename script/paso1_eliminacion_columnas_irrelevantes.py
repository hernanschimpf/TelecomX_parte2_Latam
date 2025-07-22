"""
TELECOMX - PIPELINE DE PREDICCIÓN DE CHURN
===========================================
Paso 1: Eliminación de Columnas Irrelevantes

Descripción:
    Elimina columnas que no aportan valor al análisis o a los modelos predictivos,
    como identificadores únicos y columnas con datos malformados.

Autor: Ingeniero de Datos
Fecha: 2025-07-21
"""

import pandas as pd
import os
import logging
from datetime import datetime
import sys

def create_directories():
    """Crea las carpetas necesarias si no existen"""
    directories = ['script', 'informes', 'excel', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Carpeta creada: {directory}")

# Configuración de logging
def setup_logging():
    """Configura el sistema de logging para trackear el proceso"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/paso1_eliminacion_columnas.log', mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def load_data(file_path):
    """
    Carga el dataset desde archivo CSV con detección automática de codificación
    y corrección de caracteres especiales
    
    Args:
        file_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    logger = logging.getLogger(__name__)
    
    # Lista extendida de codificaciones para probar
    encodings_to_try = [
        'cp1252',      # Windows-1252 (común en Excel)
        'latin1',      # ISO 8859-1
        'iso-8859-1',  # ISO 8859-1 alternativo
        'cp850',       # Páginas de código IBM
        'utf-8',       # UTF-8 estándar
        'utf-8-sig',   # UTF-8 con BOM
        'ansi'         # ANSI
    ]
    
    df = None
    successful_encoding = None
    
    for encoding in encodings_to_try:
        try:
            logger.info(f"Probando codificacion: {encoding}")
            
            # Cargar CSV con separador punto y coma
            df = pd.read_csv(file_path, sep=';', encoding=encoding)
            
            # Verificar que los caracteres especiales se ven bien
            # Buscar columnas con texto que contengan caracteres problemáticos
            sample_check = True
            for col in df.columns:
                if any(char in str(col) for char in ['Ã', '�', 'â€']):
                    sample_check = False
                    break
            
            # Verificar también en los datos
            if sample_check and len(df) > 0:
                for i in range(min(3, len(df))):
                    for col in df.columns:
                        cell_value = str(df.iloc[i][col])
                        if any(char in cell_value for char in ['Ã', '�', 'â€']):
                            sample_check = False
                            break
                    if not sample_check:
                        break
            
            if sample_check:
                successful_encoding = encoding
                logger.info("Dataset cargado exitosamente desde: " + file_path)
                logger.info(f"Codificacion exitosa: {encoding}")
                logger.info(f"Dimensiones iniciales: {df.shape[0]} filas x {df.shape[1]} columnas")
                
                # Mostrar muestra de columnas para verificar
                logger.info("Columnas detectadas:")
                for i, col in enumerate(df.columns[:5]):
                    logger.info(f"  {i+1}. {col}")
                
                break
            else:
                logger.warning(f"Codificacion {encoding} carga datos pero con caracteres malformados")
                continue
                
        except (UnicodeDecodeError, UnicodeError) as e:
            logger.warning(f"Error Unicode con {encoding}: {str(e)[:50]}...")
            continue
        except Exception as e:
            logger.warning(f"Error general con {encoding}: {str(e)[:50]}...")
            continue
    
    if df is None or successful_encoding is None:
            # Si todas fallaron, cargar con la mejor opción disponible
            logger.warning("Usando cp1252 como codificacion por defecto")
            df = pd.read_csv(file_path, sep=';', encoding='cp1252')
            successful_encoding = 'cp1252'
            
            logger.info("Dataset cargado con codificacion cp1252")
            logger.info(f"Dimensiones iniciales: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    return df

def analyze_columns(df):
    """
    Analiza las columnas del dataset para identificar cuáles eliminar
    Incluye métricas estadísticas detalladas
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        
    Returns:
        dict: Análisis detallado de columnas con métricas estadísticas
    """
    logger = logging.getLogger(__name__)
    logger.info("Iniciando analisis estadistico detallado de columnas...")
    
    analysis = {}
    total_rows = len(df)
    
    for col in df.columns:
        col_data = df[col].dropna()
        unique_values = col_data.nunique()
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / total_rows) * 100
        
        # Métricas estadísticas adicionales
        uniqueness_ratio = unique_values / total_rows if total_rows > 0 else 0
        completeness_ratio = (total_rows - null_count) / total_rows if total_rows > 0 else 0
        
        # Análisis de tipo de datos
        is_numeric = df[col].dtype in ['int64', 'float64']
        is_categorical = df[col].dtype == 'object'
        
        # Para columnas categóricas, analizar distribución
        value_counts = None
        mode_frequency = 0
        if is_categorical and len(col_data) > 0:
            value_counts = col_data.value_counts()
            mode_frequency = value_counts.iloc[0] / len(col_data) if len(value_counts) > 0 else 0
        
        # Detectar patrones problemáticos en los datos
        has_malformed_data = False
        if is_categorical and len(col_data) > 0:
            sample_values = col_data.head().astype(str).tolist()
            has_malformed_data = (
                any(val.startswith("{'") or val.startswith("'Total'") for val in sample_values) or
                col in ['Estructura_Cargos_Original', 'Cargo_Mensual']  # Columnas conocidas con problemas
            )
        
        analysis[col] = {
            'unique_values': unique_values,
            'total_rows': total_rows,
            'null_count': null_count,
            'null_percentage': round(null_percentage, 2),
            'uniqueness_ratio': round(uniqueness_ratio, 4),
            'completeness_ratio': round(completeness_ratio, 4),
            'sample_values': col_data.unique()[:5].tolist() if len(col_data) > 0 else [],
            'data_type': str(df[col].dtype),
            'is_numeric': is_numeric,
            'is_categorical': is_categorical,
            'mode_frequency': round(mode_frequency, 4) if mode_frequency > 0 else 0,
            'has_malformed_data': has_malformed_data,
            'value_distribution': value_counts.head().to_dict() if value_counts is not None else {}
        }
        
        # Log de métricas clave
        logger.info(f"Columna '{col}':")
        logger.info(f"  - Valores únicos: {unique_values:,} de {total_rows:,} (ratio: {uniqueness_ratio:.4f})")
        logger.info(f"  - Completitud: {completeness_ratio:.2%}")
        logger.info(f"  - Valores nulos: {null_percentage:.1f}%")
        if mode_frequency > 0:
            logger.info(f"  - Frecuencia del valor más común: {mode_frequency:.2%}")
    
    return analysis

def identify_irrelevant_columns(df, analysis):
    """
    Identifica columnas irrelevantes basándose en criterios estadísticos específicos
    
    Args:
        df (pd.DataFrame): Dataset
        analysis (dict): Análisis de columnas con métricas
        
    Returns:
        dict: Columnas categorizadas para eliminación con justificación estadística
    """
    logger = logging.getLogger(__name__)
    logger.info("APLICANDO CRITERIOS ESTADISTICOS PARA ELIMINACION DE COLUMNAS")
    logger.info("=" * 60)
    
    columns_to_remove = {
        'identificadores_unicos': [],
        'datos_malformados': [],
        'un_solo_valor': [],
        'muchos_nulos': [],
        'alta_cardinalidad': []
    }
    
    reasons = {}
    detailed_metrics = {}
    
    # CRITERIO 1: Identificadores únicos (Uniqueness Ratio >= 0.95)
    logger.info("CRITERIO 1: IDENTIFICADORES UNICOS")
    logger.info("Métrica: Uniqueness Ratio >= 0.95")
    logger.info("-" * 40)
    
    for col, stats in analysis.items():
        if (stats['uniqueness_ratio'] >= 0.95 or 
            ('id' in col.lower() and stats['uniqueness_ratio'] > 0.9)):
            columns_to_remove['identificadores_unicos'].append(col)
            reasons[col] = f"Identificador único - Uniqueness Ratio: {stats['uniqueness_ratio']:.4f}"
            detailed_metrics[col] = {
                'criterion': 'Identificador único',
                'uniqueness_ratio': stats['uniqueness_ratio'],
                'threshold': 0.95,
                'unique_values': stats['unique_values'],
                'total_rows': stats['total_rows']
            }
            logger.info(f"  ELIMINAR: {col}")
            logger.info(f"    Ratio de unicidad: {stats['uniqueness_ratio']:.4f} >= 0.95")
            logger.info(f"    Valores únicos: {stats['unique_values']:,} de {stats['total_rows']:,}")
        else:
            # Log para columnas que NO se eliminan por este criterio
            logger.info(f"  MANTENER: {col}")
            logger.info(f"    Ratio de unicidad: {stats['uniqueness_ratio']:.4f} < 0.95")
            logger.info(f"    Valores únicos: {stats['unique_values']:,} de {stats['total_rows']:,}")
    
    # CRITERIO 2: Datos malformados (Detectados por patrones específicos)
    logger.info("\nCRITERIO 2: DATOS MALFORMADOS")
    logger.info("Métrica: Detección de patrones problemáticos")
    logger.info("-" * 40)
    
    for col, stats in analysis.items():
        if (stats['has_malformed_data'] and 
            col not in [item for sublist in columns_to_remove.values() for item in sublist]):
            columns_to_remove['datos_malformados'].append(col)
            reasons[col] = f"Datos malformados - Contiene estructuras no válidas"
            detailed_metrics[col] = {
                'criterion': 'Datos malformados',
                'pattern_detected': True,
                'sample_problematic_values': stats['sample_values'][:2]
            }
            logger.info(f"  ELIMINAR: {col}")
            logger.info(f"    Patrón problemático detectado")
            logger.info(f"    Muestra de valores: {stats['sample_values'][:2]}")
    
    # CRITERIO 3: Un solo valor único (Uniqueness Ratio = 0 o unique_values = 1)
    logger.info("\nCRITERIO 3: COLUMNAS CON UN SOLO VALOR")
    logger.info("Métrica: unique_values = 1")
    logger.info("-" * 40)
    
    for col, stats in analysis.items():
        if (stats['unique_values'] == 1 and 
            col not in [item for sublist in columns_to_remove.values() for item in sublist]):
            columns_to_remove['un_solo_valor'].append(col)
            reasons[col] = f"Sin variabilidad - Solo 1 valor único: '{stats['sample_values'][0] if stats['sample_values'] else 'N/A'}'"
            detailed_metrics[col] = {
                'criterion': 'Sin variabilidad',
                'unique_values': 1,
                'single_value': stats['sample_values'][0] if stats['sample_values'] else 'N/A'
            }
            logger.info(f"  ELIMINAR: {col}")
            logger.info(f"    Valores únicos: 1")
            logger.info(f"    Valor único: '{stats['sample_values'][0] if stats['sample_values'] else 'N/A'}'")
    
    # CRITERIO 4: Muchos valores nulos (Null Percentage > 80%)
    logger.info("\nCRITERIO 4: MUCHOS VALORES NULOS")
    logger.info("Métrica: Null Percentage > 80%")
    logger.info("-" * 40)
    
    for col, stats in analysis.items():
        if (stats['null_percentage'] > 80 and 
            col not in [item for sublist in columns_to_remove.values() for item in sublist]):
            columns_to_remove['muchos_nulos'].append(col)
            reasons[col] = f"Demasiados nulos - {stats['null_percentage']:.1f}% > 80%"
            detailed_metrics[col] = {
                'criterion': 'Muchos nulos',
                'null_percentage': stats['null_percentage'],
                'threshold': 80.0,
                'completeness_ratio': stats['completeness_ratio']
            }
            logger.info(f"  ELIMINAR: {col}")
            logger.info(f"    Porcentaje de nulos: {stats['null_percentage']:.1f}% > 80%")
            logger.info(f"    Completitud: {stats['completeness_ratio']:.2%}")
    
    # CRITERIO 5: Alta cardinalidad en categóricas (Excluir variables financieras importantes)
    logger.info("\nCRITERIO 5: ALTA CARDINALIDAD EN CATEGORICAS")
    logger.info("Métrica: Columnas categóricas con Uniqueness Ratio > 0.5")
    logger.info("Excepción: Variables financieras y numéricas importantes")
    logger.info("-" * 40)
    
    # Variables importantes que NO deben eliminarse por alta cardinalidad
    important_variables = [
        'cargo_mensual', 'cargo_total', 'facturacion_mensual', 'facturacion_diaria',
        'meses_cliente', 'abandono_cliente'
    ]
    
    for col, stats in analysis.items():
        # Verificar si es una variable financiera/numérica importante
        is_important_numeric = any(important_var in col.lower() for important_var in important_variables)
        
        # Verificar si parece ser numérica pero está codificada como object
        is_likely_numeric = False
        if stats['is_categorical'] and len(stats['sample_values']) > 0:
            # Verificar si los valores parecen numéricos
            sample_str_values = [str(v) for v in stats['sample_values'][:3]]
            numeric_count = 0
            for val in sample_str_values:
                try:
                    float(val.replace(',', '.'))
                    numeric_count += 1
                except:
                    pass
            is_likely_numeric = numeric_count >= len(sample_str_values) * 0.7  # 70% de valores numéricos
        
        if (stats['is_categorical'] and 
            stats['uniqueness_ratio'] > 0.5 and 
            stats['uniqueness_ratio'] < 0.95 and  # No ya capturadas como ID
            not is_important_numeric and  # No es variable financiera importante
            not is_likely_numeric and  # No parece ser numérica malformada
            col not in [item for sublist in columns_to_remove.values() for item in sublist]):
            
            columns_to_remove['alta_cardinalidad'].append(col)
            reasons[col] = f"Alta cardinalidad categórica - {stats['uniqueness_ratio']:.4f} > 0.5"
            detailed_metrics[col] = {
                'criterion': 'Alta cardinalidad categórica',
                'uniqueness_ratio': stats['uniqueness_ratio'],
                'threshold': 0.5,
                'unique_values': stats['unique_values']
            }
            logger.info(f"  ELIMINAR: {col}")
            logger.info(f"    Ratio de unicidad: {stats['uniqueness_ratio']:.4f} > 0.5")
            logger.info(f"    Valores únicos: {stats['unique_values']:,}")
        elif (stats['is_categorical'] and stats['uniqueness_ratio'] > 0.5 and 
              (is_important_numeric or is_likely_numeric)):
            logger.info(f"  MANTENER: {col} (Variable numérica/financiera importante)")
            logger.info(f"    Ratio de unicidad: {stats['uniqueness_ratio']:.4f} > 0.5")
            logger.info(f"    Razón: {'Variable financiera' if is_important_numeric else 'Parece numérica malformada'}")
            logger.info(f"    Muestra: {stats['sample_values'][:3]}")
    
    logger.info("\nCOLUMNAS QUE SE MANTIENEN:")
    logger.info("-" * 40)
    all_removed = [item for sublist in columns_to_remove.values() for item in sublist]
    for col, stats in analysis.items():
        if col not in all_removed:
            logger.info(f"  MANTENER: {col}")
            logger.info(f"    Valores únicos: {stats['unique_values']:,}, Ratio: {stats['uniqueness_ratio']:.4f}")
            logger.info(f"    Nulos: {stats['null_percentage']:.1f}%, Tipo: {stats['data_type']}")
    
    # Todas las columnas a eliminar
    all_columns_to_remove = []
    for category in columns_to_remove.values():
        all_columns_to_remove.extend(category)
    
    logger.info("=" * 60)
    logger.info(f"RESUMEN: {len(all_columns_to_remove)} columnas identificadas para eliminación")
    
    return columns_to_remove, reasons, all_columns_to_remove, detailed_metrics

def remove_irrelevant_columns(df, columns_to_remove):
    """
    Elimina las columnas irrelevantes del dataset
    
    Args:
        df (pd.DataFrame): Dataset original
        columns_to_remove (list): Lista de columnas a eliminar
        
    Returns:
        pd.DataFrame: Dataset sin columnas irrelevantes
    """
    logger = logging.getLogger(__name__)
    
    # Verificar que las columnas existen
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Columnas no encontradas: {missing_columns}")
    
    # Eliminar columnas
    df_clean = df.drop(columns=existing_columns)
    
    logger.info(f"Eliminadas {len(existing_columns)} columnas irrelevantes")
    logger.info(f"Dimensiones despues de limpieza: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")
    
    return df_clean

def validate_data_integrity(df_original, df_clean, removed_columns):
    """
    Valida la integridad de los datos después de la eliminación
    
    Args:
        df_original (pd.DataFrame): Dataset original
        df_clean (pd.DataFrame): Dataset limpio
        removed_columns (list): Columnas eliminadas
        
    Returns:
        dict: Resultados de validación
    """
    logger = logging.getLogger(__name__)
    logger.info("Validando integridad de datos...")
    
    validation = {
        'filas_originales': len(df_original),
        'filas_finales': len(df_clean),
        'columnas_originales': len(df_original.columns),
        'columnas_finales': len(df_clean.columns),
        'columnas_eliminadas': len(removed_columns),
        'filas_perdidas': len(df_original) - len(df_clean),
        'integridad_filas': len(df_original) == len(df_clean),
        'variable_objetivo_presente': 'Abandono_Cliente' in df_clean.columns
    }
    
    # Verificaciones adicionales
    if validation['integridad_filas']:
        logger.info("Integridad de filas: CORRECTA")
    else:
        logger.warning(f"Se perdieron {validation['filas_perdidas']} filas")
    
    if validation['variable_objetivo_presente']:
        logger.info("Variable objetivo 'Abandono_Cliente': PRESENTE")
    else:
        logger.error("Variable objetivo 'Abandono_Cliente': AUSENTE")
    
    return validation

def generate_report(analysis, columns_categorized, reasons, validation, detailed_metrics, timestamp):
    """
    Genera un informe detallado del proceso de eliminación con métricas estadísticas
    
    Args:
        analysis (dict): Análisis de columnas
        columns_categorized (dict): Columnas categorizadas
        reasons (dict): Razones de eliminación
        validation (dict): Resultados de validación
        detailed_metrics (dict): Métricas estadísticas detalladas
        timestamp (str): Timestamp del proceso
        
    Returns:
        str: Contenido del informe
    """
    
    report = f"""
================================================================================
TELECOMX - INFORME DE ELIMINACIÓN DE COLUMNAS IRRELEVANTES
================================================================================
Fecha y Hora: {timestamp}
Paso: 1 - Eliminación de Columnas Irrelevantes

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Dataset Original: {validation['filas_originales']} filas x {validation['columnas_originales']} columnas
• Dataset Final: {validation['filas_finales']} filas x {validation['columnas_finales']} columnas
• Columnas Eliminadas: {validation['columnas_eliminadas']}
• Integridad de Filas: {'✅ CORRECTA' if validation['integridad_filas'] else '❌ COMPROMETIDA'}
• Variable Objetivo: {'✅ PRESENTE' if validation['variable_objetivo_presente'] else '❌ AUSENTE'}

================================================================================
MÉTODOS Y CRITERIOS ESTADÍSTICOS APLICADOS
================================================================================

🔬 CRITERIO 1: IDENTIFICADORES ÚNICOS
   Métrica: Uniqueness Ratio >= 0.95
   Justificación: Columnas con 95%+ valores únicos no aportan capacidad predictiva
   
🔬 CRITERIO 2: DATOS MALFORMADOS  
   Métrica: Detección de patrones estructurales incorrectos
   Justificación: Datos no procesables por algoritmos de ML
   
🔬 CRITERIO 3: SIN VARIABILIDAD
   Métrica: Unique Values = 1
   Justificación: Columnas constantes no proporcionan información discriminatoria
   
🔬 CRITERIO 4: ALTA PRESENCIA DE NULOS
   Métrica: Null Percentage > 80%
   Justificación: Información insuficiente para entrenamiento confiable
   
🔬 CRITERIO 5: ALTA CARDINALIDAD CATEGÓRICA
   Métrica: Uniqueness Ratio > 0.5 en variables categóricas
   Justificación: Demasiadas categorías pueden causar overfitting

================================================================================
COLUMNAS ELIMINADAS CON JUSTIFICACIÓN ESTADÍSTICA
================================================================================
"""
    
    # Agregar detalles estadísticos para cada columna eliminada
    for category_name, columns in columns_categorized.items():
        if not columns:
            continue
            
        category_titles = {
            'identificadores_unicos': '🚫 IDENTIFICADORES ÚNICOS',
            'datos_malformados': '🚫 DATOS MALFORMADOS', 
            'un_solo_valor': '🚫 SIN VARIABILIDAD',
            'muchos_nulos': '🚫 ALTA PRESENCIA DE NULOS',
            'alta_cardinalidad': '🚫 ALTA CARDINALIDAD CATEGÓRICA'
        }
        
        report += f"\n{category_titles.get(category_name, f'🚫 {category_name.upper()}')}\n"
        
        for col in columns:
            metrics = detailed_metrics.get(col, {})
            report += f"\n   📊 {col}:\n"
            report += f"      Razón: {reasons.get(col, 'No especificado')}\n"
            
            # Añadir métricas específicas según el criterio
            if col in detailed_metrics:
                metric_data = detailed_metrics[col]
                if 'uniqueness_ratio' in metric_data:
                    report += f"      Uniqueness Ratio: {metric_data['uniqueness_ratio']:.4f}\n"
                if 'unique_values' in metric_data:
                    report += f"      Valores únicos: {metric_data['unique_values']:,}\n"
                if 'null_percentage' in metric_data:
                    report += f"      Porcentaje nulos: {metric_data['null_percentage']:.1f}%\n"
                if 'completeness_ratio' in metric_data:
                    report += f"      Completitud: {metric_data['completeness_ratio']:.2%}\n"
                if 'single_value' in metric_data:
                    report += f"      Valor único: '{metric_data['single_value']}'\n"
                if 'sample_problematic_values' in metric_data:
                    report += f"      Muestra problemática: {metric_data['sample_problematic_values']}\n"
    
    report += f"""

================================================================================
MÉTRICAS ESTADÍSTICAS RESUMEN
================================================================================
• Total de columnas analizadas: {len(analysis)}
• Columnas eliminadas: {validation['columnas_eliminadas']}
• Columnas mantenidas: {validation['columnas_finales']}
• Tasa de eliminación: {(validation['columnas_eliminadas']/len(analysis)*100):.1f}%

Criterios aplicados:
  - Identificadores únicos: {len(columns_categorized['identificadores_unicos'])} columnas
  - Datos malformados: {len(columns_categorized['datos_malformados'])} columnas
  - Sin variabilidad: {len(columns_categorized['un_solo_valor'])} columnas
  - Muchos nulos: {len(columns_categorized['muchos_nulos'])} columnas
  - Alta cardinalidad: {len(columns_categorized['alta_cardinalidad'])} columnas
================================================================================
ANÁLISIS ESTADÍSTICO DETALLADO POR COLUMNA
================================================================================
"""
    
    for col, stats in analysis.items():
        all_removed = []
        for category in columns_categorized.values():
            all_removed.extend(category)
        
        status = "🚫 ELIMINADA" if col in all_removed else "✅ MANTENIDA"
        reason = f" - {reasons[col]}" if col in reasons else ""
        
        report += f"""
📈 {col} {status}{reason}
   • Tipo de dato: {stats['data_type']}
   • Valores únicos: {stats['unique_values']:,} de {stats['total_rows']:,}
   • Uniqueness Ratio: {stats['uniqueness_ratio']:.4f}
   • Completitud: {stats['completeness_ratio']:.2%} ({stats['null_percentage']:.1f}% nulos)
   • Muestra de valores: {', '.join(map(str, stats['sample_values'][:3]))}"""
        
        if stats['mode_frequency'] > 0:
            report += f"\n   • Frecuencia del valor más común: {stats['mode_frequency']:.2%}"
        
        if stats['value_distribution']:
            top_values = list(stats['value_distribution'].items())[:3]
            distribution_str = ", ".join([f"{k}: {v}" for k, v in top_values])
            report += f"\n   • Distribución top 3: {distribution_str}"
    
    report += """
================================================================================
COLUMNAS FINALES PARA MODELADO
================================================================================
"""
    
    # Obtener columnas finales (todas menos las eliminadas)
    all_removed = []
    for category in columns_categorized.values():
        all_removed.extend(category)
    
    final_columns = [col for col in analysis.keys() if col not in all_removed]
    
    report += f"Total de columnas para modelado: {len(final_columns)}\n\n"
    
    # Variable objetivo
    if 'Abandono_Cliente' in final_columns:
        report += "🎯 VARIABLE OBJETIVO:\n"
        report += "   • Abandono_Cliente\n\n"
    
    # Variables predictoras
    predictor_columns = [col for col in final_columns if col != 'Abandono_Cliente']
    report += f"📊 VARIABLES PREDICTORAS ({len(predictor_columns)} columnas):\n"
    for i, col in enumerate(predictor_columns, 1):
        report += f"   {i:2d}. {col}\n"
    
    report += """
================================================================================
ANÁLISIS DETALLADO DE COLUMNAS ORIGINALES
================================================================================
"""
    
    for col, stats in analysis.items():
        status = "🚫 ELIMINADA" if col in all_removed else "✅ MANTENIDA"
        reason = f" - {reasons[col]}" if col in reasons else ""
        
        report += f"""
{col} {status}{reason}
   • Valores únicos: {stats['unique_values']:,}
   • Valores nulos: {stats['null_count']:,} ({stats['null_percentage']}%)
   • Tipo de dato: {stats['data_type']}
   • Muestra: {', '.join(map(str, stats['sample_values']))}
"""
    
    report += f"""
================================================================================
VALIDACIÓN DE INTEGRIDAD
================================================================================
• Filas originales: {validation['filas_originales']:,}
• Filas finales: {validation['filas_finales']:,}
• Filas perdidas: {validation['filas_perdidas']:,}
• Columnas originales: {validation['columnas_originales']}
• Columnas finales: {validation['columnas_finales']}
• Columnas eliminadas: {validation['columnas_eliminadas']}

Verificaciones:
{'✅' if validation['integridad_filas'] else '❌'} Integridad de filas mantenida
{'✅' if validation['variable_objetivo_presente'] else '❌'} Variable objetivo presente

================================================================================
SIGUIENTE PASO RECOMENDADO
================================================================================
Paso 2: Análisis Exploratorio de Datos (EDA)
- Distribución de la variable objetivo
- Análisis de correlaciones
- Detección de outliers
- Análisis de variables categóricas vs numéricas

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(df_clean, report_content, timestamp):
    """
    Guarda el dataset limpio y el informe en sus respectivas carpetas
    
    Args:
        df_clean (pd.DataFrame): Dataset limpio
        report_content (str): Contenido del informe
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    
    # Guardar dataset limpio con codificación UTF-8 y BOM para Excel
    excel_filename = f"excel/telecomx_paso1_sin_columnas_irrelevantes_{timestamp}.csv"
    df_clean.to_csv(excel_filename, sep=';', index=False, encoding='utf-8-sig')
    logger.info(f"Dataset limpio guardado: {excel_filename}")
    logger.info("Archivo guardado con codificacion UTF-8-sig (compatible con Excel)")
    
    # Guardar informe
    report_filename = f"informes/paso1_eliminacion_columnas_irrelevantes_informe_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Informe guardado: {report_filename}")
    
    return excel_filename, report_filename

def main():
    """Función principal que ejecuta todo el proceso"""
    
    # Crear directorios primero, luego configurar logging
    create_directories()
    logger = setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("INICIANDO PASO 1: ELIMINACION DE COLUMNAS IRRELEVANTES")
    logger.info("=" * 70)
    
    try:
        # 1. Cargar datos
        input_file = "telecomx_dataframe_LIMPIO.csv"  # Archivo en la raíz del proyecto
        df_original = load_data(input_file)
        
        # 2. Analizar columnas
        analysis = analyze_columns(df_original)
        
        # 3. Identificar columnas irrelevantes
        columns_categorized, reasons, all_columns_to_remove, detailed_metrics = identify_irrelevant_columns(df_original, analysis)
        
        # 4. Eliminar columnas irrelevantes
        df_clean = remove_irrelevant_columns(df_original, all_columns_to_remove)
        
        # 5. Validar integridad
        validation = validate_data_integrity(df_original, df_clean, all_columns_to_remove)
        
        # 6. Generar informe
        report_content = generate_report(analysis, columns_categorized, reasons, validation, detailed_metrics, timestamp)
        
        # 7. Guardar archivos
        excel_file, report_file = save_files(df_clean, report_content, timestamp)
        
        # 8. Resumen final
        logger.info("=" * 70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info(f"Columnas eliminadas: {len(all_columns_to_remove)}")
        logger.info(f"Columnas finales: {df_clean.shape[1]}")
        logger.info(f"Archivo resultante: {excel_file}")
        logger.info(f"Informe generado: {report_file}")
        logger.info("=" * 70)
        
        print("\nSIGUIENTE PASO:")
        print("   Ejecutar Paso 2: Analisis Exploratorio de Datos (EDA)")
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        raise

if __name__ == "__main__":
    main()