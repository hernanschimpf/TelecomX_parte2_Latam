"""
TELECOMX - PIPELINE DE PREDICCIÓN DE CHURN
===========================================
Paso 2: Encoding de Variables Categóricas

Descripción:
    Transforma las variables categóricas a formato numérico para hacerlas 
    compatibles con los algoritmos de machine learning. Utiliza un enfoque
    moderado con label encoding para binarias, ordinal encoding para variables
    con orden natural, y one-hot encoding para categóricas con <5 categorías.

Autor: Ingeniero de Datos
Fecha: 2025-07-21
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import sys
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
            logging.FileHandler('logs/paso2_encoding_variables.log', mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def find_latest_paso1_file():
    """
    Encuentra el archivo más reciente del Paso 1 en la carpeta excel
    
    Returns:
        str: Ruta al archivo más reciente del Paso 1
    """
    logger = logging.getLogger(__name__)
    
    excel_files = [f for f in os.listdir('excel') if f.startswith('telecomx_paso1_sin_columnas_irrelevantes_')]
    
    if not excel_files:
        raise FileNotFoundError("No se encontró ningún archivo del Paso 1 en la carpeta excel/")
    
    # Ordenar por fecha de modificación y tomar el más reciente
    excel_files.sort(key=lambda x: os.path.getmtime(os.path.join('excel', x)), reverse=True)
    latest_file = os.path.join('excel', excel_files[0])
    
    logger.info(f"Archivo del Paso 1 encontrado: {latest_file}")
    return latest_file

def load_data(file_path):
    """
    Carga el dataset del Paso 1 con detección automática de codificación
    
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
            logger.info(f"Dimensiones iniciales: {df.shape[0]} filas x {df.shape[1]} columnas")
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

def analyze_categorical_variables(df):
    """
    Analiza las variables categóricas para determinar la estrategia de encoding
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        
    Returns:
        dict: Análisis detallado de variables categóricas
    """
    logger = logging.getLogger(__name__)
    logger.info("INICIANDO ANALISIS DE VARIABLES CATEGORICAS")
    logger.info("=" * 60)
    
    categorical_analysis = {}
    
    # Identificar variables categóricas (object type)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remover la variable objetivo del análisis si está como categórica
    if 'Abandono_Cliente' in categorical_columns:
        categorical_columns.remove('Abandono_Cliente')
    
    logger.info(f"Variables categoricas encontradas: {len(categorical_columns)}")
    
    for col in categorical_columns:
        unique_values = df[col].nunique()
        value_counts = df[col].value_counts()
        null_count = df[col].isnull().sum()
        
        # Obtener valores únicos
        unique_vals = df[col].dropna().unique().tolist()
        
        # Determinar tipo de variable categórica
        var_type = determine_categorical_type(col, unique_vals, unique_values)
        
        # Determinar estrategia de encoding
        encoding_strategy = determine_encoding_strategy(var_type, unique_values)
        
        categorical_analysis[col] = {
            'unique_values': unique_values,
            'null_count': null_count,
            'null_percentage': (null_count / len(df)) * 100,
            'values_list': unique_vals,
            'value_distribution': value_counts.to_dict(),
            'variable_type': var_type,
            'encoding_strategy': encoding_strategy,
            'data_type': str(df[col].dtype)
        }
        
        logger.info(f"\nVariable: {col}")
        logger.info(f"  Tipo: {var_type}")
        logger.info(f"  Valores unicos: {unique_values}")
        logger.info(f"  Estrategia: {encoding_strategy}")
        logger.info(f"  Valores: {unique_vals}")
    
    return categorical_analysis

def determine_categorical_type(column_name, unique_values, count_unique):
    """
    Determina el tipo de variable categórica
    
    Args:
        column_name (str): Nombre de la columna
        unique_values (list): Lista de valores únicos
        count_unique (int): Número de valores únicos
        
    Returns:
        str: Tipo de variable categórica
    """
    
    # Variables binarias
    if count_unique == 2:
        return "binaria"
    
    # Variables ordinales (con orden lógico)
    ordinal_patterns = {
        'tipo_contrato': ['Mes a Mes', 'Un Año', 'Dos Años'],
        'contrato': ['Mes a Mes', 'Un Año', 'Dos Años']
    }
    
    col_lower = column_name.lower()
    for pattern, order in ordinal_patterns.items():
        if pattern in col_lower:
            # Verificar si los valores coinciden con el patrón ordinal
            if all(val in unique_values for val in order if val in unique_values):
                return "ordinal"
    
    # Variables nominales
    return "nominal"

def determine_encoding_strategy(var_type, unique_count):
    """
    Determina la estrategia de encoding según el tipo y cardinalidad
    
    Args:
        var_type (str): Tipo de variable categórica
        unique_count (int): Número de valores únicos
        
    Returns:
        str: Estrategia de encoding a aplicar
    """
    
    if var_type == "binaria":
        return "label_encoding"
    elif var_type == "ordinal":
        return "ordinal_encoding"
    elif var_type == "nominal" and unique_count <= 5:
        return "one_hot_encoding"
    elif var_type == "nominal" and unique_count > 5:
        return "frequency_encoding"  # Para alta cardinalidad
    else:
        return "mantener_original"

def apply_label_encoding(df, column, analysis_data):
    """
    Aplica label encoding a una variable binaria
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Nombre de la columna
        analysis_data (dict): Datos del análisis de la columna
        
    Returns:
        pd.DataFrame, dict: Dataset modificado y metadatos del encoding
    """
    logger = logging.getLogger(__name__)
    
    # Crear nueva columna con sufijo
    new_column = f"{column}_encoded"
    
    # Aplicar label encoding
    le = LabelEncoder()
    df[new_column] = le.fit_transform(df[column].fillna('Missing'))
    
    # Metadatos del encoding
    encoding_metadata = {
        'original_column': column,
        'new_column': new_column,
        'encoding_type': 'label_encoding',
        'label_mapping': dict(zip(le.classes_, le.transform(le.classes_))),
        'null_handling': 'Missing category created'
    }
    
    logger.info(f"Label encoding aplicado a {column}")
    logger.info(f"  Mapeo: {encoding_metadata['label_mapping']}")
    
    return df, encoding_metadata

def apply_ordinal_encoding(df, column, analysis_data):
    """
    Aplica ordinal encoding a una variable ordinal
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Nombre de la columna
        analysis_data (dict): Datos del análisis de la columna
        
    Returns:
        pd.DataFrame, dict: Dataset modificado y metadatos del encoding
    """
    logger = logging.getLogger(__name__)
    
    # Crear nueva columna con sufijo
    new_column = f"{column}_encoded"
    
    # Definir orden para variables ordinales conocidas
    ordinal_orders = {
        'tipo_contrato': ['Mes a Mes', 'Un Año', 'Dos Años']
    }
    
    col_lower = column.lower()
    order = None
    for pattern, defined_order in ordinal_orders.items():
        if pattern in col_lower:
            # Filtrar solo los valores que existen en los datos
            order = [val for val in defined_order if val in analysis_data['values_list']]
            break
    
    if order is None:
        # Si no hay orden predefinido, usar orden alfabético
        order = sorted(analysis_data['values_list'])
    
    # Aplicar ordinal encoding
    oe = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', unknown_value=-1)
    df[new_column] = oe.fit_transform(df[[column]].fillna('Missing'))
    
    # Metadatos del encoding
    encoding_metadata = {
        'original_column': column,
        'new_column': new_column,
        'encoding_type': 'ordinal_encoding',
        'ordinal_order': order,
        'null_handling': 'Missing category created'
    }
    
    logger.info(f"Ordinal encoding aplicado a {column}")
    logger.info(f"  Orden: {order}")
    
    return df, encoding_metadata

def apply_one_hot_encoding(df, column, analysis_data):
    """
    Aplica one-hot encoding a una variable nominal
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Nombre de la columna
        analysis_data (dict): Datos del análisis de la columna
        
    Returns:
        pd.DataFrame, dict: Dataset modificado y metadatos del encoding
    """
    logger = logging.getLogger(__name__)
    
    # Aplicar one-hot encoding con prefijo
    dummy_df = pd.get_dummies(df[column], prefix=column, dummy_na=True)
    
    # Agregar las nuevas columnas al dataset
    df = pd.concat([df, dummy_df], axis=1)
    
    # Metadatos del encoding
    encoding_metadata = {
        'original_column': column,
        'new_columns': dummy_df.columns.tolist(),
        'encoding_type': 'one_hot_encoding',
        'categories_created': len(dummy_df.columns),
        'null_handling': 'Separate category if nulls exist'
    }
    
    logger.info(f"One-hot encoding aplicado a {column}")
    logger.info(f"  Columnas creadas: {len(dummy_df.columns)}")
    logger.info(f"  Nuevas columnas: {dummy_df.columns.tolist()}")
    
    return df, encoding_metadata

def apply_frequency_encoding(df, column, analysis_data):
    """
    Aplica frequency encoding a una variable de alta cardinalidad
    
    Args:
        df (pd.DataFrame): Dataset
        column (str): Nombre de la columna
        analysis_data (dict): Datos del análisis de la columna
        
    Returns:
        pd.DataFrame, dict: Dataset modificado y metadatos del encoding
    """
    logger = logging.getLogger(__name__)
    
    # Crear nueva columna con sufijo
    new_column = f"{column}_freq_encoded"
    
    # Calcular frecuencias
    freq_map = df[column].value_counts().to_dict()
    
    # Aplicar frequency encoding
    df[new_column] = df[column].map(freq_map).fillna(0)
    
    # Metadatos del encoding
    encoding_metadata = {
        'original_column': column,
        'new_column': new_column,
        'encoding_type': 'frequency_encoding',
        'frequency_mapping': freq_map,
        'null_handling': 'Mapped to 0'
    }
    
    logger.info(f"Frequency encoding aplicado a {column}")
    logger.info(f"  Valores unicos mapeados: {len(freq_map)}")
    
    return df, encoding_metadata

def perform_encoding(df, categorical_analysis):
    """
    Ejecuta el encoding de todas las variables categóricas
    
    Args:
        df (pd.DataFrame): Dataset original
        categorical_analysis (dict): Análisis de variables categóricas
        
    Returns:
        pd.DataFrame, dict: Dataset con encoding aplicado y metadatos
    """
    logger = logging.getLogger(__name__)
    logger.info("EJECUTANDO ESTRATEGIAS DE ENCODING")
    logger.info("=" * 60)
    
    df_encoded = df.copy()
    encoding_metadata = {}
    
    for column, analysis in categorical_analysis.items():
        strategy = analysis['encoding_strategy']
        
        logger.info(f"\nProcesando {column} con estrategia: {strategy}")
        
        if strategy == "label_encoding":
            df_encoded, metadata = apply_label_encoding(df_encoded, column, analysis)
        elif strategy == "ordinal_encoding":
            df_encoded, metadata = apply_ordinal_encoding(df_encoded, column, analysis)
        elif strategy == "one_hot_encoding":
            df_encoded, metadata = apply_one_hot_encoding(df_encoded, column, analysis)
        elif strategy == "frequency_encoding":
            df_encoded, metadata = apply_frequency_encoding(df_encoded, column, analysis)
        else:
            logger.info(f"  Manteniendo {column} sin cambios")
            metadata = {
                'original_column': column,
                'encoding_type': 'sin_cambios',
                'reason': 'No cumple criterios para encoding'
            }
        
        encoding_metadata[column] = metadata
    
    return df_encoded, encoding_metadata

def analyze_correlations(df, encoded_columns):
    """
    Analiza correlaciones entre variables encoded y la variable objetivo
    
    Args:
        df (pd.DataFrame): Dataset con encoding aplicado
        encoded_columns (list): Lista de columnas nuevas creadas
        
    Returns:
        dict: Análisis de correlaciones
    """
    logger = logging.getLogger(__name__)
    logger.info("ANALIZANDO CORRELACIONES POST-ENCODING")
    logger.info("=" * 50)
    
    correlation_analysis = {}
    
    # Verificar que existe la variable objetivo
    if 'Abandono_Cliente' not in df.columns:
        logger.warning("Variable objetivo 'Abandono_Cliente' no encontrada")
        return correlation_analysis
    
    # Seleccionar solo columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        # Calcular matriz de correlación
        corr_matrix = numeric_df.corr()
        
        # Correlaciones con variable objetivo
        target_correlations = corr_matrix['Abandono_Cliente'].abs().sort_values(ascending=False)
        
        # Correlaciones altas entre variables (multicolinealidad)
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        correlation_analysis = {
            'correlation_matrix': corr_matrix,
            'target_correlations': target_correlations,
            'high_correlations': high_correlations,
            'encoded_columns_correlations': {col: target_correlations.get(col, 0) for col in encoded_columns if col in target_correlations}
        }
        
        logger.info(f"Variables mas correlacionadas con objetivo:")
        for var, corr in target_correlations.head(10).items():
            logger.info(f"  {var}: {corr:.4f}")
        
        if high_correlations:
            logger.warning(f"Correlaciones altas detectadas: {len(high_correlations)}")
            for item in high_correlations[:5]:
                logger.warning(f"  {item['var1']} - {item['var2']}: {item['correlation']:.4f}")
    
    return correlation_analysis

def calculate_vif(df, encoded_columns):
    """
    Calcula el Factor de Inflación de Varianza para detectar multicolinealidad
    
    Args:
        df (pd.DataFrame): Dataset
        encoded_columns (list): Columnas encoded para analizar
        
    Returns:
        dict: Análisis de VIF
    """
    logger = logging.getLogger(__name__)
    logger.info("CALCULANDO VIF PARA DETECCION DE MULTICOLINEALIDAD")
    logger.info("=" * 50)
    
    try:
        # Seleccionar solo columnas numéricas encoded
        numeric_encoded = [col for col in encoded_columns if col in df.select_dtypes(include=[np.number]).columns]
        
        if len(numeric_encoded) < 2:
            logger.info("Insuficientes variables numéricas para calcular VIF")
            return {}
        
        # Preparar datos para VIF
        vif_df = df[numeric_encoded].dropna()
        
        # Calcular VIF para cada variable
        vif_data = []
        for i, col in enumerate(vif_df.columns):
            try:
                vif_value = variance_inflation_factor(vif_df.values, i)
                vif_data.append({'variable': col, 'vif': vif_value})
            except:
                vif_data.append({'variable': col, 'vif': np.nan})
        
        # Convertir a DataFrame para mejor manejo
        vif_results = pd.DataFrame(vif_data).sort_values('vif', ascending=False)
        
        # Identificar variables con VIF alto (>10 indica multicolinealidad severa)
        high_vif = vif_results[vif_results['vif'] > 10]
        
        vif_analysis = {
            'vif_results': vif_results,
            'high_vif_variables': high_vif,
            'multicollinearity_detected': len(high_vif) > 0
        }
        
        logger.info("Variables con VIF mas alto:")
        for _, row in vif_results.head(10).iterrows():
            logger.info(f"  {row['variable']}: {row['vif']:.2f}")
        
        if len(high_vif) > 0:
            logger.warning(f"Multicolinealidad detectada en {len(high_vif)} variables (VIF > 10)")
        
        return vif_analysis
        
    except Exception as e:
        logger.error(f"Error al calcular VIF: {str(e)}")
        return {}

def validate_encoding_integrity(df_original, df_encoded, encoding_metadata):
    """
    Valida la integridad del proceso de encoding
    
    Args:
        df_original (pd.DataFrame): Dataset original
        df_encoded (pd.DataFrame): Dataset con encoding aplicado
        encoding_metadata (dict): Metadatos del encoding
        
    Returns:
        dict: Resultados de validación
    """
    logger = logging.getLogger(__name__)
    logger.info("VALIDANDO INTEGRIDAD DEL ENCODING")
    logger.info("=" * 40)
    
    validation = {
        'filas_originales': len(df_original),
        'filas_finales': len(df_encoded),
        'columnas_originales': len(df_original.columns),
        'columnas_finales': len(df_encoded.columns),
        'columnas_agregadas': len(df_encoded.columns) - len(df_original.columns),
        'integridad_filas': len(df_original) == len(df_encoded),
        'variables_encoded': len(encoding_metadata),
        'estrategias_aplicadas': {}
    }
    
    # Contar estrategias aplicadas
    for column, metadata in encoding_metadata.items():
        strategy = metadata.get('encoding_type', 'unknown')
        validation['estrategias_aplicadas'][strategy] = validation['estrategias_aplicadas'].get(strategy, 0) + 1
    
    # Verificar variable objetivo
    validation['variable_objetivo_preservada'] = 'Abandono_Cliente' in df_encoded.columns
    
    if validation['variable_objetivo_preservada']:
        objetivo_original = df_original['Abandono_Cliente'].value_counts()
        objetivo_final = df_encoded['Abandono_Cliente'].value_counts()
        validation['distribucion_objetivo_preservada'] = objetivo_original.equals(objetivo_final)
    
    # Log de validación
    logger.info(f"Filas: {validation['filas_originales']} -> {validation['filas_finales']}")
    logger.info(f"Columnas: {validation['columnas_originales']} -> {validation['columnas_finales']} (+{validation['columnas_agregadas']})")
    logger.info(f"Variables encoded: {validation['variables_encoded']}")
    logger.info(f"Estrategias aplicadas: {validation['estrategias_aplicadas']}")
    
    if validation['integridad_filas']:
        logger.info("Integridad de filas: CORRECTA")
    else:
        logger.warning("ALERTA: Se perdieron filas durante el encoding")
    
    if validation['variable_objetivo_preservada']:
        logger.info("Variable objetivo: PRESERVADA")
    else:
        logger.error("ERROR: Variable objetivo no encontrada")
    
    return validation

def generate_visualizations(df, correlation_analysis, timestamp):
    """
    Genera visualizaciones del proceso de encoding
    
    Args:
        df (pd.DataFrame): Dataset con encoding aplicado
        correlation_analysis (dict): Análisis de correlaciones
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    logger.info("GENERANDO VISUALIZACIONES")
    
    try:
        # Configurar estilo
        plt.style.use('default')
        
        # 1. Matriz de correlación con variable objetivo
        if 'target_correlations' in correlation_analysis:
            plt.figure(figsize=(12, 8))
            target_corr = correlation_analysis['target_correlations'].head(15)
            
            colors = ['red' if x < 0 else 'blue' for x in target_corr.values]
            bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
            
            plt.yticks(range(len(target_corr)), target_corr.index)
            plt.xlabel('Correlación con Abandono_Cliente')
            plt.title('Top 15 Variables - Correlación con Variable Objetivo')
            plt.grid(axis='x', alpha=0.3)
            
            # Añadir valores en las barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01 if width >= 0 else width - 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', 
                        ha='left' if width >= 0 else 'right', 
                        va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'graficos/paso2_correlaciones_objetivo_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Grafico de correlaciones generado")
        
        # 2. Distribución de la variable objetivo
        if 'Abandono_Cliente' in df.columns:
            plt.figure(figsize=(8, 6))
            
            objetivo_counts = df['Abandono_Cliente'].value_counts()
            objetivo_pct = df['Abandono_Cliente'].value_counts(normalize=True) * 100
            
            bars = plt.bar(['No Abandono (0)', 'Abandono (1)'], objetivo_counts.values, 
                          color=['lightblue', 'salmon'], alpha=0.8)
            
            plt.ylabel('Cantidad de Clientes')
            plt.title('Distribución de Variable Objetivo: Abandono_Cliente')
            
            # Añadir porcentajes en las barras
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}\n({objetivo_pct.iloc[i]:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'graficos/paso2_distribucion_objetivo_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Grafico de distribucion objetivo generado")
    
    except Exception as e:
        logger.error(f"Error al generar visualizaciones: {str(e)}")

def generate_report(categorical_analysis, encoding_metadata, correlation_analysis, vif_analysis, validation, timestamp):
    """
    Genera un informe detallado del proceso de encoding
    
    Args:
        categorical_analysis (dict): Análisis de variables categóricas
        encoding_metadata (dict): Metadatos del encoding
        correlation_analysis (dict): Análisis de correlaciones
        vif_analysis (dict): Análisis de VIF
        validation (dict): Resultados de validación
        timestamp (str): Timestamp del proceso
        
    Returns:
        str: Contenido del informe
    """
    
    report = f"""
================================================================================
TELECOMX - INFORME DE ENCODING DE VARIABLES CATEGÓRICAS
================================================================================
Fecha y Hora: {timestamp}
Paso: 2 - Encoding de Variables Categóricas

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Dataset Original: {validation['filas_originales']:,} filas x {validation['columnas_originales']} columnas
• Dataset Final: {validation['filas_finales']:,} filas x {validation['columnas_finales']} columnas
• Columnas Agregadas: {validation['columnas_agregadas']}
• Variables Procesadas: {validation['variables_encoded']}
• Integridad de Filas: {'✅ CORRECTA' if validation['integridad_filas'] else '❌ COMPROMETIDA'}
• Variable Objetivo: {'✅ PRESERVADA' if validation['variable_objetivo_preservada'] else '❌ AUSENTE'}

================================================================================
METODOLOGÍA APLICADA - ENFOQUE MODERADO
================================================================================

🔬 ESTRATEGIAS DE ENCODING UTILIZADAS:

1. LABEL ENCODING
   • Aplicado a: Variables binarias (2 valores únicos)
   • Justificación: Mapeo directo 0/1 para algoritmos de ML
   • Ventaja: Mantiene simplicidad y interpretabilidad

2. ORDINAL ENCODING  
   • Aplicado a: Variables con orden lógico natural
   • Justificación: Preserva relación ordinal entre categorías
   • Ejemplo: Mes a Mes < Un Año < Dos Años

3. ONE-HOT ENCODING
   • Aplicado a: Variables nominales con ≤5 categorías
   • Justificación: Evita asunción de orden en datos nominales
   • Control: Limitado a baja cardinalidad para evitar explosión dimensional

4. FREQUENCY ENCODING
   • Aplicado a: Variables nominales con >5 categorías
   • Justificación: Reduce dimensionalidad manteniendo información
   • Método: Mapeo por frecuencia de aparición

================================================================================
ANÁLISIS DETALLADO POR VARIABLE
================================================================================
"""
    
    # Análisis por variable
    for column, analysis in categorical_analysis.items():
        encoding_info = encoding_metadata.get(column, {})
        strategy = analysis['encoding_strategy']
        
        report += f"""
📊 {column}
   • Tipo Original: {analysis['variable_type']} ({analysis['unique_values']} valores únicos)
   • Estrategia Aplicada: {strategy.upper()}
   • Valores Nulos: {analysis['null_count']} ({analysis['null_percentage']:.1f}%)
   • Valores Únicos: {analysis['values_list']}
   • Distribución: {dict(list(analysis['value_distribution'].items())[:3])}
"""
        
        if encoding_info.get('encoding_type') == 'label_encoding':
            report += f"   • Mapeo: {encoding_info.get('label_mapping', {})}\n"
        elif encoding_info.get('encoding_type') == 'ordinal_encoding':
            report += f"   • Orden: {encoding_info.get('ordinal_order', [])}\n"
        elif encoding_info.get('encoding_type') == 'one_hot_encoding':
            report += f"   • Columnas Creadas: {len(encoding_info.get('new_columns', []))}\n"
            report += f"   • Nuevas Variables: {encoding_info.get('new_columns', [])}\n"
        elif encoding_info.get('encoding_type') == 'frequency_encoding':
            freq_map = encoding_info.get('frequency_mapping', {})
            top_freq = dict(list(freq_map.items())[:3]) if freq_map else {}
            report += f"   • Top Frecuencias: {top_freq}\n"
    
    # Resumen de estrategias aplicadas
    report += f"""
================================================================================
RESUMEN DE ESTRATEGIAS APLICADAS
================================================================================
"""
    
    for strategy, count in validation['estrategias_aplicadas'].items():
        strategy_name = {
            'label_encoding': 'Label Encoding',
            'ordinal_encoding': 'Ordinal Encoding', 
            'one_hot_encoding': 'One-Hot Encoding',
            'frequency_encoding': 'Frequency Encoding',
            'sin_cambios': 'Sin Cambios'
        }.get(strategy, strategy)
        
        report += f"• {strategy_name}: {count} variables\n"
    
    # Análisis de correlaciones
    if correlation_analysis and 'target_correlations' in correlation_analysis:
        report += f"""
================================================================================
ANÁLISIS DE CORRELACIONES CON VARIABLE OBJETIVO
================================================================================

Top 10 Variables Más Correlacionadas con Abandono_Cliente:
"""
        
        target_corr = correlation_analysis['target_correlations']
        for i, (var, corr) in enumerate(target_corr.head(10).items(), 1):
            direction = "Positiva" if corr > 0 else "Negativa"
            strength = "Fuerte" if abs(corr) > 0.5 else "Moderada" if abs(corr) > 0.3 else "Débil"
            report += f"{i:2d}. {var}: {corr:.4f} ({direction}, {strength})\n"
        
        # Variables encoded específicamente
        if 'encoded_columns_correlations' in correlation_analysis:
            encoded_corr = correlation_analysis['encoded_columns_correlations']
            if encoded_corr:
                report += f"\nCorrelaciones de Variables Encoded:\n"
                for var, corr in sorted(encoded_corr.items(), key=lambda x: abs(x[1]), reverse=True):
                    report += f"• {var}: {corr:.4f}\n"
    
    # Análisis de multicolinealidad
    if correlation_analysis and 'high_correlations' in correlation_analysis:
        high_corr = correlation_analysis['high_correlations']
        if high_corr:
            report += f"""
================================================================================
DETECCIÓN DE MULTICOLINEALIDAD - CORRELACIONES ALTAS (>0.8)
================================================================================

⚠️  Se detectaron {len(high_corr)} pares de variables con correlación alta:
"""
            for item in high_corr[:10]:
                report += f"• {item['var1']} ↔ {item['var2']}: {item['correlation']:.4f}\n"
            
            if len(high_corr) > 10:
                report += f"... y {len(high_corr) - 10} pares adicionales\n"
        else:
            report += f"""
================================================================================
DETECCIÓN DE MULTICOLINEALIDAD - CORRELACIONES ALTAS (>0.8)
================================================================================

✅ No se detectaron correlaciones problemáticas entre variables.
"""
    
    # Análisis de VIF
    if vif_analysis and 'vif_results' in vif_analysis:
        report += f"""
================================================================================
ANÁLISIS VIF - FACTOR DE INFLACIÓN DE VARIANZA
================================================================================

Interpretación VIF:
• VIF < 5: Sin multicolinealidad
• VIF 5-10: Multicolinealidad moderada  
• VIF > 10: Multicolinealidad severa

Top 10 Variables por VIF:
"""
        
        vif_results = vif_analysis['vif_results']
        for _, row in vif_results.head(10).iterrows():
            vif_val = row['vif']
            if pd.isna(vif_val):
                status = "No calculable"
            elif vif_val < 5:
                status = "✅ Bajo"
            elif vif_val < 10:
                status = "⚠️ Moderado"
            else:
                status = "🚫 Alto"
            
            report += f"• {row['variable']}: {vif_val:.2f} ({status})\n"
        
        if vif_analysis.get('multicollinearity_detected', False):
            high_vif = vif_analysis['high_vif_variables']
            report += f"\n⚠️ ALERTA: {len(high_vif)} variables con VIF > 10 detectadas.\n"
            report += "Considerar eliminación o combinación de variables problemáticas.\n"
    
    report += f"""
================================================================================
IMPACTO EN DIMENSIONALIDAD
================================================================================

Análisis del cambio dimensional:
• Columnas originales: {validation['columnas_originales']}
• Columnas finales: {validation['columnas_finales']}
• Incremento: {validation['columnas_agregadas']} columnas ({(validation['columnas_agregadas']/validation['columnas_originales']*100):.1f}%)

Distribución del incremento por estrategia:
"""
    
    # Calcular columnas agregadas por estrategia
    columns_by_strategy = {}
    for column, metadata in encoding_metadata.items():
        strategy = metadata.get('encoding_type', 'unknown')
        if strategy == 'one_hot_encoding':
            new_cols = len(metadata.get('new_columns', []))
            columns_by_strategy[strategy] = columns_by_strategy.get(strategy, 0) + new_cols
        elif strategy in ['label_encoding', 'ordinal_encoding', 'frequency_encoding']:
            columns_by_strategy[strategy] = columns_by_strategy.get(strategy, 0) + 1
    
    for strategy, count in columns_by_strategy.items():
        strategy_name = {
            'label_encoding': 'Label Encoding',
            'ordinal_encoding': 'Ordinal Encoding',
            'one_hot_encoding': 'One-Hot Encoding', 
            'frequency_encoding': 'Frequency Encoding'
        }.get(strategy, strategy)
        report += f"• {strategy_name}: +{count} columnas\n"
    
    # Validación de integridad
    report += f"""
================================================================================
VALIDACIÓN DE INTEGRIDAD DE DATOS
================================================================================

Verificaciones realizadas:
✅ Número de filas: {validation['filas_originales']:,} → {validation['filas_finales']:,} 
   {'(Sin pérdida)' if validation['integridad_filas'] else '(⚠️ PÉRDIDA DETECTADA)'}

✅ Variable objetivo preservada: {'Sí' if validation['variable_objetivo_preservada'] else 'No'}
   {'(Distribución mantenida)' if validation.get('distribucion_objetivo_preservada', True) else '(⚠️ Distribución alterada)'}

✅ Variables categóricas procesadas: {validation['variables_encoded']}/{len(categorical_analysis)}

✅ Nuevas variables numéricas disponibles para ML: {validation['columnas_finales'] - validation['columnas_originales']}

================================================================================
RECOMENDACIONES PARA SIGUIENTE PASO
================================================================================

Basado en el análisis realizado:

1. SELECCIÓN DE CARACTERÍSTICAS:
   • Considerar eliminar variables con VIF > 10 si se detectaron
   • Evaluar variables con correlación muy baja con objetivo (<0.1)
   • Revisar pares de variables con correlación > 0.9

2. PREPARACIÓN PARA MODELADO:
   • Dataset listo para algoritmos de ML
   • {validation['columnas_finales']} características disponibles
   • Variable objetivo balanceada: verificar distribución

3. PRÓXIMOS PASOS SUGERIDOS:
   • Paso 3: Normalización/Escalado de variables numéricas
   • Paso 4: División en conjuntos de entrenamiento/validación/test
   • Paso 5: Selección de características (feature selection)

================================================================================
ARCHIVOS GENERADOS
================================================================================

• Dataset procesado: excel/telecomx_paso2_encoding_aplicado_{timestamp}.csv
• Gráficos: graficos/paso2_correlaciones_objetivo_{timestamp}.png
• Gráficos: graficos/paso2_distribucion_objetivo_{timestamp}.png
• Informe: informes/paso2_encoding_variables_categoricas_informe_{timestamp}.txt

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_files(df_encoded, report_content, timestamp):
    """
    Guarda el dataset procesado y el informe en sus respectivas carpetas
    
    Args:
        df_encoded (pd.DataFrame): Dataset con encoding aplicado
        report_content (str): Contenido del informe
        timestamp (str): Timestamp para nombres de archivo
    """
    logger = logging.getLogger(__name__)
    
    # Guardar dataset procesado
    excel_filename = f"excel/telecomx_paso2_encoding_aplicado_{timestamp}.csv"
    df_encoded.to_csv(excel_filename, sep=';', index=False, encoding='utf-8-sig')
    logger.info(f"Dataset procesado guardado: {excel_filename}")
    
    # Guardar informe
    report_filename = f"informes/paso2_encoding_variables_categoricas_informe_{timestamp}.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logger.info(f"Informe guardado: {report_filename}")
    
    return excel_filename, report_filename

def main():
    """Función principal que ejecuta todo el proceso de encoding"""
    
    # Crear directorios y configurar logging
    create_directories()
    logger = setup_logging()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("INICIANDO PASO 2: ENCODING DE VARIABLES CATEGORICAS")
    logger.info("=" * 70)
    
    try:
        # 1. Encontrar y cargar el archivo del Paso 1
        input_file = find_latest_paso1_file()
        df_original = load_data(input_file)
        
        # 2. Analizar variables categóricas
        categorical_analysis = analyze_categorical_variables(df_original)
        
        # 3. Aplicar estrategias de encoding
        df_encoded, encoding_metadata = perform_encoding(df_original, categorical_analysis)
        
        # 4. Obtener lista de columnas nuevas para análisis
        all_new_columns = []
        for metadata in encoding_metadata.values():
            if 'new_column' in metadata:
                all_new_columns.append(metadata['new_column'])
            elif 'new_columns' in metadata:
                all_new_columns.extend(metadata['new_columns'])
        
        # 5. Análisis de correlaciones
        correlation_analysis = analyze_correlations(df_encoded, all_new_columns)
        
        # 6. Análisis de VIF (multicolinealidad)
        vif_analysis = calculate_vif(df_encoded, all_new_columns)
        
        # 7. Validar integridad del proceso
        validation = validate_encoding_integrity(df_original, df_encoded, encoding_metadata)
        
        # 8. Generar visualizaciones
        generate_visualizations(df_encoded, correlation_analysis, timestamp)
        
        # 9. Generar informe detallado
        report_content = generate_report(
            categorical_analysis, encoding_metadata, correlation_analysis, 
            vif_analysis, validation, timestamp
        )
        
        # 10. Guardar archivos resultantes
        excel_file, report_file = save_files(df_encoded, report_content, timestamp)
        
        # 11. Resumen final
        logger.info("=" * 70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info(f"Variables categóricas procesadas: {validation['variables_encoded']}")
        logger.info(f"Dimensionalidad: {validation['columnas_originales']} → {validation['columnas_finales']} columnas")
        logger.info(f"Archivo resultante: {excel_file}")
        logger.info(f"Informe generado: {report_file}")
        logger.info("=" * 70)
        
        print("\nSIGUIENTE PASO:")
        print("   Ejecutar Paso 3: Normalización/Escalado de Variables Numéricas")
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        raise

if __name__ == "__main__":
    main()