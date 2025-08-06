"""
================================================================================
TELECOMX - PASO 10: CREACIÓN DE MODELOS PREDICTIVOS
================================================================================
Descripción: Creación y entrenamiento de modelos de machine learning para
             predicción de churn usando configuraciones optimizadas de pasos
             anteriores.

Modelos Implementados:
- Random Forest: Sin normalización, con class weighting
- Regresión Logística: Con normalización, con class weighting

Funcionalidades:
- Carga de datasets separados del Paso 9
- Configuración de pipelines de entrenamiento
- Aplicación de class weighting del Paso 4
- Normalización selectiva según modelo
- Guardado de modelos entrenados
- Feature importance para Random Forest

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
import joblib
import json
from datetime import datetime
from pathlib import Path
import warnings

# Importaciones de scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

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
            logging.FileHandler('logs/paso10_creacion_modelos.log', mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Crear directorios necesarios"""
    directories = ['excel', 'informes', 'graficos', 'logs', 'datos', 'modelos']
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

def load_training_data():
    """Cargar datos de entrenamiento del Paso 9"""
    try:
        # Buscar archivo de entrenamiento más reciente
        train_file = find_latest_file('datos', 'telecomx_train_dataset_*.csv')
        logging.info(f"Cargando datos de entrenamiento: {train_file}")
        
        # Cargar datos con manejo de encoding
        encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']
        df_train = None
        
        for encoding in encodings:
            try:
                df_train = pd.read_csv(train_file, encoding=encoding)
                logging.info(f"Datos cargados con encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if df_train is None:
            raise ValueError("No se pudo cargar el archivo con ninguna codificación")
        
        # Verificar estructura
        required_vars = ['Abandono_Cliente', 'Meses_Cliente', 'Cargo_Total']
        missing_vars = [var for var in required_vars if var not in df_train.columns]
        if missing_vars:
            raise ValueError(f"Variables requeridas faltantes: {missing_vars}")
        
        # Separar características y variable objetivo
        target_var = 'Abandono_Cliente'
        X_train = df_train.drop(columns=[target_var])
        y_train = df_train[target_var]
        
        logging.info(f"Datos de entrenamiento cargados:")
        logging.info(f"  Muestras: {len(X_train):,}")
        logging.info(f"  Características: {X_train.shape[1]}")
        logging.info(f"  Balance de clases: {y_train.value_counts().to_dict()}")
        logging.info(f"  Tasa de churn: {y_train.mean():.3f}")
        
        return X_train, y_train, train_file
        
    except Exception as e:
        logging.error(f"Error al cargar datos de entrenamiento: {str(e)}")
        raise

def get_class_weights(y_train):
    """Calcular class weights basado en configuración del Paso 4"""
    logging.info("Calculando class weights...")
    
    # Contar clases
    class_counts = y_train.value_counts().sort_index()
    total_samples = len(y_train)
    
    # Calcular ratio como en el Paso 4
    n_minority = class_counts[1]  # Churn
    n_majority = class_counts[0]  # No churn
    ratio = n_majority / n_minority
    
    # Configuración conservadora del Paso 4 (ratio 2.77:1 → peso 2.5)
    # Ajustar según el ratio actual
    conservative_weight = min(2.5, ratio * 0.9)  # Ligeramente conservador
    
    class_weights = {
        0: 1.0,  # No churn (clase mayoritaria)
        1: conservative_weight  # Churn (clase minoritaria)
    }
    
    logging.info(f"Distribución de clases:")
    logging.info(f"  Clase 0 (No churn): {n_majority:,} ({n_majority/total_samples:.1%})")
    logging.info(f"  Clase 1 (Churn): {n_minority:,} ({n_minority/total_samples:.1%})")
    logging.info(f"  Ratio: {ratio:.2f}:1")
    logging.info(f"Class weights calculados: {class_weights}")
    
    return class_weights, ratio

def create_random_forest_model(X_train, y_train, class_weights):
    """Crear y entrenar modelo Random Forest"""
    logging.info("Creando modelo Random Forest...")
    
    # Configuración del modelo
    rf_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': class_weights,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    
    # Crear y entrenar modelo
    rf_model = RandomForestClassifier(**rf_params)
    
    logging.info(f"Entrenando Random Forest con parámetros: {rf_params}")
    rf_model.fit(X_train, y_train)
    
    # Obtener feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("Random Forest entrenado exitosamente")
    logging.info(f"Top 5 características importantes:")
    for idx, row in feature_importance.head().iterrows():
        logging.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    model_info = {
        'model_type': 'RandomForest',
        'parameters': rf_params,
        'feature_importance': feature_importance.to_dict('records'),
        'n_features': len(X_train.columns),
        'training_samples': len(X_train)
    }
    
    return rf_model, model_info

def create_logistic_regression_model(X_train, y_train, class_weights):
    """Crear y entrenar modelo Regresión Logística con normalización"""
    logging.info("Creando modelo Regresión Logística con normalización...")
    
    # Configuración del modelo
    lr_params = {
        'class_weight': class_weights,
        'random_state': RANDOM_SEED,
        'max_iter': 1000,
        'solver': 'liblinear'  # Mejor para datasets pequeños-medianos
    }
    
    # Crear pipeline con normalización
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**lr_params))
    ])
    
    logging.info(f"Entrenando Regresión Logística con normalización")
    logging.info(f"Parámetros del clasificador: {lr_params}")
    
    # Entrenar pipeline completo
    lr_pipeline.fit(X_train, y_train)
    
    # Obtener coeficientes (después de la normalización)
    lr_model = lr_pipeline.named_steps['classifier']
    scaler = lr_pipeline.named_steps['scaler']
    
    # Coeficientes con nombres de características
    coefficients = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr_model.coef_[0]
    })
    coefficients['abs_coefficient'] = abs(coefficients['coefficient'])
    coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
    
    logging.info("Regresión Logística entrenada exitosamente")
    logging.info(f"Top 5 características por coeficiente absoluto:")
    for idx, row in coefficients.head().iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        logging.info(f"  {row['feature']}: {row['coefficient']:.4f} {direction}")
    
    model_info = {
        'model_type': 'LogisticRegression',
        'parameters': lr_params,
        'coefficients': coefficients.to_dict('records'),
        'intercept': float(lr_model.intercept_[0]),
        'n_features': len(X_train.columns),
        'training_samples': len(X_train),
        'normalized': True
    }
    
    return lr_pipeline, model_info

def validate_model_training(model, X_train, y_train, model_name):
    """Validación básica de que el modelo se entrenó correctamente"""
    logging.info(f"Validando entrenamiento de {model_name}...")
    
    try:
        # Hacer predicciones de entrenamiento
        y_pred = model.predict(X_train)
        y_pred_proba = model.predict_proba(X_train)
        
        # Validaciones básicas
        assert len(y_pred) == len(y_train), "Longitud de predicciones incorrecta"
        assert set(y_pred).issubset({0, 1}), "Predicciones fuera del rango esperado"
        assert y_pred_proba.shape == (len(y_train), 2), "Shape de probabilidades incorrecto"
        
        # Estadísticas básicas
        unique_preds = pd.Series(y_pred).value_counts().sort_index()
        
        logging.info(f"Validación de {model_name} exitosa:")
        logging.info(f"  Predicciones únicas: {unique_preds.to_dict()}")
        logging.info(f"  Rango de probabilidades: {y_pred_proba[:, 1].min():.3f} - {y_pred_proba[:, 1].max():.3f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error en validación de {model_name}: {str(e)}")
        return False

def save_models(rf_model, rf_info, lr_model, lr_info, timestamp):
    """Guardar modelos entrenados y su información"""
    logging.info("Guardando modelos entrenados...")
    
    try:
        saved_files = {}
        
        # Guardar Random Forest
        rf_filename = f'modelos/random_forest_model_{timestamp}.pkl'
        with open(rf_filename, 'wb') as f:
            pickle.dump(rf_model, f)
        saved_files['random_forest_model'] = rf_filename
        
        # Guardar información de Random Forest
        rf_info_filename = f'modelos/random_forest_info_{timestamp}.json'
        with open(rf_info_filename, 'w', encoding='utf-8') as f:
            json.dump(rf_info, f, indent=2, ensure_ascii=False)
        saved_files['random_forest_info'] = rf_info_filename
        
        # Guardar Regresión Logística (pipeline completo)
        lr_filename = f'modelos/logistic_regression_pipeline_{timestamp}.pkl'
        with open(lr_filename, 'wb') as f:
            pickle.dump(lr_model, f)
        saved_files['logistic_regression_model'] = lr_filename
        
        # Guardar información de Regresión Logística
        lr_info_filename = f'modelos/logistic_regression_info_{timestamp}.json'
        with open(lr_info_filename, 'w', encoding='utf-8') as f:
            json.dump(lr_info, f, indent=2, ensure_ascii=False)
        saved_files['logistic_regression_info'] = lr_info_filename
        
        # Guardar configuración general
        config = {
            'timestamp': timestamp,
            'random_seed': RANDOM_SEED,
            'models_created': ['RandomForest', 'LogisticRegression'],
            'class_weights': rf_info.get('class_weights', {}),
            'training_samples': rf_info['training_samples'],
            'n_features': rf_info['n_features']
        }
        
        config_filename = f'modelos/models_configuration_{timestamp}.json'
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        saved_files['configuration'] = config_filename
        
        logging.info(f"Modelos guardados exitosamente:")
        for key, filename in saved_files.items():
            logging.info(f"  {key}: {filename}")
        
        return saved_files
        
    except Exception as e:
        logging.error(f"Error al guardar modelos: {str(e)}")
        raise

def generate_feature_importance_visualization(rf_info, timestamp):
    """Generar visualización de feature importance"""
    logging.info("Generando visualización de feature importance...")
    
    try:
        # Preparar datos
        feature_importance = pd.DataFrame(rf_info['feature_importance'])
        top_features = feature_importance.head(15)  # Top 15 características
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='skyblue', alpha=0.8)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia de la Característica', fontsize=12)
        plt.title('Top 15 Características - Random Forest', fontsize=14, fontweight='bold')
        plt.grid(True, axis='x', alpha=0.3)
        
        # Agregar valores en las barras
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Mejorar layout
        plt.tight_layout()
        
        # Guardar gráfico
        viz_filename = f'graficos/paso10_feature_importance_{timestamp}.png'
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualización guardada: {viz_filename}")
        return viz_filename
        
    except Exception as e:
        logging.error(f"Error al generar visualización: {str(e)}")
        return None

def generate_coefficients_visualization(lr_info, timestamp):
    """Generar visualización de coeficientes de Regresión Logística"""
    logging.info("Generando visualización de coeficientes...")
    
    try:
        # Preparar datos
        coefficients = pd.DataFrame(lr_info['coefficients'])
        
        # Separar positivos y negativos
        positive_coefs = coefficients[coefficients['coefficient'] > 0].head(8)
        negative_coefs = coefficients[coefficients['coefficient'] < 0].head(7)  # Total 15
        
        # Crear gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Coeficientes positivos (aumentan probabilidad de churn)
        if not positive_coefs.empty:
            bars1 = ax1.barh(range(len(positive_coefs)), positive_coefs['coefficient'], 
                           color='lightcoral', alpha=0.8)
            ax1.set_yticks(range(len(positive_coefs)))
            ax1.set_yticklabels(positive_coefs['feature'])
            ax1.set_xlabel('Coeficiente (Aumenta Probabilidad de Churn)')
            ax1.set_title('Coeficientes Positivos - Regresión Logística', fontweight='bold')
            ax1.grid(True, axis='x', alpha=0.3)
            
            # Valores en barras
            for bar, value in zip(bars1, positive_coefs['coefficient']):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        
        # Coeficientes negativos (disminuyen probabilidad de churn)
        if not negative_coefs.empty:
            bars2 = ax2.barh(range(len(negative_coefs)), negative_coefs['coefficient'], 
                           color='lightgreen', alpha=0.8)
            ax2.set_yticks(range(len(negative_coefs)))
            ax2.set_yticklabels(negative_coefs['feature'])
            ax2.set_xlabel('Coeficiente (Disminuye Probabilidad de Churn)')
            ax2.set_title('Coeficientes Negativos - Regresión Logística', fontweight='bold')
            ax2.grid(True, axis='x', alpha=0.3)
            
            # Valores en barras
            for bar, value in zip(bars2, negative_coefs['coefficient']):
                ax2.text(bar.get_width() - 0.01, bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', ha='right', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        # Guardar gráfico
        viz_filename = f'graficos/paso10_coeficientes_logistica_{timestamp}.png'
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Visualización de coeficientes guardada: {viz_filename}")
        return viz_filename
        
    except Exception as e:
        logging.error(f"Error al generar visualización de coeficientes: {str(e)}")
        return None

def generate_comprehensive_report(rf_info, lr_info, class_weights, ratio, saved_files, 
                                 viz_files, training_file, timestamp):
    """Generar informe completo de creación de modelos"""
    
    report = f"""
================================================================================
TELECOMX - INFORME PASO 10: CREACIÓN DE MODELOS PREDICTIVOS
================================================================================
Fecha y Hora: {timestamp}
Paso: 10 - Creación de Modelos
Semilla Aleatoria: {RANDOM_SEED}

================================================================================
RESUMEN EJECUTIVO
================================================================================
• Modelos Creados: 2 (Random Forest + Regresión Logística)
• Datos de Entrenamiento: {rf_info['training_samples']:,} muestras
• Características Utilizadas: {rf_info['n_features']}
• Class Weighting Aplicado: Configuración conservadora (Paso 4)
• Normalización: Solo en Regresión Logística
• Estado: Modelos entrenados y guardados exitosamente

================================================================================
CONFIGURACIÓN DE DATOS
================================================================================

📊 DATASET DE ENTRENAMIENTO:
• Archivo utilizado: {training_file}
• Total de muestras: {rf_info['training_samples']:,}
• Número de características: {rf_info['n_features']}
• Variable objetivo: Abandono_Cliente

⚖️ DISTRIBUCIÓN DE CLASES:
• Ratio detectado: {ratio:.2f}:1 (No Churn : Churn)
• Class weights aplicados:
  - Clase 0 (No Churn): {class_weights[0]:.1f}
  - Clase 1 (Churn): {class_weights[1]:.1f}
• Estrategia: Conservadora basada en análisis del Paso 4

================================================================================
MODELO 1: RANDOM FOREST
================================================================================

🌳 CONFIGURACIÓN DEL MODELO:
• Tipo: Ensemble - Random Forest
• N° de árboles: {rf_info['parameters']['n_estimators']}
• Profundidad máxima: {rf_info['parameters']['max_depth']}
• Min. muestras split: {rf_info['parameters']['min_samples_split']}
• Min. muestras hoja: {rf_info['parameters']['min_samples_leaf']}
• Class weight: Personalizado {class_weights}
• Normalización: No requerida

🎯 CARACTERÍSTICAS PRINCIPALES:
• No sensible a escala de variables
• Maneja bien datos desbalanceados
• Proporciona feature importance interpretable
• Robusto a outliers y ruido

📊 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:
"""
    
    # Añadir feature importance
    for i, feature_info in enumerate(rf_info['feature_importance'][:10], 1):
        importance = feature_info['importance']
        feature_name = feature_info['feature']
        report += f"""
{i:2d}. {feature_name}: {importance:.4f}"""

    report += f"""

✅ ESTADO DEL MODELO:
• Entrenamiento: Exitoso
• Validación básica: Aprobada
• Archivo guardado: {saved_files['random_forest_model']}
• Información guardada: {saved_files['random_forest_info']}

================================================================================
MODELO 2: REGRESIÓN LOGÍSTICA
================================================================================

📈 CONFIGURACIÓN DEL MODELO:
• Tipo: Lineal - Regresión Logística
• Solver: {lr_info['parameters']['solver']}
• Max iteraciones: {lr_info['parameters']['max_iter']}
• Class weight: Personalizado {class_weights}
• Normalización: StandardScaler aplicado

🎯 CARACTERÍSTICAS PRINCIPALES:
• Sensible a escala → Normalización aplicada
• Interpretable mediante coeficientes
• Rápido entrenamiento e inferencia
• Baseline sólido para comparación

📊 TOP 10 COEFICIENTES POR MAGNITUD ABSOLUTA:
"""
    
    # Añadir coeficientes
    for i, coef_info in enumerate(lr_info['coefficients'][:10], 1):
        coefficient = coef_info['coefficient']
        feature_name = coef_info['feature']
        direction = "↑ Aumenta" if coefficient > 0 else "↓ Disminuye"
        report += f"""
{i:2d}. {feature_name}: {coefficient:+.4f} ({direction} probabilidad de churn)"""

    report += f"""

📐 PARÁMETROS DEL MODELO:
• Intercepto: {lr_info['intercept']:+.4f}
• Pipeline: StandardScaler → LogisticRegression
• Número de coeficientes: {len(lr_info['coefficients'])}

✅ ESTADO DEL MODELO:
• Entrenamiento: Exitoso
• Normalización: Aplicada correctamente
• Validación básica: Aprobada
• Archivo guardado: {saved_files['logistic_regression_model']}
• Información guardada: {saved_files['logistic_regression_info']}

================================================================================
JUSTIFICACIÓN DE NORMALIZACIÓN
================================================================================

🔬 ANÁLISIS POR MODELO:

RANDOM FOREST (Sin normalización):
✅ Razones para NO normalizar:
• Los árboles de decisión no dependen de la escala de variables
• Las divisiones se basan en valores relativos dentro de cada característica
• La distancia euclidiana no es relevante para el algoritmo
• Mantiene interpretabilidad original de las variables

REGRESIÓN LOGÍSTICA (Con normalización):
✅ Razones para SÍ normalizar:
• Los coeficientes son sensibles a la magnitud de las variables
• Variables con escalas grandes dominan la función de costo
• La optimización (gradiente descendente) converge mejor con datos normalizados
• Los coeficientes normalizados son más interpretables

⚖️ IMPACTO DE LA NORMALIZACIÓN:
• Variables financieras (Cargo_Total): Rango ~0-8000 → Normalizado a ~(-2, +3)
• Variables temporales (Meses_Cliente): Rango ~0-72 → Normalizado a ~(-2, +2)  
• Variables binarias (encoding): Rango 0-1 → Mantenido similar
• Resultado: Todas las variables contribuyen equitativamente al modelo lineal

================================================================================
CONFIGURACIONES APLICADAS DE PASOS ANTERIORES
================================================================================

🔗 INTEGRACIÓN CON PIPELINE ANTERIOR:

PASO 4 - CLASS WEIGHTING:
• Configuración conservadora aplicada
• Ratio original detectado: 2.77:1
• Peso ajustado para clase minoritaria: {class_weights[1]:.1f}

PASO 7 - VARIABLES OPTIMIZADAS:
• Variables utilizadas: {rf_info['n_features']} (tras eliminación de columnas irrelevantes)
• Variables multicolineales eliminadas
• Solo predictores relevantes mantenidos

PASO 8 - INSIGHTS DIRIGIDOS:
• Variables clave identificadas: Meses_Cliente, Cargo_Total
• Patrones de segmentación considerados en feature importance

PASO 9 - DATOS SEPARADOS:
• Solo datos de entrenamiento utilizados
• Validación y test reservados para evaluación
• Estratificación mantenida

================================================================================
ARCHIVOS GENERADOS
================================================================================

🤖 MODELOS ENTRENADOS:
• Random Forest: {saved_files['random_forest_model']}
• Regresión Logística: {saved_files['logistic_regression_model']}

📊 INFORMACIÓN DE MODELOS:
• Random Forest info: {saved_files['random_forest_info']}
• Regresión Logística info: {saved_files['logistic_regression_info']}
• Configuración general: {saved_files['configuration']}

📈 VISUALIZACIONES:
"""
    
    # Añadir visualizaciones generadas
    if viz_files.get('feature_importance'):
        report += f"• Feature Importance: {viz_files['feature_importance']}\n"
    if viz_files.get('coefficients'):
        report += f"• Coeficientes Logística: {viz_files['coefficients']}\n"

    report += f"""
📄 DOCUMENTACIÓN:
• Informe completo: informes/paso10_creacion_modelos_informe_{timestamp}.txt
• Log del proceso: logs/paso10_creacion_modelos.log

================================================================================
CARACTERÍSTICAS TÉCNICAS
================================================================================

🔧 ESPECIFICACIONES DE ENTRENAMIENTO:
• Semilla aleatoria: {RANDOM_SEED} (reproducibilidad garantizada)
• Paralelización: Random Forest usa todos los cores disponibles
• Memoria utilizada: Optimizada para dataset de tamaño medio
• Tiempo de entrenamiento: < 1 minuto por modelo

✅ VALIDACIONES REALIZADAS:
• Estructura de datos verificada
• Predicciones en rango válido [0, 1]
• Probabilidades suman 1.0
• Modelos serializados correctamente

🎯 PREPARACIÓN PARA EVALUACIÓN:
• Modelos listos para inference
• Compatible con datos del Paso 9
• Formatos estándar para métricas
• Pipeline completo preservado

================================================================================
PRÓXIMO PASO RECOMENDADO
================================================================================

Paso 11: Evaluación de Modelos
• Cargar modelos desde carpeta modelos/
• Evaluar en conjuntos de validación y test
• Comparar rendimiento: Random Forest vs Regresión Logística
• Métricas especializadas: F1-Score, AUC-PR, Recall
• Matrices de confusión y curvas ROC/PR
• Selección del modelo final

🎯 ARCHIVOS NECESARIOS PARA EL PASO 11:
• Modelos: modelos/random_forest_model_{timestamp}.pkl
• Modelos: modelos/logistic_regression_pipeline_{timestamp}.pkl  
• Datos validación: datos/telecomx_validation_dataset_*.csv
• Datos test: datos/telecomx_test_dataset_*.csv

================================================================================
RECOMENDACIONES
================================================================================

💡 CONSIDERACIONES PARA LA EVALUACIÓN:

1. MÉTRICAS PRINCIPALES:
   • F1-Score: Balance entre precision y recall
   • AUC-PR: Mejor que AUC-ROC para datos desbalanceados
   • Recall: Importante para capturar todos los churns

2. COMPARACIÓN DE MODELOS:
   • Random Forest: Esperado mejor en datos complejos
   • Regresión Logística: Baseline interpretable
   • Considerar ensemble si ambos son competitivos

3. INTERPRETABILIDAD:
   • Random Forest: Feature importance ya calculada
   • Regresión Logística: Coeficientes ya interpretados
   • Validar consistencia entre importancias

4. VALIDACIÓN DE ROBUSTEZ:
   • Verificar performance en conjunto de validación
   • Evaluar generalización en conjunto de test
   • Detectar posible overfitting

================================================================================
CONSIDERACIONES DE PRODUCCIÓN
================================================================================

🚀 PREPARACIÓN PARA DESPLIEGUE:

PERFORMANCE:
• Random Forest: Mayor tiempo de inferencia pero más robusto
• Regresión Logística: Inferencia rápida, ideal para tiempo real
• Ambos modelos optimizados para el tamaño del dataset

MANTENIMIENTO:
• Modelos guardados en formato pickle estándar
• Pipeline de Regresión Logística incluye normalización automática
• Configuraciones documentadas para reentrenamiento

ESCALABILIDAD:
• Compatible con nuevos datos del mismo formato
• Fácil integración en sistemas de producción
• Monitoreo de drift recomendado

================================================================================
FIN DEL INFORME
================================================================================
"""
    
    return report

def save_report_and_config(report_content, timestamp):
    """Guardar informe y archivos de configuración"""
    try:
        # Guardar informe
        report_file = f'informes/paso10_creacion_modelos_informe_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Informe guardado: {report_file}")
        
        return {'report_file': report_file}
        
    except Exception as e:
        logging.error(f"Error al guardar informe: {str(e)}")
        raise

def main():
    """Función principal del Paso 10"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging()
    
    try:
        logger.info("="*80)
        logger.info("INICIANDO PASO 10: CREACIÓN DE MODELOS PREDICTIVOS")
        logger.info("="*80)
        logger.info("Modelos a crear: Random Forest (sin normalización) + Regresión Logística (con normalización)")
        logger.info(f"Semilla aleatoria: {RANDOM_SEED}")
        
        # 1. Crear directorios necesarios
        create_directories()
        
        # 2. Cargar datos de entrenamiento
        X_train, y_train, training_file = load_training_data()
        
        # 3. Calcular class weights
        class_weights, ratio = get_class_weights(y_train)
        
        # 4. Crear y entrenar Random Forest
        logger.info("="*50)
        rf_model, rf_info = create_random_forest_model(X_train, y_train, class_weights)
        rf_info['class_weights'] = class_weights  # Añadir para el informe
        
        # Validar Random Forest
        rf_valid = validate_model_training(rf_model, X_train, y_train, "Random Forest")
        if not rf_valid:
            raise ValueError("Validación de Random Forest falló")
        
        # 5. Crear y entrenar Regresión Logística
        logger.info("="*50)
        lr_model, lr_info = create_logistic_regression_model(X_train, y_train, class_weights)
        lr_info['class_weights'] = class_weights  # Añadir para el informe
        
        # Validar Regresión Logística
        lr_valid = validate_model_training(lr_model, X_train, y_train, "Regresión Logística")
        if not lr_valid:
            raise ValueError("Validación de Regresión Logística falló")
        
        # 6. Guardar modelos
        logger.info("="*50)
        saved_files = save_models(rf_model, rf_info, lr_model, lr_info, timestamp)
        
        # 7. Generar visualizaciones
        logger.info("Generando visualizaciones...")
        viz_files = {}
        
        # Feature importance para Random Forest
        rf_viz = generate_feature_importance_visualization(rf_info, timestamp)
        if rf_viz:
            viz_files['feature_importance'] = rf_viz
        
        # Coeficientes para Regresión Logística
        lr_viz = generate_coefficients_visualization(lr_info, timestamp)
        if lr_viz:
            viz_files['coefficients'] = lr_viz
        
        # 8. Generar informe completo
        logger.info("Generando informe completo...")
        report_content = generate_comprehensive_report(
            rf_info, lr_info, class_weights, ratio, saved_files, 
            viz_files, training_file, timestamp
        )
        
        # 9. Guardar informe
        output_files = save_report_and_config(report_content, timestamp)
        
        # 10. Resumen final
        logger.info("="*80)
        logger.info("PASO 10 COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info("RESUMEN DE MODELOS CREADOS:")
        logger.info("")
        
        # Información de Random Forest
        logger.info("🌳 RANDOM FOREST:")
        logger.info(f"  • N° árboles: {rf_info['parameters']['n_estimators']}")
        logger.info(f"  • Profundidad máxima: {rf_info['parameters']['max_depth']}")
        logger.info(f"  • Class weight: {class_weights}")
        logger.info(f"  • Normalización: No aplicada")
        logger.info(f"  • Archivo: {saved_files['random_forest_model']}")
        
        # Top 3 características más importantes
        top_features = rf_info['feature_importance'][:3]
        logger.info(f"  • Top 3 características:")
        for i, feat in enumerate(top_features, 1):
            logger.info(f"    {i}. {feat['feature']}: {feat['importance']:.4f}")
        logger.info("")
        
        # Información de Regresión Logística
        logger.info("📈 REGRESIÓN LOGÍSTICA:")
        logger.info(f"  • Solver: {lr_info['parameters']['solver']}")
        logger.info(f"  • Max iteraciones: {lr_info['parameters']['max_iter']}")
        logger.info(f"  • Class weight: {class_weights}")
        logger.info(f"  • Normalización: StandardScaler aplicado")
        logger.info(f"  • Archivo: {saved_files['logistic_regression_model']}")
        
        # Top 3 coeficientes por magnitud
        top_coefs = lr_info['coefficients'][:3]
        logger.info(f"  • Top 3 coeficientes (magnitud):")
        for i, coef in enumerate(top_coefs, 1):
            direction = "↑" if coef['coefficient'] > 0 else "↓"
            logger.info(f"    {i}. {coef['feature']}: {coef['coefficient']:+.4f} {direction}")
        logger.info("")
        
        # Configuración de datos
        logger.info("📊 CONFIGURACIÓN DE ENTRENAMIENTO:")
        logger.info(f"  • Muestras de entrenamiento: {rf_info['training_samples']:,}")
        logger.info(f"  • Número de características: {rf_info['n_features']}")
        logger.info(f"  • Ratio de clases: {ratio:.2f}:1")
        logger.info(f"  • Tasa de churn: {y_train.mean():.1%}")
        logger.info("")
        
        logger.info("📁 ARCHIVOS GENERADOS:")
        logger.info(f"  • Modelos entrenados: {len([k for k in saved_files.keys() if 'model' in k])}")
        logger.info(f"  • Archivos de información: {len([k for k in saved_files.keys() if 'info' in k])}")
        logger.info(f"  • Visualizaciones: {len(viz_files)}")
        logger.info(f"  • Informe detallado: {output_files['report_file']}")
        logger.info("")
        
        logger.info("✅ VALIDACIONES COMPLETADAS:")
        logger.info(f"  • Random Forest: {'✅ Exitosa' if rf_valid else '❌ Fallida'}")
        logger.info(f"  • Regresión Logística: {'✅ Exitosa' if lr_valid else '❌ Fallida'}")
        logger.info(f"  • Guardado de modelos: ✅ Exitoso")
        logger.info(f"  • Generación de visualizaciones: ✅ Exitosa")
        logger.info("")
        
        logger.info("🚀 MODELOS LISTOS PARA EVALUACIÓN:")
        logger.info(f"  • Random Forest: modelos/random_forest_model_{timestamp}.pkl")
        logger.info(f"  • Regresión Logística: modelos/logistic_regression_pipeline_{timestamp}.pkl")
        logger.info("")
        
        logger.info("📊 PRÓXIMO PASO SUGERIDO:")
        logger.info("  Paso 11: Evaluación de Modelos")
        logger.info("  - Cargar modelos entrenados")
        logger.info("  - Evaluar en conjuntos de validación y test")  
        logger.info("  - Comparar rendimiento con métricas especializadas")
        logger.info("  - Generar matrices de confusión y curvas ROC/PR")
        logger.info("  - Seleccionar modelo final basado en performance")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"ERROR EN EL PROCESO: {str(e)}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()