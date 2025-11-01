# entrenar_y_evaluar_modelo.py
"""
Script de Entrenamiento y Evaluación de Modelo de Principio a Fin.

Este script realiza los siguientes pasos:
1. Carga el conjunto de datos 'online_news_modified.csv'.
2. Prepara los datos creando una variable objetivo binaria ('popular').
3. Define un pipeline de scikit-learn para el preprocesamiento de datos, que incluye:
   - Imputación de valores faltantes.
   - Recorte de valores atípicos (outliers).
   - Escalado de características.
4. Define un pipeline de modelo completo, encadenando el preprocesamiento con el
   clasificador RandomForestClassifier optimizado.
5. Divide los datos en conjuntos de entrenamiento y prueba de forma estratificada.
6. Entrena el pipeline completo con los datos de entrenamiento.
7. Evalúa el rendimiento del modelo en el conjunto de prueba.
8. Guarda el objeto del pipeline final entrenado y las visualizaciones de rendimiento
   (matriz de confusión e importancia de características).
"""

# --- Importaciones Generales ---
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Importaciones de Scikit-learn ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# --- [CONFIGURACIÓN] - Todas las variables definidas por el usuario se encuentran aquí ---
# ===============================================================
RUTA_DATOS = '../data/raw/online_news_modified.csv'
VARIABLE_OBJETIVO = 'popular'
COLUMNAS_A_ELIMINAR = ['timedelta', 'shares', 'popular']
PROPORCION_PRUEBA = 0.2
ESTADO_ALEATORIO = 42
RUTA_SALIDA_MODELO = 'pipeline_modelo_final.joblib'
MEJORES_PARAMETROS_RF = {
    'n_estimators': 356,
    'max_depth': 23,
    'random_state': ESTADO_ALEATORIO, # Asegura la reproducibilidad del modelo
    'n_jobs': -1  # Utiliza todos los núcleos de CPU disponibles
}
# ===============================================================


# --- [COMPONENTES DEL PIPELINE] - Transformadores Personalizados ---
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Un transformador personalizado de scikit-learn para recortar valores atípicos (outliers)
    usando el método de Rango Intercuartílico (IQR).
    
    Este paso es crucial para hacer que los modelos (especialmente los lineales) y los
    escaladores sean más robustos frente a valores extremos.
    """
    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        """Calcula y almacena los límites inferiores y superiores para cada columna."""
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.iqr_multiplier * IQR
            self.upper_bounds_[col] = Q3 + self.iqr_multiplier * IQR
        return self

    def transform(self, X, y=None):
        """Recorta los datos en X basándose en los límites calculados en 'fit'."""
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(
                lower=self.lower_bounds_.get(col), 
                upper=self.upper_bounds_.get(col)
            )
        return X_df.values


# --- [FUNCIONES LÓGICAS] - Pasos encapsulados del proceso ---
def cargar_y_preparar_datos(ruta: str) -> tuple:
    """Carga los datos desde una ruta, convierte tipos y crea la variable objetivo binaria."""
    print(f"Cargando datos desde {ruta}...")
    df = pd.read_csv(ruta)
    
    df_numeric = pd.DataFrame()
    for col in df.columns:
        if col != 'url':
            df_numeric[col] = pd.to_numeric(df[col], errors='coerce')

    df_numeric.dropna(subset=['shares'], inplace=True)
    
    umbral = df_numeric['shares'].median()
    df_numeric[VARIABLE_OBJETIVO] = (df_numeric['shares'] > umbral).astype(int)
    print(f"Datos cargados. Umbral de popularidad (mediana de 'shares'): {umbral:.0f}")

    X = df_numeric.drop(columns=COLUMNAS_A_ELIMINAR)
    y = df_numeric[VARIABLE_OBJETIVO]
    
    return X, y

def construir_pipeline() -> Pipeline:
    """Construye y devuelve el pipeline de modelo completo de scikit-learn."""
    print("Construyendo el pipeline de modelo completo...")
    
    pipeline_preprocesamiento = Pipeline(steps=[
        ('imputador', SimpleImputer(strategy='median')),
        ('recortador_outliers', OutlierClipper(iqr_multiplier=1.5)),
        ('escalador', StandardScaler())
    ])

    pipeline_completo = Pipeline(steps=[
        ('preprocesamiento', pipeline_preprocesamiento),
        ('clasificador', RandomForestClassifier(**MEJORES_PARAMETROS_RF))
    ])
    
    print("Pipeline creado exitosamente:")
    print(pipeline_completo)
    return pipeline_completo

def evaluar_modelo(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, nombres_caracteristicas: list):
    """Evalúa el pipeline entrenado y guarda las visualizaciones de rendimiento."""
    print("\nEvaluando el rendimiento del modelo en el conjunto de prueba...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nAccuracy Final en Prueba: {accuracy:.4f}")
    print(f"AUC Final en Prueba:      {auc:.4f}")
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred, target_names=['No Popular', 'Popular']))

    # --- Generar y Guardar Visualizaciones ---
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Popular', 'Popular'], yticklabels=['No Popular', 'Popular'])
    plt.title('Matriz de Confusión en el Conjunto de Prueba')
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.savefig("matriz_confusion.png")
    print("\nMatriz de confusión guardada en matriz_confusion.png")

    # Importancia de Características
    importancias = pipeline.named_steps['clasificador'].feature_importances_
    df_importancia = pd.DataFrame({
        'Característica': nombres_caracteristicas, 
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importancia', y='Característica', data=df_importancia.head(20))
    plt.title('Top 20 Características más Importantes')
    plt.tight_layout()
    plt.savefig("importancia_caracteristicas.png")
    print("Gráfico de importancia de características guardado en importancia_caracteristicas.png")


# --- [EJECUCIÓN PRINCIPAL] - Orquesta todo el flujo de trabajo ---
def main():
    """Función principal para ejecutar el proceso de entrenamiento y evaluación del modelo."""
    # 1. Cargar Datos
    X, y = cargar_y_preparar_datos(RUTA_DATOS)
    
    # 2. Dividir Datos
    # Usamos stratify=y para asegurar que los conjuntos de entrenamiento y prueba tengan
    # la misma proporción de artículos populares/no populares que el dataset original.
    # Esto es crucial para la correcta evaluación del modelo.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=ESTADO_ALEATORIO, stratify=y
    )
    
    # 3. Construir Pipeline
    pipeline_modelo = construir_pipeline()
    
    # 4. Entrenar Modelo
    print("\nEntrenando el modelo final...")
    pipeline_modelo.fit(X_train, y_train)
    print("Entrenamiento completado.")
    
    # 5. Evaluar Modelo
    evaluar_modelo(pipeline_modelo, X_test, y_test, nombres_caracteristicas=X.columns.tolist())
    
    # 6. Guardar Modelo Final
    print(f"\nGuardando el pipeline entrenado en '{RUTA_SALIDA_MODELO}'...")
    joblib.dump(pipeline_modelo, RUTA_SALIDA_MODELO)
    print("\n✅ ¡Proceso de entrenamiento, evaluación y guardado del modelo completado exitosamente!")

if __name__ == "__main__":
    main()