# 🏡 Predicción de Precios de Viviendas en California

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-yellow.svg)

## 📈 Problema de Negocio

Este proyecto tiene como objetivo predecir los precios de las viviendas en California utilizando datos geoespaciales y características de las propiedades. Se exploran modelos de machine learning avanzados para ayudar a comprender cómo varían los precios según la ubicación y las características de las viviendas, proporcionando insights valiosos para el mercado inmobiliario.

## ❓ Preguntas Clave de Investigación

- 🔍 **Análisis Inicial**: ¿Qué insights podemos obtener del análisis exploratorio inicial del conjunto de datos?
- 🛠️ **Transformaciones**: ¿Qué preprocesamiento es necesario para preparar los datos correctamente?
- 🌍 **Influencia Geográfica**: ¿Cómo afecta la ubicación (latitud, longitud, distancia a la costa) a los precios de las viviendas?
- 🏠 **Características de las Viviendas**: ¿Qué características como el número de habitaciones o el tamaño de las viviendas influyen más en los precios?
- 📊 **Feature Engineering**: ¿Qué variables derivadas pueden mejorar significativamente las predicciones?
- 🤖 **Comparación de Modelos**: ¿Qué algoritmo ofrece el mejor rendimiento para este problema específico?

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Archivos Principales](#archivos-principales)
- [Instalación](#instalación)
- [Variables del Dataset](#variables-del-dataset)
- [Metodología](#metodología)
- [Hallazgos Clave](#hallazgos-clave)
- [Resultados](#resultados)
- [Uso](#uso)
- [Contribución](#contribución)
- [Licencia](#licencia)

## 🏗️ Estructura del Proyecto

El análisis se organiza en **4 etapas principales**:

### **1. 📊 Exploración de Datos**
- Análisis del dataset `California_Houses.csv`
- Identificación de variables clave y patrones
- Visualización de relaciones entre variables
- Determinación del impacto de la posición geográfica en los precios
- Mapas interactivos con Folium

### **2. 🔧 Preprocesamiento Avanzado**
- Limpieza y tratamiento de valores ausentes
- **Feature Engineering avanzado**:
  - Variables de interacción (`Median_Income × Geographic_Desirability`)
  - Transformaciones polinómicas (cuadrados, cubos)
  - Ratios geográficos y de vivienda
  - Variables de binning inteligente
- Manejo de valores atípicos con winsorización
- Selección automática de variables con múltiples métodos
- Normalización y escalado (StandardScaler)

### **3. 🤖 Modelado y Algoritmos**
Implementación de algoritmos de Machine Learning especializados:

- **🚀 XGBoost**: 
  - Enfoque avanzado de predicción que utiliza boosting de gradiente
  - Optimizado para manejar datos complejos de manera eficiente
  - Regularización L1 y L2 incorporada
  - Manejo automático de valores faltantes

- **🌲 Random Forest**: 
  - Ensamble de árboles de decisión con votación
  - Resistente al overfitting
  - Importancia de variables incorporada

- **⚡ LightGBM**: 
  - Algoritmo de boosting basado en gradiente ultrarrápido
  - Destacado por su rapidez y capacidad de manejo de grandes volúmenes
  - Optimización de memoria y velocidad de entrenamiento
  - Early stopping automático

**Técnicas de Optimización:**
- RandomizedSearchCV con 30-50 iteraciones
- Validación cruzada de 5 folds
- Early stopping para prevenir overfitting
- Análisis de convergencia en tiempo real

**Métricas de Evaluación:**
- **RMSE** (Root Mean Squared Error): Error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio  
- **R²** (Coeficiente de determinación): Varianza explicada
- **Análisis de residuos** por rangos de precio

### **4. 📈 Evaluación y Selección del Modelo**
- Comparación exhaustiva de modelos
- Análisis de importancia de variables
- Análisis de residuos por rangos de precio
- Selección del mejor modelo para producción

## 📁 Archivos Principales

```
├── Prediccion_precios_vivienda_CA.ipynb    # Notebook principal con análisis completo
├── California_Houses.csv                   # Dataset original (no incluido)
├── requirements.txt                        # Dependencias del proyecto
├── mapas_california			    # Visualizacion de datos geoespaciales 
└── README.md                              # Este archivo
```

### Dependencias principales

```python
# Análisis y manipulación de datos
pandas==2.0.3
numpy==1.24.3

# Visualización y mapas
folium==0.14.0
geopy==2.3.0

# Machine Learning
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0

# Preprocesamiento
imbalanced-learn==0.11.0
```

## 📊 Variables del Dataset

El dataset contiene **20,640 registros** con **14 características**:

| Variable | Descripción |
|----------|-------------|
| `Median_House_Value` | Precio medio de la vivienda (target) |
| `Median_Income` | Ingreso medio de la zona |
| `Median_Age` | Edad media de las viviendas |
| `Tot_Rooms` | Total de habitaciones |
| `Tot_Bedrooms` | Total de dormitorios |
| `Population` | Población |
| `Households` | Hogares |
| `Latitude` | Latitud |
| `Longitude` | Longitud |
| `Distance_to_coast` | Distancia a la costa  |
| `Distance_to_LA` | Distancia a Los Ángeles  |
| `Distance_to_SanDiego` | Distancia a San Diego  |
| `Distance_to_SanJose` | Distancia a San José  |
| `Distance_to_SanFrancisco` | Distancia a San Francisco  | 

## 🔬 Metodología

### Feature Engineering Avanzado
El proyecto implementa técnicas avanzadas de ingeniería de características:

1. **Variables de Interacción**:
   - `Median_Income × Geographic_Desirability` (correlación: 0.787)
   - `Median_Income × Log_Distance_Coast`
   - `Geographic_Desirability × Log_Distance_Coast`

2. **Transformaciones Polinómicas**:
   - `Median_Income_squared` (correlación: 0.692)
   - `Geographic_Desirability_squared`

3. **Ratios Informativos**:
   - `Bedroom_Income_Ratio`
   - `Coast_to_City_Ratio`
   - `Desirability_per_Income`

### Selección Automática de Variables
- **Correlación** con variable objetivo
- **Importancia Random Forest**
- **Regularización Lasso**
- **Información Mutua**
- Ranking combinado con eliminación de redundancias

### Optimización de Modelos
- **RandomizedSearchCV** con 30-50 iteraciones
- **Validación cruzada** de 5 folds
- **Early stopping** para evitar overfitting
- **Análisis de convergencia**

## 🎯 Hallazgos Clave

### Variables Más Importantes
1. **`Median_Income_x_Geographic_Desirability`**: 0.787 correlación ⭐
2. **`Median_Income_squared`**: 0.692 correlación
3. **`Median_Income`**: 0.683 correlación
4. **`Bedroom_Income_Ratio`**: 0.652 correlación
5. **`Is_Max_Value`**: 0.574 correlación

### Insights Geográficos
- **Proximidad a la costa**: Factor determinante (+47% correlación)
- **Distancia a San Francisco**: Mayor impacto que otras ciudades principales
- **Geographic_Desirability**: Variable compuesta exitosa que combina ubicación
- **Patrón geoespacial**: Precios más altos en áreas costeras y cerca de Silicon Valley

### 🌐 Visualización Geoespacial
- **Mapas interactivos** con Folium para visualizar distribución de precios
- **Heatmaps geográficos** usando coordenadas de latitud y longitud  
- **Análisis de clusters** por zonas geográficas
- **Correlaciones espaciales** entre ubicación y características de vivienda

### Feature Engineering Impact
- **+15% mejora** en correlación con variable objetivo
- **De 19 → 27 variables** finales (eliminando redundancias)
- **8 interacciones** creadas exitosamente
- **Eliminación automática** de variables redundantes (correlación >0.95)

## 📈 Resultados

### Rendimiento de Modelos

| Modelo | R² Test | RMSE Test | MAE Test | Tiempo |
|--------|---------|-----------|----------|---------|
| **LightGBM** | **0.876** | **$41,250** | **$28,900** | 45s |
| XGBoost | 0.871 | $42,100 | $29,400 | 89s |
| Random Forest | 0.864 | $43,200 | $30,100 | 67s |

### Análisis por Rangos de Precio

| Rango | Casos | R² | MAE |
|-------|-------|----|----|
| Bajo (<$200k) | 45.2% | 0.823 | $22,100 |
| Medio ($200k-$350k) | 35.8% | 0.861 | $28,900 |
| Alto ($350k-$500k) | 15.4% | 0.789 | $39,200 |
| Premium (>$500k) | 3.6% | 0.654 | $67,800 |

## 💻 Uso

### Ejecutar el análisis completo

```bash
# Iniciar Jupyter Notebook
jupyter notebook Prediccion_precios_vivienda_CA.ipynb
```

### Usar modelos individuales

```python
# Feature Engineering
from feature_engineering import prepare_data_for_modeling
data_dict = prepare_data_for_modeling(data_processed)

# Entrenar modelos
from xgboost_model import run_xgboost_analysis
from lightgbm_model import run_lightgbm_analysis
from random_forest_model import run_random_forest_analysis

# Ejecutar análisis completo
xgb_results = run_xgboost_analysis(data_dict, optimize=True)
lgb_results = run_lightgbm_analysis(data_dict, optimize=True)
rf_results = run_random_forest_analysis(data_dict, optimize=True)

# Comparar resultados
print(f"XGBoost R²: {xgb_results['metrics']['test_r2']:.4f}")
print(f"LightGBM R²: {lgb_results['metrics']['test_r2']:.4f}")
print(f"Random Forest R²: {rf_results['metrics']['test_r2']:.4f}")
```

### Hacer predicciones

```python
# Cargar mejor modelo
best_model = lgb_results['model']

# Hacer predicciones
predictions = best_model.predict(X_new)
print(f"Precio estimado: ${predictions[0]:,.0f}")
```

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Si deseas contribuir:

1. **Fork** el repositorio
2. Crea una rama para tu feature:
   ```bash
   git checkout -b feature/nueva-mejora
   ```
3. **Commit** tus cambios:
   ```bash
   git commit -m 'Añadir nueva mejora'
   ```
4. **Push** a la rama:
   ```bash
   git push origin feature/nueva-mejora
   ```
5. Abre un **Pull Request**

### Áreas de mejora

- [ ] Implementar redes neuronales (Deep Learning)
- [ ] Añadir datos externos (económicos, demográficos)
- [ ] Crear API REST para predicciones
- [ ] Desarrollar dashboard interactivo
- [ ] Implementar modelos de series temporales

## 📝 Conclusiones del Proyecto

Este proyecto ofrece un **marco sólido y completo** para la predicción de precios de vivienda en California con los siguientes logros:

### 🎯 **Resultados Clave:**
- **Mejor modelo**: LightGBM con R² = 0.876 (87.6% de varianza explicada)
- **Precisión**: RMSE de $41,250 en predicciones de precios
- **Feature Engineering**: Mejora del 15% en correlación con variable objetivo
- **Automatización**: Pipeline completo de preprocesamiento a predicción

### 🌟 **Aportes Técnicos:**
- **Variable estrella**: `Median_Income_x_Geographic_Desirability` (correlación 0.787)
- **Optimización avanzada**: RandomizedSearchCV con early stopping
- **Visualización geoespacial**: Mapas interactivos con Folium
- **Análisis exhaustivo**: Comparación de 3 algoritmos de ML

### 🏠 **Insights de Negocio:**
- La **ubicación geográfica** es el factor más determinante en precios
- **Ingreso mediano** combinado con **deseabilidad geográfica** predice el 78.7% de la variación
- **Proximidad a la costa** incrementa valores inmobiliarios significativamente
- **San Francisco** tiene mayor impacto en precios que Los Ángeles

### 📊 **Aplicaciones Prácticas:**
- **Tasación automatizada** de propiedades
- **Estrategias de inversión** inmobiliaria
- **Análisis de mercado** por zonas geográficas
- **Predicción de tendencias** de valorización

A través de modelos avanzados y visualizaciones geoespaciales interactivas, este proyecto proporciona un **análisis profundo y accionable** de las variables que influyen en los precios del mercado inmobiliario californiano.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

⭐ Si este proyecto te fue útil, ¡dale una estrella en GitHub!
