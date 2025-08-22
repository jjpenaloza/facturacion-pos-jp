# 📊 Predicción de Facturación POS en Ecuador – Demo Interactivo

Este repositorio contiene el **prototipo interactivo en Streamlit** desarrollado en el marco del proyecto de titulación *“Diseño de un modelo predictivo del monto de facturación de Puntos de Venta Electrónicos (POS) en Ecuador”*.  

El sistema permite **cargar y ejecutar modelos predictivos entrenados con PyCaret** para tres entidades procesadoras de POS: **Datafast, Banco del Austro y Medianet**, generando pronósticos de facturación mensual con horizonte de hasta 12 meses.

---

## 🚀 Características principales
- Modelos de series temporales entrenados y ajustados con AutoML (PyCaret TS).  
- Validación mediante *time-series cross-validation* (fold=3).  
- Métricas reportadas: **MAPE, RMSSE, RMSE, MAE, SMAPE, R²**.  
- Cumplimiento de criterios de aceptación: **MAPE ≤ 10% y RMSSE ≤ 1** (tras el ajuste de hiperparámetros).  
- Demo interactivo en **Streamlit** para visualizar resultados y escenarios.  
- Versionamiento del código y modelos mediante **GitHub**.  
- Reproducibilidad garantizada con `environment.yaml`.

---

## 📂 Estructura del repositorio
```
.
├─ app/streamlit_app.py        # Script principal de la app Streamlit
├─ modelos/                    # Modelos entrenados (.pkl)
├─ data/                       # Datos históricos y plantillas de exógenas
├─ environment.yaml            # Definición del ambiente reproducible
└─ README.md                   # Este documento
```

---

## ⚙️ Instalación y ejecución local

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/proyecto-pos-ecuador.git
cd proyecto-pos-ecuador
```

### 2. Crear el entorno
```bash
conda env create -f environment.yaml
conda activate proyectomdb311
```

### 3. Ejecutar la aplicación
```bash
streamlit run app/streamlit_app.py
```

La aplicación estará disponible en [http://localhost:8501](http://localhost:8501).

---

## 📊 Uso de la aplicación
1. Seleccionar la entidad operadora (**Datafast, Banco del Austro o Medianet**).  
2. Definir el **horizonte de predicción** (hasta 12 meses).  
3. Cargar datos exógenos futuros (ejemplo disponible en `data/exog_sample_future.csv`).  
4. Visualizar predicciones gráficas y descargar resultados en CSV.  

---

## 🔄 Reproducibilidad
- **Versionamiento en GitHub**: histórico completo de cambios del proyecto.  
- **Ambiente controlado**: instalación desde `environment.yaml` asegura versiones exactas de librerías.  
- **Modelos persistidos**: entrenados y guardados con `exp.save_model()`, garantizando consistencia con el demo.  
- **Semilla fija (`session_id=123`)**: garantiza resultados replicables en cada corrida.  

---

## 📑 Documentación técnica
- **Model Cards** por entidad (ubicadas en `/docs`): describen datos usados, variables exógenas, métricas de validación y supuestos.  
- **Guía de despliegue**: instrucciones para correr la app localmente o en contenedores Docker.  
- **Runbook**: incluye pruebas de humo y acciones en caso de fallos (ej. ausencia de exógenas, drift de datos).

---

## 🧪 Validación
Los modelos fueron evaluados en tres particiones temporales (*fold=3*) con horizonte de 12 meses, alcanzando métricas finales:  

- **Datafast**: MAPE 7.69%, RMSSE 0.41 ✅  
- **Banco del Austro**: MAPE 9.85%, RMSSE 0.89 ✅  
- **Medianet**: MAPE 9.92%, RMSSE 0.61 ✅  

---

## 📌 Licencia
Uso académico y de investigación. No debe emplearse directamente para decisiones financieras sin validación adicional.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
