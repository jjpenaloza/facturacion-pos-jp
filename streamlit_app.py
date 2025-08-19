import streamlit as st
import joblib  # mejor que pickle para sklearn
from pathlib import Path
from pycaret.time_series import *
#import pycaret.containers.models.time_series
from pycaret.time_series import load_model, predict_model

import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import altair as alt
from datetime import date, timedelta
import numpy as np
from pycaret.time_series import predict_model # Importar predict_model específicamente
from sktime.datatypes import check_raise # Importar para diagnóstico de sktime
import prophet # Importación para modelos Prophet
# *** NUEVA IMPORTACIÓN ESPECÍFICA PARA RESOLVER ProphetPeriodPatched ***
import pycaret.containers.models.time_series 

# --- Constantes y Funciones de Carga de Datos ---
# Ajusta esta URL a tu data maestra. Esta data será el histórico para graficar.
RUTA_DATA_MAESTRA_GIT = "https://github.com/jjpenaloza/pos_prediction/blob/main/silver_serie_temporal_pos.csv?raw=true"

# --- RUTAS DE LOS MODELOS ---
# Define el directorio base donde se encuentran tus modelos
MODELOS_DIR = Path("modelos")

# Diccionario que mapea la entidad a la ruta de su modelo
MODEL_PATHS = {
    "datafast": MODELOS_DIR / "datafast_monto_facturado_model.pkl",
    "medianet": MODELOS_DIR / "medianet_monto_facturado_model.pkl",
    "banco del austro": MODELOS_DIR / "baustro_monto_facturado_model.pkl",
}

@st.cache_data(show_spinner=False)
def _get_data_plantilla_hardcoded() -> pd.DataFrame:
    """Retorna los datos de la plantilla codificados en el script."""
    return pd.read_csv("https://github.com/jjpenaloza/pos_prediction/blob/main/exog_template.csv?raw=true")

@st.cache_data(show_spinner=False)
def _leer_data_maestra_git() -> pd.DataFrame:
    """Carga la data maestra para la predicción desde un archivo CSV en GitHub."""
    try:
        df = pd.read_csv(RUTA_DATA_MAESTRA_GIT)
        # Asegúrate de que la columna 'fecha' sea tipo datetime y el índice
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.set_index('fecha')
        return df
    except Exception as e:
        st.error(f"Error al cargar la data maestra desde la URL: {e}")
        return pd.DataFrame()

def get_plantilla_de_datos_entidad(entidad: str) -> pd.DataFrame:
    """Filtra y devuelve todos los registros de una entidad como plantilla."""
    df = _get_data_plantilla_hardcoded()
    if df.empty:
        return pd.DataFrame()
    # Asegúrate de que la columna 'fecha' sea tipo datetime y el índice
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.set_index('fecha')
    return df.loc[df["entidad"].str.lower() == entidad]

@st.cache_resource
def _load_model(entidad: str):
    """Carga el modelo pre-entrenado para la entidad seleccionada."""
    model_path = MODEL_PATHS.get(entidad)
    if not model_path or not model_path.exists():
        st.error(f"Error: No se encontró el modelo para '{entidad}'. Ruta esperada: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo para '{entidad}': {e}")
        return None

# --- Inicializar el estado de la sesión ---
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'entidad_graficada' not in st.session_state:
    st.session_state.entidad_graficada = "..."
if 'entidad_seleccionada_anterior' not in st.session_state:
    st.session_state.entidad_seleccionada_anterior = None
if 'horizonte_anterior' not in st.session_state:
    st.session_state.horizonte_anterior = None

# --- Funciones de control ---
def get_controls():
    st.subheader("Controles y Filtros")
    
    entidad_seleccionada = st.selectbox(
        "Selecciona la entidad:",
        options=["Datafast", "Medianet", "Banco del Austro"]
    )
    
    horizonte = st.select_slider(
        "Selecciona el horizonte (en meses):",
        options=list(range(1, 13))
    )
    
    return entidad_seleccionada.lower(), horizonte

import streamlit as st

def show_data_inputs(entidad: str, horizonte: int):
    # Mensaje general
    st.write(f"Valores de las variables exógenas para {entidad.capitalize()} para los próximos {horizonte} meses:")

    # --- init seguro de session_state.input_data ---
    st.session_state.setdefault("input_data", {})
    for i in range(horizonte):
        st.session_state.input_data.setdefault(i, {})
        st.session_state.input_data[i].setdefault("fecha", f"Fecha {i+1}")
        st.session_state.input_data[i].setdefault("nro_puntos_ventas", "")
        st.session_state.input_data[i].setdefault("oferta_monetaria_m1", "")

    # --- Encabezados (una sola vez) ---
    h1, h2, h3 = st.columns([3, 4, 4])
    with h1: st.markdown("**Fecha**")
    with h2: st.markdown("**Puntos de Venta**")
    with h3: st.markdown("**Oferta Monetaria M1**")
    st.divider()

    # --- Form para evitar re-runs por cada input ---
    with st.form(f"form_parametros_{entidad}"):
        for i in range(horizonte):
            month_data = st.session_state.input_data.get(i, {})
            fecha = month_data.get("fecha", f"Fecha {i+1}")

            col1, col2, col3 = st.columns([3, 4, 4])

            # Columna 1: fecha como "subtítulo/label" (si la quisieras editable, cámbiala por st.text_input)
            with col1:
                st.caption(fecha)

            # Columna 2: número de puntos de venta
            with col2:
                val_pv = month_data.get("nro_puntos_ventas", "")
                try:
                    def_pv = int(val_pv) if val_pv not in ("", None) and str(val_pv).strip() != "" else 0
                except Exception:
                    def_pv = 0

                st.number_input(
                    label="Puntos de Venta",
                    key=f"puntos_ventas_{i}_{entidad}",
                    value=def_pv,
                    step=1,
                    format="%d",
                    label_visibility="collapsed",
                )

            # Columna 3: oferta monetaria M1
            with col3:
                val_m1 = month_data.get("oferta_monetaria_m1", "")
                try:
                    def_m1 = int(val_m1) if val_m1 not in ("", None) and str(val_m1).strip() != "" else 0
                except Exception:
                    def_m1 = 0

                st.number_input(
                    label="Oferta Monetaria M1",
                    key=f"oferta_m1_{i}_{entidad}",
                    value=def_m1,
                    step=1,
                    format="%d",
                    label_visibility="collapsed",
                )

            # Separador fino entre filas
            st.markdown("<div style='height:0.25rem;'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Guardar cambios", use_container_width=True)

    # --- Al enviar, volcamos al session_state UNA sola vez ---
    if submitted:
        for i in range(horizonte):
            pv = st.session_state.get(f"puntos_ventas_{i}_{entidad}", 0)
            m1 = st.session_state.get(f"oferta_m1_{i}_{entidad}", 0)

            # Si tu lógica downstream espera strings, los dejamos como str
            st.session_state.input_data[i]["nro_puntos_ventas"] = str(pv) if pv is not None else ""
            st.session_state.input_data[i]["oferta_monetaria_m1"] = str(m1) if m1 is not None else ""

        st.success("Parámetros guardados.")

def create_chart_data(entidad: str, horizonte: int):
    # 1. Preparar los datos exógenos futuros para la predicción
    data_list = []
    for i in range(horizonte):
        row = st.session_state.input_data.get(i, {})
        row_dict = {
            'fecha': pd.to_datetime(row.get('fecha')),
            'nro_puntos_ventas': pd.to_numeric(row.get('nro_puntos_ventas'), errors='coerce'),
            'oferta_monetaria_m1': pd.to_numeric(row.get('oferta_monetaria_m1'), errors='coerce'),
            'nro_dias_feriados': pd.to_numeric(row.get('nro_dias_feriados'), errors='coerce'),
            'shock_pandemia': pd.to_numeric(row.get('shock_pandemia'), errors='coerce'),
            'fin_de_anio': pd.to_numeric(row.get('fin_de_anio'), errors='coerce'),
            'mes_especial': pd.to_numeric(row.get('mes_especial'), errors='coerce'),
            'entidad': entidad.capitalize()
        }
        data_list.append(row_dict)

    future_exog_df = pd.DataFrame(data_list)
    future_exog_df = future_exog_df.set_index('fecha') # Establecer 'fecha' como índice

    # 2. Cargar datos históricos y preparar para el cálculo de lags
    historical_data = _leer_data_maestra_git()
    if historical_data.empty:
        st.warning("No se pudieron cargar los datos históricos. No se pueden calcular lags.")
        return pd.DataFrame()

    historical_data_entidad = historical_data[historical_data['entidad'].str.lower() == entidad].copy()
    
    # Obtener los últimos valores históricos de las variables que tienen lags
    cols_for_lag_calculation = ['nro_puntos_ventas', 'oferta_monetaria_m1']
    last_historical_values = historical_data_entidad[cols_for_lag_calculation].dropna().iloc[[-1]].copy()

    # Concatenar el último registro histórico con los datos exógenos futuros
    # Esto es crucial para que `shift(1)` calcule correctamente el primer lag futuro
    # Asegurar que el índice es de tipo datetime para la concatenación
    last_historical_values.index = pd.to_datetime(last_historical_values.index)
    
    # Crear un DataFrame temporal para calcular los lags, incluyendo el último punto histórico
    temp_combined_exog = pd.concat([last_historical_values, future_exog_df[cols_for_lag_calculation]])

    # Calcular los lags
    temp_combined_exog['lag1_nro_puntos_ventas'] = temp_combined_exog['nro_puntos_ventas'].shift(1)
    temp_combined_exog['lag1_oferta_monetaria_m1'] = temp_combined_exog['oferta_monetaria_m1'].shift(1)

    # Añadir los lags calculados al DataFrame de exógenas futuras (future_exog_df)
    # Es importante tomar solo los lags correspondientes al periodo de predicción
    future_exog_df['lag1_nro_puntos_ventas'] = temp_combined_exog['lag1_nro_puntos_ventas'].loc[future_exog_df.index]
    future_exog_df['lag1_oferta_monetaria_m1'] = temp_combined_exog['lag1_oferta_monetaria_m1'].loc[future_exog_df.index]
    
    # Identificar todas las columnas exógenas que el modelo espera, incluyendo las originales y los lags.
    # Esta lista DEBE COINCIDIR con las características que tu modelo usó durante el entrenamiento.
    expected_exog_cols = [
        'nro_puntos_ventas', 'oferta_monetaria_m1', 'nro_dias_feriados', 
        'shock_pandemia', 'fin_de_anio', 'mes_especial',
        'lag1_nro_puntos_ventas', 'lag1_oferta_monetaria_m1'
    ]

    # Seleccionar solo las columnas esperadas para X y asegurar el orden
    X_for_prediction = future_exog_df[expected_exog_cols]
    
    # Rellenar cualquier NaN remanente en X_for_prediction con 0.
    X_for_prediction = X_for_prediction.fillna(0)

    # Convertir el índice de X_for_prediction a PeriodIndex
    try:
        if not isinstance(X_for_prediction.index, pd.PeriodIndex):
            X_for_prediction.index = X_for_prediction.index.to_period('M') # O 'MS' si es inicio de mes
        #st.write(f"Índice de X_for_prediction después de conversión: {X_for_prediction.index.dtype}")
    except Exception as e:
        st.error(f"Error al convertir el índice de X_for_prediction a PeriodIndex: {e}")
        return pd.DataFrame()


    # Diagnóstico sktime: Comprobar el formato de X_for_prediction
    #st.write("DataFrame 'X' enviado al modelo para predicción:")
    #st.dataframe(X_for_prediction)
    #st.write(f"Información del índice de X_for_prediction: {X_for_prediction.index.dtype}")
    #st.write(f"Primeros 5 registros de X_for_prediction:\n{X_for_prediction.head()}")
    #st.write(f"Conteo de valores NaN por columna en X_for_prediction:\n{X_for_prediction.isnull().sum()}")

    try:
        check_raise(X_for_prediction, "pd.DataFrame", scitype="Series")
        #st.success("Formato de 'X_for_prediction' verificado por sktime como compatible 'pd.DataFrame' (scitype Series).")
    except Exception as e:
        st.error(f"Diagnóstico sktime para 'X_for_prediction' falló. Por favor, revisa los detalles de 'X' arriba. Error: {e}")
        return pd.DataFrame()

    # 3. Cargar el modelo
    model = _load_model(entidad)
    if model is None:
        return pd.DataFrame()

    # 4. Realizar la predicción
    try:
        # Definir el horizonte de predicción (forecast horizon)
        fh = np.arange(1, horizonte + 1)
        
        predictions = predict_model(model, fh=fh, X=X_for_prediction)
        
        # Convertir el índice de PeriodIndex a DatetimeIndex para la gráfica y concatenación
        predictions.index = predictions.index.to_timestamp()
        
        predictions = predictions.reset_index().rename(columns={'index': 'fecha'})
        predictions['tipo'] = 'Predicción'
        predictions = predictions.rename(columns={'y_pred': 'monto_facturado'})
        predictions = predictions[['fecha', 'monto_facturado', 'tipo']]

        st.write("Predicciones generadas:")
        st.dataframe(predictions)

        # 5. Cargar datos históricos y combinarlos para graficar
        full_historical_data = _leer_data_maestra_git()
        if full_historical_data.empty:
            st.warning("No se pudieron cargar los datos históricos para la gráfica.")
            return predictions

        historical_data_for_chart = full_historical_data[full_historical_data['entidad'].str.lower() == entidad].copy()
        historical_data_for_chart['tipo'] = 'Histórico'
        historical_data_for_chart = historical_data_for_chart.reset_index()
        historical_data_for_chart = historical_data_for_chart[['fecha', 'monto_facturado', 'tipo']]

        # Combinar datos históricos y predicciones
        combined_data = pd.concat([historical_data_for_chart, predictions])
        combined_data['fecha'] = pd.to_datetime(combined_data['fecha'])
        combined_data = combined_data.sort_values('fecha')

        return combined_data

    except Exception as e:
        st.error(f"Error al generar la predicción: {e}")
        return pd.DataFrame()


def create_chart(entidad: str):
    st.subheader(f"Gráfica de Monto Facturado para {entidad.capitalize()}")
    if st.session_state.chart_data is not None and not st.session_state.chart_data.empty:
        chart = alt.Chart(st.session_state.chart_data).mark_line(point=True).encode(
            x=alt.X('fecha', title='Fecha', axis=alt.Axis(format="%Y-%m")),
            y=alt.Y('monto_facturado', title='Monto Facturado'),
            color=alt.Color('tipo', title='Tipo de Dato', legend=alt.Legend(title="Leyenda")),
            tooltip=['fecha', 'monto_facturado', 'tipo']
        ).properties(
            title=f'Monto Facturado Histórico y Predicción para {entidad.capitalize()}'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Presiona 'Predecir' para generar la gráfica de predicción.")

# --- Layout Principal de la Aplicación ---
def main():
    st.title("Panel de Predicción de Entidades")
    # col1, col2 = st.columns([12, 24])
    
    with st.container():
        entidad_display, horizonte = get_controls()
        
        if (entidad_display != st.session_state.entidad_seleccionada_anterior or
            horizonte != st.session_state.horizonte_anterior):
            
            st.session_state.input_data = {}
            plantilla_df = get_plantilla_de_datos_entidad(entidad_display)
            
            last_historical_date = None
            historical_data_check = _leer_data_maestra_git()
            if not historical_data_check.empty:
                filtered_hist = historical_data_check[historical_data_check['entidad'].str.lower() == entidad_display]
                if not filtered_hist.empty:
                    last_historical_date = filtered_hist.index.max()

            for i in range(horizonte):
                if last_historical_date:
                    current_input_date = last_historical_date + pd.DateOffset(months=i+1)
                else:
                    current_input_date = date.today() + timedelta(days=30 * i) 

                plantilla_data_row = {}
                if not plantilla_df.empty:
                    if i < len(plantilla_df):
                        plantilla_data_row = plantilla_df.iloc[i].to_dict()
                    else:
                        plantilla_data_row = plantilla_df.iloc[-1].to_dict()
                
                st.session_state.input_data[i] = {
                    'fecha': str(current_input_date.date()),
                    'nro_puntos_ventas': plantilla_data_row.get('nro_puntos_ventas', ''),
                    'oferta_monetaria_m1': plantilla_data_row.get('oferta_monetaria_m1', ''),
                    'nro_dias_feriados': plantilla_data_row.get('nro_dias_feriados', 0),
                    'shock_pandemia': plantilla_data_row.get('shock_pandemia', 0),
                    'fin_de_anio': plantilla_data_row.get('fin_de_anio', 0),
                    'mes_especial': plantilla_data_row.get('mes_especial', 0)
                }
        
        st.session_state.entidad_seleccionada_anterior = entidad_display
        st.session_state.horizonte_anterior = horizonte
        show_data_inputs(entidad_display, horizonte)

        st.write("")
        if st.button("Predecir"):
            st.success("Generando predicción...")
            st.session_state.chart_data = create_chart_data(entidad_display, horizonte)
            st.session_state.entidad_graficada = entidad_display
    
    st.divider()            

    with st.container():
        create_chart(st.session_state.entidad_graficada)

if __name__ == "__main__":
    main()
