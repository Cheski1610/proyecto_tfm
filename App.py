# Importar Librerías
import streamlit as st
from PIL import Image
import io
import base64
import markdown2
from datetime import datetime
from funciones import MultiAgent, MultiAgentM
import warnings


warnings.filterwarnings("ignore")

# Importar imagen
imagen_1 = Image.open("Portada-Agent.png")

# Configuración de la página

def run():

    # Configurar sidebar
    st.sidebar.markdown("<h7 style='color:#097da6;'>Ingresa el API Key de GROQ para generar el análisis:</h7>", unsafe_allow_html=True)
    api_key = st.sidebar.text_input("API Key", value="", label_visibility="collapsed")
    st.sidebar.markdown("<h7 style='color:#097da6; text-align: justify;'>Si no sabes como generar el API Key, en este [Tutorial](https://www.youtube.com/watch?v=YPghgcC4p-E) se explica muy fácilmente. Este servicio de GROQ es totalmente gratuito.</h7>", unsafe_allow_html=True)

    # Cambiar el color de fondo de la página
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: 
        radial-gradient(circle at 20% 20%, rgba(255, 255, 255, 0.5), transparent),
        radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.3), transparent),
        linear-gradient(to bottom, #2596BE, #f5e6d3);
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True) 

    st.image(imagen_1, use_container_width='always') #imagen de portada

    st.markdown("<h4 style='color:#097da6; text-align: justify;'>Soy un Agente creado para ayudarte a analizar variables económicas.</h4>", unsafe_allow_html=True)

    st.markdown("<h5 style='color:#097da6;'>¿Cuál variable te interesa analizar? </h5>", unsafe_allow_html=True)

    # Crear un menú de selección

    variable = st.selectbox("", ["TC de Compra", "TC de Venta", "Inflación", "PIB", "Tasa Desempleo", "IMAE", "TBP", "TRI 6M"], 
                            index=None, placeholder="Selecciona una opción", label_visibility="collapsed")
    
    dicc_indicador = {"TC de Compra": ["317","D"], "TC de Venta": ["318","D"], "Inflación": ["89635","M"], "PIB": ["90633","Q"], 
                      "Tasa Desempleo": ["23630","M"], "IMAE": ["87764","M"], "TBP": ["423","D"], "TRI 6M": ["41206","D"]}

    if variable != None:
        freq = dicc_indicador[variable][1]
        if freq == "D":
            periodicidad = "Diaria"
            value = 180
        elif freq == "M":
            periodicidad = "Mensual"
            value = 6
        elif freq == "Q":
            periodicidad = "Trimestral"
            value = 2
        else:
            periodicidad = "Anual"
            value = 1

        st.markdown(f"<h7 style='color:#097da6;'>**Periodicidad**: {periodicidad}</h7>", unsafe_allow_html=True)

        st.markdown("<h5 style='color:#097da6;'>Selecciona el tipo de análisis:</h5>", unsafe_allow_html=True)
        
        tipo_analisis = st.selectbox("", ["Autorregresivo", "Regresión Múltiple"], 
                                    index=None, placeholder="Selecciona una opción", label_visibility="collapsed")
        
        # TODO: Análisis Autoregresivo
        if tipo_analisis == "Autorregresivo":
            st.markdown("<h7 style='color:#097da6;'>Completa lo siguiente para obtener los datos del [BCCR Servicio Web](https://www.bccr.fi.cr/indicadores-economicos/servicio-web)</h7>", unsafe_allow_html=True)

            Name = st.text_input("**Nombre:**", "")
            Email = st.text_input("**Correo Electrónico:**", "")
            Token = st.text_input("**Token BCCR:**", "")
            Indicador = dicc_indicador[variable][0]

            col1, col2 = st.columns(2, gap="small")
            with col1:
                fecha_inicio = st.date_input("**Fecha Inicial:**", datetime(2020, 1, 1))
            with col2:
                fecha_fin = st.date_input("**Fecha Final:**", datetime.today().date())
            
            fecha_inicio = fecha_inicio.strftime("%Y/%m/%d")
            fecha_fin = fecha_fin.strftime("%Y/%m/%d")

            periodos = st.number_input("**Número de periodos a predecir:**", min_value=1, max_value=180, value=value)

            if st.button("Generar Reporte", type="primary"):
                # Crear el multiagente
                agent = MultiAgent(variable, Indicador, fecha_inicio, fecha_fin, Name, Email, Token, periodos, freq, api_key)
                report, fig, resume = agent.run()
                
                # Mostrar el reporte
                st.markdown("<h5 style='color:#097da6;'>REPORTE</h5>", unsafe_allow_html=True)
                st.markdown(report, unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown(resume, unsafe_allow_html=True)

                # Convertir la imagen a base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Crear el contenido HTML
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Reporte - {variable}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3, h4, h5 {{ color: #097da6; }}
                        .content {{ margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>Reporte de Análisis: {variable}</h1>
                    <div class="content">
                        {markdown2.markdown(report)}
                    </div>
                    <div class="content">
                        <img src="data:image/png;base64,{img_str}" alt="Gráfico de proyección" style="max-width: 100%;">
                    </div>
                    <div class="content">
                        {markdown2.markdown(resume)}
                    </div>
                </body>
                </html>
                """

                # Crear botón de descarga
                st.download_button(
                    label="Descargar Reporte",
                    data=html_content,
                    file_name=f"Reporte_{variable}.html",
                    mime="text/html",
                )

        # TODO: Análisis Multivariable
        elif tipo_analisis == "Regresión Múltiple":
            st.markdown("<h7 style='color:#097da6;'>Completa lo siguiente para obtener los datos del [BCCR Servicio Web](https://www.bccr.fi.cr/indicadores-economicos/servicio-web)</h7>", unsafe_allow_html=True)

            Name = st.text_input("**Nombre:**", "")
            Email = st.text_input("**Correo Electrónico:**", "")
            Token = st.text_input("**Token BCCR:**", "")
            Indicador = dicc_indicador[variable][0]

            col1, col2 = st.columns(2, gap="small")
            with col1:
                fecha_inicio = st.date_input("**Fecha Inicial:**", datetime(2020, 1, 1))
            with col2:
                fecha_fin = st.date_input("**Fecha Final:**", datetime.today().date())
            
            fecha_inicio = fecha_inicio.strftime("%Y/%m/%d")
            fecha_fin = fecha_fin.strftime("%Y/%m/%d")

            regresores = st.multiselect("**Selecciona las variables predictoras:**", 
                                        [var for var in ["TC de Compra", "TC de Venta", "Inflación", "PIB", "Tasa Desempleo", "IMAE", "TBP", "TRI 6M"] if var != variable])
            
            dicc_indicador = {key: value for key, value in dicc_indicador.items() if key in regresores or key == variable}
            dicc_indicador = {variable: dicc_indicador[variable]} | {key: dicc_indicador[key] for key in regresores}

            periodos = st.number_input("**Número de periodos a predecir:**", min_value=1, max_value=180, value=value)

            if st.button("Generar Reporte", type="primary"):
                # Crear el multiagente
                agent = MultiAgentM(dicc_indicador, fecha_inicio, fecha_fin, Name, Email, Token, periodos, freq, api_key)
                # report, fig, resume = agent.run()
                report, fig, resume = agent.run()
                
                # Mostrar el reporte
                st.markdown("<h5 style='color:#097da6;'>REPORTE</h5>", unsafe_allow_html=True)
                st.markdown(report, unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown(resume, unsafe_allow_html=True)

                # Convertir la imagen a base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Crear el contenido HTML
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Reporte - {variable}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1, h2, h3, h4, h5 {{ color: #097da6; }}
                        .content {{ margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>Reporte de Análisis: {variable}</h1>
                    <div class="content">
                        {markdown2.markdown(report)}
                    </div>
                    <div class="content">
                        <img src="data:image/png;base64,{img_str}" alt="Gráfico de proyección" style="max-width: 100%;">
                    </div>
                    <div class="content">
                        {markdown2.markdown(resume)}
                    </div>
                </body>
                </html>
                """

                # Crear botón de descarga
                st.download_button(
                    label="Descargar Reporte",
                    data=html_content,
                    file_name=f"Reporte_{variable}.html",
                    mime="text/html",
                )

if __name__ == "__main__":
    run()