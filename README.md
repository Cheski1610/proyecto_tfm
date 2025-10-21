## Proyecto Final de Máster

Repositorio del Trabajo de Fin de Máster (TFM) del programa **Data Science & Business Analytics**. El proyecto entrega un asistente web que automatiza el análisis de indicadores macroeconómicos costarricenses combinando series temporales, modelos de regresión y resumen inteligente de noticias.

![Portada](Portada-Agent.png)

- Acceso al agente desplegado en Azure: [Agente Económico](https://proyecto-tfmds.azurewebsites.net/)

---

### 1. Objetivo del proyecto

- Democratizar el análisis económico poniendo a disposición de usuarios no técnicos un asistente conversacional capaz de:
  - Descargar datos oficiales del Banco Central de Costa Rica (BCCR) mediante su servicio SOAP.
  - Generar proyecciones con Prophet en enfoques autorregresivos y de regresión múltiple.
  - Resumir noticias recientes para contextualizar los resultados del modelo utilizando LLMs alojados en Groq.

### 2. Arquitectura funcional

| Capa | Tecnologías | Descripción |
| --- | --- | --- |
| Interfaz | Streamlit | UI que guía la captura de parámetros (API keys, token BCCR, fechas, variables) y muestra reportes.|
| Orquestación | Clases `MultiAgent` y `MultiAgentM` | Enlazan agentes especializados para datos, proyecciones, gráficos y análisis de noticias. |
| Modelado | Prophet (Facebook) | Pronóstico de series temporales y regresión con múltiples regresores. |
| Enriquecimiento | Groq + `smolagents` | Resume noticias relevantes de DuckDuckGo usando modelos `deepseek-r1`. |
| Datos externos | SOAP BCCR, ScraperAPI | Descarga de indicadores económicos y resultados de búsqueda filtrados. |
| Despliegue | Azure App Service / Docker | La aplicación se expone como contenedor Streamlit en Azure. |

### 3. Flujo general del asistente

1. Usuario define indicador, horizonte y tipo de análisis (autorregresivo o multivariable).
2. `IndicatorData`/`IndicatorDataM` consulta el servicio SOAP del BCCR.
3. `PredictionAgent`/`PredictionAgentM` entrena Prophet y genera pronósticos según la frecuencia del indicador.
4. `ChartAgent` genera la visualización con Matplotlib.
5. `NewsAgent` busca noticias, las depura con BeautifulSoup y produce un resumen estructurado vía Groq.
6. Streamlit muestra reporte, gráfico, resumen y genera un HTML descargable.

### 4. Requerimientos principales

- Python 3.10+
- Dependencias listadas en `requirements.txt`
- Token vigente para el servicio web del BCCR (solicitar en su [portal oficial](https://www.bccr.fi.cr/indicadores-economicos/servicio-web))
- Cuenta Groq con API Key gratuita
- API Key de [ScraperAPI](https://www.scraperapi.com/) (para mejorar la estabilidad de las búsquedas DuckDuckGo)

### 5. Configuración local rápida

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Definir variables de entorno (ejemplo en Windows PowerShell):

```powershell
setx GROQ_API_KEY "tu_api_key_groq"
setx SCRAPE_API "tu_api_key_scraper"
```

> También se aceptan las variables en un archivo `.env` gracias a `python-dotenv`.

### 6. Ejecución local

```powershell
streamlit run App.py
```

La interfaz abrirá en `http://localhost:8501`. Ingresar token del BCCR, API key de Groq y seleccionar variable a analizar.

### 7. Uso de la aplicación

- **Autorregresivo**: Pronósticos con la misma variable como predictor principal.
- **Regresión múltiple**: Permite seleccionar variables adicionales (tipo de cambio, actividad económica, tasas) como regresores.
- **Reporte descargable**: Streamlit genera un HTML con texto del informe, gráfico y análisis de noticias.
- **Resumen de noticias**: El modelo `deepseek-r1` (vía Groq) resume hallazgos clave y discute la coherencia con la proyección.

### 8. Despliegue en Azure (resumen)

- Contenedor Docker basado en `Dockerfile` y orquestado con `docker_compose.yml`.
- Publicación en Azure App Service for Containers.
- Variables de entorno (`GROQ_API_KEY`, `SCRAPE_API`) configuradas en la instancia.

### 9. Estructura relevante del repositorio

```
App.py                  # Interfaz Streamlit y lógica de interacción
funciones.py            # Definición de agentes de datos, predicción, gráficos y noticias
Dockerfile              # Imagen para despliegue en Azure/App Service
docker_compose.yml      # Compose para entorno local orquestado
requirements.txt        # Dependencias del proyecto
Documento TFM/          # Material de soporte del proyecto académico
```

### 10. Ejecución en GitHub Codespaces

1. Desde la vista del repositorio en GitHub haz clic en `Code` → pestaña `Codespaces` → `Create codespace on main`.
2. El contenedor de desarrollo (`.devcontainer/devcontainer.json`) crea automáticamente un entorno virtual, actualiza `pip` e instala `requirements.txt` al primer arranque. Si necesitas reinstalar paquetes más adelante, activa el entorno (`source .venv/bin/activate` en Linux, `.\.venv\Scripts\activate` en Windows) y vuelve a ejecutar `pip install -r requirements.txt`.
3. Configura secretos para las credenciales necesarias:
	- En GitHub ve a `Settings` → `Codespaces` → `Secrets` → `New repository secret`.
	- Los secretos se inyectan como variables de entorno al iniciar; también puedes definir variables temporales con `export VARIABLE=valor` en el terminal.
4. Inicia la aplicación dentro del codespace:

	```bash
	streamlit run App.py
	```

5. Codespaces detectará el puerto `8501` (preconfigurado en `devcontainer.json`) y mostrará un aviso. Selecciona “Open in Browser” para acceder a la interfaz de Streamlit.