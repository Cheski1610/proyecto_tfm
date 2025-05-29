import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xml.etree.ElementTree as ET
import requests
import instructor
from smolagents import CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool
from prophet import Prophet
from groq import Groq
from pydantic import BaseModel

# Agente autorregresivo
##----------------------------------------------------##

# Definir el agente para obtener los datos
class IndicatorData(CodeAgent):
    def __init__(self, variable, indicator, start_date, end_date, name, email, token):
        self.variable = variable
        self.indicator = indicator
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.email = email
        self.token = token
    
    def run(self):
        # Definir la URL del servicio SOAP
        url = "https://gee.bccr.fi.cr/Indicadores/Suscripciones/WS/wsindicadoreseconomicos.asmx"
        # Definir los parámetros SOAP
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
          <soap:Body>
            <ObtenerIndicadoresEconomicos xmlns="http://ws.sdde.bccr.fi.cr">
              <Indicador>{self.indicator}</Indicador>
              <FechaInicio>{self.start_date}</FechaInicio>
              <FechaFinal>{self.end_date}</FechaFinal>
              <Nombre>{self.name}</Nombre>
              <SubNiveles>N</SubNiveles>
              <CorreoElectronico>{self.email}</CorreoElectronico>
              <Token>{self.token}</Token>
            </ObtenerIndicadoresEconomicos>
          </soap:Body>
        </soap:Envelope>"""
        # Headers necesarios para la solicitud SOAP
        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': 'http://ws.sdde.bccr.fi.cr/ObtenerIndicadoresEconomicos'
        }
        # Realizar la solicitud POST
        response = requests.post(url, data=soap_request, headers=headers)
        # Parsear el XML de la respuesta
        root = ET.fromstring(response.content)
        # Extraer los datos de interés
        data = []
        for item in root.findall(".//INGC011_CAT_INDICADORECONOMIC"):
            cod_ind = item.find("COD_INDICADORINTERNO").text
            fecha = item.find("DES_FECHA").text.replace("T00:00:00-06:00", "")  # Limpiar la fecha
            valor = float(item.find("NUM_VALOR").text)  # Convertir a flotante
        # Agregar a la lista de datos
            data.append({
                "CÓDIGO_IND": cod_ind,
                "FECHA": fecha,
                "VALOR": valor,
                "INDICADOR": self.variable
            })
        # Crear el DataFrame
        df_Ind = pd.DataFrame(data)
        # Imprimir detalles
        report = f"## Datos del indicador {df_Ind['INDICADOR'][0]}"
        report += f"\n\n**{self.variable} al Cierre:** {df_Ind['VALOR'].iloc[-1]}"
        report += "\n\n**Período seleccionado:**"
        report += f"\n\n**Fecha inicial:** {self.start_date}"
        report += f"\n\n**Fecha final:** {self.end_date}\n\n"
        report += f"\n\n**Cantidad de registros base:** {len(df_Ind)}"
        report += f"\n\n**Datos Importantes de la Proyección:**"
        return report, df_Ind
    
# Definir el agente para graficar los datos
class ChartAgent(CodeAgent):
    def __init__(self, df_Ind, forecast):
        self.df = df_Ind
        self.forecast = forecast
    
    def run(self):
        # Convertir la columna FECHA a datetime
        self.df["FECHA"] = pd.to_datetime(self.df["FECHA"])
        # Crear la gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.df["FECHA"], self.df["VALOR"], color="green", label=f"{self.df['INDICADOR'][0]}")
        ax.plot(self.forecast["ds"], self.forecast["yhat"], color="red", linestyle="--", label="Predicción")
        ax.set_title(f"Indicador {self.df['INDICADOR'][0]}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=45)
        plt.legend()
        return fig

# Definir el agente para proyectar los datos
class PredictionAgent(CodeAgent):
    def __init__(self, df_Ind, Periodos, freq):
        self.df = df_Ind
        self.Periodos = Periodos
        self.freq = freq
    
    def run(self):
        # Crear el modelo Prophet
        model = Prophet()
        # Ajustar el modelo
        model.fit(self.df.rename(columns={"FECHA": "ds", "VALOR": "y"}))
        # Crear un DataFrame con las fechas a predecir
        future = model.make_future_dataframe(periods=self.Periodos, freq=self.freq)
        # Realizar la proyección
        forecast = model.predict(future)
        return forecast[['ds','yhat']]

# Definir el agente para obtener noticias
class NewsAgent(CodeAgent):
    def __init__(self, variable, end_date, forecast, api_key):
        self.variable = variable
        self.end_date = end_date
        self.forecast = forecast
        self.client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON) # Patch Groq() con instructor

    def run(self):
        # Definir herramienta de búsqueda
        search_tool = DuckDuckGoSearchTool()
        # Definir el tema de la búsqueda
        query = f"Noticias de Costa Rica de {self.variable} cercanas al {self.end_date}"
        # Realizar la búsqueda
        results = search_tool(query)
        # Definir salida estructurada
        class NewsResume(BaseModel):
            cantidad_noticias: int
            cadena_pensamiento_1: str
            cadena_pensamiento_2: str
            cadena_pensamiento_3: str
            cadena_pensamiento_4: str
            cadena_pensamiento_5: str
            cadena_pensamiento_6: str
            cadena_pensamiento_7: str
            cadena_pensamiento_8: str
            cadena_pensamiento_9: str
            cadena_pensamiento_10: str
            Analisis_noticias: str
        # Definir system prompt
        system_prompt = """
        Eres un asistente virtual especializado en el análisis de variables económicas. A partir de un conjunto de noticias que se te proporcionan, tu tarea es:

        1. Identificar y analizar de forma individual cómo cada noticia se relaciona con una variable económica específica.
        2. Posteriormente, integrar la información de las distintas noticias para detectar patrones, validar supuestos y generar una base sólida para el análisis.
        3. Evaluar una proyección que se te indica respecto a la variable económica, determinando en qué medida es consistente o factible en función de la evidencia encontrada en las noticias.
        4. Elaborar una conclusión argumentada sobre lo que puede esperarse en los próximos meses para la variable económica analizada, basándote exclusivamente en la información contenida en las noticias.

        Aspectos estructurales a considerar:

        * Toda tu Cadena de Pensamiento debe estar en español, al igual que tu respuesta final.
        * El análisis debe ser riguroso, ordenado y con enfoque crítico.
        * Justifica claramente tus evaluaciones y conclusiones, especificando los elementos que respaldan tu juicio sobre la proyección dada.
        """
        # Crear el resumen de las noticias con GROQ
        prompt = f"""La variable económica de interés es: {self.variable}. 
        Genera un resumen de las siguientes noticias que han sido recopiladas:\n\n{results}.\n\n Considera lo indicado 
        en estas noticias para evaluar la proyección generada a la fecha más lejana, 
        la proyección es de {self.forecast['yhat'].iloc[-1].round(2)}."""
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            response_model=NewsResume
        )
        # Definir estructura de parte final del reporte
        news = f"\n\n## Análisis de noticias actuales:\n\n"
        news += f"**Cantidad de Noticias Analizadas:** {chat_completion.cantidad_noticias}\n\n"
        news += f"**GROQ (DeepSeek-R1):**\n\n"
        news += f"**Resumen de la Cadena de Pensamiento**\n\n- {chat_completion.cadena_pensamiento_1}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_2}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_3}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_4}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_5}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_6}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_7}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_8}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_9}\n\n"
        news += f"- {chat_completion.cadena_pensamiento_10}\n\n"
        news += f"**Análisis de las Noticias**\n\n{chat_completion.Analisis_noticias}\n\n"
        return news

# Definir el multiagente
class MultiAgent(ToolCallingAgent):
    def __init__(self, variable, indicator, start_date, end_date, name, email, token, Periodos, freq, api_key):
        self.variable = variable
        self.indicator = indicator
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.email = email
        self.token = token
        self.Periodos = Periodos
        self.freq = freq
        self.api_key = api_key

    def run(self):
        # Crear el agente para obtener los datos
        agent1 = IndicatorData(self.variable, self.indicator, self.start_date, self.end_date, self.name, self.email, self.token)
        report, df_Ind = agent1.run()
        # Crear el agente para predecir los datos
        agent3 = PredictionAgent(df_Ind, self.Periodos, self.freq)
        forecast = agent3.run()
        # Crear el agente para graficar los datos
        agent2 = ChartAgent(df_Ind, forecast)
        fig = agent2.run()
        # crear el agente para obtener noticias
        agent4 = NewsAgent(self.variable, self.end_date, forecast, self.api_key)
        resume = agent4.run()
        # Crear el reporte final
        report += f"\n\n**Fecha final de proyección:** {forecast['ds'].iloc[-1].strftime('%Y/%m/%d')}"
        report += f"\n\n**Predicción del indicador:** {forecast['yhat'].iloc[-1].round(2)}"
        report += "\n\n## Gráfica del indicador"
        return report, fig, resume
    
##----------------------------------------------------##

# Agente Regresión Múltiple
##----------------------------------------------------##

# Definir el agente para obtener los datos
class IndicatorDataM(CodeAgent):
    def __init__(self, dicc, start_date, end_date, name, email, token):
        self.dicc = dicc
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.email = email
        self.token = token

    def run(self):
        # Definir la URL del servicio SOAP
        url = "https://gee.bccr.fi.cr/Indicadores/Suscripciones/WS/wsindicadoreseconomicos.asmx"
        # Crear un DataFrame vacío
        df_Ind = pd.DataFrame()
        # Iterar sobre los indicadores
        for variable, indicador in self.dicc.items():
            # Definir los parámetros SOAP
            soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
            <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <ObtenerIndicadoresEconomicos xmlns="http://ws.sdde.bccr.fi.cr">
                <Indicador>{indicador[0]}</Indicador>
                <FechaInicio>{self.start_date}</FechaInicio>
                <FechaFinal>{self.end_date}</FechaFinal>
                <Nombre>{self.name}</Nombre>
                <SubNiveles>N</SubNiveles>
                <CorreoElectronico>{self.email}</CorreoElectronico>
                <Token>{self.token}</Token>
                </ObtenerIndicadoresEconomicos>
            </soap:Body>
            </soap:Envelope>"""
            # Headers necesarios para la solicitud SOAP
            headers = {
                'Content-Type': 'text/xml; charset=utf-8',
                'SOAPAction': 'http://ws.sdde.bccr.fi.cr/ObtenerIndicadoresEconomicos'
            }
            # Realizar la solicitud POST
            response = requests.post(url, data=soap_request, headers=headers)
            # Parsear el XML de la respuesta
            root = ET.fromstring(response.content)
            # Extraer los datos de interés
            data = []
            for item in root.findall(".//INGC011_CAT_INDICADORECONOMIC"):
                cod_ind = item.find("COD_INDICADORINTERNO").text
                fecha = item.find("DES_FECHA").text.replace("T00:00:00-06:00", "")  # Limpiar la fecha
                valor = float(item.find("NUM_VALOR").text)  # Convertir a flotante
            # Agregar a la lista de datos
                data.append({
                    "CÓDIGO_IND": cod_ind,
                    "FECHA": fecha,
                    "VALOR": valor,
                    "INDICADOR": variable
                })
            # Crear el DataFrame
            df = pd.DataFrame(data)
            # Concatenar los DataFrames
            df_Ind = pd.concat([df_Ind, df])
            # Formatear dataframe
            df_pivot = df_Ind.pivot(index="FECHA", columns="INDICADOR", values="VALOR").reset_index()
            # Reordenar las columnas según el diccionario
            columnas = ['FECHA'] + [col for col in self.dicc.keys() if col in df_pivot.columns]
            df_pivot = df_pivot[columnas]
        df_Ind_s = df_pivot.copy()
        columns =  " - ".join(df_Ind_s.columns[2:].values.tolist())
        # Imprimir detalles
        df_report = df_Ind_s.dropna(subset=df_Ind_s.columns[1])
        report = f"## Datos del indicador {df_Ind_s.columns[1]}"
        report += f"\n\n**{df_Ind_s.columns[1]} al Cierre:** {df_report.iloc[-1, 1]:,.2f}"
        report += f"\n\n**Fecha de último dato disponible:** {df_report['FECHA'].iloc[-1]}"
        report += "\n\n**Período seleccionado:**"
        report += f"\n\n**Fecha inicial:** {self.start_date}"
        report += f"\n\n**Fecha final:** {self.end_date}"
        report += f"\n\n**Cantidad de registros base:** {len(df_report)}"
        report += f"\n\n**Datos Importantes de la Proyección:**"
        report += f"\n\n**Cantidad de Predictores adicionales:** {len(df_Ind_s.columns[2:])}"
        report += f"\n\n**Predictores adicionales:** {columns}"
        return report, df_Ind_s
    
# Definir el agente para graficar los datos
class ChartAgentM(CodeAgent):
    def __init__(self, df_Ind, forecast):
        self.df = df_Ind
        self.forecast = forecast
    
    def run(self):
        # Convertir la columna FECHA a datetime
        self.df['FECHA'] = pd.to_datetime(self.df['FECHA'])
        # Crear la gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        df_chart = self.df.dropna(subset=self.df.columns[1])
        ax.plot(df_chart["FECHA"], df_chart[df_chart.columns[1]], color="green", label=f"{df_chart.columns[1]}")
        ax.plot(self.forecast["ds"], self.forecast["yhat"], color="red", linestyle="--", label="Predicción")
        ax.set_title(f"Indicador {df_chart.columns[1]}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xticks(rotation=45)
        plt.legend()
        return fig
    
# Definir el agente para proyectar los datos
class PredictionAgentM(CodeAgent):
    def __init__(self, df_Ind, Periodos, dicc, freq, end_date):
        self.df = df_Ind
        self.Periodos = Periodos
        self.dicc = dicc
        self.freq = freq
        self.end_date = end_date
    
    def run(self):
        # Crear dataframe de proyecciones
        df_proy = pd.DataFrame()
        # Crear modelos para cada indicador
        for col in self.df.columns[2:]:
            # Crear el modelo Prophet
            model = Prophet()
            # Mantener dataframe sin NA's
            df_n = self.df[["FECHA", col]].dropna()
            # Ajustar el modelo
            model.fit(df_n.rename(columns={"FECHA": "ds", col: "y"}))
            # Crear un DataFrame con las fechas a predecir
            if self.dicc[col][1] == 'D':
                future = model.make_future_dataframe(
                    periods=int(self.Periodos * (1 if self.freq == 'D' else 30 if self.freq == 'M' else 90)), freq='D')
            elif self.dicc[col][1] == 'M':
                future = model.make_future_dataframe(
                    periods=int(self.Periodos * (1 if self.freq == 'M' else 1/30 if self.freq == 'D' else 3)), freq='M')
            elif self.dicc[col][1] == 'Q':
                future = model.make_future_dataframe(
                    periods=int(self.Periodos * (1 if self.freq == 'Q' else 1/90 if self.freq == 'D' else 1/3)), freq='Q')
            # Realizar la proyección
            forecast = model.predict(future)
            # Agregar a dataframe
            df_proy_1 = pd.DataFrame()
            df_proy_1[f"{col}_proy"] = forecast["yhat"]
            df_proy = pd.concat([df_proy, df_proy_1], axis=1)
        
        # Formatear dataframe de acuerdo con peridiocidad
        df_or = self.df.copy()
        if self.dicc[df_or.columns[1]][1] == 'D':
            # Completar NA's
            df_or.fillna(method='ffill', inplace=True)
            # Borrar NA's restantes
            df_or.dropna(inplace=True)
        elif self.dicc[df_or.columns[1]][1] == 'M':
            # Borrar NA's de la segunda columna
            df_or.dropna(subset=[list(self.dicc.keys())[0]], inplace=True)
            # Completar NA's
            df_or.fillna(method='ffill', inplace=True)
            # Borrar NA's restantes
            df_or.dropna(inplace=True)
        elif self.dicc[df_or.columns[1]][1] == 'Q':
            # Borrar NA's
            df_or.dropna(inplace=True)
        
        #Obtener última fecha
        last_date = pd.to_datetime(df_or["FECHA"].iloc[-1])
        # Cambiar nombres dataframe
        df_or.columns = ['ds', 'y'] + [col for col in df_or.columns[2:]]
        # Crear el modelo Prophet
        model = Prophet()
        # Agregar regresores
        for col in df_or.columns[2:]:
            model.add_regressor(col)
        # Ajustar el modelo
        model.fit(df_or)
        # Crear un DataFrame con las fechas a predecir
        future = model.make_future_dataframe(periods=self.Periodos, freq=self.freq)
        df_proy_2 = pd.DataFrame()
        for col in df_or.columns[2:]:
            if self.freq == 'D': # TODO: Listo
                if self.dicc[col][1] == 'D':
                    future[col] = pd.concat([df_or[col], df_proy[f"{col}_proy"][-self.Periodos:]], ignore_index=True)
                elif self.dicc[col][1] == 'M':
                    factor = 30
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    df_proy_2[f"{col}_proy"] = np.repeat(df_proy_3[f"{col}_proy"].values, factor)
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"][-self.Periodos:]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
                elif self.dicc[col][1] == 'Q':
                    factor = 90
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    df_proy_2[f"{col}_proy"] = np.repeat(df_proy_3[f"{col}_proy"].values, factor)
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"][-self.Periodos:]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
            elif self.freq == 'M':# TODO: Listo
                if self.dicc[col][1] == 'M':
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    future[col] = pd.concat([df_or[col], df_proy_3[f"{col}_proy"][-self.Periodos:]], ignore_index=True) 
                elif self.dicc[col][1] == 'D':
                    factor = self.Periodos*30
                    df_proy_2[f"{col}_proy"] = df_proy[f"{col}_proy"][-factor:].values
                    dates_next_months = [(last_date + pd.DateOffset(months=i)).to_period("M").end_time.normalize() for i in range(1, self.Periodos+1)]
                    star_date = pd.to_datetime(self.end_date)
                    dates_next = pd.date_range(start=star_date + pd.Timedelta(days=1), periods=factor, freq='D')
                    df_proy_2['Fecha'] = dates_next
                    df_proy_2 = df_proy_2[df_proy_2['Fecha'].isin(dates_next_months)]
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
                elif self.dicc[col][1] == 'Q':
                    factor = 3
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    df_proy_2[f"{col}_proy"] = np.repeat(df_proy_3[f"{col}_proy"].values, factor)
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"][-self.Periodos:]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
            elif self.freq == 'Q':# TODO: Listo
                dates_next_quarters = [(last_date + pd.DateOffset(months=i*3)).to_period("Q").end_time.normalize() for i in range(1, self.Periodos+1)]
                if self.dicc[col][1] == 'Q': 
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    future[col] = pd.concat([df_or[col], df_proy_3[f"{col}_proy"][-self.Periodos:]], ignore_index=True)
                elif self.dicc[col][1] == 'D': 
                    factor = self.Periodos*90
                    df_proy_2[f"{col}_proy"] = df_proy[f"{col}_proy"][-factor:].values
                    star_date = pd.to_datetime(self.end_date)
                    dates_next = pd.date_range(start=star_date + pd.Timedelta(days=1), periods=factor, freq='D')
                    df_proy_2['Fecha'] = dates_next
                    df_proy_2 = df_proy_2[df_proy_2['Fecha'].isin(dates_next_quarters)]
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
                elif self.dicc[col][1] == 'M':
                    factor = self.Periodos*3
                    df_proy_3 = df_proy.copy()
                    df_proy_3.dropna(inplace=True) # !Cambio fundamental
                    df_proy_2[f"{col}_proy"] = df_proy_3[f"{col}_proy"][-factor:].values
                    star_date = pd.to_datetime(self.end_date)
                    dates_next = pd.date_range(start=star_date + pd.Timedelta(days=1), periods=factor, freq='M')
                    df_proy_2['Fecha'] = dates_next
                    df_proy_2 = df_proy_2[df_proy_2['Fecha'].isin(dates_next_quarters)]
                    future[col] = pd.concat([df_or[col], df_proy_2[f"{col}_proy"]], ignore_index=True)
                    df_proy_2 = pd.DataFrame()
        # Realizar la proyección
        future = future.dropna()
        forecast = model.predict(future)
        # Retornar proyecciones
        return forecast[['ds','yhat']]
    
# Definir el multiagente
class MultiAgentM(ToolCallingAgent):
    def __init__(self, dicc, start_date, end_date, name, email, token, Periodos, freq, api_key):
        self.dicc = dicc
        self.start_date = start_date
        self.end_date = end_date
        self.name = name
        self.email = email
        self.token = token
        self.Periodos = Periodos
        self.freq = freq
        self.api_key = api_key

    def run(self):
        # Crear el agente para obtener los datos
        agent1 = IndicatorDataM(self.dicc, self.start_date, self.end_date, self.name, self.email, self.token)
        report, df_Ind_2 = agent1.run()
        # Crear el agente para predecir los datos
        agent3 = PredictionAgentM(df_Ind_2, self.Periodos, self.dicc, self.freq, self.end_date)
        forecast = agent3.run()
        # Crear el agente para graficar los datos
        agent2 = ChartAgentM(df_Ind_2, forecast)
        fig = agent2.run()
        # Crear el agente para obtener noticias
        agent4 = NewsAgent(df_Ind_2.columns[1], self.end_date, forecast, self.api_key)
        resume = agent4.run()
        # Crear el reporte final
        report += f"\n\n**Fecha final de proyección:** {forecast['ds'].iloc[-1].strftime('%Y/%m/%d')}"
        report += f"\n\n**Predicción del indicador:** {forecast['yhat'].iloc[-1].round(2)}"
        report += "\n\n## Gráfica del indicador"
        return report, fig, resume