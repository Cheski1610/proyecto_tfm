FROM python:3.10.11
EXPOSE 8501
RUN mkdir -p /app
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "App.py" ]