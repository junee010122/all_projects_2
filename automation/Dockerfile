FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml environment.yml
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "automation_env", "/bin/bash", "-c"]

COPY . .

CMD ["conda", "run", "-n", "automation_env", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

