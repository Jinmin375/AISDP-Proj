FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/machine_learning_1/machine_learning_1.py .
CMD ["python", "machine_learning_1.py"]
