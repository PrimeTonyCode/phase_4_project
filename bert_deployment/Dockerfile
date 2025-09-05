FROM python:3.9-slim

WORKDIR /app

COPY bert_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app_bert:app", "--host", "0.0.0.0", "--port", "10000"]
