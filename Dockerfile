FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /usr/code

ENV KMP_DUPLICATE_LIB_OK=True

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["uvicorn"]

CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]