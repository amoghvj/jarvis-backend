# J.A.R.V.I.S. Backend – Telemetry Ingestion (milestone 1)

Receives telemetry events at `POST /v1/events`, validates them, and prints them to the console. No storage or processing.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger docs: http://localhost:8000/docs
- Health check: `GET /health`

Invalid JSON/payloads return 422; unexpected errors return 500.

