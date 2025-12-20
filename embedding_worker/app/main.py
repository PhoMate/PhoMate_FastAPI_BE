from fastapi import FastAPI

app = FastAPI(title="PhoMate Embedding Worker")

@app.get("/health")
def health():
    return {"status": "ok"}
