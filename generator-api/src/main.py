from fastapi import FastAPI

from increase_rate_gen.router import router as increase_rate_gen_router

app = FastAPI(title="BurnFit Routine Generation API", version="1.0.0")
app.include_router(increase_rate_gen_router)


@app.get("/")
def root():
    return {"message": "Welcome to BurnFit Routine Generation API"}
