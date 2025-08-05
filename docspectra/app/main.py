from fastapi import FastAPI
from app.router import router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocSpectra - HackRx 6.0")
app.include_router(router, prefix="/hackrx")
