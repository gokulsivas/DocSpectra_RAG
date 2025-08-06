from fastapi import FastAPI
from app.router import router
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DocSpectra - HackRx 6.0")
app.include_router(router, prefix="/hackrx")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:8000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

load_dotenv()

