import logging
import os
from contextlib import asynccontextmanager

import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .db.mongodb import MongoDB, mongodb
# from db.mongodb import MongoDB, mongodb
from .models.responses import BaseResponse
from .api.chat import router as chat_router
from .api.user import router as user_router
# from models.responses import BaseResponse
# from api.chat import router as chat_router
# from api.user import router as user_router

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    print("LIFESPAN: Starting up...")
    logger.info("LIFESPAN: Connecting to MongoDB...")
    try:
        await mongodb.connect_to_mongodb()
        print("LIFESPAN: MongoDB connected successfully")
        logger.info("LIFESPAN: MongoDB connected successfully")
    except Exception as e:
        print(f"LIFESPAN: Error connecting to MongoDB: {e}")
        logger.error(f"LIFESPAN: Error connecting to MongoDB: {e}")
        raise
    yield
    print("LIFESPAN: Shutting down...")
    logger.info("LIFESPAN: Closing MongoDB connection...")
    await mongodb.close_mongodb_connection()

# Create FastAPI backend
app = FastAPI(
    title="AI Chat API",
    description="API for managing AI chat conversations",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(user_router, prefix="/api", tags=["user"])

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse(
            statusCode=exc.status_code,
            message=exc.detail,
            data=None
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=BaseResponse(
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            data=None
        ).model_dump(),
    )

@app.get("/", response_model=BaseResponse)
async def root():
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="AI Chat API is running",
        data=None
    )

@app.get("/health", response_model=BaseResponse)
async def health_check():
    db_status = "connected" if mongodb.db is not None else "disconnected"
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message=f"API is running - Database: {db_status}",
        data={"database": db_status, "client": mongodb.client is not None}
    )

cli = typer.Typer()

@cli.command()
def runserver(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(3434, help="Port to bind"),
    reload: bool = typer.Option(True, help="Enable auto-reload"),
):
    """Run the FastAPI server."""
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )

if __name__ == "__main__":
    cli()