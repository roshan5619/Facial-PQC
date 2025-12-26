from fastapi import FastAPI, Request # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse # pyright: ignore[reportMissingImports]
from fastapi.staticfiles import StaticFiles # pyright: ignore[reportMissingImports]
import time
import os
from app.core.config import settings
from app.api.endpoints import auth, registration, admin, health
from app.db.database import Base, engine

# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Face Authentication API",
    description="AI-powered face recognition with liveness detection",
    version="1.0.0",
    debug=settings.DEBUG
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local frontend
        "http://localhost:5173",
        "https://requagnize-product-1.onrender.com",
        "https://reqagnize.netlify.app/",  # Vite dev server
        "*"  # Allow all for now (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Mount static files (if needed)
upload_dir = "data/uploads/enrollment_images"
if os.path.exists(upload_dir):
    app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(registration.router, prefix="/api/registration", tags=["Registration"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Face Authentication System API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "auth": "/api/auth",
            "registration": "/api/registration",
            "admin": "/api/admin"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    print("="*60)
    print("🚀 Face Authentication System Starting...")
    print("="*60)
    print(f"✓ Models loaded from: {settings.MODELS_DIR}")
    print(f"✓ Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'configured'}")
    print(f"✓ Debug mode: {settings.DEBUG}")
    print("="*60)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down Face Authentication System...")

if __name__ == "__main__":
    import uvicorn # pyright: ignore[reportMissingImports]
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )