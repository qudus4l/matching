import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Bind to all available network interfaces
        port=port,
        reload=False,  # Disable reload in production
        workers=1  # Use a single worker process
    ) 