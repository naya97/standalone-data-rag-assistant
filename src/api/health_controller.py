import os
from fastapi import FastAPI
from datetime import datetime
from src.infrastructure.db_manager import SQLiteManager

app = FastAPI(title="Standalone Data-RAG Assistant API")

@app.get("/health")
def health_check():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, 'data', 'db', 'local_storage.sqlite')
    
    db_status = "disconnected"
    
    try:
        db = SQLiteManager(db_path)
        result = db.execute_query("SELECT 1") 
        if result:
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "api_status": "healthy",
        "database_status": db_status,
        "timestamp": datetime.now().isoformat(),
        "environment": "development"
    }