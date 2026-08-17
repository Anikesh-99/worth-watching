"""Launch the dashboard: python scripts/serve.py  ->  http://127.0.0.1:8000"""
import uvicorn
if __name__ == "__main__":
    uvicorn.run("src.serving.app:app", host="127.0.0.1", port=8000, reload=False)
