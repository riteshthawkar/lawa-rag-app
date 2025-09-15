import os
import uvicorn
from app import app 
from modules.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)