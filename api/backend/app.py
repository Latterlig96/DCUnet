from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from service import app as service_api

app = FastAPI()
app.mount('/api/files', StaticFiles(directory='files'), name='files')

app.include_router(service_api)
