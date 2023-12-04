import os
from typing import List

import nltk
import pymorphy2
from fastapi import FastAPI, UploadFile, File
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from onto import Onto
from range import range_docs
from text import scribe_documents

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download("punkt")
morph = pymorphy2.MorphAnalyzer()
onto = None
documents = []
scribed_documents = []

@app.get("/clear")
async def root():
    global documents
    global scribed_documents
    documents = []
    scribed_documents = []
    return {"success": True}

@app.get("/scribe")
async def root():
    global scribed_documents
    scribed_documents = scribe_documents(onto, morph, documents)
    return scribed_documents

@app.get("/range")
async def root(beta: float, distance: float):
    return range_docs(beta, distance, scribed_documents, onto)

@app.post("/upload/ontology")
def upload_file(file: UploadFile):
    try:
        contents = file.file.read()
        path = f"files/ontology/{file.filename}"
        with open(path, 'wb') as f:
            f.write(contents)
        global onto
        onto = Onto.load_from_file(path)
        os.remove(path)
    except Exception:
        return {"error": True}
    finally:
        file.file.close()
    return {"success": True}


@app.post("/upload/documents")
def upload(files: List[UploadFile] = File(...)):
    global documents
    for file in files:
        try:
            contents = file.file.read()
            documents.append({"name": file.filename, "text": contents.decode("utf-8")})
        except Exception:
            return {"error": True}
        finally:
            file.file.close()

    return {"success": True}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
