# FastAPI

Python package to create REST APIs 

- Easy to learn & use
- Fast development
- Async -> High performance
- Automatic Documentation

## Installation

```bash
pip install fastapi uvicorn
```

## Execution

```bash
uvicorn main:app --reload
```

## Docs

Go to

- `http://127.0.0.1/8000/docs`

  or

- `http://127.0.0.1/8000/redoc`

  or

- `http://127.0.0.1/8000/openapi.json`

## Imports

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```

## Basic App

```python
app = FastAPI()


class Item(BaseModel):
    text: str = None
    is_done: bool = False


items = []


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/items")
def create_item(item: Item):
    items.append(item)
    return items


@app.get("/items", response_model=list[Item])
def list_items(limit: int = 10):
    return items[0:limit]


@app.get("/items/{item_id}", response_model=Item)
def get_item(item_id: int) -> Item:
    if item_id < len(items):
        return items[item_id]
    else:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
```

## Return Files

```python
import os

from fastapi import FastAPI 
from fastapi.responses import FileResponse

app = FastAPI()

path = "/home/anthony/fastapifileexample"

@app.get("/")
def index():
    return {"Hello": "World"}

@app.get("/cat", responses={200: {"description": "A picture of a cat.", "content" : {"image/jpeg" : {"example" : "No example available. Just imagine a picture of a cat."}}}})
def cat():
    file_path = os.path.join(path, "files/cat.jpg")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg", filename="mycat.jpg")
    return {"error" : "File not found!"}
```

