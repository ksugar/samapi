# SAM API

## Install

Create a conda environment.

```bash
conda create -n samapi -y python=3.11
conda activate samapi
```

Install `cudatoolkit`.

```bash
conda install -y cudatoolkit=11.3
```

Install `samapi` and its dependencies.

```bash
python -m pip install git+https://github.com/ksugar/samapi.git
```

If you are using WSL2, `LD_LIBRARY_PATH` will need to be updated as follows.

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Usage

### Launch a server

```bash
uvicorn samapi.main:app
```

The command above will launch a server at http://localhost:8000.

```
INFO:     Started server process [21258]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

For more information, see [uvicorn documentation](https://www.uvicorn.org/#command-line-options).

### Request body

```python
class SAMBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    bbox: Tuple[int, int, int, int] = Field(example=(0, 0, 0, 0))
    b64img: str
```

| key    | value                                   |
| ------ | --------------------------------------- |
| type   | One of `vit_h`, `vit_l`, or `vit_b`     |
| bbox   | Coordinate of a bbox `(x1, y1, x2, y2)` |
| b64img | Base64-encoded image data               |

### Response body

The response body contains a list of [GeoJSON Feature objects](https://geojson.org).

Supporting other formats is a future work.