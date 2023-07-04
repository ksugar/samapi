# Segment Anything Models (SAM) API

![](https://github.com/ksugar/samapi/releases/download/assets/qupath-samapi.gif)

A web API for [SAM](https://github.com/facebookresearch/segment-anything) implemented with [FastAPI](https://fastapi.tiangolo.com).

This is a part of the following paper. Please [cite](#citation) it when you use this project. You will also cite [the original SAM paper](https://arxiv.org/abs/2304.02643) and [the MobileSAM paper](https://arxiv.org/abs/2306.14289).

- Sugawara, K. [*Training deep learning models for cell image segmentation with sparse annotations.*](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1) bioRxiv 2023. doi:10.1101/2023.06.13.544786


## Install

Create a conda environment.

```bash
conda create -n samapi -y python=3.11
conda activate samapi
```

If you're using a computer with CUDA-compatible GPU, install `cudatoolkit`.

```bash
conda install -y cudatoolkit=11.8
```

If you're using a computer with CUDA-compatible GPU on Windows, install `torch` with GPU-support with the following command.

```bash
# Windows with CUDA-compatible GPU only
python -m pip install torch --index-url https://download.pytorch.org/whl/cu118
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
export PYTORCH_ENABLE_MPS_FALLBACK=1 # Required for running on Apple silicon
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

#### Endpoint `/sam/`

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

#### Endpoint `/sam/automask/`

```python
class SAMAutoMaskBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    b64img: str
    points_per_side: Optional[int] = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    stability_score_offset: float = 1.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 512 / 1500
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0
```

| key                            | value                                                                                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| type                           | One of `vit_h`, `vit_l`, or `vit_b`.                                                                                                                                            |
| b64img                         | Base64-encoded image data.                                                                                                                                                      |
| points_per_side                | The number of points to be sampled along one side of the image. The total number of points is points_per_side**2.                                                               |
| points_per_batch               | Sets the number of points run simultaneously by the model. Higher numbers may be faster but use more GPU memory.                                                                |
| pred_iou_thresh                | A filtering threshold in [0,1], using the model's predicted mask quality.                                                                                                       |
| stability_score_thresh         | A filtering threshold in [0,1], using the stability of the mask under changes to the cutoff used to binarize the model's mask predictions.                                      |
| stability_score_offset         | The amount to shift the cutoff when calculated the stability score.                                                                                                             |
| box_nms_thresh                 | The box IoU cutoff used by non-maximal suppression to filter duplicate masks.                                                                                                   |
| crop_n_layers                  | If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.                        |
| crop_nms_thresh                | The box IoU cutoff used by non-maximal suppression to filter duplicate masks between different crops.                                                                           |
| crop_overlap_ratio             | Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap. |
| crop_n_points_downscale_factor | The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.                                                                           |
| min_mask_region_area           | If >0, postprocessing will be applied to remove disconnected regions and holes in masks with area smaller than min_mask_region_area. Requires opencv.                           |

- [point_grids](https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/automatic_mask_generator.py#L86-L88) is not supported.

### Response body

The response body contains a list of [GeoJSON Feature objects](https://geojson.org).

Supporting other formats is a future work.

## Updates

- v0.2.0: Support for MPS backend (MacOS) by [@petebankhead](https://github.com/petebankhead)

## Citation

Please cite my paper on [bioRxiv](https://biorxiv.org/cgi/content/short/2023.06.13.544786v1).

```.bib
@article {Sugawara2023.06.13.544786,
	author = {Ko Sugawara},
	title = {Training deep learning models for cell image segmentation with sparse annotations},
	elocation-id = {2023.06.13.544786},
	year = {2023},
	doi = {10.1101/2023.06.13.544786},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning is becoming more prominent in cell image analysis. However, collecting the annotated data required to train efficient deep-learning models remains a major obstacle. I demonstrate that functional performance can be achieved even with sparsely annotated data. Furthermore, I show that the selection of sparse cell annotations significantly impacts performance. I modified Cellpose and StarDist to enable training with sparsely annotated data and evaluated them in conjunction with ELEPHANT, a cell tracking algorithm that internally uses U-Net based cell segmentation. These results illustrate that sparse annotation is a generally effective strategy in deep learning-based cell image segmentation. Finally, I demonstrate that with the help of the Segment Anything Model (SAM), it is feasible to build an effective deep learning model of cell image segmentation from scratch just in a few minutes.Competing Interest StatementKS is employed part-time by LPIXEL Inc.},
	URL = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786},
	eprint = {https://www.biorxiv.org/content/early/2023/06/13/2023.06.13.544786.full.pdf},
	journal = {bioRxiv}
}
```

## Acknowledgements

- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [FastAPI](https://fastapi.tiangolo.com)