import warnings
from enum import Enum
import time
from typing import Optional, List, Tuple

from fastapi import FastAPI
from geojson import Feature
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.hub import load_state_dict_from_url
import torch

from samapi.utils import decode_image, mask_to_geometry

app = FastAPI()


class ModelType(str, Enum):
    vit_h = "vit_h"
    vit_l = "vit_l"
    vit_b = "vit_b"
    vit_t = "vit_t"


SAM_CHECKPOINTS = {
    ModelType.vit_h: load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    ),
    ModelType.vit_l: load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    ),
    ModelType.vit_b: load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
    ModelType.vit_t: load_state_dict_from_url(
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    ),
}


def _get_device() -> str:
    """
    Selects the device to use for inference, based on what is available.
    :return:
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            warnings.warn(
                "MPS not available because the current MacOS version is not "
                "12.3+ and/or you do not have an MPS-enabled device on this "
                "machine - using CPU for inference"
            )
    else:
        warnings.warn("No GPU support found - using CPU for inference")

    # Make sure that the device is ready
    if device in ("cuda", "mps"):
        try:
            dummy_input = np.zeros((16, 16, 3), dtype=np.uint8)
            SamPredictor(get_sam_model(ModelType.vit_b).to(device=device)).set_image(
                dummy_input
            )
        except Exception as e:
            warnings.warn(
                f"{device} device found but got the error {str(e)} - using CPU for inference"
            )
            device = "cpu"
    return device


def get_sam_model(model_type: ModelType, checkpoint_url: Optional[str] = None):
    sam = sam_model_registry[model_type]()
    if checkpoint_url is None:
        sam.load_state_dict(SAM_CHECKPOINTS[model_type])
    else:
        sam.load_state_dict(load_state_dict_from_url(checkpoint_url))
    return sam


device = _get_device()

sam_type = ModelType.vit_h
predictor = SamPredictor(get_sam_model(sam_type).to(device=device))
last_image = None


class SAMBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    bbox: Optional[Tuple[int, int, int, int]] = Field(example=(0, 0, 0, 0))
    point_coords: Optional[Tuple[Tuple[int, int], ...]] = Field(
        example=((0, 0), (1, 0))
    )
    point_labels: Optional[Tuple[int, ...]] = Field(example=(0, 1))
    b64img: str
    b64mask: Optional[str] = None
    multimask_output: bool = False
    checkpoint_url: Optional[str] = None


@app.post("/sam/")
async def predict_sam(body: SAMBody):
    global sam_type
    global predictor
    global last_image
    if body.type != sam_type:
        predictor = SamPredictor(get_sam_model(body.type, body.checkpoint_url).to(device=device))
        sam_type = body.type
        last_image = None
    if last_image != body.b64img:
        image = _parse_image(body)
        predictor.set_image(image)
        last_image = body.b64img
    else:
        print("Keeping the previous image!")

    start_time = time.time_ns()
    masks, quality, _ = predictor.predict(
        point_coords=_parse_point_coords(body),
        point_labels=_parse_point_labels(body),
        box=_parse_bbox(body),
        mask_input=_parse_mask(body),
        multimask_output=body.multimask_output,
    )
    end_time = time.time_ns()
    print(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")

    features = []
    for obj_int, mask in enumerate(masks):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask),
                properties={
                    "object_idx": index_number,
                    "label": "object",
                    "quality": float(quality[index_number]),
                    "sam_model": body.type,
                },
            )
        )
    return features


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
    output_type: str = "Single Mask"
    include_image_edge: bool = False


@app.post("/sam/automask/")
async def automatic_mask_generator(body: SAMAutoMaskBody):
    global sam_type
    global predictor
    global last_image
    if body.type != sam_type:
        predictor = SamPredictor(get_sam_model(body.type, body.checkpoint_url).to(device=device))
        sam_type = body.type
        last_image = None
    if last_image != body.b64img:
        image = _parse_image(body)
        predictor.set_image(image)
        last_image = body.b64img
    else:
        print("Keeping the previous image!")

    mask_generator = SamAutomaticMaskGenerator(
        predictor=predictor,
        points_per_side=body.points_per_side,
        points_per_batch=body.points_per_batch,
        pred_iou_thresh=body.pred_iou_thresh,
        stability_score_thresh=body.stability_score_thresh,
        stability_score_offset=body.stability_score_offset,
        box_nms_thresh=body.box_nms_thresh,
        crop_n_layers=body.crop_n_layers,
        crop_nms_thresh=body.crop_nms_thresh,
        crop_overlap_ratio=body.crop_overlap_ratio,
        crop_n_points_downscale_factor=body.crop_n_points_downscale_factor,
        min_mask_region_area=body.min_mask_region_area,
        output_type=body.output_type,
        include_image_edge=body.include_image_edge,
    )

    image = _parse_image(body)
    start_time = time.time_ns()
    masks = mask_generator.generate(image)
    end_time = time.time_ns()
    print(f"Prediction time: {(end_time - start_time) / 1e6:.1f} ms")

    features = []
    for obj_int, mask in enumerate(masks):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask["segmentation"]),
                properties={
                    "object_idx": index_number,
                    "label": "object",
                    "quality": mask["predicted_iou"],
                    "sam_model": body.type,
                },
            )
        )
    return features


def _parse_image(body: SAMBody):
    image = decode_image(body.b64img)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    return image


def _parse_mask(body: SAMBody):
    return None if body.b64mask is None else decode_image(body.b64mask)


def _parse_bbox(body: SAMBody):
    return None if not body.bbox else np.array(body.bbox)[None]


def _parse_point_labels(body: SAMBody):
    return None if not body.point_labels else np.array(body.point_labels)


def _parse_point_coords(body: SAMBody):
    return None if not body.point_coords else np.array(body.point_coords)
