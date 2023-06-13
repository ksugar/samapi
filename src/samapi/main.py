import warnings
from enum import Enum
from typing import Optional, Tuple

from fastapi import FastAPI
from geojson import Feature
import numpy as np
from pydantic import BaseModel
from pydantic import Field
from segment_anything import sam_model_registry, SamPredictor
from torch.hub import load_state_dict_from_url
import torch

from samapi.utils import decode_image, mask_to_geometry

app = FastAPI()


class ModelType(str, Enum):
    vit_h = "vit_h"
    vit_l = "vit_l"
    vit_b = "vit_b"


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
}

def _get_device() -> str:
    """
    Selects the device to use for inference, based on what is available.
    :return:
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            return "mps"
        else:
            warnings.warn(
                "MPS not available because the current MacOS version is not "
                "12.3+ and/or you do not have an MPS-enabled device on this "
                "machine."
            )
    else:
        warnings.warn("No GPU support found - using CPU for inference")
        return "cpu"

def get_sam_model(model_type: ModelType):
    sam = sam_model_registry[model_type]()
    sam.load_state_dict(SAM_CHECKPOINTS[model_type])
    return sam

device = _get_device()
predictor = SamPredictor(get_sam_model(ModelType.vit_h).to(device=device))
sam_type = ModelType.vit_h


class SAMBody(BaseModel):
    type: Optional[ModelType] = ModelType.vit_h
    bbox: Tuple[int, int, int, int] = Field(example=(0, 0, 0, 0))
    b64img: str


@app.post("/sam/")
async def predict_sam(body: SAMBody):
    global sam_type
    global predictor
    if body.type != sam_type:
        predictor = SamPredictor(get_sam_model(body.type).to(device=device))
        sam_type = body.type
    image = decode_image(body.b64img)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(body.bbox)[None],
        multimask_output=False,
    )
    features = []
    for obj_int, mask in enumerate(masks):
        index_number = int(obj_int - 1)
        features.append(
            Feature(
                geometry=mask_to_geometry(mask),
                properties={"object_idx": index_number, "label": "object"},
            )
        )
    return features