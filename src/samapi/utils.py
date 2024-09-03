import base64
import io
import logging
import os
from typing import Tuple

from geojson import Polygon as geojson_polygon
import numpy as np
from PIL import Image
from shapely.geometry import Polygon as shapely_polygon
from skimage import measure

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())
logger = logging.getLogger("uvicorn")

try:
    Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS))
except:
    logger.warning(
        "PIL.Image.MAX_IMAGE_PIXELS is set to None, potentially exposing the system to decompression bomb attacks."
    )
    Image.MAX_IMAGE_PIXELS = None


def decode_image(b64data: str):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64data))))


def mask_to_geometry(
    mask: np.ndarray,
    scale: float = 1.0,
    offset: Tuple[int, int] = (0, 0),
    simplify_tol=None,
):
    # modified from https://github.com/MouseLand/cellpose_web/blob/main/utils.py
    mask = np.pad(mask, 1)  # handle edges properly by zero-padding
    contours_find = measure.find_contours(mask, 0.5)
    if len(contours_find) == 1:
        index = 0
    else:
        pixels = []
        for _, item in enumerate(contours_find):
            pixels.append(len(item))
        try:
            index = np.argmax(pixels)
        except:
            return geojson_polygon([])
    contour = contours_find[index]
    contour -= 1  # reset padding
    contour_as_numpy = contour[:, np.argsort([1, 0])]
    contour_as_numpy *= scale
    contour_as_numpy[:, 0] += offset[0]
    contour_as_numpy[:, 1] += offset[1]
    contour_asList = contour_as_numpy.tolist()
    if simplify_tol is not None:
        poly_shapely = shapely_polygon(contour_asList)
        poly_shapely_simple = poly_shapely.simplify(
            simplify_tol, preserve_topology=False
        )
        contour_asList = list(poly_shapely_simple.exterior.coords)
    return geojson_polygon([contour_asList])
