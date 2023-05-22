import json
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_crowdai import loadSample
from models.backbone import DetectionBranch, NonMaxSuppression, R2U_Net
from models.matching import OptimalMatching


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0] / 2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max() - X.min(), Y.max() - Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(poly):
    _result = {}
    _result["category_id"] = 100
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def prediction(filename):
    # Load network modules
    model = R2U_Net()
    model = model.train()

    head_ver = DetectionBranch()
    head_ver = head_ver.train()

    suppression = NonMaxSuppression()

    matching = OptimalMatching()
    matching = matching.train()

    # NOTE: The modules are set to .train() mode during inference to make sure that the BatchNorm layers
    # rely on batch statistics rather than the mean and variance estimated during training.
    # Experimentally, using batch stats makes the network perform better during inference.

    print("Loading pretrained model")
    model.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_backbone", map_location=torch.device("cpu")
        )
    )
    head_ver.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_seg_head", map_location=torch.device("cpu")
        )
    )
    matching.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_matching", map_location=torch.device("cpu")
        )
    )

    # Initiate the dataloader

    rgb = loadSample(filename)
    features = model(rgb)
    occupancy_grid = head_ver(features)

    _, graph_pressed = suppression(occupancy_grid)
    predictions = []
    poly = matching.predict(rgb, features, graph_pressed)

    for i, pp in enumerate(poly):
        for p in pp:
            predictions.append(single_annotation([p]))

        del features
        del occupancy_grid
        del graph_pressed
        del poly
        del rgb

    return predictions


if __name__ == "__main__":
    print(prediction("0.png"))
