import numpy as np
import cv2
import sys
import torch
from PIL import Image
import numpy as np
from time import perf_counter
from utils.common import load_checkpoint
from utils.lane_det_utils import lane_as_segmentation_inference, prob_to_lines
from utils.vis_utils import lane_detection_visualize_batched, save_images
from utils.models.lane_detection.utils import lane_pruning
from utils.models import MODELS
from utils.transforms import functional as F

BENCHMARK = False
device = torch.device("cpu")

# from configs/lane_detection/resa/resnet50_culane.py
input_sizes = (360, 640)  # defined in the pretrained model
max_lane = 4
gap = 10
ppl = 56
thresh = 0.3
dataset = "llamas"


def run():

    # parse first cli argument as image path
    img_path = sys.argv[1]

    time_start = perf_counter()

    image = Image.open(img_path).convert("RGB")

    orig_sizes = (image.height, image.width)
    original_img = F.to_tensor(image).clone().unsqueeze(0)
    image = F.resize(image, size=input_sizes)

    if BENCHMARK:
        print(f"Image resize took {perf_counter() - time_start} seconds")

    pytorch_in = F.to_tensor(image).unsqueeze(0)

    pytorch_model = get_model()
    load_checkpoint(
        net=pytorch_model,
        optimizer=None,
        lr_scheduler=None,
        filename="../resnet50_resa_tusimple_20211019.pt",
    )

    if BENCHMARK:
        for i in range(100):
            time_start = perf_counter()
            pytorch_out = pytorch_model(pytorch_in)
            print(f"Inference took {perf_counter() - time_start} seconds")
    else:
        pytorch_out = pytorch_out = pytorch_model(pytorch_in)

    keypoints = lane_as_segmentation_inference(
        None,
        pytorch_out,
        [input_sizes, orig_sizes],
        gap,
        ppl,
        thresh,
        dataset,
        max_lane,
        forward=False,  # already called model
    )

    assert len(keypoints[0]) > 0, "No lanes detected"

    keypoints = [[np.array(lane) for lane in image] for image in keypoints]

    results = lane_detection_visualize_batched(
        original_img, keypoints=keypoints, style="line"
    )

    cv2.imshow("Inferred image", results[0])
    cv2.waitKey(5000)
    save_images(results, ["./inference.png"])


def get_model():
    model = dict(
        name="RESA_Net",
        backbone_cfg=dict(
            name="predefined_resnet_backbone",
            backbone_name="resnet50",
            return_layer="layer3",
            pretrained=False,
            replace_stride_with_dilation=[False, True, True],
        ),
        reducer_cfg=dict(name="RESAReducer", in_channels=1024, reduce=128),
        spatial_conv_cfg=dict(name="RESA", num_channels=128, iteration=5, alpha=2.0),
        classifier_cfg=dict(name="BUSD", in_channels=128, num_classes=7),
        lane_classifier_cfg=dict(
            name="EDLaneExist",
            num_output=7 - 1,
            flattened_size=4400,
            dropout=0.1,
            pool="avg",
        ),
    )

    return MODELS.from_dict(model)


if __name__ == "__main__":
    run()
