import onnxruntime as ort
import numpy as np
import cv2
import sys
import torch
from PIL import Image
import numpy as np
from time import perf_counter
from utils.lane_det_utils import prob_to_lines
from utils.models.lane_detection.utils import lane_pruning
from utils.models import MODELS

BENCHMARK = False

# from configs/lane_detection/resa/resnet50_culane.py
input_sizes = (360, 640)
orig_sizes = (720, 1280)
max_lane = 6
gap = 10
ppl = 56
thresh = 0.3
dataset = "llamas"


def tensor_image_to_numpy(images):
    return (images * 255.0).cpu().numpy().astype(np.uint8)


def run():

    # parse first cli argument as image path
    img_path = sys.argv[1]

    time_start = perf_counter()

    img = cv2.imread(img_path)
    img = cv2.resize(img, input_sizes).astype(np.float32)
    height, width = img.shape[:2]

    if BENCHMARK:
        print(f"Image resize took {perf_counter() - time_start} seconds")

    ort_sess = ort.InferenceSession(
        "../resnet50_resa_tusimple_20211019.onnx",
        providers=ort.get_available_providers(),
    )
    input = img.reshape(1, 3, input_sizes[0], input_sizes[1])

    model = dict(
        name="RESA_Net",
        backbone_cfg=dict(
            name="predefined_resnet_backbone",
            backbone_name="resnet50",
            return_layer="layer3",
            pretrained=True,
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

    pytorch_model = MODELS.from_dict(model)
    pytorch_in = torch.FloatTensor(input)
    pytorch_out = pytorch_model(pytorch_in)

    if BENCHMARK:
        for i in range(100):
            time_start = perf_counter()
            outputs = ort_sess.run(
                None, {"input1": input}
            )  # TODO: set width and height dynamically
            print(f"Inference took {perf_counter() - time_start} seconds")
    else:
        outputs = ort_sess.run(None, {"input1": input})

    print(len(outputs))

    prob_map = torch.nn.functional.interpolate(
        torch.FloatTensor(outputs[0]),
        size=input_sizes,
        mode="bilinear",
        align_corners=True,
    ).softmax(dim=1)
    existence_conf = torch.FloatTensor(outputs[1]).sigmoid()

    existence = existence_conf > 0.5
    if max_lane != 0:  # Lane max number prior for testing
        existence, existence_conf = lane_pruning(
            existence, existence_conf, max_lane=max_lane
        )
    prob_map = prob_map.detach().cpu().numpy()
    existence = existence.cpu().numpy()

    # Get coordinates for lanes
    print(np.count_nonzero(torch.FloatTensor(prob_map[0]) > 0.5))
    print(prob_map[0].shape)  # should be classes, height, width

    lane_coordinates = [
        prob_to_lines(
            prob_map[0],
            existence[0],
            resize_shape=orig_sizes,
            smooth=False,
            gap=gap,
            ppl=ppl,
            thresh=thresh,
            dataset=dataset,
        )
    ]

    # print(lane_coordinates)

    # TODO: prob_map is wrong, maube the ONNX conversion did not work properly

    assert len(lane_coordinates[0]) > 0, "No lanes detected!"

    keypoints = [[np.array(lane) for lane in image] for image in lane_coordinates]


if __name__ == "__main__":
    run()
