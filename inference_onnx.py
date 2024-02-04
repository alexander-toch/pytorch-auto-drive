import onnxruntime as ort
import numpy as np
import cv2
import sys
import torch
from PIL import Image
import numpy as np
from time import perf_counter
from utils.lane_det_utils import lane_as_segmentation_inference, prob_to_lines
from utils.models.lane_detection.utils import lane_pruning
from utils.models import MODELS
from utils.transforms import functional as F
from utils.vis_utils import lane_detection_visualize_batched, save_images

BENCHMARK = True

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

    model_in = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))

    model_in = model_in.view(image.size[1], image.size[0], len(image.getbands()))
    model_in = (
        model_in.permute((2, 0, 1)).contiguous().float().div(255).unsqueeze(0).numpy()
    )
    if BENCHMARK:
        print(f"Image preprocessing took {perf_counter() - time_start} seconds")

    ort_sess = ort.InferenceSession(
        "../resnet50_resa_tusimple_20211019.onnx",
        providers=ort.get_available_providers(),
    )

    if BENCHMARK:
        for _ in range(100):
            time_start = perf_counter()
            onnx_out = ort_sess.run(
                None, {"input1": model_in}
            )  # TODO: set width and height dynamically
            print(f"Inference (model only) took {perf_counter() - time_start} seconds")
    else:
        onnx_out = ort_sess.run(None, {"input1": model_in})
    outputs = {"out": torch.Tensor(onnx_out[0]), "lane": torch.Tensor(onnx_out[1])}

    keypoints = lane_as_segmentation_inference(
        None,
        outputs,
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
        original_img, keypoints=keypoints, style="point"
    )

    cv2.imshow("Inferred image", results[0])
    cv2.waitKey(5000)
    save_images(results, ["./inference.png"])


if __name__ == "__main__":
    run()
