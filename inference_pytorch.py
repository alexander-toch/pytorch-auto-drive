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

BENCHMARK = True

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

    if BENCHMARK:
        print(f"Image resize took {perf_counter() - time_start} seconds")

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

    if BENCHMARK:
        for i in range(100):
            time_start = perf_counter()
            pytorch_out = pytorch_model(pytorch_in)
            print(f"Inference took {perf_counter() - time_start} seconds")
    else:
        pytorch_out = pytorch_out = pytorch_model(pytorch_in)

    prob_map = torch.nn.functional.interpolate(
        pytorch_out["out"],
        size=input_sizes,
        mode="bilinear",
        align_corners=True,
    ).softmax(dim=1)
    existence_conf = torch.FloatTensor(pytorch_out["lane"]).sigmoid()

    existence = existence_conf > 0.5
    if max_lane != 0:  # Lane max number prior for testing
        existence, existence_conf = lane_pruning(
            existence, existence_conf, max_lane=max_lane
        )
    prob_map = prob_map.detach().cpu().numpy()
    existence = existence.cpu().numpy()

    # Get coordinates for lanes
    # print(np.count_nonzero(torch.FloatTensor(prob_map[0]) > 0.5))
    # print(prob_map[0].shape)  # should be classes, height, width

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
    keypoints = [[np.array(lane) for lane in image] for image in lane_coordinates]

    out_images = lane_detection_visualize(
        torch.FloatTensor(input), keypoints, style="point"
    )
    # print(out_images)
    print(out_images.shape)
    print(out_images[0].shape)
    print(type(out_images[0]))

    Image.fromarray(out_images[0]).save("out.png")

    # cv2.imshow("Inferred image", out_images[0])
    # cv2.waitKey()
    # o = lane_detection_visualize_batched(outputs, style="line")
    # save_images(o, ["./test.png"])


def lane_detection_visualize(
    images,
    keypoints,
    keypoint_color=None,
    control_points=None,
    style="point",
    line_trans=0.4,
):
    print(f"Shape: {images.shape}")
    # print(type(images))

    images = images.clamp_(0.0, 1.0) * 255.0
    images = images[..., [1, 2, 0]].cpu().numpy().astype(np.uint8)

    print(f"Shape: {images.shape}")
    if keypoint_color is None:
        keypoint_color = [0, 0, 0]  # Black (sits well with lane colors)
    else:
        keypoint_color = keypoint_color[::-1]  # To BGR

    if style == "point":
        images[0] = draw_points(images[0], keypoints[0], keypoint_color)
    elif style == "line":
        overlay = images[0].copy()
        overlay = draw_points_as_lines(overlay, keypoints[0], keypoint_color)
        images[0] = (
            images[0].astype(np.float) * line_trans
            + overlay.astype(np.float) * (1 - line_trans)
        ).astype(np.uint8)
    else:
        raise ValueError(
            "Unknown keypoint visualization style: {}\nPlease use point/line".format(
                style
            )
        )
    # images = images[..., [2, 1, 0]]

    return images


def draw_points(image, points, colors, radius=5, thickness=-1):
    # Draw lines (defined by points) on an image as keypoints
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0]):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.circle(
                image,
                (int(temp[k][0]), int(temp[k][1])),
                radius=radius,
                color=color,
                thickness=thickness,
            )
    return image


def draw_points_as_lines(image, points, colors, thickness=3):
    # Draw lines (defined by points) on an image by connecting points to lines
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0] - 1):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.line(
                image,
                (int(temp[k][0]), int(temp[k][1])),
                (int(temp[k + 1][0]), int(temp[k + 1][1])),
                color=color,
                thickness=thickness,
            )
    return image


if __name__ == "__main__":
    run()
