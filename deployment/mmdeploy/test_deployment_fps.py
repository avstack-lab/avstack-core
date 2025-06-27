import argparse
import os
import time

import cv2
import numpy as np
from avstack.calibration import CameraCalibration
from avstack.geometry import GlobalOrigin3D
from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
from avstack.sensors import ImageData
from mmdeploy_runtime import Detector


def run_with_timing(detector, img, n_inferences):
    timing = []
    for i in range(n_inferences):
        t1 = time.time()
        _ = detector(img)
        t2 = time.time()
        if i > 1:  # just in case first two times are slow...
            timing.append((t2 - t1) * 1000)
    print(f"Timing Results:\n   {np.mean(timing):4.2f} +/- {np.std(timing):4.2f} ms")


def main_mmdeploy(model_path, img, n_inferences=50):
    detector = Detector(model_path=model_path, device_name="cuda", device_id=0)
    print("Running mmdeploy:")
    run_with_timing(detector, img, n_inferences)


def main_mmdet(dataset, model, img, n_inferences=50):
    detector = MMDetObjectDetector2D(
        model=model,
        dataset=dataset,
        gpu=0,
    )
    cam_calib = CameraCalibration(
        reference=GlobalOrigin3D, P=np.random.rand(3, 4), img_shape=img.shape
    )
    img = ImageData(timestamp=0, frame=0, data=img, calibration=cam_calib, source_ID=0)
    print("Running mmdetection")
    run_with_timing(detector, img, n_inferences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="coco", type=str, help="Name of the dataset to use"
    )
    parser.add_argument(
        "--model", default="cascade_rcnn", type=str, help="Name of the model to use"
    )
    parser.add_argument(
        "--runtime", default="tensorrt", type=str, help="NN runtime to use for deploy"
    )
    args = parser.parse_args()

    mmdep_model_path = os.path.join(
        "mmdeploy_models", f"{args.model}_{args.dataset}_{args.runtime}"
    )
    if not os.path.exists(mmdep_model_path):
        raise FileNotFoundError("Cannot find model for mmdeploy")

    MMDET_BASE = "../../third_party/mmdetection"
    img = cv2.imread(os.path.join(MMDET_BASE, "demo", "demo.jpg"))
    n_inferences = 20
    main_mmdeploy(mmdep_model_path, img, n_inferences=n_inferences)
    main_mmdet(args.dataset, args.model, img, n_inferences=n_inferences)
