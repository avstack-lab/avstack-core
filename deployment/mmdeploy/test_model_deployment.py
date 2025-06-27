import argparse
import os

import cv2
from mmdeploy_runtime import Detector


def main(model_path, img):
    detector = Detector(model_path=model_path, device_name="cuda", device_id=0)
    bboxes, labels, _ = detector(img)
    print(bboxes, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to converted model")
    args = parser.parse_args()

    MMDET_BASE = "../../third_party/mmdetection"
    img = cv2.imread(os.path.join(MMDET_BASE, "demo", "demo.jpg"))
    main(args.model_path, img)
