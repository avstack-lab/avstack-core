# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-08-08
# @Description:
"""

"""

import sys

from avstack.modules import perception


sys.path.append("tests/")
from utilities import get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    radar_calib,
    rad,
    box_2d,
    box_3d,
) = get_test_sensor_data()


def test_mmdet_2d_perception():
    frame = 0
    try:
        pass
    except ModuleNotFoundError as e:
        print("Cannot run mmdet test without the module")
    else:

        model_dataset_pairs = [
            ("fasterrcnn", "kitti"),
            ("fasterrcnn", "cityscapes"),
            ("fasterrcnn", "coco-person"),
            ("fasterrcnn", "carla"),
            ("cascadercnn", "carla"),
            ("rtmdet", "coco"),
        ]

        for model, dataset in model_dataset_pairs:
            detector = perception.object2dfv.MMDetObjectDetector2D(
                model=model, dataset=dataset
            )
            detections = detector(img, frame=frame, identifier="camera_objects_2d")
