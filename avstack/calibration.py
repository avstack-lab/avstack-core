from __future__ import annotations

import json
from typing import Union

import numpy as np

from avstack.geometry.refchoc import (
    PassiveReferenceFrame,
    ReferenceDecoder,
    ReferenceFrame,
)


class CalibrationEncoder(json.JSONEncoder):
    def default(self, o):
        calib_dict = {"reference": o.reference.encode()}
        if isinstance(
            o,
            (
                CameraCalibration,
                SemanticSegmentationCalibration,
                DepthCameraCalibration,
            ),
        ):
            calib_dict["P"] = o.P.tolist()
            calib_dict["img_shape"] = o.img_shape
            calib_dict["fov_horizontal"] = o.fov_horizontal
            calib_dict["fov_vertical"] = o.fov_vertical
            calib_dict["square_pixels"] = o.square_pixels
            calib_dict["channel_order"] = o.channel_order
            if isinstance(o, SemanticSegmentationCalibration):
                calib_dict["tags"] = o.tags
        elif isinstance(o, (ImuCalibration, GpsCalibration, LidarCalibration)):
            pass
        elif isinstance(o, RadarCalibration):
            calib_dict["fov_horizontal"] = o.fov_horizontal
            calib_dict["fov_vertical"] = o.fov_vertical
        calib_dict["version"] = o.__class__.__name__
        return {"calibration": calib_dict}


class CalibrationDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "calibration" in json_object:
            json_object = json_object["calibration"]
            reference = json.loads(json_object["reference"], cls=ReferenceDecoder)
            if json_object["version"] in (
                "CameraCalibration",
                "DepthCameraCalibration",
            ):
                factory = (
                    CameraCalibration
                    if json_object["version"] == "CameraCalibration"
                    else DepthCameraCalibration
                )
                return factory(
                    reference=reference,
                    P=np.array(json_object["P"]),
                    img_shape=json_object["img_shape"],
                    fov_horizontal=json_object["fov_horizontal"],
                    fov_vertical=json_object["fov_vertical"],
                    square_pixels=json_object["square_pixels"],
                    channel_order=json_object["channel_order"],
                )
            elif json_object["version"] == "SemanticSegmentationCalibration":
                return SemanticSegmentationCalibration(
                    reference=reference,
                    P=np.array(json_object["P"]),
                    img_shape=json_object["img_shape"],
                    tags={int(k): v for k, v in json_object["tags"].items()},
                    fov_horizontal=json_object["fov_horizontal"],
                    fov_vertical=json_object["fov_vertical"],
                    square_pixels=json_object["square_pixels"],
                    channel_order=json_object["channel_order"],
                )
            elif json_object["version"] == "LidarCalibration":
                return LidarCalibration(reference=reference)
            elif json_object["version"] == "GpsCalibration":
                return GpsCalibration(reference=reference)
            elif json_object["version"] == "ImuCalibration":
                return ImuCalibration(reference=reference)
            elif json_object["version"] == "RadarCalibration":
                return RadarCalibration(
                    reference=reference,
                    fov_horizontal=json_object["fov_horizontal"],
                    fov_vertical=json_object["fov_vertical"],
                )
            else:
                return Calibration(reference=reference)
        else:
            return json_object


class Calibration:
    def __init__(self, reference):
        """
        T - extrinsic matrix (rotation + translation)

        extrinsic matrix is transformation from global to local
        """
        self.reference = reference

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__} with reference: {self.reference}"

    def copy(self):
        raise NotImplementedError

    def deepcopy(self):
        raise NotImplementedError

    def encode(self):
        return json.dumps(self, cls=CalibrationEncoder)

    def allclose(self, other: Calibration):
        return self.reference.allclose(other.reference)

    def save_to_file(self, file):
        if not file.endswith(".txt"):
            file += ".txt"
        with open(file, "w") as f:
            f.write(self.encode())


class LidarCalibration(Calibration):
    def __init__(self, reference):
        super().__init__(reference)


class GpsCalibration(Calibration):
    def __init__(self, reference):
        super().__init__(reference)


class ImuCalibration(Calibration):
    def __init__(self, reference):
        super().__init__(reference)


class RadarCalibration(Calibration):
    def __init__(self, reference, fov_horizontal, fov_vertical):
        super().__init__(reference)
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical

    @property
    def min_fov(self):
        return min(self.fov_horizontal, self.fov_vertical)

    def allclose(self, other: RadarCalibration):
        return (
            self.reference.allclose(other.reference)
            and np.isclose(self.fov_horizontal, other.fov_horizontal)
            and np.isclose(self.fov_vertical, other.fov_vertical)
        )


class CameraCalibration(Calibration):
    def __init__(
        self,
        reference: Union[PassiveReferenceFrame, ReferenceFrame],
        P: np.ndarray,
        img_shape: tuple,
        focal_length: float = None,
        fov_horizontal=None,
        fov_vertical=None,
        square_pixels=False,
        channel_order="bgr",
    ):
        """
        P - intrinsic matrix
        T - extrinsic matrix (rotation + translation)
        img_shape - (height, width, channels) of image
        """
        self.img_shape = img_shape
        if len(img_shape) == 3:
            self.height, self.width, self.n_channels = img_shape
        else:
            self.height, self.width = img_shape
            self.n_channels = 1
        self.P = P
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)
        self.principal_point = np.array([self.c_u, self.c_v])  # in (x, y)
        # -- pixel size using linear approximation
        if not square_pixels:
            self.pixel_size_u = (
                fov_horizontal / (2 * self.f_u) if fov_horizontal is not None else None
            )
            self.pixel_size_v = (
                fov_vertical / (2 * self.f_v) if fov_vertical is not None else None
            )
        else:
            if (fov_horizontal is None) and (fov_vertical is None):
                self.pixel_size_u = None
                self.pixel_size_v = None
            elif (fov_horizontal is not None) and (fov_vertical is not None):
                pix_size = fov_horizontal / (2 * self.f_u)
                pix_size_2 = fov_vertical / (2 * self.f_v)
                if not np.isclose(pix_size, pix_size_2):
                    raise RuntimeError(
                        "Pixel size does not check out if square pixels assumed"
                    )
            elif fov_horizontal is not None:
                pix_size = fov_horizontal / (2 * self.f_u)
            else:
                pix_size = fov_vertical / (2 * self.f_v)
            self.pixel_size_u = self.pixel_size_v = pix_size
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
        self.square_pixels = square_pixels
        self.channel_order = channel_order
        super().__init__(reference)

    def __str__(self):
        return f"Camera Calibration with reference: {self.reference}; P:{self.P}"

    def allclose(self, other: CameraCalibration):
        return self.reference.allclose(other.reference) and np.allclose(self.P, other.P)

    def pixel_to_angle(self, point: np.ndarray):
        raise NotImplementedError


_carla_semseg_labels_colors = [
    ("Unlabeled", (0, 0, 0)),
    ("Building", (70, 70, 70)),
    ("Fence", (100, 40, 50)),
    ("Other", (55, 90, 80)),
    ("Pedestrian", (220, 20, 60)),
    ("Pole", (153, 153, 153)),
    ("Roadline", (157, 234, 50)),
    ("Road", (128, 64, 128)),
    ("SideWalk", (244, 25, 232)),
    ("Vegetation", (107, 142, 35)),
    ("Vehicles", (0, 0, 142)),
    ("Wall", (102, 102, 156)),
    ("TrafficSign", (220, 220, 0)),
    ("Sky", (70, 130, 180)),
    ("Ground", (81, 0, 81)),
    ("Bridge", (150, 100, 100)),
    ("RailTrack", (230, 150, 140)),
    ("GuardRail", (180, 165, 180)),
    ("TrafficLight", (250, 170, 30)),
    ("Static", (110, 190, 160)),
    ("Dynamic", (170, 120, 50)),
    ("Water", (45, 60, 150)),
    ("Terrain", (145, 170, 100)),
]
assert len(_carla_semseg_labels_colors) == 23
carla_semseg_tags = {i: label[0] for i, label in enumerate(_carla_semseg_labels_colors)}
carla_semseg_colors = {
    i: label[1] for i, label in enumerate(_carla_semseg_labels_colors)
}


class SemanticSegmentationCalibration(CameraCalibration):
    def __init__(
        self,
        reference,
        P: np.ndarray,
        img_shape: tuple,
        tags: dict = carla_semseg_tags,
        colors: list = carla_semseg_colors,
        fov_horizontal=None,
        fov_vertical=None,
        square_pixels=False,
        channel_order="bgr",
    ):
        super().__init__(
            reference=reference,
            P=P,
            img_shape=img_shape,
            fov_horizontal=fov_horizontal,
            fov_vertical=fov_vertical,
            square_pixels=square_pixels,
            channel_order=channel_order,
        )
        self.tags = tags
        self.colors = colors

    def __str__(self):
        return f"Semantic Segmentation Calibration with reference: {self.reference}; P:{self.P}"

    def allclose(self, other: SemanticSegmentationCalibration):
        return (
            self.reference.allclose(other.reference)
            and np.allclose(self.P, other.P)
            and (self.tags == other.tags)
        )


class DepthCameraCalibration(CameraCalibration):
    pass  # TODO
