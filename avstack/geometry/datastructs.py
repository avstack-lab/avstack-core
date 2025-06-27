import json
from typing import Union

import numpy as np

import avstack

from . import transformations as tforms
from .base import q_mult_vec
from .refchoc import (
    PassiveReferenceFrame,
    ReferenceDecoder,
    ReferenceFrame,
    Rotation,
    Vector,
)


class VectorDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "vector" in json_object:
            json_object = json_object["vector"]
            factory = Vector
        elif "position" in json_object:
            json_object = json_object["position"]
            factory = Position
        elif "velocity" in json_object:
            json_object = json_object["velocity"]
            factory = Velocity
        elif "acceleration" in json_object:
            json_object = json_object["acceleration"]
            factory = Acceleration
        else:
            return json_object
        if json_object is None:
            return None
        reference = json.loads(json_object["reference"], cls=ReferenceDecoder)
        return factory(
            x=np.array(json_object["x"]),
            reference=reference,
            n_prec=json_object["n_prec"],
        )


class RotationDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "rotation" in json_object:
            json_object = json_object["rotation"]
            factory = Rotation
        elif "attitude" in json_object:
            json_object = json_object["attitude"]
            factory = Attitude
        elif "angular" in json_object:
            json_object = json_object["angular"]
            factory = AngularVelocity
        elif "angularvelocity" in json_object:
            json_object = json_object["angularvelocity"]
            factory = AngularVelocity
        else:
            return json_object
        if json_object is None:
            return None
        reference = json.loads(json_object["reference"], cls=ReferenceDecoder)
        return factory(
            q=np.quaternion(json_object["qw"], *json_object["qv"]),
            reference=reference,
            n_prec=json_object["n_prec"],
        )


class Position(Vector):
    def _pull_from_reference(
        self, reference: Union[ReferenceFrame, PassiveReferenceFrame]
    ):
        return reference.x

    @property
    def position(self):
        return self

    @staticmethod
    def factory():
        return Position


class Velocity(Vector):
    def _pull_from_reference(
        self, reference: Union[ReferenceFrame, PassiveReferenceFrame]
    ):
        return reference.v

    @staticmethod
    def factory():
        return Velocity


class Acceleration(Vector):
    def _pull_from_reference(
        self, reference: Union[ReferenceFrame, PassiveReferenceFrame]
    ):
        return reference.acc

    @staticmethod
    def factory():
        return Acceleration


class Attitude(Rotation):
    def _pull_from_reference(
        self, reference: Union[ReferenceFrame, PassiveReferenceFrame]
    ):
        return reference.q

    @staticmethod
    def factory():
        return Attitude


class AngularVelocity(Rotation):
    def _pull_from_reference(
        self, reference: Union[ReferenceFrame, PassiveReferenceFrame]
    ):
        return reference.ang

    @staticmethod
    def factory():
        return AngularVelocity


class Pose:
    def __init__(self, position: Position, attitude: Attitude) -> None:
        self.position = position
        self.attitude = attitude

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if not isinstance(position, Position):
            raise TypeError(f"Input position type {type(position)} not allowed")
        self._position = position

    @property
    def attitude(self):
        return self._attitude

    @attitude.setter
    def attitude(self, attitude):
        if not isinstance(attitude, Attitude):
            raise TypeError(f"Input position type {type(attitude)} not allowed")
        self._attitude = attitude

    @property
    def matrix(self):
        return np.block(
            [[self.attitude.R, self.position.x], [np.zeros((1, 3)), np.ones((1, 1))]]
        )

    def change_reference(self, reference, inplace: bool):
        if inplace:
            self.position.change_reference(reference=reference, inplace=inplace)
            self.attitude.change_reference(reference=reference, inplace=inplace)
        else:
            position = self.position.change_reference(
                reference=reference, inplace=inplace
            )
            attitude = self.attitude.change_reference(
                reference=reference, inplace=inplace
            )
            return Pose(position, attitude)


class Twist:
    def __init__(self, linear: Velocity, angular: AngularVelocity) -> None:
        self.linear = linear
        self.angular = angular

    @property
    def linear(self):
        return self._linear

    @linear.setter
    def linear(self, linear):
        if not isinstance(linear, Velocity):
            raise TypeError(f"Input position type {type(linear)} not allowed")
        self._linear = linear

    @property
    def angular(self):
        return self._angular

    @angular.setter
    def angular(self, angular):
        if not isinstance(angular, AngularVelocity):
            raise TypeError(f"Input position type {type(angular)} not allowed")
        self._angular = angular

    def change_reference(self, reference, inplace: bool):
        if inplace:
            self.linear.change_reference(reference=reference, inplace=inplace)
            self.angular.change_reference(reference=reference, inplace=inplace)
        else:
            linear = self.linear.change_reference(reference=reference, inplace=inplace)
            angular = self.angular.change_reference(
                reference=reference, inplace=inplace
            )
            return Twist(linear, angular)


class _PointMatrix:
    def __init__(self, x: np.ndarray, calibration) -> None:
        self.x = x
        if isinstance(calibration, ReferenceFrame):
            calibration = avstack.calibration.Calibration(calibration)
        self.calibration = calibration

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, x):
        if len(x.shape) == 1:
            x = x[:, None]
        self._x = x

    @property
    def reference(self):
        return self.calibration.reference

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, indices):
        return self.x[indices]

    def copy(self):
        return self.__class__(self.x.copy(), self.calibration)


class PointMatrix3D(_PointMatrix):
    def change_calibration(self, calibration, inplace: bool):
        return self.change_reference(calibration.reference, inplace)

    def change_reference(self, reference, inplace: bool):
        """Change of reference frame of a vector

        Step 1: compute the differential
        Step 2: apply the differential to this object

        self.x : x_ref1_to_point_in_ref1
        diff.x : x_ref1_to_ref2_in_ref1
        diff.q : q_ref1_to_ref2

        x : x_ref2_to_point_in_ref2 <-- diff.q * (self.x - diff.x)
        """
        diff = self.calibration.reference.differential(
            reference, in_self=True
        )  # self to other
        x = q_mult_vec(diff.q, self.x[:, :3] - diff.x)
        if self.x.shape[1] > 3:
            x = np.hstack((x, self.x[:, 3:]))
        if inplace:
            self.x = x
            self.calibration.reference = reference
        else:
            try:
                calib = self.calibration.__class__(reference)
            except TypeError as e:
                calib = self.calibration.__class__(
                    reference, self.calibration.P, self.calibration.img_shape
                )
            return PointMatrix3D(x, calib)

    def filter(self, mask):
        return PointMatrix3D(self.x[mask, :], self.calibration)

    def project_to_2d(self, calibration) -> "PointMatrix2D":
        matrix_3d = self.change_calibration(calibration, inplace=False)
        pts_3d_img = matrix_3d.x[:, :3]
        pts_3d_hom = tforms.cart2hom(pts_3d_img)
        pts_2d = np.dot(pts_3d_hom, np.transpose(calibration.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        pts_2d_cam = np.concatenate((pts_2d[:, 0:2], matrix_3d.x[:, 3:]), axis=1)
        return PointMatrix2D(pts_2d_cam, calibration)


class PointMatrix2D(_PointMatrix):
    @property
    def angles(self):
        """Takes pixel coordinates and get angles

        assumes x_pixels has (0,0) in the top left

        Returns the (az, el) angles
        """
        x_centered = self.calibration.principal_point - self.x
        # azel = x_centered * np.array([self.pixel_size_u, self.pixel_size_v])
        if len(x_centered.shape) == 1:
            azel = np.array(
                [
                    np.arctan2(x_centered[0], self.calibration.f_u),
                    np.arctan2(x_centered[1], self.calibration.f_v),
                ]
            )
        else:
            azel = np.zeros_like(x_centered)[:, :2]
            azel[:, 0] = np.arctan2(x_centered[:, 0], self.calibration.f_u)
            azel[:, 1] = np.arctan2(x_centered[:, 1], self.calibration.f_v)
        return azel
