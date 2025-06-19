from __future__ import annotations

import json
import logging
from typing import Union

import numpy as np
import quaternion
from numba import jit
from scipy import sparse
from scipy.spatial import QhullError

from avstack import exceptions
from avstack.geometry import transformations as tforms

from .base import _q_mult_vec, q_mult_vec
from .coordinates import CameraCoordinates, LidarCoordinates, StandardCoordinates
from .datastructs import (
    Attitude,
    PointMatrix3D,
    Position,
    RotationDecoder,
    VectorDecoder,
)
from .refchoc import GlobalOrigin3D, ReferenceFrame, Rotation, Vector
from .utils import convex_hull_intersection, poly_area


R_stan_to_cam = StandardCoordinates.get_conversion_matrix(CameraCoordinates)
q_stan_to_cam = quaternion.from_rotation_matrix(R_stan_to_cam)
R_cam_to_stan = R_stan_to_cam.T
q_cam_to_stan = q_stan_to_cam.conjugate()


# ==============================================================================
# Define bounding boxes
# ==============================================================================


class BoxEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Box2D):
            box2d = o.box2d if isinstance(o.box2d, list) else o.box2d.tolist()
            box_dict = {
                "box": box2d,
                "calibration": o.calibration.encode(),
                "ID": o.ID,
                "obj_type": o.obj_type,
            }
            return {"box2d": box_dict}
        elif isinstance(o, Box3D):
            box_dict = {
                "position": o.position.encode(),
                "attitude": o.attitude.encode(),
                "hwl": [o.h, o.w, o.l],
                "where_is_t": o.where_is_t,
                "obj_type": o.obj_type,
                "ID": o.ID,
            }
            return {"box3d": box_dict}
        elif isinstance(o, SegMask2D):
            seg_dict = {
                "shape": o.sparse_mask.shape,
                "rows": o.sparse_mask.row.tolist(),
                "cols": o.sparse_mask.col.tolist(),
                "data": o.sparse_mask.data.tolist(),
                "calibration": o.calibration.encode(),
            }
            return {"segmask2d": seg_dict}
        else:
            raise NotImplementedError(f"{type(o)}, {o}")


class BoxDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        from ..calibration import CalibrationDecoder

        if "box2d" in json_object:
            json_object = json_object["box2d"]
            calibration = json.loads(json_object["calibration"], cls=CalibrationDecoder)
            return Box2D(
                box2d=json_object["box"],
                calibration=calibration,
                ID=json_object["ID"],
                obj_type=json_object["obj_type"],
            )
        elif "box3d" in json_object:
            json_object = json_object["box3d"]
            position = json.loads(json_object["position"], cls=VectorDecoder)
            attitude = json.loads(json_object["attitude"], cls=RotationDecoder)
            return Box3D(
                position=position,
                attitude=attitude,
                hwl=json_object["hwl"],
                where_is_t=json_object["where_is_t"],
                obj_type=json_object["obj_type"],
                ID=json_object["ID"],
            )
        elif "segmask2d" in json_object:
            json_object = json_object["segmask2d"]
            calibration = json.loads(json_object["calibration"], cls=CalibrationDecoder)
            data = np.array(json_object["data"])
            rows = np.array(json_object["rows"])
            cols = np.array(json_object["cols"])
            mask = sparse.coo_matrix(
                (data, (rows, cols)), shape=json_object["shape"]
            ).toarray()
            return SegMask2D(mask=mask, calibration=calibration)
        else:
            return json_object


class SegMask2D:
    def __init__(self, mask, calibration, ID=None) -> None:
        """Mask is a binary matrix that will be converted to a sparse matrix"""
        self.calibration = calibration
        self.mask = mask
        self.sparse_mask = sparse.coo_matrix(self.mask)
        self.img_shape = calibration.img_shape
        self.ID = ID

    @property
    def data(self):
        return self.mask

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"SegMask -- {np.sum(self.mask)} points"

    def __eq__(self, other: object) -> bool:
        if self.img_shape == other.img_shape:
            return (self.sparse_mask != other.sparse_mask).nnz == 0
        else:
            return False

    def __len__(self):
        return np.sum(self.mask)

    def encode(self):
        return json.dumps(self, cls=BoxEncoder)

    def allclose(self, other):
        return np.allclose(self.data, other.data) and self.calibration.allclose(
            other.calibration
        )


class Box2D:
    def __init__(self, box2d, calibration, ID=None, obj_type=None):
        self.calibration = calibration
        if (box2d[2] < box2d[0]) or (box2d[3] < box2d[1]):
            raise exceptions.BoundingBoxError(box2d)
        self.xmin = box2d[0]
        self.ymin = box2d[1]
        self.xmax = box2d[2]
        self.ymax = box2d[3]
        self.ID = ID
        self.obj_type = obj_type

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Box2D: [%.2f, %.2f, %.2f, %.2f]" % (
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
        )

    def __eq__(self, other):
        return np.allclose(self.box2d, other.box2d)

    @property
    def box2d(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    @property
    def center(self):
        return np.asarray([(self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2])

    @property
    def corners(self):
        return np.array(
            [
                [self.xmin, self.ymin],
                [self.xmin, self.ymax],
                [self.xmax, self.ymax],
                [self.xmax, self.ymin],
            ]
        )

    @property
    def reference(self):
        return self.calibration.reference

    def allclose(self, other):
        return self.calibration.allclose(other.calibration) and np.allclose(
            np.array(self.box2d), np.array(other.box2d)
        )

    def deepcopy(self):
        return Box2D(
            box2d=self.box2d,
            calibration=self.calibration.copy(),
            ID=self.ID,
            obj_type=self.obj_type,
        )

    def check_valid(self, im_h, im_w):
        return self._x_valid(im_w) and self._y_valid(im_h)

    def _x_valid(self, im_w):
        return (
            (0 <= self.xmin < im_w)
            and (0 <= self.xmax < im_w)
            and (self.xmin < self.xmax)
        )

    def _y_valid(self, im_h):
        return (
            (0 <= self.ymin < im_h)
            and (0 <= self.ymax < im_h)
            and (self.ymin < self.ymax)
        )

    @property
    def angles(self):
        """use the calibration matrix to get az, el angles

        Procedure:
        1. Get center of bounding box (TODO: allow for centroid of mask)
        2. Change reference to principal point
        3. Use pixel size (rad/pixel) to get angles off center
        """
        return self.calibration.pixel_to_angle(self.center)

    @property
    def w(self):
        return self.xmax - self.xmin

    @property
    def width(self):
        return self.w

    @property
    def h(self):
        return self.ymax - self.ymin

    @property
    def height(self):
        return self.h

    @property
    def box2d_xywh(self):
        return [self.xmin, self.ymin, self.w, self.h]

    @property
    def box2d_xyxy(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def as_seg_mask(self):
        mask = np.zeros(self.calibration.img_shape[:2], dtype=bool)
        mask[int(self.ymin) : int(self.ymax), int(self.xmin) : int(self.xmax)] = True
        return SegMask2D(mask, self.calibration, ID=self.ID)

    def encode(self):
        return json.dumps(self, cls=BoxEncoder)

    def IoU(self, other, check_reference=True, **kwargs):
        if isinstance(other, Box2D):
            inter = box_intersection(self.box2d, other.box2d, **kwargs)
            union = box_union(self.box2d, other.box2d, **kwargs)
            iou = inter / union
        elif isinstance(other, Box3D):
            iou = other.IoU(self, check_reference=check_reference, **kwargs)
        else:
            raise NotImplementedError(type(other))
        return iou

    def squeeze(self, im_h, im_w, inplace):
        x_min_s = int(max(0, min(self.xmin, im_w - 1)))
        y_min_s = int(max(0, min(self.ymin, im_h - 1)))
        x_max_s = int(max(0, min(self.xmax, im_w - 1)))
        y_max_s = int(max(0, min(self.ymax, im_h - 1)))
        if inplace:
            self.xmin = x_min_s
            self.ymin = y_min_s
            self.xmax = x_max_s
            self.ymax = y_max_s
        else:
            return Box2D(
                [x_min_s, y_min_s, x_max_s, y_max_s],
                self.calibration,
                obj_type=self.obj_type,
                ID=self.ID,
            )

    def add_noise(self, noise_variance):
        """Add noise to each component

        NOTE: also assumes no cross correlations for now

        noise must be Gaussian of the form: [xmin, xmax, ymin, ymax]
        """
        if noise_variance is None:
            return
        assert len(noise_variance) == 4
        noise_samples = np.random.randn(len(noise_variance))
        noisy_vals = [np.sqrt(nv) * ns for nv, ns in zip(noise_variance, noise_samples)]
        self.xmin += noisy_vals[0]
        self.ymin += noisy_vals[1]
        self.xmax += noisy_vals[2]
        self.ymax += noisy_vals[3]
        self.squeeze(
            self.calibration.img_shape[0], self.calibration.img_shape[1], inplace=True
        )

    def upscale_to_3d(self, z_to_box: float, height: float = 2.0) -> "Box3D":
        """Use the similar triangles principle with focal length"""
        # get 3d coordinates of the corners
        corners_center = self.corners - self.calibration.principal_point
        x = corners_center[:, 0] * z_to_box / self.calibration.f_u
        y = corners_center[:, 1] * z_to_box / self.calibration.f_v
        z = z_to_box

        # reconstruct a bounding box
        dx = abs(max(x) - min(x))
        dy = abs(max(y) - min(y))
        width = min(dx, dy)
        length = max(dx, dy)
        hwl = [height, width, length]

        # position is of the center of the object
        position = Position(
            x=np.array([np.mean(x), np.mean(y), z - height / 2]),
            reference=self.reference,
        )

        # for now ignore the yaw angle
        # attitude is in camera reference frame
        attitude = Attitude(
            q=q_cam_to_stan,
            reference=self.reference,
        )

        # construct the bounding box
        box = Box3D(
            position=position, attitude=attitude, hwl=hwl, obj_type=self.obj_type
        )
        return box

    def change_reference(self, reference, inplace: bool):
        raise NotImplementedError


class Box3D:
    def __init__(
        self,
        position: Position,
        attitude: Attitude,
        hwl: list,
        where_is_t: str = "center",
        enforce_mins: bool = True,
        obj_type=None,
        ID=None,
    ):
        """
        q: q_XXX_to_box, normally XXX := sensor reference
        """
        h, w, l = hwl
        assert isinstance(where_is_t, str)
        self.where_is_t = where_is_t
        if enforce_mins:
            h = max(0.5, h)
            w = max(0.5, w)
            l = max(0.5, l)
        self.h, self.w, self.l = h, w, l
        assert isinstance(position, Position)
        self.position = position
        assert isinstance(attitude, Attitude)
        self.attitude = attitude
        self.obj_type = obj_type
        self.ID = ID

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            "Box3D=[h: %.2f, w: %.2f, l: %.2f] x (x: %.2f y: %.2f, z: %.2f)\n  q: %s with reference: %s"
            % (
                self.h,
                self.w,
                self.l,
                self.t[0],
                self.t[1],
                self.t[2],
                str(self.q),
                str(self.reference),
            )
        )

    def __eq__(self, other):
        c1 = (self.h == other.h) and (self.w == other.w) and (self.l == other.l)
        c2 = self.t == other.t
        c3 = self.q == other.q
        return c1 and c2 and c3

    @property
    def box(self):
        return self

    @property
    def box3d(self):
        return np.array([self.t[0], self.t[1], self.t[2], *self.hwl, *self.euler])

    @property
    def t(self):
        return self.position

    @t.setter
    def t(self, t):
        assert isinstance(t, Position)
        self.position = t

    @property
    def x(self):
        return self.position

    @property
    def q(self):
        return self.attitude

    @q.setter
    def q(self, q):
        assert isinstance(q, Attitude)
        self.attitude = q

    @property
    def reference(self):
        return self.position.reference

    def as_reference(self):
        pos = self.position.x if self.position else np.zeros((3,))
        att = self.attitude.q if self.attitude else np.quaternion(1)
        ref = self.reference
        return ReferenceFrame(x=pos, q=att, reference=ref, timestamp=self.t)

    def copy(self):
        return Box3D(
            self.position.copy(), self.attitude.copy(), self.hwl, self.where_is_t
        )

    def encode(self):
        return json.dumps(self, cls=BoxEncoder)

    def deepcopy(self):
        return Box3D(
            self.position.deepcopy(),
            self.attitude.deepcopy(),
            self.hwl,
            self.where_is_t,
        )

    def allclose(self, other):
        c1 = (
            np.isclose(self.h, other.h)
            and np.isclose(self.w, other.w)
            and np.isclose(self.l, other.l)
        )
        c2 = self.t.allclose(other.t)
        c3 = self.q.allclose(other.q)
        return c1 and c2 and c3

    @property
    def size(self):
        return np.array([self.h, self.w, self.l])

    @property
    def hwl(self):
        return self.size

    @property
    def lwh(self):
        return list(reversed(self.hwl))

    @property
    def volume(self):
        return self.h * self.w * self.l

    @property
    def center(self):
        if self.where_is_t == "center":
            return self.t
        elif self.where_is_t == "bottom":
            # HACK: assumes in ground plane...
            return self.t + np.array([0, 0, self.h / 2])
        else:
            raise NotImplementedError

    @property
    def center_global(self):
        t_glob = self.t.change_reference(GlobalOrigin3D, inplace=False)
        if self.where_is_t == "bottom":
            t_glob.x[2] += self.h / 2  # to convert to center
        return t_glob

    @property
    def yaw(self):
        """Here, yaw is 0 forward which is NOT the KITTI standard"""
        yaw = tforms.transform_orientation(self.q.q, "quat", "euler")[2]
        return yaw

    @property
    def euler(self):
        """Returns [roll, pitch, yaw]"""
        return tforms.transform_orientation(self.q.q, "quat", "euler")

    @property
    def corners(self):
        return compute_box_3d_corners(box3d=self)

    @property
    def corners_global(self):
        """Compute global corners by inverting the reference"""
        corners = self.corners
        corners.change_reference(GlobalOrigin3D, inplace=True)
        return corners

    @property
    def corners_global_without_pitch_roll(self):
        raise NotImplementedError

    def add_noise(self, noise_variance):
        """Add noise to each component

        NOTE: no noise can be added to angle yet
        NOTE: also assumes no cross correlations for now

        noise must be Gaussian of the form: [h, w, l, x, y, z]
        """
        if noise_variance is None:
            return
        assert len(noise_variance) == 6
        noise_samples = np.random.randn(len(noise_variance))
        noisy_vals = [np.sqrt(nv) * ns for nv, ns in zip(noise_variance, noise_samples)]
        self.h += noisy_vals[0]
        self.w += noisy_vals[1]
        self.l += noisy_vals[2]
        self.t += np.array(noisy_vals[3:6])

    def center_box(self, inplace=True):
        assert inplace, "Only doing inplace for now"
        if self.where_is_t == "center":
            pass
        else:
            self.t.change_reference(GlobalOrigin3D, inplace=True)
            self.t += np.array([0, 0, self.h / 2])
            self.t.change_reference(self.reference, inplace=True)
            self.where_is_t = "center"

    def change_reference(self, reference_new, inplace: bool):
        if inplace:
            self.t.change_reference(reference_new, inplace=inplace)
            self.q.change_reference(reference_new, inplace=inplace)
        else:
            newt = self.t.change_reference(reference_new, inplace=inplace)
            newr = self.q.change_reference(reference_new, inplace=inplace)
            newself = Box3D(
                newt,
                newr,
                [self.h, self.w, self.l],
                where_is_t=self.where_is_t,
                ID=self.ID,
            )
            return newself

    def distance(self, other, *args, **kwargs):
        if isinstance(other, Box3D):
            return self.position.distance(other.position, *args, **kwargs)
        elif isinstance(other, (Position, np.ndarray)):
            return self.position.distance(other, *args, **kwargs)
        else:
            raise NotImplementedError(type(other))

    def IoU(
        self,
        other,
        metric="3D",
        run_angle_check=True,
        error_on_angle_check=False,
        check_reference=True,
        debug: bool = True,
        **kwargs,
    ):
        """
        IMPORTANT NOTE: THIS METRIC ONLY WORKS WITH A YAW ANGLE
        (YAW AS DEFINED IN THE STANDARD FRAME) IT DOES NOT WORK
        WHEN THERE ARE PITCH AND ROLL ANGLES

        Check those angles in the global frame as follows:
        self.q        := q_O_2_V
        self.reference.q := q_OG_2_O

        q_OG_2_V = q_O_2_V * q_OG_2_O

        Which simplifies to:
        q_OG_2_V = self.q * self.reference.q

        """
        if isinstance(other, Box3D):
            if self.center_global.distance(other.center_global) > (
                max(self.size) / 2 + max(other.size) / 2
            ):
                return 0.0

            if run_angle_check:
                eps = 0.05
                q_stand_to_box = (
                    self.q.q * self.reference.integrate(start_at=GlobalOrigin3D).q
                )
                # q_stand_to_box = self.q * self.reference.q
                if (abs(q_stand_to_box.x) > eps) or (abs(q_stand_to_box.y) > eps):
                    msg = (
                        f"Not a good idea to run this IoU equation"
                        f" when there are more than just yaw angles in the"
                        f" global frame...{q_stand_to_box}"
                    )
                    if error_on_angle_check:
                        raise ValueError(msg)
                    else:
                        if debug:
                            logging.warning(msg)

            if metric == "3D":
                # c1 = self.corners_gloabal_without_pitch_roll
                # c2 = other.corners_global_without_pitch_roll
                c1 = self.corners_global
                c2 = other.corners_global
                inter = box_intersection(c1, c2, up="+z")
                union = box_union(c1, c2, up="+z")
                iou = inter / union
                if (iou < 0) or (iou > 2.0):
                    iou = max(
                        1.0, iou
                    )  # BUG: IoU can be >1.0 if roll and pitch angles are non-zero since the formula used approximates intersection equation as only yaw angle
                    import pdb

                    pdb.set_trace()
                    inter = box_intersection(c1, c2, up="+z")
                    union = box_union(c1, c2, up="+z")
                    raise RuntimeError("Invalid iou output")
            else:
                raise NotImplementedError(metric)
            iou = max(0.0, min(1.0, iou))
        elif isinstance(other, Box2D):
            box2d_self = self.project_to_2d_bbox(
                other.calibration, check_reference=check_reference
            )
            iou = other.IoU(box2d_self)
        else:
            raise NotImplementedError(type(other))
        return iou

    def rotate(self, q: Union[Attitude, Rotation, np.quaternion], inplace):
        """Rotates the attitude AND the translation of the box"""
        if isinstance(q, (Attitude, Rotation)):
            if not self.reference == q.reference:
                raise NotImplementedError(
                    "To apply rotation, must have identical references"
                )
            else:
                q = q.q
        if inplace:
            self.position = q_mult_vec(q, self.position)
            self.attitude.q = q * self.q.q
        else:
            pos = q_mult_vec(q, self.position)
            rot = Attitude(q * self.q.q, self.reference)
            return Box3D(pos, rot, self.size, ID=self.ID)

    def rotate_attitude(self, q, inplace):
        """Rotates the attitude of the box only"""
        if inplace:
            self.attitude = q * self.q
        else:
            rot = q * self.q
            return Box3D(self.position.copy(), rot, self.size, ID=self.ID)

    def translate(self, L, inplace):
        """Translates the position of the box"""
        if isinstance(L, (Position, Vector)):
            if not self.reference == L.reference:
                raise NotImplementedError(
                    "To apply translation, must have identical references"
                )
        if inplace:
            self.position += L
        else:
            return Box3D(self.position + L, self.attitude.copy(), self.size, ID=self.ID)

    def project_to_2d_bbox(self, calib, check_reference=True):
        """Project 3D bounding box into a 2D bounding box"""
        return proj_3d_bbox_to_2d_bbox(self, calib, check_reference).squeeze(
            calib.height, calib.width, inplace=False
        )

    def project_corners_to_2d_image_plane(self, calib, check_reference=True):
        """Project 3D bounding box corners only image plane"""
        return proj_3d_bbox_to_image_plane(self, calib, check_reference=check_reference)


# ==============================================================================
# Other Utilities
# ==============================================================================


def box_area(corners):
    """Compute 2D box area
    corners - an array of [xmin, ymin, xmax, ymax]
    """
    return (corners[2] - corners[0] + 1) * (corners[3] - corners[1] + 1)


def box_volume(corners):
    """Compute 3D box volume
    corners1: numpy array (8,3), assume up direction is +Z
    X - forward, Y - left, Z - up
    Corners start in upper top left and go counterclockwise on top plane
    then lower top left and go counterclockwise on bottom plane


          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6
    """
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def _box_intersection_2d(corners1, corners2):
    """Compute intersection of 2D boxes"""
    xA = max(corners1[0], corners2[0])
    yA = max(corners1[1], corners2[1])
    xB = min(corners1[2], corners2[2])
    yB = min(corners1[3], corners2[3])
    return max(0, xB - xA + 1) * max(0, yB - yA + 1)


def sort_corners_invariant(mat):
    """Sort matrices to be permutation-invariant

    Sort according to the proper bounding box corner ordering
    Need to flip 2 <--> 3 and  6 <--> 7 to preserve order

    Negatives to allow sorting descending
    """
    # get ordering over the columns manually...
    mat *= -1
    mat = -np.sort(mat.view("f8,f8,f8"), order=["f2", "f1", "f0"], axis=0).view(
        np.float64
    )
    mat[[2, 3], :] = mat[[3, 2], :]
    mat[[6, 7], :] = mat[[6, 7], :]
    return mat


def _box_intersection_3d(corners1, corners2, up="+z"):
    """Compute 3D bounding box IoU.

    From: https://github.com/AlienCat-K/3D-IoU-Python/blob/master/3D-IoU-Python.py
    Input:
        corners1: numpy array (8,3), assume up direction is +Z
        corners2: numpy array (8,3), assume up direction is +Z
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -    z    x
    -     --     |   -
    5-----6      |  -
                 | -
        y <------|-

    Can enforce the order by sorting in the starndard coordinates as follows:

    z, descending -- first
    y, descending -- second
    x, descending -- third
    """
    assert corners1.shape == corners2.shape, f"{corners1}, {corners2}"
    if np.any(np.isnan(corners1)) or np.any(np.isnan(corners2)):
        print(corners1)
        print(corners2)
        raise RuntimeError("NaN somewhere!")

    # Add some very small noise
    noise = 1e-12 * np.linalg.norm(corners2[0, :] - corners2[1, :])
    corners2 = corners2 + noise

    # Put in the right coordinate frame
    if up == "-y":
        R = CameraCoordinates.get_conversion_matrix(LidarCoordinates)
        corners1 = (R @ corners1.T).T
        corners2 = (R @ corners2.T).T
    elif up == "+z":
        pass
    else:
        raise NotImplementedError(up)

    # -- Get the top-down intersection (BEV)
    rect1 = [(-corners1[i, 1], corners1[i, 0]) for i in range(4)]
    rect2 = [(-corners2[i, 1], corners2[i, 0]) for i in range(4)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    if np.all(rect1 == rect2):
        inter = None
        inter_area = box_volume(corners1)
    else:
        try:
            inter_bev, inter_area_bev = convex_hull_intersection(rect1, rect2)
        except QhullError as e:
            inter = None
            inter_area = 0

    # -- Get the vertical intersection ratio
    top1, bot1 = max(corners1[:, 2]), min(corners1[:, 2])
    top2, bot2 = max(corners2[:, 2]), min(corners2[:, 2])
    z_overlap = max(0.0, min(top1, top2) - max(bot1, bot2))

    return inter_area_bev * z_overlap


def box_intersection(corners1, corners2, up="+z", **kwargs):
    if len(corners1) != len(corners2):
        raise RuntimeError(f"{len(corners1)} vs {len(corners2)} corners")
    if len(corners1) == 4:
        return _box_intersection_2d(corners1, corners2)
    elif len(corners1) == 8:
        return _box_intersection_3d(corners1, corners2, up=up)
    else:
        raise NotImplementedError(f"{len(corners1)} not implemented")


def _box_union_2d(corners1, corners2):
    return (
        box_area(corners1) + box_area(corners2) - box_intersection(corners1, corners2)
    )


def _box_union_3d(corners1, corners2, up):
    return (
        box_volume(corners1)
        + box_volume(corners2)
        - box_intersection(corners1, corners2, up=up)
    )


def box_union(corners1, corners2, up="+z", **kwargs):
    if len(corners1) != len(corners2):
        raise RuntimeError(f"{len(corners1)} vs {len(corners2)} corners")
    if len(corners1) == 4:
        return _box_union_2d(corners1, corners2)
    elif len(corners1) == 8:
        return _box_union_3d(corners1, corners2, up=up)
    else:
        raise NotImplementedError(f"{len(corners1)} not implemented")


def compute_box_size(corners_3d, heading_angle):
    """
    Takes an object's set of corners and gets h, w, l

    Corners are specified in the image rectangular coordinates
    Assumes corner convention in order to specify which is which
    """
    R = tforms.roty(heading_angle)
    box_oriented = corners_3d @ R  # invert the attitude
    l = max(box_oriented[:, 0]) - min(box_oriented[:, 0])
    h = max(box_oriented[:, 1]) - min(box_oriented[:, 1])
    w = max(box_oriented[:, 2]) - min(box_oriented[:, 2])
    return (l, w, h)


def compute_box_3d_corners(box3d):
    """A wrapper around the jit compiled method"""
    qcon = box3d.q.q.conjugate()
    qcs = qcon.w
    qcr = qcon.vec
    qcm = qcon.w**2 + qcon.x**2 + qcon.y**2 + qcon.z**2
    corners = _compute_box_3d_corners(
        box3d.t.x, box3d.h, box3d.w, box3d.l, qcs, qcr, qcm, box3d.where_is_t
    )
    return PointMatrix3D(corners, box3d.reference)


@jit(nopython=True, parallel=True)
def _compute_box_3d_corners(
    t: np.ndarray,
    h: float,
    w: float,
    l: float,
    qcs: float,
    qcr: float,
    qcm: float,
    where_is_t: str,
):
    """Computes the 3D bounding box in the box's coordinate frame

    X - forward, Y - left, Z - up
    Corners start in upper top left and go counterclockwise on top plane
    then lower top left and go counterclockwise on bottom plane


          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6

    Starts with box in standard frame, then converts to reference

    Conversion process is as follows:

    box3d.q         := q_O_2_V --> reference to vehicle body frame
    box3d.t         := x_O_2_V_in_O -- > reference to vehicle body frame in reference frame
    corners_3d_base := x_V_2_pts_in_V --> vehicle body frame to corner points in body frame

    To get corners in the reference frame, do the following:
    x_V_2_pts_in_O = q_O_2_V.conjugate() * x_V_2_pts_in_V
    x_O_2_pts_in_O = x_V_2_pts_in_O + x_O_2_V_in_O

    This expands to:
    corners_out =  box3d.q.conjugate() * corners_3d_base + box3d.t

    """
    x_corners = np.array(
        [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2], dtype=np.float64
    )
    y_corners = np.array(
        [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2], dtype=np.float64
    )
    z_corners = np.array([h, h, h, h, 0, 0, 0, 0], dtype=np.float64)
    if where_is_t == "bottom":
        pass
    elif where_is_t == "center":
        z_corners -= h / 2
    else:
        raise NotImplementedError("Cannot handle this t position")

    # -- convert to the desired coordinates in the box's frame
    corners_3d_base = np.column_stack((x_corners, y_corners, z_corners))
    # NOTE: no need to do anything with the new reference here...corners will assume
    # the reference of the vehicle

    # -- apply box rotation quaternion, then translation
    corners_3d = _q_mult_vec(qcs, qcr, qcm, corners_3d_base) + t

    return corners_3d


def compute_orientation_3d(box3d, P):
    """Takes an object and a projection matrix (P) and projects the 3d
    object orientation vector into the image plane.
    Returns:
        orientation_2d: (2,2) array in left image coord.
        orientation_3d: (2,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = tforms.roty(box3d.yaw)

    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, box3d.l], [0, 0], [0, 0]])

    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + box3d.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + box3d.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + box3d.t[2]

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)

    # project orientation into the image plane
    orientation_2d = tforms.project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


def proj_3d_bbox_to_image_plane(box3d, calib_img, check_reference=True):
    """
    Squeeze:
    Using the squeeze via-line method, the following combinations of points
    must be checked

          0-----3
         -     --
        -     - -
       -     -  -
      -     -   -
     -     -    -
    1-----2     7
    -     -    -
    -     -   -
    -     -  -
    -     - -
    -     --
    5-----6

    All pairs:
      0 -- 1
      0 -- 3
      0 -- 4
      1 -- 2
      1 -- 5
      2 -- 3
      2 -- 6
      3 -- 7
      4 -- 5
      4 -- 7
      5 -- 6
      6 -- 7
    """
    # Get 3D box points
    if check_reference:
        box3d = box3d.change_reference(calib_img.reference, inplace=False)
    box3d_pts_3d = box3d.corners

    # Project into image plane
    corners_3d_in_image = tforms.project_to_image(box3d_pts_3d, calib_img.P)
    return corners_3d_in_image


def proj_3d_bbox_to_2d_bbox(box3d, calib_img, check_reference=True):
    # Get 3D box points
    corners_3d_in_image = proj_3d_bbox_to_image_plane(box3d, calib_img, check_reference)

    # Get mins and maxes to make 2D bbox
    xmin, xmax = np.min(corners_3d_in_image[:, 0]), np.max(corners_3d_in_image[:, 0])
    ymin, ymax = np.min(corners_3d_in_image[:, 1]), np.max(corners_3d_in_image[:, 1])
    box2d = np.array([xmin, ymin, xmax, ymax])
    return Box2D(box2d, calib_img, obj_type=box3d.obj_type, ID=box3d.ID)
