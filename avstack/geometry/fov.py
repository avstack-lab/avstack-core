import itertools
import json
from typing import TYPE_CHECKING, Dict, FrozenSet, Union


if TYPE_CHECKING:
    from avstack.geometry.bbox import Box3D

import numpy as np
from shapely import Point as ShapelyPoint
from shapely import Polygon as ShapelyPolygon
from shapely.prepared import prep

from avstack.config import GEOMETRY
from avstack.geometry.datastructs import PointMatrix3D
from avstack.geometry.refchoc import ReferenceDecoder, ReferenceFrame
from avstack.geometry.utils import in_polygon, parallel_in_polygon


def box_in_fov(box: "Box3D", fov: Union["Shape", np.ndarray]):
    # get the corner points as a list of vectors then run fov
    corners = [corner for corner in box.corners]
    return any(points_in_fov(corners, fov))


def box_center_in_fov(box: "Box3D", fov: Union["Shape", np.ndarray]):
    return points_in_fov(box.position.x, fov)


def points_in_fov(points, fov: Union["Shape", np.ndarray]):
    try:
        if isinstance(points[0], (list, np.ndarray)):
            in_fov = fov.check_points(points)
        else:
            in_fov = fov.check_point(points)
    except (TypeError, AttributeError) as e:
        try:
            if isinstance(points[0], (list, np.ndarray)):
                in_fov = parallel_in_polygon(points, fov)
            else:
                in_fov = in_polygon(points, fov)
        except Exception:
            raise e
    return in_fov


def get_disjoint_fov_subsets(fovs: Dict[int, "Shape"]) -> Dict[FrozenSet[int], "Shape"]:
    """compute all disjoint FOV subsets from all agents

    A disjoint subsets are subsets that have no overlap with each other.

    Computing all subsets amounts to looping over the 2^n number of
    distinct agent combinations and checking for FOV overlap. Some
    optimizations and caching can be done for efficiency.

    Reducing all subsets to "maximal" disjoint subsets amounts to
    pruning the list of subsets if the list of agent IDs is a strict subset
    of a larger list of agent IDs with overlap
    """
    agent_IDs = list(fovs.keys())
    fov_subsets = {
        frozenset({ID}): fov for ID, fov in fovs.items()
    }  # dict of agent ID sets to the subset
    overlap_ID_combos = {(ID,) for ID in agent_IDs}
    n = len(fovs)  # choose from all agents
    k = 2  # choose pairs of agents

    # Compute all disjoint subsets of overlap
    while True:
        # check all unique combinations of k agents for overlap
        found_a_combo = False
        agent_combos = itertools.combinations(range(n), k)
        for combo in agent_combos:
            IDs = frozenset([agent_IDs[idx] for idx in combo])

            # check if the km1 combos are valid
            all_subsets_valid = True
            all_km1_combos = list(itertools.combinations(IDs, k - 1))
            for km1_combo in all_km1_combos:
                if km1_combo not in overlap_ID_combos:
                    all_subsets_valid = False
                    break

            # check overlaps only if all subsets are valid
            if all_subsets_valid:
                # get starting point and remaining IDs
                IDs_use = all_km1_combos[0]  # just pick first arbitrarily
                overlap = fov_subsets[frozenset(IDs_use)]
                IDs_rem = IDs.difference(IDs_use)

                # check the remaining ID FOVs for overlap
                for ID in IDs_rem:
                    overlap = overlap.intersection(fovs[ID])
                    if overlap is None:
                        break
                else:
                    # modify the km1 subsets to reflect the disjoint nature
                    for km1_combo in all_km1_combos:
                        fov_subsets[frozenset(km1_combo)] = fov_subsets[
                            frozenset(km1_combo)
                        ].difference(overlap)
                    # add the new set to the dict
                    found_a_combo = True
                    fov_subsets[IDs] = overlap

        # check if we need to increment k and keep going
        if not found_a_combo:
            break  # we can't possibly find a more maximal one
        else:
            k += 1

    return fov_subsets


class FieldOfViewEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Polygon):
            fov_dict = {
                "boundary": o.boundary.tolist(),
                "reference": o.reference.encode(),
                "frame": o.frame,
                "timestamp": o.timestamp,
            }
            return {"polygon": fov_dict}
        else:
            raise NotImplementedError(f"{type(o)}, {o}")


class FieldOfViewDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "polygon" in json_object:
            json_object = json_object["polygon"]
            return Polygon(
                boundary=np.array(json_object["boundary"]),
                reference=json.loads(json_object["reference"], cls=ReferenceDecoder),
                frame=int(json_object["frame"]),
                timestamp=float(json_object["timestamp"]),
            )
        else:
            return json_object


class Shape:
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "Shape"

    @property
    def area(self):
        raise NotImplementedError

    def change_reference(self, reference: ReferenceFrame, inplace: bool):
        raise NotImplementedError

    def check_point(self, point: np.ndarray) -> bool:
        raise NotImplementedError

    def check_points(self, point: np.ndarray) -> np.ndarray[bool]:
        raise NotImplementedError
    
    def intersects(self, other: "Shape") -> bool:
        raise NotImplementedError
    
    def intersection(self, other: "Shape") -> float:
        raise NotImplementedError


@GEOMETRY.register_module()
class Polygon(Shape):
    def __init__(
        self,
        boundary: np.ndarray,
        reference: ReferenceFrame,
        frame: int = None,
        timestamp: float = None,
    ):
        self.polygon = ShapelyPolygon(boundary)
        self.reference = reference
        self.frame = frame
        self.timestamp = timestamp

    def __str__(self) -> str:
        return f"Polygon with boundary {self.boundary} at reference {self.reference}"

    @property
    def boundary(self):
        return self.polygon.boundary.coords._coords

    def change_reference(self, reference: ReferenceFrame, inplace: bool):
        """Change the polygon reference frame

        HACK: Assumes that the boundary is a 2D BEV set of points in X-Y plane"""
        boundary = np.concatenate(
            (self.boundary, np.zeros((self.boundary.shape[0], 1))), axis=1
        )
        boundary = PointMatrix3D(x=boundary, calibration=self.reference)
        boundary = boundary.change_reference(reference, inplace=False)[:, :2]

        if inplace:
            self.polygon = ShapelyPolygon(boundary)
            self.reference = reference
        else:
            return Polygon(boundary=boundary, reference=reference)

    def check_point(self, point: np.ndarray) -> bool:
        """Checks whether a point is within the polygon"""
        return self.polygon.contains(ShapelyPoint(*point))

    def check_points(self, points: np.ndarray) -> np.ndarray[bool]:
        """Checks a batch of points to be in a polygon"""
        prep_poly = prep(self.polygon)
        points_shape = [ShapelyPoint(*point) for point in points]
        return filter(prep_poly.contains, points_shape)

    def distance_point(self, point: np.ndarray) -> float:
        return self.polygon.exterior.distance(ShapelyPoint(*point))

    def distance_points(self, points: np.ndarray) -> np.ndarray[float]:
        """Checks distance of a batch of points"""
        return np.array([self.distance_point(point) for point in points])
    
    def intersects(self, other: "Polygon") -> bool:
        return self.polygon.intersects(other.polygon)
    
    def intersection(self, other: "Polygon") -> float:
        return self.polygon.intersection(other.polygon)

    def encode(self):
        return json.dumps(self, cls=FieldOfViewEncoder)


@GEOMETRY.register_module()
class Wedge(Shape):
    def __init__(self, radius: float, angle_start: float, angle_stop: float) -> None:
        """Define a wedge as a part of a circle

        Assumes unit circle convention where:
            (1, 0) is at 0
            (0, 1) is at pi/2
            (-1,0) is at pi or -pi
            (0,-1) is at -pi/2
        """
        self.radius = radius
        self.angle_start = angle_start
        self.angle_stop = angle_stop

    @property
    def angle_range(self):
        return (self.angle_stop % (2 * np.pi)) - (self.angle_start % (2 * np.pi))

    @property
    def area(self):
        return self.radius**2 * self.angle_range / 2

    def check_point(self, point: np.ndarray):
        if len(point.shape) == 1:
            point = point[:, None]
        flag_rng = np.linalg.norm(point, axis=0) <= self.radius
        az = np.arctan2(point[1, :], point[0, :])
        flag_az_1 = self.angle_start <= az
        flag_az_2 = az <= self.angle_stop
        return flag_rng & flag_az_1 & flag_az_2


@GEOMETRY.register_module()
class Circle(Shape):
    def __init__(self, radius: float, center: np.ndarray = np.zeros((2,))) -> None:
        self.radius = radius
        self.center = center
        if len(center) != 2:
            raise ValueError(f"Cannot handle center of len {len(center)}")

    def __str__(self) -> str:
        return f"Circle with radius {self.radius} at center {self.center}"

    def check_point(self, point: np.ndarray):
        return (
            np.linalg.norm(point[: len(self.center)] - self.center, axis=0)
            <= self.radius
        )

    def intersection(self, other: "Shape") -> Union["Shape", None]:
        if isinstance(other, (Sphere, Circle)):
            d = np.linalg.norm(self.center - other.center)
            if (d + self.radius) < other.radius:
                return self
            elif (d + other.radius) < self.radius:
                return other
            elif d < (self.radius + other.radius):
                overlap = Vesica([self, other])
            else:
                overlap = None
        elif isinstance(other, Vesica):
            overlap = other.intersection(self)
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "Shape") -> Union["Shape", None]:
        raise NotImplementedError

    def change_reference(self, reference):
        pass


@GEOMETRY.register_module()
class Sphere(Circle):
    def __init__(self, radius: float, center: np.ndarray = np.zeros((3,))) -> None:
        self.radius = radius
        self.center = center


@GEOMETRY.register_module()
class Vesica(Shape):
    def __init__(self, circles):
        """Shape formed by intersection of circles"""
        self.circles = circles

    def intersection(self, other: "Shape") -> Union["Shape", None]:
        if isinstance(other, Circle):
            for circle in self.circles:
                if np.linalg.norm(circle.center - other.center) >= max(
                    circle.radius, other.radius
                ):
                    overlap = None
                    break
            else:
                overlap = Vesica(self.circles + [other])
        elif isinstance(other, Vesica):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return overlap

    def difference(self, other: "Shape") -> Union["Shape", None]:
        raise NotImplementedError


# class FieldOfView:
#     def check_point(self, point: np.ndarray):
#         raise NotImplementedError

#     def intersection(self, other: "FieldOfView") -> Union["FieldOfView", None]:
#         raise NotImplementedError

#     def difference(self, other: "FieldOfView") -> Union["FieldOfView", None]:
#         raise NotImplementedError


# class ParametricFieldOfView(FieldOfView):
#     def __init__(self) -> None:
#         """Uses a particular shape to form a field of view"""
#         super().__init__()

#     def check_point(self, point: np.ndarray):
#         return any([shape.check_point(point) for shape in self.shapes])


# def fov_from_radar(range_doppler):
#     raise NotImplementedError


# def fov_from_lidar(pc: LidarData):
#     """Use LiDAR data to estimate the field of view"""
#     raise NotImplementedError
