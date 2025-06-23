from __future__ import annotations

import json
from typing import List

import numpy as np

from avstack.config import REFERENCE

from . import transformations as tforms
from .base import fastround, q_mult_vec


class VectorEncoder(json.JSONEncoder):
    def default(self, o):
        v_dict = {
            "x": o.x.tolist(),
            "reference": o.reference.encode(),
            "n_prec": o.n_prec,
        }
        return {type(o).__name__.lower(): v_dict}


class RotationEncoder(json.JSONEncoder):
    def default(self, o):
        q_dict = {
            "qw": o.q.w,
            "qv": o.q.vec.tolist(),
            "reference": o.reference.encode(),
            "n_prec": o.n_prec,
        }
        return {type(o).__name__.lower(): q_dict}


class ReferenceEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PassiveReferenceFrame):
            ref_dict = {"frame_id": o.frame_id, "timestamp": o.timestamp}
        else:
            if o.reference is None:
                reference = None
            else:
                reference = o.reference.encode()
            ref_dict = {
                "x": o.x.tolist(),
                "qw": o.q.w,
                "qv": o.q.vec.tolist(),
                "v": o.v.tolist(),
                "acc": o.acc.tolist(),
                "angw": o.ang.w,
                "angv": o.ang.vec.tolist(),
                "reference": reference,
                "handedness": o.handedness,
                "n_prec": o.n_prec,
                "level": o.level,
            }
        return {"reference": ref_dict}


class ReferenceDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(json_object):
        if "reference" in json_object:
            if json_object["reference"] is None:
                return GlobalOrigin3D
            else:
                if isinstance(json_object["reference"], ReferenceFrame):
                    return json_object["reference"]
                else:
                    if "x" in json_object:
                        reference = json.loads(
                            json_object["reference"], cls=ReferenceDecoder
                        )
                        return ReferenceFrame(
                            x=np.array(json_object["x"]),
                            q=np.quaternion(json_object["qw"], *json_object["qv"]),
                            v=np.array(json_object["v"]),
                            acc=np.array(json_object["acc"]),
                            ang=np.quaternion(
                                json_object["angw"], *json_object["angv"]
                            ),
                            reference=reference,
                            handedness=json_object["handedness"],
                            n_prec=json_object["n_prec"],
                        )
                    else:
                        json_object = json_object["reference"]
                        return PassiveReferenceFrame(
                            frame_id=json_object["frame_id"],
                            timestamp=json_object["timestamp"],
                        )
        else:
            return json_object


@REFERENCE.register_module()
class PassiveReferenceFrame:
    def __init__(
        self,
        frame_id: str,
        timestamp: float,
    ) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp

    def __str__(self):
        return f"PassiveReferenceFrame - {self.frame_id} frame at {self.timestamp} time"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: PassiveReferenceFrame):
        if isinstance(other, PassiveReferenceFrame):
            return (self.frame_id == other.frame_id) and (
                self.timestamp == other.timestamp
            )
        else:
            raise NotImplementedError(
                f"Cannot check equality between passive reference frame and {type(other)}"
            )

    def encode(self):
        return json.dumps(self, cls=ReferenceEncoder)


@REFERENCE.register_module()
class ReferenceFrame:
    def __init__(
        self,
        x: np.ndarray,
        q: np.quaternion,
        reference: ReferenceFrame,
        v: np.ndarray = None,
        acc: np.ndarray = None,
        ang: np.quaternion = None,
        handedness="right",
        n_prec=8,
        from_frame="",
        to_frame="",
        timestamp=0.0,
    ) -> None:
        self.n_prec = n_prec
        self.dim = len(x)
        if v is None:
            v = np.zeros((self.dim,), dtype=float)
        if acc is None:
            acc = np.zeros((self.dim,), dtype=float)
        if ang is None:
            ang = np.quaternion(1)

        # -- add double linkage
        self._point_from = []
        if reference is not None:
            reference._point_from.append(self)

        # -- set fields
        self.x = np.array(x, float) if isinstance(x, list) else x.astype(float)
        self.q = np.quaternion(*q) if isinstance(q, list) else q
        self.v = v.astype(float)
        self.acc = acc.astype(float)
        self.ang = ang

        # -- store other things
        self.reference = reference
        if reference is None:
            self.level = 0
            self.is_global_origin = True
            self._hash = hash(self.x.tobytes() + self.q.vec.tobytes())
            self._global_integrated = None  # will need to reintegrate
            if not np.all(x == 0):
                raise ValueError("Without input reference, must be a global origin")
        else:
            self.level = self.reference.level + 1
            self.is_global_origin = False
            self._hash = None
            self._global_integrated = None  # will need to reintegrate
        self.handedness = handedness
        assert self.handedness == "right"
        self._fixed = None
        self.from_frame = from_frame
        self.to_frame = to_frame
        self.timestamp = timestamp

        # -- everything must trace back to the global origin
        ref_check = self
        for _ in range(self.level):
            ref_check = ref_check.reference
        assert ref_check.is_global_origin

    @property
    def fixed(self):
        if self._fixed is None:
            self._fixed = (
                np.allclose(self.v, np.zeros((self.dim,)))
                and np.allclose(self.acc, np.zeros((self.dim,)))
                and np.allclose(self.ang.vec, np.zeros((self.dim,)))
            )
        return self._fixed

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be of type np.ndarray")
        y = np.empty_like(x)
        self._x = fastround(x, self.n_prec, y)
        self.set_reupdate()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if not isinstance(q, np.quaternion):
            raise TypeError(f"q must be of type np.quaternion -- is {type(q)}")
        # y = np.empty((3,))
        self._q = np.quaternion(
            # np.round(q.w, self.n_prec), *fastround(q.vec, self.n_prec, y)
            q.w,
            *q.vec,
        )
        self.set_reupdate()

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v: np.ndarray):
        if not isinstance(v, np.ndarray):
            raise TypeError("v must be of type np.ndarray")
        y = np.empty_like(v)
        self._v = fastround(v, self.n_prec, y)
        self.set_reupdate()
        self._fixed = None

    @property
    def acc(self):
        return self._acc

    @acc.setter
    def acc(self, acc: np.ndarray):
        if not isinstance(acc, np.ndarray):
            raise TypeError("acc must be of type np.ndarray")
        y = np.empty_like(acc)
        self._acc = fastround(acc, self.n_prec, y)
        self.set_reupdate()
        self._fixed = None

    @property
    def ang(self):
        return self._ang

    @ang.setter
    def ang(self, ang: np.quaternion):
        if not isinstance(ang, np.quaternion):
            raise TypeError(f"ang must be of type np.quaternion - is {type(ang)}")
        # y = np.empty((3,))
        self._ang = np.quaternion(
            # np.round(ang.w, self.n_prec), *fastround(ang.vec, self.n_prec, y)
            ang.w,
            *ang.vec,
        )
        self.set_reupdate()
        self._fixed = None

    @property
    def ancestors(self):
        ancestors = [self]
        ref_check = self
        while ref_check.level > 0:
            ancestors.append(ref_check.reference)
            ref_check = ref_check.reference
        return ancestors

    def __str__(self) -> str:
        if self.level == 0:
            return f"GlobalOrigin"
        else:
            return f"ReferenceFrame level {self.level}, x: {self.x}, q: {self.q}, v: {self.v}, acc: {self.acc}, ang: {self.ang}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        """Lazy hashing -- but still need to check upstreams"""
        if self._hash is None:
            self._hash = hash(
                self.x.tobytes()
                + self.q.vec.tobytes()
                + self.v.tobytes()
                + self.acc.tobytes()
                + self.ang.tobytes()
            )
        return self._hash + hash(self.reference)

    def __eq__(self, other: ReferenceFrame):
        if isinstance(other, ReferenceFrame):
            if hash(self) == hash(other):
                return True
            else:
                return False
        else:
            raise NotImplementedError(
                f"Cannot check equality between reference frame and {type(other)}"
            )

    def deepcopy(self):
        return ReferenceFrame(
            x=self.x.copy(),
            q=self.q.copy(),
            reference=self.reference.copy(),
        )

    def copy(self):
        if self.is_global_origin:
            return self
        else:
            return ReferenceFrame(
                x=self.x.copy(),
                q=self.q.copy(),
                reference=self.reference.copy(),
            )

    def set_reupdate(self):
        self._hash = None
        self._global_integrated = None
        for ref in self._point_from:
            ref.set_reupdate()

    def get_ground_projected_plane(self) -> List[np.ndarray]:
        ref_gp = self.get_ground_projected_reference()
        v1 = np.array([np.sqrt(2) / 2, 0, 0])
        v2 = np.array([0, np.sqrt(2) / 2, 0])
        plane = [
            Vector(v1, ref_gp).change_reference(self, inplace=False, angle_only=True).x,
            Vector(v2, ref_gp).change_reference(self, inplace=False, angle_only=True).x,
        ]
        return plane

    def get_ground_projected_reference(self):
        if self == GlobalOrigin3D:
            return self
        elif self.reference != GlobalOrigin3D:
            ref = self.integrate(start_at=GlobalOrigin3D)
        else:
            ref = self
        x_new = np.array([ref.x[0], ref.x[1], 0])  # flat on ground
        yaw_old = tforms.transform_orientation(ref.q, "quat", "euler")[2]
        q_new = tforms.transform_orientation(
            [0, 0, yaw_old], "euler", "quat"
        )  # keep old yaw
        ref_new = ReferenceFrame(
            x=x_new,
            q=q_new,
            reference=ref.reference,
            from_frame=self.from_frame,
            to_frame=self.to_frame + "/ground",
            timestamp=self.timestamp,
        )
        return ref_new

    def get_static_reference(self):
        # integrate reference frame
        if self == GlobalOrigin3D:
            return self
        elif self.reference != GlobalOrigin3D:
            ref = self.integrate(start_at=GlobalOrigin3D)
        else:
            ref = self

        # set new reference without velocity
        ref_new = ReferenceFrame(
            x=ref.x,
            q=ref.q,
            reference=ref.reference,
            from_frame=self.from_frame,
            to_frame=self.to_frame + "/static",
            timestamp=self.timestamp,
        )

        return ref_new

    def encode(self):
        return json.dumps(self, cls=ReferenceEncoder)

    def allclose(self, other: ReferenceFrame):
        """Check if two are nearly the same via the differential"""
        if self == other:
            return True
        else:
            diff = self.differential(other)
            return (
                np.allclose(
                    diff.x,
                    np.zeros(
                        self.dim,
                    ),
                )
                and np.allclose(
                    diff.q.vec,
                    np.zeros(
                        self.dim,
                    ),
                )
                and np.allclose(
                    diff.v,
                    np.zeros(
                        self.dim,
                    ),
                )
                and np.allclose(
                    diff.acc,
                    np.zeros(
                        self.dim,
                    ),
                )
                and np.allclose(
                    diff.ang.vec,
                    np.zeros(
                        self.dim,
                    ),
                )
            )

    def check_hash_trail(self):
        target = self
        for anc in self.ancestors:
            if target._upstream_hash == anc._hash:
                pass
            else:
                return False
            target = anc
        return True

    def integrate(self, start_at: ReferenceFrame):
        """Integrate the transformation from the global

        Each reference frame is first a rotation then a
        translation away from the previous corodinate frame.
        Therefore, to integrate, apply the following.
        Consider the transformation A --> B --> C. Let's
        say we have already integrated A --> B. Now we
        wish to finalize with C.

        x_A_to_C_in_A = q_B_to_A * x_B_to_C_in_B + x_A_to_B_in_A
        q_A_to_C      = q_B_to_C * q_A_to_B
        """
        if (self._global_integrated is not None) and (start_at.is_global_origin):
            return self._global_integrated
        else:
            ancestors = self.ancestors
            started = False
            for anc in reversed(ancestors):
                if not started:
                    if anc == start_at:
                        x_tot = np.zeros((self.dim,))
                        q_tot = np.quaternion(1)
                        v_tot = np.zeros((self.dim,))
                        acc_tot = np.zeros((self.dim,))
                        ang_tot = np.quaternion(1)
                        started = True
                else:
                    x_tot = q_mult_vec(q_tot.conjugate(), anc.x) + x_tot
                    if not anc.fixed:
                        v_tot = q_mult_vec(q_tot.conjugate(), anc.v) + v_tot
                        acc_tot = q_mult_vec(q_tot.conjugate(), anc.acc) + acc_tot
                        ang_tot = anc.ang * ang_tot  # TODO: this is definitely wrong
                    q_tot = anc.q * q_tot
            if not started:
                raise RuntimeError("Could not find place to start integration")
            integrated = ReferenceFrame(
                x=x_tot, q=q_tot, v=v_tot, acc=acc_tot, ang=ang_tot, reference=start_at
            )
            if start_at.is_global_origin:
                self._global_integrated = integrated
            return integrated

    def common_ancestor(self, other: ReferenceFrame, exact=True):
        """Search back to get a common ancestor

        This will at least always get back to the global origin
        """
        for ref1 in self.ancestors:
            for ref2 in other.ancestors:
                if exact:
                    if ref1 == ref2:
                        return ref1
                else:
                    if ref1.allclose(ref2):
                        return ref1
        else:
            raise RuntimeError("Should have gotten back to global origin")

    def differential(self, other: ReferenceFrame, in_self: bool = True):
        """Get the differential between two frames  (**self_to_other** default)

        Step 1: find a common ancestor
        Step 2: compute ancestor to self
        Step 3: compute ancestor to other
        Step 4: compute differential via ancestor
        """
        anc = self.common_ancestor(other)
        A_2_self = self.integrate(start_at=anc)
        A_2_other = other.integrate(start_at=anc)

        if in_self:
            x_self_to_other_in_A = A_2_other.x - A_2_self.x
            x_self_to_other_in_self = q_mult_vec(A_2_self.q, x_self_to_other_in_A)
            q_self_to_other = A_2_other.q * A_2_self.q.conjugate()
            if not (self.fixed and other.fixed):
                v_self_to_other_in_A = A_2_other.v - A_2_self.v
                v_self_to_other_in_self = q_mult_vec(A_2_self.q, v_self_to_other_in_A)
                acc_self_to_other_in_A = A_2_other.acc - A_2_self.acc
                acc_self_to_other_in_self = q_mult_vec(
                    A_2_self.q, acc_self_to_other_in_A
                )
                ang_self_to_other = (
                    A_2_other.ang * A_2_self.ang.conjugate()
                )  # TODO this is definitely wrong
                ref = ReferenceFrame(
                    x=x_self_to_other_in_self,
                    q=q_self_to_other,
                    v=v_self_to_other_in_self,
                    acc=acc_self_to_other_in_self,
                    ang=ang_self_to_other,
                    reference=self,
                )
            else:
                ref = ReferenceFrame(
                    x=x_self_to_other_in_self, q=q_self_to_other, reference=self
                )
            return ref
        else:
            x_other_to_self_in_A = A_2_self.x - A_2_other.x
            x_other_to_self_in_other = q_mult_vec(A_2_other.q, x_other_to_self_in_A)
            q_other_to_self = A_2_self.q * A_2_other.q.conjugate()
            if not (self.fixed and other.fixed):
                v_other_to_self_in_A = A_2_self.v - A_2_other.v
                v_other_to_self_in_other = q_mult_vec(A_2_other.v, v_other_to_self_in_A)
                acc_other_to_self_in_A = A_2_self.acc - A_2_other.acc
                acc_other_to_self_in_other = q_mult_vec(
                    A_2_other.acc, acc_other_to_self_in_A
                )
                ang_other_to_self = (
                    A_2_self.ang * A_2_other.ang.conjugate()
                )  # TODO this is definitely wrong
                ref = ReferenceFrame(
                    x=x_other_to_self_in_other,
                    q=q_other_to_self,
                    v=v_other_to_self_in_other,
                    acc=acc_other_to_self_in_other,
                    ang=ang_other_to_self,
                    reference=other,
                )
            else:
                ref = ReferenceFrame(
                    x=x_other_to_self_in_other, q=q_other_to_self, reference=other
                )
            return ref

    def engross(self):
        """Incorporate the frame shift as the former reference"""
        return ReferenceFrame(
            x=np.zeros(
                self.dim,
            ),
            q=np.quaternion(1),
            reference=self,
            n_prec=self.n_prec,
        )

    def as_passive_frame(self):
        return PassiveReferenceFrame(frame_id=self.to_frame, timestamp=self.timestamp)


GlobalOrigin3D = ReferenceFrame(
    x=np.zeros((3,)),
    q=np.quaternion(1),
    v=np.zeros((3,)),
    acc=np.zeros((3,)),
    ang=np.quaternion(1),
    reference=None,
    handedness="right",
    to_frame="world",
    from_frame="world",
)

GlobalOrigin2D = ReferenceFrame(
    x=np.zeros((2,)),
    q=np.quaternion(1),
    v=np.zeros((2,)),
    acc=np.zeros((2,)),
    ang=np.quaternion(1),
    reference=None,
    handedness="right",
    to_frame="world",
    from_frame="world",
)


class Vector:
    def __init__(
        self, x: np.ndarray, reference: ReferenceFrame, n_prec: int = 8
    ) -> None:
        self.n_prec = n_prec
        self.x = np.asarray(x, dtype=float)
        self.reference = reference

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        y = np.empty_like(x)
        self._x = fastround(x, self.n_prec, y)

    @property
    def finite(self):
        return np.all(np.isfinite(self.x))

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        assert isinstance(reference, (PassiveReferenceFrame, ReferenceFrame))
        self._reference = reference

    def __str__(self):
        return f"{type(self)} - {self.x}, {self.reference}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.x)

    def __iter__(self):
        return self.x

    def __getitem__(self, key: int):
        return self.x[key]

    def __setitem__(self, key: int, value: float):
        # self.x[key] = np.round(value, self.n_prec)
        self.x[key] = value

    def __neg__(self):
        return self.factory()(-self.x, self.reference, n_prec=self.n_prec)

    def __add__(self, other: Vector, inplace: bool = False):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                other = other.change_reference(self.reference, inplace=False)
            other = other.x
        elif isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform addition
        if inplace:
            self.x = self.x + other
        else:
            return self.factory()(self.x + other, self.reference, self.n_prec)

    def __sub__(self, other: Vector):
        return -(-self + other)  # have to do this weird order!!

    def __mul__(self, other: Vector, inplace: bool = False):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                other = other.change_reference(self.reference, inplace=False)
            other = other.x
        elif isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform multiplication
        if inplace:
            self.x = self.x * other
        else:
            return self.factory()(self.x * other, self.reference, self.n_prec)

    def __truediv__(self, other, inplace: bool = False):
        # Wrapping
        if isinstance(other, (int, np.ndarray, float)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform division
        if inplace:
            self.x = self.x / other
        else:
            return self.factory()(self.x / other, self.reference, self.n_prec)

    def __matmul__(self, other):
        # Perform wrapping
        if isinstance(other, Vector):
            if self.reference != other.reference:
                other = other.change_reference(self.reference, inplace=False)
            other = other.x
        elif isinstance(other, (np.ndarray)):
            pass
        else:
            raise NotImplementedError(type(other))

        # Perform dot product
        return self.x @ other

    def deepcopy(self):
        return self.factory()(self.x.copy(), self.reference.copy())

    def copy(self):
        return self.factory()(self.x, self.reference)

    def encode(self):
        return json.dumps(self, cls=VectorEncoder)

    def allclose(self, other: Vector):
        if self.reference == other.reference:
            return np.allclose(self.x, other.x)
        else:
            other = other.change_reference(self.reference, inplace=False)
            return np.allclose(self.x, other.x)

    def change_reference(
        self, reference: ReferenceFrame, inplace: bool, angle_only: bool = False
    ):
        """Change of reference frame of a vector

        Step 1: compute the differential (self to other)
        Step 2: apply the differential to this object

        self.x : x_ref1_to_point_in_ref1
        diff.x : x_ref1_to_ref2_in_ref1
        diff.q : q_ref1_to_ref2

        x : x_ref2_to_point_in_ref2 <-- diff.q * (self.x - diff.x)
        """
        diff = self.reference.differential(reference, in_self=True)  # self to other
        diff_x = self._pull_from_reference(diff)
        if angle_only:
            x = q_mult_vec(diff.q, self.x)
        else:
            x = q_mult_vec(diff.q, self.x - diff_x)
        if inplace:
            self.x = x
            self.reference = reference
        else:
            return self.factory()(x, reference, self.n_prec)

    def in_global(self):
        return self.change_reference(GlobalOrigin3D, inplace=False)

    @staticmethod
    def factory():
        return Vector

    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.x

    def sqrt(self):
        return np.sqrt(self.x)

    def norm(self):
        return np.linalg.norm(self.x)

    def unit(self):
        return self.factory()(self.x / np.linalg.norm(self.x), self.reference)

    def distance(self, other, check_reference=True):
        if isinstance(other, ReferenceFrame):
            s2 = self.change_reference(other, inplace=False)
            return s2.norm()
        else:
            # try to pull off "position" first
            try:
                other = other.position
            except AttributeError:
                pass

            # compute the distance
            if check_reference:
                return (self - other).norm()
            else:
                return np.linalg.norm(self.x - other.x)


class VectorHeadTail:
    def __init__(
        self,
        head: np.ndarray,
        tail: np.ndarray,
        reference: ReferenceFrame,
        n_prec: int = 8,
    ) -> None:
        self.n_prec = n_prec
        self.head = Vector(head, reference, n_prec=n_prec)
        self.tail = Vector(tail, reference, n_prec=n_prec)

    def change_reference(
        self, reference: ReferenceFrame, inplace: bool, angle_only: bool = False
    ):
        if inplace:
            self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            self.tail.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
        else:
            head = self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            tail = self.head.change_reference(
                reference, inplace=inplace, angle_only=angle_only
            )
            return VectorHeadTail(head.x, tail.x, reference)


class Spherical(Vector):
    def __init__(
        self, x: np.ndarray, reference: ReferenceFrame, n_prec: int = 8, wrapping="v1"
    ) -> None:
        """

        wrapping:
            - v1: azimuth wraps to [-pi, pi]
                  elevation wraps [-pi/2, pi/2]
            - v2: azimuth wraps to [0, 2pi]
                  elevation wraps to [-pi/2, pi/2]
        """
        super().__init__(x, reference, n_prec)
        assert wrapping in ["v1", "v2"]
        if self.wrapping == "v1":
            self.wrap_az = lambda az: (az + np.pi) % (2 * np.pi) - np.pi
            self.wrap_el = lambda el: (el + np.pi / 2) % np.pi - np.pi / 2
        elif self.wrapping == "v2":
            self.wrap_az = lambda az: az % (2 * np.pi)
            self.wrap_el = lambda el: (el + np.pi / 2) % np.pi - np.pi / 2
        else:
            raise NotImplementedError(wrapping)
        self.wrapping = wrapping

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if len(x) > 1:
            x[1] = self.wrap_az(x[1])
        if len(x) > 2:
            x[2] = self.wrap_el(x[2])
        y = np.empty_like(x)
        self._x = fastround(x, self.n_prec, y)


class Rotation:
    """
    Rotation is defined as q_reference_to_object
    """

    def __init__(self, q: np.quaternion, reference: ReferenceFrame, n_prec=8) -> None:
        self.n_prec = n_prec
        if isinstance(q, np.quaternion):
            pass
        elif isinstance(q, np.ndarray) and ((q.shape == (3, 3)) or (q.shape == (2, 2))):
            q = tforms.transform_orientation(q, "dcm", "quat")
        else:
            raise ValueError(f"{type(q)} must be quaternion or 3x3 or 2x2")
        self.q = q
        self.reference = reference

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, reference):
        assert isinstance(reference, (PassiveReferenceFrame, ReferenceFrame))
        self._reference = reference

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        # y = np.empty_like(q.vec)
        self._q = np.quaternion(
            q.w,
            *q.vec,
            # np.round(q.w, self.n_prec), *fastround(q.vec, self.n_prec, y)
        )

    @property
    def qw(self):
        return self.q.w

    @property
    def qx(self):
        return self.q.x

    @property
    def qy(self):
        return self.q.y

    @property
    def qz(self):
        return self.q.z

    @property
    def R(self):
        return tforms.transform_orientation(self.q, "quat", "dcm")

    @property
    def euler(self):
        return tforms.transform_orientation(self.q, "quat", "euler")

    @property
    def forward_vector(self):
        return self.R[0, :]

    @property
    def left_vector(self):
        return self.R[1, :]

    @property
    def up_vector(self):
        return self.R[2, :]

    @property
    def yaw(self):
        return self.euler[2]

    def conjugate(self):
        return self.factory()(self.q.conjugate(), self.reference)

    def __str__(self):
        return f"{type(self)} - {self.q}, {self.reference}"

    def __repr__(self):
        return str(self)

    def __mul__(self, other: Rotation, inplace: bool = False):
        if self.reference != other.reference:
            other = other.change_reference(self.reference, inplace=False)
        if inplace:
            self.q = self.q * other.q
        else:
            return self.factory()(self.q * other.q, self.reference, self.n_prec)

    def __matmul__(self, other):
        raise NotImplementedError

    def deepcopy(self):
        return self.factory()(self.q.copy(), self.reference.copy())

    def copy(self):
        return self.factory()(self.q, self.reference)

    def encode(self):
        return json.dumps(self, cls=RotationEncoder)

    def allclose(self, other: Rotation):
        if self.reference == other.reference:
            return np.allclose(self.q.vec, other.q.vec)
        else:
            other = other.change_reference(self.reference, inplace=False)
            return np.allclose(self.q.vec, other.q.vec)

    def angle_between(self, other: Rotation):
        if self.reference != other.reference:
            other = other.change_reference(self.reference, inplace=False)
        return 2 * np.arcsin(np.linalg.norm((self.q * other.q.conjugate()).vec))

    def change_reference(self, reference: ReferenceFrame, inplace: bool):
        """Change of reference frame of a vector

        Step 1: compute the differential
        Step 2: apply the differential to this object
        TODO: could do angles only...

        self.q : q_ref1_to_point
        diff.q : q_ref1_to_ref2

        q : q_ref2_to_point <-- q_ref1_to_point * q_ref1_to_ref2.conjugate()
        """
        diff = self.reference.differential(reference, in_self=True)  # self to other
        diff_q = self._pull_from_reference(diff)
        q = self.q * diff_q.conjugate()
        if inplace:
            self.q = q
            self.reference = reference
        else:
            return self.factory()(q, reference, self.n_prec)

    def _pull_from_reference(self, reference: ReferenceFrame):
        return reference.q

    @staticmethod
    def factory():
        return Rotation
