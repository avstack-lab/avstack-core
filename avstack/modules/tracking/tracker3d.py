import numpy as np

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.environment.objects import VehicleState
from avstack.geometry import Attitude, Box3D, Position, Velocity
from avstack.geometry import transformations as tforms

from ..assignment import build_A_from_iou, gnn_single_frame_assign, greedy_assignment
from . import libraries
from .base import _TrackingAlgorithm
from .tracks import (
    BasicBoxTrack3D,
    XyzFromRazelRrtTrack,
    XyzFromRazelTrack,
    XyzFromXyzTrack,
)


@MODELS.register_module()
class GroundTruthTracker(_TrackingAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ground_truth = True

    def __call__(self, t, frame, detections, ground_truth, **kwargs):
        return ground_truth.objects


@MODELS.register_module()
class BasicBoxTracker3D(_TrackingAlgorithm):
    dimensions = 3

    def __init__(
        self,
        threshold_confirmed=3,
        threshold_coast=3,
        v_max=60,  # meters per second
        assign_metric="center_dist",
        assign_radius=4,
        cost_threshold=0,
        **kwargs,
    ):
        super().__init__(
            assign_metric=assign_metric,
            assign_radius=assign_radius,
            threshold_confirmed=threshold_confirmed,
            threshold_coast=threshold_coast,
            cost_threshold=cost_threshold,
            v_max=v_max,
            **kwargs,
        )

    def spawn_track_from_detection(self, detection):
        return BasicBoxTrack3D(
            self.timestamp,
            box3d=detection.box3d,
            reference=detection.box3d.reference,
            obj_type=detection.obj_type,
        )


@MODELS.register_module()
class BasicBoxTrackerFusion3Stage(_TrackingAlgorithm):
    dimensions = 3

    def __init__(
        self,
        threshold_confirmed_3d=3,
        threshold_confirmed_2d=3,
        threshold_coast_3d=3,
        threshold_coast_2d=3,
        threshold_assoc_2=3,
        threshold_assoc_3=-0.2,
        **kwargs,
    ):
        self.tracks = []
        self.threshold_confirmed_3d = threshold_confirmed_3d
        self.threshold_confirmed_2d = threshold_confirmed_2d
        self.threshold_coast_3d = threshold_coast_3d
        self.threshold_coast_2d = threshold_coast_2d
        self.threshold_assoc_2 = threshold_assoc_2
        self.threshold_assoc_3 = threshold_assoc_3

        self.calib_cam = None
        super().__init__(**kwargs)

    @property
    def tracks_confirmed(self):
        return [
            trk
            for trk in self.tracks
            if (trk.n_updates_3d >= self.threshold_confirmed_3d)
            and (trk.n_updates_2d >= self.threshold_confirmed_2d)
        ]

    def track(self, detections, platform, **kwargs):
        """
        :detections_2d
        :detections_3d

        Use 3-stage association from EagerMOT

        ASSUMPTION: only 1 camera used for now
        """
        t = detections.timestamp
        for trk in self.tracks:
            trk.change_reference(platform, inplace=True)
            trk.predict(t)

        if detections is not None:
            # -- change reference frame
            detections_2d = [
                det.change_reference(platform, inplace=False)
                for det in detections["2d"]
            ]
            detections_3d = [
                det.change_reference(platform, inplace=False)
                for det in detections["3d"]
            ]

            # -- STAGE 1: assignment between detections
            boxes_2d = [det.box2d for det in detections_2d]
            obj_types_2d = [det.obj_type for det in detections_2d]
            boxes_3d = [det.box3d for det in detections_3d]
            obj_types_3d = [det.obj_type for det in detections_3d]

            if len(detections_2d) > 0:
                if self.calib_cam is None:
                    self.calib_cam = detections_2d[0].box.calibration
                boxes_3d_in_2d = [
                    box.project_to_2d_bbox(calib=self.calib_cam) for box in boxes_3d
                ]
                A = build_A_from_iou(boxes_2d, boxes_3d_in_2d)
                assign_sol_1 = gnn_single_frame_assign(A, cost_threshold=-0.10)
                lone_2d = [boxes_2d[i_2d] for i_2d in assign_sol_1.unassigned_rows]
                lone_2d_to_det_map = {
                    i: k for i, k in enumerate(assign_sol_1.unassigned_rows)
                }
                lone_3d = [boxes_3d[i_3d] for i_3d in assign_sol_1.unassigned_cols]
                lone_3d_to_det_map = {
                    i: k for i, k in enumerate(assign_sol_1.unassigned_cols)
                }
                fused_detections = [
                    (boxes_2d[i], boxes_3d[j])
                    for i, j in assign_sol_1.assignment_tuples
                ]
                fused_to_det_map = {
                    i: (k1, k2)
                    for i, (k1, k2) in enumerate(assign_sol_1.assignment_tuples)
                }
            else:
                lone_2d = []
                lone_3d = boxes_3d
                lone_2d_to_det_map = {}
                lone_3d_to_det_map = {i: i for i in range(len(boxes_3d))}
                fused_detections = []
                fused_to_det_map = {}

            # -- STAGE 2: assignment between fused and lone 3d to tracks
            i = 0
            A = np.inf * np.ones(
                (len(fused_detections) + len(lone_3d), len(self.tracks))
            )
            for ds in (fused_detections, lone_3d):
                for d_ in ds:
                    if i < len(fused_detections):
                        d_ = d_[1]  # index into tuple for 3D box
                    for j, t in enumerate(self.tracks):
                        if t.box3d is None:
                            continue
                        Bi = np.array(
                            [
                                t.box3d.t[0],
                                t.box3d.t[1],
                                t.box3d.t[2],
                                t.box3d.h,
                                t.box3d.w,
                                t.box3d.l,
                            ]
                        )
                        Bj = np.array([d_.t[0], d_.t[1], d_.t[2], d_.h, d_.w, d_.l])
                        alpha = 2 - np.cos((t.yaw - d_.yaw) % (np.pi / 2))
                        A[i, j] = np.linalg.norm(Bi - Bj) * alpha
                    i += 1
            assign_sol_2 = greedy_assignment(A, threshold=self.threshold_assoc_2)
            lone_fused = [
                fused_detections[j]
                for j in assign_sol_2.unassigned_rows
                if j < len(fused_detections)
            ]
            lone_fused_to_det_map = {
                i: fused_to_det_map[k]
                for i, k in enumerate(assign_sol_2.unassigned_rows)
                if k < len(fused_detections)
            }
            lone_tracks = [self.tracks[k] for k in assign_sol_2.unassigned_cols]
            lone_track_to_track_map = {
                i: k for i, k in enumerate(assign_sol_2.unassigned_cols)
            }

            # -- STAGE 3: assignment between lone fused and lone 2d to tracks via 2D
            i = 0
            A = np.zeros((len(lone_fused) + len(lone_2d), len(lone_tracks)))
            for ds in (lone_fused, lone_2d):
                for d_ in ds:
                    if i < len(lone_fused):
                        d_ = d_[0]  # index into tuple for 2D box
                    for j, t in enumerate(lone_tracks):
                        if t.box2d is None:
                            t_box = t.box3d.project_to_2d_bbox(self.calib_cam)
                        else:
                            t_box = t.box2d
                        A[i, j] = -d_.IoU(t_box)
                    i += 1
            assign_sol_3 = greedy_assignment(A, threshold=self.threshold_assoc_3)

            # -- update tracks
            # ----- from assignment 2
            for i_det, j_trk in assign_sol_2.assignment_tuples:
                if i_det < len(fused_detections):
                    d_2 = fused_detections[i_det]
                    o2d = obj_types_2d[fused_to_det_map[i_det][0]]
                    o3d = obj_types_3d[fused_to_det_map[i_det][1]]
                else:
                    d_2 = lone_3d[i_det - len(fused_detections)]
                    o3d = obj_types_3d[
                        lone_3d_to_det_map[i_det - len(fused_detections)]
                    ]
                self.tracks[j_trk].update(d_2, o3d, platform)
            # ----- from assignment 3
            for i_det, j_trk in assign_sol_3.assignment_tuples:
                if i_det < len(lone_fused):
                    d_3 = lone_fused[i_det]
                    o2d = obj_types_2d[lone_fused_to_det_map[i_det][0]]
                    o3d = obj_types_3d[lone_fused_to_det_map[i_det][1]]
                else:
                    d_3 = lone_2d[i_det - len(lone_fused)]
                    o2d = obj_types_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
                self.tracks[lone_track_to_track_map[j_trk]].update(d_3, o2d, platform)

            # -- unassigned dets for new tracks
            # ----- unassigned from the 3D to 3D step
            for i_det in assign_sol_2.unassigned_rows:
                if i_det < len(fused_detections):
                    continue
                else:
                    d3d = lone_3d[i_det - len(fused_detections)]
                    o3d = obj_types_3d[
                        lone_3d_to_det_map[i_det - len(fused_detections)]
                    ]
                    self.tracks.append(
                        BasicJointBoxTrack3D(self.timestamp, None, d3d, platform, o3d)
                    )
            # ----- unassigned from the 2D to 2D step
            for i_det in assign_sol_3.unassigned_rows:
                if i_det < len(lone_fused):
                    d2d, d3d = lone_fused[i_det]
                    o2d, o3d = (
                        obj_types_2d[lone_fused_to_det_map[i_det][0]],
                        obj_types_3d[lone_fused_to_det_map[i_det][1]],
                    )
                    self.tracks.append(
                        BasicJointBoxTrack3D(self.timestamp, d2d, d3d, platform, o2d)
                    )
                else:
                    d2d = lone_2d[i_det - len(lone_fused)]
                    o2d = obj_types_2d[lone_2d_to_det_map[i_det - len(lone_fused)]]
                    self.tracks.append(
                        BasicJointBoxTrack3D(self.timestamp, d2d, None, platform, o2d)
                    )

            # -- prune dead tracks
            self.tracks = [
                trk
                for trk in self.tracks
                if (trk.dt_coast_2d < self.threshold_coast_2d)
                and (trk.dt_coast_3d < self.threshold_coast_3d)
            ]

        return self.tracks_confirmed


class _BaseCenterTracker(_TrackingAlgorithm):
    def __init__(
        self,
        threshold_confirmed=3,
        threshold_coast=3,
        v_max=60,  # meters per second
        assign_metric="center_dist",
        assign_radius=8,
        **kwargs,
    ):
        super().__init__(
            assign_metric=assign_metric,
            assign_radius=assign_radius,
            threshold_confirmed=threshold_confirmed,
            threshold_coast=threshold_coast,
            cost_threshold=0,  # bc we are subtracting off assign radius
            v_max=v_max,
            **kwargs,
        )

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError


@MODELS.register_module()
class BasicXyzTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromXyzTrack(
            t0=self.timestamp,
            xyz=detection.xyz,
            reference=detection.reference,
            obj_type=detection.obj_type,
        )


@MODELS.register_module()
class BasicRazelTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromRazelTrack(
            t0=self.timestamp,
            razel=detection.razel,
            reference=detection.reference,
            obj_type=detection.obj_type,
        )


@MODELS.register_module()
class BasicRazelRrtTracker(_BaseCenterTracker):
    dimensions = 3

    def spawn_track_from_detection(self, detection):
        return XyzFromRazelRrtTrack(
            t0=self.timestamp,
            razelrrt=detection.razelrrt,
            reference=detection.reference,
            obj_type=detection.obj_type,
        )


# ==============================================================
# EXTERNALS
# ==============================================================


@MODELS.register_module()
class Ab3dmotTracker(_TrackingAlgorithm):
    dimensions = 3

    def __init__(self, **kwargs):
        self.iframe = 0
        self.tracker = libraries.AB3DMOT.model.AB3DMOT()
        self.n_tracks_total = 0
        self.n_tracks_last = 0
        self._ID_set = set()
        self.reference = None
        super().__init__(**kwargs)

    @property
    def n_tracks(self):
        return self.n_tracks_total

    @property
    def n_tracks_active(self):
        return self.n_tracks_last

    @property
    def tracks(self):
        return self.tracker.trackers

    @tracks.setter
    def tracks(self, tracks):
        self.tracker.trackers = tracks

    def track(self, detections, platform, **kwargs):
        """
        :detections - list of class Detection

        AB3DMOT expects detections in camera coordinates, so do conversion
        """
        t = detections.timestamp
        if detections is None:
            raise NotImplementedError("Need to implement prediction only")

        detections = DataContainer(
            frame=detections.frame,
            timestamp=detections.timestamp,
            data=[det.change_reference(platform, inplace=False) for det in detections],
            source_identifier=detections.source_identifier,
        )
        if (len(detections) > 0) and (self.reference is None):
            self.reference = detections[0].box.reference
            R = tforms.transform_orientation(self.reference.q, "quat", "dcm")
            up_vec = np.round(R[:, 2])
            self.z_up = np.allclose(up_vec, np.array([0, 0, 1]))

        # -- get information on the object
        ori_array = np.asarray([d.box.yaw for d in detections]).reshape((-1, 1))
        score = 1
        other_array = np.asarray(
            [[d.obj_type, None, None, None, None, None] for d in detections]
        )
        if len(other_array.shape) == 1:
            other_array = other_array[:, None]
        additional_info = np.concatenate((ori_array, other_array), axis=1)

        # --- make detections format
        dets = np.asarray(
            [
                [
                    d.box.h,
                    d.box.w,
                    d.box.l,
                    d.box.t[0],
                    d.box.t[1],
                    d.box.t[2],
                    d.box.yaw,
                ]
                for d in detections
            ]
        )
        if len(dets.shape) == 1:
            dets = dets[:, None]
        dets_all = {"dets": dets, "info": additional_info}

        # --- update
        tracks = self.tracker.update(t, dets_all, self.z_up)
        return self._format_tracks(tracks, t)

    def _format_tracks(self, tracks, t):
        # --- put in common format
        tracks_format = []
        for d in tracks:
            ID = d[7]
            pos = Position(d[3:6], self.reference)
            vel = Velocity(d[15:18], self.reference)
            acc = None
            h, w, l, yaw = d[0], d[1], d[2], d[6]
            q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            rot = Attitude(q, self.reference)
            vs = VehicleState(obj_type=d[9], ID=ID)
            vs.set(
                t=t,
                position=pos,
                box=Box3D(pos, rot, [h, w, l]),
                velocity=vel,
                acceleration=acc,
                attitude=rot,
                angular_velocity=None,
            )
            tracks_format.append(vs)
            # n_updates=d[18],
            # score=(d[14] if d[14] is not None else 1.0))
        self._ID_set = self._ID_set.union(set([t.ID for t in tracks_format]))
        self.n_tracks_total = len(self._ID_set)
        self.n_tracks_last = len(tracks_format)
        return tracks_format


@MODELS.register_module()
class EagermotTracker(_TrackingAlgorithm):
    dimensions = 3

    def __init__(self, plus=False, n_box_confirmed=0, n_joint_coast=np.inf, **kwargs):
        self.tracker = libraries.EagerMOT.model.EagerMOT(
            plus, n_box_confirmed, n_joint_coast
        )
        super().__init__(**kwargs)

    def track(self, t, frame, detections, platform, **kwargs):
        if detections is None:
            raise NotImplementedError("Need to implement this")
        # -- change reference frame
        detections_2d = DataContainer(
            frame=detections["2d"].frame,
            timestamp=detections["2d"].timestamp,
            data=[
                det.change_reference(platform, inplace=False)
                for det in detections["2d"]
            ],
            source_identifier=detections["2d"].source_identifier,
        )
        detections_3d = DataContainer(
            frame=detections["3d"].frame,
            timestamp=detections["3d"].timestamp,
            data=[
                det.change_reference(platform, inplace=False)
                for det in detections["3d"]
            ],
            source_identifier=detections["3d"].source_identifier,
        )
        tracks = self.tracker(t, detections_2d, detections_3d, platform)
        return self._format_tracks(tracks, detections_2d, detections_3d)

    @property
    def n_tracks(self):
        return len(self.tracker.tracks)

    @property
    def n_tracks_confirmed(self):
        return len(self.tracker.tracks_confirmed)

    def _format_tracks(self, tracks, detections_2d, detections_3d):
        tracks_format = []
        if isinstance(detections_2d, dict):
            frame = detections_2d[list(detections_2d.keys())[0]].frame
            timestamp = detections_2d[list(detections_2d.keys())[0]].timestamp
        else:
            frame = detections_2d.frame
            timestamp = detections_2d.frame
        for ID, trk in tracks.items():
            x, y, z, yaw, h, w, l, vx, vy, vz = trk.kf.x[:, 0]
            q = tforms.transform_orientation([0, 0, yaw], "euler", "quat")
            reference = trk.box3d.reference
            vs = VehicleState(obj_type=None, ID=ID)
            pos = Position(np.array([x, y, z]), reference)
            rot = Attitude(q, reference)
            vel = Velocity(np.array([vx, vy, vz]), reference)
            vs.set(
                t=timestamp,
                position=pos,
                box=Box3D(pos, rot, [h, w, l]),
                velocity=vel,
                acceleration=None,
                attitude=rot,
                angular_velocity=None,
            )
            tracks_format.append(vs)
        return tracks_format


# class Chaser3DBoxTracker(_TrackingAlgorithm):
#     def __init__(self, **kwargs):
#         self.tracker = chaser.get_tracker(
#             association="gnn", dim=3, model="cv", maneuver="low", use_box=True
#         )
#         super().__init__(**kwargs)

#     def track(self, t, frame, detections, platform, **kwargs):
#         if detections is None:
#             raise NotImplementedError("Need to implement prediction only")
#         detections_3d = [det.change_reference(platform, inplace=False) for det in detections["object_3d"]]
#         msmts = self._convert_dets_to_msmts(detections_3d)
#         self.tracker.process_msmts(msmts)
#         return self._format_tracks(self.tracker.confirmed_tracks, detections_3d)

#     def _convert_dets_to_msmts(self, dets):
#         msmts = []
#         for d in dets:
#             if isinstance(d, BoxDetection):
#                 if isinstance(d.box, Box3D):
#                     r = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.1])
#                     m = estimators.measurements.BoxMeasurement_3D_XYZHWLYaw(
#                         source_ID=d.source_ID,
#                         t=dets.timestamp,
#                         r=r,
#                         x=d.box.t[0],
#                         y=d.box.t[1],
#                         z=d.box.t[2],
#                         h=d.box.h,
#                         w=d.box.w,
#                         l=d.box.l,
#                         yaw=d.box.yaw,
#                     )
#                     msmts.append(m)
#                 elif isinstance(d.box, Box2D):
#                     raise NotImplementedError
#                     r = np.array([5, 5, 5, 5])
#                     m = estimators.measurements.BoxMeasurement_2D_XYXY(
#                         source_ID=d.source_ID,
#                         t=dets.timestamp,
#                         r=r,
#                         x_min=d.box.xmin,
#                         y_min=d.box.ymin,
#                         x_max=d.box.xmax,
#                         y_max=d.box.ymax,
#                     )
#                     msmts.append(m)
#                 else:
#                     raise NotImplementedError(type(d))
#             else:
#                 raise NotImplementedError(type(d))
#         return msmts

#     def _format_tracks(self, tracks, detections_3d):
#         tracks_format = []
#         if isinstance(detections_3d, dict):
#             frame = detections_3d[list(detections_3d.keys())[0]].frame
#             timestamp = detections_3d[list(detections_3d.keys())[0]].timestamp
#         else:
#             frame = detections_3d.frame
#             timestamp = detections_3d.frame
#         for trk in tracks:
#             x, y, z, vx, vy, vz, h, w, l, yaw = trk.filter.x_vector
#             q = None
#             raise
#             vs = VehicleState(obj_type=None, ID=ID)
#             vs.set(
#                 t=timestamp,
#                 position=np.array([x, y, z]),
#                 box=Box3D([h, w, l, np.zeros((3,)), q], StandardCoordinates),
#                 velocity=np.array([vx, vy, vz]),
#                 acceleration=None,
#                 angular_velocity=None,
#             )
#             tracks_format.append(vs)
#         return tracks_format
