import datetime
from typing import List, Union

import numpy as np

from avstack.config import MODELS, ConfigDict
from avstack.datastructs import DataContainer
from avstack.environment.objects import VehicleState
from avstack.geometry import Box2D, Box3D, PassiveReferenceFrame, ReferenceFrame
from avstack.modules.perception.detections import (
    BoxDetection,
    CentroidDetection,
    RazDetection,
    RazelDetection,
    RazelRrtDetection,
)
from avstack.utils.decorators import apply_hooks

from ..assignment import gnn_single_frame_assign
from ..base import BaseModule
from .tracks import BasicBoxTrack3D


class _TrackingAlgorithm(BaseModule):
    def __init__(
        self,
        assign_metric="IoU",
        assign_radius=4,
        threshold_confirmed=3,
        threshold_coast=3,
        cost_threshold=-0.10,
        clusterer: ConfigDict = {"type": "BasicClusterer", "assign_radius": 1.0},
        run_clustering: bool = False,
        v_max=None,
        check_reference=True,
        ID=None,
        t0=0.0,
        P0=None,
        name="tracking",
        **kwargs,
    ):
        """Base class for tracking algorithm

        Cost threshold means any cost higher than this value is rejected
        """
        super().__init__(name=name, **kwargs)

        self.ID = ID
        self.iframe = -1
        self.frame = 0
        self.timestamp = t0.timestamp() if isinstance(t0, datetime.datetime) else t0
        self.tracks = DataContainer(
            frame=self.frame, timestamp=self.timestamp, source_identifier="", data=[]
        )
        self.assign_metric = assign_metric
        self.assign_radius = assign_radius
        if assign_metric == "center_dist":
            assert (
                cost_threshold == 0
            ), "Cost threshold should be 0 to let dist threshold work"
        elif assign_metric == "IoU":
            assert cost_threshold < 0, "Cost threshold must be negative in IoU mode"
        else:
            raise NotImplementedError(assign_metric)
        self.cost_threshold = cost_threshold
        self.last_assignment = None
        self.threshold_confirmed = threshold_confirmed
        self.threshold_coast = threshold_coast
        self.check_reference = check_reference
        self.clusterer = MODELS.build(clusterer)
        self.run_clustering = run_clustering
        self.v_max = v_max
        self.P0 = P0

    @property
    def tracks(self):
        return self._tracks

    @tracks.setter
    def tracks(self, tracks):
        self._tracks = tracks

    @property
    def confirmed_tracks(self):
        return self.tracks_confirmed

    @property
    def tracks_confirmed(self):
        trks_confirmed = [trk for trk in self.tracks if trk.confirmed]
        return DataContainer(self.frame, self.timestamp, trks_confirmed, self.name)

    @property
    def tracks_active(self):
        trks_active = [trk for trk in self.tracks if trk.active]
        return DataContainer(self.frame, self.timestamp, trks_active, self.name)

    @apply_hooks
    def __call__(
        self,
        detections,
        platform: Union[PassiveReferenceFrame, ReferenceFrame],
        **kwargs,
    ):
        if not isinstance(detections, DataContainer):
            raise ValueError(
                f"Detections are {type(detections)}, must be DataContainer"
            )
        self.timestamp = float(detections.timestamp)
        self.frame = int(detections.frame)
        self.iframe += 1
        tracks = self.track(detections, platform, **kwargs)
        return tracks

    def reset(self):
        self.tracks = []
        self.iframe = -1
        self.frame = 0
        self.timestamp = 0
        self.last_assignment = None

    def get_assignment_matrix(self, dets, tracks):
        A = np.zeros((len(dets), len(tracks)))
        for i, det_ in enumerate(dets):
            # -- pull off detection state
            if isinstance(det_, (VehicleState, BoxDetection, BasicBoxTrack3D)):
                det = det_.box
            elif isinstance(det_, (Box3D, Box2D)):
                det = det_
            elif isinstance(det_, RazelRrtDetection):
                det = det_.xyzrrt  # use the cartesian coordinates for gating
            elif isinstance(det_, RazelDetection):
                det = det_.xyz  # use the cartesian coordinates for gating
            elif isinstance(det_, RazDetection):
                det = det_.xy  # use the cartesian coordinates for gating
            elif isinstance(det_, CentroidDetection):
                if self.dimensions == 3:
                    det = det_.xyz
                elif self.dimensions == 2:
                    det = det_.xy
                else:
                    raise NotImplementedError(self.dimensions)
            else:
                raise NotImplementedError(type(det_))

            for j, trk_ in enumerate(tracks):
                # -- pull off track state
                if isinstance(
                    det_, (VehicleState, BasicBoxTrack3D, BoxDetection, Box3D, Box2D)
                ):
                    try:
                        trk = trk_.as_object().box
                    except AttributeError:
                        trk = trk_.box2d
                    else:
                        if trk is None:
                            trk = trk_.box2d
                elif isinstance(det_, RazelRrtDetection):
                    trk = np.array([*trk_.x[:3], trk_.rrt])
                elif isinstance(det_, RazelDetection):
                    trk = trk_.x[:3]
                elif isinstance(det_, RazDetection):
                    trk = trk_.x[:2]
                elif isinstance(det_, CentroidDetection):
                    if self.dimensions == 3:
                        trk = trk_.x[:3]
                    elif self.dimensions == 2:
                        trk = trk_.x[:2]
                    else:
                        raise NotImplementedError(self.dimensions)
                else:
                    raise NotImplementedError(type(det_))

                # -- either way, change origin and use radius to filter coarsely
                try:
                    if self.check_reference:
                        err = False
                        if isinstance(det.reference, ReferenceFrame):
                            if det.reference != trk.reference:
                                err = True
                        elif isinstance(det.reference, PassiveReferenceFrame):
                            if det.reference.frame_id != trk.reference.frame_id:
                                err = True
                        else:
                            raise NotImplementedError(
                                f"type of {type(det.reference)} not understood"
                            )
                        if err:
                            raise RuntimeError(
                                "Should have performed reference transformations earlier..."
                            )
                except AttributeError as e:
                    pass

                # -- gating
                if self.assign_radius is not None:
                    if isinstance(
                        det_, (BasicBoxTrack3D, VehicleState, BoxDetection, Box3D)
                    ):
                        dist = det.t.distance(
                            trk.t, check_reference=self.check_reference
                        )
                    else:
                        dist = np.linalg.norm(trk - det)

                # -- use the metric of choice
                if self.assign_metric == "IoU":
                    cost = -det.IoU(
                        trk, check_reference=self.check_reference
                    )  # lower is better
                elif self.assign_metric == "center_dist":
                    cost = dist - self.assign_radius  # lower is better
                else:
                    raise NotImplementedError(self.assign_metric)

                # -- store result
                A[i, j] = cost
        return A

    def spawn_track_from_detection(self, detection):
        raise NotImplementedError

    def predict_tracks(self, timestamp, platform, check_reference):
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.timestamp()
        tracks_active = self.tracks_active  # pull from the beginning
        for trk in tracks_active:
            if platform and check_reference:
                trk.change_reference(platform, inplace=True)
            trk.predict(timestamp)
            if self.v_max is not None:
                if trk.velocity.norm() > self.v_max:
                    trk.active = False
        return tracks_active

    def track(
        self,
        detections: DataContainer,
        platform: ReferenceFrame,
        change_in_place=False,
        trks_observable: List = None,
        *args,
        **kwargs,
    ):
        """Basic tracking implementation

        Note: detections being None means only do a prediction but don't penalize misses

        if we observe a track but technically it was in unobservable,
        just go ahead and update it anyway....only use unobservable to
        handle penalties for misses
        """

        t = detections.timestamp

        # -- propagation
        tracks_active = self.predict_tracks(
            t, platform=platform, check_reference=self.check_reference
        )
        if not trks_observable:
            trks_observable = tracks_active

        # for trk in tracks_active:
        #     if platform and check_reference:
        #         trk.change_reference(platform, inplace=True)
        #     trk.predict(t)
        #     if self.v_max is not None:
        #         if trk.velocity.norm() > self.v_max:
        #             trk.active = False

        # -- loop over each sensor providing detections
        if detections is not None:
            if not isinstance(detections, dict):
                detections = {"sensor_1": detections}
            for sensor, dets in detections.items():
                # -- change to platform reference
                if platform and self.check_reference:
                    if not change_in_place:
                        dets = [
                            det.change_reference(platform, inplace=False)
                            for det in dets
                        ]
                    else:
                        for det in dets:
                            det.change_reference(platform, inplace=True)

                # -- assignment with active and ALL tracks
                A = self.get_assignment_matrix(dets, tracks_active)
                assign_sol = gnn_single_frame_assign(
                    A, cost_threshold=self.cost_threshold
                )
                self.last_assignment = assign_sol

                # -- update tracks with associations
                for i_det, j_trk in assign_sol.assignment_tuples:
                    tracks_active[j_trk].update(z=dets[i_det].z, R=dets[i_det].R)

                # -- unassigned dets for new tracks
                for i_det in assign_sol.unassigned_rows:
                    self.tracks.append(self.spawn_track_from_detection(dets[i_det]))

                # -- tell unassigned tracks we missed them UNLESS they're whitelisted
                for j_trk in assign_sol.unassigned_cols:
                    if tracks_active[j_trk] in trks_observable:
                        tracks_active[j_trk].missed()

                # -- run clustering on tracks
                if self.run_clustering:
                    clusters = self.clusterer(
                        objects=self.tracks,
                        agent_ID="central",
                        frame=self.frame,
                        timestamp=self.timestamp,
                    )
                    # -- remove "duplicate" tracks based on clustering
                    for clust in clusters:
                        if len(clust) > 1:
                            idx_max_updates = np.argmax(
                                [trk.n_updates for trk in clust.objects]
                            )
                            for i, trk in enumerate(clust.objects):
                                if i != idx_max_updates:
                                    self.tracks.remove(trk)

            # -- prune dead tracks -- only in a non-predict_only state
            self.tracks = [
                trk
                for trk in self.tracks
                if (trk.dt_coast < self.threshold_coast) and trk.active
            ]

        return self.tracks_confirmed
