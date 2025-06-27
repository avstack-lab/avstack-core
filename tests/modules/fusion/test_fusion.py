from copy import deepcopy

import numpy as np

from avstack.geometry import Attitude, GlobalOrigin3D, Position, bbox
from avstack.modules import clustering, fusion, tracking


def get_n_tracks(n_tracks=2):
    t0 = 0.0
    obj_type = "car"
    pos = Position(np.array([-1, -2, -3]), GlobalOrigin3D)
    rot = Attitude(np.quaternion(1), GlobalOrigin3D)
    box1 = bbox.Box3D(pos, rot, [2, 2, 4])
    tracks = [
        tracking.tracker3d.BasicBoxTrack3D(t0, deepcopy(box1), GlobalOrigin3D, obj_type)
        for _ in range(n_tracks)
    ]
    return tracks


def test_no_fusion():
    fuser = fusion.track_to_track.NoFusion()
    clusterer = clustering.NoClustering()
    tracks = get_n_tracks(n_tracks=2)
    clusters = clusterer({0: tracks}, frame=0, timestamp=0)
    tracks_fused = [fuser(cluster) for cluster in clusters]
    assert [tracks_fused[0][0], tracks_fused[1][0]] == tracks


def test_ci_fusion_base_naive_bayes():
    n_dim = 9
    n_agents = 5
    x = np.random.randn(n_dim)
    P = 20 * np.eye(9) + np.random.rand(n_dim, n_dim)
    xs = [x for _ in range(n_agents)]
    Ps = [P for _ in range(n_agents)]
    x_f, P_f = fusion.track_to_track.ci_fusion(xs, Ps, weights="uniform")
    assert np.allclose(x_f, x)
    assert np.allclose(P_f, P)


def test_fusion_into_BoxTrack3D():
    tracks = get_n_tracks(n_tracks=4)
    fuser = fusion.CovarianceIntersectionFusionToBox()
    fused_out = fuser(tracks)
    assert isinstance(fused_out, tracking.tracker3d.BasicBoxTrack3D)
