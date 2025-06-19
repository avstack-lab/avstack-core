import numpy as np

from avstack.geometry import Attitude, GlobalOrigin3D, Position, q_cam_to_stan
from avstack.geometry.bbox import Box2D, Box3D, SegMask2D
from avstack.geometry.transformations import transform_orientation

from .detections import BoxDetection, MaskDetection


car_classes = ["car", "Car", "vehicle"]
ped_classes = ["pedestrian", "walker", "person", "Pedestrian", "rider"]
bic_classes = ["bicycle", "cyclist", "Cyclist", "cycler"]
ignore_classes = ["traffic_cone", "barrier", "trailer", "train"]

k_classes = [
    ("Car", car_classes + ["bus", "truck"]),
    ("Pedestrian", ped_classes),
    ("Cyclist", bic_classes),
    ("ignore", ignore_classes + ["motorcycle"]),
]

nu_classes = [
    ("car", car_classes),
    ("truck", ["truck"]),
    ("construction_vehicle", ["construction_vehicle"]),
    ("van", ["van"]),
    ("bicycle", bic_classes),
    ("motorcycle", ["motorcycle"]),
    ("bus", ["bus"]),
    ("person", ped_classes),
    ("ignore", ignore_classes),
]

ci_classes = [
    ("person", ped_classes),
    ("car", car_classes),
    ("truck", ["truck"]),
    ("bus", ["bus"]),
    ("train", ["train"]),
    ("motorcycle", ["motorcycle"]),
    ("bicycle", ["bicycle"]),
]

carla_clases = [
    ("car", car_classes),
    ("truck", ["truck", "van", "bus"]),
    ("bicycle", bic_classes),
    ("motorcycle", ["motorcycle"], ("ignore", ignore_classes)),
]

rccars_oneclass_classes = [("car", car_classes)]
coco_person_classes = [("person", ped_classes)]
coco_classes = [("person", ped_classes), ("car", car_classes), ("bicycle", bic_classes)]

class_maps = {
    "kitti": {k: ks[0] for ks in k_classes for k in ks[1]},
    "cityscapes": {k: ks[0] for ks in ci_classes for k in ks[1]},
    "nuscenes": {k: ks[0] for ks in nu_classes for k in ks[1]},
    "nuimages": {k: ks[0] for ks in nu_classes for k in ks[1]},
    "carla-joint": {k: ks[0] for ks in carla_clases for k in ks[1]},
    "carla-vehicle": {k: ks[0] for ks in carla_clases for k in ks[1]},
    "carla-infrastructure": {k: ks[0] for ks in carla_clases for k in ks[1]},
    "rccars-oneclass": {k: ks[0] for ks in rccars_oneclass_classes for k in ks[1]},
    "coco-person": {k: ks[0] for ks in coco_person_classes for k in ks[1]},
    "coco": {k: ks[0] for ks in coco_classes for k in ks[1]},
}


def convert_mm2d_to_avstack(
    result_,
    calib,
    source_identifier,
    dataset,
    score_thresh,
    whitelist,
    class_names,
    is_deploy,
):
    if is_deploy:
        bboxes, labels, _ = result_
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        segms = None
    else:
        try:
            segms = result_.pred_instances.masks.cpu().numpy()  # segms are N x H x W
        except AttributeError:
            segms = None
        bboxes = result_.pred_instances.bboxes.cpu().numpy()
        labels = result_.pred_instances.labels.cpu().numpy().astype(int)
        scores = result_.pred_instances.scores.cpu().numpy().astype(float)

    # -- filter by score
    if score_thresh > 0:
        assert bboxes is not None and bboxes.shape[1] == 4
        scores_pre = scores.copy()
        inds = scores > score_thresh
        if segms is not None:
            segms = segms[inds, :, :]
        scores = scores[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    if segms is None:
        segms = [None] * len(labels)

    # -- object types
    obj_type_text = []
    try:
        for label in labels:
            if (class_names is not None) and (
                class_names[label] in class_maps[dataset]
            ):
                obj_type_text.append(class_maps[dataset][class_names[label]])
            else:
                obj_type_text.append(class_names[label])
    except IndexError as e:
        print(label, len(class_names), dataset)
        raise e

    # -- make objects
    dets = []
    for segm, bbox, obj_type, score in zip(segms, bboxes, obj_type_text, scores):
        if obj_type in whitelist:
            box_2d = Box2D(bbox, calib)
            if segm is None:
                det = BoxDetection(
                    data=box_2d,
                    noise=np.array([5, 5, 5, 5]) ** 2,
                    source_identifier=source_identifier,
                    reference=calib.reference,
                    obj_type=obj_type,
                    score=score,
                )
            else:
                det = MaskDetection(
                    box=box_2d,
                    mask=SegMask2D(segm, calib),
                    source_identifier=source_identifier,
                    reference=calib.reference,
                    obj_type=obj_type,
                    score=score,
                )
            dets.append(det)

    return dets


def convert_mm3d_to_avstack(
    result_,
    calib,
    model_3d,
    obj_map,
    whitelist,
    input_data,
    source_identifier,
    dataset,
    thresh=0.5,
    dist_min=2.0,
    do_projection=False,
    front_only=False,
    prune_low=False,
    thresh_low=-3,
    prune_duplicate=False,
    thresh_duplicate=0.5,
    verbose=False,
    nominal_height=1.8,
    **kwargs,
):
    dets = []
    # -- parse object information
    if "lidar" in input_data.lower():
        bboxes = result_.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        labels = result_.pred_instances_3d.labels_3d.cpu().numpy()
        scores = result_.pred_instances_3d.scores_3d.cpu().numpy()
    elif ("cam" in input_data.lower()) or ("image" in input_data.lower()):
        bboxes = result_.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        labels = result_.pred_instances_3d.labels_3d.cpu().numpy()
        scores = result_.pred_instances_3d.scores_3d.cpu().numpy()
    else:
        raise NotImplementedError(input_data)

    # -- convert boxes
    prev_locs = []
    if do_projection:
        gp_reference = calib.reference.get_ground_projected_reference()
    for box, label, score in zip(bboxes, labels, scores):
        obj_type = class_maps[dataset][obj_map[label]]
        if obj_type in whitelist:
            if score > thresh:
                # get info from detections
                cent = box[:3]
                if np.linalg.norm(cent) < dist_min:
                    continue
                if (
                    ("pointpillars" in model_3d.cfg.filename)
                    or ("3dssd" in model_3d.cfg.filename)
                    or ("ssn" in model_3d.cfg.filename)
                ):
                    h, w, l = box[5].item(), box[4].item(), box[3].item()
                    if dataset == "kitti":
                        yaw = box[6]
                        q_S_2_obj = transform_orientation([0, 0, yaw], "euler", "quat")
                        q_O_2_obj = q_S_2_obj  # sensor is our reference
                        x_O_2_obj_in_O = cent
                        if "ssn" in model_3d.cfg.filename:
                            x_O_2_obj_in_O[2] += h  # whoops
                            where_is_t = "center"
                        else:
                            where_is_t = "bottom"
                        reference = calib.reference
                    elif dataset == "nuscenes":
                        yaw = box[6]
                        if "pointpillars" in model_3d.cfg.filename:
                            q_O1_2_obj = transform_orientation(
                                [0, 0, -yaw], "euler", "quat"
                            )
                            q_O_2_O1 = transform_orientation(
                                [0, 0, -np.pi / 2], "euler", "quat"
                            )
                            q_O_2_obj = q_O1_2_obj * q_O_2_O1
                        else:
                            q_O1_2_obj = transform_orientation(
                                [0, 0, yaw], "euler", "quat"
                            )
                            q_O_2_obj = q_O1_2_obj
                        x_O_2_obj_in_O = cent
                        where_is_t = "bottom"
                        reference = calib.reference
                    elif "carla" in dataset:
                        if do_projection:
                            reference = gp_reference
                        else:
                            reference = calib.reference
                        yaw = box[6]
                        dx = nominal_height - reference.x[2]
                        q_O_2_obj = transform_orientation([0, 0, yaw], "euler", "quat")
                        x_O_2_obj_in_O = cent
                        x_O_2_obj_in_O[2] += dx
                        where_is_t = "bottom"  # box is bottom-centered
                    else:
                        raise NotImplementedError(dataset)
                elif "pgd" in model_3d.cfg.filename:
                    w, h, l = box[5].item(), box[4].item(), box[3].item()
                    if dataset == "kitti":
                        yaw = box[6] - np.pi / 2
                        q_L_2_obj = transform_orientation([0, 0, -yaw], "euler", "quat")
                        q_C_2_obj = q_L_2_obj * q_cam_to_stan
                        x_C_2_obj_in_C = cent
                        where_is_t = "bottom"
                        reference = calib.reference  # camera reference
                    elif dataset == "nuscenes":
                        yaw = box[6] - np.pi / 2
                        q_L_2_obj = transform_orientation([0, 0, -yaw], "euler", "quat")
                        q_C_2_obj = q_L_2_obj * q_cam_to_stan
                        x_C_2_obj_in_C = cent
                        where_is_t = "bottom"
                        reference = calib.reference
                    else:
                        raise NotImplementedError(dataset)
                    x_O_2_obj_in_O = x_C_2_obj_in_C
                    q_O_2_obj = q_C_2_obj
                else:
                    raise NotImplementedError(model_3d.cfg.filename)

                # make box output
                pos = Position(x_O_2_obj_in_O, reference)
                rot = Attitude(q_O_2_obj, reference)
                box3d = Box3D(pos, rot, [h, w, l], where_is_t=where_is_t)

                # invert the projection if necessary
                if do_projection:
                    box3d.change_reference(calib.reference, inplace=True)

                # -- pruning
                # ---- too low
                if prune_low and (
                    box3d.t.change_reference(GlobalOrigin3D, inplace=False)[2]
                    < thresh_low
                ):
                    if verbose:
                        print("Box registered uncharacteristically low...skipping")
                    continue
                # ---- too close to another object
                if prune_duplicate:
                    distances = np.array(
                        [
                            np.linalg.norm(x_O_2_obj_in_O - obj_loc)
                            for obj_loc in prev_locs
                        ]
                    )
                    if any(distances <= thresh_duplicate):
                        if verbose:
                            print("Object was too close to another")
                        continue

                # ---- we made it!
                prev_locs.append(x_O_2_obj_in_O)
                dets.append(
                    BoxDetection(
                        data=box3d,
                        noise=np.array([1, 1, 1, 0.25, 0.25, 0.25]) ** 2,
                        source_identifier=source_identifier,
                        reference=box3d.reference,
                        obj_type=obj_type,
                        score=score,
                    )
                )

    return dets
