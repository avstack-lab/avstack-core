import os
import tempfile

import numpy as np
from cv2 import imwrite

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.modules.perception import utils
from avstack.modules.perception.base import _MMObjectDetector, _PerceptionAlgorithm


# ===========================================================================
# TRUTH OPERATIONS
# ===========================================================================


@MODELS.register_module()
class GroundTruth2DFvObjectDetector(_PerceptionAlgorithm):
    MODE = "object_2d"

    def __call__(self, ground_truth, *args, **kwargs):
        raise NotImplementedError


# ===========================================================================
# CUSTOM OPERATIONS
# ===========================================================================


@MODELS.register_module()
class InfrastructureCameraTo3d(_PerceptionAlgorithm):
    """
    This algorithm is suitable for 2D infrastructure sensors that have some
    non-zero pitch angle. Under the assumption of a flat ground plane and
    precise knowledge of the calibration matrix for the sensor, this algorithm
    computes the full 3D positioning information assuming the object is on
    the ground plane. It then uses and optimization procedure to fit a 3D
    bounding box that matches the 2D detection from the camera

    The procedure is as follows:
    1. Get 2D mask detections from a camera algorithm
    2. From the centroid of the mask, get azimuth and elevation angles
    3. Approximating the ground as flat, use sensor calibration to predict range
    4. Fit 3D bounding box corners that are consistent with this 2D box
    5. Build a box object from the corners
    """

    MODE = "object_3d"

    def __init__(self, detector, algorithm="range_from_ground", **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.algorithm = algorithm

    def _execute(self, data, identifier, *args, **kwargs):
        # -- get detections from mmdet
        detections = self.detector(data, identifier, *args, **kwargs)

        # -- use approximation of a flat ground plane to get 3D positioning
        # TODO: add possibility of roll angle?
        # TODO: account for RefChoc
        z = data.reference.x[2]
        pitch = data.reference.euler[1]  # pitch down is positive

        # -- from the detections, compute the azimuth and elevation angles
        for det in detections:
            az, el = det.box.angles
            view_angle = pitch + el  # TODO: check this out...
            rng = z / np.sin(view_angle)

        # -- fit a 3D bounding box that is consistent with the 2D projection
        raise

        # -- from the 3D bounding box corners, build the box object
        raise


# ===========================================================================
# MM DETECTION OPERATIONS
# ===========================================================================


@MODELS.register_module()
class MMDetObjectDetector2D(_MMObjectDetector):
    MODE = "object_2d"

    def __init__(
        self,
        model="fasterrcnn",
        dataset="kitti",
        deploy=False,
        threshold=None,
        gpu=0,
        epoch="latest",
        deploy_runtime="tensorrt",
        **kwargs,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            deploy=deploy,
            deploy_runtime=deploy_runtime,
            threshold=threshold,
            gpu=gpu,
            epoch=epoch,
            **kwargs,
        )

    def _execute(self, data, identifier, eval_method="data", **kwargs):
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=True)

        # -- flip data to bgr
        img = data.bgr_image

        # -- inference
        result_ = self.run_mm_inference(img, eval_method)

        # -- postprocess objects
        detections = utils.convert_mm2d_to_avstack(
            result_,
            data.calibration,
            identifier,
            self.label_dataset_override,
            self.threshold,
            self.whitelist,
            self.class_names,
            self.deploy,
        )
        return DataContainer(data.frame, data.timestamp, detections, identifier)

    def run_mm_inference(self, image, eval_method):
        if self.deploy:
            return self.run_mm_from_deploy(self.model, image)
        else:
            return self.run_mm_from_checkpoint(
                self.inference_detector, self.model, image, eval_method
            )

    @staticmethod
    def run_mm_from_deploy(model, image):
        return model(image)

    @staticmethod
    def run_mm_from_checkpoint(inference_detector, model, image, eval_method):
        if eval_method == "file":
            with tempfile.TemporaryDirectory() as temp_dir:
                fd_data, data_file = tempfile.mkstemp(suffix=".png", dir=temp_dir)
                os.close(fd_data)  # need to start with the file closed...
                imwrite(data_file, image)
                result_ = inference_detector(model, data_file)
        elif eval_method == "data":
            result_ = inference_detector(model, image)
        else:
            raise NotImplementedError(eval_method)
        return result_

    @staticmethod
    def parse_mm_model_from_deploy(model, dataset):
        raise NotImplementedError

    @staticmethod
    def parse_mm_model_from_checkpoint(model, dataset, epoch):
        input_data = "camera"
        label_dataset_override = dataset
        epoch_str = "latest" if epoch == "latest" else "epoch_{}".format(epoch)
        do_projection = False
        if model == "yolov3":
            raise NotImplementedError("yolo not trained yet")
        elif model == "rtmdet":
            if dataset == "coco":
                threshold = 0.5
                config_file = "configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
                checkpoint_file = "checkpoints/coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
            elif dataset == "rccars-oneclass":
                threshold = 0.5
                config_file = "configs/rccars/rtmdet_m_8xb32-300e_rccars-oneclass.py"
                checkpoint_file = (
                    f"work_dirs/rtmdet_m_8xb32-300e_rccars-oneclass/{epoch_str}.pth"
                )
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model in ["fasterrcnn", "faster_rcnn"]:
            if dataset == "kitti":
                threshold = 0.5
                config_file = "configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py"
                checkpoint_file = "checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
                label_dataset_override = "cityscapes"
            elif dataset == "nuscenes":
                threshold = 0.7
                config_file = "configs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.py"
                # config_file = "work_dirs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.py"
                checkpoint_file = (
                    "work_dirs/nuscenes/faster_rcnn_r50_fpn_1x_nuscenes.pth"
                )
            elif dataset == "carla-vehicle":
                threshold = 0.7
                config_file = "work_dirs/carla/faster_rcnn_r50_fpn_1x_carla_vehicle.py"
                checkpoint_file = (
                    f"work_dirs/carla/faster_rcnn_r50_fpn_1x_carla_vehicle.pth"
                )
            elif dataset == "carla-infrastructure":
                threshold = 0.7
                config_file = "work_dirs/faster_rcnn_r50_fpn_1x_carla_infrastructure.py"
                checkpoint_file = (
                    f"work_dirs/faster_rcnn_r50_fpn_1x_carla_infrastructure.pth"
                )
            elif dataset == "carla-joint":
                threshold = 0.7
                config_file = "work_dirs/faster_rcnn_r50_fpn_1x_carla_joint/faster_rcnn_r50_fpn_1x_carla_joint.py"
                checkpoint_file = (
                    f"work_dirs/faster_rcnn_r50_fpn_1x_carla_joint/{epoch_str}.pth"
                )
            elif dataset == "cityscapes":
                threshold = 0.5
                config_file = "configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes.py"
                checkpoint_file = "checkpoints/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
            elif dataset == "coco-person":
                threshold = 0.25
                config_file = (
                    "configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py"
                )
                checkpoint_file = "checkpoints/coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model in ["cascadercnn", "cascade_rcnn"]:
            if dataset == "carla-vehicle":
                threshold = 0.5
                config_file = "work_dirs/carla/cascade-rcnn_r50_fpn_1x_carla_vehicle.py"
                checkpoint_file = (
                    f"work_dirs/carla/cascade-rcnn_r50_fpn_1x_carla_vehicle.pth"
                )
            elif dataset == "carla-infrastructure":
                threshold = 0.5
                config_file = (
                    "work_dirs/carla/cascade-rcnn_r50_fpn_1x_carla_infrastructure.py"
                )
                checkpoint_file = (
                    f"work_dirs/carla/cascade-rcnn_r50_fpn_1x_carla_infrastructure.pth"
                )
            elif dataset == "coco":
                threshold = 0.5
                config_file = "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py"
                checkpoint_file = "checkpoints/coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        elif model == "htc":
            if dataset == "nuimages":
                threshold = 0.7
                config_file = "configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e-1xb16_nuim.py"
                checkpoint_file = "checkpoints/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth"
            else:
                raise NotImplementedError(f"{model}, {dataset} not compatible yet")
        else:
            raise NotImplementedError(model)
        return (
            threshold,
            config_file,
            checkpoint_file,
            input_data,
            label_dataset_override,
            do_projection,
        )
