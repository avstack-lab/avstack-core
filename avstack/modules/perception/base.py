from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer


import itertools
import os

from avstack import __file__ as avfile
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule


class _PerceptionAlgorithm(BaseModule):
    next_id = itertools.count()

    def __init__(self, name="perception", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.ID = next(self.next_id)
        self.iframe = -1

    @apply_hooks
    def __call__(self, data, frame=-1, *args, **kwargs) -> "DataContainer":
        self.iframe += 1
        if data is None:
            return None
        else:
            detections = self._execute(
                data, frame=frame, identifier=self.name, *args, **kwargs
            )
            return detections


mmdep_model_root = os.path.join(
    os.path.dirname(os.path.dirname(avfile)),
    "deployment",
    "mmdeploy",
    "mmdeploy_models",
)
mm2d_root = os.path.join(
    os.path.dirname(os.path.dirname(avfile)), "third_party", "mmdetection"
)
mm3d_root = os.path.join(
    os.path.dirname(os.path.dirname(avfile)), "third_party", "mmdetection3d"
)
mmseg_root = os.path.join(
    os.path.dirname(os.path.dirname(avfile)), "third_party", "mmsegmentation"
)


class _MMBase(_PerceptionAlgorithm):
    @staticmethod
    def map_checkpoint_to_latest(mm_root, checkpoint_file):
        if os.path.exists(os.path.dirname(os.path.join(mm_root, checkpoint_file))):
            with open(
                os.path.join(
                    os.path.dirname(os.path.join(mm_root, checkpoint_file)),
                    "last_checkpoint",
                ),
                "r",
            ) as f:
                chk_path = f.readlines()[0].rstrip()
            return chk_path
        else:
            return ""

    def parse_mm_model(self):
        raise NotImplementedError("Implement this in the subclass.")


class _MMSegmenter(_MMBase):
    def __init__(
        self,
        model: str,
        dataset: str,
        gpu: int = 0,
        iteration: Union[str, int] = "latest",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        config_file, checkpoint_file = self.parse_mm_model_from_checkpoint(
            model, dataset, iteration
        )
        self.model = self.load_model_from_checkpoint(
            config_file=config_file,
            checkpoint_file=checkpoint_file,
            gpu=gpu,
        )

    @staticmethod
    def parse_mm_model_from_checkpoint(model, dataset, iteration):
        raise NotImplementedError

    def load_model_from_checkpoint(self, config_file, checkpoint_file, gpu):
        from mmseg.apis import init_model

        # load the model
        if "last_checkpoint" in checkpoint_file:
            chk_path = self.map_checkpoint_to_latest(mmseg_root, checkpoint_file)
        else:
            chk_path = os.path.join(mmseg_root, checkpoint_file)
        cfg_path = os.path.join(mmseg_root, config_file)
        model = init_model(cfg_path, chk_path)
        return model


class _MMObjectDetector(_MMBase):
    def __init__(
        self,
        model,
        dataset,
        deploy,
        gpu=0,
        epoch="latest",
        threshold=None,
        deploy_runtime="tensorrt",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset = dataset.lower()
        self.model_name = model

        # Initialize model
        (
            self.threshold,
            config_file,
            checkpoint_file,
            self.input_data,
            label_dataset_override,
            self._do_projection,
        ) = self.parse_mm_model_from_checkpoint(model, dataset, epoch)
        self.class_names = self.parse_mm_object_classes(label_dataset_override)[0]
        if threshold is not None:
            print(f"Overriding default threshold of {self.threshold} with {threshold}")
            self.threshold = threshold

        # Get label mapping
        all_objs, _ = self.parse_mm_object_classes(label_dataset_override)
        self.obj_map = {i: n for i, n in enumerate(all_objs)}
        _, self.whitelist = self.parse_mm_object_classes(label_dataset_override)
        self.label_dataset_override = label_dataset_override

        # Load the model/paths
        self.deploy = deploy
        if self.deploy:
            self.model = self.load_model_from_deploy(
                model, dataset, deploy_runtime, gpu
            )
        else:
            self.model = self.load_model_from_checkpoint(
                config_file, checkpoint_file, gpu
            )

    @staticmethod
    def parse_mm_model_from_checkpoint(model, dataset, epoch):
        raise NotImplementedError

    def load_model_from_deploy(self, model, dataset, deploy_runtime, gpu):
        from mmdeploy_runtime import Detector

        model_path = os.path.join(
            mmdep_model_root, f"{model}_{dataset}_{deploy_runtime}"
        )
        if os.path.exists(model_path):
            model = Detector(
                model_path=model_path,
                device_name="cuda",
                device_id=gpu,
            )
        else:
            raise FileNotFoundError(f"Cannot find deploy model at {model_path}")
        return model

    def load_model_from_checkpoint(self, config_file, checkpoint_file, gpu):
        # Find model and checkpoint paths
        mod_path = os.path.join(mm2d_root, config_file)

        # HACK: map 'latest' to the checkpoint
        if not os.path.exists(mod_path):
            mod_path = os.path.join(mm3d_root, config_file)
            if "latest" in checkpoint_file:
                chk_path = self.map_checkpoint_to_latest(mm3d_root, checkpoint_file)
            else:
                chk_path = os.path.join(mm3d_root, checkpoint_file)
            if not os.path.exists(mod_path):
                raise FileNotFoundError(f"Cannot find {config_file} config")
            if not os.path.exists(chk_path):
                raise FileNotFoundError(f"Cannot find {chk_path} checkpoint")
        else:
            if "latest" in checkpoint_file:
                chk_path = self.map_checkpoint_to_latest(mm2d_root, checkpoint_file)
            else:
                chk_path = os.path.join(mm2d_root, checkpoint_file)
        if not os.path.exists(chk_path):
            raise FileNotFoundError(
                f"Cannot find {checkpoint_file} checkpoint\n(tried {chk_path})\nmm3d root: {mm3d_root}\nmm2d root: {mm2d_root}"
            )

        # set up inference model settings
        if self.MODE == "object_3d":
            if self.model_name == "3dssd":
                assert gpu == 0, "For some reason, 3dssd must be on gpu 0"
            from mmdet3d.utils import register_all_modules

            if self.input_data == "camera":
                from mmdet3d.apis import inference_mono_3d_detector, init_model

                self.inference_detector = inference_mono_3d_detector
                self.inference_mode = "from_mono"
            elif self.input_data == "lidar":
                from mmdet3d.apis import inference_detector, init_model

                self.inference_detector = inference_detector
                self.inference_mode = "from_lidar"
            else:
                raise NotImplementedError(self.input_data)
            register_all_modules(init_default_scope=True)
            model = init_model(mod_path, chk_path, device=f"cuda:{gpu}")
        elif self.MODE in ["object_2d", "instance_segmentation"]:
            from mmdet.apis import inference_detector, init_detector
            from mmdet.utils import register_all_modules

            self.inference_detector = inference_detector
            register_all_modules(init_default_scope=True)
            model = init_detector(mod_path, chk_path, device=f"cuda:{gpu}")
        else:
            raise NotImplementedError(self.MODE)
        return model

    @staticmethod
    def parse_mm_object_classes(dataset):
        if dataset == "kitti":
            all_objs = ["Car", "Pedestrian", "Cyclist"]
            whitelist = all_objs
        elif dataset == "nuscenes":
            all_objs = [
                "barrier",
                "traffic_cone",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "car",
                "bus",
                "construction_vehicle",
                "trailer",
                "truck",
            ]
            whitelist = all_objs
        elif dataset == "nuimages":
            all_objs = (
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
                "traffic_cone",
                "barrier",
            )
            whitelist = (
                "car",
                "truck",
                "trailer",
                "bus",
                "construction_vehicle",
                "bicycle",
                "motorcycle",
                "pedestrian",
            )
        elif dataset in [
            "carla-vehicle",
            "carla-joint",
            "carla-infrastructure",
        ]:
            all_objs = ["car", "bicycle", "truck", "motorcycle"]
            whitelist = all_objs
        elif dataset == "rccars-oneclass":
            all_objs = ["car"]
            whitelist = all_objs
        elif dataset == "cityscapes":
            all_objs = [
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            ]
            whitelist = all_objs
        elif dataset == "coco-person":
            all_objs = ["person"]
            whitelist = all_objs
        elif dataset == "coco":
            all_objs = [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
            whitelist = all_objs  # ["person", "bicycle", "car"]
        else:
            raise NotImplementedError(dataset)
        return all_objs, whitelist
