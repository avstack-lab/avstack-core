import os

import cv2


try:
    import torch
except ModuleNotFoundError:
    print("No torch found.")

try:
    from midas.model_loader import load_model
except ModuleNotFoundError:
    print("MIDAS module not found (this is not an error).")

from avstack import __file__ as avfile
from avstack.config import MODELS
from avstack.modules.perception.base import _PerceptionAlgorithm
from avstack.sensors import DepthImageData


midas_root = os.path.join(
    os.path.dirname(os.path.dirname(avfile)), "third_party", "MiDaS"
)


@MODELS.register_module()
class MidasDepthEstimator(_PerceptionAlgorithm):
    MODE = "monocular_depth"

    def __init__(
        self,
        model="dpt_beit_base_384",
        gpu=0,
        optimize=False,
        height=None,
        side=False,
        square=False,
        *args,
        **kwargs,
    ):
        """Run MonoDepthNN to compute depth maps.

        Args:
            model (str): the name of the model
            optimize (bool): optimize the model to half-floats on CUDA?
            side (bool): RGB and depth side by side in output images?
            height (int): inference encoder image height
            square (bool): resize to a square resolution?
        """
        super().__init__(*args, **kwargs)
        self.model_name = model
        if optimize:
            raise NotImplementedError

        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        )
        print("Device: %s" % self.device)
        model_weights = os.path.join(midas_root, "weights", model + ".pt")
        if not os.path.exists(model_weights):
            raise FileNotFoundError(
                "{} weights not found at {}".format(model, model_weights)
            )

        # load the model
        self.model, self.transform, net_w, net_h = load_model(
            self.device, model_weights, self.model_name, optimize, height, square
        )
        self.input_size = (net_w, net_h)

    def preprocessing(self, image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        return image

    def process(self, image, target_size):
        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction

    def _execute(self, data, identifier, **kwargs):
        # make sure image is RGB
        original_image_rgb = self.preprocessing(data.rgb_image)
        image = self.transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = self.process(image, original_image_rgb.shape[1::-1])

        # store output
        img_out = DepthImageData(
            timestamp=data.timestamp,
            frame=data.frame,
            data=prediction,
            encoding="midas",
            calibration=data.calibration,
            source_ID=-1,
            source_name=identifier,
        )
        return img_out
