import json
from pathlib import Path
from typing import Optional

from xhquant.api import HMONNXInference


from ....utils import DeviceDtypeMixin


class SD3Inference(DeviceDtypeMixin):
    def __init__(self, model_config_file: Path, fast_mode: bool = False):
        super().__init__()
        self.fast_mode = fast_mode
        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info

        self.mmdit_hmonnx_file = str(model_dir / meta_info["mmdit_hmonnx"])
        self.clip_l_hmonnx_file = str(model_dir / meta_info["clip_l_hmonnx"])
        self.clip_hmonnx_file = str(model_dir / meta_info["clip_hmonnx"])
        self.vae_hmonnx_file = str(model_dir / meta_info["vae_hmonnx"])
        self.t5_hmonnx_file = str(model_dir / meta_info["t5_hmonnx"])

        self.mmdit_session: Optional[HMONNXInference] = None
        self.clip_l_session: Optional[HMONNXInference] = None
        self.clip_session: Optional[HMONNXInference] = None
        self.vae_session: Optional[HMONNXInference] = None
        self.t5_session: Optional[HMONNXInference] = None

        self.init_mmdit()
        self.init_clip_l()
        self.init_clip()
        self.init_vae()
        self.init_t5()

    def init_clip(self):
        if self.clip_session is not None:
            return
        self.clip_session = HMONNXInference(self.clip_hmonnx_file)
        if self.fast_mode:
            self.clip_session.to_fast_mode()
        self.clip_session.exec_device = self._exec_device
        self.clip_session.to(self.device)

    @property
    def width(self):
        return self.meta_info["width"]

    @property
    def height(self):
        return self.meta_info["height"]

    @property
    def guidance_scale(self):
        return self.meta_info["guidance_scale"]

    def init_mmdit(self):
        if self.mmdit_session is not None:
            return
        self.mmdit_session = HMONNXInference(self.mmdit_hmonnx_file)
        if self.fast_mode:
            self.mmdit_session.to_fast_mode()
        self.mmdit_session.exec_device = self._exec_device
        self.mmdit_session.to(self.device)

    def init_clip_l(self):
        if self.clip_l_session is not None:
            return
        self.clip_l_session = HMONNXInference(self.clip_l_hmonnx_file)
        if self.fast_mode:
            self.clip_l_session.to_fast_mode()
        self.clip_l_session.exec_device = self._exec_device
        self.clip_l_session.to(self.device)

    def init_clip(self):
        if self.clip_session is not None:
            return
        self.clip_session = HMONNXInference(self.clip_hmonnx_file)
        if self.fast_mode:
            self.clip_session.to_fast_mode()
        self.clip_session.exec_device = self._exec_device
        self.clip_session.to(self.device)

    def init_vae(self):
        if self.vae_session is not None:
            return
        self.vae_session = HMONNXInference(self.vae_hmonnx_file)
        if self.fast_mode:
            self.vae_session.to_fast_mode()
        self.vae_session.exec_device = self._exec_device
        self.vae_session.to(self.device)

    def init_t5(self):
        if self.t5_session is not None:
            return
        self.t5_session = HMONNXInference(self.t5_hmonnx_file)
        if self.fast_mode:
            self.t5_session.to_fast_mode()
        self.t5_session.exec_device = self._exec_device
        self.t5_session.to(self.device)
