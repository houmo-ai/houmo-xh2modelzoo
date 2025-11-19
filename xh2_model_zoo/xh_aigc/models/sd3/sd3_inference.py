import json
from pathlib import Path
from typing import Optional

import torch
from xhquant.api import HMONNXGoldenInference

from ....utils import DeviceDtypeMixin


class SD3Inference(DeviceDtypeMixin):
    def __init__(
        self, model_config_file: Path, fast_mode: bool = False, save_golden: bool = False, device: str = "cuda"
    ):
        super().__init__(device)
        self.fast_mode = fast_mode
        model_dir = Path(model_config_file).parent
        meta_info = json.load(open(model_config_file, "r"))
        self.meta_info = meta_info
        self.save_golden = save_golden

        self.mmdit_hmonnx_file = str(model_dir / meta_info["mmdit_hmonnx"])
        self.clip_l_hmonnx_file = str(model_dir / meta_info["clip_l_hmonnx"])
        self.clip_hmonnx_file = str(model_dir / meta_info["clip_hmonnx"])
        self.vae_hmonnx_file = str(model_dir / meta_info["vae_hmonnx"])
        self.t5_hmonnx_file = str(model_dir / meta_info["t5_hmonnx"])
        self.vae_encoder_hmonnx_file = str(model_dir / meta_info["vae_encoder_hmonnx"])

        self.mmdit_session: Optional[HMONNXGoldenInference] = None
        self.clip_l_session: Optional[HMONNXGoldenInference] = None
        self.clip_session: Optional[HMONNXGoldenInference] = None
        self.vae_session: Optional[HMONNXGoldenInference] = None
        self.t5_session: Optional[HMONNXGoldenInference] = None
        self.vae_encoder_session: Optional[HMONNXGoldenInference] = None

        self.init_mmdit()
        self.init_clip_l()
        self.init_clip()
        self.init_vae()
        self.init_t5()
        self.init_vae_encoder()

        # self.device = self.execution_device = device

    def init_clip(self):
        if self.clip_session is not None:
            return
        self.clip_session = HMONNXGoldenInference(self.clip_hmonnx_file)
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
        self.mmdit_session = HMONNXGoldenInference(self.mmdit_hmonnx_file)
        self.mmdit_session.initialize()
        self.mmdit_session._session.eval()
        if self.fast_mode:
            self.mmdit_session.to_fast_mode()
        self.mmdit_session.exec_device = self._exec_device
        self.mmdit_session.to(self.device)
        self.mmdit_session.save_golden = self.save_golden
        self.mmdit_session.golden_dir = str(Path("work_dirs_develop") / "mmdit_golden")

    def init_clip_l(self):
        if self.clip_l_session is not None:
            return
        self.clip_l_session = HMONNXGoldenInference(self.clip_l_hmonnx_file)
        self.clip_l_session.initialize()
        self.clip_l_session._session.eval()
        if self.fast_mode:
            self.clip_l_session.to_fast_mode()
        self.clip_l_session.exec_device = self._exec_device
        self.clip_l_session.to(self.device)
        self.clip_l_session.save_golden = self.save_golden
        self.clip_l_session.golden_dir = str(Path("work_dirs_develop") / "clip_l_golden")

    def init_clip(self):
        if self.clip_session is not None:
            return
        self.clip_session = HMONNXGoldenInference(self.clip_hmonnx_file)
        self.clip_session.initialize()
        self.clip_session._session.eval()
        if self.fast_mode:
            self.clip_session.to_fast_mode()
        self.clip_session.exec_device = self._exec_device
        self.clip_session.to(self.device)
        self.clip_session.save_golden = self.save_golden
        self.clip_session.golden_dir = str(Path("work_dirs_develop") / "clip_golden")

    def init_vae(self):
        if self.vae_session is not None:
            return
        self.vae_session = HMONNXGoldenInference(self.vae_hmonnx_file)
        self.vae_session.initialize()
        self.vae_session._session.eval()
        if self.fast_mode:
            self.vae_session.to_fast_mode()
        self.vae_session.exec_device = self._exec_device
        self.vae_session.to(self.device)
        self.vae_session.save_golden = self.save_golden
        self.vae_session.golden_dir = str(Path("work_dirs_develop") / "vae_golden")

    def init_vae_encoder(self):
        if self.vae_encoder_session is not None:
            return
        self.vae_encoder_session = HMONNXGoldenInference(self.vae_encoder_hmonnx_file)
        self.vae_encoder_session.initialize()
        self.vae_encoder_session._session.eval()
        if self.fast_mode:
            self.vae_encoder_session.to_fast_mode()
        self.vae_encoder_session.exec_device = self._exec_device
        self.vae_encoder_session.to(self.device)
        self.vae_encoder_session.save_golden = self.save_golden
        self.vae_encoder_session.golden_dir = str(Path("work_dirs_develop") / "vae_encoder_golden")

    def init_t5(self):
        if self.t5_session is not None:
            return
        self.t5_session = HMONNXGoldenInference(self.t5_hmonnx_file)
        self.t5_session.initialize()
        self.t5_session._session.eval()
        if self.fast_mode:
            self.t5_session.to_fast_mode()
        self.t5_session.exec_device = self._exec_device
        self.t5_session.to(self.device)
        self.t5_session.save_golden = self.save_golden
        self.t5_session.golden_dir = str(Path("work_dirs_develop") / "t5_golden")
