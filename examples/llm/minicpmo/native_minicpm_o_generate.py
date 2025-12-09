import argparse
import math
import tempfile
from functools import partial
from types import MethodType

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from xh_model_zoo.utils.image import impad, imrescale


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents


def _get_sliced_images(self, image, max_slice_nums=None, rescale_size=None):
    slice_images = self._old_get_sliced_images(image, max_slice_nums)
    img_max_w, img_max_h = rescale_size
    if rescale_size is not None:
        for i, image in enumerate(slice_images):

            img = np.array(image)
            img, scale_factor = imrescale(
                img,
                (img_max_w, img_max_h),
                interpolation="bilinear",
                return_scale=True,
                backend="cv2",
            )
            pad_img = impad(img, shape=(img_max_h, img_max_w), pad_val=0)
            image = Image.fromarray(pad_img)
            slice_images[i] = image

    return slice_images


def _audio_feature_extract_fixed_length(self, *args, **kwargs):
    audio_features, audio_feature_lens_list, audio_ph_list = self._old_audio_feature_extract(*args, **kwargs)
    # self.feature_extractor.nb_max_frames = 3000
    audio_features = torch.nn.functional.pad(
        audio_features,
        (0, self.feature_extractor.nb_max_frames - audio_features.shape[-1]),
        mode="constant",
        value=0,
    )
    return audio_features, audio_feature_lens_list, audio_ph_list


def main(args):
    # load omni model default, the default init_vision/init_audio/init_tts is True
    # if load vision-only model, please set init_audio=False and init_tts=False
    # if load audio-only model, please set init_vision=False
    model_dir = args.model_dir
    logger.info(f"{model_dir}")
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        attn_implementation="sdpa",  # sdpa or flash_attention_2
        torch_dtype=torch.float16,
        init_vision=True,
        init_audio=True,
        init_tts=True,
    )

    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # In addition to vision-only mode, tts processor and vocos also needs to be initialized
    model.init_tts()

    video_path = args.video
    # if use voice clone prompt, please set ref_audio
    ref_audio_path = args.audio
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
    # or use default prompt
    # sys_msg = model.get_sys_prompt(mode='omni', language='en')

    contents = get_video_chunk_content(video_path)
    msg = {"role": "user", "content": contents}
    msgs = [sys_msg, msg]

    # please set generate_audio=True and output_audio_path to save the tts result
    generate_audio = True
    output_audio_path = "output.wav"

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    image_processor = processor.image_processor
    image_processor._old_get_sliced_images = image_processor.get_sliced_images
    image_processor.get_sliced_images = MethodType(
        partial(_get_sliced_images, rescale_size=[40 * 14, 40 * 14]), image_processor
    )

    processor._old_audio_feature_extract = processor.audio_feature_extract
    processor.audio_feature_extract = MethodType(_audio_feature_extract_fixed_length, processor)
    # The person in the picture is skiing down a snowy slope
    res = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        processor=processor,
        sampling=True,
        temperature=0.5,
        max_new_tokens=4096,
        omni_input=True,  # please set omni_input=True when omni inference
        use_tts_template=True,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        max_slice_nums=1,
        use_image_id=False,
        return_dict=True,
    )
    logger.info(res)

    # processor.image_processor.get_sliced_images = processor.image_processor._old_get_sliced_images
    # del processor.image_processor._old_get_sliced_images

    ## You will get the answer: The person in the picture is skiing down a snowy slope.
    # import IPython
    # IPython.display.Audio('output.wav')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model-dir", type=str, default="weights/MiniCPM-o-2_6")
    parser.add_argument("--video", type=str, default="weights/MiniCPM-o-2_6/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="weights/MiniCPM-o-2_6/assets/demo.wav")
    args = parser.parse_args()
    main(args)
