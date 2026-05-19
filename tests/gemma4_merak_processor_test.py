from pathlib import Path
import wave

import numpy as np
from PIL import Image
import torch


MODEL_DIR = Path("/data01/datasets/gemma-4-E4B-it")
if not MODEL_DIR.exists():
    MODEL_DIR = Path("/data01/datasets/gemma-4-E4B")


def _load_wav_mono_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        pcm_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / float(1 << 15)
    else:
        raise ValueError(f"Unsupported test WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio)


def _compute_audio_soft_token_count(num_feature_frames: int) -> int:
    tokens = num_feature_frames
    for _ in range(2):
        tokens = (tokens + 2 - 3) // 2 + 1
    return tokens


def test_gemma4_processor_expands_image_placeholders():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (448, 448), color="white")},
                {"type": "text", "text": "Describe the image briefly."},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(messages)
    token_text = processor.tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=False)

    assert "<|image>" in token_text
    assert token_text.count("<|image|>") > 1
    assert "pixel_values" in model_inputs
    assert "image_position_ids" in model_inputs
    assert "mm_token_type_ids" in model_inputs
    assert model_inputs["image_position_ids"].shape[-1] == 2


def test_gemma4_processor_full_mode_keeps_rectangular_image_tokens():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    model_inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (320, 224), color="white")},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ]
    )

    image_token_count = int((model_inputs["input_ids"] == processor.tokenizer.image_token_id).sum().item())

    assert image_token_count == 280


def test_gemma4_processor_full_export_mode_forces_fixed_square_contract():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import configure_gemma4_visual_processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    configure_gemma4_visual_processor(
        processor,
        export_mode="full",
        max_size_h=448,
        max_size_w=448,
        patch_size=16,
        image_seq_length=280,
    )
    model_inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (320, 224), color="white")},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ]
    )

    image_token_count = int((model_inputs["input_ids"] == processor.tokenizer.image_token_id).sum().item())
    real_patch_count = int((~(model_inputs["image_position_ids"] == -1).all(dim=-1)).sum().item())

    assert image_token_count == 280
    assert model_inputs["pixel_values"].shape[1] == 2520
    assert model_inputs["image_position_ids"].shape[1] == 2520
    assert real_patch_count == 2304


def test_gemma4_processor_compact_mode_forces_square_image_contract():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import configure_gemma4_visual_processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    configure_gemma4_visual_processor(
        processor,
        export_mode="compact",
        max_size_h=448,
        max_size_w=448,
        patch_size=16,
        image_seq_length=256,
    )
    model_inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (320, 224), color="white")},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ]
    )

    image_token_count = int((model_inputs["input_ids"] == processor.tokenizer.image_token_id).sum().item())
    real_patch_count = int((~(model_inputs["image_position_ids"] == -1).all(dim=-1)).sum().item())

    assert image_token_count == 256
    assert model_inputs["pixel_values"].shape[1] == 256
    assert model_inputs["image_position_ids"].shape[1] == 256
    assert real_patch_count == 256


def test_gemma4_processor_uses_official_turn_tokens_without_double_bos():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))

    model_inputs = processor.apply_chat_template([{"role": "user", "content": "Say hello."}])
    input_ids = model_inputs["input_ids"][0].tolist()

    assert input_ids.count(processor.tokenizer.bos_token_id) == 1
    assert input_ids[0] == processor.tokenizer.bos_token_id
    assert input_ids[1] == processor.tokenizer.convert_tokens_to_ids("<|turn>")

    token_text = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
    assert token_text.startswith("<bos><|turn>user\nSay hello.<turn|>\n<|turn>model\n")
    assert "<start_of_turn>" not in token_text
    assert "<end_of_turn>" not in token_text


def test_gemma4_processor_renders_image_blocks_like_official_template():
    from transformers import AutoProcessor

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    official_processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (448, 448), color="white")},
                {"type": "text", "text": "Describe the image briefly."},
            ],
        }
    ]

    rendered, _, _, _ = processor._render_messages(messages, add_generation_prompt=True)
    official_rendered = official_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    assert rendered == official_rendered


def test_gemma4_processor_matches_official_audio_template_tokenization():
    from transformers import AutoProcessor

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    official_processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    audio = np.zeros(16000, dtype=np.float32)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio, "sampling_rate": 16000},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(messages)
    official_inputs = official_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    assert torch.equal(model_inputs["input_ids"], official_inputs["input_ids"])
    assert torch.equal(model_inputs["attention_mask"], official_inputs["attention_mask"])
    assert torch.equal(model_inputs["mm_token_type_ids"], official_inputs["mm_token_type_ids"])


def test_gemma4_processor_expands_audio_placeholders():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    processor.config.audio_feature_length = 400
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(messages)
    token_text = processor.tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=False)

    assert "<|audio>" in token_text
    assert token_text.count("<|audio|>") > 1
    assert "input_features" in model_inputs
    assert "input_features_mask" in model_inputs
    assert "mm_token_type_ids" in model_inputs
    assert model_inputs["input_features"].shape == (1, 400, 128)
    assert model_inputs["input_features_mask"].shape == (1, 400)
    assert int(model_inputs["input_features_mask"].sum().item()) == 99


def test_gemma4_processor_resizes_audio_placeholder_count_after_feature_truncation():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_processor import XHGemma4Processor

    processor = XHGemma4Processor.from_pretrained(str(MODEL_DIR))
    processor.config.audio_feature_length = 400
    audio = _load_wav_mono_float32(Path("examples/llm/minicpmo/assets/mimick.wav"))

    model_inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio, "sampling_rate": 16000},
                    {"type": "text", "text": "Please transcribe the audio."},
                ],
            }
        ]
    )

    expected_audio_token_count = _compute_audio_soft_token_count(int(model_inputs["input_features_mask"].sum().item()))
    actual_audio_token_count = int((model_inputs["input_ids"] == processor.tokenizer.audio_token_id).sum().item())

    assert model_inputs["input_features"].shape == (1, 400, 128)
    assert expected_audio_token_count == 100
    assert actual_audio_token_count == expected_audio_token_count
    assert model_inputs["mm_token_type_ids"].shape == model_inputs["input_ids"].shape


def test_gemma4_moe_visual_processor_contract_without_upsample():
    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import XHGemma4MoeVisualProcessor

    processor = XHGemma4MoeVisualProcessor(pooling_kernel_size=1, image_seq_length=256)
    model_inputs = processor(images=Image.new("RGB", (448, 448), color="white"), return_tensors="pt")
    valid_mask = ~(model_inputs["image_position_ids"] == -1).all(dim=-1)

    assert model_inputs["pixel_values"].shape == (1, 280, 768)
    assert int(valid_mask.sum().item()) == 256
    assert model_inputs["num_soft_tokens_per_image"] == [256]


def test_gemma4_moe_visual_processor_contract_with_upsample():
    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_visual_model import XHGemma4MoeVisualProcessor

    processor = XHGemma4MoeVisualProcessor(pooling_kernel_size=3, image_seq_length=280)
    model_inputs = processor(images=Image.new("RGB", (448, 448), color="white"), return_tensors="pt")
    valid_mask = ~(model_inputs["image_position_ids"] == -1).all(dim=-1)

    assert model_inputs["pixel_values"].shape == (1, 2520, 768)
    assert int(valid_mask.sum().item()) == 2304
    assert model_inputs["num_soft_tokens_per_image"] == [256]
