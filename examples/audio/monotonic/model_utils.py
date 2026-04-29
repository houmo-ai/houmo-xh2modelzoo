import copy
from pathlib import Path

import torch
import torchaudio
from torch import nn


class BiLSTMCompat(nn.Module):
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        if not lstm.bidirectional or lstm.num_layers != 1:
            raise ValueError("BiLSTMCompat only supports 1-layer bidirectional LSTM")

        self.hidden_size = lstm.hidden_size
        self.weight_ih_l0 = nn.Parameter(lstm.weight_ih_l0.detach().clone())
        self.weight_hh_l0 = nn.Parameter(lstm.weight_hh_l0.detach().clone())
        self.bias_ih_l0 = nn.Parameter(lstm.bias_ih_l0.detach().clone())
        self.bias_hh_l0 = nn.Parameter(lstm.bias_hh_l0.detach().clone())

        self.weight_ih_l0_reverse = nn.Parameter(lstm.weight_ih_l0_reverse.detach().clone())
        self.weight_hh_l0_reverse = nn.Parameter(lstm.weight_hh_l0_reverse.detach().clone())
        self.bias_ih_l0_reverse = nn.Parameter(lstm.bias_ih_l0_reverse.detach().clone())
        self.bias_hh_l0_reverse = nn.Parameter(lstm.bias_hh_l0_reverse.detach().clone())
        self.register_buffer("initial_hidden", torch.zeros((1, self.hidden_size), dtype=torch.float32), persistent=False)
        self.register_buffer("initial_cell", torch.zeros((1, self.hidden_size), dtype=torch.float32), persistent=False)
        self.fixed_seq_len: int | None = None

    def _run_direction(
        self,
        inputs: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len, _ = inputs.shape
        if self.fixed_seq_len is not None:
            seq_len = self.fixed_seq_len
        hidden = self.initial_hidden
        cell = self.initial_cell
        outputs = []
        for index in range(seq_len):
            gates = torch.matmul(inputs[:, index, :], weight_ih.transpose(0, 1))
            gates = gates + torch.matmul(hidden, weight_hh.transpose(0, 1)) + bias_ih + bias_hh
            hidden_size = self.hidden_size
            input_gate = gates[:, 0:hidden_size]
            forget_gate = gates[:, hidden_size : 2 * hidden_size]
            cell_gate = gates[:, 2 * hidden_size : 3 * hidden_size]
            output_gate = gates[:, 3 * hidden_size : 4 * hidden_size]
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_gate = torch.tanh(cell_gate)
            output_gate = torch.sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * cell_gate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, inputs: torch.Tensor):
        forward_out = self._run_direction(
            inputs,
            self.weight_ih_l0,
            self.weight_hh_l0,
            self.bias_ih_l0,
            self.bias_hh_l0,
        )
        backward_out = self._run_direction(
            torch.flip(inputs, dims=[1]),
            self.weight_ih_l0_reverse,
            self.weight_hh_l0_reverse,
            self.bias_ih_l0_reverse,
            self.bias_hh_l0_reverse,
        )
        backward_out = torch.flip(backward_out, dims=[1])
        output = torch.cat([forward_out, backward_out], dim=-1)
        return output, (None, None)


class StaticSinusoidalPositionEncoder(nn.Module):
    def __init__(self, max_length: int, input_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        positions = torch.arange(1, max_length + 1, dtype=dtype).unsqueeze(0)
        log_timescale_increment = torch.log(torch.tensor([10000.0], dtype=dtype)) / (input_dim / 2 - 1)
        inv_timescales = torch.exp(torch.arange(input_dim / 2, dtype=dtype) * (-log_timescale_increment))
        inv_timescales = inv_timescales.reshape(1, -1)
        scaled_time = positions.reshape(1, max_length, 1) * inv_timescales.reshape(1, 1, -1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        self.register_buffer("position_encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_encoding


class StaticSANMEncoder(nn.Module):
    def __init__(self, model: nn.Module, max_seq_len: int, input_dim: int):
        super().__init__()

        from funasr.models.sanm.attention import MultiHeadedAttentionSANM
        from funasr.models.sanm.encoder import EncoderLayerSANMExport
        from funasr.models.transformer.embedding import SinusoidalPositionEncoder, StreamSinusoidalPositionEncoder

        self.model = copy.deepcopy(model)
        self.embed = self.model.embed
        if isinstance(self.embed, StreamSinusoidalPositionEncoder):
            self.embed = None
        elif isinstance(self.embed, SinusoidalPositionEncoder):
            self.embed = StaticSinusoidalPositionEncoder(max_seq_len, input_dim)

        if hasattr(self.model, "encoders0"):
            for index, layer in enumerate(self.model.encoders0):
                if isinstance(layer.self_attn, MultiHeadedAttentionSANM):
                    layer.self_attn = StaticMultiHeadedAttentionSANMExport(layer.self_attn)
                self.model.encoders0[index] = EncoderLayerSANMExport(layer)

        for index, layer in enumerate(self.model.encoders):
            if isinstance(layer.self_attn, MultiHeadedAttentionSANM):
                layer.self_attn = StaticMultiHeadedAttentionSANMExport(layer.self_attn)
            self.model.encoders[index] = EncoderLayerSANMExport(layer)

        self.output_scale = float(self.model._output_size) ** 0.5
        mask = torch.ones((1, max_seq_len), dtype=torch.float32)
        self.register_buffer("mask_3d_btd", mask[:, :, None])
        self.register_buffer("mask_4d_bhlt", (1.0 - mask[:, None, None, :]) * -10000.0)

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        mask = (self.mask_3d_btd, self.mask_4d_bhlt)
        xs_pad = speech * self.output_scale
        if self.embed is not None:
            xs_pad = self.embed(xs_pad)

        if hasattr(self.model, "encoders0"):
            xs_pad, _ = self.model.encoders0(xs_pad, mask)
        xs_pad, _ = self.model.encoders(xs_pad, mask)
        xs_pad = self.model.after_norm(xs_pad)
        return xs_pad, speech_lengths

    def get_predictor_mask(self) -> torch.Tensor:
        return self.mask_3d_btd.transpose(1, 2)


class StaticMultiHeadedAttentionSANMExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.d_k = model.d_k
        self.h = model.h
        self.linear_out = model.linear_out
        self.linear_q_k_v = model.linear_q_k_v
        self.fsmn_block = model.fsmn_block
        self.left_padding, self.right_padding = model.pad_fn.padding
        self.all_head_size = self.h * self.d_k
        self.hidden_size = self.linear_out.out_features
        self.register_buffer("left_pad", torch.zeros((1, self.hidden_size, self.left_padding), dtype=torch.float32), persistent=False)
        self.register_buffer("right_pad", torch.zeros((1, self.hidden_size, self.right_padding), dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, mask):
        mask_3d_btd, mask_4d_bhlt = mask
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask_3d_btd)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask_4d_bhlt)
        return att_outs + fsmn_memory

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_qkv(self, x: torch.Tensor):
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        return self.transpose_for_scores(q), self.transpose_for_scores(k), self.transpose_for_scores(v), v

    def forward_fsmn(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = inputs * mask
        x = inputs.transpose(1, 2)
        if self.left_padding > 0:
            x = torch.cat((self.left_pad, x), dim=2)
        if self.right_padding > 0:
            x = torch.cat((x, self.right_pad), dim=2)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x = x + inputs
        x = x * mask
        return x

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        context_layer = torch.matmul(attn, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.linear_out(context_layer)


def load_auto_model(model_dir: Path, model_revision: str):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    monotonic_pipeline = pipeline(
        task=Tasks.speech_timestamp,
        model=str(model_dir),
        model_revision=model_revision,
    )
    auto_model = monotonic_pipeline.model.model
    auto_model.model = auto_model.model.cpu().float().eval()
    predictor = auto_model.model.predictor
    if hasattr(predictor, "blstm") and getattr(predictor.blstm, "bidirectional", False):
        predictor.blstm = BiLSTMCompat(predictor.blstm)
    return auto_model


def build_export_encoder(model: nn.Module, max_seq_len: int, feats_dim: int) -> nn.Module:
    return StaticSANMEncoder(model.encoder, max_seq_len=max_seq_len, input_dim=feats_dim).cpu().float().eval()


def read_text(text: str | None, text_path: Path) -> str:
    if text is not None:
        return text.strip()
    return text_path.read_text(encoding="utf-8").strip()


def extract_inputs(auto_model, audio_path: Path, text: str):
    frontend = auto_model.kwargs["frontend"]
    tokenizer = auto_model.kwargs["tokenizer"]

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != frontend.fs:
        waveform = torchaudio.functional.resample(waveform, sample_rate, frontend.fs)

    speech, speech_lengths = frontend(waveform, torch.tensor([waveform.shape[1]]))
    token_ids = tokenizer.encode(text)
    token_num = torch.tensor([len(token_ids) + 1], dtype=torch.int32)
    token_list = tokenizer.ids2tokens(token_ids)
    return speech.float(), speech_lengths.to(dtype=torch.int32), token_num, token_list