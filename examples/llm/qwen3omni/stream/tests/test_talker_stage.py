# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for HMONNXTalkerStage."""

import torch

from talker_stage import HMONNXTalkerStage
from events import TalkerInputDecode, TalkerInputPrefill, TalkerStepOutput


class TestHMONNXTalkerStage:
    def _make_stage(self, thinker_hidden_size=64, num_code_groups=4, num_lm_heads=8):
        """Create a TalkerStage with fake sessions."""
        from conftest import FakeHMONNXSession

        talker_kv = {"shape": [1, 4, 64, 128], "num_decoder_layers": 4}
        predictor_kv = {"shape": [1, 2, 32, 128], "num_decoder_layers": 2}

        return HMONNXTalkerStage(
            talker_prefill_session=FakeHMONNXSession(),
            talker_decode_session=FakeHMONNXSession(),
            predictor_prefill_session=FakeHMONNXSession(),
            predictor_decode_session=FakeHMONNXSession(),
            talker_kv_cache_info=talker_kv,
            predictor_kv_cache_info=predictor_kv,
            static_prefill_len=16,
            static_predictor_prefill_len=4,
            num_code_groups=num_code_groups,
            num_lm_heads=num_lm_heads,
            thinker_hidden_size=thinker_hidden_size,
            talker_hidden_size=thinker_hidden_size,
        )

    def test_prefill_emits_code(self):
        stage = self._make_stage()
        prefill = TalkerInputPrefill(
            hidden_state=torch.randn(1, 10, 64),
            role_mask=torch.ones(1, 10, 1),
            bypass_embeds=torch.randn(1, 10, 64),
            bypass_mask=torch.ones(1, 10, 1),
        )
        outputs = list(stage.run(iter([prefill])))
        # Should emit full residual-code frames.
        assert len(outputs) >= 1
        assert all(isinstance(o, TalkerStepOutput) for o in outputs)
        assert all(o.residual_codes.shape[-1] == 4 for o in outputs)

    def test_prefill_then_decode(self):
        stage = self._make_stage(num_code_groups=2)
        prefill = TalkerInputPrefill(
            hidden_state=torch.randn(1, 8, 64),
            role_mask=torch.ones(1, 8, 1),
            bypass_embeds=torch.randn(1, 8, 64),
            bypass_mask=torch.ones(1, 8, 1),
        )
        decode = TalkerInputDecode(
            codec_embedding=torch.randn(1, 1, 64),
            generation_step=0,
        )
        finished = TalkerInputDecode(
            codec_embedding=torch.empty(0),
            is_finished=True,
        )
        outputs = list(stage.run(iter([prefill, decode, finished])))
        # prefill and decode each generate one full residual-code frame, then finished signal
        assert len(outputs) >= 3
        assert outputs[-1].is_finished is True

    def test_decode_without_prefill(self):
        stage = self._make_stage(num_code_groups=2)
        decode = TalkerInputDecode(
            codec_embedding=torch.randn(1, 1, 64),
            generation_step=0,
        )
        finished = TalkerInputDecode(
            codec_embedding=torch.empty(0),
            is_finished=True,
        )
        outputs = list(stage.run(iter([decode, finished])))
        # Should handle decode-only gracefully (initializes cache)
        assert len(outputs) >= 1

    def test_reset_clears_state(self):
        stage = self._make_stage()
        prefill = TalkerInputPrefill(
            hidden_state=torch.randn(1, 5, 64),
            role_mask=torch.ones(1, 5, 1),
            bypass_embeds=torch.randn(1, 5, 64),
            bypass_mask=torch.ones(1, 5, 1),
        )
        list(stage.run(iter([prefill])))
        assert stage._generation_step >= 0

        stage.reset()
        assert stage._generation_step == -1
        assert stage._step == 0

    def test_multiple_decode_steps(self):
        stage = self._make_stage(num_code_groups=2)
        prefill = TalkerInputPrefill(
            hidden_state=torch.randn(1, 6, 64),
            role_mask=torch.ones(1, 6, 1),
            bypass_embeds=torch.randn(1, 6, 64),
            bypass_mask=torch.ones(1, 6, 1),
        )
        decodes = [
            TalkerInputDecode(codec_embedding=torch.randn(1, 1, 64), generation_step=i)
            for i in range(3)
        ]
        finished = TalkerInputDecode(codec_embedding=torch.empty(0), is_finished=True)

        inputs = [prefill] + decodes + [finished]
        outputs = list(stage.run(iter(inputs)))
        # prefill: 1 frame, 3 decodes: 3 frames, + 1 finished
        assert len(outputs) >= 5
        assert outputs[-1].is_finished is True
