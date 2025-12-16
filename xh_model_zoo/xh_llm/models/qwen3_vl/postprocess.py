from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import torch

class VLLMPresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    等价于 vLLM 的 presence_penalty 实现：
    logits -= presence_penalty * 1_{token 出现在已生成 token 中}

    注意：
    - 只对“生成部分”的 token 生效，不包含 prompt 部分，
      所以需要在初始化时告诉它 prompt 的长度。
    - 兼容 batch / beam，因为 Transformers 在每一步传进来的
      input_ids 形状都是 [batch*num_beams, cur_len]。
    """

    def __init__(self, presence_penalty: float, prompt_length: int):
        if not isinstance(presence_penalty, (int, float)):
            raise TypeError("presence_penalty must be a float.")
        self.presence_penalty = float(presence_penalty)
        self.prompt_length = int(prompt_length)

    def __call__(
        self,
        input_ids: torch.LongTensor,  # [batch*beam, cur_len]
        scores: torch.FloatTensor,  # [batch*beam, vocab_size]
    ) -> torch.FloatTensor:

        # 零就别折腾了，直接返回
        if self.presence_penalty == 0.0:
            return scores

        batch_size, cur_len = input_ids.shape
        _, vocab_size = scores.shape
        device = scores.device

        # 还没开始真正生成（只有 prompt），不施加惩罚
        if cur_len <= self.prompt_length:
            return scores

        # 只统计“已生成部分”的 token（不包含 prompt）
        gen_tokens = input_ids[:, self.prompt_length :]  # [B, gen_len]

        # 统计每个序列中各个 token 的出现次数（与 vLLM 的 get_token_bin_counts_and_mask 一致）
        # bin_counts 形状：[B, vocab_size+1]，+1 是预留给 padding 的占位 ID
        bin_counts = torch.zeros(
            (batch_size, vocab_size + 1),
            dtype=torch.long,
            device=device,
        )
        # 注意：gen_tokens 的值必须 < vocab_size，否则会越界；
        # 正常 HF 模型的 token_id 范围是 [0, vocab_size-1]，不会用到 vocab_size 这个占位。
        ones = torch.ones_like(gen_tokens, dtype=torch.long, device=device)
        bin_counts.scatter_add_(dim=1, index=gen_tokens, src=ones)

        # 去掉占位列，只保留真实 vocab 范围
        bin_counts = bin_counts[:, :vocab_size]  # [B, V]
        # 是否出现过（>0）
        mask = bin_counts > 0  # [B, V], bool

        # logits -= presence_penalty * mask
        scores = scores - self.presence_penalty * mask.to(scores.dtype)
        return scores
