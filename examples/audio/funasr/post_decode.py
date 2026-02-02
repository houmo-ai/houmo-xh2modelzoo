import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Union
import numpy as np

pred_bias = 1

class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[List, str],
    ):

        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]



def _decode_one(am_score: np.ndarray, valid_token_num: int) -> List[str]:
    global pred_bias
    global converter
    yseq = am_score.argmax(axis=-1)
    score = am_score.max(axis=-1)
    score = np.sum(score, axis=-1)

    # pad with mask tokens to ensure compatibility with sos/eos tokens
    # asr_model.sos:1  asr_model.eos:2
    yseq = np.array([1] + yseq.tolist() + [2])
    hyp = Hypothesis(yseq=yseq, score=score)

    # remove sos/eos and get results
    last_pos = -1
    token_int = hyp.yseq[1:last_pos].tolist()

    # remove blank symbol id, which is assumed to be 0
    token_int = list(filter(lambda x: x not in (0, 2), token_int))

    # Change integer-ids to tokens
    token = converter.ids2tokens(token_int)
    token = token[: valid_token_num - pred_bias]
    # texts = sentence_postprocess(token)
    return token


def post_decode( am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [
            _decode_one(am_score, token_num)
            for am_score, token_num in zip(am_scores, token_nums)
        ]


token_list_path = Path(__file__).with_name("tokens.json")
if not token_list_path.exists():
    raise FileNotFoundError(f"tokens.json not found: {token_list_path}")
with token_list_path.open("r", encoding="utf-8") as f:
    token_list = json.load(f)
converter = TokenIDConverter(token_list)
