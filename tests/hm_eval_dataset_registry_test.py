from hm_eval.core.dataset_registry import DatasetRegistry


def test_ceval_zero_shot_is_explicitly_passed_to_evalscope():
    args = DatasetRegistry().build_dataset_args("ceval")
    assert args == {"ceval": {"few_shot_num": 0}}


def test_positive_default_few_shot_is_preserved():
    args = DatasetRegistry().build_dataset_args("mmlu")
    assert args == {"mmlu": {"few_shot_num": 5}}


def test_call_site_can_override_few_shot_num():
    args = DatasetRegistry().build_dataset_args("ceval", few_shot_num=3)
    assert args == {"ceval": {"few_shot_num": 3}}
