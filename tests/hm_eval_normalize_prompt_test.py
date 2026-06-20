from hm_eval.core.normalize import get_system_message, rewrite_messages_for_dataset


def test_get_system_message_preserves_exact_underscore_dataset_names():
    assert get_system_message("mmlu_pro") is not None
    assert "Answer: A" in get_system_message("mmlu_pro")


def test_ceval_rewrite_adds_answer_only_suffix_once():
    messages = [{"role": "user", "content": "问题：1+1=？\n选项：\nA. 1\nB. 2"}]
    once = rewrite_messages_for_dataset(messages, "ceval")
    twice = rewrite_messages_for_dataset(once, "ceval")

    suffix = "请不要解释，不要列步骤，只输出一行：答案：A/B/C/D。"
    assert once[0]["content"].endswith(suffix)
    assert twice[0]["content"].count(suffix) == 1
