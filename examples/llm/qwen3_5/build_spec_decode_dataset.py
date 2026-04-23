"""Generate 256-case evaluation JSONL for Qwen3.5 speculative-decoding bench.

Categories (approximately balanced):
  humanities, social, tech, math, tool_calls, coding

Each prompt's character length is placed on a roughly uniform distribution
across 100 to 1000 characters. Short seeds are extended with a neutral
"context" suffix so the final prompt hits its target length bin without
changing the underlying question.

Usage:
    python build_spec_decode_dataset.py --output spec_decode_eval_prompts.jsonl --n 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List


# --- Category seed prompts (kept <= 200 chars each; will be padded to hit bin) ---

HUMANITIES = [
    "请用通俗易懂的语言解释《史记》在中国史学中的地位，并举一个你最欣赏的人物传记作为例子。",
    "比较莎士比亚《哈姆雷特》与曹禺《雷雨》中父子冲突的异同。",
    "What are the main ideas of Kant's categorical imperative? Give a concrete everyday example.",
    "从诗经到唐诗，简述中国古典诗歌形式上的演变脉络。",
    "Summarise the role of the Silk Road in cultural exchange between East and West.",
    "请评价鲁迅《狂人日记》的文学价值与思想意义。",
    "Describe three differences between Greek and Roman mythology.",
    "用通俗语言解释什么是文艺复兴，它对欧洲近代科学产生了哪些影响。",
    "Discuss the impact of the printing press on European religious history.",
    "介绍苏东坡的生平及其在宋代文化史上的地位。",
    "用三段话说明佛教从印度传入中国的过程及其中国化特点。",
    "Explain the significance of Virginia Woolf's 'A Room of One's Own'.",
    "What are the main themes in Tolstoy's 'War and Peace'?",
    "简述王阳明心学的核心主张，并对比朱熹理学。",
    "介绍敦煌莫高窟的历史与艺术价值，列举两个你认为最重要的洞窟。",
    "How did the Enlightenment reshape Western political thought? Mention Voltaire, Rousseau and Locke.",
    "请解释京剧脸谱颜色所代表的性格含义，并举例说明。",
    "Discuss the symbolism of light and darkness in Milton's Paradise Lost.",
    "中国书法有哪些主要字体？请对比楷、行、草三种的风格差异。",
    "Explain the difference between stoicism and epicureanism as schools of Greek philosophy.",
]

SOCIAL = [
    "分析社交媒体对当代青年人际关系的双面影响，并给出若干可行建议。",
    "What are the main arguments for and against universal basic income?",
    "请讨论人口老龄化对中国城市公共服务体系提出了哪些挑战。",
    "Explain how inflation affects household savings and investment decisions.",
    "请比较计划经济与市场经济在资源配置效率上的典型差异。",
    "How does gerrymandering affect democratic representation in the United States?",
    "从社会学角度分析『内卷』一词流行背后反映的结构性压力。",
    "Discuss how GDP as a measure of well-being has been criticised since the 1970s.",
    "请说明教育公平与机会公平的关系，并结合高考制度谈一谈。",
    "What sociological theories explain urban-rural migration in developing economies?",
    "请用案例说明政府财政政策与货币政策的区别和配合。",
    "Discuss how household debt levels influence macroeconomic stability.",
    "从经济学角度分析房地产价格对居民消费的影响。",
    "Explain the concept of soft power and give two contemporary examples.",
    "请评价共享经济模式（如网约车、共享单车）对传统行业的冲击。",
    "What are the main causes and consequences of income inequality in OECD countries?",
    "如何理解『碳中和』政策对中国制造业转型的长远影响？",
    "Discuss how public pensions are funded in the US, UK and China, and highlight one challenge each system faces.",
    "分析国际贸易中的比较优势理论及其对发展中国家的启示。",
    "Explain how central banks use interest rates to manage inflation and employment.",
]

TECH = [
    "请通俗解释什么是大语言模型，以及它与传统自然语言处理方法的主要区别。",
    "Describe how HTTPS protects data in transit. Include the role of certificates.",
    "比较关系型数据库和文档型数据库的使用场景，各举一个例子。",
    "Explain how a CPU pipeline works and why branch prediction is important.",
    "请介绍 Transformer 架构中自注意力机制的基本原理。",
    "What is Kubernetes, and what problem does it solve compared to running containers directly?",
    "请解释什么是零信任安全架构，并与传统边界安全做比较。",
    "How does a distributed consensus algorithm like Raft ensure fault tolerance?",
    "请介绍 RISC-V 与 ARM 架构的区别，并讨论 RISC-V 的开源优势。",
    "Explain how TLS 1.3 handshakes differ from TLS 1.2 and why it is more secure.",
    "请通俗解释什么是 CDN，以及它如何降低延迟。",
    "Describe the CAP theorem and explain why you cannot have all three guarantees simultaneously.",
    "请解释操作系统的虚拟内存机制，并简述其对进程隔离的作用。",
    "What is a vector database, and how is it used together with large language models?",
    "请介绍 Rust 的所有权模型及其相较 C++ 的安全优势。",
    "Explain how GPUs parallelise matrix multiplication and why they dominate deep learning workloads.",
    "请说明 RAG（检索增强生成）框架的基本组成与典型适用场景。",
    "What is an end-to-end encrypted messenger, and how does Signal's protocol guarantee forward secrecy?",
    "请通俗解释什么是区块链的共识机制，并对比 PoW 与 PoS。",
    "Describe the main differences between ONNX and TorchScript as model deployment formats.",
]

MATH = [
    "证明：对任意正整数 n，1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2。",
    "Solve the integral ∫ x * e^x dx and show each step clearly.",
    "求方程 x^3 - 6x^2 + 11x - 6 = 0 的所有实数解，并说明因式分解过程。",
    "Prove that the sum of the first n odd positive integers equals n^2.",
    "已知 sin(x) + cos(x) = 1/2，x ∈ [0, 2π)，求 sin(2x) 的所有可能取值。",
    "Show that the square root of 2 is irrational using a proof by contradiction.",
    "求函数 f(x) = x^2 * ln(x) 在 x>0 上的最小值，并说明所用方法。",
    "Compute the determinant of the 3x3 matrix [[1,2,3],[0,1,4],[5,6,0]] and describe the method.",
    "利用数学归纳法证明：对任意整数 n ≥ 1， 2^n > n 。",
    "Find the eigenvalues and eigenvectors of the matrix [[4,1],[2,3]].",
    "一个袋子里有 5 个红球、3 个蓝球、2 个白球，任取 3 个，求恰好取到 2 红 1 蓝的概率。",
    "Use Taylor expansion to approximate ln(1.1) to four decimal places and estimate the error.",
    "证明：若 a、b、c 为正实数且 a+b+c=1，则 (1-a)(1-b)(1-c) ≥ 8abc。",
    "Solve the differential equation dy/dx = y/x with initial condition y(1) = 2.",
    "求极限 lim_{x→0} (sin(x) - x) / x^3 ，并说明使用的展开。",
    "Prove that for any triangle, the medians intersect at a single point (the centroid).",
    "在极坐标下求曲线 r = 1 + cos(θ) 所围成的图形面积。",
    "Given a random variable X ~ N(0,1), compute P(|X| > 1.96) using the standard normal table.",
    "求方程组 {x + 2y - z = 3; 2x - y + 3z = 7; -x + y + 2z = 4} 的解，并说明消元过程。",
    "Show that e^(iπ) + 1 = 0 follows from the Taylor series of exp, sin and cos.",
]

TOOL_CALLS = [
    "你是一个出行助手，可以调用 get_weather(city)、book_train(from, to, date) 两个工具。用户说：'帮我查下明天北京到上海的天气，并订一张上午的高铁。' 请给出调用顺序和参数。",
    "You are a shopping agent with tools search_product(query), add_to_cart(id, qty), checkout(). The user says 'Buy me two USB-C chargers under $30.' Show the tool call plan.",
    "你可以调用 currency_convert(amount, from, to) 和 news_search(topic, date) 。用户说：'明天我要去东京出差，给我换 500 美元到日元，并搜一下东京今天的重要新闻。'",
    "Given tools calendar_list(date), calendar_create(title, start, end), email_send(to, subject, body). User: 'Find my next free 1-hour slot on Friday and invite alice@example.com to lunch.' Output a JSON plan.",
    "你是数据库助手，可用工具 run_sql(query), describe_table(name)。用户说：'表 orders 最近 7 天的总销售额是多少？先看看表结构再写 SQL。'",
    "You are an ops assistant with tools kube_get(resource, ns), kube_restart(pod, ns), log_tail(pod, ns, lines). User: 'My pod api-server-7 in ns prod keeps crashing, help.' Write the exact tool calls to start investigating.",
    "给你两个工具：geocode(address) 和 route(from, to, mode)。用户说：'帮我规划从北京西站到颐和园的公交路线。'请写出调用。",
    "You have tools file_read(path), file_write(path, content), shell_run(cmd). User: 'In repo /tmp/proj, append the line 'echo done' to scripts/run.sh and show the new content.' Give step-by-step tool calls.",
    "可用工具 weibo_search(query), translate(text, to_lang)。用户说：'看下今天微博上关于 AI 芯片的热点，并把前三条翻成英文。'请给出调用计划。",
    "You are a customer-service bot with tools get_order(id), refund(id, amount), notify(user, msg). User: 'My order 10342 arrived damaged, please refund $29.99 and let me know.' Respond with the tool sequence.",
    "你是一个财报分析助手，可用工具 fetch_10k(ticker, year), compute_ratio(name, a, b)。用户说：'分析 AAPL 2023 的净利率和毛利率，并比 2022 变化。'",
    "You have tools flight_search(from, to, date), hotel_search(city, checkin, checkout), weather(city, date). User: 'Plan a 3-day trip from SFO to Tokyo starting next Monday.' Output a JSON plan with tool calls.",
    "可用工具 github_list_pr(repo, state), github_merge_pr(repo, id)。用户说：'合并 github.com/acme/app 上 open 且通过 CI 的所有 PR。'写出执行步骤。",
    "Tools: lark_create_doc(title, content), lark_share(doc_id, user). User: 'Draft a meeting note titled \"2026 Q2 规划\" with sections Agenda/Decisions/Actions, and share with ceo@acme.com.' Emit the tool calls.",
    "你是个代码评审助手，工具有 git_diff(path), review_comment(path, line, msg)。用户说：'请 review 当前变更，对 src/utils.py 第 42 行空指针风险评论。'",
    "You have tools send_sms(phone, msg), address_lookup(user_id). User: 'Tell contact 12345 that their package is delayed one day.' Write the ordered tool calls.",
    "可用工具 search_papers(query, year), cite(paper_id, format)。用户说：'帮我找 2023 年 NeurIPS 上关于 Speculative Decoding 的三篇论文，并生成 BibTeX。'",
    "Tools: s3_list(bucket, prefix), s3_get(bucket, key), zip_files(paths, output). User: 'Download all JSON files under reports/2025-04 in bucket acme-data and zip them to /tmp/april.zip.' Give the plan.",
    "你是个运维助手，工具 ssh_run(host, cmd), alert_resolve(id)。用户说：'节点 db-02 磁盘告警 123，帮我清理 /var/log 下 7 天前日志并关闭告警。'",
    "Tools: stock_quote(ticker), portfolio_hold(user_id). User: 'How much is my NVDA position worth right now for user-42?' Respond with tool calls only.",
    "请演示如何调用 tool.math.add(a,b) 和 tool.math.mul(a,b) 完成表达式 (3+4)*(5+6) 的计算。输出严格 JSON 工具调用。",
]

CODING = [
    "请用 Python 实现一个函数 is_palindrome(s)，忽略大小写和非字母数字字符，判断字符串是否回文，并写出 3 个测试用例。",
    "Write a Rust function that computes the nth Fibonacci number using memoisation. Include at least one unit test.",
    "请写一段 SQL 查询 orders 表：每个 customer 最近一次下单的金额和日期，返回按金额降序的前 10 行。",
    "Implement a TypeScript debounce(fn, ms) utility with leading/trailing options and proper typing.",
    "请用 Go 写一个并发安全的计数器结构体 Counter，支持 Inc/Dec/Value 方法，并给出一个并发测试例子。",
    "Write a Python decorator @retry(n, delay) that retries a function n times with a fixed delay on exception. Include an example usage.",
    "请用 C++ 实现 LRU 缓存（put/get 均摊 O(1)），并说明你用到的数据结构。",
    "In Java, write a class BoundedBlockingQueue using ReentrantLock and Condition, with put(), take() methods.",
    "请用 Python 写一个函数 group_anagrams(words) 把同字母异序的单词分在一组，返回列表的列表。",
    "Write a React hook useLocalStorage(key, initial) that syncs state to localStorage and handles SSR safely.",
    "给定一棵二叉树，请用 Python 写函数 is_balanced(root) 判断是否是高度平衡，返回 True/False。",
    "Write a Dockerfile for a FastAPI app exposing port 8000 with uvicorn and multi-stage build to minimise image size.",
    "请用 SQL 写一个查询：找出 users 表中 email 重复的所有用户 id，按 email 分组返回。",
    "Write a Python function that merges two sorted lists in O(n+m) without using sorted() or heapq.",
    "请用 Python 写一个基于 argparse 的小 CLI，支持子命令 add/list/remove，管理一个本地 todo.json 文件。",
    "Implement a binary search tree in Kotlin with insert, lookup, and in-order iterator.",
    "请用 JavaScript 写一个函数 throttle(fn, ms) 并详细解释它和 debounce 的区别。",
    "Write a Bash script that finds all files larger than 100MB under a given directory and prints them sorted by size descending.",
    "请用 Python 实现 Trie 结构，支持 insert(word)、search(word)、startsWith(prefix)，给出 3 个测试用例。",
    "Write a CUDA kernel (pseudo-code acceptable) for element-wise vector add C = A + B and explain grid/block sizing.",
    "请用 Python 写一个基于 asyncio 的简单 TCP echo server，监听 0.0.0.0:9000。",
]


def _pad_to_length(base: str, target_len: int, category: str, idx: int) -> str:
    """Pad prompt to roughly target_len chars by appending a neutral 'context' block.

    We never change the core request, only append a disclaimer/example spec that
    keeps the prompt coherent."""
    if len(base) >= target_len:
        return base[:target_len]
    pad_block = (
        f"\n\n附加说明 ({category} #{idx}): "
        "请给出清晰的结构化回答，必要时使用列表、代码块或小节标题。"
        "如有歧义请按最常见的解释作答，并在末尾用一句话总结你的关键结论。"
        "回答需兼顾准确性与可读性，不要省略关键推理步骤。"
    )
    need = target_len - len(base)
    filler_chunks = []
    block_len = len(pad_block)
    while need > 0:
        if need >= block_len:
            filler_chunks.append(pad_block)
            need -= block_len
        else:
            filler_chunks.append(pad_block[:need])
            need = 0
    return base + "".join(filler_chunks)


def build_dataset(
    n_total: int = 256,
    seed: int = 1024,
) -> List[dict]:
    rng = random.Random(seed)
    categories = [
        ("humanities", HUMANITIES),
        ("social", SOCIAL),
        ("tech", TECH),
        ("math", MATH),
        ("tool_calls", TOOL_CALLS),
        ("coding", CODING),
    ]

    # even split with leftover distributed to first categories
    base_per = n_total // len(categories)
    extra = n_total - base_per * len(categories)
    per_cat = [base_per + (1 if i < extra else 0) for i in range(len(categories))]

    cases: List[dict] = []
    case_id = 0
    for (cat_name, seeds), count in zip(categories, per_cat):
        # Uniform target length bins 100..1000 within this category.
        bins = [int(100 + (1000 - 100) * (i + 0.5) / count) for i in range(count)]
        rng.shuffle(bins)
        for i in range(count):
            seed_prompt = seeds[i % len(seeds)]
            target_len = bins[i]
            padded = _pad_to_length(seed_prompt, target_len, cat_name, i)
            cases.append({
                "id": f"case_{case_id:04d}",
                "category": cat_name,
                "prompt": padded,
                "char_length": len(padded),
            })
            case_id += 1
    rng.shuffle(cases)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "spec_decode_eval_prompts.jsonl"),
    )
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    cases = build_dataset(n_total=args.n, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in cases:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # quick stats
    by_cat: dict = {}
    for c in cases:
        by_cat.setdefault(c["category"], []).append(c["char_length"])
    print(f"Wrote {len(cases)} cases to {output}")
    for cat, lens in sorted(by_cat.items()):
        print(
            f"  {cat:<12} n={len(lens):>3}  len min={min(lens):>4}  "
            f"max={max(lens):>4}  mean={sum(lens) / len(lens):6.1f}"
        )


if __name__ == "__main__":
    main()
