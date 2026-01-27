# Copyright 2025 HOUMO AI
#
# File: eval_ppl.py
# Description:
#   Perplexity evaluation module for language models.
#   This module provides functions for evaluating model perplexity on
#   various datasets including WikiText2, PTB, and C4.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .datautils import get_loaders


@torch.no_grad()
def evaluate(lm, model_path=None, seed=0, output_dir="data"):
    results = {}
    # if args.multigpu:
    #     map_layers_to_multi_gpus(lm.model.model.layers)
    #     input_device = lm.model.model.layers[0].device
    #     output_device = lm.model.model.layers[-1].device
    #     assert input_device == output_device
    #     lm._device = input_device
    #     lm.model.model.embed_tokens.to(input_device)
    #     lm.model.model.norm.to(output_device)
    #     lm.model.lm_head.to(output_device)
    # else:
    #     lm.model = lm.model.to(lm.device)

    if True:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f"{output_dir}/testloader__{dataset}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                # logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=seed,
                    model=model_path,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            with tqdm(range(nsamples)) as pbar:
                for i in pbar:
                    batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                        lm.device
                    )
                    outputs = lm.model(batch)
                    hidden_states = outputs[0]
                    logits = lm.lm_head(hidden_states)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                        :, 1:
                    ].to(lm.lm_head.weight.device)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    neg_log_likelihood = loss.float() * lm.seqlen
                    nlls.append(neg_log_likelihood)
                    tmp_ppl = torch.exp(
                        torch.stack(nlls).sum() / ((i + 1) * lm.seqlen)
                    ).item()
                    pbar.set_postfix_str(f"--{tmp_ppl:4.4}")
                    # if i == args.limit:
                    #     break
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            # logger.info(f'{dataset} : {ppl.item()}')
            lm.config.use_cache = use_cache
            results[dataset] = round(ppl.item(), 4)
        results["ppl_avg"] = round(sum(results.values()) / len(results.values()), 4)

    """
    if args.eval_mmlu:
        # eval quantized model on MMLU
        from mmlu_eval import run_mmlu_eval
        for num_few_shots in [0, 5]:
            save_dir = os.path.join(args.output_dir, "mmlu", f"{num_few_shots}-shot")
            run_mmlu_eval(lm.model, lm.tokenizer, args.net, num_few_shots, args.mmlu_data_dir, save_dir,logger)

    if args.eval_QA:
        lm.model.eval()
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM
        hflm = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer, batch_size=args.lm_eval_batch_size)

        task_manager = lm_eval.tasks.TaskManager(include_path="./datasets/lm_eval_configs/tasks",
                                                 include_defaults=False)
        task_names = lm_eval_utils.pattern_match(['hellaswag', 'winogrande', 'piqa', 'lambada_openai', 'arc_easy', 'arc_challenge'], task_manager.all_tasks)
        # ['hellaswag', 'winogrande', 'piqa', 'lambada_openai', 'arc_easy', 'arc_challenge']
        task_results = {}
        for task_name in task_names:
            print(f'eval {task_name}------')
            import logging
            logging.disable(logging.CRITICAL)
            task_result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size,
                                             task_manager=task_manager)['results']
            task_result = task_result[task_name]
            logger.info(task_result)
            acc = round(task_result.get('acc_norm,none', task_result['acc,none']) * 100, 4)
            task_results[task_name] = acc
            logging.disable(logging.NOTSET)
            logger.info(f"{task_name}_acc: {acc}%")
        metric_vals = {task: result for task, result in task_results.items()}
        task_results['acc_avg']=round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        results.update(task_results)
        logger.info(results)
    """

    return results
