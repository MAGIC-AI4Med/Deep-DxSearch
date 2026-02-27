# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any
import json
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("deep_dx_search")
class DeepDxSearchRewardManagerWithSave(AbstractRewardManager):
    """The reward manager with deep diagnosis and search result saving."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", save_path=None) -> None:
        """
        Initialize the DeepDxSearchRewardManagerWithSave instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console.
            compute_score: A function to compute the reward score.
            reward_fn_key: The key used to access the data source. Defaults to "data_source".
            save_path: The path to save the detailed search/reward logs (jsonl format).
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.save_path = save_path
        self.is_token_apply = True
        
        # ============= 新增：初始化 trajectory 统计字典 =============
        self.trajectory_stats = {}  # {trajectory_string: count}
        # ============= 新增结束 =============

    def __call__(self, data: DataProto, return_dict: bool = False, curr_save_path=None) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards and save detailed diagnosis logs.
        """
        # Determine the save path for this specific call (allow override)
        save_path = curr_save_path if curr_save_path is not None else self.save_path

        # If there is rm score, we directly return rm score.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # Handle file opening if path is provided
        save_file = None
        if save_path is not None:
            try:
                save_file = open(save_path, 'a', encoding='utf-8')
            except Exception as e:
                print(f"[DeepDxSearch] Warning: Failed to open save_path {save_path}: {e}")

        print(f"Current trajectory status: \n{self.trajectory_stats}")
        print(f"Batch size: {len(data)}")

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode (Separate decoding is cleaner in the new version)
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            # Extract extended info
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            
            # Pack info for the score function
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            
            # ============= 修改：传入 trajectory_stats =============
            # Compute Score
            score_result = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                tokenizer=self.tokenizer,
                trajectory_stats=self.trajectory_stats  # 新增参数
            )
            # ============= 修改结束 =============
            
            # Initialize log dictionary for saving
            log_entry = {
                'data_source': data_source,
                'sequences_str': sequences_str, 
                'ground_truth': ground_truth,
            }

            # ========================= 修改核心逻辑 =========================
            
            # Case 1: 如果是 Tuple (现在返回 7 个值)
            if isinstance(score_result, tuple):
                # ============= 修改：解包 7 个值 =============
                final_score, detailed_reason, refer_adjust, lookup_score, search_score, answer_score, trajectory, diversity_coefficient = score_result
                # ============= 修改结束 =============
                
                # 1. 确定用于训练的标量 reward
                reward = final_score
                
                # 2. 填入 log_entry (对应你要求的字段映射)
                log_entry['score'] = final_score
                log_entry['reason'] = detailed_reason
                log_entry['refer_score'] = refer_adjust
                log_entry['lookup_score'] = lookup_score  
                log_entry['search_score'] = search_score  
                log_entry['answer_score'] = answer_score
                # ============= 新增：记录 trajectory 和 diversity_coefficient =============
                log_entry['trajectory'] = trajectory
                log_entry['diversity_coefficient'] = diversity_coefficient
                # ============= 新增结束 =============
                
                # 3. 填入 reward_extra_info (用于回传给主进程记录指标)
                # 注意：这里我们把子项分数也存进去，方便看曲线
                reward_extra_info['score'].append(final_score)
                reward_extra_info['refer_score'].append(refer_adjust)
                reward_extra_info['lookup_score'].append(lookup_score)  # 原来的search_score改为lookup_score
                reward_extra_info['search_score'].append(search_score)   # 新增search_score
                reward_extra_info['answer_score'].append(answer_score)
                # ============= 新增：添加到 reward_extra_info =============
                reward_extra_info['trajectory'].append(trajectory)
                reward_extra_info['diversity_coefficient'].append(diversity_coefficient)
                # ============= 新增结束 =============
                
                # ============= 新增：更新 trajectory_stats =============
                if trajectory is not None:  # 只有有效的trajectory才统计
                    if trajectory in self.trajectory_stats:
                        self.trajectory_stats[trajectory] += 1
                    else:
                        self.trajectory_stats[trajectory] = 1
                # ============= 新增结束 =============

            # Case 2: 如果是 Dict (保留原有逻辑以兼容其他代码)
            elif isinstance(score_result, dict):
                reward = score_result["score"]
                for key, value in score_result.items():
                    reward_extra_info[key].append(value)
                    log_entry[key] = value
            
            # Case 3: 如果是单个数值 (float/int)
            else:
                reward = score_result
                log_entry['score'] = score_result
                reward_extra_info['score'].append(reward)

            # ========================= 逻辑结束 =========================

            reward_tensor[i, valid_response_length - 1] = reward

            # Write to file
            if save_file is not None:
                save_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            # Console Printing (Debug)
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"[data_source] {data_source}")
                print(f"[ground_truth] {ground_truth}")
                
                # 针对 Tuple 情况的专门打印
                if isinstance(score_result, tuple):
                    print(f"[score] {final_score}")
                    print(f"[reason] {detailed_reason[:100]}...")
                    print(f"[refer_score] {refer_adjust}")
                    print(f"[lookup_score] {lookup_score}")  # 原来的search_score改为lookup_score
                    print(f"[search_score] {search_score}")   # 新增search_score打印
                    print(f"[answer_score] {answer_score}")
                    # ============= 新增：打印 trajectory 和 diversity_coefficient =============
                    print(f"[trajectory] {trajectory}")
                    print(f"[diversity_coefficient] {diversity_coefficient:.3f}")
                    print(f"[trajectory_stats_total] {sum(self.trajectory_stats.values())}")
                    # ============= 新增结束 =============
                elif isinstance(score_result, dict):
                     for key, value in score_result.items():
                        print(f"[{key}] {value}")
                else:
                    print(f"[score] {score_result}")
                print('-' * 20)

        # Close file
        if save_file is not None:
            save_file.close()

        print(f"Current trajectory status after step: \n{self.trajectory_stats}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor