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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import time
import requests
from functools import wraps
from typing import Union

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
import concurrent.futures
from vllm import SamplingParams
import time
import json
from match.CaseMatchService import CaseMatchService
from search.PhenotypeSearchService import PhenotypeSearchService
# from search.GeneralSearchService import GeneralSearchService
# from search.CombinedSearchService import CombinedSearchService

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/remote-home/share/huggingface/Qwen2.5-3B-Instruct")
        search_token_ids = tokenizer.encode('</search>', add_special_tokens=False)
        print("Test Generate Timing BEGIN ###################\n")
        start_time = time.time()
        # users can customize different sampling_params at different run
        with self.update_sampling_params(stop_token_ids=search_token_ids):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)
        print("Test Generate Timing END ###################\n")
        end_time = time.time()
        print(f"Test Generate Time: {end_time - start_time:.2f} seconds")
        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)


class vLLMRolloutMeta(vLLMRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.case_path = config.match_source
        self.pheno_path = config.search_source
        self.match_top_n = config.match_top_n
        self.search_max_n = config.search_max_n
        self.matcher = CaseMatchService(source_path=self.case_path)
        self.searcher = PhenotypeSearchService(map_path=self.pheno_path)

    def batch_match(self, query: Union[str, List[str]], top_n=5) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        diseases_list = []
        for q in query:
            result = self.matcher.match_diseases(q, top_n)
            diseases_list.append(result)
        return diseases_list

    def batch_search(self, query: Union[str, List[str]], max_n=10) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        phenos_list = []
        for q in query:
            result = self.searcher.get_phenotypes_for_diseases(q, max_n)
            phenos_list.append(result)
        return phenos_list
    
    def extract_match_content(self, text: str) -> str:
        try:
            start_tag = '<match>'
            end_tag = '</match>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""


    def extract_search_content(self, text: str) -> str:
        try:
            start_tag = '<search>'
            end_tag = '</search>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        # print("BEGIN GENERATION\n\n\n\n")
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)

        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]
            
            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            # print("BEGIN ACTIVE INDICESn\n\n\n")
            # generate until all inputs are finished
            while active_indices:
                # only process the active inputs
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                print("Generate Timing BEGIN ###################\n")
                start_time = time.time()
                # generate in batch, according to active max tokens
                with self.update_sampling_params(n=1, stop=['</match>', '</search>'], max_tokens=max(active_max_tokens), detokenize=True):
                    outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False
                    )
                end_time = time.time()
                print("Generate Timing END ###################\n")
                print(f"Generate Time: {end_time - start_time:.2f} seconds")
                print("Search Timimg BEGIN ###################\n")
                start_time = time.time()
                # print("BEGIN SEARCHING\n\n\n\n\n")
                # collect the queries to search
                search_queries = []
                search_indices = []
                match_queries = []
                match_indices = []

                # process each output
                new_active_indices = []
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[0][i].tolist()
                    if self.tokenizer.eos_token_id in output_ids:
                        first_eos_idx = output_ids.index(self.tokenizer.eos_token_id)
                    else:
                        first_eos_idx = len(output_ids)
                    
                    if self.tokenizer.pad_token_id in output_ids:
                        first_pad_idx = output_ids.index(self.tokenizer.pad_token_id)
                    else:
                        first_pad_idx = len(output_ids)
                    
                    finish_reason = outputs[2][i]
                    stop_reason = outputs[3][i]

                    if finish_reason == 'stop' and isinstance(stop_reason, str):
                        output_ids = output_ids[:first_pad_idx]
                        output_str = self.tokenizer.decode(output_ids)
                        if '</match>' in stop_reason:
                            # 处理match请求
                            match_content = self.extract_match_content(output_str)
                            match_queries.append(match_content)
                            match_indices.append(idx)
                            new_active_indices.append(idx)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                        elif '</search>' in stop_reason:
                            # 处理search请求
                            search_content = self.extract_search_content(output_str)
                            search_queries.append(search_content)
                            search_indices.append(idx)
                            new_active_indices.append(idx)
                            curr_inputs[idx] += output_ids
                            result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == None:
                        # output eos, indicating finished; truncate from the first eos token
                        output_ids = output_ids[:first_eos_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'stop' and stop_reason == self.tokenizer.pad_token_id:
                        # for instruction model, there is a chance that the end is endoftext, not im_end, this case needs special handling
                        output_ids = output_ids[:first_pad_idx+1]
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                    elif finish_reason == 'length':
                        # output is too long
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [1] * len(output_ids)
                        
                # batch process
                if match_queries:
                    match_results = self.batch_match(match_queries, self.match_top_n)
                    for idx, result in zip(match_indices, match_results):
                        output_ids = self.tokenizer.encode(f" <refer>\n{result}\n</refer>")
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)
                if search_queries:
                    search_results = self.batch_search(search_queries, self.search_max_n)
                    for idx, result in zip(search_indices, search_results):
                        # update the output, add the search result
                        output_ids = self.tokenizer.encode(f" <result>\n{result}\n</result>")
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)

                # check if need to truncate for active indices
                length_checked_active_indices = []
                for idx in active_indices:
                    assert len(curr_inputs[idx]) - len(init_inputs[idx]) == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    if len(curr_inputs[idx]) - len(init_inputs[idx]) >= self.config.response_length:
                        curr_inputs[idx] = init_inputs[idx] \
                            + curr_inputs[idx][len(init_inputs[idx]):len(init_inputs[idx])+self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    else:
                        curr_max_tokens[idx] = self.config.response_length - len(curr_inputs[idx]) + len(init_inputs[idx])
                        if idx in new_active_indices:
                            length_checked_active_indices.append(idx)
                active_indices = length_checked_active_indices
                end_time = time.time()
                print("Search Timimg END ###################\n")
                print(f"Search Time: {end_time - start_time:.2f} seconds")

            output_ids_list = []
            # collect the results
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    output_ids_list.append(curr_inputs[idx][input_len:])

        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            response_list.append(response)
            result_mask_list_padded.append(result_mask)
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
    


class vLLMRolloutWithSearch(vLLMRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer
        self.case_path = config.match_source
        self.pheno_path = config.search_source
        self.match_top_n = config.match_top_n
        # self.wiki_max_n = config.search_wiki_max_n
        # self.pubmed_max_n = config.search_pubmed_max_n
        self.search_max_n = config.search_max_n
        # self.wiki_port = config.search_wiki_port
        # self.pubmed_port = config.search_pubmed_port
        # self.server_port = config.server_port
        self.matcher = CaseMatchService(source_path=self.case_path)
        # self.general_searcher = GeneralSearchService(wiki_port=self.wiki_port, pubmed_port=self.pubmed_port)
        self.orphanet_searcher = PhenotypeSearchService(map_path=self.pheno_path)
        # self.searcher = CombinedSearchService(map_path=self.pheno_path, wiki_port=self.wiki_port, pubmed_port=self.pubmed_port)
    

    def batch_match(self, query: Union[str, List[str]], top_n=5) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        cases_list = []
        for q in query:
            result = self.matcher.match_cases(q, top_n)
            cases_list.append(result)
        return cases_list

    def batch_search(self, query: Union[str, List[str]], max_n=10) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        phenos_list = []
        for q in query:
            # general_result = self.general_searcher.get_docs_for_diseases(q, self.wiki_max_n, max_n)
            # phenotype_result = self.searcher.get_phenotypes_for_diseases(q, max_n)
            result = self.searcher.get_combined_results(q, self.wiki_max_n, max_n)
            # json.dumps(results, indent=2, ensure_ascii=False)
            phenos_list.append(result)
        return phenos_list

    def batch_search_orpha(self, query: Union[str, List[str]], max_n=10) -> List[str]:
        if len(query) == 0:
            return 'invalid query'

        if isinstance(query, str):
            query = [query]

        phenos_list = []
        for q in query:
            # general_result = self.general_searcher.get_docs_for_diseases(q, self.wiki_max_n, max_n)
            # phenotype_result = self.searcher.get_phenotypes_for_diseases(q, max_n)
            result = self.orphanet_searcher.get_phenotypes_for_diseases(q, max_n)
            phenos_list.append(json.dumps(result, ensure_ascii=False))
        return phenos_list

    def batch_summarize(self, search_results: List[dict], max_tokens=100) -> List[dict]:
        """
        Process all disease search results efficiently in a single batch
        
        Args:
            search_results: List of dictionaries, each containing disease->source->content mappings
            max_tokens: Maximum number of tokens for each summary
        
        Returns:
            List of dictionaries with the same structure but with summarized content
        """
        if not search_results:
            return []
        
        print("Summarize Timing BEGIN ###################\n")
        start_time = time.time()
        
        # Step 1: Prepare a batch for the LLM by decoupling all disease data
        text_inputs = []
        mapping_data = []  # Store batch_idx, disease, source for reconstruction
        
        for batch_idx, disease_data in enumerate(search_results):
            for disease, sources_data in disease_data.items():
                for source, content in sources_data.items():
                    # Skip if no content or marked as "no reference"
                    # if not content or content == "no reference":
                    #     continue
                    if not content:
                        continue
                    # if source == "orphanet":
                    #     text_inputs.
                        
                    # Create a prompt for the LLM
                    prompt = (
                        f"<|im_start|>system\n\nYou are a phenotype summarization assistant. Based on the search result of a disease, you should:"
                        f"If the search source is orphanet, please keep the answer just a direct copy of the content. "
                        f"If the search source is wikipedia or pubmed, please try to extract related phenotypes from all the content and collate them into a phenotype list (if more than 10 phenotypes, just keep 10 of them)"
                        f"Your answer should be as brief as possible without any explanation."
                        f"Formulate your answer into json schema like '[pheno1, pheno2, ...]' If no phenotype can be extracted, please just answer 'no reference'."
                        f"\n<|im_end|>\n<|im_start|>user\n"
                        f"The search content of '{disease}' from source: {source} is {content}<|im_end|>\n<|im_start|>assistant\n"
                    )
                    
                    text_inputs.append(prompt)
                    mapping_data.append((batch_idx, disease, source))
        
        if not text_inputs:
            print("No valid disease data to summarize")
            return search_results
        
        # Step 2: Send the entire batch to the LLM
        print(f"Sending batch of {len(text_inputs)} prompts to summarization service...")
        try:
            response = requests.post(
                f"http://localhost:{self.server_port}/generate",
                json={
                    "text": text_inputs,
                    "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}
                }
            )
            print("response Success")
            if response.status_code != 200:
                print(f"Error from summarization service: {response.status_code}")
                return search_results
                
            result = response.json()
            summaries = [item['text'] for item in result]
            # print("step2 success")
            # Step 3: Reconstruct the results
            # Initialize results structure to match input structure
            summarized_results = []
            for _ in range(len(search_results)):
                summarized_results.append({})
            # print("Initialize results success")
            # Process each summary
            for i, summary in enumerate(summaries):
                batch_idx, disease, source = mapping_data[i]
                
                # Truncate summary if needed
                words = summary.split()
                if len(words) > 50:
                    truncated_summary = ' '.join(words[:50]) + "..."
                else:
                    truncated_summary = summary
                
                # Add to reconstructed results
                if disease not in summarized_results[batch_idx]:
                    summarized_results[batch_idx][disease] = {}
                
                summarized_results[batch_idx][disease][source] = truncated_summary
            # print("Process summary success")
            # Preserve original structure for any diseases/sources that weren't summarized
            for batch_idx, (original, summarized) in enumerate(zip(search_results, summarized_results)):
                # print("first line")
                for disease, sources in original.items():
                    # print("second line")
                    if disease not in summarized:
                        # print("If yes")
                        summarized[disease] = {}
                    for source, content in sources.items():
                        # print("third line")
                        if source not in summarized.get(disease, {}):
                            summarized[batch_idx][disease][source] = content
            
            end_time = time.time()
            print("Summarize Timing END ###################\n")
            print(f"Summarize Time: {end_time - start_time:.2f} seconds")
            
            return summarized_results
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return search_results
    
    def extract_match_content(self, text: str) -> str:
        try:
            start_tag = '<match>'
            end_tag = '</match>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""


    def extract_search_content(self, text: str) -> str:
        try:
            start_tag = '<search>'
            end_tag = '</search>'
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        ori_input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = ori_input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, ori_input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        with self.update_sampling_params(**kwargs):
            # prepare n copies for each input
            curr_inputs = []
            for input_ids in idx_list:
                for _ in range(self.sampling_params.n):
                    curr_inputs.append(input_ids.copy())
            init_inputs = [ids.copy() for ids in curr_inputs]
            
            # track the status of each input
            curr_max_tokens = [self.sampling_params.max_tokens] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))

            # collect the result mask of each rollout
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            
            # generate until all inputs are finished
            while active_indices:
                print(len(active_indices))
                # Process the entire active batch at once
                active_inputs = [curr_inputs[i] for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                print("Generate Timing BEGIN ###################\n")
                start_time = time.time()
                
                # generate the entire batch with a single call
                with self.update_sampling_params(n=1, max_tokens=max(active_max_tokens)):
                    outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=active_inputs,
                        use_tqdm=False
                    )
                
                end_time = time.time()
                print("Generate Timing END ###################\n")
                print(f"Generate Time: {end_time - start_time:.2f} seconds")
                print("Search Timing BEGIN ###################\n")
                start_time = time.time()
                
                # Collect search and match queries from all outputs at once
                search_queries = []
                search_indices = []
                match_queries = []
                match_indices = []
                new_active_indices = []
                
                # Process all outputs in the batch
                for i, idx in enumerate(active_indices):
                    output_ids = outputs[0][i].tolist()
                    output_str = self.tokenizer.decode(output_ids)
                    
                    # Check for search and match tags
                    search_tag_pos = output_str.find('</search>')
                    match_tag_pos = output_str.find('</match>')
                    eos_pos = -1 if self.tokenizer.eos_token_id not in output_ids else output_str.find(self.tokenizer.decode([self.tokenizer.eos_token_id]))
                    pad_pos = -1 if self.tokenizer.pad_token_id not in output_ids else output_str.find(self.tokenizer.decode([self.tokenizer.pad_token_id]))
                    
                    # Determine where to truncate based on what comes first
                    positions = [pos for pos in [search_tag_pos, match_tag_pos, eos_pos, pad_pos] if pos >= 0]
                    truncate_pos = min(positions) if positions else len(output_str)
                    
                    if search_tag_pos >= 0 and search_tag_pos == truncate_pos:
                        # Handle search request
                        search_content = self.extract_search_content(output_str[:truncate_pos + len('</search>')])
                        search_queries.append(search_content)
                        search_indices.append(idx)
                        new_active_indices.append(idx)
                        
                        # Get token IDs for the truncated output
                        truncated_output_ids = self.tokenizer.encode(output_str[:truncate_pos + len('</search>')])
                        curr_inputs[idx] += truncated_output_ids
                        result_mask_list[idx] += [1] * len(truncated_output_ids)
                        
                    elif match_tag_pos >= 0 and match_tag_pos == truncate_pos:
                        # Handle match request
                        match_content = self.extract_match_content(output_str[:truncate_pos + len('</match>')])
                        match_queries.append(match_content)
                        match_indices.append(idx)
                        new_active_indices.append(idx)
                        
                        # Get token IDs for the truncated output
                        truncated_output_ids = self.tokenizer.encode(output_str[:truncate_pos + len('</match>')])
                        curr_inputs[idx] += truncated_output_ids
                        result_mask_list[idx] += [1] * len(truncated_output_ids)
                        
                    else:
                        # No search/match or they're not the first stopping condition
                        # Just add the current output to results and don't mark as active
                        if pad_pos >= 0 and pad_pos == truncate_pos:
                            truncate_pos = pad_pos
                        elif eos_pos >= 0 and eos_pos == truncate_pos:
                            truncate_pos = eos_pos + len(self.tokenizer.decode([self.tokenizer.eos_token_id]))
                        
                        # Get token IDs for the truncated output
                        truncated_output_ids = self.tokenizer.encode(output_str[:truncate_pos])
                        curr_inputs[idx] += truncated_output_ids
                        result_mask_list[idx] += [1] * len(truncated_output_ids)
                
                # Process all search and match requests in batches
                if match_queries:
                    match_results = self.batch_match(match_queries, self.match_top_n)
                    for idx, result in zip(match_indices, match_results):
                        refer_text = f" <refer>\n{result}\n</refer>"
                        output_ids = self.tokenizer.encode(refer_text)
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)
                        
                if search_queries:
                    # search_results = self.batch_search(search_queries, self.search_max_n)
                    # print("External Help Begin ###################\n")
                    # llm_inference_begin = time.time()
                    # llm_results = self.batch_summarize(search_results)
                    # results_json = [json.dumps(summary, ensure_ascii=False).replace('\\\\', '\\').replace('\\"', '"') for summary in llm_results]
                    # assert len(results_json) == len(search_results)
                    # llm_inference_end = time.time()
                    # print(f"Help Time: {llm_inference_end - llm_inference_begin:.2f} seconds")
                    # print("External Help End ###################\n")
                    # assert len(search_indices) == len(results_json)

                    results_json = self.batch_search_orpha(search_queries, self.search_max_n)
                    for idx, result in zip(search_indices, results_json):
                        # result_text = f" <result>\n{result}\n</result>"
                        # result = "{\"Cystic Fibrosis\": {\"orphanet\": [\"Exocrine pancreatic insufficiency\", \"Malabsorption\", \"Bronchiectasis\", \"Recurrent respiratory infections\", \"Airway obstruction\", \"Elevated sweat chloride\", \"Absent vas deferens\", \"Failure to thrive\", \"Male infertility\"], \"wikipedia\": \"Doc 1(Title: Cystic Fibrosis Canada. Canadian researchers are viewed as leaders in the global effort to find a cure or control for cystic fibrosis. In 1989, Canadian researchers, funded by Cystic Fibrosis Canada, discovered the gene responsible for cystic fibrosis, and they continue to play a leading role in developing new treatments. Publications As well as general information about cystic fibrosis in Canada and resources for teachers, parents, and health care professionals, Cystic Fibrosis Canada publishes newsletters and reports covering such areas as research and training grants, clinical services and annual data on patients with cystic fibrosis. Kin Canada Since 1964, Kin Canada, a Canadian service organisation, has supported Cystic Fibrosis Canada, raising over $42 million in support of cystic fibrosis research and care. See also Shinerama List of cystic fibrosis organizations Cystic Fibrosis Foundation Cystic Fibrosis Trust References) \\n\", \"pubmed\": \"Doc 1(Title: Predicting the risk of cystic fibrosis with echogenic fetal bowel and one cystic fibrosis mutation. To assess fetal risk for cystic fibrosis when echogenic bowel and one cystic fibrosis mutation are detected. A hypothetical cohort of 1000 women with singleton pregnancies and echogenic fetal bowel during the second trimester was used to determine the probability of cystic fibrosis when one cystic fibrosis transmembrane conductance regulator mutation was detected. The risk of cystic fibrosis was calculated using the range of prevalence of cystic fibrosis in fetuses with echogenic bowel reported in the literature. Risk calculations for fetuses of Ashkenazi Jewish, Northern European, African-American, Hispanic, and Asian descent accounted for carrier frequencies and mutation detection rates specific to each ethnic group. As the assumed prevalence of cystic fibrosis increases from 1-25%, the probability that a white fetus with one mutation and echogenic fetal bowel actually has cystic fibrosis increases from 4.8% to 62.5%. Assuming a 2% risk of cystic fibrosis with echogenic fetal bowel, an Ashkenazi Jewish fetus and an Asian fetus with echogenic bowel and one mutation have a 3.1% and 72% risk of cystic fibrosis, respectively. The probability of cystic fibrosis in a nonwhite fetus is between those two extremes. The probability of cystic fibrosis after detection of echogenic bowel and one cystic fibrosis mutation varied among ethnic groups. Even at the highest prevalence of cystic fibrosis, most white fetuses will not have cystic fibrosis. In nonwhite populations almost half of these fetuses will have cystic fibrosis, even at the lowest prevalence of cystic fibrosis.) \\n\"}, \"Hemophilia\": {\"orphanet\": \"no reference\", \"wikipedia\": \"Doc 1(Title: World Federation of Hemophilia. The World Federation of Hemophilia (WFH) is an international non-profit organization dedicated to improving the lives of people with hemophilia (also spelled haemophilia) and other genetic bleeding disorders. It educates people with bleeding disorders and lobbies for improved medical treatment. 75% of people in the world with bleeding disorders do not know it and do not receive care. The WFH was established by Frank Schnabel in 1963 and has its headquarters in Montreal, Canada. It has member organizations in 147 countries and official recognition from the World Health Organization. The current President is Cesar Garrido. World Hemophilia Day World Hemophilia Day is held annually on April 17 by the WFH. It is an awareness day for hemophilia and other bleeding disorders, which also serves to raise funds and attract volunteers for the WFH. It was started in 1989; April 17 was chosen in honor of Frank Schnabel's birthday.) \\n\", \"pubmed\": \"Doc 1(Title: Clinical, instrumental, serological and histological findings suggest that hemophilia B may be less severe than hemophilia A. Recent evidence suggests that patients with severe hemophilia B may have a less severe disease compared to severe hemophilia A. To investigate clinical, radiological, laboratory and histological differences in the arthropathy of severe hemophilia A and hemophilia B, 70 patients with hemophilia A and 35 with hemophilia B with at least one joint bleeding were consecutively enrolled. Joint bleedings (&lt;10, 10-50, &gt;50), regimen of treatment (prophylaxis/on demand), World Federation of Hemophilia, Pettersson and ultrasound scores, serum soluble RANK ligand and osteoprotegerin were assessed in all patients. RANK, RANK ligand and osteoprotegerin expression was evaluated in synovial tissue from 18 hemophilia A and 4 hemophilia B patients. The percentage of patients with either 10-50 or more than 50 hemarthrosis was greater in hemophilia A than in hemophilia B (P&lt;0.001 and P=0.03, respectively), while that with less than 10 hemarthrosis was higher in hemophilia B (P&lt;0.0001). World Federation of Hemophilia (36.6 vs. 20.2; P&lt;0.0001) and ultrasound (10.9 vs. 4.3; P&lt;0.0001) score mean values were significantly higher in hemophilia A patients. Serum osteoprotegerin and soluble RANK ligand were decreased in hemophilia A versus hemophilia B (P&lt;0.0001 and P=0.006, respectively). Osteoprotegerin expression was markedly reduced in synovial tissue from hemophilia A patients. In conclusion, the reduced number of hemarthrosis, the lower World Federation of Hemophilia and ultrasound scores, and higher osteoprotegerin expression in serum and synovial tissue in hemophilia B suggest that hemophilia B is a less severe disease than hemophilia A. Osteoprotegerin reduction seems to play a pivotal role in the progression of arthropathy in hemophilia A.) \\n\"}, \"Duchenne Muscular Dystrophy\": {\"orphanet\": [\"Delayed speech and language development\", \"Global developmental delay\", \"Motor delay\", \"Specific learning disability\", \"Flexion contracture\", \"Cardiomyopathy\", \"Respiratory insufficiency\", \"Waddling gait\", \"Scoliosis\", \"Skeletal muscle atrophy\"], \"wikipedia\": \"Doc 1(Title: Duchenne muscular dystrophy. Biostrophin is a delivery vector for gene therapy in the treatment of Duchenne muscular dystrophy and Becker muscular dystrophy. References Further reading External links CDC's National Center on Birth Defects and Developmental Disabilities (previously listed below as \\\"Duchenne/Becker Muscular Dystrophy, NCBDDD\\\") at CDC Genes and Disease Page at NCBI Muscular dystrophy Wikipedia medicine articles ready to translate X-linked recessive disorders) \\n\", \"pubmed\": \"Doc 1(Title: A survey of the feasibility of developing osteoporosis clinical trials in Duchenne muscular dystrophy: Survey of the opinion of young people with Duchenne muscular dystrophy, families and clinicians. Given the extent of osteoporosis in people with Duchenne muscular dystrophy treated with glucocorticoids and the limited evidence of bone-protective therapies, clinical trials are needed. We conducted surveys to obtain the opinion of young people with Duchenne muscular dystrophy, parents/guardians and neuromuscular clinicians on the feasibility of osteoporosis clinical trials in this population. Online surveys were sent to three groups: (a) people with a confirmed diagnosis of Duchenne muscular dystrophy (≥14 years), (b) parents and guardians and (c) neuromuscular clinicians in the UK NorthStar Clinical Network. Surveys (a) and (b) were distributed via the UK Duchenne muscular dystrophy Registry. Survey respondents included 52 people with Duchenne muscular dystrophy with a median age of 17 years (range: 14, 40) and 183 parents/guardians. Fourteen out of 23 (61%) NorthStar centres responded. Of the 52 people with Duchenne muscular dystrophy, 13 (25%) were very concerned about their bone health and 21 (40%) were slightly concerned. Of the 183 parents/guardians, 75 (41%) were very concerned about their son's bone health and 90 (49%) were slightly concerned. Fractures and quality of life were the top two main outcome measures identified by people with Duchenne muscular dystrophy. Fractures and bone density were the top two main outcome measures identified by parents/guardians and neuromuscular clinicians. Thirty percent of people with Duchenne muscular dystrophy and 40% of parents/guardians would not take part if an osteoporosis trial involved a placebo that was administered parenterally. Only 2 of the 14 NorthStar centres (14%) would enrol people with Duchenne muscular dystrophy if a parenteral placebo was used in an osteoporosis trial in Duchenne muscular dystrophy. There is great awareness of bone health and the need for bone-protective trials among people with Duchenne muscular dystrophy and their carers. However, a proportion of people with Duchenne muscular dystrophy and parents are reluctant to participate in a placebo-controlled osteoporosis trial that included a parenteral therapy. A larger proportion of health care experts are unwillin"
                        # print(result)
                        # with open("/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangya-24047/Qiaoyu/DiagX/src/verl/workers/rollout/ResTest/wuwuwu.json", 'w') as f:
                        #     json.dump(result, f, indent=4)
                        # result = "\"orphanet\": [\"Exocrine pancreatic insufficiency\", \"Malabsorption\", \"Bronchiectasis\", \"Recurrent respiratory infections\", \"Airway obstruction\", \"Elevated sweat chloride\", \"Absent vas deferens\", \"Failure to thrive\", \"Male infertility\"]"
                        result_text = f" <result>\n{result}\n</result>"
                        output_ids = self.tokenizer.encode(result_text)
                        curr_inputs[idx] += output_ids
                        result_mask_list[idx] += [0] * len(output_ids)
                # Check length constraints for ALL indices that were active this round
                # Not just the new active ones
                length_checked_active_indices = []
                for idx in active_indices:  # <- Now we check ALL active indices, not just new_active_indices
                    response_length = len(curr_inputs[idx]) - len(init_inputs[idx])
                    assert response_length == len(result_mask_list[idx]), f"curr_inputs: {len(curr_inputs[idx])}, init_inputs: {len(init_inputs[idx])}, result_mask_list: {len(result_mask_list[idx])}"
                    
                    if response_length >= self.config.response_length:
                        # Truncate if exceeded max length - make sure we're strictly truncating
                        response_part = curr_inputs[idx][len(init_inputs[idx]):]
                        result_mask_part = result_mask_list[idx]
                        
                        # Strict truncation to response_length
                        curr_inputs[idx] = init_inputs[idx] + response_part[:self.config.response_length]
                        result_mask_list[idx] = result_mask_part[:self.config.response_length]
                        # Not adding to active indices as this is done generating
                    elif idx in new_active_indices:
                        # Only keep active if it was marked as active and still has room to grow
                        curr_max_tokens[idx] = self.config.response_length - response_length
                        length_checked_active_indices.append(idx)
                
                # Update active indices for next iteration
                active_indices = length_checked_active_indices
                
                end_time = time.time()
                print("Search Timing END ###################\n")
                print(f"Search Time: {end_time - start_time:.2f} seconds")

            # Perform a final length check on all indices before collecting results
            for idx in range(len(curr_inputs)):
                response_length = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_length > self.config.response_length:
                    # Truncate if exceeded max length
                    response_part = curr_inputs[idx][len(init_inputs[idx]):]
                    result_mask_part = result_mask_list[idx]
                    
                    # Strict truncation to response_length
                    curr_inputs[idx] = init_inputs[idx] + response_part[:self.config.response_length]
                    result_mask_list[idx] = result_mask_part[:self.config.response_length]

            # Collect the final results
            output_ids_list = []
            for i, input_ids in enumerate(idx_list):
                for j in range(self.sampling_params.n):
                    idx = i * self.sampling_params.n + j
                    input_len = len(input_ids)
                    # Final check to ensure response lengths are correct
                    response_part = curr_inputs[idx][input_len:]
                    if len(response_part) > self.config.response_length:
                        print(f"Warning: Final response length still too long: {len(response_part)} > {self.config.response_length}")
                        response_part = response_part[:self.config.response_length]
                        result_mask_list[idx] = result_mask_list[idx][:self.config.response_length]
                    output_ids_list.append(response_part)

        # Debug: Print the lengths of all outputs
        # print(f"Final output lengths: {[len(ids) for ids in output_ids_list]}")
        # print(f"Final mask lengths: {[len(mask) for mask in result_mask_list]}")

        # Process results into the expected format
        response_list = []
        result_mask_list_padded = []
        for output_ids, result_mask in zip(output_ids_list, result_mask_list):
            assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
            # Double-check we're not exceeding max length
            if len(output_ids) > self.config.response_length:
                output_ids = output_ids[:self.config.response_length]
                result_mask = result_mask[:self.config.response_length]
            
            response = torch.tensor(output_ids, device=ori_input_ids.device)
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            result_mask = torch.tensor(result_mask, device=ori_input_ids.device)
            result_mask = pad_sequence_to_length(result_mask, self.config.response_length, 0)
            response_list.append(response)
            result_mask_list_padded.append(result_mask)
        
        # Verify all tensors are the same shape before stacking
        shapes = [tensor.shape for tensor in response_list]
        assert all(shape == shapes[0] for shape in shapes), f"Inconsistent tensor shapes: {shapes}"
        
        response = torch.stack(response_list, dim=0)
        result_mask = torch.stack(result_mask_list_padded, dim=0)

        if self.config.n > 1 and do_sample:
            ori_input_ids = ori_input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([ori_input_ids, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # Position IDs for the response
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Result mask: result part is 0, other part is 1
        loss_mask = result_mask * response_attention_mask
        
        # All TP ranks should contain the same data here
        batch = TensorDict({
            'prompts': ori_input_ids,
            'responses': response,
            'input_ids': seq,  # Here input_ids become the whole sentences
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }, batch_size=batch_size)
        # requests.post(f"http://localhost:{self.server_port}/flush_cache")
        # Free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)