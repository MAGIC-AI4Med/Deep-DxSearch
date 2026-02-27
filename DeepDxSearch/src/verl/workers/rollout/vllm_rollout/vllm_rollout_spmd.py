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
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator
import json

import cloudpickle as pickle
import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest
from dataclasses import dataclass
from typing import Optional, Union, List

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.model import get_lora_rank_from_adapter
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

# Import retrieval services - adjust paths as needed
try:
    from match.CaseMatchService import CaseMatchService
    from lookup.PhenotypeLookupService import PhenotypeLookupService
    from search.KnowledgeSearchService import KnowledgeSearchService
    # from search.WikipediaSearchService import WikiSearchService
    # from search.PubmedSearchService import PubmedSearchService
    # from search.TextbookSearchService import TextbookSearchService
    RETRIEVAL_SERVICES_AVAILABLE = True
except ImportError:
    RETRIEVAL_SERVICES_AVAILABLE = False
    print("Warning: Retrieval services not found, mock services will be used or errors will occur.")
    CaseMatchService = None
    PhenotypeLookupService = None
    KnowledgeSearchService = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            lora_rank = get_lora_rank_from_adapter(lora_adapter_path)
        else:
            lora_rank = model_config.lora_rank

        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
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

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # print("using vllm_rollout to return batch results")
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

""" Here is the multi-turn adaption for DeepDxSearch """

@dataclass
class RetrievalConfig:
    """Configuration for retrieval functionality (including lookup, match, and search)."""
    
    # Special tags for lookup match, and search operations
    lookup_start_tag: str = "<lookup>"
    lookup_end_tag: str = "</lookup>"
    match_start_tag: str = "<match>"
    match_end_tag: str = "</match>"
    search_start_tag: str = "<search>"
    search_end_tag: str = "</search>"
    
    # returned passive tags
    lookup_result_start_tag: str = "<guide>"
    lookup_result_end_tag: str = "</guide>"
    match_result_start_tag: str = "<refer>"
    match_result_end_tag: str = "</refer>"
    search_result_start_tag: str = "<result>"
    search_result_end_tag: str = "</result>"
    
    # Retrieval parameters
    match_source: Optional[str] = None
    lookup_source: Optional[str] = None
    search_summarize: bool = False # Whether to use LLM summarization in SearchManager
    search_ports: Optional[dict] = None # Ports for wiki, pmc, etc.

    match_top_n: int = 5
    lookup_max_n: int = 10
    
    # Safety limits
    max_iterations: int = 20
    
    @classmethod
    def from_config(cls, config: RolloutConfig) -> "RetrievalConfig":
        """Create RetrievalConfig from RolloutConfig."""
        return cls(
            match_source=config.get("match_source", None),
            lookup_source=config.get("lookup_source", None),
            match_top_n=config.get("match_top_n", 5),
            lookup_max_n=config.get("lookup_max_n", 10),
            max_iterations=config.get("retrieval_max_iterations", 20),
            search_summarize=config.get("search_summarize", False),
            search_ports=config.get("search_ports", {}), # Expects dict like {'wiki': 8001, ...}

            # Tags
            lookup_start_tag=config.get("lookup_start_tag", "<lookup>"),
            lookup_end_tag=config.get("lookup_end_tag", "</lookup>"),
            match_start_tag=config.get("match_start_tag", "<match>"),
            match_end_tag=config.get("match_end_tag", "</match>"),
            search_start_tag=config.get("search_start_tag", "<search>"),
            search_end_tag=config.get("search_end_tag", "</search>"),
            
            lookup_result_start_tag=config.get("lookup_result_start_tag", "<guide>"),
            lookup_result_end_tag=config.get("lookup_result_end_tag", "</guide>"),
            match_result_start_tag=config.get("match_result_start_tag", "<refer>"),
            match_result_end_tag=config.get("match_result_end_tag", "</refer>"),
            search_result_start_tag=config.get("search_result_start_tag", "<result>"),
            search_result_end_tag=config.get("search_result_end_tag", "</result>"),
        )


class vLLMRolloutWithDeepDxSearch(vLLMRollout):
    """
    vLLM Rollout with iterative retrieval (lookup/match) support.
    
    This class extends vLLMRollout to support:
    - Iterative generation with intermediate stops at lookup/match tags
    - Batch processing of lookup and match queries
    - Loss masking for retrieval results (so they don't contribute to training loss)
    
    All original vLLMRollout features are preserved:
    - LoRA support
    - Log probability calculation
    - Sampling parameter customization
    - Multi-turn conversation support
    
    Key improvements:
    1. Configurable special tags via RetrievalConfig
    2. Proper error handling for retrieval service failures
    3. Cleaner code structure with separated concerns
    4. Safety limits to prevent infinite loops
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        # Initialize parent class with all original functionality
        super().__init__(config, model_config, device_mesh)
        
        # Store tokenizer reference for convenience
        self.tokenizer = model_config.tokenizer
        
        # Initialize retrieval configuration
        self.retrieval_config = RetrievalConfig.from_config(config)
        
        # Initialize retrieval services
        self._init_retrieval_services()
        
        # Pre-compute special token sequences for efficient detection
        self._init_special_tokens()
        
        logger.info(f"vLLMRolloutWithDeepDxSearch initialized with retrieval_config: {self.retrieval_config}")

    def _init_retrieval_services(self):
        """Initialize retrieval services (lookuper and matcher) with error handling."""
        self.matcher = None
        self.lookuper = None
        self.searcher = None
        
        if not RETRIEVAL_SERVICES_AVAILABLE:
            logger.warning(
                "Retrieval services (CaseMatchService, PhenotypeLookupService) not available. "
                "Retrieval functionality will be disabled. Install the required packages to enable."
            )
            return
            
        try:
            if self.retrieval_config.match_source:
                self.matcher = CaseMatchService(source_path=self.retrieval_config.match_source)
                logger.info(f"Initialized CaseMatchService (matcher) with source: {self.retrieval_config.match_source}")
        except Exception as e:
            logger.error(f"Failed to initialize CaseMatchService (matcher): {e}")
        # import ipdb; ipdb.set_trace()
        try:
            if self.retrieval_config.lookup_source:
                self.lookuper = PhenotypeLookupService(map_path=self.retrieval_config.lookup_source)
                logger.info(f"Initialized PhenotypeLookupService (lookuper) with source: {self.retrieval_config.lookup_source}")
        except Exception as e:
            logger.error(f"Failed to initialize PhenotypeLookupService (lookuper): {e}")
        try:
            # import ipdb; ipdb.set_trace()
            # Get ports from config or use defaults matching KnowledgeSearchService.py
            ports = self.retrieval_config.search_ports or {}
            self.searcher = KnowledgeSearchService(
                wiki_port=ports.get('wiki_port', 8001),
                pmc_port=ports.get('pmc_port', 8000),
                book_port=ports.get('book_port', 8002),
                sgl_port=ports.get('sgl_port', None)
            )
            logger.info("Initialized SearchManager (searcher).")
        except Exception as e:
            logger.error(f"Failed to initialize SearchManager: {e}")

    def _init_special_tokens(self):
        """Pre-compute special token sequences for efficient detection."""
        # Encode end tags for stop token detection
        self.lookup_end_token_ids = self.tokenizer.encode(
            self.retrieval_config.lookup_end_tag, add_special_tokens=False
        )
        self.match_end_token_ids = self.tokenizer.encode(
            self.retrieval_config.match_end_tag, add_special_tokens=False
        )
        self.search_end_token_ids = self.tokenizer.encode(
            self.retrieval_config.search_end_tag, add_special_tokens=False
        )
        
        logger.debug(f"Lookup end token IDs: {self.lookup_end_token_ids}")
        logger.debug(f"Match end token IDs: {self.match_end_token_ids}")
        logger.debug(f"Search end token IDs: {self.search_end_token_ids}")

    def _extract_tag_content(self, text: str, start_tag: str, end_tag: str) -> str:
        """
        Extract content between start and end tags (last occurrence).
        
        Args:
            text: Input text containing the tags
            start_tag: Opening tag
            end_tag: Closing tag
            
        Returns:
            Content between tags, or empty string if not found
        """
        try:
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            return ""

    def _extract_lookup_content(self, text: str) -> str:
        """Extract content from <lookup>...</lookup> tags."""
        return self._extract_tag_content(
            text,
            self.retrieval_config.lookup_start_tag,
            self.retrieval_config.lookup_end_tag
        )

    def _extract_match_content(self, text: str) -> str:
        """Extract content from <match>...</match> tags."""
        return self._extract_tag_content(
            text, 
            self.retrieval_config.match_start_tag, 
            self.retrieval_config.match_end_tag
        )
    def _extract_search_content(self, text: str) -> str:
        """Extract content from <search>...</search> tags."""
        return self._extract_tag_content(
            text, 
            self.retrieval_config.search_start_tag, 
            self.retrieval_config.search_end_tag
        )

    def batch_lookup(self, queries: Union[str, List[str]], max_n: Optional[int] = None) -> List[str]:
        """
        Batch process lookup queries.
        
        Args:
            queries: Single query or list of queries
            max_n: Maximum number of results per query (defaults to config value)
            
        Returns:
            List of lookup results as JSON strings
        """
        if not queries:
            return []
            
        if isinstance(queries, str):
            queries = [queries]
            
        if max_n is None:
            max_n = self.retrieval_config.lookup_max_n
            
        if self.lookuper is None:
            logger.warning("Lookuper not initialized, returning empty results")
            return ["no reference available"] * len(queries)
            
        results = []
        for query in queries:
            try:
                result = self.lookuper.get_phenotypes_for_diseases(query)
                results.append(json.dumps(result, ensure_ascii=False) if result else "no reference available")
            except Exception as e:
                logger.error(f"Lookup error for query '{query[:50]}...': {e}")
                results.append("no reference available")
                
        return results

    def batch_match(self, queries: Union[str, List[str]], top_n: Optional[int] = None) -> List[str]:
        """
        Batch process match queries.
        
        Args:
            queries: Single query or list of queries
            top_n: Number of top matches to return (defaults to config value)
            
        Returns:
            List of match results as strings
        """
        if not queries:
            return []
            
        if isinstance(queries, str):
            queries = [queries]
            
        if top_n is None:
            top_n = self.retrieval_config.match_top_n
            
        if self.matcher is None:
            logger.warning("Matcher not initialized, returning empty results")
            return ["no reference available"] * len(queries)
            
        results = []
        for query in queries:
            try:
                result = self.matcher.match_cases(query, top_n)
                results.append(result if result else "no reference available")
            except Exception as e:
                logger.error(f"Match error for query '{query[:50]}...': {e}")
                results.append("no reference available")
                
        return results

    # def batch_search(self, queries: Union[str, List[str]]) -> List[str]:
    #     """
    #     Args:
    #         queries: Single query string (e.g., "|WIKI| x |PMC| y") or list of strings
            
    #     Returns:
    #         List of search results formatted as JSON strings
    #     """
    #     if not queries:
    #         return []
            
    #     if isinstance(queries, str):
    #         queries = [queries]
            
    #     if self.searcher is None:
    #         logger.warning("Searcher not initialized, returning empty results")
    #         return ["no search results available"] * len(queries)
            
    #     try:
    #         # SearchManager returns List[Dict[str, str]]
    #         # e.g., [{"Q1: query": "result"}, ...]
    #         results_dicts = self.searcher.process_batch(
    #             queries, 
    #             summarize=self.retrieval_config.search_summarize
    #         )
            
    #         # Convert dicts to string representation for the LLM
    #         results_strs = []
    #         for res_dict in results_dicts:
    #             if not res_dict:
    #                 results_strs.append("no results found")
    #             else:
    #                 # Dump the dict to JSON string so LLM can read the structured results
    #                 results_strs.append(json.dumps(res_dict, ensure_ascii=False))
    #         return results_strs
            
    #     except Exception as e:
    #         logger.error(f"Search error: {e}")
    #         return ["error during search"] * len(queries)

    def batch_search(self, queries: Union[str, List[str]]) -> List[str]:
        """
        Args:
            queries: Single query string (e.g., "|WIKI| x |PMC| y") or list of strings
            
        Returns:
            List of search results formatted as JSON strings
        """
        # 1. 统一输入格式
        if not queries:
            return []
            
        if isinstance(queries, str):
            queries = [queries]
            
        # 2. 检查 Searcher 是否初始化 (如果连Searcher都没有，就不用广播了，大家都返回空)
        if self.searcher is None:
            logger.warning("Searcher not initialized, returning empty results")
            return ["no search results available"] * len(queries)

        # 3. 分布式广播逻辑 (解决死锁的核心)
        if torch.distributed.is_initialized():
            # 获取全局 Rank
            rank = torch.distributed.get_rank()
            # 获取 TP 大小 (从 config 中读取，默认为 1)
            tp_size = self.config.get("tensor_model_parallel_size", 1)
            
            # 计算当前进程在 TP 组内的局部 Rank (0, 1, ..., tp_size-1)
            # 假设连续的 rank 组成一个 TP 组
            local_rank = rank % tp_size
            
            # 计算广播源 Rank (当前组的第一个 Rank)
            src_rank = rank - local_rank
            
            # 准备广播容器 (List[List[str]])
            # 这是一个包含一个元素的列表，元素就是我们要广播的结果列表
            object_list = [None]
            
            # === 只有组长 (Rank 0) 执行实际的搜索 ===
            if local_rank == 0:
                try:
                    # 执行网络请求 (可能耗时，可能带有不确定性)
                    results_dicts = self.searcher.process_batch(
                        queries, 
                        summarize=self.retrieval_config.search_summarize
                    )
                    
                    # 将结果处理为字符串，确保存入容器
                    results_strs = []
                    for res_dict in results_dicts:
                        if not res_dict:
                            results_strs.append("no results found")
                        else:
                            # 转换为 JSON 字符串
                            results_strs.append(json.dumps(res_dict, ensure_ascii=False))
                    
                    object_list[0] = results_strs
                    
                except Exception as e:
                    logger.error(f"Search error in Rank {rank}: {e}")
                    # 发生错误时，也要生成一个一致的错误列表广播出去
                    object_list[0] = ["error during search"] * len(queries)
            
            # === 关键步骤：广播 ===
            # src_rank 将 object_list 内容发送给组内所有其他进程
            # 这一步是阻塞的，确保所有人在这一行代码处数据对其
            torch.distributed.broadcast_object_list(object_list, src=src_rank)
            
            # 取出广播后的结果
            final_results = object_list[0]
            return final_results

        else:
            # === 单卡/非分布式模式 (Fallback) ===
            try:
                results_dicts = self.searcher.process_batch(
                    queries, 
                    summarize=self.retrieval_config.search_summarize
                )
                
                results_strs = []
                for res_dict in results_dicts:
                    if not res_dict:
                        results_strs.append("no results found")
                    else:
                        results_strs.append(json.dumps(res_dict, ensure_ascii=False))
                return results_strs
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return ["error during search"] * len(queries)


    def _find_stop_position(
        self, 
        output_ids: List[int], 
        output_str: str
    ) -> tuple[int, str]:
        """
        Find the first stopping position and its type.
        
        Args:
            output_ids: List of output token IDs
            output_str: Decoded output string
            
        Returns:
            Tuple of (truncate_position_in_string, stop_type)
            stop_type is one of: 'lookup', 'match', 'eos', 'pad', 'none'
        """
        # Find positions of different stop conditions in the string
        lookup_pos = output_str.find(self.retrieval_config.lookup_end_tag)
        match_pos = output_str.find(self.retrieval_config.match_end_tag)
        search_pos = output_str.find(self.retrieval_config.search_end_tag)
        
        # Find EOS position by looking at token IDs
        eos_pos = -1
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id in output_ids:
            eos_idx = output_ids.index(self.tokenizer.eos_token_id)
            # Decode up to (not including) EOS to get string position
            eos_pos = len(self.tokenizer.decode(output_ids[:eos_idx], skip_special_tokens=False))
            
        # Find PAD position
        pad_pos = -1
        if self.pad_token_id is not None and self.pad_token_id in output_ids:
            pad_idx = output_ids.index(self.pad_token_id)
            pad_pos = len(self.tokenizer.decode(output_ids[:pad_idx], skip_special_tokens=False))
        
        # Collect valid positions
        positions = {
            'lookup': lookup_pos,
            'match': match_pos,
            'search': search_pos,
            'eos': eos_pos,
            'pad': pad_pos,
        }
        
        valid_positions = {k: v for k, v in positions.items() if v >= 0}
        
        if not valid_positions:
            return len(output_str), 'none'
            
        # Find the earliest stopping position
        stop_type = min(valid_positions, key=valid_positions.get)
        truncate_pos = valid_positions[stop_type]
        
        # Adjust truncate position to include the tag itself
        if stop_type == 'lookup':
            truncate_pos += len(self.retrieval_config.lookup_end_tag)
        elif stop_type == 'match':
            truncate_pos += len(self.retrieval_config.match_end_tag)
        elif stop_type == 'search':
            truncate_pos += len(self.retrieval_config.search_end_tag)
        elif stop_type == 'eos':
            # Include the EOS token
            truncate_pos += len(self.tokenizer.decode([self.tokenizer.eos_token_id], skip_special_tokens=False))
            
        return truncate_pos, stop_type

    def _truncate_to_response_length(
        self,
        curr_input: List[int],
        init_input: List[int],
        result_mask: List[int],
        response_length: int
    ) -> tuple[List[int], List[int]]:
        """
        Truncate input and mask to response length limit.
        
        Args:
            curr_input: Current full input token IDs
            init_input: Initial prompt token IDs
            result_mask: Mask for the response part
            response_length: Maximum response length
            
        Returns:
            Tuple of (truncated_input, truncated_mask)
        """
        response_part = curr_input[len(init_input):]
        
        if len(response_part) > response_length:
            response_part = response_part[:response_length]
            result_mask = result_mask[:response_length]
            
        return init_input + response_part, result_mask

    def _prepare_lora_requests(self, batch_size: int) -> Optional[List[LoRARequest]]:
        """
        Prepare LoRA requests if LoRA is enabled.
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            List of LoRARequest objects or None if LoRA is not enabled
        """
        if not self.lora_kwargs:
            return None
            
        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if len(lora_int_ids) > 0:
            lora_int_id = lora_int_ids[0]
            return [
                LoRARequest(
                    lora_name=f"{lora_int_id}", 
                    lora_int_id=lora_int_id, 
                    lora_path="/simon-stub-path"
                )
            ] * batch_size
        return None

    def _get_sampling_kwargs(self, prompts: DataProto) -> dict:
        """
        Get sampling parameters based on generation mode.
        
        Args:
            prompts: Input batch with meta information
            
        Returns:
            Dictionary of sampling parameter overrides
        """
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample:
            return {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,
            }
        elif is_validate:
            return {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        return {}

    @GPUMemoryLogger(role="vllm rollout with retrieval", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences with iterative retrieval (lookup/match) support.
        
        This method implements iterative generation:
        1. Generate until a lookup/match tag or EOS is found
        2. If lookup/match tag, execute the query and append results
        3. Continue generation until done or max length reached
        
        The method preserves all original vLLMRollout features:
        - LoRA support
        - Log probability calculation (note: only for non-search generations)
        - Sampling parameter customization
        
        Args:
            prompts: Input batch containing:
                - batch["input_ids"]: prompt token IDs [bsz, prompt_length]
                - batch["attention_mask"]: attention mask [bsz, prompt_length]
                - batch["position_ids"]: position IDs [bsz, prompt_length]
                - meta_info["eos_token_id"]: EOS token ID
                
        Returns:
            DataProto with generated sequences:
                - prompts: [bsz, prompt_length]
                - responses: [bsz, response_length]
                - input_ids: [bsz, prompt_length + response_length]
                - attention_mask: [bsz, prompt_length + response_length]
                - position_ids: [bsz, prompt_length + response_length]
                - response_mask: [bsz, response_length] - 1 for LLM tokens, 0 for retrieval/padding
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        batch_size = idx.size(0)
        
        # Prepare input token lists
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object
            )
        
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        raw_prompt_ids_list = list(non_tensor_batch.pop("raw_prompt_ids"))
        
        # Get sampling parameters based on mode
        sampling_kwargs = self._get_sampling_kwargs(prompts)
        
        # Prepare LoRA requests if enabled
        lora_requests = self._prepare_lora_requests(batch_size)
        
        with self.update_sampling_params(**sampling_kwargs):
            # Initialize tracking structures
            curr_inputs = [list(ids) for ids in raw_prompt_ids_list]
            init_inputs = [ids.copy() for ids in curr_inputs]
            
            curr_max_tokens = [self.config.response_length] * len(curr_inputs)
            active_indices = list(range(len(curr_inputs)))
            result_mask_list = [[] for _ in range(len(curr_inputs))]
            
            # For log probability tracking (simplified - only final generation)
            # Note: Full log prob tracking across iterations would require more complex handling
            
            iteration = 0
            
            while active_indices and iteration < self.retrieval_config.max_iterations:
                iteration += 1
                logger.debug(f"Retrieval iteration {iteration}, active samples: {len(active_indices)}")
                
                # Prepare active inputs for generation
                active_inputs = [{"prompt_token_ids": curr_inputs[i]} for i in active_indices]
                active_max_tokens = [curr_max_tokens[i] for i in active_indices]
                
                # Prepare LoRA requests for active samples
                active_lora_requests = None
                if lora_requests is not None:
                    active_lora_requests = [lora_requests[i] for i in active_indices]
                
                start_time = time.time()
                
                # Generate with stop tokens for retrieval (lookup/match)
                with self.update_sampling_params(
                    n=1, 
                    max_tokens=max(active_max_tokens),
                ):
                    outputs = self.inference_engine.generate(
                        prompts=active_inputs,
                        sampling_params=self.sampling_params,
                        lora_request=active_lora_requests,
                        use_tqdm=False,
                    )
                
                gen_time = time.time() - start_time
                logger.debug(f"Generation time: {gen_time:.2f}s for {len(active_indices)} samples")
                
                # Process outputs and collect retrieval (lookup/match) queries
                lookup_queries = []
                lookup_indices = []
                match_queries = []
                match_indices = []
                search_queries = []
                search_indices = []
                new_active_indices = []
                
                start_time = time.time()
                
                for i, output_idx in enumerate(active_indices):
                    output = outputs[i]
                    output_ids = list(output.outputs[0].token_ids)
                    output_str = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                    
                    truncate_pos, stop_type = self._find_stop_position(output_ids, output_str)
                    truncated_str = output_str[:truncate_pos]
                    truncated_ids = self.tokenizer.encode(truncated_str, add_special_tokens=False)
                    
                    if stop_type == 'lookup':
                        lookup_content = self._extract_lookup_content(truncated_str)
                        if lookup_content:
                            lookup_queries.append(lookup_content)
                            lookup_indices.append(output_idx)
                            new_active_indices.append(output_idx)
                        else:
                            logger.warning(f"Sample {output_idx}: Found lookup end tag but couldn't extract content")
                            # import ipdb
                            # ipdb.set_trace()
                        curr_inputs[output_idx].extend(truncated_ids)
                        result_mask_list[output_idx].extend([1] * len(truncated_ids))
                        
                    elif stop_type == 'match':
                        match_content = self._extract_match_content(truncated_str)
                        if match_content:
                            match_queries.append(match_content)
                            match_indices.append(output_idx)
                            new_active_indices.append(output_idx)
                        else:
                            logger.warning(f"Sample {output_idx}: Found match end tag but couldn't extract content")
                        curr_inputs[output_idx].extend(truncated_ids)
                        result_mask_list[output_idx].extend([1] * len(truncated_ids))
                    
                    elif stop_type == 'search':
                        search_content = self._extract_search_content(truncated_str)
                        if search_content:
                            search_queries.append(search_content)
                            search_indices.append(output_idx)
                            new_active_indices.append(output_idx)
                        else:
                            logger.warning(f"Sample {output_idx}: Found search end tag but empty content")
                        
                        curr_inputs[output_idx].extend(truncated_ids)
                        result_mask_list[output_idx].extend([1] * len(truncated_ids))
                        
                    else:
                        # EOS, PAD, or max length reached - done generating
                        curr_inputs[output_idx].extend(truncated_ids)
                        result_mask_list[output_idx].extend([1] * len(truncated_ids))
                
                # Batch process match queries
                if match_queries:
                    logger.debug(f"Processing {len(match_queries)} match queries")
                    match_results = self.batch_match(match_queries)
                    for i, result in zip(match_indices, match_results):
                        result_text = (
                            f" {self.retrieval_config.match_result_start_tag} "
                            f"{result} "
                            f"{self.retrieval_config.match_result_end_tag}\n"
                        )
                        result_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        curr_inputs[i].extend(result_ids)
                        result_mask_list[i].extend([0] * len(result_ids))  # 0 for retrieval content
                
                # Batch process lookup queries
                if lookup_queries:
                    logger.debug(f"Processing {len(lookup_queries)} lookup queries")
                    lookup_results = self.batch_lookup(lookup_queries)
                    for i, result in zip(lookup_indices, lookup_results):
                        result_text = (
                            f" {self.retrieval_config.lookup_result_start_tag} "
                            f"{result} "
                            f"{self.retrieval_config.lookup_result_end_tag}\n"
                        )
                        result_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        curr_inputs[i].extend(result_ids)
                        result_mask_list[i].extend([0] * len(result_ids))  # 0 for retrieval content

                if search_queries:
                    logger.debug(f"Processing {len(search_queries)} search queries")
                    search_results = self.batch_search(search_queries)
                    
                    for i, result in zip(search_indices, search_results):
                        # Format: <result> {json_content} </result>
                        # result = "no result found."
                        result_text = (
                            f" {self.retrieval_config.search_result_start_tag} "
                            f"{result} "
                            f"{self.retrieval_config.search_result_end_tag}\n"
                        )
                        result_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        curr_inputs[i].extend(result_ids)
                        # 0 mask ensures search results don't affect training loss
                        result_mask_list[i].extend([0] * len(result_ids))
                
                retrieval_time = time.time() - start_time
                if match_queries or lookup_queries:
                    logger.debug(f"Retrieval (lookup/match) time: {retrieval_time:.2f}s")
                
                # Check length constraints and update active indices
                length_checked_active_indices = []
                for i in active_indices:
                    response_length = len(curr_inputs[i]) - len(init_inputs[i])
                    
                    if response_length >= self.config.response_length:
                        # Truncate to max length
                        curr_inputs[i], result_mask_list[i] = self._truncate_to_response_length(
                            curr_inputs[i],
                            init_inputs[i],
                            result_mask_list[i],
                            self.config.response_length
                        )
                    elif i in new_active_indices:
                        # Still active and has room to grow
                        curr_max_tokens[i] = self.config.response_length - response_length
                        length_checked_active_indices.append(i)
                
                active_indices = length_checked_active_indices
            
            if iteration >= self.retrieval_config.max_iterations:
                logger.warning(
                    f"Reached maximum iterations ({self.retrieval_config.max_iterations}). "
                    f"Some samples may not have completed generation."
                )
            
            # Final length check on all outputs
            for i in range(len(curr_inputs)):
                curr_inputs[i], result_mask_list[i] = self._truncate_to_response_length(
                    curr_inputs[i],
                    init_inputs[i],
                    result_mask_list[i],
                    self.config.response_length
                )
            
            # Collect final outputs
            output_ids_list = []
            for i, init_input in enumerate(init_inputs):
                response_ids = curr_inputs[i][len(init_input):]
                output_ids_list.append(response_ids)
        
        # Pad responses and masks to fixed length
        response = pad_2d_list_to_length(
            output_ids_list, 
            self.pad_token_id, 
            max_length=self.config.response_length
        ).to(idx.device)
        
        result_mask = pad_2d_list_to_length(
            result_mask_list,
            0,  # Pad with 0 (masked)
            max_length=self.config.response_length
        ).to(idx.device)
        
        # Build final sequence
        seq = torch.cat([idx, response], dim=-1)
        
        # Compute position IDs for response
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        
        # Handle qwen2vl mrope case (batch size, 4, seq len)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, position_ids.size(1), -1
            )
        
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # Compute attention mask for response
        response_attention_mask = get_response_mask(
            response_id=response, 
            eos_token=eos_token_id, 
            dtype=attention_mask.dtype
        )

        # import ipdb
        # ipdb.set_trace()
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Compute response mask: 1 for LLM generated tokens, 0 for retrieval/padding
        # This ensures retrieval content doesn't contribute to training loss
        response_mask = result_mask * response_attention_mask
        
        # Build output batch
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": response_mask,
            },
            batch_size=batch_size,
        )
        
        # Note: rollout_log_probs is not computed for search iterations
        # as the generation is split across multiple calls. If needed,
        # log probs should be recomputed by the actor model.
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


""" Modification End """

# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
