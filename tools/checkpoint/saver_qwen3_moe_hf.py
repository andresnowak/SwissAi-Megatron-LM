"""HF saver for Apertus and Llama models."""

import sys
import os
import gc
import json
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from abc import ABC, abstractmethod

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3MoeForCausalLM, Qwen3MoeConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from schema_hf import get_qwen3_moe_schema

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir,
                    os.path.pardir)))
try:
    from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
except ModuleNotFoundError:
    print("Unable to import Megatron. Exiting.")
    exit(1)

def add_arguments(parser):
    group = parser.add_argument_group(title="Llama HF saver.")
    group.add_argument(
        "--hf-tokenizer",
        type=str,
        default=None,
        help="Example: epfl-llm/meditron-70b",
    )
    group.add_argument(
        "--check-eq-hf",
        type=str,
        default=None,
        help="check equality with HF model, e.g. epfl-llm/meditron-70b",
    )
    group.add_argument(
        "--save-chat-model",
        action='store_true',
        help="flag to save chat model or not",
    )


def perform_check(
    state_dict: dict[str, torch.Tensor], ref_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Given a reference state dict, check that state_dict is equal to it
    then pop the keys from ref_state_dict
    """
    for key in state_dict:
        assert torch.equal(ref_state_dict[key], state_dict[key])
        ref_state_dict.pop(key)
    return ref_state_dict


def save_layer(
    state_dict: dict[str, torch.Tensor],
    index_dict: dict,
    dir_path: str,
    filename: str,
    check_reference: bool = False,
    ref_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """check state dict against a reference one if needed
    update index_dict
    save state dict
    """
    if check_reference:
        ref_state_dict = perform_check(state_dict, ref_state_dict)
    for layer_name, weight_matrix in state_dict.items():
        index_dict["weight_map"][layer_name] = filename
        index_dict["metadata"]["total_size"] += weight_matrix.numel()
    print(f"saving state dict to {dir_path}/{filename}")
    torch.save(state_dict, f"{dir_path}/{filename}")
    return index_dict, ref_state_dict


def pad_weight(md, orig_word_embed, true_vocab_size):
    if true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        md.checkpoint_args.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, md.checkpoint_args)

        # Cut out extra padding we don't need
        if orig_vocab_size > md.checkpoint_args.padded_vocab_size:
            full_word_embed = orig_word_embed[0:md.checkpoint_args.padded_vocab_size,:]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < md.checkpoint_args.padded_vocab_size:
            padding_size = md.checkpoint_args.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
            "If you've changed the tensor parallel size this could cause problems.")
        md.checkpoint_args.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed
    return full_word_embed


class HFCheckpointSaver(ABC):
    def __init__(self, args, queue: mp.Queue):
        self.args = args
        self.queue = queue

        self.md = self.queue_get()

        self.verify_compatibility_args(self.md)

    def queue_get(self, name=None) -> dict:
        val = self.queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and self.args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(self, msg):
        if not self.args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def receive_model(self):
        pass

    @abstractmethod
    def receive_lm(self, schema):
        pass

    @abstractmethod
    def verify_compatibility_args(self, md):
        pass


class HFCheckpointSaverQwen3MoE(HFCheckpointSaver):
    def verify_compatibility_args(self, md):
        ### Verify compatibility of args
        if not hasattr(md, "checkpoint_args"):
            raise ValueError("missing checkpoint_args in metadata")
        if md.model_type != "GPT":
            raise ValueError("wrong model_type in metadata. must be GPT")
        if md.checkpoint_args.position_embedding_type != "rope":
            raise ValueError("Qwen model must use RoPE")
        if md.checkpoint_args.normalization != "RMSNorm":
            raise ValueError("Qwen model must use RMSNorm")
        if not md.checkpoint_args.disable_bias_linear:
            raise ValueError("Qwen model must not use linear bias")
        if not md.checkpoint_args.swiglu:
            raise ValueError("Qwen model must use swiglu")

    def _split_qkv_weights(self, qkv_weight):
        """Split packed QKV weight into separate Q, K, V tensors for HF format."""
        # Qwen3-MoE uses GQA (grouped query attention)
        head_size = self.md.hidden_size // self.md.num_attention_heads
        num_query_groups = self.md.checkpoint_args.num_query_groups
        heads_per_group = self.md.num_attention_heads // num_query_groups
        qkv_total_heads = self.md.num_attention_heads + 2 * num_query_groups

        # Reshape to [qkv_total_heads, head_size, hidden_size]
        qkv_weight = qkv_weight.reshape([qkv_total_heads, head_size, self.md.hidden_size])

        # Create slices for Q, K, V
        q_slice = torch.cat([
            torch.arange(
                (heads_per_group + 2) * i,
                (heads_per_group + 2) * i + heads_per_group,
            )
            for i in range(num_query_groups)
        ])
        k_slice = torch.arange(heads_per_group, qkv_total_heads, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_heads, (heads_per_group + 2))

        q_weight = qkv_weight[q_slice].reshape(-1, self.md.hidden_size)
        k_weight = qkv_weight[k_slice].reshape(-1, self.md.hidden_size)
        v_weight = qkv_weight[v_slice].reshape(-1, self.md.hidden_size)

        return q_weight, k_weight, v_weight

    def combine_experts_to_3d(self, expert_weights):
        """
        Combine per-expert weights into 3D tensor format expected by HF Qwen3-MoE.
        Input: [num_experts, ...] tensor or list of tensors
        Output: 3D tensor [num_experts, ...]
        """
        if isinstance(expert_weights, list):
            return torch.stack(expert_weights, dim=0)
        return expert_weights

    def receive_lm(self, schema):
        # Receive embeddings
        embeddings_msg = self.queue_get("embeddings")
        params_dict = {}
        # Store embeddings for potential use in lm_head if tied
        self.embeddings = pad_weight(self.md, embeddings_msg["word embeddings"], self.md.true_vocab_size)
        params_dict["word_embeddings"] = self.embeddings
        schema.set(self.state_dict, params_dict)

        # Receive layers
        for i in range(self.md.num_layers):
            message = self.queue_get(f"transformer layer {i}")
            params_dict = {}

            # Layer norms
            params_dict["input_norm_weight"] = message["input norm weight"]
            params_dict["post_norm_weight"] = message["post norm weight"]

            if self.md.norm_has_bias:
                params_dict["input_norm_bias"] = message["input norm bias"]
                params_dict["post_norm_bias"] = message["post norm bias"]

            # Split QKV into separate Q, K, V
            q_weight, k_weight, v_weight = self._split_qkv_weights(message["qkv weight"])
            params_dict["q_proj_weight"] = q_weight
            params_dict["k_proj_weight"] = k_weight
            params_dict["v_proj_weight"] = v_weight

            # Dense/output projection
            params_dict["dense_weight"] = message["dense weight"]

            # Q/K norms (Qwen3 specific)
            if hasattr(self.md.checkpoint_args, 'qk_layernorm') and self.md.checkpoint_args.qk_layernorm:
                params_dict["q_norm_weight"] = message["q norm weight"]
                params_dict["k_norm_weight"] = message["k norm weight"]

            # Router
            params_dict["router_weight"] = message["router weight"]

            # Expert weights - need to collect from per-expert keys and stack them
            # The message will have keys like "mlp l0 weight W.0", "mlp l0 weight W.1", etc.
            num_experts = self.md.num_experts

            # Collect gate and up weights for each expert
            expert_gates = []
            expert_ups = []
            expert_downs = []

            for expert_idx in range(num_experts):
                # SwiGLU: separate W (gate) and V (up) projections
                expert_gates.append(message[f"mlp l0 weight W.{expert_idx}"])
                expert_ups.append(message[f"mlp l0 weight V.{expert_idx}"])

                expert_downs.append(message[f"mlp l1 weight.{expert_idx}"])

            # Stack into [num_experts, ...] tensors
            expert_gate_stacked = torch.stack(expert_gates, dim=0)  # [num_experts, intermediate, hidden]
            expert_up_stacked = torch.stack(expert_ups, dim=0)    # [num_experts, intermediate, hidden]
            # Concatenate gate and up into gate_up_proj: [num_experts, 2*intermediate, hidden]
            expert_gate_up = torch.cat([expert_gate_stacked, expert_up_stacked], dim=1)
            params_dict["experts_weight_gate_up"] = expert_gate_up

            expert_down_stacked = torch.stack(expert_downs, dim=0)  # [num_experts, hidden, intermediate]
            params_dict["experts_weight_down"] = expert_down_stacked

            schema.set_layer(self.state_dict, i, params_dict)

        # Final layer
        final_norm_msg = self.queue_get("final norm")
        params_dict = {"final_norm": final_norm_msg["weight"]}

        if not self.md.checkpoint_args.untie_embeddings_and_output_weights:  # tied embeddings and lm-head
            params_dict["lm_head"] = self.embeddings
        else:
            params_dict["lm_head"] = pad_weight(self.md, self.queue_get("output layer")["weight"], self.md.true_vocab_size)

        schema.set(self.state_dict, params_dict)

        
    def receive_model(self):
        language_model_prefix = ""
        language_layer_prefix = "model.layers"
        language_schema = get_qwen3_moe_schema(
            prefix=language_model_prefix,
            layer_prefix=language_layer_prefix,
        )

        self.receive_lm(language_schema)

    def save(self):
        self.md = self.queue_get()
        
        self.state_dict = {}

        self.receive_model()

        self.save_state_dict_to_hf_checkpoint()

        print("Done!")

    def save_state_dict_to_hf_checkpoint(self):
        torch_dtype = torch.float32
        if self.md.checkpoint_args.bf16:
            torch_dtype = torch.bfloat16
            if self.md.checkpoint_args.fp16:
                raise ValueError("bf16 and fp16 cannot be both set.")
        elif self.md.checkpoint_args.fp16:
            torch_dtype = torch.float16
            if self.md.checkpoint_args.bf16:
                raise ValueError("bf16 and fp16 cannot be both set.")

        ### init
        save_dir = Path(self.args.save_dir)
        save_dir.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix=str(save_dir/"tmp")) as tmp_save_dir:
            index_dict = {
                "weight_map": {},
                "metadata": {"total_size": 0},
            }
            tokenizer = None
            ref_state_dict = None

            ### prepare a reference model if needed
            if self.args.check_eq_hf:
                print(f"preparing checks with given HF model {self.args.check_eq_hf}")
                ref_model = AutoModelForCausalLM.from_pretrained(self.args.check_eq_hf)
                ref_state_dict = ref_model.state_dict()
            
            ### save tokenizer conf files
            if self.args.hf_tokenizer:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.args.hf_tokenizer)
                print(f"saving tokenizer to {self.args.save_dir}")
                tokenizer.save_pretrained(self.args.save_dir)
            elif self.md.checkpoint_args.tokenizer_type == "HuggingFaceTokenizer":
                from transformers import AutoTokenizer
        
                tokenizer = AutoTokenizer.from_pretrained(self.md.checkpoint_args.tokenizer_model)
                print(f"saving tokenizer to {self.args.save_dir}")
                tokenizer.save_pretrained(self.args.save_dir)
            else:
                print("no HF tokenizer specified, skipping saving tokenizer")
            
            qwen3_moe_conf = Qwen3MoeConfig(
                vocab_size=self.md.checkpoint_args.padded_vocab_size,
                hidden_size=self.md.hidden_size,
                intermediate_size=self.md.checkpoint_args.ffn_hidden_size,
                num_hidden_layers=self.md.num_layers,
                num_attention_heads=self.md.num_attention_heads,
                num_key_value_heads=self.md.checkpoint_args.num_query_groups,
                hidden_act="silu", # SwiGLU
                max_position_embeddings=self.md.checkpoint_args.max_position_embeddings,
                rope_theta=self.md.checkpoint_args.rotary_base,
                rope_scaling={
                    "rope_type": "llama_3",
                    "factor": self.md.checkpoint_args.rope_scaling_factor,
                    "original_max_position_embeddings": self.md.checkpoint_args.max_position_embeddings,
                    "high_freq_factor": 4.0,
                    "low_freq_fraction": 1.0,
                } if self.md.checkpoint_args.rope_scaling else None,
                attention_bias=self.md.checkpoint_args.add_qkv_bias if hasattr(self.md.checkpoint_args, "add_qkv_bias") else False,
                mlp_bias=self.md.checkpoint_args.add_bias_linear,
                num_experts=self.md.num_experts,
                num_experts_per_tok=self.md.checkpoint_args.moe_router_topk,
                tie_word_embeddings=not self.md.checkpoint_args.untie_embeddings_and_output_weights,
                rms_norm_eps=self.md.checkpoint_args.norm_epsilon,
                model_type="qwen3_moe",
                architectures=["Qwen3MoeForCausalLM"],
                qk_norm=self.md.checkpoint_args.qk_layernorm if hasattr(self.md.checkpoint_args, "qk_layernorm") else False,
                post_norm=self.md.checkpoint_args.post_layernorm if hasattr(self.md.checkpoint_args, "post_layernorm") else False,
                attention_dropout=self.md.checkpoint_args.attention_dropout,
            )

            if self.args.hf_tokenizer or self.md.checkpoint_args.tokenizer_type == "HuggingFaceTokenizer":
                qwen3_moe_conf.pad_token_id = tokenizer.pad_token_id
                qwen3_moe_conf.bos_token_id = tokenizer.bos_token_id
                qwen3_moe_conf.eos_token_id = tokenizer.eos_token_id

            print(f"saving config.json to {tmp_save_dir}")
            qwen3_moe_conf.save_pretrained(tmp_save_dir)

            ### save index dict
            index_dict = {
                "weight_map": {},
                "metadata": {"total_size": 0},
            }

            # Save each layer as a separate file
            for key, value in self.state_dict.items():
                # Determine which file this weight should go to
                if "embed_tokens" in key:
                    filename = "pytorch_model-embedding.bin"
                elif "lm_head" in key or "model.norm" in key:
                    filename = "pytorch_model-lm-head.bin"
                else:
                    # Extract layer number from key like "model.layers.0.xxx"
                    layer_num = int(key.split(".")[2])
                    filename = f"pytorch_model-{layer_num + 1}.bin"

                index_dict["weight_map"][key] = filename
                index_dict["metadata"]["total_size"] += value.numel()

            # Group weights by filename
            files_dict = {}
            for key, value in self.state_dict.items():
                filename = index_dict["weight_map"][key]
                if filename not in files_dict:
                    files_dict[filename] = {}
                files_dict[filename][key] = value

            # Save each file
            for filename, weights in files_dict.items():
                print(f"saving state dict to {tmp_save_dir}/{filename}")
                torch.save(weights, f"{tmp_save_dir}/{filename}")

            # Update total size based on dtype
            index_dict["metadata"]["total_size"] *= {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
            }[torch_dtype]

            print(f"saving {tmp_save_dir}/pytorch_model.bin.index.json")
            with open(f"{tmp_save_dir}/pytorch_model.bin.index.json", "w") as f:
                json.dump(index_dict, f)

            ### load then save model in HF format
            # Make space so we can load the model properly now.
            del self.state_dict
            gc.collect()

            print(f"Loading the converted pytorch checkpoint in a Qwen3-MoE HF model from {tmp_save_dir}")
            model = Qwen3MoeForCausalLM.from_pretrained(
                str(tmp_save_dir), torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )

        # Avoid saving this as part of the config.
        del model.config._name_or_path
        model.config.torch_dtype = torch_dtype
        print(f"Saving in the Transformers safe tensors format to {self.args.save_dir}")
        model.save_pretrained(self.args.save_dir, safe_serialization=True)

        ### save chat config
        generation_config = (
            GenerationConfig(
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                bos_token_id=qwen3_moe_conf.bos_token_id if hasattr(qwen3_moe_conf, 'bos_token_id') else None,
                eos_token_id=qwen3_moe_conf.eos_token_id if hasattr(qwen3_moe_conf, 'eos_token_id') else None,
            )
            if self.args.save_chat_model
            else GenerationConfig(
                _from_model_config=True,
                bos_token_id=qwen3_moe_conf.bos_token_id if hasattr(qwen3_moe_conf, 'bos_token_id') else None,
                eos_token_id=qwen3_moe_conf.eos_token_id if hasattr(qwen3_moe_conf, 'eos_token_id') else None,
            )
        )
        print(f"Saving chat config to {self.args.save_dir}")
        generation_config.save_pretrained(self.args.save_dir)


def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().
    """
    saver = HFCheckpointSaverQwen3MoE(args, queue)
    try:
        saver.save()
    except Exception as e:
        raise e