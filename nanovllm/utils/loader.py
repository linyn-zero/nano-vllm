import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):  # .safetensors 键值对
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # print(f"{weight_name}, {f.get_tensor(weight_name).shape}")            
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)  # 找到 Qwen3ForCausalLM(hf_config) 的需要 weight 的 param
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)  # 找到 param 的 weight_load
                    weight_loader(param, f.get_tensor(weight_name))  # 用 weight_load 导入权重
