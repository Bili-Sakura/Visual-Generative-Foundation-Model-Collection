from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper

from .backbone_diffmoe import DiT as DiffMoEBackbone
from .backbone_dit import DiT as DiTBackbone
from .backbone_ecdit import DiT as ECDiTBackbone
from .backbone_promoe_ec import DiT as ProMoEECBackbone
from .backbone_promoe_tc import DiT as ProMoETCBackbone
from .backbone_tcdit import DiT as TCDiTBackbone
from .modeling_promoe_common import AttrDict


@dataclass
class ProMoETransformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor
    loss_strategy: Optional[str] = None
    layer_idx_list: Optional[Tuple[int, ...]] = None
    ones_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    pred_c_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    capacity_pred_loss_weight: Optional[float] = None


_BACKBONES = {
    "dit": DiTBackbone,
    "tcdit": TCDiTBackbone,
    "ecdit": ECDiTBackbone,
    "diffmoe": DiffMoEBackbone,
    "promoe_tc": ProMoETCBackbone,
    "promoe_ec": ProMoEECBackbone,
}


class ProMoETransformer2DModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, architecture: str = "promoe_tc", model_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if architecture not in _BACKBONES:
            raise ValueError(f"Unsupported architecture: {architecture}. Valid: {sorted(_BACKBONES)}")
        model_config = model_config or {}
        self.architecture = architecture
        self.model_config = model_config
        self.backbone = _BACKBONES[architecture](**self._prepare_config(model_config))
        self.in_channels = getattr(self.backbone, "in_channels", model_config.get("in_channels", 4))
        self.out_channels = getattr(self.backbone, "out_channels", model_config.get("in_channels", 4))

    def _prepare_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        prepared = {}
        for key, value in model_config.items():
            prepared[key] = AttrDict.from_data(value)
        return prepared

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.LongTensor] = None,
        context: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ProMoETransformer2DModelOutput, Tuple[torch.Tensor, ...]]:
        labels = class_labels if class_labels is not None else context
        if labels is None:
            raise ValueError("Either `class_labels` or `context` must be provided.")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=hidden_states.dtype)
        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype).flatten()
        if timestep.numel() == 1:
            timestep = timestep.repeat(labels.shape[0])

        sample = self.backbone(hidden_states, timestep, labels, **kwargs)
        if isinstance(sample, tuple):
            if len(sample) == 6 and sample[1] == "Capacity_Pred":
                output = ProMoETransformer2DModelOutput(
                    sample=sample[0],
                    loss_strategy=sample[1],
                    layer_idx_list=tuple(sample[2]),
                    ones_list=tuple(sample[3]),
                    pred_c_list=tuple(sample[4]),
                    capacity_pred_loss_weight=float(sample[5]),
                )
            else:
                output = ProMoETransformer2DModelOutput(sample=sample[0])
        else:
            output = ProMoETransformer2DModelOutput(sample=sample)

        if not return_dict:
            if output.loss_strategy is None:
                return (output.sample,)
            return (
                output.sample,
                output.loss_strategy,
                output.layer_idx_list,
                output.ones_list,
                output.pred_c_list,
                output.capacity_pred_loss_weight,
            )
        return output
