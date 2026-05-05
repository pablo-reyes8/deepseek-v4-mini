from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mHC_residuals_utils import collapse_residual_stream, expand_residual_stream


def normalize_devices(devices: Optional[list[str | torch.device]]) -> list[torch.device]:
    if not devices:
        return [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    return [torch.device(device) for device in devices]


def infer_auto_balance(n_layers: int, n_devices: int) -> list[int]:
    if n_layers <= 0:
        raise ValueError(f"n_layers must be > 0, got {n_layers}")
    if n_devices <= 0:
        raise ValueError(f"n_devices must be > 0, got {n_devices}")
    base = n_layers // n_devices
    rem = n_layers % n_devices
    return [base + (i < rem) for i in range(n_devices)]


def build_block_device_map(
    n_layers: int,
    devices: list[str | torch.device],
    balance: Optional[list[int]] = None,
) -> list[torch.device]:
    normalized = normalize_devices(devices)
    if balance is None:
        balance = infer_auto_balance(n_layers, len(normalized))
    if len(balance) != len(normalized):
        raise ValueError(
            f"balance length must match devices length. Got {len(balance)} and {len(normalized)}"
        )
    if sum(balance) != n_layers:
        raise ValueError(f"sum(balance) must equal n_layers={n_layers}, got {sum(balance)}")
    if any(x <= 0 for x in balance):
        raise ValueError(f"balance entries must be > 0, got {balance}")

    block_devices: list[torch.device] = []
    for device, count in zip(normalized, balance):
        block_devices.extend([device] * int(count))
    return block_devices


class ModelParallelDeepSeekV4LM(nn.Module):
    """Layerwise educational model-parallel wrapper for DeepSeekV4LM.

    This is not tensor parallelism. It places whole blocks on devices and moves
    activations between them. It is intentionally simple and testable on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        devices: Optional[list[str | torch.device]] = None,
        balance: Optional[list[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.devices = normalize_devices(devices)
        self.block_devices = build_block_device_map(
            n_layers=self.config.n_layers,
            devices=self.devices,
            balance=balance,
        )

        self.input_device = self.block_devices[0]
        self.output_device = self.block_devices[-1]
        self._place_modules()

    def _place_modules(self) -> None:
        self.model.embedding.to(self.input_device)
        for block, device in zip(self.model.blocks, self.block_devices):
            block.to(device)
        if getattr(self.model, "mhc_readout", None) is not None:
            self.model.mhc_readout.to(self.output_device)
        self.model.final_norm.to(self.output_device)
        self.model.lm_head.to(self.output_device)
        if getattr(self.model, "mtp_head", None) is not None:
            self.model.mtp_head.to(self.output_device)

    def _to_block_device(self, value: Optional[torch.Tensor], device: torch.device):
        if value is None:
            return None
        return value.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mtp_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        return_aux: bool = False,
        need_weights: bool = False,
    ) -> dict[str, Any]:
        B, T = self.model._validate_input_ids(input_ids)
        del B, T

        input_ids = input_ids.long()
        if labels is not None:
            self.model._validate_labels(labels, input_ids)
            labels = labels.long()

        attention_mask = self.model._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        input_ids_first = input_ids.to(self.input_device)
        x = self.model.embedding(input_ids_first)
        block_aux_list = []

        if self.model.use_mhc:
            X = expand_residual_stream(
                x,
                n_hc=self.config.n_hc,
                mode=self.config.mhc_expand_mode,
            )
            for block, device in zip(self.model.blocks, self.block_devices):
                X = X.to(device)
                block_input_ids = input_ids.to(device)
                block_attention_mask = self._to_block_device(attention_mask, device)
                block_position_ids = self._to_block_device(position_ids, device)

                if return_aux or need_weights or self.config.ffn_type == "moe":
                    X, block_aux = block(
                        X,
                        attention_mask=block_attention_mask,
                        position_ids=block_position_ids,
                        start_pos=start_pos,
                        input_ids=block_input_ids,
                        return_aux=return_aux,
                        need_weights=need_weights,
                        collect_moe_aux=self.config.ffn_type == "moe",
                    )
                    block_aux_list.append(block_aux)
                else:
                    X = block(
                        X,
                        attention_mask=block_attention_mask,
                        position_ids=block_position_ids,
                        start_pos=start_pos,
                        input_ids=block_input_ids,
                        return_aux=False,
                        need_weights=False,
                        collect_moe_aux=False,
                    )

            X = X.to(self.output_device)
            if self.config.mhc_collapse_mode == "readout":
                x = self.model.mhc_readout(X)
            else:
                x = collapse_residual_stream(X, mode=self.config.mhc_collapse_mode)

        else:
            for block, device in zip(self.model.blocks, self.block_devices):
                x = x.to(device)
                block_input_ids = input_ids.to(device)
                block_attention_mask = self._to_block_device(attention_mask, device)
                block_position_ids = self._to_block_device(position_ids, device)

                if return_aux or need_weights or self.config.ffn_type == "moe":
                    x, block_aux = block(
                        x,
                        attention_mask=block_attention_mask,
                        position_ids=block_position_ids,
                        start_pos=start_pos,
                        input_ids=block_input_ids,
                        return_aux=return_aux,
                        need_weights=need_weights,
                        collect_moe_aux=self.config.ffn_type == "moe",
                    )
                    block_aux_list.append(block_aux)
                else:
                    x = block(
                        x,
                        attention_mask=block_attention_mask,
                        position_ids=block_position_ids,
                        start_pos=start_pos,
                        input_ids=block_input_ids,
                        return_aux=False,
                        need_weights=False,
                        collect_moe_aux=False,
                    )

        x = x.to(self.output_device)
        hidden_states = self.model.final_norm(x)
        logits = self.model.lm_head(hidden_states)

        labels_out = labels.to(self.output_device) if labels is not None else None
        attention_mask_out = attention_mask.to(self.output_device) if attention_mask is not None else None

        lm_loss = None
        if labels_out is not None:
            lm_loss = self.model._compute_lm_loss(
                logits=logits,
                labels=labels_out,
                attention_mask=attention_mask_out,
            )

        mtp_loss = None
        mtp_outputs = None
        if self.model.use_mtp:
            if mtp_labels is None and labels_out is not None:
                from src.deepseek_mtp import build_mtp_labels

                mtp_labels = build_mtp_labels(
                    input_ids=input_ids.to(self.output_device),
                    mtp_depth=self.config.mtp_depth,
                    ignore_index=self.model.ignore_index,
                    pad_token_id=self.model.pad_token_id,
                )
            elif mtp_labels is not None:
                mtp_labels = mtp_labels.to(self.output_device)

            mtp_outputs = self.model.mtp_head(
                hidden_states,
                mtp_labels=mtp_labels,
                return_aux=return_aux,
            )
            mtp_loss = mtp_outputs["mtp_loss"]

        moe_aux_loss = self.model._collect_moe_aux_loss(
            block_aux_list=block_aux_list,
            device=logits.device,
            dtype=logits.dtype,
        )
        has_moe_aux_loss = bool(
            self.config.ffn_type == "moe"
            and (
                self.config.balance_loss_weight > 0
                or self.config.sequence_balance_loss_weight > 0
            )
        )
        if not has_moe_aux_loss:
            moe_aux_loss = None

        loss = None
        if lm_loss is not None:
            loss = lm_loss
            if mtp_loss is not None:
                loss = loss + mtp_loss.to(dtype=loss.dtype)
            if moe_aux_loss is not None:
                loss = loss + moe_aux_loss.to(dtype=loss.dtype)

        aux: dict[str, Any] = {}
        if return_aux or need_weights:
            aux["blocks"] = block_aux_list
            aux["labels_are_shifted"] = self.model.labels_are_shifted
            aux["ignore_pad_token_in_loss"] = self.model.ignore_pad_token_in_loss
            if mtp_outputs is not None:
                aux["mtp"] = mtp_outputs.get("aux", {})
            if attention_mask_out is not None:
                aux["attention_mask"] = attention_mask_out

        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "moe_aux_loss": moe_aux_loss,
            "hidden_states": hidden_states if return_aux else None,
            "aux": aux,
        }


def wrap_model_parallel(
    model: nn.Module,
    devices: Optional[list[str | torch.device]] = None,
    balance: Optional[list[int]] = None,
) -> ModelParallelDeepSeekV4LM:
    return ModelParallelDeepSeekV4LM(model=model, devices=devices, balance=balance)
