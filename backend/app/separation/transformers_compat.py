from __future__ import annotations

"""Runtime compatibility patches for the vendored QuarkAudio-UniSE code.

The vendored LLM (vendor/unified-audio/QuarkAudio-UniSE/model/llm/llm.py) was
written against transformers ~4.4x and relies on two things that changed by
transformers 4.57:

1. ``LlamaModel._update_causal_mask`` (private) no longer exists — its
   replacement is ``transformers.masking_utils.create_causal_mask``.
2. ``LlamaDecoderLayer.forward`` now returns the hidden-states tensor
   directly instead of a tuple, so the vendored ``layer_outputs[0]`` would
   silently slice the batch dimension.

Both patches are applied only when the installed transformers lacks the old
API, so the vendored code keeps working unmodified against old and new
versions alike.
"""


def patch_vendored_unise() -> None:
    """Apply compat patches. Must run after `model` (vendored) is importable,
    but is safe to call before or after model construction and repeatedly."""
    from transformers.models.llama.modeling_llama import LlamaModel

    if hasattr(LlamaModel, "_update_causal_mask"):
        return  # old transformers: the vendored code path works as written

    from transformers.masking_utils import create_causal_mask

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        return create_causal_mask(
            config=self.config,
            input_embeds=input_tensor,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

    LlamaModel._update_causal_mask = _update_causal_mask

    from model.llm import llm as vendored_llm  # vendored package; caller sets sys.path

    def llm_forward(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        position_ids=None,
        cache_position=None,
        **kwargs,
    ):
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    vendored_llm.CustomLlamaModel.llm_forward = llm_forward
