import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from typing import Optional, Tuple, Union, List, Dict
import math
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
import logging
import torch.nn.functional as F
import torch.nn.init as init
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

@dataclass
class MoEConfig:
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_size: int = 4096
    capacity_factor: float = 1.25

class DeepSeekConfig(PretrainedConfig):
    model_type = "deepseek"
    
    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1.0e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        # DeepSeek specific configs
        moe_config: Optional[MoEConfig] = None,
        num_latent_heads: int = 4,  # For MHLA
        latent_size: int = 32,
        max_sequence_length: int = 512,  # New parameter
        compression_ratio: float = 0.125,  # New parameter for MHLA
        num_shared_experts: int = 0,  # New parameter for MoE
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        
        # DeepSeek specific attributes
        self.moe_config = moe_config or MoEConfig()
        self.num_latent_heads = num_latent_heads
        self.latent_size = latent_size
        self.max_sequence_length = max_sequence_length
        self.compression_ratio = compression_ratio
        self.num_shared_experts = num_shared_experts
        self.latent_size = int(max_sequence_length * compression_ratio)
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def _init_weights(module, std=0.041666666666666664):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention module from DeepSeek"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_latent_heads = config.num_key_value_heads
        self.latent_size = config.latent_size
        
        # Initialize parameters first
        self.latent = nn.Parameter(torch.empty(1, self.num_key_value_heads, self.latent_size, self.head_dim))
        self.latent_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.latent_o_proj = nn.Linear(self.num_key_value_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Now we can safely get the device
        device = next(self.parameters()).device
        
        # Create config object for LlamaRotaryEmbedding with all required attributes
        rotary_config = type('RotaryConfig', (), {
            'max_position_embeddings': config.max_position_embeddings,
            'head_dim': self.head_dim,
            'rope_theta': config.rope_theta,
            'rope_scaling': None,
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'device': device
        })()
        
        # Initialize LlamaRotaryEmbedding with config object
        self.rotary_emb = LlamaRotaryEmbedding(rotary_config)
        
        self._init_weights()
    
    def _init_weights(self):
        init.normal_(self.latent, std=0.02)
    
    def forward(self, hidden_states, cos=None, sin=None, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        # Regular attention projections with proper head dimensions
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Create position ids for rotary embeddings
        position_ids = torch.arange(seq_length, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get rotary embeddings from LlamaRotaryEmbedding
        cos, sin = self.rotary_emb(x=hidden_states, position_ids=position_ids)
        
        # Apply rotary embeddings using Llama's implementation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Apply rotary embeddings with proper arguments
        q, k = apply_rotary_pos_emb(q=q, k=k, cos=cos, sin=sin)
        
        # Rest of the attention computation
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        
        # Repeat k,v for regular attention
        k_regular = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        v_regular = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        
        # Regular attention scores
        attn_scores = torch.matmul(q, k_regular.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v_regular)
        
        # Process latent attention
        latent = self.latent.expand(batch_size, -1, -1, -1)  # [batch, num_key_value_heads, latent_size, head_dim]
        latent_flat = latent.reshape(-1, self.head_dim)
        latent_proj = self.latent_proj(latent_flat)
        latent = latent_proj.reshape(batch_size, self.num_key_value_heads, self.latent_size, self.head_dim)
        
        # Compute latent attention scores
        latent_scores = torch.matmul(latent, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            latent_mask = attention_mask[:, :self.num_key_value_heads, -1:, :]
            latent_mask = latent_mask.expand(-1, -1, self.latent_size, -1)
            latent_scores = latent_scores + latent_mask
        latent_probs = F.softmax(latent_scores, dim=-1)
        latent_out = torch.matmul(latent_probs, v)  # [batch, num_key_value_heads, latent_size, head_dim]
        
        # Process outputs
        attn_out = attn_out.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_out = attn_out.reshape(batch_size, seq_length, self.hidden_size)
        attn_out = self.o_proj(attn_out)
        
        # Process latent output
        latent_out = latent_out.permute(0, 2, 1, 3).contiguous()  # [batch, latent_size, num_key_value_heads, head_dim]
        latent_out = latent_out.reshape(batch_size, self.latent_size, self.num_key_value_heads * self.head_dim)
        latent_out = self.latent_o_proj(latent_out)  # Project to hidden_size
        
        # Average and expand latent output
        latent_out = latent_out.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]
        latent_out = latent_out.expand(-1, seq_length, -1)  # [batch, seq_len, hidden_size]
        
        return attn_out + latent_out


class MoELayer(nn.Module):
    """Mixture of Experts layer with loss-free load balancing"""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_config.num_experts
        self.num_experts_per_tok = config.moe_config.num_experts_per_tok
        self.expert_size = config.moe_config.expert_size
        self.hidden_size = config.hidden_size
        self.capacity_factor = config.moe_config.capacity_factor
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.expert_size),
                nn.SiLU(),
                nn.Linear(self.expert_size, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
        # Router for expert selection
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Load balancing parameters
        self.router_z_loss_coef = 0.001  # coefficient for auxiliary loss
        self.router_aux_loss_coef = 0.001  # coefficient for z-loss
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch, seq, num_experts]
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs, 
            k=self.num_experts_per_tok,
            dim=-1
        )
        
        # Normalize probabilities of selected experts
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute load balancing auxiliary loss
        # Ideal load would be uniform distribution across experts
        ideal_load = torch.ones_like(router_probs) / self.num_experts
        aux_loss = F.kl_div(
            router_probs.log(), 
            ideal_load,
            reduction='batchmean'
        ) * self.router_aux_loss_coef
        
        # Compute router z-loss to encourage sharp routing decisions
        z_loss = torch.mean(torch.square(router_logits)) * self.router_z_loss_coef
        
        # Combine losses
        router_loss = aux_loss + z_loss
        
        # Dispatch to experts
        expert_outputs = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            # Get tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_outputs[expert_mask] += expert_output
                
        return expert_outputs, router_loss


class DeepSeekBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadLatentAttention(config)
        self.moe = MoELayer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, cos, sin, attention_mask=None):
        # Pre-norm for attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        
        # Pre-norm for MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, router_loss


class DeepSeekModel(PreTrainedModel):
    config_class = DeepSeekConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create causal mask for attention
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to float and expand dims
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
            
            # Combine masks
            combined_mask = attention_mask.unsqueeze(-2) + causal_mask.to(dtype=torch.float32)
        else:
            combined_mask = causal_mask.to(dtype=torch.float32)
            combined_mask = (1.0 - combined_mask) * torch.finfo(torch.float32).min
            combined_mask = combined_mask.unsqueeze(0)  # Add batch dimension
        
        # Forward through layers with None for cos, sin since they're computed in attention
        for layer in self.layers:
            hidden_states, router_loss = layer(
                hidden_states,
                cos=None,
                sin=None,
                attention_mask=combined_mask
            )
        
        hidden_states = self.norm(hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, seq_length, seq_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        
        # Convert to float and apply causal mask
        expanded_attn_mask = expanded_attn_mask.to(dtype=torch.float32)
        if combined_attention_mask is not None:
            expanded_attn_mask = expanded_attn_mask + combined_attention_mask.to(dtype=torch.float32)
        
        # Convert to attention mask format (0 for attend, large negative for mask)
        expanded_attn_mask = (1.0 - expanded_attn_mask) * torch.finfo(torch.float32).min

        return expanded_attn_mask


class DeepSeekForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DeepSeekConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.main_input_name = "input_ids"
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        # Update lm_head weights if using weight tying
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        # Update input embeddings if using weight tying
        if self.config.tie_word_heads:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Create position_ids from attention_mask
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


def _make_causal_mask(
    input_ids_shape: torch.Size,
    device: torch.device,
    past_key_values_length: int = 0,
):
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] >= seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # Handle case where mask might have extra dimensions
    if mask.dim() > 2:
        mask = mask.squeeze(1).squeeze(1)
    
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_length, src_length)
    return expanded_mask 