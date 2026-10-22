import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        causal_mask = torch.ones(
            length, length, dtype=torch.bool, device=x.device
        ).triu(1)
        result, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return self.dropout(result)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.n_layers))
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        logits = self.output(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generation expects input_ids with shape (1, sequence)")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.max_seq_len :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = torch.topk(
                        next_logits, min(top_k, next_logits.shape[-1])
                    ).values[:, -1, None]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, -torch.inf
                    )
                next_token = torch.multinomial(
                    torch.softmax(next_logits, dim=-1), num_samples=1
                )
                input_ids = torch.cat((input_ids, next_token), dim=1)
                if eos_id is not None and int(next_token.item()) == eos_id:
                    break
        finally:
            self.train(was_training)
        return input_ids
