"""Coarse-to-Fine prototype head (CoFInAl-inspired).

Reformula la salida del Student como una jerarquía de prototipos: K_coarse
prototipos de calidad gruesa (e.g., 10 niveles de [0,1]) y K_fine prototipos
finos por cada coarse. La predicción se hace mediante similitud coseno con
prototipos aprendibles, y la pérdida es CrossEntropy contra el bin coarse +
fine de la etiqueta.

Para inferencia, se calcula la esperanza:
    score_pred = Σ_c p_coarse(c) · Σ_f p_fine(f|c) · score_bin(c, f)

Referencia: Zhou et al. CoFInAl (IJCAI 2024). https://arxiv.org/abs/2404.13999
Esta implementación es una adaptación ligera (no copia el método completo
del paper, que usa instruction alignment textual; aquí mantenemos sólo la
estructura de prototipos coarse→fine sin alineación textual).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    """
    Genera dos distribuciones jerárquicas: coarse (K_c bins) y fine (K_f bins
    por cada coarse). Devuelve un dict con `coarse_logits` [B, K_c],
    `fine_logits` [B, K_c, K_f] y `score_pred` [B] (esperanza).
    """

    def __init__(self,
                 in_features: int,
                 num_coarse: int = 10,
                 num_fine: int = 5,
                 dropout: float = 0.5,
                 score_min: float = 0.0,
                 score_max: float = 1.0):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.score_min = score_min
        self.score_max = score_max
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, 256)
        # prototipos: K_c × 256 para coarse, K_c × K_f × 256 para fine
        self.coarse_proto = nn.Parameter(torch.randn(num_coarse, 256) * 0.02)
        self.fine_proto = nn.Parameter(torch.randn(num_coarse, num_fine, 256) * 0.02)

    def _bin_centers(self, device) -> torch.Tensor:
        """Centros de bin combinado coarse_idx*K_f + fine_idx → score escalar."""
        coarse_centers = torch.linspace(
            self.score_min, self.score_max, self.num_coarse, device=device
        )  # [K_c]
        # offset fino dentro de cada coarse: distribuir K_f bins simétricamente
        coarse_width = (self.score_max - self.score_min) / max(self.num_coarse - 1, 1)
        fine_offsets = (torch.arange(self.num_fine, device=device).float() - (self.num_fine - 1) / 2.0) \
                       * (coarse_width / max(self.num_fine, 1))
        # centers[c, f] = coarse_centers[c] + fine_offsets[f]
        centers = coarse_centers.unsqueeze(1) + fine_offsets.unsqueeze(0)  # [K_c, K_f]
        return centers

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() == 4:
            x = x.unsqueeze(2)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = F.normalize(self.proj(x), dim=1)  # [B, 256]

        coarse_proto_n = F.normalize(self.coarse_proto, dim=1)         # [K_c, 256]
        fine_proto_n = F.normalize(self.fine_proto, dim=2)             # [K_c, K_f, 256]

        # logits coarse: cosine sim entre z y K_c prototipos
        coarse_logits = z @ coarse_proto_n.t()                          # [B, K_c]

        # logits fine por coarse: para cada coarse, similitud con K_f
        # z: [B, 256] -> [B, 1, 1, 256]
        z_e = z.unsqueeze(1).unsqueeze(1)
        fine_logits = (z_e * fine_proto_n.unsqueeze(0)).sum(dim=-1)     # [B, K_c, K_f]

        # esperanza para inferencia
        with torch.no_grad():
            p_coarse = F.softmax(coarse_logits, dim=1)                  # [B, K_c]
            p_fine = F.softmax(fine_logits, dim=2)                      # [B, K_c, K_f]
            centers = self._bin_centers(z.device)                        # [K_c, K_f]
            joint = p_coarse.unsqueeze(2) * p_fine                       # [B, K_c, K_f]
            score_pred = (joint * centers.unsqueeze(0)).sum(dim=(1, 2))  # [B]

        return {
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "score_pred": score_pred,
        }


def assign_bins(scores: torch.Tensor,
                num_coarse: int,
                num_fine: int,
                score_min: float = 0.0,
                score_max: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte scores escalares en (coarse_idx, fine_idx) para CrossEntropy."""
    rng = score_max - score_min
    norm = (scores - score_min) / rng                                    # [B] en [0,1]
    norm = norm.clamp(0.0, 1.0 - 1e-6)
    total_bins = num_coarse * num_fine
    flat_idx = (norm * total_bins).long()                                # [B] en [0, total_bins-1]
    coarse_idx = flat_idx // num_fine
    fine_idx = flat_idx % num_fine
    coarse_idx = coarse_idx.clamp(0, num_coarse - 1)
    fine_idx = fine_idx.clamp(0, num_fine - 1)
    return coarse_idx, fine_idx
