# Plan Fase 9 — Robustez empírica (alineado con asesor)

> Pivote final tras feedback del asesor del 2026-05-05. La tesis se reformula
> como **"Pipeline liviano para AQA basado en arquitecturas modernas de bajo
> costo computacional"**. La contribución central es el pipeline (no la KD,
> no MUSDL, no SD-KD). Las fases 7 y 8 se preservan como evidencia de
> respaldo, pero el énfasis se mueve a robustez empírica del pipeline.
>
> Documento maestro del asesor: `recomendaciones y plan a seguir segun mi
> asesor.txt`.

---

## Cambios estructurales aprobados

- **Título nuevo:** *"Pipeline liviano para Action Quality Assessment basado
  en arquitecturas modernas de bajo costo computacional"*.
- **Contribución central:** Students livianos modernos (TSM-MobileNetV2 y
  MobileNetV3) con pipeline moderno alcanzan rendimiento competitivo
  respecto al Teacher I3D, validado en tres dominios.
- **KD pasa a análisis marginal:** los resultados negativos de la fase 7
  refuerzan que la KD no es necesaria con pipeline fuerte.
- **Descartado de fase 8:** MUSDL como contribución, CORAL, CoFInAl, SD-KD,
  multi-Teacher ensemble. No los pide el asesor y rompen la narrativa.
- **Conservado de fase 8:** el hallazgo MUSDL TSM-MBv2 + JIGSAWS = +0.053
  SRCC entra como nota en "trabajo futuro" (línea de continuación).

## Experimentos pendientes

### E5 — Réplicas por semilla (baselines AQA-7)

Objetivo: mostrar que el SRCC reportado en E1 no es artefacto de una
inicialización favorable. Se añaden 2 semillas adicionales (`0` y `7`)
para los dos Students en AQA-7. Se reporta media ± std sobre 3 semillas
(42, 0, 7) en la tabla principal de la tesis.

| Run | Config | Seed |
|---|---|---|
| `aqa7_tsm_mbv2_baseline_seed0` | `student_tsm_mbv2.yaml` | 0 |
| `aqa7_tsm_mbv2_baseline_seed7` | `student_tsm_mbv2.yaml` | 7 |
| `aqa7_mbv3_baseline_seed0` | `student_mbv3.yaml` | 0 |
| `aqa7_mbv3_baseline_seed7` | `student_mbv3.yaml` | 7 |

ETA: 30 min × 4 = 2 horas.

### E6 — Sin pretrain ImageNet

Objetivo: aislar el aporte del preentrenamiento ImageNet en el rendimiento
del pipeline. TSM-MobileNetV2 con pesos aleatorios sobre AQA-7.

Implementación: nuevo flag o config `student_tsm_mbv2_nopretrain.yaml` con
`pretrained: false` en la construcción del modelo. Posiblemente requiere
modificación menor de `build_model()` para pasar el flag.

ETA: ~60 min.

### E7 — Sin componente temporal explícito

Objetivo: aislar el aporte del Temporal Shift Module (TSM). Se entrena un
MobileNetV2 plano (sin TSM module inyectado) sobre AQA-7. Si la ablación
muestra que TSM aporta poco en este dataset, el resultado matiza la tesis.
Si aporta, refuerza la importancia del componente temporal eficiente.

Implementación: nueva variante `mobilenetv2_video.py` (análoga a
`mobilenetv3_video.py`) que aplica MobileNetV2 frame-by-frame sin TSM.

ETA: ~60 min implementación + ~30 min entrenamiento.

### E1 extendido — SlowFast-R50 Teacher en AQA-7

Objetivo: añadir un Teacher 3D moderno (SlowFast 2019) además del I3D
(2017). Permite mostrar dónde se ubican los Students respecto al
paradigma 3D actual, no solo el histórico.

Implementación: usar `pytorchvideo.hub.slowfast_r50` pre-entrenado en
Kinetics-400, fine-tune en AQA-7 con la misma receta del Teacher I3D.

ETA: ~2 horas.

---

## Orden de ejecución

1. **9A** — E6 implementación (modificación de `build_model` para flag
   `pretrained=False`).
2. **9B** — E7 implementación (nuevo `mobilenetv2_video.py`).
3. **9C** — Cadena nocturna en este orden:
   - E5 × 4 (semillas 0 y 7 × 2 Students) — 2h
   - E6 × 1 (sin pretrain) — 1h
   - E7 × 1 (sin TSM) — 0.5h
   - E1 extendido — SlowFast Teacher — 2h
4. **9D** — Tabla unificada y `RESULTADOS_FASE_9.md`.
5. **9E** — Avisar al usuario para reescribir LaTeX.

ETA total: ~6 horas de cómputo + 1–2 horas de implementación.

## Reglas operativas (heredadas de fase 8)

- Branch git: `phase-8-novel-contribution` (renombre opcional a `phase-9`
  al cierre).
- Sin commits a `main` hasta cierre.
- BITACORA viva: cada hallazgo, error, decisión.
- Sin reescribir LaTeX hasta tener RESULTADOS_FASE_9.md aprobado.
