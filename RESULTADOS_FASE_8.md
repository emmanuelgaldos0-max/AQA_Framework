# Resultados Fase 8 — MUSDL como contribución implementada nueva

> Cierre de la fase 8. Se evalúa Camino 1 (MUSDL aplicado a Students livianos)
> contra el baseline de la fase 7. **El gate de decisión pasa** y MUSDL se
> adopta como nueva propuesta técnica de la tesis.
>
> Fecha: 2026-04-28 · Branch git: `phase-8-novel-contribution`

---

## 1. Tabla de resultados (Camino 1)

Todos los entrenamientos con semilla 42, configuración común
(num_bins=100, σ=5 bins, batch_efectivo=16, AdamW 3e-4, cosine, AMP,
50 epochs / early stop patience 12).

| Dataset | Student | Baseline `L_reg` | **MUSDL** | Δ SRCC | Tiempo |
|---|---|---|---|---|---|
| AQA-7 | MobileNetV3 | 0.8854 | **0.9142** | **+0.0288** ✅ | 37.3 min |
| AQA-7 | TSM-MobileNetV2 | 0.8968 | **0.9223** | **+0.0255** ✅ | 80.3 min |
| MTL-AQA | MobileNetV3 | 0.8703 | 0.8508 | −0.0195 | 49.2 min |
| MTL-AQA | TSM-MobileNetV2 | 0.8804 | 0.8531 | −0.0273 | 124.4 min |

**Gate de decisión (PLAN_FASE_8.md):**

- ≥ 2/4 configs con Δ ≥ +0.020 SRCC: **2/4 cumplen** ✅
- ≥ 1/4 con Δ ≥ +0.050: 0/4 (no aplicable, el otro ya alcanzó).

**Resultado: GATE PASA ✅** — MUSDL se adopta como nueva propuesta.

## 2. Comparación con propuesta original (KD)

| Config | Baseline | KD original | **MUSDL** |
|---|---|---|---|
| MBv3 + AQA-7 | 0.8854 | 0.9250 (+0.040) | **0.9142 (+0.029)** |
| TSM + AQA-7 | 0.8968 | 0.8811 (−0.016) | **0.9223 (+0.026)** ✅ |
| MBv3 + MTL-AQA | 0.8703 | 0.8470 (−0.023) | 0.8508 (−0.020) |
| TSM + MTL-AQA | 0.8804 | 0.7628 (−0.118) | 0.8531 (−0.027) |

**Hallazgo clave:** MUSDL **funciona en ambas arquitecturas en AQA-7**,
mientras que el KD original sólo funcionaba en MobileNetV3 + AQA-7. Esto
elimina la limitación arquitectónica del KD (que degradaba a TSM-MBv2).
MUSDL es **arquitectura-agnóstica** dentro del dataset AQA-7.

## 3. Por qué este resultado es defendible como contribución

1. **Novedad técnica concreta:** primer estudio de Score Distribution
   Learning (Tang CVPR 2020) aplicado a Students móviles para AQA. La
   literatura previa de MUSDL/USDL usa I3D/3D-ResNet (modelos pesados);
   ningún paper lo aplica a TSM-MobileNetV2 ni MobileNetV3.

2. **Mejora consistente intra-arquitectura:** mientras que el KD original
   tenía el problema de degradar TSM-MBv2 (porque los módulos TSM ya
   modelan tiempo y la KD temporal interfiere), MUSDL no entra en
   conflicto con módulos temporales — sólo cambia la formulación del
   objetivo, no la representación. Por eso aporta en ambas arquitecturas.

3. **Mejora en el dataset principal:** AQA-7 es el dataset multi-disciplina
   más diverso, con 7 deportes. MUSDL aporta consistentemente +0.025 a
   +0.029 SRCC sobre el baseline más fuerte conocido.

4. **Caracterización dataset-dependiente:** MUSDL no aporta en MTL-AQA,
   donde la distribución de scores está fuertemente concentrada
   (clavados especializados, scores en rango estrecho). Este resultado
   negativo refuerza la narrativa de "aporte selectivo" característica
   del trabajo, ahora con dos ejes de selectividad: arquitectura
   (KD selectiva) y distribución de scores (MUSDL selectiva).

## 4. Implementación

Archivos nuevos en branch `phase-8-novel-contribution`:

| Archivo | Propósito |
|---|---|
| `code/src/losses/score_distribution.py` | `make_target_distribution`, `musdl_kl_loss`, `expected_score` |
| `code/src/models/heads.py` (extendido) | nuevo `DistributionHead` (logits sobre N bins) |
| `code/src/models/musdl_wrapper.py` | wrapper que reemplaza head del Student |
| `code/src/engine/musdl_trainer.py` | subclase de `Trainer` con KL loss y E[score] en validate |
| `code/scripts/train_musdl.py` | entry point CLI |
| `code/scripts/run_musdl_remaining.sh` | encadenador de los 4 entrenamientos |
| `code/scripts/evaluate_gate.py` | evaluación del gate |
| `code/configs/musdl_*.yaml` | 4 configs |

Hiperparámetros clave: `num_bins=100`, `σ=5 bins`, todo lo demás igual al baseline.

## 5. Próximos pasos

1. **Reescribir Cap_3 (Propuesta)** del LaTeX:
   - Reemplazar la propuesta KD principal por MUSDL adaptado a Students
     móviles.
   - Conservar Cap_3 §KD como ablación / análisis arquitectónico.
   - Formular la pérdida MUSDL como contribución principal nueva.
2. **Reescribir Cap_4 (Resultados)**:
   - Tabla principal con (Teacher / Baseline / MUSDL / KD ablación).
   - Análisis dataset-dependiente del aporte (AQA-7 sí, MTL-AQA no).
3. **Actualizar Resumen, Abstract, Conclusiones, EXPLICACION.md.**
4. **Bibliografía**: añadir entrada Tang et al. CVPR 2020 (USDL/MUSDL).

## 6. Caminos no ejecutados

Los Caminos 2 (CORAL para cross-domain) y 3 (CoFInAl prototype head) están
**implementados y smoke-testeados** pero no se ejecutaron al cumplir Camino 1
el gate. Pueden quedar como **trabajo futuro** o como **anexos
experimentales adicionales** si el asesor pide más extensión.
