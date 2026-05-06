# Propuesta Fase 8 — Pivote técnico hacia una contribución implementada novedosa

> Documento de transición: cierra la fase 7 (KD original con resultados parcialmente
> negativos) y abre la fase 8 (búsqueda de un componente técnico nuevo con
> probabilidad real de aporte positivo).
>
> Autor: Emmanuel Galdos · Fecha: 2026-04-27

---

## 1. Qué obtuvimos en la propuesta inicial (fase 7)

La propuesta original era un framework de destilación de conocimiento (KD)
Teacher (I3D) → Student (TSM-MobileNetV2 / MobileNetV3) con tres pérdidas:

```
L_total = α·L_reg + β·L_att + γ·L_temp
```

donde `L_att` alinea mapas de atención espacio-temporal y `L_temp` alinea
representaciones intermedias entre Teacher y Student.

**Resultados experimentales (semilla 42, sobre los 3 datasets):**

| Configuración | Baseline (`L_reg`) | + KD (`L_total`) | Δ |
|---|---|---|---|
| MBv3 + AQA-7 | 0.8854 | **0.9250** | **+0.040** ✅ |
| TSM-MBv2 + AQA-7 | 0.8968 | 0.8811 | −0.016 |
| MBv3 + MTL-AQA | 0.8703 | 0.8470 | −0.023 |
| TSM-MBv2 + MTL-AQA | 0.8804 | 0.7628 | −0.118 |
| MBv3 + JIGSAWS | 0.8368 | 0.7907 | −0.046 |
| TSM-MBv2 + JIGSAWS | 0.8283 | 0.3435 | −0.485 |

**Conclusión empírica:** la KD propuesta sólo aporta en 1 de 6 configuraciones.
En las otras 5 degrada (de levemente a catastróficamente).

**Contribuciones obtenidas (camino C, narrativa actual):**

1. Pipeline eficiente: Students livianos modernos cierran la brecha al Teacher
   sin destilación (<0.01 SRCC) usando 9% de los FLOPs.
2. Caracterización arquitectónica: el KD aporta sólo cuando el Student es
   puramente espacial (MBv3) y el dominio es rico (AQA-7).
3. Límites cross-domain: AQA-7 → JIGSAWS no transfiere zero-shot.

**Problema con esta narrativa:** las contribuciones son **observacionales**, no
**implementacionales**. El curso exige un componente técnico nuevo con
resultados positivos demostrables, no sólo una caracterización empírica.

---

## 2. Qué estamos explorando ahora (fase 8)

Tras una revisión de literatura AQA 2023–2026, se identificaron tres caminos
con alta probabilidad de aporte real, técnicamente novedosos en el contexto
de Students livianos para AQA, y viables en RTX 3060 Mobile (6 GB VRAM):

### Camino 1 — MUSDL + Multi-task auxiliary heads

Reformula el objetivo de regresión: en lugar de MSE/MAE sobre el score, predecir
una distribución gaussiana sobre puntuaciones y minimizar la divergencia KL
contra una gaussiana centrada en la etiqueta real.

Adicionalmente, añadir cabezas auxiliares de multi-task:
- En MTL-AQA: predecir Difficulty Degree (DD) y subcomponentes de score.
- En AQA-7: predecir clase de acción (la disciplina deportiva).

**Por qué podría funcionar mejor:**
- Ataca el ruido de etiqueta del juez subjetivo.
- USDL/MUSDL (Tang CVPR 2020) reporta +1–3 SRCC sobre I3D base; ningún paper
  lo ha aplicado a Students móviles → es una contribución abierta.
- Es ortogonal a la KD que falló: no hay Teacher en el loop, evita el
  problema arquitectónico de TSM-MBv2.
- Las cabezas auxiliares aprovechan etiquetas que ya tenemos pero no usábamos.

**Esperado:** +1 a +3 SRCC en 4 configuraciones deportivas (MBv2/MBv3 ×
MTL-AQA/AQA-7). No aplica en JIGSAWS porque sólo tiene un score por trial.

### Camino 2 — Adaptación de dominio (CORAL / MMD) para cross-domain

Añadir una pérdida de alineación de momentos (CORAL) o discrepancia maximum
mean (MMD) entre las features del Student en `source` (AQA-7) y `target`
(JIGSAWS sin etiquetas) durante el entrenamiento.

**Por qué podría funcionar mejor:**
- Ataca **directamente** el peor número actual: AQA-7 → JIGSAWS = 0.0 SRCC.
- El survey AQA 2025 identifica el domain shift como gap abierto principal.
- PHI (TIP 2025) reporta hasta +24% SRCC atacando domain gaps.
- Convertir un fallo de la fase 7 en una mejora cuantificable.

**Esperado:** SRCC en cross-domain pasa de 0.0 a ≥ 0.3 (mejora notoria).

### Camino 3 — Coarse-to-Fine prototype head (CoFInAl-inspired)

Reformula la salida como prototipos jerárquicos (10 grados gruesos × 5 sub-grados
finos) y entrena con InfoNCE contrastivo en el espacio de prototipos.

**Por qué podría funcionar mejor:**
- CoFInAl (IJCAI 2024) reporta +3–5% SRCC sobre backbones congelados con sólo
  +0.5 M parámetros — exactamente nuestro escenario.
- Es plug-on sobre TSM-MBv2 / MBv3 sin reescribir el backbone.
- Defendible como aporte arquitectónico nuevo (head novedoso, no loss).

**Esperado:** +0.5 a +2 SRCC en AQA-7 / MTL-AQA.

---

## 3. Criterio de éxito para la fase 8

Una técnica se considera **aporte real** y queda elegida como la nueva
propuesta de la tesis si cumple **al menos uno** de los siguientes:

- Mejora ≥ +0.020 SRCC sobre el baseline `L_reg` en al menos 2 de 4
  configuraciones intra-domain (MBv2/MBv3 × MTL-AQA/AQA-7).
- Mejora ≥ +0.05 SRCC en una sola configuración intra-domain.
- Mejora ≥ +0.10 SRCC en alguna configuración cross-domain (de un valor
  base cercano a 0).

Si una técnica cumple, se detiene la búsqueda y se procede a redactar la
nueva propuesta en el documento LaTeX. Si no cumple, se pasa al siguiente
camino. Si los tres caminos fallan, se vuelve al camino C (narrativa
caracterización) como fallback aceptado.
