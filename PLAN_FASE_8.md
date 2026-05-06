# Plan de trabajo Fase 8 — Búsqueda de contribución implementada novedosa

> Plan operativo paralelo a `PROPUESTA_FASE_8.md`. Ejecuta los tres caminos en
> orden de probabilidad de éxito × bajo esfuerzo. Cada camino tiene gate de
> decisión: si pasa el criterio de éxito, queda elegido y se cierra la fase.
>
> Bitácora viva: `BITACORA.md` se actualiza en cada hito (errores, métricas,
> desviaciones, scripts nuevos).

---

## Hardware y entorno

- GPU: NVIDIA RTX 3060 Mobile, 6 GB VRAM
- venv: `code/.venv` (Python 3.10, torch 2.3.1+cu121)
- Dataset paths: `Datasets/AQA-7/`, `Datasets/MTL-AQA/`, `Datasets/JIGSAWS/`
- Repo de código: `code/`
- Branch git de fase 8: `phase-8-novel-contribution`

---

## Fase 8.0 — Setup y verificación (1 día)

- [ ] Crear branch `phase-8-novel-contribution` desde `main`.
- [ ] Verificar que `.venv` activa y `torch.cuda.is_available()`.
- [ ] Verificar que los 3 datasets están preprocesados y accesibles.
- [ ] Verificar que los checkpoints baseline (`L_reg` solo) existen para
      MBv2/MBv3 × {AQA-7, MTL-AQA, JIGSAWS}; si no, re-entrenar como referencia.
- [ ] Snapshot de SRCC baseline para comparar contra resultados de fase 8.

---

## Fase 8.1 — Camino 1: MUSDL + Multi-task heads (3–7 días)

**Implementación:**

- [ ] `code/src/losses/score_distribution_loss.py`
  - Convierte el score escalar `y` en una gaussiana discreta sobre el rango del
    dataset (bins de 1 punto; rango específico por dataset).
  - σ aprendible o fija (probar σ = 5, σ = 10, σ aprendible).
  - Salida: probabilidad sobre N bins. Pérdida: `KL(p_pred || p_target)`.
- [ ] `code/src/models/heads/aux_heads.py`
  - `MTLAuxHead`: predice DD y rotation_type (clasificación).
  - `AQA7AuxHead`: predice action_class (7 clases).
- [ ] Modificar `code/src/models/student.py` para soportar cabeza dual
      (regresión distribucional + auxiliar).
- [ ] Configs YAML: `configs/musdl_mtl.yaml`, `configs/musdl_aqa7.yaml`
      (MBv2 y MBv3 cada uno).

**Entrenamientos (4 configs intra-domain, ~6 horas total):**

- [ ] `MBv2 + AQA-7` con MUSDL + aux action_class.
- [ ] `MBv3 + AQA-7` con MUSDL + aux action_class.
- [ ] `MBv2 + MTL-AQA` con MUSDL + aux DD.
- [ ] `MBv3 + MTL-AQA` con MUSDL + aux DD.

**Decisión gate:**

- ✅ Si ≥ 2/4 mejoran ≥ +0.020 SRCC → **ELEGIDO**, pasar a Fase 8.5.
- ✅ Si ≥ 1/4 mejora ≥ +0.05 SRCC → **ELEGIDO**, pasar a Fase 8.5.
- ❌ Si ninguno cumple → bitácora completa, pasar a Camino 2.

---

## Fase 8.2 — Camino 2: CORAL/MMD para cross-domain (5–8 días)

**Implementación:**

- [ ] `code/src/losses/coral_loss.py`: pérdida CORAL (alineación de matrices
      de covarianza) entre features `source` y `target`.
- [ ] `code/src/datasets/dual_loader.py`: dataloader que entrega batches
      mixtos source (con label) + target (sin label).
- [ ] Modificar `code/src/engine/trainer.py` para incorporar el término
      λ·L_CORAL.
- [ ] Configs YAML: `configs/coral_aqa7_to_jigsaws.yaml`,
      `configs/coral_mtl_to_aqa7.yaml`.

**Entrenamientos (2 transferencias × 2 arch × λ ∈ {0.01, 0.1, 1.0} = 12 corridas):**

- [ ] AQA-7 → JIGSAWS (TSM-MBv2 y MBv3).
- [ ] MTL-AQA → AQA-7 (TSM-MBv2 y MBv3).

**Decisión gate:**

- ✅ Si AQA-7 → JIGSAWS sube ≥ +0.10 SRCC → **ELEGIDO**, pasar a Fase 8.5.
- ✅ Si MTL-AQA → AQA-7 sube ≥ +0.05 SRCC → **ELEGIDO**, pasar a Fase 8.5.
- ❌ Si nada cumple → bitácora completa, pasar a Camino 3.

---

## Fase 8.3 — Camino 3: CoFInAl-inspired prototype head (4–6 días)

**Implementación:**

- [ ] `code/src/models/heads/prototype_head.py`: 10 prototipos coarse +
      5 fine cada uno, parametrizados.
- [ ] `code/src/losses/infonce_proto.py`: InfoNCE contrastivo sobre
      prototipos.
- [ ] Configs YAML correspondientes.

**Entrenamientos (4 configs intra-domain):**

- [ ] MBv2/MBv3 × AQA-7/MTL-AQA con prototype head.

**Decisión gate:** misma regla que Fase 8.1.

---

## Fase 8.4 — Fallback (si los 3 caminos fallan)

- [ ] Documentar exhaustivamente en BITACORA por qué cada camino no aportó.
- [ ] Volver al "Camino C" (narrativa de caracterización) ya plasmada en el
      LaTeX actual; añadir los 3 fracasos como evidencia adicional de la
      conclusión.

---

## Fase 8.5 — Cierre y notificación al usuario

- [ ] Generar tabla comparativa final con todos los caminos probados.
- [ ] Crear documento `RESULTADOS_FASE_8.md` con números concretos.
- [ ] Generar Grad-CAM y métricas de eficiencia para el método elegido.
- [ ] Avisar al usuario con resumen 1-página y esperar luz verde para
      reescribir la tesis.

---

## Reglas operativas

1. **Bitácora obligatoria:** cada error, hallazgo, hiperparámetro encontrado
   por barrido, OOM, NaN, tiempo de training real → BITACORA.md (sección
   nueva por camino).
2. **Sin commits a `main` durante la fase 8.** Todo va a la branch
   `phase-8-novel-contribution`. Merge sólo al cierre.
3. **Reproducibilidad:** semilla 42 en todo. Si se experimenta con otras,
   anotarlas explícitamente.
4. **Sin reescribir LaTeX hasta cerrar fase 8.** El documento se mantiene
   con la narrativa Camino C como respaldo. Si fase 8 entrega resultado
   positivo, se reescribe; si no, se preserva.
5. **Configuración hardware-aware:** todo debe correr en RTX 3060 6 GB. Si
   un experimento no cabe en VRAM, se ajusta batch + grad_accum y se anota.
6. **Métrica clave:** SRCC (Spearman). PLCC y MAE como secundarias.
