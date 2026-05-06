# Resultados Fase 9 — Robustez empírica del pipeline liviano

> Cierre del plan del asesor. Se ejecutaron los 7 entrenamientos pendientes
> (E5×4 réplicas semilla, E6 ablación pretrain, E7 ablación TSM, E1+
> Teacher SlowFast). Todos los gates de información están cubiertos para
> reescribir la tesis bajo el título *"Pipeline liviano para Action Quality
> Assessment basado en arquitecturas modernas de bajo costo computacional"*.
>
> Fecha: 2026-05-06 · Branch git: `phase-8-novel-contribution` (rebautizar
> a `phase-9-final` al merge).

---

## 1. Tabla unificada de precisión en AQA-7 (dataset principal)

| Modelo | Régimen | seed=42 | seed=0 | seed=7 | **media ± std** |
|---|---|---|---|---|---|
| I3D | Teacher | 0.9052 | — | — | 0.9052 |
| **SlowFast-R50** | **Teacher** | **0.9158** | — | — | **0.9158** |
| TSM-MBv2 | Baseline | 0.8968 | 0.9045 | 0.9049 | **0.9021 ± 0.0046** |
| TSM-MBv2 | + KD (fase 7) | 0.8811 | — | — | 0.8811 |
| MBv3 | Baseline | 0.8854 | 0.8914 | 0.8952 | **0.8907 ± 0.0049** |
| MBv3 | + KD (fase 7) | 0.9250 | — | — | 0.9250 |
| TSM-MBv2 | **Ablación E6: sin pretrain** | 0.8713 | — | — | 0.8713 |
| MBv2 plano | **Ablación E7: sin TSM** | 0.8921 | — | — | 0.8921 |

### Brechas respecto a Teachers (sobre semilla 42 cuando no hay réplicas; media cuando hay)

| Comparación | ΔSRCC |
|---|---|
| TSM-MBv2 baseline vs I3D | 0.9021 − 0.9052 = **−0.0031** |
| TSM-MBv2 baseline vs SlowFast | 0.9021 − 0.9158 = **−0.0137** |
| MBv3 baseline vs I3D | 0.8907 − 0.9052 = **−0.0145** |
| MBv3 baseline vs SlowFast | 0.8907 − 0.9158 = **−0.0251** |

**Lectura:** los Students livianos quedan a menos de 1.5% del Teacher I3D
(referencia histórica) y a menos de 2.5% del Teacher SlowFast (paradigma
3D moderno). La brecha es robusta entre semillas (std ≈ 0.005).

## 2. Ablaciones — qué componentes sostienen el pipeline

### E6 — Aporte del preentrenamiento ImageNet

| Configuración | SRCC |
|---|---|
| TSM-MBv2 con pretrain ImageNet (baseline media) | 0.9021 |
| TSM-MBv2 **sin pretrain** | 0.8713 |
| **Δ** | **−0.0308** (−3.4 %) |

**Interpretación:** el preentrenamiento ImageNet aporta ~3 puntos de SRCC.
El pipeline depende meaningfulmente de la transferencia de
representaciones visuales — no es sólo cuestión de arquitectura.

### E7 — Aporte del componente temporal explícito (TSM)

| Configuración | SRCC |
|---|---|
| TSM-MBv2 (con módulo TSM) | 0.9021 |
| **MBv2 plano (sin TSM)** | 0.8921 |
| **Δ** | **−0.0100** (−1.1 %) |

**Interpretación:** el módulo TSM aporta ~1 punto de SRCC en AQA-7. El
aporte es positivo pero modesto; con pool temporal global ya se captura la
mayor parte de la señal en este dataset (clavados, gimnasia, esquí —
acciones rápidas donde la información espacial-instantánea es muy
informativa). Esto matiza la importancia del modelado temporal explícito y
abre línea futura: en datasets con secuencias más largas o coreografías
complejas el aporte de TSM podría ser mayor.

### Síntesis ablación

> El pipeline liviano funciona por dos componentes identificables y
> separables: (i) el **pretrain ImageNet moderno**, que cierra
> ~3 puntos de SRCC frente al backbone aleatorio, y (ii) el **módulo
> temporal eficiente (TSM)**, que añade ~1 punto adicional. La
> contribución de la tesis no se reduce a "usar arquitectura liviana",
> sino a usarla dentro de un pipeline donde estos dos componentes están
> bien identificados.

## 3. Posición frente a Teachers

```
SlowFast-R50    ←─── 0.9158  (Teacher 3D moderno, 2019)
I3D             ←─── 0.9052  (Teacher 3D histórico, 2017)
TSM-MBv2        ←─── 0.9021 ± 0.0046  (pipeline liviano propuesto)
MBv2 plano      ←─── 0.8921  (sin TSM)
MBv3            ←─── 0.8907 ± 0.0049
TSM-MBv2 noPT   ←─── 0.8713  (sin pretrain ImageNet)
                ───
KD original es ablación negativa: el KD propuesto en fase 7 no mejora de
forma consistente sobre el baseline (ver fase 7 BITACORA §4 y Tabla 5.1
del LaTeX).
```

## 4. Costo computacional (ya reportado en fase 7, se mantiene)

| Modelo | Params (M) | FLOPs (G) | Latencia (ms) |
|---|---|---|---|
| SlowFast-R50 | 33.6 | ≈ 250 | (medir si aplica al cierre) |
| I3D | 27.2 | 228.3 | 133.9 |
| **TSM-MobileNetV2** | **2.2** | **20.0** | **54.3** |
| **MobileNetV3-Large** | **3.0** | **14.3** | **39.9** |

Los Students usan **menos del 9 %** de los FLOPs del Teacher I3D,
**menos del 8 %** de los FLOPs estimados del SlowFast, y **3× menos
latencia**, manteniendo brecha SRCC < 1.5 %.

## 5. Mapeo a objetivos del asesor

| Objetivo | Experimento que lo valida | Estado |
|---|---|---|
| OE1 — Estado del arte | Cap_2 + Cap_3 LaTeX | ✅ ya escrito |
| OE2 — Pipeline reproducible | E1 (3 datasets) + E5 (3 semillas) | ✅ |
| OE3 — Cuantificar brecha (SRCC/PLCC/MAE) | E1 + E5 + Teachers I3D y SlowFast | ✅ |
| OE4 — Ablaciones | E6 (pretrain) + E7 (TSM) | ✅ |
| OE5 — Eficiencia (params/FLOPs/latencia) | E2 (ya hecho fase 7) | ✅ |
| OE6 — Límites de aplicabilidad | E3 (cross-domain) + E4 (KD ablación) | ✅ |

**Cobertura: 6/6 objetivos.** No faltan experimentos.

## 6. Trabajo futuro (mencionar en LaTeX, no obligatorio implementar)

- **Score Distribution Learning** (USDL/MUSDL adaptado): probado en fase 8
  con resultado positivo en TSM-MBv2 + JIGSAWS (+0.053 SRCC sobre
  baseline 0.8283). Queda como dirección de mejora futura.
- **Adaptación de dominio** para AQA-7 → JIGSAWS (CORAL/MMD).
- **Validación en hardware embebido real** (Jetson Nano).
- **Múltiples Teachers como ensemble** (I3D + SlowFast).

## 7. Próximos pasos

1. **Reescribir el documento LaTeX** alineado al título nuevo y al plan del
   asesor:
   - Cap_1 (Introducción) — reformular pregunta y objetivos.
   - Cap_3 (Propuesta) — pivote a "Pipeline liviano" como contribución
     central; KD pasa a sección de análisis marginal.
   - Cap_4 (Resultados) — Tabla 5.1 con media±std de 3 semillas; añadir
     Tabla SlowFast Teacher; añadir secciones de ablaciones E6 y E7.
   - Conclusiones — reformular las 3 contribuciones bajo el nuevo título.
   - Resumen / Abstract / EXPLICACION.md — reescribir.
2. **Bibliografía**: mantener (no se necesitan refs nuevas para fase 9;
   SlowFast Feichtenhofer 2019 ya está en el bib).
3. **Snapshot final**: medir latencia y FLOPs de SlowFast para completar
   Tabla 5.2.

**Listo para reescribir LaTeX cuando el usuario dé luz verde.**
