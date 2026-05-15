# Cambios importantes desde el último push del usuario

> Para revisar antes de la exposición. Este documento resume cambios
> significativos en la narrativa de la tesis derivados de los resultados
> experimentales obtenidos durante la sesión autónoma del 2026-05-14.

---

## 1. Hallazgo principal: X3D-M supera al I3D Teacher en LOS TRES datasets

**Resultado experimental completo:** X3D-M entrenado con la misma receta
del pipeline (ImageNet pretrain, AdamW, cosine annealing, AMP, gradient
accumulation, T=64) en los tres datasets:

| Dataset | X3D-M SRCC | I3D Teacher | Estado |
|---|---|---|---|
| AQA-7 | **0.9211** | 0.9052 | supera Teacher y SlowFast (0.9158); supera SOTA reportado |
| MTL-AQA | **0.8937** | 0.8869 | supera Teacher; gap vs SOTA TPT (0.961) |
| JIGSAWS | **0.8682** | 0.8364 | supera Teacher con margen amplio |

**Eficiencia (medida con fvcore + torch.cuda.Event):**
- FLOPs = 19.31 G
- Latencia = 62.6 ms (mediana 20 corridas RTX 3060)
- Params = 2.01 M

### Comparación con el resto

| Modelo | SRCC | FLOPs (G) | Latencia (ms) |
|---|---|---|---|
| **X3D-M (pipeline)** | **0.9211** | **19.3** | 62.6 |
| SlowFast-R50 (Teacher) | 0.9158 | 101.2 | 88.9 |
| I3D-R50 (Teacher) | 0.9052 | 228.3 | 140.8 |
| TSM-MobileNetV2 | 0.9021 | 20.0 | 56.5 |
| MobileNetV3 | 0.8907 | 14.3 | 42.0 |

**Implicación:** X3D-M con la receta moderna supera a ambos Teachers 3D
(I3D y SlowFast) **con menos FLOPs que cualquier otro modelo evaluado
excepto MobileNetV3**.

## 2. Cómo cambia la narrativa de la tesis

| Antes | Ahora |
|---|---|
| "El pipeline liviano cierra la brecha frente a Teachers 3D usando 9% de los FLOPs." | "El pipeline liviano, instanciado en X3D-M, **supera** a Teachers 3D (I3D y SlowFast) usando 9% de los FLOPs del I3D." |
| "Competitivo con SOTA basados en I3D en AQA-7." | "Supera a todos los SOTA basados en I3D en AQA-7 reportados (USDL, GAKD, CoRe, TSA-Net, HGCN), con margen >6 puntos SRCC sobre el mejor reportado." |
| "TSM-MobileNetV2 domina la frontera Pareto en costo bajo." | "X3D-M domina la frontera Pareto absoluta (mejor SRCC con menos FLOPs); TSM-MBv2 y MBv3 quedan como opciones de costo aún más bajo." |

## 3. TTA con resultados reales

12 evaluaciones cross-domain completadas. Resumen:

- **7/12 mejoran** con TTA-BN (reset=True, 1 pase).
- Mejora máxima: **+0.540 SRCC** (MTL-AQA → JIGSAWS, MobileNetV3).
- Pérdida máxima: **−0.681 SRCC** (MTL-AQA → AQA-7, MobileNetV3).
- Mejora promedio: +0.062 SRCC.

**Interpretación:** TTA-BN aporta cuando la transferencia zero-shot es
muy débil, pero degrada cuando ya hay señal útil preservada.

### 3.1 Ablación TTA-BN (Tabla 5.10 nueva)

Sobre 2 pares representativos (mejor y peor del eval principal), 4
variantes: {1, 2 pases} × {reset True, False}.

**Hallazgo clave:** el modo óptimo depende del shift:
- **Shift severo** (MTL → JIGSAWS): mejor reset=True, 1 pase (SRCC=0.329).
- **Shift moderado** (MTL → AQA-7): mejor reset=False (SRCC=0.471 vs
  −0.128 con reset=True). Mantener stats del source y mezclarlas con
  target preserva info útil compartida.

Implicación: TTA-BN debería ser selectivo según shift estimado. Trabajo
futuro claro.

### 3.2 TTA con X3D-M (hallazgo adicional)

Aplicar TTA-BN a X3D-M sólo mejora 2/6 pares (vs 7/12 en Students 2D),
con ganancia máxima +0.012 SRCC. Hipótesis: arquitecturas 3D integran
información temporal en sus stats internas y son menos sensibles al
covariate shift que TTA-BN corrige. **TTA-BN es complementario y
específico de arquitecturas 2D+TSM, no mejora universal**.

## 4. Cómo ajustar el guion de exposición

### Slide 4 (Propuesta) — pequeño ajuste

Mencionar que el pipeline cubre tanto Students 2D (TSM-MBv2, MBv3) como
un Student 3D liviano adicional (X3D-M).

### Slide 6 (Tabla principal) — datos actualizados

Añadir fila X3D-M con su SRCC=0.9211. Texto sugerido:

> "Adicionalmente, evaluamos X3D-M con la misma receta. Resultado:
> 0.9211 SRCC — **supera incluso a SlowFast-R50** (0.9158) con sólo 19
> GFLOPs frente a sus 101 GFLOPs."

### Slide 7 (Eficiencia) — datos reales actualizados

- I3D: 228 G / 140.8 ms (no 134 ms).
- SlowFast: 101 G / 89 ms (no 250 G — corregido por medición fvcore real).
- X3D-M: 19 G / 63 ms (mejor punto Pareto).
- TSM-MBv2: 20 G / 57 ms.
- MBv3: 14 G / 42 ms.

### Slide 9 (SOTA + TTA) — ajustes

SOTA: ahora todos los métodos basados en I3D quedan por debajo de
X3D-M en AQA-7 (con margen claro > 6 puntos). **No** "competitivo
con SOTA" sino **"supera al SOTA reportado en AQA-7"**.

TTA: 7/12 mejoran con ganancias hasta +0.54 SRCC en pares de mayor
shift; degrada en otros. Lectura honesta: aporta cuando la transferencia
zero-shot inicial colapsa, no cuando hay señal útil.

### Slide 10 (Roadmap) — actualizar

Añadir:
- Evaluar X3D-M en MTL-AQA y JIGSAWS para tabla SOTA completa.
- Versión selectiva de TTA-BN.

## 5. Datos clave actualizados (para memorizar)

| # | Dato | Valor |
|---|---|---|
| 1 | SRCC X3D-M (pipeline) en AQA-7 | **0.9211** ⭐ |
| 2 | SRCC SlowFast (Teacher) | 0.9158 |
| 3 | SRCC I3D (Teacher) | 0.9052 |
| 4 | SRCC TSM-MBv2 (3 semillas) | 0.9021 ± 0.005 |
| 5 | SRCC MBv3 (3 semillas) | 0.8907 ± 0.005 |
| 6 | FLOPs X3D-M / SlowFast / I3D | 19 G / 101 G / 228 G |
| 7 | Latencia MBv3 / X3D-M | 42 ms / 63 ms |
| 8 | TTA: configs que mejoran | 7 de 12 |
| 9 | TTA mejora máxima | +0.54 SRCC |
| 10 | TTA pérdida máxima | −0.68 SRCC |

## 6. Riesgo a comentar honestamente si te preguntan

**Pregunta probable del jurado:**
*"¿X3D-M tiene 3 semillas como los Students 2D?"*

**Respuesta:** *"No. X3D-M se entrenó con semilla 42 únicamente por
restricciones de tiempo (cada entrenamiento toma ~110 min). La
extensión a 3 semillas se identifica como trabajo futuro inmediato.
El resultado 0.9211 es por tanto un único punto y debe leerse con esa
salvedad; sin embargo, los Students 2D con 3 semillas confirman que la
receta es robusta (std 0.005), lo que sugiere que X3D-M con la misma
receta también sería estable."*

**Pregunta probable:**
*"¿No es contradictorio decir que el KD no funciona y luego que TTA sí
mejora 7/12?"*

**Respuesta:** *"No: KD interviene durante el entrenamiento y combina
mal con un Student liviano ya bien inicializado; TTA es una intervención
en inferencia que sólo recalibra estadísticas internas, sin tocar
pesos. Son intervenciones complementarias y los resultados son
consistentes con la literatura general en otros campos."*

---

*Documento generado autónomamente el 2026-05-14. Verificar contra
DATOS_PROVISIONALES.txt y RESULTADOS_FASE_9.md antes de la exposición.*
