# Resumen Ejecutivo de la Tesis

> *Pipeline liviano para Action Quality Assessment basado en arquitecturas
> modernas de bajo costo computacional* · Emmanuel Galdos · UCSP, 2026

Documento de una página y media para entender de un vistazo qué problema
resuelve la tesis, cómo lo resuelve y qué obtiene.

---

## 1. Problema

**Action Quality Assessment (AQA)** es la tarea de que una computadora vea
un video de alguien ejecutando una acción y le ponga una nota de calidad
(qué tan bien lo hizo). Tiene aplicaciones en deporte, rehabilitación,
educación médica y cirugía robótica.

Los modelos que mejor lo hacen son **redes 3D profundas** (I3D, SlowFast),
muy precisas pero **demasiado pesadas** para correr en un teléfono, una
tableta o un sistema embebido. Eso impide su uso en aplicaciones reales
fuera de servidores potentes.

## 2. Pregunta de investigación

> ¿En qué medida un *pipeline* liviano basado en arquitecturas modernas
> (TSM-MobileNetV2, MobileNetV3), bien inicializado y entrenado con
> prácticas actuales, puede aproximarse al rendimiento de un *Teacher* 3D
> profundo en AQA a un costo computacional sustancialmente menor, y dónde
> están sus límites de aplicabilidad?

## 3. Propuesta

Un **pipeline liviano reproducible** para AQA que combina:

- Arquitecturas livianas: **TSM-MobileNetV2** (con módulo Temporal Shift)
  y **MobileNetV3** (puramente espacial).
- **Pesos pre-entrenados** modernos de ImageNet (torchvision 2024).
- **Receta de entrenamiento moderna**: AdamW + programación cosenoidal +
  precisión mixta + acumulación de gradientes.
- **Sólo pérdida de regresión** sobre el score (sin destilación obligatoria
  ni mecanismos auxiliares).

La hipótesis es que con esta combinación, las arquitecturas livianas ya
no necesitan destilación de conocimiento para acercarse al *Teacher* 3D.

## 4. Metodología

- **3 datasets** de naturaleza distinta:
  - **AQA-7** (multi-deporte: clavados, gimnasia, esquí, snowboard, etc.)
  - **MTL-AQA** (clavados especializados)
  - **JIGSAWS** (cirugía robótica con sistema *da Vinci*)
- **2 Teachers de referencia**: I3D-R50 (histórico, 2017) y SlowFast-R50
  (3D moderno, 2019).
- **2 Students** (la propuesta): TSM-MobileNetV2 y MobileNetV3.
- **Métricas de precisión**: SRCC (Spearman), PLCC (Pearson), MAE.
- **Métricas de eficiencia**: parámetros, FLOPs, latencia (RTX 3060 Mobile).
- **3 semillas aleatorias** en el dataset principal (AQA-7) para reportar
  media ± desviación estándar.
- **Ablaciones**:
  - **E6**: pipeline sin pre-entrenamiento ImageNet.
  - **E7**: pipeline sin componente temporal (TSM removido).
- **Análisis marginal**: se evalúa si añadir destilación de conocimiento
  sobre el pipeline mejora resultados.
- **Cross-domain**: transferencia *zero-shot* entre dominios.

## 5. Resultados principales

### Precisión en AQA-7 (3 semillas)

| Modelo | Régimen | SRCC |
|---|---|---|
| SlowFast-R50 | Teacher 3D moderno | 0,9158 |
| I3D-R50 | Teacher 3D histórico | 0,9052 |
| **TSM-MobileNetV2** | **Pipeline propuesto** | **0,9021 ± 0,005** |
| **MobileNetV3** | **Pipeline propuesto** | **0,8907 ± 0,005** |

→ Brecha **menor a 0,025 SRCC** frente al *Teacher* 3D moderno. Robusto
entre semillas (std ≈ 0,005).

### Eficiencia

| Modelo | Params | FLOPs | Latencia |
|---|---|---|---|
| I3D | 27 M | 228 G | 134 ms |
| **TSM-MobileNetV2** | **2,2 M** | **20 G** | **54 ms** |
| **MobileNetV3** | **3,0 M** | **14 G** | **40 ms** |

→ Students usan **menos del 9 %** de los FLOPs del Teacher I3D y son
**~3× más rápidos**.

### Ablaciones (qué componentes sostienen el pipeline)

| Ablación | Δ SRCC |
|---|---|
| Sin pre-entrenamiento ImageNet (E6) | **−0,031** |
| Sin TSM, sólo pool temporal (E7) | **−0,010** |

→ El **pre-entrenamiento es el componente más importante** (3 puntos
SRCC). El TSM aporta menos pero positivamente (1 punto).

### Análisis marginal de la destilación

| Configuración | Δ SRCC vs pipeline solo |
|---|---|
| MBv3 + AQA-7 + KD | +0,040 ✓ |
| Otras 5 configuraciones × KD | de −0,016 a −0,485 ✗ |

→ **La destilación NO aporta valor consistente** sobre el pipeline. Sólo
1 de 6 mejora; las 5 restantes degradan.

### Cross-domain

- MTL-AQA → AQA-7: SRCC ≈ 0,55 (transferencia parcial entre dominios
  deportivos).
- AQA-7 → JIGSAWS: SRCC ≈ 0 (no transfiere a cirugía).

## 6. Conclusiones

1. **Un pipeline liviano moderno cierra la brecha frente a Teachers 3D
   modernos** a menos del 2,5 % de SRCC, usando el 9 % de los FLOPs y con
   3× menos latencia. Reproducible en 3 dominios distintos.

2. **El pipeline funciona por componentes identificables**: pre-entrenamiento
   ImageNet (~3 puntos SRCC) + módulo Temporal Shift (~1 punto adicional).
   No es una caja negra.

3. **La destilación de conocimiento es innecesaria** cuando la línea base
   liviana es fuerte. Cuestiona la premisa heredada de la literatura
   previa.

## 7. Límites de aplicabilidad y trabajo futuro

- Réplicas con 3 semillas sólo en AQA-7 (los otros dos datasets, semilla
  única).
- Sin validación en hardware embebido real (Jetson Nano, móviles).
- Cross-domain a dominios radicalmente distintos (deporte → cirugía)
  requiere fine-tuning específico o adaptación de dominio.
- Líneas futuras: *Score Distribution Learning* sobre Students livianos,
  adaptación de dominio para cross-domain, validación en hardware
  embebido real.

---

*Código: https://github.com/emmanuelgaldos0-max/AQA\_Framework*
