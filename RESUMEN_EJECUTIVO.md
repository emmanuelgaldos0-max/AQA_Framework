# Resumen Ejecutivo de la Tesis (cronológico)

> Cómo evolucionó esta tesis desde su planteamiento original hasta su forma
> actual. Pensado para que cualquiera entienda en 5 minutos no sólo *qué*
> hicimos, sino *por qué* terminamos haciéndolo así.
>
> Tesis: *Pipeline liviano para Action Quality Assessment basado en
> arquitecturas modernas de bajo costo computacional* · Emmanuel Galdos · UCSP

---

## 1. Al inicio planteamos esto

La tesis arrancó como un trabajo de **destilación de conocimiento (KD)**
aplicada a *Action Quality Assessment*: la idea era usar un *Teacher* 3D
profundo (I3D) para enseñar a dos *Students* livianos
(TSM-MobileNetV2 y MobileNetV3) a evaluar la calidad de acciones en video,
combinando tres señales de supervisión:

- **Pérdida de regresión** sobre el puntaje del juez humano.
- **Pérdida de atención espacio-temporal** para imitar dónde mira el
  *Teacher*.
- **Pérdida de alineación temporal** para imitar cómo procesa la secuencia
  de cuadros.

La hipótesis original era la heredada de la literatura: *los Students
livianos no pueden cerrar la brecha frente al Teacher 3D sin algún tipo de
destilación.* La contribución de la tesis sería entonces un esquema de KD
que cierre esa brecha de forma eficiente.

## 2. Pero al ejecutar nos dimos cuenta de dos cosas

**Primero**, los *Students* entrenados **solo con la pérdida de regresión**
ya alcanzaban casi el rendimiento del *Teacher*. Bastaba con aplicar
prácticas modernas (pesos ImageNet de torchvision 2024, AdamW, programación
cosenoidal, precisión mixta, acumulación de gradientes, normalización
coherente con el *Teacher*) para que la brecha SRCC quedara en menos de
0,01 sobre los tres datasets evaluados (AQA-7, MTL-AQA, JIGSAWS). La
premisa heredada de la literatura — que la KD era necesaria para cerrar
esa brecha — **ya no se sostenía**.

**Segundo**, al añadir las pérdidas auxiliares de KD sobre esa línea base
ya fuerte, los resultados se desordenaron. De seis configuraciones
posibles (dos *Students* × tres datasets), **solo una mejoró**
(MobileNetV3 en AQA-7, +0,040 SRCC). En las otras cinco la KD degradó el
rendimiento, en algunos casos de forma severa (TSM-MobileNetV2 en JIGSAWS
cayó de 0,83 a 0,34 SRCC). En vez de cerrar la brecha, la KD propuesta
**la abría**.

## 3. Debido a eso, cambiamos la dirección

La propuesta original ya no era defendible: si la KD funcionaba en uno de
seis casos, no se podía vender como contribución central. Exploramos
alternativas técnicas (Score Distribution Learning, CORAL para adaptación
de dominio, prototipos coarse-to-fine, KD multi-Teacher), pero al
analizarlas honestamente todas eran *"métodos existentes aplicados a
nuestro setup"*: ingeniería, no novedad genuina.

Tras conversación con el asesor, **replanteamos la tesis desde otro
ángulo**. La contribución central deja de ser la KD y pasa a ser **el
propio pipeline liviano**: la observación de que con prácticas modernas
los Students ya cierran la brecha. La tesis se reformula bajo un título
nuevo —*"Pipeline liviano para Action Quality Assessment basado en
arquitecturas modernas de bajo costo computacional"*— y los experimentos
ya hechos se reorganizan al servicio de esa contribución:

- La comparación principal contra el *Teacher* I3D pasa a ser **la
  validación del pipeline**, no un benchmark suelto.
- La KD pasa a ser **análisis marginal**: muestra honestamente que sobre
  una línea base liviana fuerte la destilación no aporta consistencia.
- Se agregan los experimentos que faltaban para que el pipeline quede
  empíricamente blindado: réplicas con tres semillas, ablaciones,
  segundo *Teacher* moderno.

## 4. Y ahora tenemos esto

### Comparación principal en AQA-7 (3 semillas en los Students)

| Modelo | Régimen | SRCC |
|---|---|---|
| **SlowFast-R50** (2019) | Teacher 3D moderno | 0,9158 |
| I3D-R50 (2017) | Teacher 3D histórico | 0,9052 |
| **TSM-MobileNetV2** | **Pipeline propuesto** | **0,9021 ± 0,005** |
| **MobileNetV3** | **Pipeline propuesto** | **0,8907 ± 0,005** |

→ El pipeline liviano queda a **menos de 0,025 SRCC** del *Teacher* 3D
moderno, con desviación estándar de apenas 0,005 entre semillas.

### Eficiencia computacional

| Modelo | Params | FLOPs | Latencia |
|---|---|---|---|
| I3D | 27 M | 228 G | 134 ms |
| TSM-MobileNetV2 | 2,2 M | 20 G | 54 ms |
| MobileNetV3 | 3,0 M | 14 G | 40 ms |

→ Los *Students* usan **menos del 9 % de los FLOPs** del Teacher I3D y
son **3× más rápidos** en inferencia.

### Ablaciones (qué componentes sostienen el pipeline)

| Configuración | Δ SRCC vs pipeline completo |
|---|---|
| Sin pre-entrenamiento ImageNet | **−0,031** (es lo más importante) |
| Sin módulo Temporal Shift (TSM) | **−0,010** (aporta, pero menos) |

→ El pipeline funciona por **dos componentes identificables**, no por
magia. La mayor parte del rendimiento viene del pre-entrenamiento; el TSM
aporta una fracción menor pero positiva.

### Análisis marginal de la destilación

→ Sólo 1 de 6 configuraciones mejora con KD; las otras 5 degradan
(Δ SRCC entre −0,016 y −0,485). **La destilación es innecesaria** cuando
la línea base liviana es fuerte.

### Cross-domain

→ MTL-AQA → AQA-7 transfiere parcialmente (SRCC ≈ 0,55). AQA-7 → JIGSAWS
falla (SRCC ≈ 0). El pipeline funciona dentro del dominio entrenado,
pero la transferencia a dominios radicalmente distintos requiere
fine-tuning específico.

## 5. Por lo que la conclusión queda así

La tesis termina sosteniendo tres cosas, todas empíricamente respaldadas:

1. **El pipeline liviano moderno cierra la brecha frente a Teachers 3D**
   (incluso al SlowFast moderno), usando el 9 % de los FLOPs y con 3×
   menos latencia. Robusto entre semillas y reproducible en tres
   dominios.

2. **El pipeline es interpretable**: las ablaciones identifican que el
   pre-entrenamiento ImageNet aporta ~3 puntos SRCC y el módulo TSM ~1
   punto adicional.

3. **La destilación de conocimiento es innecesaria** cuando se dispone de
   un pipeline liviano fuerte. Esto cuestiona una premisa heredada de la
   literatura previa y replantea la prioridad de futuras propuestas en
   AQA eficiente.

El recorrido — proponer, ejecutar, encontrar que la propuesta original no
funcionaba, cuestionar honestamente por qué, y reconstruir la tesis sobre
los datos que sí teníamos — quedó documentado en la BITACORA del proyecto
y se preserva como parte de la trazabilidad metodológica.

---

*Código y artefactos: https://github.com/emmanuelgaldos0-max/AQA\_Framework*
