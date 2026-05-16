# Guion final adaptado a tus 18 slides — Exposición tesis

> *Pipeline liviano para Action Quality Assessment basado en arquitecturas
> modernas de bajo costo computacional* · Emmanuel Galdos · UCSP
>
> **Estructura:** 3 min propuesta + 7 min resultados = 10 min. 18 slides
> (~33 seg por slide en promedio). Algunos van rápido (transiciones),
> otros son los que más tiempo te llevarán.

---

## ⚠️ Antes de exponer: correcciones obligatorias en slides

Detallado al inicio de este archivo. Resumen crítico:

1. **Slides 10, 11, 13** — añadir "0." a los SRCC (832 → 0.832, etc.).
2. **Slide 10** — corregir FLOPs I3D = 228 G (no 974) y latencia = 141 ms (no 1.183).
3. **Slides 5 y 6** — corregir receta a valores reales (batch=2 efectivo 16, paciencia=12, sin warmup, steps=8).
4. **Agregar 1 slide** entre 17 (Conclusiones) y 18 (Gracias) titulado **"Trabajo Futuro"**.

---

## Distribución de tiempo por slide

| Slide | Contenido | Tiempo |
|---|---|---|
| 1 | Portada | 10 s |
| 2 | Agenda | 15 s |
| **PROPUESTA (3 min)** | | |
| 3 | Introducción (¿Qué es AQA?) | 30 s |
| 4 | Problemática | 30 s |
| 5 | Trabajos relacionados (3 papers) | 60 s |
| 6 | Solución propuesta (diagrama) | 35 s |
| 7 | Metodología (datasets) | 15 s |
| 8 | Metodología (receta) | 25 s |
| 9 | Metodología (inferencia + Pareto) | 10 s |
| **RESULTADOS (7 min)** | | |
| 10 | Métricas SRCC/PLCC/MAE | 30 s |
| 11 | Métricas FLOPs/Latencia | 30 s |
| 12 | Separador "Resultados" | 5 s |
| 13 | Resultados Anteriores (qué hice antes) | 60 s |
| 14 | Resultados Actuales — SRCC | 80 s |
| 15 | Resultados Actuales — Eficiencia | 60 s |
| 16 | Resultados Actuales — SOTA | 60 s |
| 17 | Conclusiones | 60 s |
| **NUEVO** | Trabajo Futuro | 45 s |
| 18 | Gracias | 5 s |

---

## Guion minuto a minuto

### 🟢 PROPUESTA — 3 minutos

#### Slide 1 — Portada (10 s)

> "Buenos días. Mi tesis se titula *Pipeline liviano para Action Quality
> Assessment basado en arquitecturas modernas de bajo costo
> computacional*."

#### Slide 2 — Agenda (15 s)

> "Voy a cubrir seis puntos: introducción al problema, problemática,
> solución propuesta, entorno de pruebas, métricas y resultados."

#### Slide 3 — Introducción (¿Qué es AQA?) — 30 s

> "AQA es la evaluación automática de la calidad con la que una persona
> ejecuta una acción en video. Por ejemplo, en este clavado el sistema
> da una nota de 86.5 sobre 100 según la técnica, la altura y la
> limpieza de entrada. Tiene aplicaciones en deporte, rehabilitación y
> formación médica."

#### Slide 4 — Problemática — 30 s

> "El problema: los modelos precisos para AQA son redes 3D profundas con
> alto costo computacional. Esto los hace inadecuados para dispositivos
> con recursos limitados, donde se necesita baja latencia y eficiencia
> energética — exactamente los escenarios donde AQA sería más útil:
> retroalimentación en vivo en deporte o rehabilitación."

#### Slide 5 — Trabajos Relacionados — 60 s

> "Reviso tres líneas representativas del campo:"
>
> *"Carreira y Zisserman 2017 introdujeron I3D, las Inflated 3D ConvNets
> que son la referencia histórica del campo — alta precisión pero alto
> costo computacional."*
>
> *"Yu y colaboradores 2021 propusieron Contrastive Regression para AQA,
> alcanzando estado del arte; pero siguen dependiendo de backbones
> pesados como I3D."*
>
> *"Du y colaboradores 2024 incorporaron modelos vision-language para
> mejorar la semántica; sin embargo, esto incrementa aún más la
> complejidad computacional."*
>
> "Lo común: todos sacrifican eficiencia por precisión. Mi hipótesis es
> que con prácticas modernas eso ya no es necesario."

#### Slide 6 — Solución propuesta (diagrama) — 35 s

> "Propongo un pipeline liviano que toma un clip de 64 frames a 224
> píxeles, lo procesa con tres arquitecturas eficientes en paralelo:
> TSM-MobileNetV2 (2D con módulo Temporal Shift), MobileNetV3 (2D puro
> para móvil) y X3D-M (3D ligero). Cada uno genera un score continuo
> entre 0 y 10. Comparamos el rendimiento contra los Teachers de
> referencia I3D-R50 y SlowFast-R50."

#### Slide 7 — Metodología: datasets — 15 s

> "Evaluamos en tres datasets de naturaleza distinta: AQA-7
> multi-deporte, MTL-AQA clavados especializados y JIGSAWS cirugía
> robótica. Todos los clips se preprocesan a 64 frames × 224×224 con
> normalización Kinetics-400."

#### Slide 8 — Metodología: receta — 25 s

> "La receta clave: pesos preentrenados de ImageNet versión 2024, AdamW
> como optimizador con weight decay, programación cosenoidal, precisión
> mixta, acumulación de gradientes y early stopping monitoreando SRCC.
> La pérdida es MSE para regresión simple."

#### Slide 9 — Metodología: flujo de inferencia + Pareto — 10 s

> "El flujo de inferencia es directo: clip → backbone ligero + cabeza
> de regresión → score. Y analizamos la frontera Pareto entre precisión
> y costo computacional."

---

### 🟢 RESULTADOS — 7 minutos

#### Slide 10 — Métricas: SRCC, PLCC, MAE — 30 s

> "Las tres métricas principales: SRCC mide la concordancia de orden,
> valores arriba de 0.75 son sólidos. PLCC mide relación lineal, arriba
> de 0.70 es buena calidad. MAE mide el error absoluto promedio en la
> escala del score."

#### Slide 11 — Métricas: FLOPs y Latencia — 30 s

> "Para eficiencia: FLOPs son operaciones de punto flotante. Modelos por
> debajo de 10 GFLOPs son altamente eficientes; sobre 50 GFLOPs son
> costosos. Latencia inferior a 20 ms permite tiempo real; sobre 100 ms
> ya es inadecuada para aplicaciones interactivas."

#### Slide 12 — Separador "Resultados" — 5 s

> "Paso a resultados."

#### Slide 13 — Resultados Anteriores (qué hice antes) — 60 s

> "Originalmente la tesis era sobre destilación de conocimiento. Los
> resultados anteriores muestran que TSM-MobileNetV2 con KD alcanzaba
> SRCC 0.766 frente a 0.832 del I3D Teacher — había mejora sobre el
> baseline 0.643, pero seguía por debajo del Teacher. Y la KD aumentaba
> la latencia."
>
> "El hallazgo crítico de esta fase fue: la KD original sólo funcionó
> en 1 de 6 configuraciones; en las otras 5 degradaba el rendimiento.
> Eso me llevó a cuestionar la premisa: ¿realmente necesitamos
> destilación, o un pipeline moderno bien construido cierra la brecha
> por sí solo?"

#### Slide 14 — Resultados Actuales: SRCC — 80 s

> "La respuesta es la siguiente. Con prácticas modernas:"
>
> *"I3D Teacher histórico: 0.9052. SlowFast Teacher moderno: 0.9158.
> X3D-M ligero: 0.9211. TSM-MobileNetV2 sobre tres semillas: 0.9021
> ± 0.005. MobileNetV3 sobre tres semillas: 0.8907 ± 0.005."*
>
> "**Hallazgo principal: X3D-M supera a ambos Teachers 3D.** Y los
> Students 2D quedan a menos de 0.025 SRCC del Teacher 3D moderno, con
> desviación estándar de sólo 0.005 — el resultado es robusto entre
> semillas."

#### Slide 15 — Resultados Actuales: Eficiencia — 60 s

> "Ahora la cara de la eficiencia, que es donde se ve el aporte real:"
>
> *"I3D usa 228 GFLOPs y tarda 141 ms. SlowFast 101 GFLOPs y 89 ms.
> X3D-M solamente 19 GFLOPs y 63 ms — eso es el 8.5% de los FLOPs del
> I3D con mejor SRCC. TSM-MobileNetV2 está en 20 GFLOPs y 57 ms.
> MobileNetV3 baja a 14 GFLOPs y 42 ms — el más eficiente."*
>
> "Es decir: alcanzamos o superamos a los Teachers usando una fracción
> de su cómputo. Esto sí habilita aplicaciones en tiempo real en
> dispositivos embebidos."

#### Slide 16 — Resultados Actuales: SOTA — 60 s

> "Comparado con los métodos SOTA reportados específicamente en AQA-7:
> MUSDL 2020 alcanzaba 0.85, CoRe 2021 0.84, TSA-Net 2021 0.85. Mi
> X3D-M alcanza **0.921** — supera a todos los SOTA reportados en este
> benchmark."
>
> "Sin embargo, soy honesto: AQA-7 no es el benchmark más activo en
> 2024–2026. Los papers recientes se han movido a FineDiving y LOGO,
> que no evaluamos. Y en MTL-AQA seguimos por debajo del SOTA actual
> (TPT alcanza 0.96 vs nuestro X3D-M 0.89). La contribución se ubica en
> un punto específico: **precisión competitiva o superior al SOTA en
> AQA-7 con eficiencia drásticamente mejor**."

#### Slide 17 — Conclusiones — 60 s

> "X3D-M con nuestra receta moderna logra el mejor rendimiento (0.9211
> SRCC), superando a I3D y SlowFast, mientras reduce el costo
> computacional de 101–228 GFLOPs a sólo 19 GFLOPs."
>
> "Esto sugiere que, en AQA, una arquitectura eficiente combinada con
> un pipeline moderno puede igualar o superar a modelos más pesados —
> el cuello de botella no estaba en la arquitectura, sino en la receta
> de entrenamiento que usaba la literatura previa."

#### NUEVO SLIDE — Trabajo Futuro (entre 17 y 18) — 45 s

> "Para el examen final completaré tres entregables:"
>
> 1. **Réplicas de X3D-M con tres semillas** en los tres datasets,
>    para reportar desviación estándar (hoy solo tengo una semilla).
> 2. **Evaluación en hardware embebido real** como Jetson Nano y
>    móviles, midiendo consumo energético y memoria de runtime.
> 3. **Versión de Test-Time Adaptation selectiva** que mejore la
>    transferencia cross-domain de manera robusta.

#### Slide 18 — Gracias — 5 s

> "Muchas gracias. Quedo a sus preguntas."

---

## Datos clave a memorizar (para no dudar)

| # | Dato | Valor |
|---|---|---|
| 1 | SRCC X3D-M | **0.9211** ⭐ |
| 2 | SRCC SlowFast | 0.9158 |
| 3 | SRCC I3D | 0.9052 |
| 4 | SRCC TSM-MBv2 (3 sem) | 0.9021 ± 0.005 |
| 5 | SRCC MBv3 (3 sem) | 0.8907 ± 0.005 |
| 6 | FLOPs X3D-M | 19 G |
| 7 | FLOPs I3D | 228 G |
| 8 | Latencia X3D-M | 63 ms |
| 9 | Latencia MBv3 | 42 ms |
| 10 | % FLOPs vs I3D | 8.5% (X3D-M) |

---

## Preguntas anticipadas del jurado

### P1: *"¿X3D-M tiene tres semillas como los Students 2D?"*

**Respuesta honesta:** *"No. X3D-M se entrenó con semilla 42 únicamente,
por restricciones de tiempo de entrenamiento. Las tres semillas se
ejecutaron sólo para los Students 2D en AQA-7. La extensión de
X3D-M a tres semillas está identificada como trabajo futuro inmediato."*

### P2: *"¿Es el SOTA absoluto del campo?"*

**Respuesta honesta:** *"Superamos a todos los métodos SOTA reportados
específicamente en AQA-7 (USDL, CoRe, TSA-Net, etc.). Pero los papers
más recientes 2024–2026 reportan en benchmarks distintos como
FineDiving o LOGO, que no evaluamos. En MTL-AQA, métodos con
componentes específicos como TPT superan a nuestro pipeline; sin
embargo, ningún SOTA reportado opera con menos de 100 GFLOPs."*

### P3: *"¿Por qué no usar destilación si ahora X3D-M es el mejor?"*

**Respuesta honesta:** *"Justamente eso es parte de la contribución
empírica: probamos KD con tres pérdidas y sólo mejoró en 1 de 6
configuraciones; en las demás degradaba. El experimento mostró que
con la receta moderna bien construida, la destilación se vuelve
innecesaria — el pipeline cierra la brecha por sí solo."*

### P4: *"¿Cuál es la novedad técnica original entonces?"*

**Respuesta honesta:** *"La contribución es de tipo empírico — análoga
a los papers 'Bag of Tricks' de He 2019 y 'ResNet strikes back' de
Wightman 2021: demostrar sistemáticamente que una combinación
específica de prácticas modernas (preentrenamiento V2, AdamW, cosine,
AMP, normalización compartida) eleva el techo de arquitecturas ligeras
hasta superar al SOTA reportado con backbones pesados. No reclamo
invención de nuevo método, sino caracterización empírica que
cuestiona una premisa de la literatura previa."*

### P5: *"¿Por qué los datos de Resultados Anteriores (slide 13) son distintos a los Actuales?"*

**Respuesta honesta:** *"Los resultados anteriores corresponden a la
fase inicial donde se evaluó destilación de conocimiento como
contribución central. Tras cuestionar esa premisa, reformulé la tesis
y los resultados actuales reflejan la propuesta refinada con la
receta moderna completa, incluyendo el Student 3D liviano X3D-M
añadido posteriormente."*

---

## Consejos finales

1. **Practica con cronómetro.** 10 min es estricto. Los slides 13–16
   son los más densos y los que más tiempo te tomarán.
2. **No leas las slides.** Habla mirando al jurado.
3. **Señala con cursor el valor clave** en las tablas (X3D-M 0.9211),
   no leas toda la tabla.
4. **Cuando muestres slide 14, enfatiza:** *"X3D-M supera a los dos
   Teachers"*. Es tu punto más fuerte.
5. **Cuando muestres slide 16, sé matizado:** *"superamos al SOTA
   reportado en AQA-7"*, no *"al SOTA absoluto del campo"*.
6. **Cierre fuerte:** termina con "Muchas gracias, quedo a sus
   preguntas" + pausa de 2 segundos.
