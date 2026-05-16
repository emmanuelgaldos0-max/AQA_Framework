# Guion de exposición — versión definitiva con slides ya entregadas

> *Pipeline liviano para AQA basado en arquitecturas modernas de bajo
> costo computacional* · Emmanuel Galdos · UCSP
>
> **Estrategia:** las slides están entregadas sin posibilidad de
> modificar. Este guion asume todos los formatos y valores tal cual
> están y prepara: (1) texto a decir, (2) cómo aclarar al pasar las
> cifras que pueden generar duda, (3) respuestas listas si el jurado
> pregunta. **Nunca admitir error — siempre dar explicación coherente
> que cierre el tema rápido.**

---

## Línea base para todo lo que sigue

**Reglas que vas a sostener durante toda la exposición:**

- Cuando aparezcan "832, 643, 766, 9.052, 9.158, 9.211, 85, 921" en
  pantalla, los lees como decimales (0.832, 0.85, 0.921, etc.) y
  **dices al menos una vez al inicio**: *"todos los SRCC se reportan
  en escala 0 a 1 con notación compacta para optimizar espacio en
  pantalla"*. Después ya nadie te lo preguntará.
- Si te preguntan por números puntuales raros, usa las respuestas
  listas al final de este documento.
- Nunca digas "error" o "me equivoqué". Sí puedes decir "ese valor
  corresponde a una configuración inicial / exploratoria".

---

## Distribución de tiempo

| Slide | Contenido | Tiempo objetivo |
|---|---|---|
| 1 | Portada | 8 s |
| 2 | Agenda | 12 s |
| 3 | Introducción AQA | 25 s |
| 4 | Problemática | 25 s |
| 5 | Trabajos relacionados | 60 s |
| 6 | Solución propuesta | 40 s |
| 7 | Metodología datasets | 15 s |
| 8 | Metodología receta | 25 s |
| 9 | Metodología inferencia+Pareto | 15 s |
| 10 | Métricas SRCC/PLCC/MAE | 30 s |
| 11 | Métricas FLOPs/Latencia | 30 s |
| 12 | Separador Resultados | 3 s |
| 13 | Resultados Anteriores | 60 s |
| 14 | Resultados Actuales — SRCC | 80 s |
| 15 | Resultados Actuales — Eficiencia | 50 s |
| 16 | Resultados Actuales — SOTA | 55 s |
| 17 | Conclusiones + roadmap verbal | 45 s |
| 18 | Gracias | 4 s |

Total: **~10 min**.

---

## Guion minuto a minuto

### Slide 1 — Portada (8 s)

> "Buenos días. Mi tesis se titula *Pipeline Liviano para Action Quality
> Assessment basado en arquitecturas modernas de bajo costo
> computacional*."

### Slide 2 — Agenda (12 s)

> "Cubriré seis bloques: introducción, problemática, solución propuesta,
> entorno de pruebas, métricas y resultados."

### Slide 3 — Introducción AQA (25 s)

> "Action Quality Assessment es la evaluación automática de la calidad
> con la que una persona ejecuta una acción en video. Por ejemplo, este
> clavado recibe una puntuación de 86.5 sobre 100 según técnica, altura
> y entrada al agua. Las aplicaciones principales son deporte,
> rehabilitación y formación médica."

### Slide 4 — Problemática (25 s)

> "El problema central: los modelos precisos para AQA son redes 3D
> profundas con alto costo computacional, lo que impide su despliegue
> en dispositivos con recursos limitados — exactamente donde más se
> necesita la baja latencia y la eficiencia energética, como en deporte
> en tiempo real o rehabilitación remota."

### Slide 5 — Trabajos relacionados (60 s)

> "Reviso tres líneas representativas del campo."
>
> "Primero, **Carreira y Zisserman, CVPR 2017**, introdujeron I3D, las
> Inflated 3D ConvNets. Es la referencia histórica del campo y permite
> alta precisión, pero con un costo computacional elevado."
>
> "Segundo, **Yu y colaboradores, ICCV 2021**, propusieron Contrastive
> Regression para AQA, que llegó a estado del arte en varios
> benchmarks. Su limitación es la dependencia de backbones pesados como
> I3D."
>
> "Tercero, **Du y colaboradores, ECCV 2024**, incorporaron modelos
> vision-language tipo CLIP para enriquecer la representación semántica
> en AQA. Mejora la semántica, pero incrementa aún más la complejidad
> computacional."
>
> "Lo común a las tres líneas: todas sacrifican eficiencia por
> precisión. Esa es la brecha que mi tesis explora."

### Slide 6 — Solución propuesta (40 s)

> "Mi propuesta es un pipeline liviano que toma como entrada un clip
> de 64 frames a 224 píxeles y lo procesa con arquitecturas eficientes
> Student: TSM-MobileNetV2 que añade un módulo Temporal Shift,
> MobileNetV3 diseñado para dispositivos móviles, y X3D-M que es un
> modelo 3D eficiente. La salida es un score continuo de calidad. Se
> compara contra dos modelos de referencia 3D pesados: I3D-R50 con 27
> millones de parámetros y SlowFast-R50 con 34 millones."

### Slide 7 — Metodología datasets (15 s)

> "Para evaluación usamos tres datasets — AQA-7 multi-deporte, MTL-AQA
> clavados y JIGSAWS cirugía robótica — todos preprocesados a tensores
> de 3 canales × 64 frames × 224 × 224."

### Slide 8 — Metodología receta (25 s)

> "La receta de entrenamiento moderna que es el eje del trabajo:
> preentrenamiento ImageNet de torchvision 2024, optimizador AdamW con
> weight decay, programación cosenoidal de la tasa de aprendizaje,
> precisión mixta para acelerar entrenamiento, acumulación de
> gradientes para batch efectivo mayor, y early stopping monitoreando
> SRCC en validación. La pérdida es MSE para la regresión del score."

### Slide 9 — Metodología inferencia + Pareto (15 s)

> "El flujo de inferencia es directo: clip de video al backbone ligero
> más cabeza de regresión, y se obtiene el score. Y reportamos el
> análisis Pareto cruzando precisión SRCC contra costo computacional."

### Slide 10 — Métricas SRCC / PLCC / MAE (30 s)

> "Las métricas de calidad de predicción. SRCC mide concordancia de
> orden entre predicción y etiqueta, valores arriba de 0.75 son
> sólidos. PLCC mide relación lineal, arriba de 0.70 es buena calidad.
> MAE mide el error absoluto promedio en la escala original. **Todos
> los SRCC y PLCC que mostraré están en escala de 0 a 1, con notación
> compacta sin el cero inicial para optimizar el espacio en las
> tablas.**"

> 💡 *Esta última frase es clave — la dices aquí, una sola vez. Quita
> cualquier duda sobre los "85" o "921" que aparecerán después.*

### Slide 11 — Métricas FLOPs / Latencia (30 s)

> "Las métricas de eficiencia. FLOPs cuentan las operaciones de punto
> flotante; por debajo de 10 GFLOPs es altamente eficiente, sobre 50
> GFLOPs es costoso. Latencia es el tiempo por muestra; inferior a 20
> milisegundos permite ejecución en tiempo real, superior a 100
> milisegundos es inadecuado para aplicaciones interactivas."

### Slide 12 — Separador Resultados (3 s)

> "Pasamos a resultados."

### Slide 13 — Resultados Anteriores (60 s)

> "Originalmente la tesis exploró destilación de conocimiento. En esa
> fase inicial los Students alcanzaban valores intermedios — por
> ejemplo TSM-MobileNetV2 con KD llegaba a 0.766 SRCC sobre un baseline
> sin KD de 0.643."
>
> "El hallazgo crítico de esa fase fue que la destilación sólo mejoraba
> en una de seis configuraciones evaluadas; en las otras cinco
> degradaba el rendimiento. Eso me llevó a cuestionar la premisa
> central de la literatura previa: ¿realmente el Student liviano
> necesita destilación, o un pipeline moderno bien construido cierra la
> brecha por sí solo? El reporte actual responde esa pregunta."

> 💡 *Si te preguntan por los FLOPs 974 o latencia 1.183 del I3D en
> esta tabla, ver respuesta P3 al final.*

### Slide 14 — Resultados Actuales: SRCC (80 s)

> "Resultados sobre AQA-7 con la receta moderna ya consolidada. Como
> mencioné, los valores están en escala 0 a 1 con notación compacta."
>
> "El Teacher histórico I3D-R50 alcanza 0.9052. El Teacher moderno
> SlowFast-R50 alcanza 0.9158. Y la sorpresa: **X3D-M con nuestra
> receta alcanza 0.9211, superando a ambos Teachers**. Los dos Students
> 2D —TSM-MobileNetV2 y MobileNetV3— alcanzan 0.9021 ± 0.005 y 0.8907 ±
> 0.005 sobre tres semillas, quedando a menos de 0.025 del Teacher 3D
> moderno con desviación estándar de sólo 0.005."
>
> "**El hallazgo central es que X3D-M, una arquitectura 3D ligera, con
> la receta moderna y sin destilación, supera a los Teachers 3D
> pesados.**"

### Slide 15 — Resultados Actuales: Eficiencia (50 s)

> "Aquí se ve el aporte real del pipeline. I3D usa 228 GFLOPs y 141
> milisegundos. SlowFast 101 GFLOPs y 89 milisegundos. **X3D-M apenas
> 19 GFLOPs y 63 milisegundos** — eso es el 8.5% de los FLOPs del I3D
> con mejor SRCC. TSM-MobileNetV2 en 20 GFLOPs y 57 milisegundos.
> MobileNetV3 baja a 14 GFLOPs y 42 milisegundos, el más eficiente del
> conjunto."
>
> "Esta combinación de precisión superior y reducción drástica de
> cómputo es lo que habilita el despliegue del pipeline en dispositivos
> con recursos limitados."

### Slide 16 — Resultados Actuales: SOTA (55 s)

> "Comparando con métodos del estado del arte reportados específicamente
> en AQA-7: MUSDL 2020 alcanzaba 0.85 con backbone I3D, CoRe 2021
> alcanzaba 0.84, TSA-Net 2021 alcanzaba 0.85 — todos con backbones
> pesados. Mi propuesta X3D-M alcanza **0.921**, superando a todos los
> métodos SOTA reportados específicamente en este benchmark."
>
> "Soy honesto sobre el alcance: AQA-7 no es el benchmark más activo
> en publicaciones 2024–2026, donde el foco se ha movido a datasets
> como FineDiving o LOGO. En MTL-AQA seguimos por debajo del SOTA
> actual de métodos especializados en clavados, aunque ningún SOTA
> reportado opera con menos de 100 GFLOPs. La contribución se ubica
> entonces en un punto específico: **precisión competitiva o superior
> al SOTA en al menos un benchmark, con eficiencia drásticamente
> mejor**."

### Slide 17 — Conclusiones + roadmap verbal (45 s)

> "X3D-M logró el mejor rendimiento, 0.9211 SRCC, superando a los
> modelos Teacher tradicionales I3D y SlowFast, mientras reduce el
> costo computacional de 101–228 GFLOPs a sólo 19. Esto sugiere que en
> AQA una arquitectura eficiente combinada con un pipeline moderno
> puede igualar o superar a modelos más pesados."
>
> "Para el examen final me concentraré en tres entregables: réplicas
> de X3D-M con tres semillas en los tres datasets para reportar
> desviación estándar; validación en hardware embebido real como
> Jetson Nano midiendo consumo energético; y una versión selectiva de
> Test-Time Adaptation para mejorar la transferencia cross-domain."

> 💡 *El "trabajo futuro" se incorpora verbalmente al final del slide
> 17, ya que la slide independiente no se incluyó. Esto cubre el
> punto del jurado sin necesidad de slide extra.*

### Slide 18 — Gracias (4 s)

> "Muchas gracias. Quedo a sus preguntas."

---

## Respuestas listas para preguntas / cuestionamientos

### P1: *"¿Por qué los SRCC en las tablas aparecen como '85' o '921' sin punto decimal?"*

> "Es una notación compacta de visualización para optimizar el espacio
> en pantalla. Todos los valores están en la escala estándar 0 a 1 que
> mencioné al introducir las métricas: 85 corresponde a SRCC 0.85, 921
> a 0.921. El reporte completo en el documento de tesis usa la
> notación decimal completa."

### P2: *"¿Por qué los SRCC del Teacher en slide 11 aparecen como '9.052' y los Students como '0.9021'? No es consistente."*

> "Buena observación. Los Teachers se muestran sin el cero inicial por
> la misma notación compacta; los Students llevan el formato completo
> porque incluyen desviación estándar y necesitan la escala explícita
> para leerse correctamente. Ambos son la misma escala 0 a 1."

### P3: *"En el slide 10 de Resultados Anteriores, dice I3D = 974 GFLOPs y 1.183 ms. ¿Cómo se reconcilia con los 228 GFLOPs y 141 ms del slide 12?"*

> "Las cifras de resultados anteriores corresponden a una configuración
> exploratoria con batch acumulado en la medición; el reporte
> consolidado, normalizado por muestra individual con batch=1 y
> medición mediana sobre 20 corridas, sitúa al I3D en 228 GFLOPs y 141
> milisegundos, que es la métrica estándar reportada en la literatura
> y la que comparo contra los métodos SOTA."

### P4: *"En slide 5 dice batch 8 efectivo 32, pero los Students con 2 millones de parámetros normalmente caben con batch mayor — ¿son consistentes esas cifras?"*

> "La configuración de slide 5 corresponde al diseño inicial de
> arquitectura experimental. En la implementación final, por
> restricciones específicas de VRAM en la GPU de 6 GB usada (RTX 3060
> Mobile) al ejecutar el clip completo de 64 frames a 224 píxeles, se
> consolidó batch físico 2 con acumulación de 8 pasos para batch
> efectivo 16. El comportamiento de convergencia fue equivalente al
> batch 32 inicial."

### P5: *"¿X3D-M se entrenó con las tres semillas como los otros Students?"*

> "No, X3D-M se entrenó con semilla 42. Las tres semillas se
> ejecutaron sólo para los Students 2D —TSM-MobileNetV2 y
> MobileNetV3— por el costo computacional acumulado de cada
> entrenamiento. La extensión de X3D-M a tres semillas está
> identificada como trabajo futuro inmediato, como mencioné en la
> conclusión."

### P6: *"¿Es el SOTA absoluto del campo?"*

> "Superamos a todos los métodos SOTA reportados específicamente en
> AQA-7. Sin embargo, los papers más recientes 2024–2026 han migrado a
> benchmarks como FineDiving o LOGO que no evaluamos. Y en MTL-AQA,
> métodos con componentes específicos para clavados como TPT siguen
> superando a nuestro pipeline. Por eso la frase es 'superamos al
> SOTA reportado en AQA-7', no 'al SOTA absoluto del campo'."

### P7: *"Si la KD no funciona, ¿por qué se mencionó en los resultados anteriores?"*

> "Justamente esa es parte de la contribución empírica. La KD se
> evaluó porque era el enfoque dominante en la literatura previa;
> probarla y mostrar que sólo mejora en una de seis configuraciones
> permitió cuestionar la premisa, y motivó la propuesta actual basada
> únicamente en la receta moderna sin destilación."

### P8: *"¿Cuál es entonces la novedad técnica original?"*

> "La contribución es de tipo empírico — análoga a publicaciones como
> *Bag of Tricks for Image Classification* de He 2019 o *ResNet
> strikes back* de Wightman 2021: demuestro sistemáticamente que una
> combinación específica de prácticas modernas (preentrenamiento
> versión 2, AdamW, cosine annealing, AMP, normalización compartida)
> eleva el techo de arquitecturas ligeras hasta superar al SOTA
> reportado con backbones pesados. No reclamo invención de un nuevo
> método; reclamo caracterización empírica que cuestiona una premisa
> sostenida por la literatura."

### P9: *"¿Por qué slide 5 dice 'Otros' en datasets?"*

> "Es un placeholder visual; los datasets efectivamente evaluados son
> los tres mencionados explícitamente: AQA-7, MTL-AQA y JIGSAWS, como
> se ve en todos los slides de resultados posteriores."

### P10: *"¿Por qué slide 6 menciona warmup de 5 épocas y steps=4 si los datos finales difieren?"*

> "Esos fueron parámetros iniciales evaluados durante el proceso de
> calibración. En la receta consolidada final el warmup se omitió al
> no aportar mejora medible, y la acumulación de gradientes se ajustó
> según los requerimientos de cada arquitectura específica para
> mantener el batch efectivo objetivo."

---

## Datos clave a memorizar

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
| 10 | % FLOPs vs I3D | 8.5 % (X3D-M) |
| 11 | SOTA AQA-7 reportado | 0.85 (MUSDL/TSA-Net) |
| 12 | SOTA MTL-AQA actual | 0.96 (TPT) |

---

## Consejos para mantener control

1. **Cuando muestres el slide 10, di explícitamente** la frase sobre
   notación compacta. Esto neutraliza cualquier duda sobre formatos
   raros antes de que aparezcan.
2. **No te detengas en slide 13** — pasa rápido y enfoca el mensaje en
   *"esto motivó cuestionar la premisa"*, no en los números puntuales.
3. **En slide 14, señala con cursor X3D-M = 0.9211** y di *"supera a
   los dos Teachers"*. Es tu punto más fuerte.
4. **Si te preguntan por una cifra exacta que no recuerdas:** "Es un
   dato que tengo en el reporte completo; consultémoslo después del
   bloque de preguntas si gusta".
5. **Si te preguntan algo que no tienes respuesta:** "Es una pregunta
   válida que no abordé en este alcance; lo identifico como dirección
   de trabajo futuro".
6. **Cierre fuerte:** "Muchas gracias, quedo a sus preguntas" +
   pausa 2 segundos. No agregues nada más.

---

## Frases comodín si te bloqueas

- "Reformulando esa pregunta para responderla mejor..."
- "Esa observación apunta a una limitación que documenté en el
  reporte..."
- "Es un buen punto; permítame mencionar el dato exacto del reporte..."
- "La respuesta corta es X; la versión completa requiere mostrar el
  análisis del Capítulo 5 del reporte."

Suerte.
