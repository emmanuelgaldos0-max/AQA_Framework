# Explicación de la Tesis para Público General

> Documento complementario que explica, en lenguaje sencillo y sin jerga técnica
> innecesaria, de qué trata la tesis, qué se propuso, cómo se hizo y qué se
> obtuvo. El objetivo es que cualquier persona con interés (familia, jurado de
> áreas no afines, colegas de otras disciplinas) pueda entender el trabajo en
> 10--15 minutos de lectura.

---

## 1. ¿De qué trata esta tesis?

Imagina que una gimnasta hace un salto en las olimpiadas. Un panel de jueces
humanos le pone una nota del 0 al 10 considerando técnica, altura, limpieza,
aterrizaje, etc. La pregunta es: **¿puede una computadora mirar el mismo video
y poner una nota similar a la del juez humano?**

Esa tarea se llama **Action Quality Assessment** (en español, *Evaluación de
la Calidad de Acciones*, abreviado **AQA**). No es lo mismo que "reconocer"
que alguien está haciendo un salto (eso ya está resuelto), sino juzgar
**qué tan bien** lo hace.

AQA tiene aplicaciones útiles:

- **Deporte:** entrenadores y atletas pueden recibir feedback automático.
- **Rehabilitación:** pacientes pueden hacer ejercicios en casa y la computadora evalúa si los hacen bien.
- **Educación / formación:** evaluar movimientos de personal médico en prácticas.
- **Cirugía:** medir la habilidad de un cirujano en operaciones robóticas.

## 2. El problema

Las computadoras que saben hacer AQA con buena precisión son **enormes**.
Requieren hardware potente (una tarjeta gráfica cara) y consumen mucha
memoria y energía. Eso las hace **inservibles para dispositivos cotidianos**:
un teléfono móvil, una tableta, un sistema embebido como los que usa un
robot de rehabilitación, etc.

Analogía: es como si para medir tu presión arterial necesitaras una máquina
de hospital de 50.000 dólares. Sirve, pero no puedes tener una en casa.

**La meta de esta tesis:** construir un sistema que haga AQA con precisión
parecida al modelo grande pero que pueda correr en un laptop común o incluso
un teléfono.

## 3. La idea original de la propuesta

En el mundo del aprendizaje automático (*machine learning*), cuando tienes
un modelo grande y muy bueno y quieres uno pequeño y rápido, existe una
técnica clásica llamada **destilación de conocimiento**
(*knowledge distillation*, KD).

Funciona así:

- El modelo grande se llama **Teacher** (maestro).
- El modelo pequeño se llama **Student** (estudiante).
- El Student se entrena mirando lo que hace el Teacher e intentando imitarlo.

La idea es que el Student aprenda no sólo la respuesta correcta, sino
también "cómo razona" el Teacher. En teoría, el Student resulta mejor que
si se entrenara solo.

**La propuesta original de la tesis** era: aplicar destilación de
conocimiento con tres tipos de imitación simultáneas:

1. **Imitar la respuesta final** (el puntaje que asigna el Teacher).
2. **Imitar dónde mira** (en qué parte del video enfoca la atención).
3. **Imitar cómo procesa el tiempo** (qué secuencias de imágenes considera importantes).

El modelo Teacher elegido: **I3D**, un modelo 3D profundo famoso en la
literatura (año 2017, sigue siendo referencia).

Los modelos Student elegidos: **TSM-MobileNetV2** y **MobileNetV3**, dos
arquitecturas ligeras que sí caben en un teléfono.

## 4. Conceptos clave (mini-glosario antes de seguir)

**Pipeline:** una tubería de procesamiento. En programación, significa una
secuencia de pasos que transforman datos crudos hasta un resultado final.
Por ejemplo: leer un video → extraer cuadros → normalizar colores →
entrar al modelo → salir un puntaje. Cada paso es un segmento del
"pipeline".

**Framework:** un "armazón" o conjunto de componentes reutilizables que,
juntos, permiten hacer algo. A diferencia de una sola técnica, un framework
permite configurar distintos escenarios. En esta tesis, el framework
incluye: los modelos (Teacher y Students), las tres pérdidas de
destilación, el pipeline de datos, y los scripts de entrenamiento y
evaluación. Todo esto junto es "el framework".

**Modelo:** en el contexto de esta tesis, es una red neuronal artificial:
una función matemática con millones de parámetros que se "entrenan" con
ejemplos hasta que aprende a realizar una tarea.

**Entrenar un modelo:** mostrarle miles de ejemplos (video + puntaje del
juez) y dejar que ajuste sus parámetros internos hasta que, dado un video
nuevo, prediga un puntaje razonable.

**Dataset:** colección de videos ya etiquetados (es decir, con el puntaje
del juez ya conocido). Sirven para entrenar y evaluar modelos.

**FLOPs (Floating Point Operations):** la cantidad de operaciones
matemáticas que una computadora hace para procesar un ejemplo. Cuanto más
FLOPs, más tiempo y energía consume.

**Latencia:** el tiempo que tarda un modelo en dar su respuesta después
de ver el video. Se mide en milisegundos.

**SRCC / PLCC / MAE:** tres formas de medir qué tan bueno es el modelo.
Todas comparan el puntaje que da el modelo contra el puntaje del juez
humano. SRCC y PLCC van de 0 a 1 (más alto = mejor); MAE es el error
promedio (más bajo = mejor). SRCC = 0.9 significa que el modelo ordena
los videos casi igual que el juez humano.

**Cross-domain:** entrenar el modelo con videos de un tipo (por ejemplo,
clavados olímpicos) y probarlo en otro tipo diferente (por ejemplo, tareas
de cirugía). Mide qué tan "generalista" es el modelo.

## 5. El proceso paso a paso

La tesis se ejecutó en siete fases. Aquí va el resumen ejecutivo de cada
una:

### Fase 1 — Preparación del entorno

Se instaló el software necesario (Python, PyTorch, librerías de visión por
computadora) en un laptop con tarjeta gráfica NVIDIA RTX 3060 Mobile (6 GB
de memoria de video). Se armó la estructura de carpetas del proyecto y se
escribieron las primeras pruebas automáticas para asegurar que todo
funcionara.

### Fase 2 — Datos

Se descargaron tres colecciones de videos con sus respectivas
puntuaciones:

- **AQA-7:** 1.106 videos de siete deportes distintos (clavados, gimnasia,
  esquí, snowboard, patinaje, etc.). Es el dataset principal.
- **MTL-AQA:** 1.412 videos de clavados especializados desde trampolín y
  plataforma.
- **JIGSAWS:** 206 grabaciones de cirujanos haciendo tareas básicas
  (suturar, anudar, pasar aguja) en un robot quirúrgico *da Vinci*.

Cada video se pasó por un "pipeline de preprocesamiento": extracción de 64
cuadros representativos, redimensionado a 224×224 píxeles, normalización
de colores y guardado en un formato rápido de cargar. Esto resultó en
unos 50 GB de datos listos para entrenar.

### Fase 3 — Construcción de los modelos

Se programaron las tres redes neuronales:

- **I3D** (Teacher, 27 millones de parámetros): modelo 3D profundo.
- **TSM-MobileNetV2** (Student 1, 2,2 millones de parámetros): red
  móvil con un truco para procesar el tiempo.
- **MobileNetV3** (Student 2, 3 millones de parámetros): red móvil más
  moderna pero sin el truco temporal.

También se implementaron las tres pérdidas de la destilación (regresión,
atención, alineación temporal) y el código que las combina.

### Fase 4 — Motor de entrenamiento

Se programó el "Trainer": el código que ejecuta el bucle de entrenamiento
(mostrar ejemplos, ajustar los parámetros, medir el progreso, detenerse
cuando ya no mejora, guardar el mejor resultado).

### Fase 5 — Entrenamientos

Se entrenaron 15 modelos en total, en orden:

1. **Tres Teacher I3D** (uno por dataset): 1--3 horas cada uno en el laptop.
2. **Seis Students sin destilación** (dos arquitecturas × tres datasets):
   la línea base.
3. **Seis Students con destilación** aplicando la propuesta completa: los
   experimentos principales.

Durante esta fase se encontraron y resolvieron varios problemas técnicos
(memoria de video insuficiente, la regularización BatchNorm se rompe con
grupos muy pequeños, errores numéricos en un dataset específico). Cada
problema se documentó con su causa y solución en una "bitácora" interna.

### Fase 6 — Evaluación

- **Evaluación dentro del mismo dataset** (train y test del mismo
  dominio): se rellenaron las tablas principales de la tesis.
- **Evaluación cruzada** (cross-domain): se probó cada modelo en un
  dataset distinto al de entrenamiento, sin ajuste adicional.
- **Visualización Grad-CAM:** se generaron 15 imágenes que muestran
  visualmente "dónde mira" cada modelo en el video. Esto permite
  interpretar qué está aprendiendo.

### Fase 7 — Documento final

Se volcaron todos los números reales en el documento LaTeX de la tesis y
se reformuló la narrativa para reflejar lo que efectivamente se encontró.

## 6. Los resultados

Cuando el trabajo terminó, los números contaron una historia distinta a
la esperada:

### Lo que sí funcionó muy bien

**Los modelos Students resultaron casi tan buenos como el Teacher, incluso
sin destilación:**

| Dataset | Teacher (I3D) | Student baseline | Diferencia |
|---|---|---|---|
| AQA-7 | 0,9052 | 0,8968 | apenas 0,008 |
| MTL-AQA | 0,8869 | 0,8804 | apenas 0,007 |
| JIGSAWS | 0,8364 | 0,8368 | ¡el Student es mejor! |

**Y son mucho más eficientes:**

- El Student hace 14 operaciones por cada 228 que hace el Teacher
  (**94 % menos trabajo**).
- Responde en 40 ms por video vs 134 ms del Teacher
  (**3 veces más rápido**).

Eso significa: **un modelo que sí puede correr en un teléfono o en un
dispositivo embebido, con precisión casi idéntica al modelo grande.**

### Lo que no funcionó como se esperaba

**La destilación de conocimiento (la propuesta original) no aportó lo
esperado.** De seis combinaciones (dos arquitecturas × tres datasets),
sólo una mejoró con destilación:

- MobileNetV3 en AQA-7: mejoró de 0,885 a **0,925** (+0,04). ✅

En los otros cinco casos la destilación empeoró ligeramente el
rendimiento, o lo empeoró mucho en el caso más extremo (TSM-MobileNetV2
en cirugía, donde el modelo se volvió prácticamente inútil).

### Por qué ocurrió esto

Al analizar los resultados, surgieron dos explicaciones:

1. **Los modelos livianos modernos ya son muy buenos.** Los pesos iniciales
   que vienen con PyTorch (año 2024) son considerablemente mejores que los
   de 2019 cuando se escribieron los papers originales que motivaron esta
   tesis. El "problema" que la destilación pretendía resolver ya casi no
   existe.

2. **La destilación interfiere con Students que ya modelan tiempo.** El
   TSM-MobileNetV2 tiene un mecanismo interno para procesar secuencias
   temporales. Cuando se le obliga a imitar también la forma en que el
   Teacher procesa el tiempo, las dos señales entran en conflicto y el
   modelo termina confundido. En cambio, MobileNetV3 no tiene mecanismo
   temporal propio, así que la guía del Teacher le sirve.

### Cross-domain

Cuando un modelo entrenado con videos de un dominio se prueba en otro
dominio, los resultados fueron claros:

- **Clavados (MTL-AQA) → Multi-deporte (AQA-7):** se transfiere
  parcialmente (puntaje ~0,55). Ambos son deportes, así que hay cierta
  similitud.
- **Deporte (AQA-7) → Cirugía (JIGSAWS):** falla completamente. Los
  dominios son demasiado distintos visualmente como para transferir sin
  re-entrenar.

## 7. ¿Cómo se convirtieron estos resultados en una tesis "positiva"?

Aquí está el giro importante: en lugar de presentar el trabajo como
"la propuesta original no funcionó", se reformuló en tres contribuciones
positivas y verdaderas:

### Contribución 1 — Un sistema eficiente de AQA para dispositivos con recursos limitados

Se demuestra que arquitecturas ligeras modernas, bien inicializadas y
bien entrenadas, **alcanzan casi la misma precisión que el Teacher I3D
usando menos del 10 % de sus operaciones**. Eso resuelve el problema
práctico del título de la tesis.

### Contribución 2 — Caracterización de cuándo la destilación aporta valor

Es una contribución científica: se muestra empíricamente que la
destilación espacio-temporal **sí funciona pero bajo condiciones
específicas** (Student puramente espacial + dominio rico). Esto es una
**guía accionable para la comunidad**: en qué casos vale la pena aplicar
KD y en qué casos es perjudicial.

### Contribución 3 — Límites claros para la generalización cross-domain

Se documenta qué transferencias entre dominios funcionan y cuáles
requieren estrategias adicionales. Esto abre líneas concretas de trabajo
futuro (por ejemplo, destilación multimodal o adaptación de dominio para
saltar de deporte a cirugía).

## 8. ¿Por qué todo esto se llama "framework" y no simplemente "modelo"?

Un modelo es una red neuronal con parámetros entrenados. Esta tesis
entrega **varios modelos** (tres Teacher, seis Students baseline, seis
Students con KD, además de los checkpoints cross-domain), **además del
código que los conecta**, y **además de la metodología de evaluación**.

Esa combinación —modelos + pipeline de datos + mecanismos de destilación
configurables + protocolo de evaluación + mediciones de eficiencia— es lo
que se llama **framework**: un armazón reutilizable que permite repetir,
extender o modificar el experimento sin rearmar todo desde cero.

Cualquier persona con el repositorio de la tesis puede:

1. Cambiar el dataset (añadir uno nuevo) y todo el pipeline se adapta.
2. Cambiar la arquitectura del Student (probar ResNet o EfficientNet) sin
   reescribir el entrenador.
3. Apagar o encender cualquiera de las tres pérdidas de destilación con un
   archivo de configuración.
4. Medir FLOPs y latencia con un solo comando.

Ese nivel de modularidad es lo que distingue un framework de una
implementación puntual.

## 9. Resumen en tres líneas

1. **Problema:** los modelos buenos para AQA son demasiado pesados para teléfonos.
2. **Hallazgo principal:** los modelos ligeros modernos ya son muy buenos
   sin necesidad de destilación; y cuando la destilación sí ayuda, es en un
   nicho específico (Students sin modelado temporal en datasets ricos).
3. **Contribución:** un framework abierto, reproducible y caracterizado
   empíricamente que permite hacer AQA eficiente en dispositivos con
   recursos limitados y que da una guía clara de cuándo la destilación
   vale la pena.

## 10. ¿Qué significa esto en la vida real?

- Un fisioterapeuta podría tener una app móvil que evalúa la técnica de
  ejercicios de rehabilitación en tiempo real, sin necesidad de enviar el
  video a un servidor.
- Un entrenador deportivo en un pueblo sin buena conexión podría usar un
  dispositivo embebido para dar feedback automático a atletas jóvenes.
- Un hospital que forma residentes quirúrgicos podría evaluar
  automáticamente la habilidad en simuladores (con fine-tuning específico).
- El código y los modelos están disponibles públicamente en GitHub, así
  que otros investigadores pueden construir sobre este trabajo.

---

*Documento escrito como parte de la tesis de Emmanuel Samir Galdos Rodriguez
(UCSP, 2026). Código fuente: https://github.com/emmanuelgaldos0-max/AQA\_Framework*
