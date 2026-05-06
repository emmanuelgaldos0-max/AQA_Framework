# Explicación de la Tesis para Público General

> Documento complementario que explica, en lenguaje sencillo y sin jerga técnica
> innecesaria, de qué trata la tesis, qué se propuso, cómo se hizo y qué se
> obtuvo. El objetivo es que cualquier persona con interés (familia, jurado de
> áreas no afines, colegas de otras disciplinas) pueda entender el trabajo en
> 10--15 minutos de lectura.

---

## 1. ¿De qué trata esta tesis?

Imagina que una gimnasta hace un salto en las olimpiadas. Un panel de jueces
humanos le pone una nota considerando técnica, altura, limpieza,
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

## 3. La idea de la propuesta

En lugar de proponer un método nuevo y complejo, esta tesis explora una
hipótesis directa pero importante: **¿qué pasa si simplemente armamos un
buen pipeline de entrenamiento usando arquitecturas livianas modernas?**

Hace cinco años, los modelos livianos (las redes pequeñas) eran claramente
peores que los grandes. La literatura aceptaba eso como un hecho y proponía
distintos trucos para acortar la distancia. Pero el mundo del aprendizaje
automático ha cambiado mucho desde entonces:

- Los pesos pre-entrenados que vienen con las librerías (PyTorch 2024) son
  considerablemente mejores que los de 2019.
- Optimizadores modernos como AdamW convergen mejor.
- Programar la tasa de aprendizaje con curva cosenoidal mejora resultados.
- Entrenar con precisión mixta y acumulación de gradientes permite usar
  tarjetas gráficas modestas.

Junto, todo esto puede haber cambiado el panorama. La tesis se pregunta:
**¿siguen siendo necesarios los trucos extra, o un simple pipeline moderno
ya basta?**

## 4. Conceptos clave (mini-glosario antes de seguir)

**Pipeline:** una tubería de procesamiento. En programación, una secuencia
de pasos que transforman datos crudos hasta un resultado final. Por
ejemplo: leer un video → extraer cuadros → normalizar colores → entrar al
modelo → salir un puntaje.

**Modelo:** una red neuronal artificial: una función matemática con
millones de parámetros que se "entrenan" con ejemplos hasta que aprende
a realizar una tarea.

**Entrenar un modelo:** mostrarle miles de ejemplos (video + puntaje del
juez) y dejar que ajuste sus parámetros internos hasta que, dado un video
nuevo, prediga un puntaje razonable.

**Dataset:** colección de videos ya etiquetados (con el puntaje del juez
ya conocido).

**Teacher / Student:** en aprendizaje, un modelo grande (Teacher) puede
"enseñar" a uno pequeño (Student). En esta tesis, los Teachers son I3D
(2017) y SlowFast (2019), modelos grandes y precisos. Los Students son
TSM-MobileNetV2 y MobileNetV3, modelos pequeños que sí caben en un
teléfono.

**Destilación de Conocimiento (KD):** la técnica clásica para que un
Student aprenda de un Teacher imitando su comportamiento interno. Esta
tesis evalúa si la KD sigue siendo necesaria con los pipelines modernos.

**FLOPs:** la cantidad de operaciones matemáticas que una computadora hace
para procesar un ejemplo. Cuanto más FLOPs, más tiempo y energía consume.

**Latencia:** el tiempo que tarda un modelo en dar su respuesta.

**SRCC / PLCC / MAE:** tres formas de medir qué tan bueno es el modelo.
SRCC = 0.9 significa que el modelo ordena los videos casi igual que el
juez humano.

**Pre-entrenamiento ImageNet:** los pesos iniciales de la red, aprendidos
sobre un dataset gigante de imágenes de objetos cotidianos. Usar estos
pesos como punto de partida es una práctica estándar.

**Cross-domain:** entrenar el modelo con videos de un tipo (clavados) y
probarlo en otro (cirugía), sin re-entrenar.

## 5. El proceso paso a paso

La tesis se desarrolló en varias fases. Resumen ejecutivo:

### Preparación

Se instaló el software (Python, PyTorch, librerías de visión por
computadora) en un laptop con tarjeta gráfica NVIDIA RTX 3060 Mobile (6 GB
de memoria de video). Se armó la estructura de carpetas y se escribieron
las primeras pruebas automáticas.

### Datos

Se descargaron y procesaron tres colecciones de videos:

- **AQA-7:** ~1.106 videos de siete deportes (clavados, gimnasia, esquí,
  snowboard, patinaje, etc.). Es el dataset principal.
- **MTL-AQA:** 1.412 videos de clavados especializados.
- **JIGSAWS:** 206 grabaciones de cirujanos haciendo tareas básicas
  (suturar, anudar, pasar aguja) en un robot quirúrgico *da Vinci*.

### Construcción del pipeline

Se programaron las redes neuronales:

- **I3D** y **SlowFast** (Teachers grandes, sólo de referencia).
- **TSM-MobileNetV2** y **MobileNetV3** (Students pequeños, la propuesta).

También se programó el código que entrena, evalúa y mide eficiencia.

### Entrenamiento y evaluación

Se entrenaron los modelos en los tres datasets, con réplicas en distintas
semillas aleatorias en el dataset principal para verificar estabilidad. Se
midieron las brechas de precisión y la eficiencia (FLOPs, latencia).

### Ablaciones

Para entender por qué el pipeline funciona, se hicieron experimentos
quitando una pieza a la vez:

- **Sin pre-entrenamiento ImageNet:** ¿qué tanto cambia el resultado?
- **Sin componente temporal explícito (TSM):** ¿qué tanto aporta?

### Análisis marginal de la destilación

Como complemento, se probó añadir destilación de conocimiento sobre el
pipeline para ver si aporta valor adicional.

## 6. Los resultados

### El pipeline liviano cierra la brecha frente a los Teachers

| Modelo | SRCC en AQA-7 |
|---|---|
| SlowFast (Teacher 3D moderno) | 0,9158 |
| I3D (Teacher 3D histórico) | 0,9052 |
| **TSM-MobileNetV2 (pipeline propuesto)** | **0,9021 ± 0,005** |
| **MobileNetV3 (pipeline propuesto)** | **0,8907 ± 0,005** |

La brecha entre el pipeline liviano y el Teacher 3D moderno es **menor a
0,025** (apenas 2,5%) y se mantiene robusta entre semillas (la
desviación estándar es de sólo 0,005). En los otros dos datasets (clavados
especializados y cirugía robótica) la brecha es similar o incluso menor.

### Es mucho más eficiente

| Modelo | Parámetros | FLOPs | Latencia |
|---|---|---|---|
| I3D | 27 M | 228 G | 134 ms |
| TSM-MobileNetV2 | **2,2 M** | **20 G** | **54 ms** |
| MobileNetV3 | **3,0 M** | **14 G** | **40 ms** |

Los Students hacen **menos del 9% del trabajo** del Teacher I3D y
responden **3 veces más rápido**.

### Las ablaciones cuentan una historia interpretable

- **Sin pre-entrenamiento ImageNet:** el SRCC cae 3 puntos. El
  pre-entrenamiento es el componente más importante del pipeline.
- **Sin módulo temporal (TSM):** el SRCC cae 1 punto. El componente
  temporal aporta menos de lo que se esperaría.

Conclusión: el pipeline funciona por **dos componentes identificables**, no
por magia. La mayor parte del rendimiento viene del pre-entrenamiento; el
TSM aporta una fracción menor pero positiva.

### La destilación de conocimiento, sorprendentemente, no aporta

Cuando se añade destilación de conocimiento sobre el pipeline ya fuerte,
sólo 1 de 6 configuraciones mejora; las otras 5 empeoran, algunas mucho.

**Lectura honesta:** la destilación parece **innecesaria** cuando el
pipeline ya está bien construido. Esto cuestiona la premisa de la
literatura previa, que suponía que la destilación era indispensable para
cerrar la brecha entre Teacher y Student.

### Cross-domain

- **Clavados → Multi-deporte:** transferencia parcial (SRCC ~0,55).
  Funciona porque ambos son deportes.
- **Deporte → Cirugía:** falla (SRCC ~0). Los dominios son demasiado
  distintos visualmente para una transferencia automática.

## 7. ¿Cómo se interpretan estos resultados como tesis?

La tesis aporta tres conclusiones positivas y empíricamente sólidas:

### Contribución 1 — Un pipeline liviano que funciona

Un pipeline moderno bien construido (TSM-MobileNetV2 o MobileNetV3 con
pre-entrenamiento, AdamW, cosine annealing, AMP, gradient accumulation)
**cierra la brecha frente al Teacher 3D moderno** (SlowFast) a menos del
2,5% de SRCC, usando el 9% de los FLOPs. Esto resuelve el problema
práctico del título: AQA en dispositivos con recursos limitados.

### Contribución 2 — Componentes identificables

Las ablaciones muestran que el rendimiento del pipeline tiene dos
fuentes claras: el pre-entrenamiento ImageNet (~3 puntos SRCC) y el módulo
TSM (~1 punto adicional). El pipeline no es una caja negra; es una
propuesta interpretable y reproducible.

### Contribución 3 — La destilación de conocimiento es innecesaria

Cuando la línea base liviana es fuerte, la destilación no aporta valor
consistente y puede degradar. Esto cuestiona la premisa heredada de la
literatura previa y replantea la prioridad de futuras propuestas en AQA
eficiente.

## 8. ¿Qué significa en la vida real?

- Un fisioterapeuta podría tener una app móvil que evalúa la técnica de
  ejercicios de rehabilitación en tiempo real, sin servidor.
- Un entrenador deportivo en un pueblo sin buena conexión podría usar un
  dispositivo embebido para dar feedback automático.
- Un hospital que forma residentes quirúrgicos podría evaluar
  automáticamente la habilidad en simuladores (con fine-tuning específico).
- Otros investigadores pueden construir sobre este trabajo: el código y los
  modelos están disponibles públicamente.

## 9. Resumen en tres líneas

1. **Problema:** los modelos buenos para AQA son demasiado pesados para
   teléfonos.
2. **Hallazgo principal:** un pipeline liviano moderno con prácticas
   actuales **cierra la brecha frente a los Teachers 3D** (incluso el
   SlowFast moderno), y la destilación de conocimiento ya no es
   necesaria.
3. **Contribución:** un pipeline reproducible, interpretable
   (componentes identificados por ablación) y empíricamente caracterizado
   que permite hacer AQA eficiente en dispositivos con recursos limitados.

---

*Documento escrito como parte de la tesis de Emmanuel Samir Galdos
Rodriguez (UCSP, 2026). Código fuente: https://github.com/emmanuelgaldos0-max/AQA\_Framework*
