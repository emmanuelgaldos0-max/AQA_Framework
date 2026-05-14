# Guion de exposición — Tesis (10 min)

> *Pipeline liviano para Action Quality Assessment basado en arquitecturas
> modernas de bajo costo computacional* · Emmanuel Galdos · UCSP
>
> **Estructura objetivo:** 30 % propuesta (3 min) + 70 % resultados (7 min).

---

## Slides recomendados (10 slides para 10 minutos)

1. **Portada** — título, autor, asesor, fecha.
2. **Problema** — imagen gimnasta/clavadista + ícono móvil con X.
3. **Trabajos relacionados** — 3 referencias en columna.
4. **Propuesta** — diagrama del pipeline (Teacher de referencia + Student).
5. **Lo que se hizo antes** — tabla resumen de la fase de KD.
6. **Resultados — comparación principal** — Tabla 5.1 (Teachers + Students con 3 semillas).
7. **Eficiencia computacional** — tabla FLOPs/params/latencia.
8. **Ablaciones** — qué componentes sostienen el pipeline (E6 + E7).
9. **SOTA + cross-domain con TTA** — posición vs literatura + cross-domain.
10. **Roadmap final** — qué falta para el examen final.

---

## Minuto a minuto

### 🟢 PROPUESTA — 3 minutos

#### Minuto 1 — Problema (60 s)

> "*Action Quality Assessment*, o AQA, busca que una computadora le ponga
> nota a la calidad de un movimiento humano en video — como un juez en una
> competencia de clavados o gimnasia. Tiene aplicaciones reales en
> deporte, rehabilitación física y evaluación de cirujanos."
>
> "El problema: los modelos que mejor hacen AQA son redes 3D profundas
> como I3D o SlowFast — son muy precisos pero demasiado pesados para correr
> en un teléfono, una tableta o cualquier dispositivo con recursos
> limitados."
>
> "La pregunta de mi tesis: ¿podemos hacer AQA con precisión similar pero
> usando una fracción del cómputo?"

**Datos clave a mostrar:**
- *AQA = "ponerle nota" a un movimiento*.
- Teachers 3D = ~228 GFLOPs, 27 M parámetros.
- Mi objetivo = comparable en precisión, mucho menos cómputo.

#### Minuto 2 — Trabajos relacionados (60 s)

> "Reviso tres referencias que cuentan la evolución del campo:"
>
> 1. ***Carreira & Zisserman (CVPR 2017) — I3D.*** Estableció el paradigma
>    3D inflado para reconocimiento de acciones en video. Es la base de
>    casi toda la literatura AQA posterior. Aún hoy es el Teacher estándar.
>
> 2. ***Yu et al. (ICCV 2021) — CoRe.*** Propuso *contrastive regression*
>    para AQA, fue SOTA varios años en MTL-AQA. Pero sigue usando I3D
>    pesado como backbone (~25 M parámetros, ~28 GFLOPs).
>
> 3. ***Du et al. (ECCV 2024) — VLAKL.*** Última tendencia: usar
>    *vision-language* (CLIP) para AQA semántico. Logra SOTA en varios
>    datasets pero requiere CLIP grande — sacrifica aún más eficiencia."
>
> "**Lo común:** todos sacrifican eficiencia por precisión. Lo que falta
> en la literatura es un estudio sistemático de **arquitecturas eficientes
> con prácticas modernas de entrenamiento** — eso es el hueco que ataco."

**Datos clave:**
- 3 papers, 3 épocas (2017, 2021, 2024).
- Hilo común: backbones pesados.
- Hueco: no hay benchmark eficiente sistemático.

#### Minuto 3 — Propuesta (60 s)

> "Mi propuesta es un **pipeline liviano** basado en dos arquitecturas
> móviles modernas: **TSM-MobileNetV2** y **MobileNetV3**. Tres
> ingredientes clave:"
>
> 1. **Pesos ImageNet modernos** (torchvision 2024, no los de 2019).
> 2. **Receta de entrenamiento moderna**: AdamW + cosine annealing +
>    precisión mixta + acumulación de gradientes.
> 3. **Preprocesamiento alineado con el Teacher**: misma normalización
>    Kinetics-400, mismo *clip length*.
>
> "Hipótesis: con esto el Student liviano ya cierra la brecha frente al
> Teacher 3D, y **la destilación de conocimiento ya no es necesaria**.
> Adicionalmente añadimos *Test-Time Adaptation* por BN-recalibration para
> cross-domain — primer estudio de TTA en AQA según nuestra revisión de
> literatura."

**Datos clave a mostrar (diagrama):**
- Pipeline: video → TSM-MBv2 / MBv3 → score.
- Recetas modernas como cajas alrededor.
- Comparación con Teacher I3D / SlowFast.

---

### 🟢 RESULTADOS — 7 minutos

#### Minuto 4 — Qué se hizo antes (60 s)

> "Inicialmente, la tesis era sobre destilación de conocimiento — usar el
> Teacher I3D para enseñar al Student liviano. Combiné tres pérdidas:
> regresión, atención espacio-temporal y alineación de features."
>
> "Resultado honesto: **la destilación sólo funcionó en 1 de 6
> configuraciones**. En las otras 5 degradó el rendimiento, en algunos
> casos catastróficamente (−0.48 SRCC en JIGSAWS TSM-MBv2)."
>
> "Este resultado cuestiona la premisa heredada de la literatura: 'el
> Student liviano necesita destilación'. Mostré empíricamente que **con un
> pipeline moderno bien construido esa premisa no se sostiene**."
>
> "Eso motivó pivotar la tesis: el pipeline liviano deja de ser preámbulo
> y pasa a ser **la contribución central**."

**Slide:** tabla 6×3 (configs × Δ SRCC) con un solo verde y cinco rojos.

#### Minuto 5 — Resultados: comparación principal con Teachers (60 s)

> "En AQA-7, el dataset principal, ejecuté **3 semillas** para reportar
> media y desviación estándar:"
>
> | Modelo | SRCC AQA-7 |
> |---|---|
> | **SlowFast-R50** (Teacher 3D moderno) | **0.9158** |
> | **I3D-R50** (Teacher 3D histórico) | **0.9052** |
> | TSM-MobileNetV2 (pipeline) | **0.9021 ± 0.005** |
> | MobileNetV3 (pipeline) | 0.8907 ± 0.005 |
>
> "El pipeline queda a **menos de 0.025 SRCC** del Teacher 3D moderno,
> con std de sólo 0.005 — el resultado es robusto entre semillas. En los
> otros dos datasets (MTL-AQA, JIGSAWS) la brecha es menor a 0.01 SRCC."

**Datos clave (memorizar):**
- 0.9021 vs 0.9158 → −0.014 (vs SlowFast).
- 0.9021 vs 0.9052 → −0.003 (vs I3D).
- Std 0.005 (robusto).

#### Minuto 6 — Eficiencia computacional (60 s)

> "La parte clave de la propuesta: **lo que ahorramos en cómputo**."
>
> | Modelo | Params | FLOPs | Latencia |
> |---|---|---|---|
> | I3D | 27 M | 228 G | 134 ms |
> | TSM-MobileNetV2 | **2.2 M** | **20 G** | **54 ms** |
> | MobileNetV3 | **3.0 M** | **14 G** | **40 ms** |
>
> "Usamos **menos del 9 % de los FLOPs** del Teacher I3D, parámetros 10×
> menores, latencia 3× más rápida. **40 ms por clip de 64 frames** en una
> RTX 3060 Mobile — eso ya permite tiempo real en dispositivos
> embebidos."

**Datos clave:**
- "9 % FLOPs, 3× latencia, 10× menos parámetros."
- 40 ms = tiempo real.

#### Minuto 7 — Ablaciones (60 s)

> "Para que el pipeline sea **interpretable, no una caja negra**, ejecuté
> dos ablaciones que aíslan los componentes responsables del rendimiento:"
>
> | Ablación | Δ SRCC |
> |---|---|
> | Sin preentrenamiento ImageNet | **−0.031** |
> | Sin módulo Temporal Shift (TSM) | **−0.010** |
>
> "**El preentrenamiento ImageNet es el componente más importante** — sin
> él el SRCC cae 3 puntos. El módulo TSM aporta 1 punto adicional. Esto
> convierte la afirmación 'el pipeline funciona' en una más útil: 'el
> pipeline funciona **porque** el preentrenamiento aporta 3 puntos y el
> TSM aporta 1 punto'."

**Datos clave:**
- E6: −0.031 (sin pretrain).
- E7: −0.010 (sin TSM).
- Mensaje: componentes identificables.

#### Minuto 8 — SOTA + cross-domain con TTA (60 s)

> "Comparando con el estado del arte actual (2024–2025):"
>
> | Método | AQA-7 | MTL-AQA |
> |---|---|---|
> | CoRe (2021) | 0.84 | 0.95 |
> | MUSDL (2020) | 0.85 | 0.93 |
> | TSA-Net (2021) | 0.85 | 0.94 |
> | TPT (2022) | — | **0.96** |
> | **Mío (TSM-MBv2)** | **0.90** | 0.88 |
>
> "**En AQA-7 superamos a varios SOTA basados en I3D**. En MTL-AQA quedamos
> 7 puntos detrás — los SOTA usan componentes específicos para clavados.
> Todo esto usando **una fracción de los FLOPs**, lo cual ningún SOTA
> reporta."
>
> "Adicionalmente, propusimos **Test-Time Adaptation** por BN-recalibration
> para cross-domain — primera aplicación documentada en AQA. [Mostrar
> tabla TTA con los 6 pares y las diferencias.]"

#### Minuto 9 — Discusión: por qué los resultados, posibilidades de mejora (60 s)

> "**¿Por qué no llegamos al SOTA en todos los datasets?** Honestamente:
> los métodos SOTA en MTL-AQA usan I3D de 25 M parámetros + componentes
> específicos como pose o parsing — sacrifican eficiencia. Nosotros
> sacrificamos un poco de precisión para ganar mucha eficiencia."
>
> "**¿Se puede mejorar?** Sí, identificamos tres direcciones concretas:"
>
> 1. **Score Distribution Learning** — experimentos preliminares mostraron
>    **+0.053 SRCC** en TSM-MBv2 + JIGSAWS sobre el baseline.
> 2. **Adaptación de dominio** para arreglar AQA-7 → JIGSAWS cross-domain.
> 3. **Validación en hardware embebido real** (Jetson Nano, móvil).

#### Minuto 10 — Qué se elaborará para el examen final (60 s)

> "Para el examen final, completaré tres entregables:"
>
> 1. **Benchmark Pareto completo** con X3D-M como referencia 3D eficiente
>    adicional → tabla SRCC × FLOPs × latencia que la comunidad necesita.
> 2. **Tabla TTA completa** en los 6 pares cross-domain + ablación del
>    hiperparámetro principal.
> 3. **Documento final pulido** con discusión consolidada y aclaración
>    sobre la naturaleza de la contribución (empírica + caracterización +
>    TTA novedoso)."
>
> "Gracias. Quedo a sus preguntas."

---

## Datos clave a memorizar (lista corta)

| # | Dato | Valor |
|---|---|---|
| 1 | SRCC SlowFast (Teacher moderno) | 0.9158 |
| 2 | SRCC I3D (Teacher histórico) | 0.9052 |
| 3 | SRCC TSM-MBv2 (pipeline, 3 semillas) | 0.9021 ± 0.005 |
| 4 | SRCC MobileNetV3 (pipeline, 3 semillas) | 0.8907 ± 0.005 |
| 5 | Brecha vs SlowFast | < 0.025 |
| 6 | FLOPs Teacher vs Students | 228 G vs 20 G y 14 G |
| 7 | % FLOPs vs Teacher | 9 % |
| 8 | Latencia MBv3 | 40 ms |
| 9 | Ablación pretrain | −0.031 |
| 10 | Ablación TSM | −0.010 |

## Consejos prácticos

1. **No leas las slides** — usa el guion como apoyo, habla mirando al
   jurado.
2. **Practica con cronómetro** — los minutos son rígidos. Una ronda con
   cronómetro y luego ajustes.
3. **Anticipa preguntas** — el jurado probablemente preguntará:
   - *"¿Por qué no llegan al SOTA en MTL-AQA?"* → respuesta lista (min 9).
   - *"¿Es la propuesta novedosa?"* → es una contribución empírica de
     tipo "strong baseline" + TTA novedoso; precedente: *"Bag of Tricks"*
     (He 2019), *"ResNet strikes back"* (Wightman 2021).
   - *"¿Qué garantiza que su receta es la adecuada?"* → ablaciones E6 y
     E7 identifican los componentes responsables.
4. **Cuando muestres una tabla** — señala con el cursor el valor clave,
   no leas toda la tabla.
5. **Cierre fuerte** — termina con "Gracias, quedo a sus preguntas",
   pausa de 2 segundos para dar paso a Q&A.

## Estructura del PPT sugerida (visual)

```
[Slide 1] Portada
[Slide 2] Problema (1 imagen grande + 3 bullets)
[Slide 3] Trabajos relacionados (3 papers en columna)
[Slide 4] Propuesta (diagrama pipeline)
[Slide 5] Fase 7 - KD (tabla 6×3, 1 verde, 5 rojos)
[Slide 6] Tabla 5.1 (Teachers + Students AQA-7)
[Slide 7] Eficiencia (3 columnas comparativas)
[Slide 8] Ablaciones (2 filas Δ SRCC)
[Slide 9] SOTA + TTA (2 mini tablas)
[Slide 10] Roadmap final + "Gracias"
```
