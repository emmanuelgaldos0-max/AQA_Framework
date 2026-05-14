# Criterios de auditoría para una tesis académica

> Lista de verificación que se usa para auditar el documento de tesis.
> Cubre formato, estructura, redacción, citas, figuras/tablas y aspectos
> técnicos de LaTeX.

---

## 1. Formato y layout

- **F1.1** Cada capítulo debe iniciar en página nueva (`\chapter` en LaTeX
  lo hace por defecto si la clase es `book`).
- **F1.2** Las secciones (`\section`) no deberían iniciar a menos de
  ~6 líneas del pie de página. Si una sección se inicia y sólo tiene
  1–2 líneas antes del corte, debe forzarse con `\clearpage` o
  `\newpage`.
- **F1.3** Tablas y figuras flotantes no deben separarse de la primera
  mención. La distancia máxima razonable es ~1 página de separación.
- **F1.4** No deben quedar **viudas** (línea final de párrafo aislada al
  inicio de página) ni **huérfanas** (línea inicial de párrafo aislada al
  final de página). LaTeX gestiona con `\widowpenalty` y `\clubpenalty`
  altos, pero a veces requiere `\samepage` o ajuste manual.
- **F1.5** Una página no debe contener **sólo el título** de una sección
  y nada más (o título + 1 línea). Si ocurre, forzar `\newpage` antes.
- **F1.6** Márgenes consistentes en todo el documento (definidos en
  preámbulo, no cambiar por capítulo).
- **F1.7** Interlineado consistente (usualmente 1.5 en tesis de
  pregrado/maestría).
- **F1.8** Fuentes y tamaños consistentes. Sin cambios injustificados.
- **F1.9** Numeración correcta: romana en preliminares, arábiga desde
  Capítulo 1, sin saltos.
- **F1.10** Encabezado/pie de página consistente.

## 2. Estructura académica

- **F2.1** Portada completa: título, autor, grado al que opta, asesor,
  institución, fecha.
- **F2.2** Dedicatoria (opcional).
- **F2.3** Agradecimientos.
- **F2.4** Resumen (español) y Abstract (inglés). Longitud típica:
  150–300 palabras cada uno. Sin citas, sin referencias.
- **F2.5** Índice general, índice de tablas, índice de figuras, lista de
  abreviaturas.
- **F2.6** Capítulos: introducción, marco teórico, estado del arte,
  propuesta, resultados, conclusiones.
- **F2.7** Bibliografía al final, completa y consistente.
- **F2.8** Anexos si aplica.

## 3. Redacción académica

- **F3.1** Tono impersonal. Preferir voz pasiva refleja con "se"
  (*"se evaluó"*, *"se reporta"*) o primera persona plural neutra
  (*"en este trabajo presentamos"*). **Evitar primera persona singular**
  ("yo hice", "mi propuesta").
- **F3.2** Tiempos verbales consistentes:
  - Presente: hechos generales y conocimiento del campo.
  - Pretérito perfecto/simple: experimentos realizados.
  - Futuro: trabajo futuro y planes.
- **F3.3** Sin contracciones coloquiales ("no se ha", no "no's ha").
- **F3.4** Sin coloquialismos ni jerga informal.
- **F3.5** Definir cada acrónimo en su primer uso, luego usar siempre el
  acrónimo (LaTeX `\ac{}` lo automatiza).
- **F3.6** Frases con extensión moderada. Evitar oraciones >40 palabras.
- **F3.7** Conectores lógicos entre párrafos: *"Por otro lado"*,
  *"En consecuencia"*, *"Adicionalmente"*, *"No obstante"*, etc.
- **F3.8** Uso consistente de mayúsculas: nombres propios sí, conceptos
  genéricos no (e.g., *"clavados olímpicos"*, no *"Clavados Olímpicos"*).
- **F3.9** Términos en inglés en *itálica*: *Action Quality Assessment*,
  *Teacher*, *Student*, *baseline*, etc.
- **F3.10** No usar Markdown dentro de LaTeX (`**bold**`, `*italic*`).
  Usar `\textbf{}` y `\textit{}` o `\emph{}`.

## 4. Citas y referencias

- **F4.1** Cada afirmación no trivial sobre el campo debe tener cita.
- **F4.2** Estilo de citas consistente (APA, IEEE, etc.) según la
  plantilla institucional.
- **F4.3** Bibliografía sin entradas duplicadas en `.bib`.
- **F4.4** Todas las entradas del `.bib` que se citan deben aparecer en
  la bibliografía final.
- **F4.5** Todas las `\cite{...}` deben resolverse (sin `[??]` en el
  PDF).
- **F4.6** Formato consistente de nombres (apellido, iniciales).
- **F4.7** Año correcto en cada entrada (verificar contra el paper
  original).

## 5. Tablas y figuras

- **F5.1** Numeración correlativa por capítulo (Tabla 5.1, 5.2; Fig. 5.1).
- **F5.2** Cada tabla y figura debe tener `\caption{}` descriptivo y
  autoexplicativo (un lector debe entender la tabla sólo con el caption).
- **F5.3** Cada tabla y figura debe ser referenciada en el texto antes
  de aparecer (*"En la Tabla~\ref{tab:foo} se muestra..."*).
- **F5.4** Posicionamiento: usar `[H]` (`float` package) o `[!htb]` para
  forzar cerca de la mención.
- **F5.5** En tablas numéricas: alinear decimales (`S` column de
  `siunitx` o alinear a mano con `c`).
- **F5.6** Unidades especificadas (ms, GB, %).
- **F5.7** Resaltar el mejor valor (negrita o subrayado) y explicar en
  caption qué representa.
- **F5.8** Para figuras: incluir fuente si es de terceros; si es propia,
  no requiere fuente.

## 6. Consistencia terminológica

- **F6.1** Un mismo concepto, una sola forma: no mezclar *"destilación"*
  y *"distilación"*, *"framework"* y *"marco"*, etc.
- **F6.2** Acrónimos siempre escritos igual (KD, no Kd ni kd).
- **F6.3** Nombres propios consistentes en mayúsculas/minúsculas
  (MobileNetV3 siempre, no mobilenetv3).

## 7. Ortografía y gramática

- **F7.1** Sin errores ortográficos.
- **F7.2** Acentuación correcta (esdrújulas, hiatos, agudas).
- **F7.3** Concordancia género/número.
- **F7.4** Puntuación correcta (especialmente comas, dos puntos, punto y
  coma).
- **F7.5** En español: signos de apertura y cierre (¿?, ¡!).

## 8. Coherencia, flujo y lógica argumental

- **F8.1** Cada capítulo tiene introducción breve y cierre que conecta
  al siguiente.
- **F8.2** Las secciones se ordenan lógicamente (no saltos abruptos).
- **F8.3** Referencias cruzadas: cuando una sección/capítulo se menciona
  en otro punto, usar `\ref{}`.
- **F8.4** Sin contradicciones entre secciones.
- **F8.5** El texto valida los objetivos planteados en la introducción.
- **F8.6** Las tablas y figuras del cuerpo coinciden con los números
  citados en el texto.

## 9. Aspectos técnicos LaTeX

- **F9.1** Compilar sin errores ni warnings críticos.
- **F9.2** Sin `Undefined references` (sin `??` en el PDF).
- **F9.3** Sin `Citations undefined`.
- **F9.4** Usar `~` para espacios no rompibles entre referencias y
  números (*"Tabla~\ref{tab:foo}"*, *"Fig.~\ref{fig:bar}"*).
- **F9.5** Usar `\textit{}` para itálicas en cuerpo, no `{\it ...}`.
- **F9.6** Usar `\emph{}` para énfasis lógico, `\textbf{}` para negrita.
- **F9.7** Comillas tipográficas (`` ''`) o españolas (« »), nunca
  caracteres ASCII (`"`).
- **F9.8** Evitar `Overfull \hbox` >5pt y `Underfull \hbox` masivos.
- **F9.9** Encabezados consistentes (`\pagestyle{fancy}` o `plain`).

## 10. Específicos a tesis de Ciencia de la Computación

- **F10.1** Fórmulas matemáticas numeradas (`equation`) si se referencian
  en el texto; sin numerar (`equation*` o `\[ ... \]`) si no.
- **F10.2** Pseudocódigo con `algorithm`/`algorithmic` package, no como
  texto plano.
- **F10.3** Código fuente en `\texttt{}` para nombres de archivo,
  comandos, identificadores; `lstlisting` para bloques completos.
- **F10.4** Datasets y nombres de modelo en su grafía original
  (Kinetics-400, no kinetics 400; ImageNet, no Imagenet).
- **F10.5** Hiperparámetros numéricos consistentes (decimal con punto en
  inglés; coma en español. Documentar y mantener).
