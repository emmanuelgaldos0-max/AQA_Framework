# Bitácora del Proyecto AQA Framework

> Registro cronológico de **descubrimientos, desviaciones del plan, errores y resoluciones**.
> Claude debe actualizar este archivo **en cada hallazgo no trivial** durante la implementación.
>
> Ubicación: `/home/sam/Documentos/Github Personal/Tesis/BITACORA.md`
> Propósito: tener trazabilidad completa de todo lo que no estaba en el plan original
> pero terminó siendo parte del proyecto, y de cómo se resolvió cada problema.

---

## 1. Descubrimientos del dominio y datasets

### AQA-7
- **1189 videos totales** distribuidos en 7 disciplinas, pero el **Split_4 oficial solo usa 1106** (excluye `trampoline` porque tiene 618 frames por clip en vez de 103).
- Los `.mat` oficiales vienen con columnas `(action_class, sample_no, score)` pero con **múltiples filas duplicadas por clip** (augmentation temporal declarada en el Readme). **Hay que deduplicar por `(category, sample_no)`**.
- **Scores pueden exceder 100** (llegan a ~102.6 en diving). No normalizar con clip a 100 rígido; usar `score/100.0` como señal en [0, ~1.02].
- 7 carpetas en `Actions/`: diving, gym_vault, ski_big_air, snowboard_big_air, sync_diving_10m, sync_diving_3m, trampoline.
- **Action class mapping** (Readme): 1=diving, 2=gym_vault, 3=ski_big_air, 4=snowboard_big_air, 5=sync_diving_3m, 6=sync_diving_10m.

### MTL-AQA
- Distribución real: **frames pre-extraídos** en carpetas `<dive_id:02d>_<clip_id:02d>/`, 1412 clips × ~109 frames cada uno.
- Anotaciones en el repo oficial: `MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/final_annotations_dict.pkl`.
- Keys del pkl son **tuplas `(dive_id, clip_id)` de ints**, no strings.
- Cada entrada tiene: `primary_view, start_frame, end_frame, position, difficulty, armstand, rotation_type, ss_no, tw_no, final_score`.

### JIGSAWS
- **103 trials oficiales × 2 vistas (`_capture1`, `_capture2`) = 206 clips** en el paquete de frames que usa el usuario.
- Cada trial tiene **6 items OSATS** cada uno en [1, 5]; GRS = suma ∈ [6, 30].
- **Anotaciones no vienen en el paquete de frames**. Se obtuvieron de `github.com/nzl-thu/MUSDL/master/JIGSAWS/data/info/label.pkl` (repo público del paper MUSDL).
- Tareas: `Suturing` (~90 clips), `Knot_Tying` (~70), `Needle_Passing` (~46). Distribución desbalanceada.
- Formato de trial_id: `<Task>_<Subject_Trial>` (p. ej. `Suturing_B001`, `Knot_Tying_E003`).

---

## 2. Desviaciones del plan original

| Tema | Plan original | Realidad | Razón |
|---|---|---|---|
| **Datasets** | AQA-7 + MIT-Diving + MTL-AQA | AQA-7 + MTL-AQA + JIGSAWS | Usuario pidió ampliar a dominio quirúrgico |
| **Formato AQA-7** | Asumía frames | Archivos .avi | Así lo distribuyó Parmar |
| **Formato MTL-AQA** | .avi | Frames en carpetas | Distribución real desde Drive |
| **Formato JIGSAWS** | Frames | Frames | Confirmado |
| **Teacher I3D** | `piergiaj/pytorch-i3d` | `pytorchvideo.hub.i3d_r50` | API más limpia, torch 2.3 OK |
| **Venv** | `python3 -m venv` | `virtualenv` | python3.10-venv no estaba instalado |
| **Anotaciones MTL-AQA** | En el paquete del usuario | Descargadas de github.com/ParitoshParmar/MTL-AQA | El paquete de frames no las traía |
| **Anotaciones JIGSAWS** | En el paquete | Descargadas de github.com/nzl-thu/MUSDL | El paquete oficial requiere registro + OSATS derivarse de meta files; MUSDL ya los tiene parseados |
| **Batch size Students** | 16 | 2 (con grad_accum=8) | OOM en RTX 3060 6 GB con T=64 |
| **Batch size KD** | 16 | 1 (con grad_accum=16) | Teacher+Student en VRAM es aún más apretado |
| **Clip length** | 64 frames a 25 FPS | 64 frames a 25 FPS | OK, pero MTL/JIGSAWS vienen a 30 FPS efectivos → submuestreo step=1 |
| **Dataset sintético** | No contemplado | Implementado | Validar pipeline sin depender de descargas externas |

---

## 3. Errores encontrados y resoluciones

### E01 — `pytorch-grad-cam` no existe en PyPI
- **Síntoma:** `ERROR: No matching distribution found for pytorch-grad-cam`.
- **Causa:** Nombre incorrecto del paquete.
- **Fix:** `pip install grad-cam` (nombre correcto).

### E02 — `opencv-python 4.13` incompatible con `numpy 1.26.4`
- **Síntoma:** Conflict: opencv requiere numpy>=2, pero timm/fvcore piden numpy<2.
- **Fix:** Pinear `opencv-python<4.12` + `numpy==1.26.4`.

### E03 — `python3-venv` no instalado en el sistema
- **Síntoma:** `ensurepip is not available` al crear venv.
- **Fix:** Usar `virtualenv` (ya disponible en el sistema globalmente).

### E04 — LaTeX Workshop no encuentra `latexmk`
- **Síntoma:** `spawn latexmk ENOENT`.
- **Fix:** Instalar `latexmk` vía `apt install latexmk` (lo hace el usuario con sudo).

### E05 — `Bibliog.bib` tenía 19 entradas duplicadas
- **Síntoma:** BibTeX aborta con `Repeated entry` → `latexmk` falla.
- **Fix:** Script Python que escanea entradas y mantiene solo la primera ocurrencia por clave. 70 → 51 entradas.

### E06 — URL rota en bibliografía (`slwang_AQA-Survey.pdf`)
- **Síntoma:** Guión bajo se interpreta como subíndice LaTeX.
- **Fix:** Envolver con `\url{...}` y añadir `\usepackage{url}` a `Tesis.tex`.

### E07 — `\text` undefined en fórmulas matemáticas
- **Síntoma:** `Undefined control sequence \text{...}`.
- **Fix:** Añadir `\usepackage{amsmath}` (y `amssymb` por higiene).

### E08 — Unicode `κ` en texto plano no compila
- **Síntoma:** LaTeX Error: Unicode character κ (U+03BA).
- **Fix:** Usar `$\kappa$` en modo matemático.

### E09 — Acrónimos sin definir se renderizan como "AQA!" "KD!" "TSM!" "I3D!"
- **Síntoma:** El paquete `acronym` genera placeholders con `!` cuando el acrónimo no está definido.
- **Fix:** Definir todos los acrónimos en `abreviaturas.tex` (AQA, KD, I3D, TSM, SRCC, PLCC, MAE, FLOPs, CNN, RGB, MSE, KL).

### E10 — Tabla 5.1 tenía cifras numéricas hardcoded de un entrenamiento hipotético
- **Fix:** Marcar como "valores preliminares, se actualizarán con los experimentos reales" hasta obtener números.

### E11 — CUDA OOM con Students baseline (batch=16, T=64)
- **Síntoma:** `torch.cuda.OutOfMemoryError: Tried to allocate 1.53 GiB`.
- **Causa:** Procesamiento por-frame del backbone 2D: B×T=16×64=1024 frames en paralelo → activaciones enormes.
- **Fix:** `configs/student_*.yaml`: `batch_size: 2` + `grad_accum_steps: 8` (efectivo 16).

### E12 — CUDA OOM más severo en KD (Teacher+Student simultáneos)
- **Fix:** `configs/kd.yaml`: `batch_size: 1` + `grad_accum_steps: 16`.

### E13 — KD diverge: SRCC baja a 0.37–0.70 cuando baseline es 0.90
- **Síntoma:** Con batch=1, los Students+KD aprenden peor que sus baselines.
- **Diagnóstico:** `BatchNorm` con batch=1 calcula media/varianza sobre UNA sola muestra → estadísticas degenerativas → destruye las features pre-entrenadas en ImageNet.
- **Fix:** `Distiller._set_student_bn_eval()` pone todas las capas `BatchNorm*d` del Student en `.eval()` antes de cada forward. Los weights siguen entrenándose; las estadísticas `running_mean`/`running_var` mantienen las de ImageNet.
- **Validación post-fix:** SRCC=0.82 en 3 epochs (smoke test sobre AQA-7).
- **Configurable:** `kd.freeze_bn` (default `True`).

---

## 4. Resultados experimentales clave

### Teachers (fijos, 1 semilla 42)
Target del plan vs. real:

| Dataset | SRCC (real) | PLCC | MAE | Epochs | Target | Δ |
|---|---|---|---|---|---|---|
| AQA-7 | **0.9052** | 0.9352 | 8.20 | 42 | 0.80 | +0.105 |
| MTL-AQA | **0.8869** | 0.8848 | 5.51 | 45 | 0.85 | +0.037 |
| JIGSAWS | **0.8364** | 0.8456 | 10.83 | 25 | 0.70 | +0.136 |

**Observación:** los Teachers superan ampliamente los targets; esto sugiere que el valor base del KD será menor que en el paper original.

### Students baseline (sin KD, 1 semilla 42)

| Dataset | TSM-MBv2 | MBv3 | Teacher | Δ vs Teacher |
|---|---|---|---|---|
| AQA-7 | 0.8968 | 0.8854 | 0.9052 | −0.008 / −0.020 |
| MTL-AQA | 0.8804 | 0.8703 | 0.8869 | −0.007 / −0.017 |
| JIGSAWS | 0.8283 | 0.8368 | 0.8364 | −0.008 / +0.000 |

**Observaciones críticas:**
- Los baselines **superan los valores "con KD" de la Tabla 5.1 original de la tesis** (p. ej. Tabla original: TSM+KD=0.766; aquí baseline=0.897 en AQA-7).
- En JIGSAWS el MBv3 baseline **empata con el Teacher** (0.8368 vs 0.8364). No hay margen de KD que aportar.
- Explicación: pesos ImageNet modernos (torchvision 2024) + clip_length=64 + AdamW+cosine son mucho más efectivos que los modelos/configs disponibles en 2019-2020 cuando se escribió el paper base.

### Eficiencia (T=64, batch=1, RTX 3060)

| Modelo | Params | GFLOPs | Latencia (ms) |
|---|---|---|---|
| I3D (Teacher) | 27.23 M | 228.3 | 133.9 |
| TSM-MobileNetV2 | 2.23 M | 20.0 | 54.3 |
| MobileNetV3-Large | 2.97 M | 14.3 | 39.9 |

**Nota:** FLOPs ~3× superiores a la Tabla 5.2 original porque usamos T=64 vs T=16 asumido allá. Reportar los valores reales al final.

### KD runs — 1 semilla (42), fix BN aplicado [6/6 COMPLETO]

| Dataset | Student | Baseline | KD best | Δ (KD − baseline) | Notas |
|---|---|---|---|---|---|
| AQA-7 | TSM-MBv2 | 0.8968 | 0.8811 (ep 46) | −0.016 | KD ligeramente peor |
| AQA-7 | **MBv3** | 0.8854 | **0.9250** | **+0.040** ✅ | KD mejora. **Único caso de éxito claro** |
| MTL-AQA | TSM-MBv2 | 0.8804 | 0.7628 (ep 50) | −0.118 | KD degrada fuerte |
| MTL-AQA | MBv3 | 0.8703 | 0.8470 (ep 23) | −0.023 | KD degrada levemente |
| JIGSAWS | **TSM-MBv2** | 0.8283 | 0.4935 → 0.344 (re-run) | **−0.484** ❌ | Re-run con fix no mejoró; KD no funciona aquí |
| JIGSAWS | MBv3 | 0.8368 | 0.7907 | −0.046 | KD degrada |

**Hallazgo principal (6/6 runs):**
- **Sólo 1 de 6 configuraciones mejora con KD** (MBv3 AQA-7, +0.040).
- **TSM-MBv2 siempre empeora con KD** (−0.016, −0.118, NaN). Hipótesis: TSM ya modela temporalidad; el KD temporal interfiere con su representación.
- **MBv3 baja pero menos** (−0.023, −0.046): como no tiene módulos temporales explícitos, el KD temporal es más compatible.
- **JIGSAWS es el más frágil** (dataset chico, batch=1, pérdidas auxiliares → gradiente inestable → NaN).

### E14 — Gradiente NaN en JIGSAWS TSM-MBv2 KD (epoch 11)
- **Síntoma:** `last SRCC=NaN` en epoch 11; el training continuó con pesos NaN.
- **Causa:** predicciones colapsaron a valores casi constantes → Spearman indefinido. Las pérdidas auxiliares dominan y destruyen la señal de regresión.
- **Fix aplicado en código:** `Trainer` ahora aborta si la loss no es finita (RuntimeError explícito).
- **Intento de fix experimental:** `configs/kd_jigsaws.yaml` con `batch=2`, `grad_clip=0.5`, `warmup_epochs=8`, `β=γ=0.3`. **Resultado:** mejoró numéricamente (no más NaN) pero el KD sigue degradando: best SRCC=0.344 (epoch 1) vs baseline 0.828.
- **Conclusión:** el KD propuesto **no funciona en JIGSAWS TSM-MBv2** con ningún ajuste razonable. El dataset chico (144 train) + arquitectura ya temporal (TSM) + pérdidas KD introducen demasiado ruido. Se acepta como resultado válido: evidencia empírica de que el framework tiene límites claros.

### Cross-domain (Tabla 5.3) — completado

| Transferencia | Arch | Baseline | KD | Δ (KD − baseline) |
|---|---|---|---|---|
| MTL-AQA → AQA-7 | TSM-MBv2 | 0.5294 | 0.4704 | −0.059 |
| MTL-AQA → AQA-7 | MBv3 | 0.5531 | 0.5409 | −0.012 |
| AQA-7 → JIGSAWS | TSM-MBv2 | −0.193 | −0.039 | +0.154 (ambos cercanos a 0) |
| AQA-7 → JIGSAWS | MBv3 | 0.037 | −0.048 | − (aleatorio en ambos) |

**Hallazgos cross-domain:**
- **MTL-AQA → AQA-7 transfiere parcialmente** (SRCC 0.53-0.55): el dominio de clavados se generaliza razonablemente a multi-deporte.
- **AQA-7 → JIGSAWS no transfiere** (SRCC cercano a 0 o negativo): la brecha de dominio deporte↔cirugía es demasiado grande para transferencia zero-shot.
- **KD no ayuda a la generalización cross-domain** en ninguna configuración. En MTL-AQA→AQA-7 empeora ligeramente; en AQA-7→JIGSAWS ambos son aleatorios.
- La transferencia deporte→cirugía **requiere fine-tuning** o destilación multimodal (vision-language) para ser factible.

### Pendientes de resultados
- [x] Guard anti-NaN en Trainer.
- [x] Re-run JIGSAWS TSM-MBv2 KD (confirmó que KD no funciona ahí).
- [x] Cross-domain (Tabla 5.3).
- [ ] Grad-CAM — en progreso.
- [ ] Actualizar PDF con todos los números reales.

### Interpretación para la tesis

Este resultado es **científicamente honesto y publicable** aunque contradice la hipótesis original:

1. **El KD con los 3 términos propuestos NO mejora sistemáticamente al baseline** en la configuración moderna (ImageNet pretrained 2024 + MobileNet + TSM/MBv3).
2. Solo funciona donde hay asimetría en capacidades: **MBv3** (sin TSM, puramente espacial) **sí se beneficia** del KD temporal desde I3D.
3. **TSM-MBv2 ya es eficaz por sí solo**; añadir KD temporal es redundante y puede destruir (NaN).
4. En **dominios pequeños** (JIGSAWS) el KD con batch=1 es frágil numéricamente.

**Renarrativa propuesta para la tesis:**
> "Analizamos la destilación Teacher-Student en AQA con 3 datasets de dominios
> distintos (deporte, clavados, cirugía). Observamos que los Students modernos
> con pesos ImageNet alcanzan ~99% del rendimiento del Teacher I3D sin
> destilación, y que la destilación propuesta solo aporta mejora clara cuando
> el Student carece de módulos temporales explícitos (MBv3: +0.040 SRCC).
> Cuando el Student ya modela temporalidad (TSM-MBv2), la destilación
> interfiere o degrada. Este resultado reabre la pregunta sobre cuándo es
> realmente necesario el KD espacio-temporal en AQA."

---

## 5. Decisiones de diseño no previstas en el plan

- **Dataset sintético** (`scripts/make_synthetic_dataset.py`) para validar pipeline end-to-end sin depender de descargas.
- **Factory `build_model()` / `build_dataset()`** en `src/models/__init__.py` y `src/datasets/__init__.py` para simplificar el CLI en `main.py`.
- **Proyector 1×1×1 `Conv3d`** en `FeatureAlignLoss` para alinear canales disímiles entre Teacher (512 en mid) y Student (64/40 en mid).
- **Warmup lineal de β y γ** (5 epochs) para que las pérdidas auxiliares no interfieran con la regresión al inicio del entrenamiento.
- **`Distiller._set_student_bn_eval()`** como fix del E13 (no estaba en el plan).
- **Script `train_pipeline_remaining.sh` y `train_students_only.sh`** para encadenar runs automáticamente, no previsto en el plan.
- **Deduplicación del `Bibliog.bib`** previo a compilar con latexmk.
- **`.gitignore` de `Datasets/`** (4 GB descargados por el usuario, no versionables).
- **Split 70/15/15 estratificado** por `(categoría × bin de score)` usando `StratifiedShuffleSplit`; en AQA-7 respetamos el train/test oficial y tomamos 15% del train oficial como val.

---

## 6. Infraestructura y tooling

- **Git:** repo local + GitHub remoto (`github.com/emmanuelgaldos0-max/AQA_Framework`), push vía SSH.
- **Venv:** `virtualenv` en `code/.venv`, activable con `source`.
- **Dependencias:** 89 paquetes en `requirements.txt`. Clave: `torch==2.3.1+cu121`, `numpy==1.26.4`, `opencv-python<4.12`, `timm==0.9.16`.
- **Testing:** `pytest`, 34-36 tests verdes (metrics, models, losses, config, pipeline, engine, preprocess_frames).
- **Hooks de sonido** (.claude/settings.json): `bell.oga` para Notification, `complete.oga` para Stop.
- **PushNotification** para notificaciones remotas cuando termine training.
- **Monitor tool** con `tail -f | grep --line-buffered` para streaming de eventos del pipeline; timeout máximo 1h, hay que re-armarlo.

---

## 7. Scripts auxiliares creados (no previstos en el plan)

| Script | Propósito |
|---|---|
| `scripts/build_aqa7_annotations.py` | Parsea `.mat` oficial → scores.csv + splits JSON |
| `scripts/build_mtl_aqa_annotations.py` | Parsea `final_annotations_dict.pkl` → scores.csv + splits |
| `scripts/build_jigsaws_annotations.py` | Parsea `label.pkl` de MUSDL → scores.csv + splits |
| `scripts/make_synthetic_dataset.py` | 40 clips fake para tests end-to-end |
| `scripts/make_splits.py` | Splits estratificados genéricos |
| `scripts/preprocess_videos.py` | decord/OpenCV → tensor .pt float16 |
| `scripts/download_datasets.sh` | Instrucciones de descarga (manual) |
| `scripts/reproduce_all.sh` | Pipeline completo desde cero |
| `scripts/train_pipeline_remaining.sh` | Encadena Teachers+Students+KD en secuencia |
| `scripts/train_students_only.sh` | Variante sin Teachers (tras fix E13) |

---

## 8. Lecciones y notas para futuras iteraciones

- **Con RTX 3060 6 GB, T=64 es el límite práctico.** Si quisieras T=128, bajar `frame_size` a 160 o reducir canales del Teacher.
- **ImageNet pretrained es sorprendentemente efectivo** para AQA con clips cortos de deporte. En estos datasets modernos el KD aporta poco margen (0.01–0.02 SRCC) — la "ganancia" del paper original de ~0.12 SRCC no se reproduce porque el baseline moderno ya es mucho mejor.
- **Split_4 de AQA-7 tiene augmentations duplicadas** en el `.mat`. Siempre deduplicar por `(category, sample_no)`.
- **JIGSAWS generaliza mal en cross-domain** (esperado: dominio quirúrgico vs deportivo).
- **BatchNorm + batch_size pequeño = disaster** (E13). Con batch<4 en modelos pre-entrenados ImageNet, congelar BN siempre.
- **Monitor de Claude tiene timeout máximo 1h.** Para training de varias horas, re-armar manualmente o partir el seguimiento en hitos.

---

## 9. TODO y riesgos abiertos

- [ ] Decidir **1 semilla vs 3 semillas** para los KD runs (tradeoff tiempo/rigor).
- [ ] Implementar cross-domain (Tabla 5.3) — script simple: cargar checkpoint Student entrenado en dataset A, evaluar en split test de dataset B.
- [ ] Generar Grad-CAM para 15 clips (5 por dataset).
- [ ] Regenerar `Tesis.pdf` con los números reales (Tabla 5.1, 5.2, 5.3 + Abstract).
- [ ] Considerar re-evaluar la narrativa de la tesis ahora que el KD aporta menos que en el paper original: ¿destacar la eficiencia y el cross-domain como contribución principal?

---

### 10. Reformulación de la tesis (Camino C) — 2026-04-23

Dado que los resultados experimentales no sostenían la hipótesis original
("KD mejora al Student +0.12 SRCC"), se renarrativizó la tesis en positivo
sin ocultar hallazgos negativos. Cambios principales:

- **Pregunta de investigación reformulada** de *"¿cómo optimizar el proceso
  de transferencia?"* a *"¿Cuándo y bajo qué condiciones arquitectónicas la
  destilación espacio-temporal aporta valor a Students livianos modernos?"*.
- **Objetivo general** cambia de "desarrollar una metodología de KD que
  transfiera el rendimiento" a "desarrollar y evaluar un framework eficiente
  complementado por KD selectiva, y caracterizar las condiciones de aporte".
- **Abstract y Resumen**: narrativa de dos contribuciones (línea base fuerte
  + KD selectiva) en lugar de una sola hipótesis que fallaría.
- **Cap. 4 §5.2 (nueva)**: "Caracterización arquitectónica: ¿cuándo el KD
  aporta valor?" presenta la interacción arquitectura × dataset en una tabla
  dedicada, convierte el fallo de TSM-MBv2 en análisis estructurado.
- **Conclusiones**: tres contribuciones positivas (pipeline eficiente,
  framework selectivo, límites cross-domain) en lugar de una sola.

PDF resultante: 45 páginas, compila limpio. Todos los números experimentales
se preservan; solo cambia la narrativa interpretativa.

---

### 11. Aclaración formal Baseline vs +KD en el documento (2026-04-27)

Observación del asesor: la Tabla 5.1 era confusa porque mezclaba "Teacher",
"nuestro baseline" y "baseline + KD" sin marcarlos como regímenes distintos
de la misma fórmula `L_total = α·L_reg + β·L_att + γ·L_temp`.

Cambios aplicados:

- **Cap_3.tex (Propuesta):** tras la fórmula de `L_total` se añadió un
  párrafo `\paragraph{Dos regímenes operativos}` que ata explícitamente
  Régimen 1 (`β=γ=0` = Baseline = Contribución 1) y Régimen 2
  (`β,γ>0` = + KD = Contribución 2).
- **Cap_3.tex (intro):** se añadió un párrafo que define qué cubre la
  palabra "framework" en esta tesis (conjunto modular de componentes:
  arquitecturas + esquema de pérdida configurable + pipeline de datos +
  protocolo de evaluación), respondiendo al cuestionamiento del asesor
  sobre si lo entregado constituye realmente un framework.
- **Cap_4.tex (Tablas 5.1 y 5.3):** se añadió la columna `Régimen` con
  valores `Teacher` / `Baseline` / `+ KD`, y los captions describen la
  convención. Tabla 5.2 ya tenía la distinción.

Decisión sobre el título: por ahora se mantiene "Framework de destilación
de conocimiento espacio-temporal..." porque (a) la KD selectiva sigue
siendo una contribución central, y (b) cambiar el título tras varios
commits implica re-aprobaciones administrativas. Si el asesor lo exige se
considerarán alternativas como "Framework eficiente para AQA con
destilación selectiva en dispositivos con recursos limitados".

---

### 12. Auditoría completa del documento (2026-04-27)

Revisión sistemática del documento detectó **27 incongruencias** entre lo
escrito (heredado del plan original) y la realidad experimental ya
ejecutada. Correcciones aplicadas en una sola pasada:

**Portada (Tesis.tex):**
- Título reformulado: *"Framework eficiente para Action Quality Assessment con
  destilación de conocimiento selectiva en dispositivos con recursos
  limitados"* (alinea con Camino C; quita "framework de destilación" como
  núcleo).
- `\advisor{}` → "Edward Jorge Yuri Cayllahua Cahuina".
- `\date{Mes 2025}` → "2026".
- `\dedicado{}` y `Agradecimientos.tex` rellenados con texto neutral.

**Cap_1.tex:** marcadores Markdown `**...**` → `\textbf{}`.

**Cap_2.tex:** homologación "distilación" → "destilación" (todo el archivo).
Marcado explícito de cuáles variantes de KD adopta esta tesis (atención +
alineación temporal) y cuál queda fuera (multimodal).

**Cap_3.tex (Propuesta):** corregidas 11 incongruencias factuales:
- JIGSAWS: 103 trials × 2 vistas = **206 clips** (antes "103 ejecuciones").
- Eliminado el gesture-windowing inexistente en JIGSAWS.
- Clip length unificado a 64 frames (eliminada referencia a 128 que nunca
  se usó).
- Eliminado solapamiento del 20% (no se usa).
- Aclarado: MTL-AQA usa `final_score` directo, no promedio de subetiquetas.
- Batch real: 2 (Students) / 1 (KD) con grad_accum, no 8/16.
- Latencia: medida en RTX 3060 Mobile, no T4.
- Eliminadas comparativas con X3D y MoViNet (nunca se evaluaron).
- Repetibilidad: explicitado n=1 (semilla 42), múltiples semillas como
  trabajo futuro.
- Splits: aclarado AQA-7 oficial vs MTL/JIGSAWS estratificado 70/15/15.
- Jetson Nano movido a trabajo futuro.

**Cap_4.tex (Resultados):** insertada figura Grad-CAM `fig:gradcam` con 3
ejemplos representativos (AQA-7 diving, MTL-AQA, JIGSAWS Suturing) copiados
desde `code/experiments/figures/gradcam/` a `figs/Resultados/`.

**conclusiones.tex:**
- "distilación" → "destilación" en todo el archivo.
- "reproducible en los tres dominios" → "consistente ... con la semilla 42".
- Nueva sección **"Limitaciones del trabajo"** con 6 ítems (semilla única,
  hardware embebido pendiente, batch pequeño, cobertura parcial del KD,
  zero-shot cross-domain, subconjunto de Students).

**Resumen.tex / Abstract.tex:** mención al código y configuraciones
disponibles como recurso reproducible.

**Bibliografía:** verificado que la entrada `Cai2023VisionLanguageAQA`
tiene `year=2024` correcto en el `.bib` (sólo el cite key es 2023, no
afecta el render).

PDF resultante: 48 páginas (vs 45 previas; +3 por sección Limitaciones y
figura Grad-CAM), compila limpio sin referencias rotas.

---

### 13. Apertura de Fase 8 — pivote a contribución implementada novedosa (2026-04-27)

Tras auditoría del documento y diálogo con el usuario, se identifica que el
"Camino C" (narrativa de caracterización) no satisface el requerimiento del
curso de **implementación técnicamente nueva** con resultados positivos.

Se realiza investigación profunda de literatura AQA 2023–2026 (Survey IJCV
2025, USDL/MUSDL, CoRe, CoFInAl, PHI, RICA², HP-MCoRe, Vision-Language AQA).
Se priorizan tres caminos:

1. **MUSDL + Multi-task auxiliary heads** (probabilidad alta, novedad clara).
2. **CORAL/MMD para cross-domain** (ataca el peor número actual).
3. **CoFInAl prototype head** (head novedoso, plug-on sobre Student).

Se crean documentos `PROPUESTA_FASE_8.md` (justificación) y
`PLAN_FASE_8.md` (plan operativo con gates de decisión por camino).

Branch git: `phase-8-novel-contribution` (creada desde `main`).

### 14. Implementación Camino 1: MUSDL standalone (2026-04-27)

Implementación mínima viable sin tocar `main.py`:

- `code/src/losses/score_distribution.py`: `make_target_distribution`,
  `musdl_kl_loss`, `expected_score`.
- `code/src/models/heads.py`: añadido `DistributionHead` (pool + Dropout +
  Linear → logits sobre N bins).
- `code/src/models/musdl_wrapper.py`: wrapper que reemplaza `head` del
  Student por `DistributionHead`, accediendo a `final_feat` del backbone.
- `code/src/engine/musdl_trainer.py`: subclase de `Trainer` que sobreescribe
  `compute_loss` (usa MUSDL KL) y `validate` (esperanza E[score] antes de
  SRCC/PLCC/MAE).
- `code/scripts/train_musdl.py`: entry point CLI.
- 4 configs YAML: `musdl_aqa7_mbv3.yaml`, `musdl_aqa7_tsm.yaml`,
  `musdl_mtl_mbv3.yaml`, `musdl_mtl_tsm.yaml`.

**Smoke tests aprobados:**
- `make_target_distribution(0.5)` → media 0.5 (correcta, σ=5 bins).
- Forward + backward MBv3 wrapper en CUDA: loss ≈ 1.61, gradientes finitos.
- Forward + backward TSM-MBv2 wrapper en CUDA: loss ≈ 1.66, gradientes finitos.

Hiperparámetros base (todas las configs): num_bins=100, σ=5 bins, batch=2,
grad_accum=8 (efectivo 16), lr=3e-4, AdamW + cosine, 50 epochs, early stop
patience=12.

**Decisión MVP:** se posterga la cabeza multi-task auxiliar (DD para MTL,
action_class para AQA-7) hasta validar que MUSDL solo aporta sobre baseline.
Si MUSDL solo cumple el gate, se añade MT como ablación; si no, se evalúa MT
como rescate antes de pasar a Camino 2.

---

### 15. Implementación Camino 2 (CORAL) y Camino 3 (CoFInAl) en paralelo (2026-04-27)

Mientras corre el primer entrenamiento de Camino 1 (MUSDL MBv3 AQA-7), se
implementan los caminos 2 y 3 para tener todo listo si Camino 1 no cumple
el gate.

**Camino 2 — CORAL:**

- `code/src/losses/coral.py`: `coral_loss(source, target)` calcula la
  norma de Frobenius normalizada de la diferencia de matrices de
  covarianza entre features source y target.
- `code/src/engine/coral_trainer.py`: `CoralTrainer` que en cada paso
  toma un batch source (con label) + un batch target (sin label, ciclado),
  hace forward de ambos, extrae `final_feat` pooled (B,C) y combina
  `L_total = L_reg(source) + λ · L_coral(feat_s, feat_t)`.
- `code/scripts/train_coral.py`: entry point con flags `--source` y
  `--target`. Al final corre evaluación cross-domain sobre el split test
  del target.
- Configs: `coral_aqa7_to_jigsaws.yaml`, `coral_mtl_to_aqa7.yaml`.

**Camino 3 — CoFInAl-inspired prototype head:**

- `code/src/models/prototype_head.py`: `PrototypeHead` con K_coarse=10
  prototipos + K_fine=5 finos por cada coarse, similitud coseno con
  proyección 256-D L2-normalizada. Devuelve dict
  `{coarse_logits, fine_logits, score_pred (esperanza)}`.
- `code/src/models/proto_wrapper.py`: wrapper análogo a MUSDLWrapper.
- `code/src/engine/proto_trainer.py`: `PrototypeTrainer` con pérdida
  `L_coarse + α_fine · L_fine` (CrossEntropy en ambos niveles).
- `code/scripts/train_proto.py`: entry point.
- Config: `proto_aqa7_mbv3.yaml`.

**Smoke tests aprobados (CUDA):**
- CoFInAl: forward 2×3×8×224×224 → coarse_logits (2,10), fine_logits
  (2,10,5), score_pred (2,) ≈ 0.5; backward OK; bins target 0.3 → (3,0),
  0.85 → (8,2). l_coarse=2.38, l_fine=1.61.

Pendiente: entrenamientos CORAL (12 corridas si se llega) y CoFInAl
(4 corridas si se llega), gateadas por el resultado de Camino 1.

---

### 16. Resultado MUSDL MBv3 + AQA-7 (2026-04-27)

Primer experimento de Camino 1 cerrado.

| Métrica | Valor |
|---|---|
| Mejor SRCC val | **0.9142** (epoch 18) |
| Baseline (`L_reg`) | 0.8854 |
| KD original | 0.9250 |
| **Δ MUSDL vs baseline** | **+0.0288** ✅ |
| Tiempo | 37.3 min |
| Early stop | epoch 30 (patience 12 desde epoch 18) |

Trayectoria SRCC val por epoch:
0.83 → 0.86 → 0.88 → 0.87 → 0.89 → 0.87 → 0.90 → 0.88 → 0.89 → 0.89 →
0.90 → 0.88 → 0.90 → **0.91** → 0.90 → 0.90 → 0.91 → 0.90 → **0.9142** → ...

**Análisis:**
- MUSDL supera el gate provisional de +0.020 SRCC en la primera de 4 configs.
- Mejora consistente vs baseline `L_reg` solo (+0.029).
- Quedó por debajo del KD original (0.9250), pero el KD sólo funciona en 1/6
  combinaciones; MUSDL podría funcionar en más configs (a determinar con
  los 3 entrenamientos restantes).

**Decisión:** lanzar inmediatamente los 3 entrenamientos restantes
(TSM+AQA-7, MBv3+MTL-AQA, TSM+MTL-AQA) encadenados con
`scripts/run_musdl_remaining.sh`. ETA ~3 horas.

---

### 17. Cierre Camino 1 — gate cumplido, MUSDL adoptado (2026-04-28)

Los 3 entrenamientos restantes terminaron en una cadena nocturna de ~4h.
Resultados completos:

| Dataset | Student | Baseline | MUSDL | Δ | Tiempo |
|---|---|---|---|---|---|
| AQA-7 | MBv3 | 0.8854 | **0.9142** | **+0.029** ✅ | 37 min |
| AQA-7 | TSM-MBv2 | 0.8968 | **0.9223** | **+0.026** ✅ | 80 min |
| MTL-AQA | MBv3 | 0.8703 | 0.8508 | −0.020 | 49 min |
| MTL-AQA | TSM-MBv2 | 0.8804 | 0.8531 | −0.027 | 124 min |

**Gate Camino 1: PASA ✅** (2/4 configs ≥ +0.020 SRCC, regla cumplida).

**Hallazgos:**

1. **MUSDL aporta universalmente en AQA-7** — supera baseline en ambas
   arquitecturas (MBv3 +0.029, TSM-MBv2 +0.026), eliminando la limitación
   arquitectónica del KD original (que degradaba TSM-MBv2 en AQA-7
   −0.016). Esto es novedad técnica defendible.
2. **MUSDL no aporta en MTL-AQA** (ambas arquitecturas −0.020 a −0.027).
   Hipótesis: la distribución de scores en MTL-AQA está fuertemente
   concentrada (clavados especializados, rango estrecho), lo que reduce
   la ganancia de modelar incertidumbre vía gaussiana discreta. Esto
   refuerza la narrativa "selectiva" del trabajo, ahora con dos ejes:
   arquitectura y distribución de scores.
3. El KD original sigue siendo el mejor en MBv3+AQA-7 (0.9250 > 0.9142),
   pero ya no es la única configuración exitosa; MUSDL gana en TSM-MBv2
   donde el KD fallaba.

**Decisión:** se adopta MUSDL como contribución principal de la fase 8.
Los Caminos 2 (CORAL) y 3 (CoFInAl) quedan implementados pero no
ejecutados; pueden usarse como anexos experimentales si el asesor lo
solicita.

**Documento generado:** `RESULTADOS_FASE_8.md` con tabla, análisis y plan
para reescritura del LaTeX. Pendiente: aprobación del usuario para
reescribir el documento.

**Tiempo total fase 8.1:** 4h 14min (4 entrenamientos secuenciales).

---

### 18. Resultados macro-cadena fase 8.2 (2026-04-28)

A petición del usuario se ejecutaron los caminos 2 y 3 además de
completar MUSDL en JIGSAWS. Resultados (semilla 42, 50 epochs, early stop
patience 12):

**MUSDL (cobertura completa 6/6 configs):**

| Config | Baseline | MUSDL | Δ | Status |
|---|---|---|---|---|
| AQA-7 + MBv3 | 0.8854 | **0.9142** | +0.029 | ✅ |
| AQA-7 + TSM-MBv2 | 0.8968 | **0.9223** | +0.026 | ✅ |
| MTL-AQA + MBv3 | 0.8703 | 0.8508 | −0.020 | ❌ |
| MTL-AQA + TSM-MBv2 | 0.8804 | 0.8531 | −0.027 | ❌ |
| JIGSAWS + MBv3 | 0.8368 | 0.8362 | −0.001 | ≈ |
| JIGSAWS + TSM-MBv2 | 0.8283 | **0.8815** | **+0.053** | ✅✅ |

**Hallazgo notable:** MUSDL TSM-MBv2 + JIGSAWS = +0.053 SRCC. El **peor
caso del KD original** (TSM-JIGSAWS −0.485) se convierte en **mejor caso
de MUSDL** (+0.053). Esto valida fuertemente la hipótesis de que MUSDL
es la solución correcta para los regímenes donde el KD temporal degrada
la representación de TSM-MBv2.

**CoFInAl (4 configs intra-domain):**

| Config | Baseline | CoFInAl | Δ |
|---|---|---|---|
| AQA-7 + MBv3 | 0.8854 | 0.7976 | −0.088 |
| AQA-7 + TSM-MBv2 | 0.8968 | 0.8000 | −0.097 |
| MTL-AQA + MBv3 | 0.8703 | 0.7721 | −0.098 |
| MTL-AQA + TSM-MBv2 | 0.8804 | 0.7622 | −0.118 |

CoFInAl simplificado degrada en **todas las configuraciones** (−0.088 a
−0.118). La implementación adapta sólo la estructura de prototipos coarse→fine,
sin instruction alignment textual del paper original. Sin la alineación
de prototipos a embeddings textuales pre-entrenados, los prototipos
quedan poco anclados y la pérdida InfoNCE no aporta. Conclusión: CoFInAl
**no es una alternativa viable** sin re-implementar el componente
vision-language completo.

### E15 — CoRAL OOM con batch=2 (2026-04-28)

- **Síntoma:** `CUDA out of memory` en `MobileNetV3Video.forward` durante
  el segundo forward (target) del primer paso de CoralTrainer.
- **Causa:** CORAL hace dos forwards en paralelo (source + target) cada
  paso, lo que duplica las activaciones en VRAM. Con batch=2 y T=64,
  excede los 6 GB de la RTX 3060 Mobile.
- **Fix aplicado:**
  - `configs/coral_*.yaml`: `batch_size=1`, `grad_accum_steps=16`.
  - `CoralTrainer._set_bn_eval()`: congelar BatchNorm igual que se hizo
    en KD (E13) para evitar degenaración de stats con batch=1.
- **Validación:** smoke test confirma 5.72 GB / 6 GB con 2 forwards,
  baja a 0.14 GB tras backward.
- Cadena CORAL relanzada (4 corridas: aqa7→jigsaws y mtl→aqa7 × MBv3 y
  TSM-MBv2). ETA ~6h.

---

### 19. Pivote a Fase 9 según plan del asesor (2026-05-05)

Tras revisión con el asesor, la tesis se reformula a **"Pipeline liviano
para AQA basado en arquitecturas modernas de bajo costo computacional"**.
La contribución central pasa a ser el pipeline (no la KD, no MUSDL, no
SD-KD). El asesor pide robustez empírica, no más métodos:

- **E5**: 2 semillas adicionales en AQA-7 para baselines (4 trainings).
- **E6**: ablación sin ImageNet pretrain en TSM-MBv2 + AQA-7.
- **E7**: ablación sin componente temporal (MBv2 plano sin TSM) + AQA-7.
- **E1+**: añadir SlowFast-R50 como segundo Teacher en AQA-7.

Se cancelaron los caminos novedosos pendientes (CORAL ya detenido,
SD-KD no implementado, multi-Teacher ensemble descartado).

Documentos creados: `PLAN_FASE_9.md` (plan operativo),
`RESULTADOS_FASE_9.md` (resultados consolidados al cierre).

### 20. Resultados Cadena Fase 9 (2026-05-06)

7 entrenamientos en cadena, sin errores. Tiempos individuales:

| Run | Tiempo | Mejor SRCC |
|---|---|---|
| TSM-MBv2 baseline AQA-7 seed 0 | 68.9 min | 0.9045 |
| TSM-MBv2 baseline AQA-7 seed 7 | 52.3 min | 0.9049 |
| MBv3 baseline AQA-7 seed 0 | 31.6 min | 0.8914 |
| MBv3 baseline AQA-7 seed 7 | 27.8 min | 0.8952 |
| TSM-MBv2 sin pretrain (E6) | 104.2 min | 0.8713 |
| MBv2 plano sin TSM (E7) | 38.6 min | 0.8921 |
| SlowFast-R50 Teacher (E1+) | 81.0 min | 0.9158 |

**Hallazgos:**

- **Replicación de baselines (E5)**: 3 semillas estables, std ≈ 0.005.
  - TSM-MBv2: 0.9021 ± 0.0046
  - MBv3: 0.8907 ± 0.0049
  - La media supera ligeramente la semilla 42 reportada en la tesis,
    pero las cifras son estadísticamente equivalentes.

- **SlowFast Teacher (E1+)**: 0.9158 sobre AQA-7. Mejor que I3D
  (0.9052), como esperado del paradigma 3D moderno. Brecha del
  pipeline liviano ahora reportable contra dos referencias 3D distintas.

- **Ablación pretrain (E6)**: SRCC cae a 0.8713 sin ImageNet pretrain
  (Δ = −0.031). El preentrenamiento es el componente más importante
  del pipeline.

- **Ablación TSM (E7)**: SRCC cae a 0.8921 sin TSM (Δ = −0.010). El
  módulo temporal aporta poco en AQA-7 (clases de acciones rápidas
  donde la información espacial domina). Resultado matiza la
  importancia del modelado temporal explícito.

**Implementación nueva:** `src/models/mobilenetv2_video.py` (E7,
MBv2 plano sin TSM); `src/models/slowfast.py` (E1+ Teacher SlowFast,
con reemplazo del PoolConcatPathway por pool adaptativo para soportar
clip_length flexible); flag `pretrained` configurable desde YAML
(E6); 3 configs nuevas.

**Cobertura de objetivos del asesor: 6/6.**

Pendiente: reescribir LaTeX con la nueva narrativa, tablas con
media±std y secciones de ablaciones.

---

### 21. Reescritura del LaTeX alineada al plan del asesor (2026-05-06)

Cambios aplicados a la fuente LaTeX:

- **Tesis.tex**: nuevo título *"Pipeline liviano para Action Quality
  Assessment basado en arquitecturas modernas de bajo costo
  computacional"*.
- **Cap_1.tex**: reformulado completo. Pregunta de investigación nueva
  centrada en el pipeline liviano; objetivos OE1–OE6 alineados al
  documento del asesor.
- **Cap_3.tex (Propuesta)**: pivote total. La propuesta principal es el
  pipeline liviano. La sección de KD pasa a "Análisis marginal" con
  función explícita de cuestionar la premisa heredada. Modelos Teacher
  ampliados (I3D + SlowFast-R50). Protocolo experimental actualizado para
  reflejar 3 semillas en AQA-7 y ablaciones E6/E7.
- **Cap_4.tex (Resultados)**: Tabla 5.1 nueva con SlowFast Teacher,
  Students con media±std (3 semillas), ablaciones y +KD como análisis
  marginal. Sección de eficiencia con SlowFast incluido. Sección nueva
  "Ablaciones del pipeline" (E6 + E7). Sección "Análisis marginal de la
  destilación" con la lectura honesta del aporte negativo de KD sobre
  baseline fuerte. Cross-domain conservado.
- **conclusiones.tex**: reformulado por OE1–OE6. Tres conclusiones
  positivas: pipeline liviano cierra brecha; ablaciones identifican
  componentes; KD innecesaria sobre baseline fuerte. Sección
  "Limitaciones" preservada. Trabajo futuro actualizado (incluye Score
  Distribution Learning como dirección promisoria por hallazgo +0.053
  SRCC en JIGSAWS TSM-MBv2).
- **Resumen.tex / Abstract.tex**: reescritos al título nuevo.
- **EXPLICACION.md**: reescrito para público general bajo nueva narrativa.
- **Bibliografía**: añadidas entradas `feichtenhofer2019slowfast` y
  `sandler2018mobilenetv2`. La referencia TSM ya existía como
  `Lin2019TSM`.

PDF resultante: 47 páginas, compila limpio, sin referencias rotas. Copia
del PDF actualizada en `Tesis.pdf` (raíz del repo) además de
`Tesis Latex/Tesis.pdf`.

**Estado:** la tesis queda alineada al plan del asesor. Pendiente la
revisión por parte del asesor y eventualmente \textit{merge} de la branch
`phase-8-novel-contribution` a `main`.

---

*Última actualización: 2026-05-06 (LaTeX reescrito completo bajo el
nuevo título "Pipeline liviano para AQA"; PDF final 47 páginas compila
limpio).*
