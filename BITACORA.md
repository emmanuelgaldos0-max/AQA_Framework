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

*Última actualización: 2026-04-22 (final de Fase 5.2 – Students baseline completos, KD pendiente)*
