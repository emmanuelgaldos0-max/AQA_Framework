# Auditoría de la tesis — hallazgos y recomendaciones

> Aplicación de los criterios definidos en `CRITERIOS_AUDITORIA_TESIS.md`
> al documento actual `Tesis Latex/Tesis.pdf` (47 páginas, compila limpio
> sin referencias rotas).
>
> Clasificación por severidad: 🔴 crítico (visible, afecta seriedad
> académica), 🟡 medio (mejorable, no urgente), 🟢 cosmético.

---

## 🔴 Críticos (recomiendo arreglar antes de mostrar)

### A1. Duplicación de acrónimo "Knowledge Distillation" en Cap_1

**Ubicación:** `Cap_1.tex` línea 15, renderizado en PDF página 14.

**Síntoma:** el PDF muestra
> *"destilación de conocimiento (Knowledge Distillation, Knowledge
> Distillation (KD))"*

**Causa:** el texto combina `(\textit{Knowledge Distillation}, \ac{KD})`.
En la primera invocación, `\ac{KD}` expande a *"Knowledge Distillation
(KD)"*, lo que duplica el término ya escrito manualmente.

**Fix:** reemplazar por `(\acf{KD})` o eliminar el `\textit{Knowledge
Distillation}` redundante.

### A2. Acrónimo `\ac{VRAM}` no definido

**Ubicación:** `conclusiones.tex` línea 42 (y otras menciones).

**Síntoma:** el paquete `acronym` no encuentra la entrada y el render
puede mostrarse incompleto o como placeholder según versión.

**Fix:** añadir `\acro{VRAM}{\textit{Video RAM}}` en `abreviaturas.tex`.

### A3. Secciones huérfanas al pie de página

**Ubicaciones identificadas (renderizado actual del PDF):**

- Página 17: termina con `2.1.1.` sin contenido visible debajo.
- Página 21: inicia con `3.1. Destilación en AQA: avances recientes` y
  el cuerpo del texto aparece sólo en la página 22.
- Página 24: termina con `3.1.4.` sin contenido.
- Página 28: termina con `4.2.1.` sin contenido.

**Causa:** LaTeX colocó el `\subsection` o `\subsubsection` a unas pocas
líneas del final de página y no rellenó.

**Fix:** insertar `\needspace{4\baselineskip}` (paquete `needspace`) o
`\FloatBarrier` antes del título problemático para forzar que el título
no quede aislado.

### A4. Nombre de capítulo inconsistente en "Organización de la tesis"

**Ubicación:** `Cap_1.tex` líneas 58–66.

**Síntoma:** dice *"Capítulo 5: Resultados"* y *"Capítulo 6:
Conclusiones"* pero los títulos reales son *"Pruebas y Resultados"* y
*"Conclusiones y Trabajos Futuros"*.

**Fix:** uniformar a los títulos reales.

## 🟡 Medios (mejoran calidad académica)

### B1. Primer uso de I3D sin definir como acrónimo

**Ubicación:** `Cap_1.tex` línea 9.

**Síntoma:** *"como I3D y SlowFast"* aparece sin expansión previa del
acrónimo. La expansión natural debería ocurrir aquí, no en `Cap_2`.

**Fix:** cambiar `I3D` por `\ac{I3D}` en su primer uso (y `SlowFast`
queda como nombre propio, no necesita expansión).

### B2. Términos en inglés sin itálica consistente

**Ubicaciones:** *baseline, batch, batch size, frame, frames,
gradient accumulation, pipeline, pretrain, framework, fine-tuning,
zero-shot* aparecen en algunos lugares con `\textit{}` y en otros
sin él.

**Fix:** revisar con `grep` y normalizar (preferiblemente `\textit{}`
para los anglicismos técnicos).

### B3. Acrónimo `AMP` (*Automatic Mixed Precision*) no formalizado

**Ubicación:** múltiple (`Cap_3.tex`, `conclusiones.tex`).

**Fix:** añadir `\acro{AMP}{\textit{Automatic Mixed Precision}}` y usar
`\ac{AMP}` en el primer uso.

### B4. Tablas: alineación de decimales

**Ubicación:** Tabla 5.1 (`tab:precision_aqa7`).

**Síntoma:** los valores `0.9021 ± 0.0046` se alinean con `c` (centro),
lo que dificulta comparar visualmente columnas numéricas.

**Fix:** considerar `siunitx` con columna `S` para alineación por punto
decimal, o ajustar a `r` con padding.

### B5. Tabla `tab:precision_aqa7` muestra "—" para SlowFast en PLCC/MAE

**Ubicación:** Cap_4 Tabla 5.1.

**Síntoma:** el SlowFast Teacher se reporta sólo con SRCC; PLCC y MAE
aparecen como "—". Esto es honesto pero un revisor puede preguntar por
qué no se midieron.

**Fix:** opción A — calcular PLCC y MAE del SlowFast con el checkpoint
guardado y rellenar. Opción B — añadir nota al pie explicando que se
reporta sólo SRCC por la métrica clave del campo. Recomiendo A.

### B6. Capítulos sin cierre que conecte al siguiente

**Ubicación:** Cap_2 (Marco Teórico), Cap_3 (Estado del Arte).

**Síntoma:** terminan abruptamente sin un párrafo de transición.

**Fix:** añadir 1 párrafo de cierre al final de cada uno (*"En el
siguiente capítulo se presenta..."*).

### B7. Resumen y Abstract son párrafos únicos muy largos

**Ubicación:** `Resumen.tex`, `Abstract.tex`.

**Síntoma:** ambos textos son un solo párrafo de ~300 palabras. La
norma académica suele preferir 2–3 párrafos con saltos lógicos
(problema, propuesta, resultados).

**Fix:** dividir en 2–3 párrafos sin cambiar el contenido.

### B8. Numeración mixta de acrónimos en español/inglés

**Ubicaciones:** alternativa entre "FLOPs (G)" y "GFLOPs" en distintos
puntos.

**Fix:** decidir convención (recomiendo "GFLOPs" en cuerpo, "FLOPs (G)"
sólo en cabeceras de tabla).

## 🟢 Cosméticos (opcional)

### C1. Decimales: estilo de notación

Uso mixto: `0.005` (notación inglesa) y `0,9021` (notación española).
Recomendación: dado que la tesis es en español, mantener coma decimal
en cuerpo. El `RESUMEN_EJECUTIVO.md` ya usa coma; el `.tex` usa punto.
Cambio masivo riesgoso; lo dejo como opcional.

### C2. Espacios no rompibles

Verificar que todas las referencias usen `~`:
- *"Tabla 5.1"* → *"Tabla~5.1"* (que ya usamos como `Tabla~\ref{...}`).
- *"6 GB"* → *"6~GB"* (la mayoría está bien, hay alguno suelto).

### C3. Bibliografía: formato de inicial

Algunas entradas tienen `J. Lin` y otras `Junyang Lin`. La mayoría
están bien con iniciales; verificar consistencia.

### C4. Citas sin contexto en Cap_3

Algunas citas se cierran con `[Yu et al., 2021]` directamente al final
de un párrafo sin haber explicado el trabajo. No es error, pero la
narrativa mejora si cada cita introduce mínimamente el método citado.

### C5. Footer redundante

El template UCSP imprime *"Escuela Profesional de Ciencia de la
Computación"* en cada página + número de página + cabecera con el
mismo texto. Es consistente con la plantilla institucional, pero
crea redundancia visual. Probablemente requerimiento de la
universidad — no tocar.

---

## Resumen ejecutivo de la auditoría

| Categoría | Hallazgos |
|---|---|
| 🔴 Críticos | 4 (duplicación acrónimo, VRAM no definido, 4 secciones huérfanas, nombres de cap inconsistentes) |
| 🟡 Medios | 8 (acrónimos, itálicas, alineación de tablas, métricas SlowFast, cierres de capítulo, Resumen monolítico, GFLOPs) |
| 🟢 Cosméticos | 5 (decimales español/inglés, espacios no rompibles, bibliografía, citas, footer) |
| **Total** | **17 hallazgos** |

**Compilación actual:** ✅ limpia (sin errores, sin warnings críticos,
sin referencias rotas).

**Recomendación:** aplicar los 4 hallazgos críticos (A1–A4) ya;
opcionalmente los medios (B1–B8). Los cosméticos pueden quedarse para
una pasada final antes de la entrega.

¿Procedo a aplicar los críticos automáticamente?
