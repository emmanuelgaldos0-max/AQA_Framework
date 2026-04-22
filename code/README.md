# AQA Framework — Código

Implementación del framework de destilación de conocimiento espacio-temporal para Action Quality Assessment, correspondiente a la tesis de Emmanuel Samir Galdos Rodriguez (UCSP).

## Estructura

```
code/
├── configs/        # YAML de hiperparámetros por experimento
├── data/           # datasets (ignorados en git)
├── src/
│   ├── datasets/   # loaders de MIT-Diving, AQA-7, MTL-AQA
│   ├── models/     # I3D (Teacher), TSM-MobileNetV2, MobileNetV3
│   ├── losses/     # regresión, atención KD, alineación temporal
│   ├── engine/     # Trainer, Distiller, Evaluator
│   └── utils/      # métricas (SRCC/PLCC/MAE), FLOPs, latencia
├── scripts/        # entrenamiento, evaluación, preprocesamiento
├── tests/          # pytest
├── experiments/    # checkpoints y logs (ignorados)
└── notebooks/
```

## Uso rápido

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Entrenar Teacher
python -m src.main --config configs/teacher_i3d.yaml --dataset aqa7 --seed 42

# Entrenar Student + KD
python -m src.main --config configs/kd.yaml --student tsm_mbv2 --dataset aqa7 --seed 42

# Evaluar
python -m src.main --mode eval --ckpt experiments/<run>/best.pth
```

Las métricas objetivo y el plan completo están en `../PLAN_DESARROLLO.md`.
