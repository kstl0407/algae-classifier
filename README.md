# Holographic Algae Classifier

ConvNeXt-Tiny alapú képosztályozó holografikus algaképekhez. A modell egyszerre old meg egy **5-osztályos** és egy **bináris** (chlorella / nem-chlorella) feladatot, egyetlen kettős fejű hálózattal.

## Osztályok

| Label | Osztály |
|-------|---------|
| 0 | chlorella |
| 1 | debris |
| 2 | haematococcus |
| 3 | small\_haemato |
| 4 | small\_particle |

## Adatstruktúra

```
contest/
├── train/
│   ├── class_chlorella/        # *_amp.png, *_phase.png, *_mask.png
│   ├── class_debris/
│   ├── class_haematococcus/
│   ├── class_small_haemato/
│   └── class_small_particle/
└── test/
    └── 1.png ... N.png
```

Minden mintához 3 csatorna: amplitúdó (`_amp`), fázis (`_phase`), maszk (`_mask`). Ha valamelyik hiányzik, az amplitúdó másolata kerül a helyére.


## Kimenet

| Fájl | Tartalom |
|------|----------|
| `submission_multiclass.csv` | 5-osztályos predikció (TARGET: 0–4) |
| `submission_binary.csv` | Bináris predikció (TARGET: 0=chlorella, 1=más) |
| `best_model.pth` | Legjobb checkpoint + optimális küszöbök |

## Modell

- **Alap:** ConvNeXt-Tiny (ImageNet előtanított)
- **Fejek:** multi-class (5 osztály) + binary (chlorella vs. többi)
- **Loss:** CrossEntropyLoss (chlorella 2.5× súlyozva) + Focal Loss
- **Regularizáció:** WeightedRandomSampler, OneCycleLR, early stopping (patience=7)
- **Threshold:** validációs 2D grid-kereséssel (threshold × margin), teszt adatot nem lát


