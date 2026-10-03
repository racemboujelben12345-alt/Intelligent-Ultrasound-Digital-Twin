# SCAN A — Intelligent Ultrasound Digital Twin V3

Prototype de jumeau numérique orienté ingénierie pour le monitoring statistique et temporel d'un système d'échographie SCAN A.

## Architecture

```text
Acquisition
     ↓
Digital Signature
     ↓
Statistical Baseline
     ↓
Anomaly Detection
     ↓
Digital Twin State
     ↓
Drift
     ↓
Prediction / Trend
     ↓
Validation
     ↓
Reporting
     ↓
Dashboard
```

Le pipeline exécuté sépare explicitement la baseline, la calibration temporelle et les acquisitions de test.

## Exécution

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

Les tests logiciels :

```bash
python -m pytest tests -q
```

## Données SCAN A réelles

Placer les acquisitions image dans :

```text
data/raw/scan_a/
```

Formats supportés par le pipeline image : PNG, JPG/JPEG et TIFF selon les loaders utilisés.

La source `scan_a` doit rester distincte des données de démonstration et des datasets publics.

La partition nominale actuelle est :

- **Baseline** : 30 acquisitions
- **Calibration** : 10 acquisitions nominales indépendantes
- **Test** : acquisitions restantes

La constante centrale `MIN_TOTAL_ACQUISITIONS` définit le minimum requis pour exécuter le pipeline.

## Données simulées et datasets publics

Si suffisamment d'acquisitions SCAN A réelles ne sont pas disponibles, le mode automatique peut utiliser les données simulées prévues par le projet.

Les datasets publics présents dans `data/raw/public_ultrasound/` servent au développement, au benchmark et à la vérification logicielle du pipeline. Ils ne doivent jamais être utilisés silencieusement pour construire la baseline réelle du SCAN A.

Une dégradation virtuelle constitue une **validation logicielle contrôlée** : elle ne prouve pas l'existence d'une panne physique sur le système SCAN A.

## Digital Signature

La signature numérique actuelle décrit l'acquisition par des caractéristiques quantitatives d'image, notamment :

- intensité moyenne et dispersion ;
- dynamique et contraste ;
- entropie et uniformité ;
- indicateurs de speckle ;
- densité de contours ;
- statistiques du gradient ;
- netteté Laplacienne.

La signature est ensuite comparée à une baseline statistique, notamment via la distance de Mahalanobis.

> Une distance de Mahalanobis élevée indique une déviation statistique par rapport à la population de référence. Elle n'identifie pas à elle seule la cause physique de cette déviation.

### Cas A-scan

Si le SCAN A fournit un signal A-scan 1D plutôt qu'une image, l'étage de signature doit être adapté à des caractéristiques de signal telles que l'amplitude, le SNR, la largeur des échos et les caractéristiques spectrales. L'architecture globale reste inchangée.

## Drift et tendance

Le monitoring temporel utilise une calibration indépendante et des méthodes de contrôle telles que EWMA/CUSUM.

La composante de prédiction actuelle est une **analyse de tendance temporelle**. Elle ne doit pas être interprétée comme :

- une date de panne ;
- une durée de vie résiduelle (RUL) ;
- une probabilité clinique ;
- une preuve de défaillance matérielle.

## Validation

La validation est organisée en niveaux :

1. **Validation logicielle** : tests unitaires et tests de cohérence du pipeline.
2. **Validation par données simulées** : vérification du comportement face à des déviations contrôlées.
3. **Validation expérimentale SCAN A** : répétabilité des acquisitions nominales.
4. **Validation par perturbations contrôlées**, uniquement si elles sont sûres, documentées et autorisées sur l'équipement.

Les métriques de classification disponibles dans `src/validation/metrics.py` nécessitent une vérité terrain explicitement définie. Elles ne doivent pas être calculées en inventant des labels de panne physique.

## Artefacts générés

Après `python main.py`, le dossier `outputs/` contient notamment :

- `summary.json` — résumé du run ;
- `baseline.json` — paramètres et statistiques de référence ;
- `twin_states.csv` — états temporels du Digital Twin ;
- `features.csv` — Digital Signatures des acquisitions de test ;
- `report.md` — rapport Markdown automatiquement généré.

Le dashboard Streamlit lit ces artefacts sans modifier le moteur scientifique.

## Documentation

- `docs/EXPERT_ARCHITECTURE.md`
- `docs/PROTOCOLE_EXPERIMENTAL.md`
- `docs/METHODOLOGIE.md`

## Principe scientifique

Le projet vise à construire une représentation numérique traçable du comportement observé du SCAN A à partir d'acquisitions contrôlées. La distinction entre observation statistique, simulation numérique et validation physique est conservée à chaque étape.
