# Intelligent Ultrasound Digital Twin

Prototype de jumeau numérique orienté ingénierie pour un système d'échographie.

Le projet vise à construire une représentation numérique traçable du comportement d'un système d'imagerie ultrasonore à partir de données échographiques, de caractéristiques quantitatives d'image et de simulations contrôlées.

## Objectif

Le Digital Twin combine :

```text
Ultrasound Data
      ↓
Preprocessing
      ↓
Digital Signature / Image Features
      ↓
Quality Assessment
      ↓
Statistical Baseline
      ↓
Digital Twin State
      ↓
Controlled Degradation Simulation
      ↓
Anomaly Detection
      ↓
Drift / Trend Analysis
      ↓
Prediction & Engineering Dashboard
```

L'objectif n'est pas de diagnostiquer un patient ni de prouver automatiquement une panne matérielle. Le système fournit un cadre d'analyse et de surveillance de la qualité d'imagerie.

## Data strategy

Le pipeline distingue explicitement trois origines :

- **experimental** : acquisitions provenant d'un système d'échographie physique ;
- **public** : datasets publics d'échographie utilisés pour le développement et le benchmark ;
- **simulated** : données générées ou images soumises à des dégradations contrôlées.

Cette séparation empêche une donnée publique ou simulée d'être présentée comme une mesure spécifique d'un équipement physique.

Des données provenant ultérieurement d'un équipement particulier peuvent être intégrées comme **cas expérimental / validation**, sans modifier le cœur du Digital Twin.

## Digital Signature

La signature numérique décrit une image échographique par des caractéristiques quantitatives telles que :

- intensité moyenne et dispersion ;
- dynamique et contraste ;
- entropie et uniformité ;
- indicateurs de speckle ;
- densité de contours ;
- statistiques du gradient ;
- netteté Laplacienne.

Les caractéristiques sont utilisées pour construire une représentation multivariée de l'état d'imagerie.

## Quality monitoring

Le module de qualité évalue les propriétés de l'image et leur cohérence avec une population de référence.

Les résultats peuvent inclure :

- quality score ;
- contrast ;
- noise-related indicators ;
- sharpness ;
- edge quality ;
- texture / speckle indicators.

Une distance statistique élevée indique une déviation par rapport à la population de référence ; elle n'identifie pas à elle seule la cause physique.

## Digital Twin and simulation

Le Digital Twin représente l'état observé du système d'imagerie dans l'espace des caractéristiques.

Le moteur de simulation permet de créer des scénarios contrôlés, par exemple :

- bruit croissant ;
- flou progressif ;
- réduction du contraste ;
- déplacement d'intensité ;
- modification du speckle.

Les scénarios servent à étudier la sensibilité du pipeline et à valider son comportement logiciel. Une dégradation simulée ne constitue pas une preuve de panne physique.

## Anomaly detection

Le détecteur statistique compare les signatures aux distributions de référence et produit :

- distance de Mahalanobis ;
- état du Digital Twin ;
- contributions des caractéristiques ;
- indicateurs de déviation.

## Drift and trend

Le monitoring temporel recherche une évolution persistante de la signature ou de la qualité.

Les résultats de tendance sont des indicateurs d'évolution statistique. Ils ne doivent pas être interprétés comme une date de panne, une RUL ou une probabilité clinique sans données expérimentales et validation appropriées.

## Validation strategy

La validation est organisée en niveaux :

1. **Validation logicielle** — tests unitaires et cohérence du pipeline.
2. **Validation par données publiques** — benchmark méthodologique sur échographie.
3. **Validation par simulations contrôlées** — dégradations connues et reproductibles.
4. **Validation expérimentale** — acquisitions répétées sur un système physique lorsque disponibles.
5. **Perturbations contrôlées** — uniquement si elles sont sûres, autorisées et documentées.

## Project structure

```text
src/
├── acquisition/       # ingestion, metadata, provenance
├── image_analysis/    # image features and quality
├── signature/         # baseline and statistical representation
├── anomaly/           # anomaly detection
├── digital_twin/      # twin state and orchestration
├── simulation/        # controlled degradation scenarios
├── drift/             # temporal monitoring
├── prediction/        # trend analysis
├── explainability/    # feature contributions
├── validation/        # validation metrics and experiments
└── reporting/         # engineering reports

dashboard/              # Streamlit interface
tests/                  # automated tests
docs/                   # methodology and architecture
```

## Execution

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

Tests :

```bash
python -m pytest tests -q
```

## Scientific boundary

The project is an engineering research prototype.

It is not a clinical diagnostic system, does not establish patient-specific conclusions, and does not automatically prove hardware failure. Experimental conclusions about a specific ultrasound system require traceable real acquisitions and appropriate validation.
