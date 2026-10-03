# SCAN A Digital Twin V2
Lancer : `pip install -r requirements.txt && python main.py` puis `streamlit run dashboard/app.py`.

## Vraies données SCAN A
Mets tes images (png/jpg/tif, ordre chronologique du nom de fichier) dans `data/raw/scan_a/`.
Si >= N_BASELINE + N_HOLDOUT (40) images : le pipeline les utilise automatiquement (source='scan_a').
Sinon : démo SIMULÉE (source='simulated'). Les 30 premières = baseline nominal, les 10 suivantes = holdout nominal.
Règle : datasets publics (`data/raw/public_ultrasound/`) = développement du pipeline uniquement, jamais la signature.

## Attention
Si SCAN A sort un A-scan (signal 1D) et non une image, remplace `twin/image_analysis.py` par des features du signal (amplitude, SNR, largeur d'écho, spectre).

## Modules
data_engine -> image_analysis -> signature (Mahalanobis+LedoitWolf, IQS, Health State) -> drift (EWMA/CUSUM) -> anomaly (IF, OCSVM, PCA-AE, RF scénarios) -> simulation (Fault Injection) -> adaptive -> prediction -> validation

## Datasets publics
`python tools/public_benchmark.py chemin/dataset` (images png/jpg/tif ; masques ignorés ; DICOM à convertir).

## Docs & tests
`docs/PROTOCOLE_EXPERIMENTAL.md`, `docs/METHODOLOGIE.md`, `python -m pytest tests`.
