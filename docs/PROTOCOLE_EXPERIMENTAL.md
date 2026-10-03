# Protocole expérimental SCAN A (à valider avec Dr. Nssibi)

## 1. Avant la première acquisition (Phase 1 — à remplir, rien n'est inventé)
| Question | Réponse constatée sur l'équipement |
|---|---|
| Format de sortie : image 2D, A-scan 1D, capture d'écran, export DICOM/BMP ? | |
| Réglages accessibles (gain, profondeur, fréquence, TGC, focus) | |
| Fantôme disponible (type, structures, âge) | |
| Mode d'export des données (clé USB, photo d'écran) | |
Si la sortie est un A-scan : remplacer `twin/image_analysis.py` par des features de signal.

## 2. Conditions fixes pour toute une série
Même salle, même fantôme, même position/angle du capteur (repère physique), même quantité de gel, mêmes réglages, même opérateur si possible. Noter dans un tableau : id, date/heure, opérateur, réglages, remarques.

## 3. Plan d'acquisition minimal
| Bloc | Contenu | Rôle |
|---|---|---|
| B1 | ≥ 30 acquisitions répétées, mêmes conditions | baseline (Digital Signature) |
| B2 | ≥ 10 acquisitions répétées, autre moment | holdout nominal (faux positifs) |
| B3 | Variations réelles NON destructives : gain ±, profondeur ±, gel réduit, pression/angle modifiés | sensibilité réelle, comparée aux dégradations virtuelles |
| B4 | Séries espacées dans le temps (jours/semaines) | seul moyen d'étudier une vraie évolution |

## 4. Nommage
`YYYYMMDD_HHMM_bloc_nn.png` (l'ordre alphabétique = ordre chronologique, exigé par le loader).

## 5. Critères d'acceptation
Répétabilité : CV du baseline reportés par feature (voir rapport). Une feature avec CV très élevé en conditions identiques n'est pas exploitable.
