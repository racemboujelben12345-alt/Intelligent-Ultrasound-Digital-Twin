# Protocole expérimental — Système d'échographie

## 1. Avant la première acquisition
Documenter uniquement ce qui est réellement observé sur l'équipement :

| Élément | Observation |
|---|---|
| Format de sortie : image 2D, A-scan 1D, capture, DICOM/BMP… | |
| Réglages accessibles : gain, profondeur, fréquence, TGC, focus… | |
| Sonde/probe | |
| Fantôme ou référence disponible | |
| Mode d'export | |

Si la sortie est un A-scan 1D, prévoir une branche d'analyse de signal complémentaire aux features image.

## 2. Conditions fixes
Pour une série nominale, conserver autant que possible :

- même salle ;
- même fantôme/référence ;
- même position et angle de la sonde ;
- mêmes réglages ;
- même opérateur si possible ;
- mêmes conditions de préparation.

Noter l'identifiant, la date/heure, la session, les réglages et les remarques.

## 3. Plan d'acquisition recommandé
| Bloc | Contenu | Rôle |
|---|---|---|
| B1 | acquisitions répétées sous conditions nominales | construire la référence |
| B2 | acquisitions nominales indépendantes à un autre moment | estimer les faux positifs |
| B3 | variations non destructives et autorisées d'un paramètre | tester la sensibilité du Digital Twin |
| B4 | séries espacées dans le temps | étudier une évolution réelle si disponible |

Les perturbations physiques ne doivent être réalisées que si elles sont sûres, autorisées et documentées.

## 4. Nommage
Utiliser un identifiant unique et conserver les métadonnées dans un fichier associé. L'ordre temporel doit être explicitement représenté par le timestamp plutôt que déduit uniquement du nom de fichier.

## 5. Critères de validation
Reporter :

- répétabilité des features ;
- variabilité du baseline ;
- faux positifs sur le holdout ;
- évolution du quality score ;
- réponse aux dégradations simulées ;
- cohérence entre simulation et observations expérimentales lorsque celles-ci existent.

## 6. Cas d'un appareil particulier
Si des données provenant d'un modèle précis d'échographe sont obtenues, elles sont intégrées comme **experimental case study**. Le projet conserve son architecture générique et le nom du modèle ne devient pas une dépendance du Digital Twin.
