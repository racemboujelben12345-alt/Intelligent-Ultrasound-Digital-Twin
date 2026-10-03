# Méthodologie et justification des choix

**Statut** : modèle numérique expérimental (digital shadow, niveau 1) : pas de synchronisation bidirectionnelle avec l'appareil.

## Features (12) — `image_analysis.py`
Intensité (moyenne, écart-type), contraste RMS = σ/μ, bruit = MAD du résidu image − gaussienne(1.5) (×1.4826), SNR = μ/bruit, netteté = variance du laplacien, gradient moyen, entropie (64 bins), homogénéité et contraste GLCM (16 niveaux, offset horizontal, symétrique), non-uniformité = CV des moyennes de 4×4 blocs, ratio de puissance haute fréquence (FFT, rayon > 0.25).
Limite : le « bruit » mesuré inclut le speckle ; il n'est pas un bruit électronique pur.

## Digital Signature
μ, σ estimés sur le baseline. Distance de **Mahalanobis** sur variables standardisées avec covariance **Ledoit-Wolf** (shrinkage) car n_baseline (30) est proche du nombre de features (12). Seuils : quantiles du χ² à 12 ddl (95 / 99 / 99.9 %) → Normal / Early / Significant / Critical. Hypothèse : normalité approximative ; l'écart est vérifié par le taux de faux positifs sur le holdout.

## IQS (indicateur expérimental, pas un score clinique)
IQS = 100 + 10·(s − s̄₀)/σ_s₀, avec s = −moyenne pondérée des |z| de 6 indicateurs (contraste, bruit, netteté, gradient, non-uniformité, HF). Valeur absolue car une dérive peut aller dans les deux sens (le flou *réduit* le bruit estimé). Poids égaux ; analyse de sensibilité : corrélation de Spearman avec 200 tirages Dirichlet de poids.

## Détection de dérive
Alerte Mahalanobis si ≥ 3 acquisitions consécutives hors Normal. EWMA (λ=0.2, L=3) et CUSUM unilatéral (k=0.5, h=5) sur l'IQS standardisé. Ces cartes de contrôle SPC sont le comparateur de référence des modèles ML.

## ML
Isolation Forest, One-Class SVM (entraînés sur nominal seul), reconstruction PCA (équivalent linéaire d'un autoencodeur, seuil calibré en validation croisée 5 plis). RandomForest sur scénarios simulés (étiquettes = scénarios, jamais diagnostics).

## Fault lab et validité
Les dégradations sont injectées sur des images nominales de test. Le taux de détection mesure la **limite de détection du pipeline** (sensibilité) ; il ne prouve pas la détection d'une dérive matérielle réelle (circularité : on mesure ce qu'on a injecté). La validation réelle passe par le bloc B3/B4 du protocole.

## Jumeau adaptatif
Référence glissante (30). Mise à jour seulement après 3 acquisitions nominales consécutives (contrôlée) ; comparée à la mise à jour aveugle, qui absorbe la dérive.

## Prédiction
Régression linéaire sur les 10 dernières acquisitions, intervalle de prédiction à 95 % (loi de Student), tendance Stable/Improving/Degrading selon p < 0.05, nombre d'acquisitions avant IQS = 70 (baseline − 3σ) si la pente est significativement négative. Ne prédit pas une panne.

## Datasets publics
Développement et robustesse du code (`tools/public_benchmark.py`), jamais la signature de SCAN A. Des images de patients différents ne forment pas une série temporelle d'un appareil.

## Tests
`pytest tests` (9 tests : sens physique des features, signature, cartes de contrôle, tendance, jumeau adaptatif, séparation des sources).
