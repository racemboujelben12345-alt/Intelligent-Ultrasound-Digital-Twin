# Méthodologie et justification des choix

**Statut** : prototype de jumeau numérique fondé sur les données, sans synchronisation bidirectionnelle obligatoire avec un appareil physique.

## 1. Données
Le pipeline sépare trois sources :

- **experimental** : acquisitions d'un système d'échographie physique ;
- **public** : datasets publics utilisés pour développement et benchmark ;
- **simulated** : images générées ou soumises à des dégradations contrôlées.

Cette séparation est essentielle pour éviter de présenter un benchmark public ou une simulation comme une mesure spécifique d'un équipement.

## 2. Digital Signature
Les images sont converties en un vecteur de caractéristiques quantitatives : intensité, dispersion, contraste, bruit apparent, SNR, netteté, gradient, entropie, texture/homogénéité, non-uniformité et contenu fréquentiel.

Le « bruit » image-domain peut inclure le speckle et ne doit pas être interprété comme un bruit électronique pur.

## 3. Baseline et anomalie
Une population nominale sert à estimer la variabilité multivariée. La distance de Mahalanobis et les méthodes de contrôle statistique permettent de quantifier la déviation d'une nouvelle observation.

Une distance élevée signifie une différence statistique par rapport à la référence ; elle n'identifie pas seule une cause physique.

## 4. Quality Assessment
Le score de qualité est un indicateur expérimental construit à partir de dimensions d'image telles que contraste, bruit apparent, netteté, gradient, non-uniformité et contenu haute fréquence.

Il ne s'agit ni d'un score clinique ni d'une mesure absolue de performance matérielle.

## 5. Simulation
Le moteur applique des dégradations contrôlées : bruit, flou, réduction du contraste, déplacement d'intensité et variation du speckle.

La simulation permet de tester la sensibilité et la cohérence du pipeline. Elle ne constitue pas une preuve de panne réelle.

## 6. Drift et tendance
EWMA/CUSUM et l'analyse temporelle permettent d'étudier les changements persistants de la signature.

La tendance ne doit pas être présentée comme une date de panne ou une RUL sans validation expérimentale appropriée.

## 7. ML
Les modèles non supervisés peuvent être comparés sur les données nominales. Les labels de scénarios simulés peuvent servir à tester la reconnaissance de dégradations injectées, mais ne doivent pas être décrits comme des diagnostics de panne physique.

## 8. Validation
La validation suit quatre niveaux :

1. tests logiciels ;
2. benchmark public ;
3. simulation contrôlée ;
4. acquisitions expérimentales répétées lorsque disponibles.

Les performances sur simulation mesurent la capacité du pipeline à détecter les perturbations que nous avons définies.

## 9. Principe de généralisation
Le Digital Twin est volontairement indépendant d'un modèle particulier d'échographe. Des données provenant ultérieurement d'un appareil spécifique peuvent être ajoutées comme couche expérimentale et servir à calibrer/valider le jumeau.
