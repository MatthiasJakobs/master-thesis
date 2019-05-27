# Motivation
- Motivieren mit einem use case
- In Reining et al. sollen Handlungen von Menschen im Kontext Lagerhaus erkannt werden
- Grund: Evaluieren, wie effizient der Pickingprozess ist
    - Mittels motion capturing wurden Skeletdaten (posen) erstellt.
    - Diese wurden dann als features für die action recognition verwendet
- Es gibt auch verfahren, welche pose und handlungen gemeinsam lernen. Dadurch sollen beide voneinander lernen. Ein solches verfahren schaue ich mir an, aber zunächst mal zur Inhalsübersicht.

# TOC
Improvisation

# Fundamentals HAR
1. Slide
    - Human Action Recognition
    - Beispiele: Gehen, hinlegen, telefonieren etc.
2. Slide
    - Menschliche Handlungen sind stark unterschiedlich da auch Menschen stark unterschiedlich sind

# Fundamentals Pose Estimation
1. Slide
    - Extrahierung von Gelenkpunkten aus Bild
        - Heatmap oder Regression, je nach Ansatz
    - Verwandt mit Objekterkennung
    - Es gibt verschiedene Ansätze und Anzahl an Punkten. Manchmal Gesicht dabei (siehe Abbildung, manchmal nicht)
    - Methoden für Bilder mit nur einer Person oder mehreren Personen verfügbar (nächste Folie)
    - 3D Koordinaten Approximierung möglich
2. Slide
    - Probleme (nicht nur bei Multiperson)
        - Verdeckungen und Überlappungen

# Pose estimation shallow methods
- $c$: Parameter welche gelernt werden. Sie geben an, wie die Gaussparameter sind etc.
- Chamfer Distanz gibt an wie gut matching ist

# Stacked Hourglass
- Durch Pooling Auflösung verringern und durch Upsampling wiederherstellen.
    - Features aus den Zwischenschritten über Skip-connections weiterverwenden
        - Wichtig für Kombination von feinen (Hand) und groben (allgemeine Pose) Features
- Jedes Houglass 64x64 pixel
- Maximal 3x3, inspiriert durch Inception
- Durch aufeinanderfolgen: Initiale Ergebnisse werden immer weiter verfeinert und verbessert
    - Häufiger Ansatz in Pose Estimation mittels CNNs
- Geben keine Begründung für die Residual Modules. Normalerweise z.B. damit kein vanishing gradient auftritt
    - Keine Aktivierungsfunktionen angegeben, also keine Ahnung warum kein ReLU

# Video-based HAR
## Shallow
- Video als Würfel
- Interest Points dort wo starke Veränderungen in x, y und t stattfinden
- Cluster mittels k-means, k = 4000, aus trainingsdaten

## Deep - MiCT
- 3D convolutions sind schwer dafür geeignet tiefe netze zu bilden da sie sehr viele parameter haben.
- Darum werden auch sehr sehr sehr große Datensets benötigt weil schwerer zu lernen.
    - Kurz: Geht nicht tief genug weil zu viele Ressourcen benötigt würden.
- Darum: Kombination aus tiefen 2D und 3D
- Zwei Ansätze zur Kombination:
    - Konkatenation: 3D Convolution auf Input und dann auf die 2D output feature map nochmal 2D Convolution
    - Addition: Output der 3D convolution wird addiert mit dem output einer 2D convolution auf dem letzten Frame.
    - Kombination von beiden findet statt (siehe figure)

# Method
## Multitask Learning
- Handlungserkennung und Posenbestimmung gemeinsam trainieren. Idee: Profitieren voneinander
- Pretraining von Pose
    - Verschiedene Datensätze für Pose und Action
- Optimierung (fine-tuning) passiert end-to-end
- Soft-argmax
    - Oft ist die Ausgabe von neuronalen Netzen zur Posenbestimmung eine Heatmap. Heatmap stellt Softmaxverteilung da.
    - Argmax als postprocessing schritt nötig damit exakte Koordinaten. Wenn man end-to-end trainieren will ist ein nicht-differentierbares Argmax aber problematisch.
        - Darum: soft-argmax
- Multitask CNN
    - Basiert auf Inception v4 um Features zu extrahieren
    - Prediction Block: Ähnlich zu Hourglasses von vorhin. Nach jedem Block ein Zwischenergebnis welches verfeinert wird.
    - Zu beidem keine konkreten angaben gemacht. Muss im Code checken.
    - Pose aus Heatmap:
        - Softargmax
        - Danach mittels sigmoid: Ist Gelenk zu sehen?
- Pose-based recognition
    - Pipeline zu sehen.
    - Auch hier: keine genauen Angaben zum Aufbau. Verweis auf Code
    - Action Heatmap für jede Handlung. Mittles Pooling und Softmax dann Wahrscheinlichkeit
        - Max plus min Pooling:
            - f(x) = MaxPooling(x) - MaxPooling(-x)
            - Gibt keine Erklärung warum (In order to be more sensitive to the strongest response)
    - Auch hier: Ergebnisse werden verfeinert
- Appearance-based recognition
    - Im Prinzip gleicher Aufbau wie vorher, nur mit Visual Features statt Posen
- Am Ende: Durch FC-Layer aggregation und mit softmax dann finales Ergebnis
- Losses:
    - Pose: Elastic Net Loss
    - Action: Categorical crossentropy
