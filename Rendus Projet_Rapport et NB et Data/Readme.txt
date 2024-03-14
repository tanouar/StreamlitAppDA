Dans ce répertoire, se trouvent 3 notebooks qui doivent être exécutés dans l'ordre :

-> 3 Analyse exploratoire des données.ipynb :
---------------------------------------------
input nécessaires :
    * owid.csv
    * hadcrut-surface-temperature-anomaly.csv
    * ctpzi.csv
    
output générés :
    * merged_owid_temp.csv
    * merged_owid_temp_zones.csv

-> 4 et 5 Data Visualization et Analyses stats.ipynb :
------------------------------------------------------
input nécessaire :
    * merged_owid_temp.csv

-> 6 et 7 Preprocessing et Modélisation.ipynb :
-----------------------------------------------
input nécessaire :
    * merged_owid_temp_zones.csv


Les fichiers merged_owid_temp.csv et merged_owid_temp_zones.csv fournis dans ce dossier sont là à titre informatif (ou pour tester les 2 derniers notebooks sans exécuter le 1er)