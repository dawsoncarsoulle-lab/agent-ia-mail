# données a structurer dans un fichier json

## structure du json

```json
{
  "entete" : [
    "N° fournisseur" : {}, // toujours present
    "SIREN" ou "SIRET" : {}, // pas toujours present
    "Adresse" : {}, // toujours present
    "ligne 1" : {}, // toujours present
    "ligne 2" : {}, // toujours present
    "Rue" : {}, // toujours present
    "Code postal / Ville" : {}, // toujours present
  ],

  "item" : [
    "Ref article" : {}, // toujours present
    "Qte / Facturé" : {}, // toujours present
    "Prix unitaire" : {}, // des fois present, des fois non en fonction de si il y a Prix net ou nonet vice versa
    "Prix net" : {},
    "Taxe (taux) ou valeur" : {},// pas toujours present
    "Remise : % ou net" : {},// pas toujours present
    "Description" : {}, // toujours present
  ],

  "Entete /Bas de page" : [
    "Net total TTC" : {},
    "Net total HT" : {},
    "Mnt remise" : {}
  ]
}
```
