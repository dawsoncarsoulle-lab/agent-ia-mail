PDF Agent MVP
=============

Projet d'extraction de donnees de commandes d'achat PDF vers JSON.

Le pipeline utilise deux modeles Ollama :

- `qwen2.5:7b` pour les PDF avec couche texte.
- `qwen2.5vl:7b` pour les PDF scannes ou rendus en images.

Installation Linux
------------------

```bash
chmod +x install-linux.sh
./install-linux.sh
```

Installation Windows
--------------------

Depuis PowerShell, a la racine du projet :

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install-windows.ps1
```

Utilisation
-----------

Placez les PDF a traiter dans `data/raw`, puis lancez :

```bash
make run
```

Sans `make`, vous pouvez lancer directement :

```bash
uv run python -m scripts.extract
```

Les modeles peuvent etre changes avec les variables d'environnement
`TEXT_MODEL` et `VISION_MODEL`.
