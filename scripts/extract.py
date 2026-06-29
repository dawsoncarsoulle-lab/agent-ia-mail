"""
Orchestrateur d'extraction PDF -> JSON.

Workflow :
    PDF propre (fonts détectées) -> texte pypdfium2 -> qwen2.5:7b  (Ollama texte)
    PDF scanné (aucune font)     -> PNG pypdfium2   -> qwen2.5vl:7b (Ollama vision)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, TypeVar

import ollama
import pypdfium2 as pdfium

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 2500
DEFAULT_TIMEOUT_SECONDS = 1000
MIN_TEXT_CHARS = 100  # seuil : en dessous => considéré comme scanné
ACCOUNTING_TOLERANCE = 0.05

RAW_DIR = Path("data/raw")
PAGES_ROOT = Path("pages")
TEMPLATE_PATH = Path("template.json")
EXTRACTED_DIR = Path("extracted")
REPORTS_DIR = Path("reports")
VALIDATION_DIR = Path("validation")

T = TypeVar("T")


def round_money(value: float) -> float:
    return round(value + 0.0000001, 2)


# ---------------------------------------------------------------------------
# Détection : PDF propre ou scanné ?
# ---------------------------------------------------------------------------


def has_text_layer(pdf_path: Path) -> bool:
    """
    Retourne True si le PDF contient une couche texte exploitable.
    Stratégie double :
      1. pdffonts (subprocess) : rapide, fiable si disponible.
      2. Fallback pypdfium2 : extraction texte page 0, seuil MIN_TEXT_CHARS.
    """
    # Stratégie 1 : pdffonts
    import subprocess

    try:
        result = subprocess.run(
            ["pdffonts", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [L for L in result.stdout.splitlines() if L.strip()]
        # Les 2 premières lignes sont l'en-tête du tableau
        if len(lines) > 2:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # pdffonts non dispo → fallback

    # Stratégie 2 : fallback pypdfium2
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_textpage().get_text_range()
            if len(text) >= MIN_TEXT_CHARS:
                doc.close()
                return True
        doc.close()
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Extraction du texte brut (branche "propre")
# ---------------------------------------------------------------------------


def extract_full_text(pdf_path: Path) -> str:
    """
    Extrait le texte d'un PDF propre en préservant autant que possible
    la mise en page visuelle, surtout les colonnes des tableaux.

    Priorité :
      1. pdftotext -layout : conserve mieux l'alignement des colonnes.
      2. fallback pypdfium2 : disponible via la dépendance déjà utilisée.

    Les marqueurs === PAGE N === restent importants pour aider le LLM
    à séparer les pages et les blocs fournisseur / acheteur.
    """
    import subprocess

    # Stratégie 1 : Poppler / pdftotext avec préservation de layout.
    # Option -layout = garde l'ordre visuel et les espaces des colonnes.
    # Option -nopgbrk = évite les caractères form-feed entre les pages.
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-nopgbrk", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        text = result.stdout.strip()
        if text:
            # pdftotext -nopgbrk ne donne pas toujours des séparateurs fiables.
            # Pour garder une structure simple et stable, on met tout dans PAGE 1
            # quand on ne peut pas découper proprement. Pour les PDF multi-pages,
            # la branche fallback ci-dessous garde des marqueurs par page si besoin.
            return f"=== PAGE 1 ===\n{text}"
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
    ):
        pass

    # Stratégie 2 : fallback pypdfium2, moins bon pour les tableaux mais robuste.
    doc = pdfium.PdfDocument(str(pdf_path))
    parts: list[str] = []
    try:
        for i, page in enumerate(doc, start=1):
            t = page.get_textpage().get_text_range().strip()
            if t:
                parts.append(f"=== PAGE {i} ===\n{t}")
    finally:
        doc.close()
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Rendu PNG (branche "scanné")
# ---------------------------------------------------------------------------


def render_pdf_to_pages(pdf_path: Path, dpi: int = 200, force: bool = False) -> Path:
    output_dir = PAGES_ROOT / pdf_path.stem

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72

    for index, page in enumerate(doc, start=1):
        output_path = output_dir / f"page-{index:03d}.png"
        if output_path.exists() and not force:
            continue
        image = page.render(
            scale=int(scale)
        ).to_pil()  # changement => cast de scale en Int()
        image.save(output_path)

    doc.close()
    return output_dir


# ---------------------------------------------------------------------------
# Prompt partagé
# ---------------------------------------------------------------------------


def build_prompt(template: dict[str, Any], mode: str) -> str:
    """
    mode = "text" | "vision"
    Le prompt est identique ; seule l'intro change légèrement.
    """
    template_text = json.dumps(template, indent=2, ensure_ascii=False)
    intro = (
        "Tu es un extracteur de données pour commandes d'achat fournisseurs.\n"
        "Le texte du document t'est fourni directement."
        if mode == "text"
        else "Tu es un extracteur de données pour commandes d'achat fournisseurs scannées.\n"
        "Le document t'est fourni sous forme d'image(s)."
    )

    return f"""{intro}

Le document peut être nommé :
- Bon de commande
- Commande achat
- B.C. fournisseur
- Purchase order
- Ordre achat
- Ref. achat
- Bon fournisseur
- Demande achat

Retourne uniquement un JSON valide conforme exactement à ce template :

{template_text}

Règles générales :
- Respecte exactement les clés du template.
- Ne crée aucune clé supplémentaire.
- N'invente jamais de valeur.
- Si une information est réellement absente ou illisible, mets null.
- Une chaîne vide doit devenir null.
- Attention : 0, 0%, 0,00 EUR et 0.00 ne sont pas null. Ce sont des valeurs présentes.
- Retourne uniquement le JSON final.
- Ne retourne pas de markdown.
- Ne retourne pas d'explication.
- Ne retourne pas le texte OCR.
- Ne mets pas de commentaires dans le JSON.

Numéro document :
- Le champ entete.numero_facture contient TOUJOURS le numéro principal du document.
- Même si le champ s'appelle numero_facture, il doit recevoir le numéro de commande quand le document est une commande d'achat.
- Si le document affiche "N° commande", "N commande", "Numéro commande", "PO Number", "Document", "Ref. achat" ou "Commande no", extrais cette valeur dans entete.numero_facture.
- Ne mets jamais entete.numero_facture à null si un numéro de commande est visible.

Fournisseur :
- Extraire uniquement les informations du fournisseur, vendeur ou émetteur.
- Ne jamais prendre les informations du client, acheteur, donneur d'ordre, bill to, ship to.
- Le nom fournisseur va dans entete.adresse.ligne_1.

Numéro fournisseur :
- entete.numero_fournisseur doit contenir un vrai code fournisseur (ex: FOU-00219, SUP-7810).
- Ne mets pas le SIREN, le SIRET, la TVA ou une référence article dans numero_fournisseur.

SIREN / SIRET :
- SIREN = exactement 9 chiffres. SIRET = exactement 14 chiffres.
- Ne jamais mettre un numéro de TVA commençant par FR dans siren ou siret.

Adresse fournisseur :
- rue = lignes contenant rue, avenue, boulevard, route, chemin, impasse, allée, quai, cours, place.
- code_postal_ville = code postal 5 chiffres + ville.
- Ne mets jamais un code postal + ville dans adresse.rue.

Items :
- Crée un item par ligne article visible.
- Ignore les lignes de total, TVA globale, remise globale, conditions, commentaires.
- Les lignes négatives sont valides (remise commerciale, avoir).

Colonnes articles :
- Ref / Code / Item = ref_article.
- Designation / Description / Libellé = description.
- Qte / Qty / Quantité = qte_facturee.
- PU HT / Prix unitaire / Unit price = prix_unitaire.
- Total HT / Net / Montant HT = prix_net.
- Rem. / Remise / Discount = remise.
- TVA / Taxe / Tax = taxe.

Règles de montants :
- Les montants doivent être numériques : "1 225,16 EUR" → 1225.16.
- Ne confonds pas prix_unitaire (prix d'une unité) et prix_net (total de la ligne).

Taxe :
- Respecte exactement le format visible : "20%" → "20%", "V20" → "V20".
- Ne transforme jamais "20%" en "V20%" ou "V20" en "20%".
- Ne confonds jamais TVA et remise.

Bas de page :
- net_total_ht = Total HT / Net total HT / Subtotal excl. tax.
- net_total_ttc = Total TTC / Grand total / Net TTC.
- mnt_remise = Remise totale / Discount total.
""".strip()


# ---------------------------------------------------------------------------
# Appels Ollama
# ---------------------------------------------------------------------------


def extract_order_number(text: str) -> str | None:
    """
    Extrait le numéro de commande/document par regex avant l'appel LLM.
    Le numéro doit contenir au moins un chiffre et un tiret ou slash (ex: CA-2026-001).
    Évite que le modèle laisse numero_facture à null par confusion de nom de champ.
    """
    # Un "vrai" numéro de commande contient au moins un chiffre ET fait >= 4 caractères
    NUM = r"([A-Z0-9][A-Z0-9\-_/]{3,})"  # alphanum avec tirets, au moins 4 chars
    patterns = [
        # "N° commande: CA-2026-001" ou "BON DE COMMANDE N° commande: CA-2026-001"
        rf"(?:N[°o°]\s*commande|Num[eé]ro\s+commande)\s*[:\-]?\s*{NUM}",
        # "BON DE COMMANDE N° CA-2026-001"
        rf"BON DE COMMANDE\s+N[°o]?\s*[:\-]?\s*{NUM}",
        # "PO Number: PO-1234"
        rf"(?:PO\s*Number|Purchase\s*Order\s*N[°o]?|Order\s*N[°o]?)\s*[:\-]?\s*{NUM}",
        # "Ref. achat: RA-001"
        rf"(?:Ref\.?\s*achat|R[eé]f[eé]rence\s+commande|Commande\s+no)\s*[:\-]?\s*{NUM}",
        # "Document: DOC-2026-001"
        rf"(?:Document|N[°o]\s*document)\s*[:\-]?\s*{NUM}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).strip()
            # Filtre : doit contenir au moins un chiffre
            if re.search(r"\d", candidate):
                return candidate
    return None


def extract_supplier_facts(text: str) -> dict[str, str | None]:
    """
    Extrait de façon déterministe les données du bloc FOURNISSEUR par regex.
    On cherche le texte entre le marqueur FOURNISSEUR et le prochain marqueur de bloc.

    Retourne un dict avec les clés : numero_fournisseur, siret, siren.
    """
    result: dict[str, str | None] = {
        "numero_fournisseur": None,
        "siret": None,
        "siren": None,
    }

    # Marqueurs qui délimitent le début du bloc fournisseur
    START_MARKERS = (
        r"(?:FOURNISSEUR|EMETTEUR|EMETTEUR\s*/\s*FOURNISSEUR|VENDOR|SUPPLIER)"
    )
    # Marqueurs qui indiquent la fin du bloc
    STOP_MARKERS = r"(?:ACHETEUR|DONNEUR\s+D.ORDRE|ADRESSE\s+DE\s+LIVRAISON|CONDITIONS|BILL\s+TO|SHIP\s+TO|CLIENT|Ref\.\s+Designation|=== PAGE)"

    block = ""

    # Cas fréquent des PDF propres : ACHETEUR et FOURNISSEUR sont deux colonnes
    # sur les mêmes lignes. On prend alors uniquement la colonne de droite.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.search(r"\bFOURNISSEUR\b|\bVENDOR\b|\bSUPPLIER\b", line, re.IGNORECASE):
            right_column: list[str] = []
            for next_line in lines[index + 1 :]:
                if re.search(
                    r"\bADRESSE\s+DE\s+LIVRAISON\b|\bCONDITIONS\b",
                    next_line,
                    re.IGNORECASE,
                ):
                    break

                parts = [
                    part.strip()
                    for part in re.split(r"\s{2,}", next_line)
                    if part.strip()
                ]
                if len(parts) >= 2:
                    right_column.append(parts[-1])
                elif parts and not re.search(r"\bACHETEUR\b", parts[0], re.IGNORECASE):
                    right_column.append(parts[0])

            if right_column:
                block = "\n".join(right_column)
                break

    if not block:
        match = re.search(
            rf"{START_MARKERS}\s*\r?\n(.*?)(?={STOP_MARKERS})",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return result

        block = match.group(1)

    # Code fournisseur
    code_match = re.search(
        r"(?:Code\s+fournisseur|N[°o]\s*fournisseur|Num[eé]ro\s+fournisseur"
        r"|Compte\s+frn|Cpte\s+frn|Compte\s+fournisseur|Vendor\s+ID"
        r"|Supplier\s+ID|Code\s+tiers|Tiers\s+fournisseur)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-_]{2,})",
        block,
        re.IGNORECASE,
    )
    if code_match:
        result["numero_fournisseur"] = code_match.group(1).strip()

    # SIRET (14 chiffres)
    siret_match = re.search(r"SIRET\s*[:\-]?\s*([\d\s]{14,17})", block, re.IGNORECASE)
    if siret_match:
        digits = re.sub(r"\D", "", siret_match.group(1))
        if len(digits) == 14:
            result["siret"] = digits
            result["siren"] = digits[:9]

    # SIREN seul (9 chiffres) si pas de SIRET
    if not result["siret"]:
        siren_match = re.search(
            r"SIREN\s*[:\-]?\s*([\d\s]{9,12})", block, re.IGNORECASE
        )
        if siren_match:
            digits = re.sub(r"\D", "", siren_match.group(1))
            if len(digits) == 9:
                result["siren"] = digits

    return result


def call_ollama_text(
    text: str,
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
) -> str:
    # Extractions déterministes avant le LLM
    order_number = extract_order_number(text)
    supplier = extract_supplier_facts(text)

    order_hint = (
        f"IMPORTANT : numero de commande extrait = [{order_number}]. "
        f"Mets [{order_number}] dans entete.numero_facture.\n"
        if order_number
        else "Cherche le numero de commande et mets-le dans entete.numero_facture.\n"
    )

    supplier_hint = ""
    if supplier["numero_fournisseur"]:
        supplier_hint += (
            f"- entete.numero_fournisseur = [{supplier['numero_fournisseur']}]\n"
        )
    if supplier["siret"]:
        supplier_hint += f"- entete.siret = [{supplier['siret']}]\n"
        supplier_hint += f"- entete.siren = [{supplier['siren']}]\n"
    if supplier_hint:
        supplier_hint = (
            "IMPORTANT : valeurs fournisseur extraites automatiquement, tu DOIS les utiliser :\n"
            + supplier_hint
        )

    prompt = build_prompt(template, mode="text")
    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "=== INSTRUCTIONS CRITIQUES POUR CE DOCUMENT TEXTE ===\n"
                    f"1. NUMERO DE COMMANDE : {order_hint}"
                    "2. FOURNISSEUR vs ACHETEUR : repère le bloc intitulé FOURNISSEUR/VENDOR/SUPPLIER/EMETTEUR. "
                    "   Extrais UNIQUEMENT ce bloc. IGNORE ACHETEUR, CLIENT, DONNEUR D'ORDRE, LIVRAISON, BILL TO, SHIP TO.\n"
                    f"3. DONNEES FOURNISSEUR : {supplier_hint if supplier_hint else 'extrait le code fournisseur, SIRET et SIREN du bloc FOURNISSEUR.'}\n"
                    "=== FIN INSTRUCTIONS ===\n\n"
                    f"--- TEXTE DU DOCUMENT ---\n{text}"
                ),
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )

    raw = response["message"]["content"]

    # Filet de sécurité : force les valeurs extraites si le LLM les a ratées
    try:
        data = json.loads(clean_json_output(raw))
        entete = data.get("entete", {})
        if order_number and is_missing(entete.get("numero_facture")):
            entete["numero_facture"] = order_number
        if supplier["numero_fournisseur"] and is_missing(
            entete.get("numero_fournisseur")
        ):
            entete["numero_fournisseur"] = supplier["numero_fournisseur"]
        if supplier["siret"] and is_missing(entete.get("siret")):
            entete["siret"] = supplier["siret"]
        if supplier["siren"] and is_missing(entete.get("siren")):
            entete["siren"] = supplier["siren"]
        raw = json.dumps(data, ensure_ascii=False)
    except (json.JSONDecodeError, KeyError):
        pass

    return raw


def call_ollama_vision(
    pages_dir: Path,
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
    max_pages: int | None,
) -> str:
    images = sorted(pages_dir.glob("page-*.png"))
    if max_pages is not None:
        images = images[:max_pages]
    if not images:
        raise RuntimeError(f"Aucune image dans {pages_dir}")

    prompt = build_prompt(template, mode="vision")
    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(img) for img in images],
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )
    return response["message"]["content"]


def call_ollama_totals_retry_text(
    text: str,
    current_doc: dict[str, Any],
    issues: list[str],
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
) -> str:
    template_text = json.dumps(template, indent=2, ensure_ascii=False)
    current_json = json.dumps(current_doc, indent=2, ensure_ascii=False)

    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {
                "role": "user",
                "content": f"""
Tu dois vérifier les totaux comptables d'une extraction JSON.

Retourne uniquement un JSON complet conforme exactement à ce template :

{template_text}

Règles :
- Ne modifie pas les champs déjà cohérents sauf nécessité comptable.
- Recalcule somme_items_ht = somme(items[].prix_net).
- Recalcule total_brut_calcule = somme(items[].qte_facturee * items[].prix_unitaire).
- Recalcule remise_calculee = total_brut_calcule - somme_items_ht.
- Vérifie la TVA par taux depuis items[].taxe.
- net_total_ht doit être cohérent avec somme_items_ht et avec le total visible dans le document.
- mnt_remise ne doit jamais recevoir Total brut HT.
- net_total_ttc doit être cohérent avec net_total_ht + TVA.
- Quand les totaux de bas de page sont visibles dans le document, privilégie les valeurs explicitement écrites.

Incohérences détectées :
{json.dumps(issues, ensure_ascii=False)}

JSON courant :
{current_json}

Texte du document :
{text}
""".strip(),
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )
    return response["message"]["content"]


def call_ollama_totals_retry_vision(
    pages_dir: Path,
    current_doc: dict[str, Any],
    issues: list[str],
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
    max_pages: int | None,
) -> str:
    images = sorted(pages_dir.glob("page-*.png"))
    if max_pages is not None:
        images = images[:max_pages]
    if not images:
        raise RuntimeError(f"Aucune image dans {pages_dir}")

    template_text = json.dumps(template, indent=2, ensure_ascii=False)
    current_json = json.dumps(current_doc, indent=2, ensure_ascii=False)

    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {
                "role": "user",
                "content": f"""
Tu dois vérifier les totaux comptables d'une extraction JSON depuis l'image du document.

Retourne uniquement un JSON complet conforme exactement à ce template :

{template_text}

Règles :
- Ne modifie pas les champs déjà cohérents sauf nécessité comptable.
- Recalcule somme_items_ht = somme(items[].prix_net).
- Recalcule total_brut_calcule = somme(items[].qte_facturee * items[].prix_unitaire).
- Recalcule remise_calculee = total_brut_calcule - somme_items_ht.
- Vérifie la TVA par taux depuis items[].taxe.
- net_total_ht doit être cohérent avec somme_items_ht et avec le total visible dans le document.
- mnt_remise ne doit jamais recevoir Total brut HT.
- net_total_ttc doit être cohérent avec net_total_ht + TVA.
- Quand les totaux de bas de page sont visibles, privilégie les valeurs explicitement écrites.

Incohérences détectées :
{json.dumps(issues, ensure_ascii=False)}

JSON courant :
{current_json}
""".strip(),
                "images": [str(img) for img in images],
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Post-traitement (repris de extract_vision_ollama.py)
# ---------------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def clean_json_output(content: str) -> str:
    content = content.strip()
    for prefix in ("```json", "```"):
        if content.startswith(prefix):
            content = content.removeprefix(prefix).strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]
    return content.strip()


def empty_strings_to_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: empty_strings_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [empty_strings_to_none(i) for i in value]
    if isinstance(value, str) and not value.strip():
        return None
    return value


def stringify_if_present(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def digits_only(value: Any) -> str | None:
    text = stringify_if_present(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_siren(value: Any) -> str | None:
    digits = digits_only(value)
    return digits if digits and len(digits) == 9 else None


def normalize_siret(value: Any) -> str | None:
    text = stringify_if_present(value)
    if text and text.upper().startswith("FR"):
        return None
    digits = digits_only(value)
    return digits if digits and len(digits) == 14 else None


def parse_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return value
    text = str(value).strip()
    normalized = (
        text.replace("\u202f", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("€", "")
        .replace("EUR", "")
        .replace("eur", "")
    )
    match = re.search(r"-?\d+(?:[.,]\d+)?", normalized)
    if not match:
        return None
    raw = match.group(0).replace(",", ".")
    try:
        n = float(raw)
        return int(n) if n.is_integer() else n
    except ValueError:
        return None


def parse_percent(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if re.fullmatch(r"V\d+(?:[.,]\d+)?", text.upper()):
        return 0.0
    text = text.replace(",", ".").replace("%", "").strip()
    try:
        return float(text) / 100.0
    except ValueError:
        return 0.0


def parse_discount_rate(value: Any) -> float | None:
    text = stringify_if_present(value)
    if text is None:
        return None
    normalized = text.strip().replace(",", ".").replace("%", "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        return None
    rate = float(normalized) / 100.0
    return rate if 0 <= rate < 1 else None


def format_percent(rate: float) -> str:
    value = round(rate * 100, 4)
    return f"{value:g}%"


def normalize_taxe_value(value: Any) -> str | None:
    text = stringify_if_present(value)
    if text is None:
        return None
    normalized = text.strip().upper().replace(" ", "").replace(",", ".")
    if normalized == "V20%":
        return "20%"
    if normalized == "V10%":
        return "10%"
    if normalized in {"V5.5%", "V5,5%"}:
        return "5.5%"
    return text


def merge_with_template(template: Any, data: Any) -> Any:
    if isinstance(template, dict):
        if not isinstance(data, dict):
            data = {}
        return {k: merge_with_template(tv, data.get(k)) for k, tv in template.items()}
    if isinstance(template, list):
        item_template = template[0] if template else None
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []
        return [merge_with_template(item_template, item) for item in data]
    return data if data is not None else template


def repair_tax_discount_confusion(item: dict[str, Any]) -> dict[str, Any]:
    remise = stringify_if_present(item.get("remise"))
    taxe = stringify_if_present(item.get("taxe"))
    if taxe is None and remise is not None:
        normalized = remise.upper().replace(",", ".")
        if re.fullmatch(r"V\d+(?:\.\d+)?", normalized) or normalized in {
            "20%",
            "10%",
            "5.5%",
            "5,5%",
        }:
            item["taxe"] = remise
            item["remise"] = None
    return item


def infer_line_amounts(item: dict[str, Any]) -> bool:
    qte, pu, net = (
        item.get("qte_facturee"),
        item.get("prix_unitaire"),
        item.get("prix_net"),
    )
    remise_rate = parse_discount_rate(item.get("remise"))
    changed = False

    if qte is not None and pu is not None and net is not None:
        gross = float(qte) * float(pu)
        if remise_rate is not None:
            expected_net = round_money(gross * (1.0 - remise_rate))
            if abs(float(net) - expected_net) > ACCOUNTING_TOLERANCE:
                item["prix_net"] = expected_net
                changed = True
            return changed
        if gross > 0 and float(net) <= gross + ACCOUNTING_TOLERANCE:
            calculated_rate = max(0.0, 1.0 - float(net) / gross)
            item["remise"] = format_percent(calculated_rate)
            changed = True
        return changed

    if remise_rate is None:
        return changed

    multiplier = 1.0 - remise_rate
    if net is None and qte is not None and pu is not None:
        item["prix_net"] = round_money(float(qte) * float(pu) * multiplier)
        return True
    if pu is None and qte not in (None, 0) and net is not None and multiplier > 0:
        item["prix_unitaire"] = round_money(float(net) / (float(qte) * multiplier))
        return True
    if qte is None and pu not in (None, 0) and net is not None and multiplier > 0:
        calculated_qte = float(net) / (float(pu) * multiplier)
        item["qte_facturee"] = (
            int(round(calculated_qte))
            if abs(calculated_qte - round(calculated_qte)) <= 0.0001
            else round(calculated_qte, 4)
        )
        return True

    return changed


def infer_missing_line_net_from_footer(doc: dict[str, Any]) -> bool:
    net_total_ht = doc.get("bas_de_page", {}).get("net_total_ht")
    if net_total_ht is None:
        return False

    items = doc.get("items", [])
    missing = [item for item in items if item.get("prix_net") is None]
    known = [item.get("prix_net") for item in items if item.get("prix_net") is not None]
    if len(missing) != 1 or len(known) != len(items) - 1:
        return False

    residual = round_money(float(net_total_ht) - sum(float(value) for value in known))
    if residual < 0:
        return False
    missing[0]["prix_net"] = residual
    return True


def infer_line_from_total_discount(doc: dict[str, Any]) -> bool:
    total_discount = doc.get("bas_de_page", {}).get("mnt_remise")
    if total_discount is None or float(total_discount) < 0:
        return False

    items = doc.get("items", [])
    candidates: list[dict[str, Any]] = []
    known_discount = 0.0

    for item in items:
        qte = item.get("qte_facturee")
        pu = item.get("prix_unitaire")
        net = item.get("prix_net")
        if qte is not None and pu is not None and net is not None:
            known_discount += float(qte) * float(pu) - float(net)
        elif net is not None and ((qte is None) != (pu is None)):
            candidates.append(item)
        else:
            return False

    if len(candidates) != 1:
        return False

    item = candidates[0]
    line_discount = round_money(float(total_discount) - known_discount)
    gross = round_money(float(item["prix_net"]) + line_discount)
    if line_discount < 0 or gross <= 0:
        return False

    if item.get("prix_unitaire") is None and item.get("qte_facturee") not in (None, 0):
        item["prix_unitaire"] = round_money(gross / float(item["qte_facturee"]))
    elif item.get("qte_facturee") is None and item.get("prix_unitaire") not in (
        None,
        0,
    ):
        calculated_qte = gross / float(item["prix_unitaire"])
        item["qte_facturee"] = (
            int(round(calculated_qte))
            if abs(calculated_qte - round(calculated_qte)) <= 0.0001
            else round(calculated_qte, 4)
        )
    else:
        return False

    item["remise"] = format_percent(line_discount / gross)
    return True


def parse_tax_rate(value: Any) -> float | None:
    text = stringify_if_present(value)
    if text is None:
        return None

    normalized = text.strip().upper().replace(" ", "").replace(",", ".")

    compact_vat_code = normalized.startswith("V")
    if compact_vat_code:
        normalized = normalized[1:]

    if normalized.endswith("%"):
        normalized = normalized[:-1]

    try:
        percentage = float(normalized)
        if compact_vat_code and percentage > 20:
            percentage /= 10.0
        rate = percentage / 100.0
    except ValueError:
        return None

    if rate < 0 or rate > 1:
        return None

    return rate


def compute_accounting_totals(doc: dict[str, Any]) -> dict[str, Any]:
    gross_total = 0.0
    net_total = 0.0
    tax_bases: dict[str, float] = {}
    complete_lines = 0
    taxed_lines = 0
    items = doc.get("items", [])

    for item in items:
        qte = item.get("qte_facturee")
        pu = item.get("prix_unitaire")
        net = item.get("prix_net")

        if qte is None or pu is None or net is None:
            continue

        gross_total += float(qte) * float(pu)
        net_total += float(net)
        complete_lines += 1

        rate = parse_tax_rate(item.get("taxe"))
        if rate is not None:
            taxed_lines += 1
            key = f"{rate:.4f}"
            tax_bases[key] = tax_bases.get(key, 0.0) + float(net)

    vat_by_rate = {
        key: round_money(base * float(key)) for key, base in tax_bases.items()
    }
    total_vat = round_money(sum(vat_by_rate.values()))
    net_total = round_money(net_total)
    gross_total = round_money(gross_total)
    discount_total = round_money(gross_total - net_total)

    return {
        "item_count": len(items),
        "complete_lines": complete_lines,
        "all_lines_complete": bool(items) and complete_lines == len(items),
        "all_tax_rates_known": bool(items) and taxed_lines == len(items),
        "gross_total": gross_total,
        "sum_items_ht": net_total,
        "discount_total": discount_total,
        "tax_bases": {key: round_money(value) for key, value in tax_bases.items()},
        "vat_by_rate": vat_by_rate,
        "total_vat": total_vat,
        "expected_ttc": round_money(net_total + total_vat),
    }


def accounting_issues(doc: dict[str, Any]) -> list[str]:
    totals = compute_accounting_totals(doc)

    if not totals["all_lines_complete"]:
        return []

    bas = doc.get("bas_de_page", {})
    issues: list[str] = []
    net_total_ht = bas.get("net_total_ht")
    mnt_remise = bas.get("mnt_remise")
    net_total_ttc = bas.get("net_total_ttc")

    if (
        net_total_ht is not None
        and abs(float(net_total_ht) - float(totals["sum_items_ht"]))
        > ACCOUNTING_TOLERANCE
    ):
        issues.append(
            "bas_de_page.net_total_ht incohérent avec la somme items[].prix_net"
        )

    if (
        mnt_remise is not None
        and totals["discount_total"] >= 0
        and abs(float(mnt_remise) - float(totals["discount_total"]))
        > ACCOUNTING_TOLERANCE
    ):
        issues.append(
            "bas_de_page.mnt_remise incohérent avec total brut calculé - somme HT"
        )

    if (
        net_total_ttc is not None
        and totals["all_tax_rates_known"]
        and abs(float(net_total_ttc) - float(totals["expected_ttc"]))
        > ACCOUNTING_TOLERANCE
    ):
        issues.append(
            "bas_de_page.net_total_ttc incohérent avec net_total_ht + TVA calculée"
        )

    return issues


def repair_accounting_totals(doc: dict[str, Any]) -> dict[str, Any]:
    totals = compute_accounting_totals(doc)

    if not totals["all_lines_complete"]:
        return doc

    bas = doc.get("bas_de_page", {})

    net_total_ht = bas.get("net_total_ht")
    if (
        net_total_ht is None
        or abs(float(net_total_ht) - float(totals["sum_items_ht"]))
        > ACCOUNTING_TOLERANCE
    ):
        bas["net_total_ht"] = totals["sum_items_ht"]

    current_discount = bas.get("mnt_remise")
    if totals["discount_total"] >= 0 and (
        current_discount is None
        or abs(float(current_discount) - float(totals["discount_total"]))
        > ACCOUNTING_TOLERANCE
    ):
        bas["mnt_remise"] = totals["discount_total"]

    current_ttc = bas.get("net_total_ttc")
    if totals["all_tax_rates_known"] and (
        current_ttc is None
        or abs(float(current_ttc) - float(totals["expected_ttc"]))
        > ACCOUNTING_TOLERANCE
    ):
        bas["net_total_ttc"] = totals["expected_ttc"]

    return doc


def postprocess_document(
    doc: dict[str, Any], *, repair_accounting: bool = True
) -> dict[str, Any]:
    doc = empty_strings_to_none(doc)
    entete = doc.get("entete", {})
    adresse = entete.get("adresse", {})

    entete["numero_facture"] = stringify_if_present(entete.get("numero_facture"))
    entete["numero_fournisseur"] = stringify_if_present(
        entete.get("numero_fournisseur")
    )
    entete["date"] = stringify_if_present(entete.get("date"))

    siren_digits = digits_only(entete.get("siren"))
    siret_digits = digits_only(entete.get("siret"))

    entete["siren"] = normalize_siren(entete.get("siren"))
    entete["siret"] = normalize_siret(entete.get("siret"))

    if siren_digits and len(siren_digits) == 14:
        entete["siret"] = siren_digits
        entete["siren"] = siren_digits[:9]
    if siret_digits and len(siret_digits) == 9 and not entete["siren"]:
        entete["siren"] = siret_digits
    if entete["siret"] and not entete["siren"]:
        entete["siren"] = entete["siret"][:9]

    adresse["ligne_1"] = stringify_if_present(adresse.get("ligne_1"))
    adresse["ligne_2"] = stringify_if_present(adresse.get("ligne_2"))
    adresse["rue"] = stringify_if_present(adresse.get("rue"))
    adresse["code_postal_ville"] = stringify_if_present(
        adresse.get("code_postal_ville")
    )

    numero_fournisseur = entete.get("numero_fournisseur")
    if numero_fournisseur and not re.search(r"\d", str(numero_fournisseur)):
        if is_missing(adresse.get("ligne_1")):
            adresse["ligne_1"] = str(numero_fournisseur).strip()
        entete["numero_fournisseur"] = None

    for item in doc.get("items", []):
        item["ref_article"] = stringify_if_present(item.get("ref_article"))
        item["description"] = stringify_if_present(item.get("description"))
        item["taxe"] = normalize_taxe_value(item.get("taxe"))
        item["remise"] = stringify_if_present(item.get("remise"))
        item["qte_facturee"] = parse_number(item.get("qte_facturee"))
        item["prix_unitaire"] = parse_number(item.get("prix_unitaire"))
        item["prix_net"] = parse_number(item.get("prix_net"))
        repair_tax_discount_confusion(item)

    bas = doc.get("bas_de_page", {})
    bas["net_total_ttc"] = parse_number(bas.get("net_total_ttc"))
    bas["net_total_ht"] = parse_number(bas.get("net_total_ht"))
    bas["mnt_remise"] = parse_number(bas.get("mnt_remise"))

    # Resolve line equations first, then use an explicit footer total to fill
    # the only missing line amount. A second pass can derive another field.
    for item in doc.get("items", []):
        infer_line_amounts(item)
    if (
        bas["mnt_remise"] is not None
        and abs(float(bas["mnt_remise"])) <= ACCOUNTING_TOLERANCE
    ):
        for item in doc.get("items", []):
            if item.get("remise") is None:
                item["remise"] = "0%"
    for item in doc.get("items", []):
        infer_line_amounts(item)
    if infer_missing_line_net_from_footer(doc):
        for item in doc.get("items", []):
            infer_line_amounts(item)
    if infer_line_from_total_discount(doc):
        for item in doc.get("items", []):
            infer_line_amounts(item)

    if repair_accounting:
        repair_accounting_totals(doc)

    return doc


def validate_document(doc: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    entete = doc.get("entete", {})
    adresse = entete.get("adresse", {})
    items = doc.get("items", [])
    bas = doc.get("bas_de_page", {})

    if is_missing(entete.get("numero_facture")):
        errors.append("entete.numero_facture manquant")
    if is_missing(entete.get("numero_fournisseur")):
        warnings.append("entete.numero_fournisseur manquant")
    if is_missing(entete.get("date")):
        warnings.append("entete.date absente")
    if is_missing(entete.get("siren")) and is_missing(entete.get("siret")):
        warnings.append("entete.siren ou entete.siret absent")
    if is_missing(adresse.get("ligne_1")):
        errors.append("entete.adresse.ligne_1 manquant")
    if is_missing(adresse.get("rue")):
        warnings.append("entete.adresse.rue manquant")
    if is_missing(adresse.get("code_postal_ville")):
        warnings.append("entete.adresse.code_postal_ville manquant")
    if not items:
        errors.append("items manquant : aucun article détecté")
    for idx, item in enumerate(items, 1):
        p = f"items[{idx}]"
        if is_missing(item.get("ref_article")):
            errors.append(f"{p}.ref_article manquant")
        if is_missing(item.get("qte_facturee")):
            errors.append(f"{p}.qte_facturee manquant")
        if is_missing(item.get("description")):
            errors.append(f"{p}.description manquant")
        if is_missing(item.get("prix_unitaire")) and is_missing(item.get("prix_net")):
            errors.append(f"{p}: prix_unitaire ou prix_net obligatoire")
    if is_missing(bas.get("net_total_ttc")):
        warnings.append("bas_de_page.net_total_ttc absent")
    if is_missing(bas.get("net_total_ht")):
        warnings.append("bas_de_page.net_total_ht absent")

    warnings.extend(accounting_issues(doc))

    return {
        "status": "OK" if not errors else "NEEDS_REVIEW",
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------


class ExtractionTimeoutError(RuntimeError):
    pass


def run_with_timeout(fn: Callable[..., T], *args: Any, timeout_seconds: int) -> T:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise ExtractionTimeoutError(f"timeout après {timeout_seconds}s") from exc


# ---------------------------------------------------------------------------
# Traitement d'un document
# ---------------------------------------------------------------------------


def process_document(
    pdf_path: Path,
    template: dict[str, Any],
    text_model: str,
    vision_model: str,
    num_ctx: int,
    num_predict: int,
    max_pages: int | None,
    timeout_seconds: int,
    dpi: int,
    skip_existing: bool,
    force_render: bool,
) -> None:
    pdf_name = pdf_path.stem
    print(f"\n{'─' * 60}", flush=True)

    EXTRACTED_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    VALIDATION_DIR.mkdir(exist_ok=True)

    extracted_path = EXTRACTED_DIR / f"{pdf_name}.json"
    validation_path = VALIDATION_DIR / f"{pdf_name}_validation.json"
    raw_output_path = REPORTS_DIR / f"{pdf_name}_raw_output.txt"

    if skip_existing and extracted_path.exists() and validation_path.exists():
        print(f"SKIP (déjà traité) : {pdf_name}", flush=True)
        return

    # --- Détection ---
    is_clean = has_text_layer(pdf_path)
    mode = "text" if is_clean else "vision"
    model_used = text_model if is_clean else vision_model
    print(
        f"{'📄 PDF propre' if is_clean else '🖼  PDF scanné'} → {mode} ({model_used}) : {pdf_name}",
        flush=True,
    )

    try:
        source_text: str | None = None
        source_pages_dir: Path | None = None

        def do_extract() -> str:
            nonlocal source_text, source_pages_dir
            if is_clean:
                source_text = extract_full_text(pdf_path)
                return call_ollama_text(
                    source_text, template, text_model, num_ctx, num_predict
                )
            else:
                source_pages_dir = render_pdf_to_pages(
                    pdf_path, dpi=dpi, force=force_render
                )
                return call_ollama_vision(
                    source_pages_dir,
                    template,
                    vision_model,
                    num_ctx,
                    num_predict,
                    max_pages,
                )

        raw_content = run_with_timeout(do_extract, timeout_seconds=timeout_seconds)
        raw_output_path.write_text(raw_content, encoding="utf-8")

        cleaned = clean_json_output(raw_content)
        model_data = json.loads(cleaned)

        doc = merge_with_template(template, model_data)
        doc = postprocess_document(doc, repair_accounting=False)

        retry_issues = accounting_issues(doc)
        did_retry = False

        if retry_issues:
            print("   ↻ Retry vérification des totaux", flush=True)
            did_retry = True

            if is_clean and source_text is not None:
                retry_raw = call_ollama_totals_retry_text(
                    source_text,
                    doc,
                    retry_issues,
                    template,
                    text_model,
                    num_ctx,
                    num_predict,
                )
            elif source_pages_dir is not None:
                retry_raw = call_ollama_totals_retry_vision(
                    source_pages_dir,
                    doc,
                    retry_issues,
                    template,
                    vision_model,
                    num_ctx,
                    num_predict,
                    max_pages,
                )
            else:
                retry_raw = raw_content

            raw_output_path.write_text(
                f"{raw_content}\n\n--- RETRY_TOTALS ---\n{retry_raw}",
                encoding="utf-8",
            )
            retry_data = json.loads(clean_json_output(retry_raw))
            doc = merge_with_template(template, retry_data)
            doc = postprocess_document(doc, repair_accounting=False)

        doc = postprocess_document(doc, repair_accounting=True)
        doc["_meta"] = {
            "mode": mode,
            "model": model_used,
            "accounting_retry": did_retry,
            "accounting_retry_issues": retry_issues,
        }

        extracted_path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        validation = validate_document(doc)
        report = {
            "pdf_name": pdf_name,
            "mode": mode,
            "model": model_used,
            "status": validation["status"],
            "extracted_file": str(extracted_path),
            "raw_output_file": str(raw_output_path),
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "accounting_retry": did_retry,
            "accounting_retry_issues": retry_issues,
        }
        validation_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        status_icon = "✅" if validation["status"] == "OK" else "⚠️ "
        print(f"{status_icon} {validation['status']} : {pdf_name}", flush=True)
        if validation["errors"]:
            for e in validation["errors"]:
                print(f"   ❌ {e}", flush=True)
        if validation["warnings"]:
            for w in validation["warnings"]:
                print(f"   ⚠  {w}", flush=True)

    except ExtractionTimeoutError as err:
        _write_error(validation_path, pdf_name, mode, model_used, "timeout", str(err))
        print(f"❌ TIMEOUT : {pdf_name} → {err}", flush=True)

    except json.JSONDecodeError as err:
        _write_error(
            validation_path, pdf_name, mode, model_used, "invalid_json", str(err)
        )
        print(f"❌ JSON invalide : {pdf_name}", flush=True)

    except Exception as err:
        reason = (
            "context_size_exceeded"
            if "exceeds the available context size" in str(err)
            else "exception"
        )
        _write_error(validation_path, pdf_name, mode, model_used, reason, str(err))
        print(f"❌ ERREUR : {pdf_name} → {err}", flush=True)


def _write_error(
    path: Path, pdf_name: str, mode: str, model: str, reason: str, error: str
) -> None:
    path.write_text(
        json.dumps(
            {
                "pdf_name": pdf_name,
                "mode": mode,
                "model": model,
                "status": "FAILED",
                "reason": reason,
                "error": error,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------


def list_pdfs(only: str | None) -> list[Path]:
    pdfs = sorted(RAW_DIR.rglob("*.pdf"))
    if only:
        pdfs = [p for p in pdfs if p.stem == only or p.name == only]
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrateur PDF → JSON : branche texte (propre) ou vision (scanné)."
    )
    parser.add_argument(
        "--only", default=None, help="Traiter un seul PDF (nom ou stem)."
    )
    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL,
        help=f"Modèle Ollama texte. Défaut: {DEFAULT_TEXT_MODEL}",
    )
    parser.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=f"Modèle Ollama vision. Défaut: {DEFAULT_VISION_MODEL}",
    )
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limite le nb d'images pour la branche vision.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Résolution PNG pour la branche vision."
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="Retraite même si déjà extrait."
    )
    parser.add_argument(
        "--force-render",
        action="store_true",
        help="Recrée les PNG même s'ils existent.",
    )

    args = parser.parse_args()

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template introuvable : {TEMPLATE_PATH}")

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    pdfs = list_pdfs(args.only)

    if not pdfs:
        print("Aucun PDF trouvé.")
        return

    print(f"📂 {len(pdfs)} PDF(s) à traiter", flush=True)
    print(f"   Modèle texte  : {args.text_model}", flush=True)
    print(f"   Modèle vision : {args.vision_model}", flush=True)

    for pdf_path in pdfs:
        process_document(
            pdf_path=pdf_path,
            template=template,
            text_model=args.text_model,
            vision_model=args.vision_model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout_seconds,
            dpi=args.dpi,
            skip_existing=not args.no_skip_existing,
            force_render=args.force_render,
        )

    print(f"\n{'─' * 60}", flush=True)
    print("✅ Terminé.", flush=True)


if __name__ == "__main__":
    main()
