from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

import ollama
import pypdfium2 as pdfium

DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 2500
DEFAULT_TIMEOUT_SECONDS = 1000

PAGES_ROOT = Path("pages")
RAW_DIR = Path("data/raw")
TEMPLATE_PATH = Path("template.json")

EXTRACTED_DIR = Path("extracted")
REPORTS_DIR = Path("reports")
VALIDATION_DIR = Path("validation")

T = TypeVar("T")


class ExtractionTimeoutError(RuntimeError):
    pass


def load_template() -> dict[str, Any]:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template introuvable : {TEMPLATE_PATH}")

    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"page-(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 999999, path.name


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def clean_json_output(content: str) -> str:
    content = content.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()

    if content.startswith("```"):
        content = content.removeprefix("```").strip()

    if content.endswith("```"):
        content = content.removesuffix("```").strip()

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]

    return content.strip()


def empty_strings_to_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: empty_strings_to_none(val) for key, val in value.items()}

    if isinstance(value, list):
        return [empty_strings_to_none(item) for item in value]

    if isinstance(value, str) and not value.strip():
        return None

    return value


def stringify_if_present(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text


def digits_only(value: Any) -> str | None:
    text = stringify_if_present(value)
    if text is None:
        return None

    digits = re.sub(r"\D", "", text)
    return digits or None


def normalize_siren(value: Any) -> str | None:
    digits = digits_only(value)

    if digits and len(digits) == 9:
        return digits

    return None


def normalize_siret(value: Any) -> str | None:
    text = stringify_if_present(value)

    # TVA FR du style FR42396720310 : ce n'est PAS un SIRET.
    if text and text.upper().startswith("FR"):
        return None

    digits = digits_only(value)

    if digits and len(digits) == 14:
        return digits

    return None


def parse_number(value: Any) -> int | float | None:
    if value is None:
        return None

    if isinstance(value, int | float):
        return value

    text = str(value).strip()
    if not text:
        return None

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
        number = float(raw)
    except ValueError:
        return None

    if number.is_integer():
        return int(number)

    return number


def parse_percent(value: Any) -> float:
    if value is None:
        return 0.0

    text = str(value).strip()
    if not text:
        return 0.0

    # V20 est un code TVA, pas une remise.
    if re.fullmatch(r"V\d+(?:[.,]\d+)?", text.upper()):
        return 0.0

    text = text.replace(",", ".").replace("%", "").strip()

    try:
        return float(text) / 100.0
    except ValueError:
        return 0.0


def merge_with_template(template: Any, data: Any) -> Any:
    """
    Garde uniquement les clés présentes dans le template.
    Les clés inventées par le modèle sont supprimées.
    """
    if isinstance(template, dict):
        if not isinstance(data, dict):
            data = {}

        return {
            key: merge_with_template(template_value, data.get(key))
            for key, template_value in template.items()
        }

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
    """
    Corrige un cas fréquent où le modèle met V20/20% dans remise au lieu de taxe.
    """
    remise = stringify_if_present(item.get("remise"))
    taxe = stringify_if_present(item.get("taxe"))

    if taxe is None and remise is not None:
        normalized = remise.upper().replace(",", ".")
        looks_like_tax = bool(
            re.fullmatch(r"V\d+(?:\.\d+)?", normalized)
            or normalized in {"20%", "10%", "5.5%", "5,5%"}
        )

        if looks_like_tax:
            item["taxe"] = remise
            item["remise"] = None

    return item


def repair_line_amounts(item: dict[str, Any]) -> dict[str, Any]:
    """
    Recalcule prix_net si le modèle a lu quantité + prix unitaire mais a raté Net/Total HT.
    On le fait seulement si prix_net est absent.
    """
    qte = item.get("qte_facturee")
    pu = item.get("prix_unitaire")
    net = item.get("prix_net")

    if net is not None or qte is None or pu is None:
        return item

    remise_rate = parse_percent(item.get("remise"))

    if remise_rate < 0 or remise_rate >= 1:
        remise_rate = 0.0

    item["prix_net"] = round(float(qte) * float(pu) * (1.0 - remise_rate), 2)
    return item


def repair_missing_discount_from_amounts(item: dict[str, Any]) -> dict[str, Any]:
    """
    Déduit une remise positive quand quantité, PU et net ligne la rendent évidente.
    On ne crée pas de 0% pour éviter d'inventer une colonne remise absente.
    """
    if not is_missing(item.get("remise")):
        return item

    qte = item.get("qte_facturee")
    pu = item.get("prix_unitaire")
    net = item.get("prix_net")

    if qte is None or pu is None or net is None:
        return item

    gross = float(qte) * float(pu)

    if gross <= 0:
        return item

    discount_rate = 1.0 - (float(net) / gross)

    if discount_rate <= 0.0001 or discount_rate >= 1:
        return item

    percent = round(discount_rate * 100, 2)
    item["remise"] = f"{percent:g}%"
    return item


def extract_pdf_text(pdf_name: str) -> str:
    pdf_path = RAW_DIR / f"{pdf_name}.pdf"

    if not pdf_path.exists():
        return ""

    document = pdfium.PdfDocument(str(pdf_path))
    parts: list[str] = []

    try:
        for page in document:
            text = page.get_textpage().get_text_range()
            if text.strip():
                parts.append(text)
    finally:
        document.close()

    return "\n".join(parts)


def normalize_text_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def extract_supplier_block_from_text(text: str) -> list[str]:
    lines = normalize_text_lines(text)
    start_index: int | None = None

    supplier_headers = {
        "FOURNISSEUR",
        "EMETTEUR",
        "EMETTEUR / FOURNISSEUR",
        "VENDOR",
        "SUPPLIER",
    }
    stop_headers = {
        "ACHETEUR",
        "ADRESSE DE LIVRAISON",
        "CONDITIONS",
        "DONNEUR ORDRE",
        "DONNEUR D'ORDRE",
        "BILL TO",
        "SHIP TO",
        "CLIENT",
    }

    for index, line in enumerate(lines):
        normalized = line.upper().replace("É", "E").replace("È", "E")
        if normalized in supplier_headers:
            start_index = index + 1
            break

    if start_index is None:
        return []

    block: list[str] = []
    for line in lines[start_index:]:
        normalized = line.upper().replace("É", "E").replace("È", "E")

        if normalized in stop_headers:
            break

        if re.match(r"^(REF\.?|REF ARTICLE|REFERENCE)\b", normalized):
            break

        block.append(line)

    return block


def extract_supplier_code_from_lines(lines: list[str]) -> str | None:
    pattern = re.compile(
        r"\b(?:code\s+fournisseur|n[°o]?\s*fournisseur|num[eé]ro\s+fournisseur|"
        r"compte\s+frn|cpte\s+frn|compte\s+fournisseur|vendor\s+id|"
        r"vendor\s+account|supplier\s+id|code\s+tiers|tiers\s+fournisseur)\b"
        r"\s*:?\s*([A-Z0-9][A-Z0-9_-]{2,})",
        re.IGNORECASE,
    )

    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1).strip(" .;,:")

    return None


def repair_entete_from_pdf_text(doc: dict[str, Any], pdf_text: str) -> dict[str, Any]:
    if not pdf_text.strip():
        return doc

    supplier_lines = extract_supplier_block_from_text(pdf_text)

    if not supplier_lines:
        return doc

    entete = doc.get("entete", {})
    adresse = entete.get("adresse", {})

    supplier_code = extract_supplier_code_from_lines(supplier_lines)
    if supplier_code and is_missing(entete.get("numero_fournisseur")):
        entete["numero_fournisseur"] = supplier_code

    for line in supplier_lines:
        if re.search(r"\b(SIREN|SIRET|TVA|code fournisseur|compte frn|cpte frn)\b", line, re.IGNORECASE):
            continue

        if re.search(r"\b\d{5}\b", line):
            if is_missing(adresse.get("code_postal_ville")):
                adresse["code_postal_ville"] = line
            continue

        if re.search(
            r"\b(rue|avenue|av\.|boulevard|bd|route|chemin|impasse|all[eé]e|quai|cours|place|passage)\b",
            line,
            re.IGNORECASE,
        ):
            if is_missing(adresse.get("rue")):
                adresse["rue"] = line
            continue

        if is_missing(adresse.get("ligne_1")):
            adresse["ligne_1"] = line

    for line in supplier_lines:
        if re.search(r"\bSIRET\b", line, re.IGNORECASE):
            siret = normalize_siret(line)
            if siret:
                entete["siret"] = siret
                entete["siren"] = siret[:9]
        elif re.search(r"\bSIREN\b", line, re.IGNORECASE):
            siren = normalize_siren(line)
            if siren and is_missing(entete.get("siren")):
                entete["siren"] = siren

    return doc


def parse_item_rows_from_pdf_text(text: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    for line in normalize_text_lines(text):
        match = re.match(
            r"^(?P<ref>[A-Z0-9][A-Z0-9_-]+)\s+"
            r"(?P<description>.+?)\s+"
            r"(?P<qte>-?\d+(?:[,.]\d+)?)\s+"
            r"(?P<unite>[A-Za-zÀ-ÿ0-9_-]+)\s+"
            r"(?P<pu>-?\d[\d\s]*[,.]\d{2})(?:\s+EUR)?\s+"
            r"(?P<remise>-?\d+(?:[,.]\d+)?%?)\s+"
            r"(?P<taxe>V?\d+(?:[,.]\d+)?%?)\s+"
            r"(?P<net>-?\d[\d\s]*[,.]\d{2})(?:\s+EUR)?$",
            line,
            re.IGNORECASE,
        )

        if not match:
            continue

        ref = match.group("ref").strip()
        rows[ref] = {
            "ref_article": ref,
            "description": match.group("description").strip(),
            "qte_facturee": parse_number(match.group("qte")),
            "prix_unitaire": parse_number(match.group("pu")),
            "prix_net": parse_number(match.group("net")),
            "taxe": normalize_taxe_value(match.group("taxe")),
            "remise": stringify_if_present(match.group("remise")),
        }

    return rows


def repair_items_from_pdf_text(doc: dict[str, Any], pdf_text: str) -> dict[str, Any]:
    if not pdf_text.strip():
        return doc

    rows = parse_item_rows_from_pdf_text(pdf_text)

    if not rows:
        return doc

    for item in doc.get("items", []):
        ref = stringify_if_present(item.get("ref_article"))

        if not ref or ref not in rows:
            continue

        row = rows[ref]

        for key in (
            "description",
            "qte_facturee",
            "prix_unitaire",
            "prix_net",
            "taxe",
            "remise",
        ):
            if is_missing(item.get(key)) and not is_missing(row.get(key)):
                item[key] = row[key]

    return doc


def postprocess_document(doc: dict[str, Any]) -> dict[str, Any]:
    doc = empty_strings_to_none(doc)

    entete = doc.get("entete", {})
    adresse = entete.get("adresse", {})

    entete["numero_facture"] = stringify_if_present(entete.get("numero_facture"))
    entete["numero_fournisseur"] = stringify_if_present(
        entete.get("numero_fournisseur")
    )
    entete["date"] = stringify_if_present(entete.get("date"))

    raw_siren = entete.get("siren")
    raw_siret = entete.get("siret")

    siren_digits = digits_only(raw_siren)
    siret_digits = digits_only(raw_siret)

    entete["siren"] = normalize_siren(raw_siren)
    entete["siret"] = normalize_siret(raw_siret)

    # Cas fréquent : le modèle met un SIRET dans le champ SIREN.
    if siren_digits and len(siren_digits) == 14:
        entete["siret"] = siren_digits
        entete["siren"] = siren_digits[:9]

    # Cas fréquent : le modèle met un SIREN dans le champ SIRET.
    if siret_digits and len(siret_digits) == 9 and not entete["siren"]:
        entete["siren"] = siret_digits

    # Cas normal : si SIRET est présent, on peut déduire le SIREN.
    if entete["siret"] and not entete["siren"]:
        entete["siren"] = entete["siret"][:9]

    adresse["ligne_1"] = stringify_if_present(adresse.get("ligne_1"))
    adresse["ligne_2"] = stringify_if_present(adresse.get("ligne_2"))
    adresse["rue"] = stringify_if_present(adresse.get("rue"))
    adresse["code_postal_ville"] = stringify_if_present(
        adresse.get("code_postal_ville")
    )

    numero_fournisseur = entete.get("numero_fournisseur")

    # Si le modèle met un nom de société dans numero_fournisseur,
    # on le déplace vers adresse.ligne_1 si cette ligne est vide.
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
        repair_line_amounts(item)
        repair_missing_discount_from_amounts(item)

    bas_de_page = doc.get("bas_de_page", {})
    bas_de_page["net_total_ttc"] = parse_number(bas_de_page.get("net_total_ttc"))
    bas_de_page["net_total_ht"] = parse_number(bas_de_page.get("net_total_ht"))
    bas_de_page["mnt_remise"] = parse_number(bas_de_page.get("mnt_remise"))

    return doc


def validate_document(doc: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    entete = doc.get("entete", {})
    adresse = entete.get("adresse", {})
    items = doc.get("items", [])
    bas_de_page = doc.get("bas_de_page", {})

    if is_missing(entete.get("numero_facture")):
        errors.append("entete.numero_facture manquant")

    # Un vrai document peut ne pas contenir de code fournisseur : warning, pas erreur bloquante.
    if is_missing(entete.get("numero_fournisseur")):
        warnings.append("entete.numero_fournisseur manquant")
    else:
        numero_fournisseur = str(entete.get("numero_fournisseur"))
        if not re.search(r"\d", numero_fournisseur):
            warnings.append("entete.numero_fournisseur semble ne pas être un numéro")

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

    for index, item in enumerate(items, start=1):
        prefix = f"items[{index}]"

        if is_missing(item.get("ref_article")):
            errors.append(f"{prefix}.ref_article manquant")

        if is_missing(item.get("qte_facturee")):
            errors.append(f"{prefix}.qte_facturee manquant")

        if is_missing(item.get("description")):
            errors.append(f"{prefix}.description manquant")

        if is_missing(item.get("prix_unitaire")) and is_missing(item.get("prix_net")):
            errors.append(f"{prefix}: prix_unitaire ou prix_net obligatoire")

    if is_missing(bas_de_page.get("net_total_ttc")):
        warnings.append("bas_de_page.net_total_ttc absent")

    if is_missing(bas_de_page.get("net_total_ht")):
        warnings.append("bas_de_page.net_total_ht absent")

    return {
        "status": "OK" if not errors else "NEEDS_REVIEW",
        "errors": errors,
        "warnings": warnings,
    }


def build_prompt(template: dict[str, Any]) -> str:
    template_text = json.dumps(template, indent=2, ensure_ascii=False)

    return f"""
Tu es un extracteur de données pour commandes d'achat fournisseurs scannées.

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
- Exemple : "N° commande: CA-2026-014" donne "numero_facture": "CA-2026-014".
- Ne mets jamais entete.numero_facture à null si un numéro de commande est visible.

Fournisseur :
- Extraire uniquement les informations du fournisseur, vendeur ou émetteur.
- Ne jamais prendre les informations du client, acheteur, donneur d'ordre, bill to, ship to, client interne, adresse de livraison ou adresse de facturation client.
- Les blocs fournisseur peuvent être nommés :
  - Fournisseur
  - Emetteur
  - Emetteur / Fournisseur
  - Vendor
  - Supplier
  - A commander à
  - B.C. fournisseur
  - Vendeur
  - Prestataire
- Le nom fournisseur est souvent la première ligne en gras ou en majuscules après le libellé fournisseur/vendor/supplier.
- Ce nom doit aller dans entete.adresse.ligne_1.
- Ne mets pas une ligne d'adresse dans entete.adresse.ligne_1 si un nom de société fournisseur est visible.
- Si plusieurs blocs d'adresse existent, choisis toujours le bloc fournisseur/vendor/supplier, pas le bloc client/donneur d'ordre/livraison.

Numéro fournisseur :
- entete.numero_fournisseur doit contenir un vrai code fournisseur.
- Cherche les libellés :
  - Code fournisseur
  - No fournisseur
  - N° fournisseur
  - Numéro fournisseur
  - Compte frn
  - Cpte frn
  - Compte fournisseur
  - Vendor ID
  - Vendor account
  - Supplier ID
  - Code tiers
  - Tiers fournisseur
- Exemples valides : FOU-00219, SUP-7810, FRN-80421, GRN-6207, NHP-55201, SP-03188.
- Le numéro fournisseur n'est jamais le nom de l'entreprise.
- Ne mets pas le SIREN, le SIRET, la TVA ou une référence article dans numero_fournisseur.
- Si une ligne visible dans le bloc fournisseur contient un libellé de code fournisseur
  suivi d'une valeur, extrais uniquement la valeur après ":" ou après le libellé.
- Exemples génériques :
  - "Code fournisseur: XXX-12345" donne "numero_fournisseur": "XXX-12345".
  - "Compte frn : ABC-9876" donne "numero_fournisseur": "ABC-9876".
  - "Cpte frn 6207" donne "numero_fournisseur": "6207".
- Le code fournisseur est souvent proche du SIREN/SIRET dans le bloc fournisseur :
  lis les lignes juste au-dessus de SIREN/SIRET avant de décider que numero_fournisseur est null.

SIREN / SIRET :
- SIREN = exactement 9 chiffres.
- SIRET = exactement 14 chiffres.
- Extraire le SIREN/SIRET du fournisseur, pas celui du client, donneur d'ordre, bill to, ship to ou livraison.
- Ne jamais mettre un numéro de TVA commençant par FR dans siren ou siret.
- Si un numéro commence par FR, c'est une TVA intracommunautaire, pas un SIREN/SIRET.
- Si SIRET est présent, SIREN peut être déduit avec les 9 premiers chiffres.
- Si plusieurs SIRET sont visibles, choisir celui du bloc fournisseur/vendor/supplier.

Adresse fournisseur :
- Dans le bloc fournisseur, chaque ligne doit être classée correctement.
- Une ligne contenant un nom de voie comme "rue", "avenue", "boulevard", "route", "chemin", "impasse", "allée", "quai", "cours", "place" va TOUJOURS dans adresse.rue.
- Une ligne composée d'un code postal à 5 chiffres suivi d'une ville va TOUJOURS dans adresse.code_postal_ville.
- Ne mets jamais un code postal + ville dans adresse.rue.
- Ne mets jamais une rue dans adresse.ligne_2.
- Exemple :
  TechniSud Materiel
  41 rue Paul Langevin
  13013 Marseille

  donne :
  ligne_1 = "TechniSud Materiel"
  ligne_2 = null
  rue = "41 rue Paul Langevin"
  code_postal_ville = "13013 Marseille"

Découpage adresse :
- Une ligne contenant seulement un code postal et une ville va dans adresse.code_postal_ville.
- Une ligne contenant "rue", "avenue", "av.", "boulevard", "bd", "route", "chemin", "impasse", "allée", "allee", "quai", "cours", "place", "passage" va dans adresse.rue.
- Une ligne contenant "ZI", "ZA", "ZAC", "zone", "zone artisanale", "bâtiment", "batiment", "dépôt", "depot", "service", "atelier", "comptoir", "à l'attention de", "a l attention de" va dans adresse.ligne_2 si ce n'est pas une voie complète.
- Si une ligne contient à la fois un complément et une voie, sépare-les si possible.
- Exemple : "ZI du Lac - Batiment C, 5 Avenue du Lac, 75008 Paris" devient :
  - ligne_1 : nom fournisseur
  - ligne_2 : "ZI du Lac - Batiment C"
  - rue : "5 Avenue du Lac"
  - code_postal_ville : "75008 Paris"
- Exemple : "Comptoir pro Zone artisanale 9 impasse des Metiers 13011 Marseille" devient :
  - ligne_2 : "Comptoir pro"
  - rue : "Zone artisanale 9 impasse des Metiers"
  - code_postal_ville : "13011 Marseille"
- Ne mets jamais un code postal + ville dans adresse.rue.
- Ne mets jamais une rue dans adresse.code_postal_ville.
- Ne mets jamais une voie comme "24 avenue Jean Jaures" ou "88 route de Paris" dans ligne_2 : elle doit aller dans rue.

Items :
- Crée un item par ligne article visible.
- Si un tableau article est visible, items ne doit pas être vide.
- Si tu ne lis pas certains champs d'une ligne, crée quand même l'item avec les champs lisibles et null pour les champs illisibles.
- Ignore les lignes de total, TVA globale, remise globale, conditions, commentaires, signatures et notes manuscrites.
- Les lignes négatives sont valides : remise commerciale, avoir, correction, frais déduits.
- Une ligne négative peut avoir prix_unitaire négatif et prix_net négatif.
- Ne supprime pas une ligne négative si elle fait partie du tableau articles.

Colonnes articles :
- Ref / Réf / Code / Item / Ref article / Article = ref_article.
- Designation / Désignation / Description / Libellé = description.
- Qte / Qté / Qty / Quantité / Qté fact. = qte_facturee.
- PU HT / P.U. HT / Prix unit. / Prix unitaire / Unit price = prix_unitaire.
- Total HT / Net / Line net / Montant HT / Montant net / Total ligne = prix_net.
- Rem. / Rem.% / Remise / Rabais / Discount / Disc. = remise.
- TVA / Taxe / Tax / VAT = taxe.
- Les colonnes courtes à droite du tableau comme "Rem." et "TVA" sont importantes :
  lis chaque cellule de ces colonnes, même si elle contient seulement "0%", "5%" ou "V20".

Règles de montants :
- Les montants doivent être numériques.
- "1 225,16 EUR" devient 1225.16.
- "80,40 €" devient 80.40.
- "80.40 EUR" devient 80.40.
- Les quantités doivent être numériques.
- "10 Pièce" devient 10.
- "5 cartons" devient 5.
- Si prix_net est visible dans la colonne Net, Total HT, Montant HT, Montant net ou Line net, il doit être extrait.
- Ne laisse pas prix_net à null si une valeur est visible, même négative.
- Si prix_unitaire est visible dans la colonne PU HT, P.U. HT, Prix unitaire ou Unit price, il doit être extrait.
- Ne confonds pas prix_unitaire et prix_net :
  - prix_unitaire = prix d'une seule unité
  - prix_net = montant total de la ligne après quantité et remise éventuelle
- Si prix_net est absent mais calculable avec quantité × prix_unitaire après remise, tu peux le calculer seulement si c'est évident.
- Si une ligne a une remise négative ou un prix négatif visible, conserve le signe négatif.

Taxe :
- Respecte exactement le format visible dans le document.
- Si la taxe visible est "20%", retourne "20%".
- Si la taxe visible est "5,5%", retourne "5,5%".
- Si la taxe visible est "5.5%", retourne "5.5%".
- Si la taxe visible est "V20", retourne "V20".
- Si la taxe visible est "V10", retourne "V10".
- Si la taxe visible est "V5.5", retourne "V5.5".
- Ne transforme jamais "20%" en "V20%".
- Ne transforme jamais "V20" en "20%".
- Le format "V20%" est invalide sauf s'il est explicitement visible dans le document.
- Si la colonne TVA/Taxe contient V20 sur une ligne, items[].taxe = "V20".
- Ne mets pas taxe à null si une valeur est visible dans la colonne TVA/Taxe, même à droite du tableau.
- Ne confonds jamais TVA et remise.

Remise :
- Si une cellule Remise/Rabais/Discount est vide, mets null.
- Si la cellule contient 0%, mets "0%".
- Si la cellule contient 0,00 ou 0.00, mets 0 ou "0%" selon le format visible.
- Si la cellule contient 5%, 7%, 10%, 12%, etc., extrais cette valeur.
- Ne mets pas null quand la valeur visible est 0%.
- Ne mets jamais la TVA dans remise.
- Si prix_net = quantité × prix_unitaire, alors il n'y a probablement pas de remise, même si la TVA vaut 20%.
- Si une ligne contient Rem. = vide et TVA = V20 ou 20%, alors remise = null et taxe = V20 ou 20%.

Bas de page :
- net_total_ht correspond à Total HT, Net total HT, Total excl. tax, Subtotal excl. tax, Sous-total HT.
- net_total_ttc correspond à Net TTC, Grand total, Total TTC, Total incl. tax.
- mnt_remise correspond à Remise totale, Remise lignes, Discount total, Total discount.
- Si Remise totale vaut 0,00 EUR, retourne 0.0 et non null.
- Ne confonds pas Total TVA avec net_total_ttc.
- Ne confonds pas Total brut HT avec net_total_ht si un Net total HT est visible.
- Si plusieurs totaux sont visibles, privilégie les libellés "Net total HT" et "Net total TTC".

Contrôle de cohérence :
- Vérifie mentalement que les lignes extraites correspondent au tableau visible.
- Vérifie que l'adresse fournisseur ne mélange pas deux blocs différents.
- Vérifie que taxe et remise ne sont pas inversées.
- Vérifie que les items ne sont pas vides si un tableau article est visible.
- Vérifie que le JSON final est valide.
""".strip()


def run_with_timeout(fn: Callable[..., T], *args: Any, timeout_seconds: int) -> T:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise ExtractionTimeoutError(
                f"timeout après {timeout_seconds} secondes"
            ) from exc


def select_page_images(pages_dir: Path, max_pages: int | None) -> list[Path]:
    images = sorted(pages_dir.glob("page-*.png"), key=page_sort_key)

    if max_pages is not None:
        images = images[:max_pages]

    return images


def extract_one_document(
    pdf_name: str,
    pages_dir: Path,
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
    max_pages: int | None,
) -> str:
    images = select_page_images(pages_dir, max_pages)

    if not images:
        raise RuntimeError(f"Aucune image trouvée dans {pages_dir}")

    prompt = build_prompt(template)

    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image) for image in images],
            }
        ],
        options={
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )

    return response["message"]["content"]


def process_document(
    pdf_name: str,
    pages_dir: Path,
    template: dict[str, Any],
    model: str,
    num_ctx: int,
    num_predict: int,
    max_pages: int | None,
    timeout_seconds: int,
    skip_existing: bool,
) -> None:
    print(f"Extraction : {pdf_name}", flush=True)

    EXTRACTED_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    VALIDATION_DIR.mkdir(exist_ok=True)

    raw_output_path = REPORTS_DIR / f"{pdf_name}_raw_output.txt"
    extracted_path = EXTRACTED_DIR / f"{pdf_name}.json"
    validation_path = VALIDATION_DIR / f"{pdf_name}_validation.json"

    if skip_existing and extracted_path.exists() and validation_path.exists():
        print(f"SKIP existing : {pdf_name}", flush=True)
        return

    try:
        raw_content = run_with_timeout(
            extract_one_document,
            pdf_name,
            pages_dir,
            template,
            model,
            num_ctx,
            num_predict,
            max_pages,
            timeout_seconds=timeout_seconds,
        )

        raw_output_path.write_text(raw_content, encoding="utf-8")

        cleaned_content = clean_json_output(raw_content)
        model_data = json.loads(cleaned_content)

        strict_data = merge_with_template(template, model_data)
        strict_data = postprocess_document(strict_data)
        pdf_text = extract_pdf_text(pdf_name)
        strict_data = repair_entete_from_pdf_text(strict_data, pdf_text)
        strict_data = repair_items_from_pdf_text(strict_data, pdf_text)
        strict_data = postprocess_document(strict_data)

        extracted_path.write_text(
            json.dumps(strict_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        validation = validate_document(strict_data)

        validation_report = {
            "pdf_name": pdf_name,
            "status": validation["status"],
            "extracted_file": str(extracted_path),
            "raw_output_file": str(raw_output_path),
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }

        validation_path.write_text(
            json.dumps(validation_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"{validation['status']} : {pdf_name}", flush=True)

    except ExtractionTimeoutError as error:
        error_report = {
            "pdf_name": pdf_name,
            "status": "FAILED",
            "reason": "timeout",
            "error": str(error),
            "timeout_seconds": timeout_seconds,
        }

        validation_path.write_text(
            json.dumps(error_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"FAILED TIMEOUT : {pdf_name} -> {error}", flush=True)

    except json.JSONDecodeError as error:
        error_report = {
            "pdf_name": pdf_name,
            "status": "FAILED",
            "reason": "invalid_json",
            "error": str(error),
            "raw_output_file": str(raw_output_path),
        }

        validation_path.write_text(
            json.dumps(error_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"FAILED JSON : {pdf_name}", flush=True)

    except Exception as error:
        reason = "exception"
        text = str(error)

        if (
            "exceeds the available context size" in text
            or "exceed_context_size" in text
        ):
            reason = "context_size_exceeded"

        error_report = {
            "pdf_name": pdf_name,
            "status": "FAILED",
            "reason": reason,
            "error": text,
        }

        validation_path.write_text(
            json.dumps(error_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"FAILED : {pdf_name} -> {error}", flush=True)


def list_page_dirs(only: str | None = None) -> list[Path]:
    if not PAGES_ROOT.exists():
        raise FileNotFoundError(f"Dossier introuvable : {PAGES_ROOT}")

    page_dirs = sorted([path for path in PAGES_ROOT.iterdir() if path.is_dir()])

    if only:
        page_dirs = [path for path in page_dirs if path.name == only]

    return page_dirs


def normalize_taxe_value(value: Any) -> str | None:
    text = stringify_if_present(value)
    if text is None:
        return None

    normalized = text.strip().upper().replace(" ", "").replace(",", ".")

    # Formats hybrides invalides souvent inventés par le modèle.
    # Quand le modèle mélange "V20" et "20%", on garde le format taux.
    if normalized == "V20%":
        return "20%"
    if normalized == "V10%":
        return "10%"
    if normalized in {"V5.5%", "V5,5%"}:
        return "5.5%"

    return text


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        help="Traiter uniquement un dossier précis dans pages/, ex: 'CA-536-démo'",
        default=None,
    )

    parser.add_argument(
        "--model",
        help=f"Modèle Ollama à utiliser, défaut: {DEFAULT_MODEL}",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--num-ctx",
        help=f"Taille contexte Ollama, défaut: {DEFAULT_NUM_CTX}",
        type=int,
        default=DEFAULT_NUM_CTX,
    )

    parser.add_argument(
        "--num-predict",
        help=f"Nombre max de tokens générés, défaut: {DEFAULT_NUM_PREDICT}",
        type=int,
        default=DEFAULT_NUM_PREDICT,
    )

    parser.add_argument(
        "--timeout-seconds",
        help=f"Timeout par document, défaut: {DEFAULT_TIMEOUT_SECONDS}",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )

    parser.add_argument(
        "--max-pages",
        help="Limiter le nombre d'images envoyées par document. Utile pour tester avec une seule image.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-skip-existing",
        help="Retraiter même si extracted/*.json et validation/*.json existent déjà.",
        action="store_true",
    )

    args = parser.parse_args()

    if args.max_pages is not None and args.max_pages < 1:
        raise ValueError("--max-pages doit être supérieur ou égal à 1")

    template = load_template()
    page_dirs = list_page_dirs(args.only)

    if not page_dirs:
        print("Aucun dossier à traiter.")
        return

    for pages_dir in page_dirs:
        process_document(
            pdf_name=pages_dir.name,
            pages_dir=pages_dir,
            template=template,
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            max_pages=args.max_pages,
            timeout_seconds=args.timeout_seconds,
            skip_existing=not args.no_skip_existing,
        )


if __name__ == "__main__":
    main()
