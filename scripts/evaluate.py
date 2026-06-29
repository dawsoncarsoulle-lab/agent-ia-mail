"""
Evaluation des extractions JSON contre data/raw/expected_extractions.json.

Usage:
    uv run python -m scripts.evaluate
    uv run python -m scripts.evaluate --only commande_achat_CA-2026-201_fax_low_contrast

Sorties:
    evaluation/<stem>_eval.json
    reports/scoreboard.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_PATH = Path("data/raw/expected_extractions.json")
EXTRACTED_DIR = Path("extracted")
EVALUATION_DIR = Path("evaluation")
SCOREBOARD_PATH = Path("reports/scoreboard.json")

IGNORE_KEYS = {"_meta", "_test_difficulty", "_note"}

# Poids simples. Tu peux les ajuster selon ce qui est critique pour Cerealog.
FIELD_WEIGHTS = {
    "entete.numero_facture": 12,
    "entete.numero_fournisseur": 8,
    "entete.date": 6,
    "entete.siren": 5,
    "entete.siret": 7,
    "entete.adresse.ligne_1": 5,
    "entete.adresse.rue": 4,
    "entete.adresse.code_postal_ville": 4,
    "bas_de_page.net_total_ht": 8,
    "bas_de_page.net_total_ttc": 8,
    "bas_de_page.mnt_remise": 4,
    "items.ref_article": 8,
    "items.description": 8,
    "items.qte_facturee": 8,
    "items.prix_unitaire": 8,
    "items.prix_net": 8,
    "items.taxe": 4,
    "items.remise": 4,
}


@dataclass
class Check:
    path: str
    expected: Any
    actual: Any
    ok: bool
    weight: int


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Normalise les espaces multiples, accents non gérés volontairement.
        text = re.sub(r"\s+", " ", text)
        # SIREN/SIRET affichés avec espaces.
        if re.fullmatch(r"[\d ]{9,17}", text):
            return re.sub(r"\D", "", text)
        # Montants français.
        money = text.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
        money = money.replace("EUR", "").replace("€", "").replace(",", ".")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", money):
            n = float(money)
            return int(n) if n.is_integer() else round(n, 2)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        return text.lower()
    if isinstance(value, float):
        return round(value, 2)
    return value


def equalish(expected: Any, actual: Any) -> bool:
    e = normalize_scalar(expected)
    a = normalize_scalar(actual)
    if isinstance(e, (int, float)) and isinstance(a, (int, float)):
        return abs(float(e) - float(a)) <= 0.02
    return e == a


def get_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def item_key(item: dict[str, Any]) -> str | None:
    ref = normalize_scalar(item.get("ref_article"))
    return str(ref) if ref else None


def evaluate_one(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    checks: list[Check] = []

    # Champs hors items.
    for path, weight in FIELD_WEIGHTS.items():
        if path.startswith("items."):
            continue
        checks.append(
            Check(
                path=path,
                expected=get_path(expected, path),
                actual=get_path(actual, path),
                ok=equalish(get_path(expected, path), get_path(actual, path)),
                weight=weight,
            )
        )

    expected_items = expected.get("items") or []
    actual_items = actual.get("items") or []
    actual_by_ref = {
        key: item
        for item in actual_items
        if isinstance(item, dict) and (key := item_key(item))
    }

    for idx, exp_item in enumerate(expected_items):
        if not isinstance(exp_item, dict):
            continue
        key = item_key(exp_item)
        act_item = actual_by_ref.get(key) if key else None
        if (
            act_item is None
            and idx < len(actual_items)
            and isinstance(actual_items[idx], dict)
        ):
            act_item = actual_items[idx]

        for item_field in [
            k.removeprefix("items.") for k in FIELD_WEIGHTS if k.startswith("items.")
        ]:
            path = f"items[{idx + 1}].{item_field}"
            weight = FIELD_WEIGHTS[f"items.{item_field}"]
            expected_value = exp_item.get(item_field)
            actual_value = (
                act_item.get(item_field) if isinstance(act_item, dict) else None
            )
            checks.append(
                Check(
                    path=path,
                    expected=expected_value,
                    actual=actual_value,
                    ok=equalish(expected_value, actual_value),
                    weight=weight,
                )
            )

    total = sum(c.weight for c in checks)
    gained = sum(c.weight for c in checks if c.ok)
    score = round((gained / total) * 100, 2) if total else 0.0
    failed = [c.__dict__ for c in checks if not c.ok]

    return {
        "score": score,
        "status": "OK" if score >= 90 else "NEEDS_REVIEW" if score >= 70 else "FAILED",
        "passed_points": gained,
        "total_points": total,
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score les extractions contre expected_extractions.json"
    )
    parser.add_argument("--only", default=None, help="stem ou nom PDF à évaluer")
    args = parser.parse_args()

    if not EXPECTED_PATH.exists():
        raise FileNotFoundError(EXPECTED_PATH)

    expected_all = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    EVALUATION_DIR.mkdir(exist_ok=True)
    SCOREBOARD_PATH.parent.mkdir(exist_ok=True)

    scoreboard = []
    for pdf_name, expected in expected_all.items():
        stem = Path(pdf_name).stem
        if args.only and args.only not in {stem, pdf_name}:
            continue

        extracted_path = EXTRACTED_DIR / f"{stem}.json"
        if not extracted_path.exists():
            scoreboard.append(
                {"pdf_name": pdf_name, "status": "MISSING_EXTRACTION", "score": 0}
            )
            continue

        actual = json.loads(extracted_path.read_text(encoding="utf-8"))
        result = evaluate_one(expected, actual)
        result["pdf_name"] = pdf_name
        result["extracted_file"] = str(extracted_path)

        out = EVALUATION_DIR / f"{stem}_eval.json"
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        scoreboard.append(
            {
                "pdf_name": pdf_name,
                "status": result["status"],
                "score": result["score"],
                "failed_checks": len(result["failed_checks"]),
                "evaluation_file": str(out),
            }
        )

    SCOREBOARD_PATH.write_text(
        json.dumps(scoreboard, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for row in scoreboard:
        print(f"{row['score']:>6}  {row['status']:<18} {row['pdf_name']}")
    print(f"\nRapport: {SCOREBOARD_PATH}")


if __name__ == "__main__":
    main()
