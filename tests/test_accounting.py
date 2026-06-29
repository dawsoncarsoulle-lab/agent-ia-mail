from __future__ import annotations

import unittest

from scripts.extract import (
    accounting_issues,
    compute_accounting_totals,
    postprocess_document,
)


def make_doc(items: list[dict], bas_de_page: dict) -> dict:
    return {
        "entete": {
            "numero_facture": "DOC-1",
            "numero_fournisseur": "SUP-1",
            "date": "01/01/2026",
            "siren": "123456789",
            "siret": "12345678900011",
            "adresse": {
                "ligne_1": "Supplier",
                "ligne_2": None,
                "rue": "1 rue Test",
                "code_postal_ville": "75000 Paris",
            },
        },
        "items": items,
        "bas_de_page": bas_de_page,
    }


class AccountingTests(unittest.TestCase):
    def test_inconsistent_ttc_is_repaired_when_all_tax_rates_are_known(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 1,
                    "prix_unitaire": 100,
                    "prix_net": 100,
                    "taxe": "20%",
                    "remise": 0,
                    "description": "A",
                }
            ],
            {"net_total_ht": 100, "net_total_ttc": 105, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["net_total_ttc"], 120)
        self.assertEqual(accounting_issues(result), [])

    def test_numeric_discount_repairs_inconsistent_generated_net(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 25,
                    "prix_unitaire": 1.2,
                    "prix_net": 30,
                    "taxe": "V20",
                    "remise": 10,
                    "description": "A",
                }
            ],
            {"net_total_ht": None, "net_total_ttc": None, "mnt_remise": None},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["items"][0]["prix_net"], 27)
        self.assertEqual(result["bas_de_page"]["mnt_remise"], 3)

    def test_compact_vat_code_v55_means_five_point_five_percent(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 1,
                    "prix_unitaire": 100,
                    "prix_net": 100,
                    "taxe": "V55",
                    "remise": 0,
                    "description": "A",
                }
            ],
            {"net_total_ht": None, "net_total_ttc": None, "mnt_remise": None},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["net_total_ttc"], 105.5)

    def test_missing_unit_price_is_inferred_with_explicit_zero_total_discount(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 4,
                    "prix_unitaire": None,
                    "prix_net": 50,
                    "taxe": "20%",
                    "remise": None,
                    "description": "A",
                }
            ],
            {"net_total_ht": 50, "net_total_ttc": 60, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["items"][0]["prix_unitaire"], 12.5)
        self.assertEqual(result["items"][0]["remise"], "0%")

    def test_missing_unit_price_is_inferred_from_total_discount(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 2,
                    "prix_unitaire": 10,
                    "prix_net": 20,
                    "taxe": "20%",
                    "remise": "0%",
                    "description": "A",
                },
                {
                    "ref_article": "B",
                    "qte_facturee": 4,
                    "prix_unitaire": None,
                    "prix_net": 36,
                    "taxe": "20%",
                    "remise": None,
                    "description": "B",
                },
            ],
            {"net_total_ht": 56, "net_total_ttc": 67.2, "mnt_remise": 4},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["items"][1]["prix_unitaire"], 10)
        self.assertEqual(result["items"][1]["remise"], "10%")

    def test_only_missing_line_net_is_inferred_from_footer(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 2,
                    "prix_unitaire": None,
                    "prix_net": None,
                    "taxe": "20%",
                    "remise": None,
                    "description": "A",
                },
                {
                    "ref_article": "B",
                    "qte_facturee": 1,
                    "prix_unitaire": 20,
                    "prix_net": 20,
                    "taxe": "20%",
                    "remise": "0%",
                    "description": "B",
                },
            ],
            {"net_total_ht": 50, "net_total_ttc": 60, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["items"][0]["prix_net"], 30)
        self.assertEqual(result["items"][0]["prix_unitaire"], 15)

    def test_ambiguous_missing_unit_prices_are_not_inferred(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": ref,
                    "qte_facturee": 2,
                    "prix_unitaire": None,
                    "prix_net": 20,
                    "taxe": "20%",
                    "remise": None,
                    "description": ref,
                }
                for ref in ("A", "B")
            ],
            {"net_total_ht": 40, "net_total_ttc": 48, "mnt_remise": None},
        )

        result = postprocess_document(doc)

        self.assertIsNone(result["items"][0]["prix_unitaire"])
        self.assertIsNone(result["items"][1]["prix_unitaire"])

    def test_zero_discount_totals_are_kept(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 2,
                    "prix_unitaire": 10,
                    "prix_net": 20,
                    "taxe": "20%",
                    "remise": "0%",
                    "description": "A",
                }
            ],
            {"net_total_ht": 20, "net_total_ttc": 24, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["net_total_ht"], 20)
        self.assertEqual(result["bas_de_page"]["mnt_remise"], 0)
        self.assertEqual(accounting_issues(result), [])

    def test_line_discount_total_is_calculated_from_items(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 10,
                    "prix_unitaire": 10,
                    "prix_net": 90,
                    "taxe": "20%",
                    "remise": "10%",
                    "description": "A",
                }
            ],
            {"net_total_ht": 90, "net_total_ttc": 108, "mnt_remise": None},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["mnt_remise"], 10)
        self.assertEqual(accounting_issues(result), [])

    def test_multiple_tax_rates_are_grouped(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 1,
                    "prix_unitaire": 100,
                    "prix_net": 100,
                    "taxe": "5.5%",
                    "remise": "0%",
                    "description": "A",
                },
                {
                    "ref_article": "B",
                    "qte_facturee": 1,
                    "prix_unitaire": 50,
                    "prix_net": 50,
                    "taxe": "20%",
                    "remise": "0%",
                    "description": "B",
                },
            ],
            {"net_total_ht": 150, "net_total_ttc": 165.5, "mnt_remise": 0},
        )

        totals = compute_accounting_totals(postprocess_document(doc))

        self.assertEqual(totals["tax_bases"], {"0.0550": 100.0, "0.2000": 50.0})
        self.assertEqual(totals["total_vat"], 15.5)
        self.assertEqual(totals["expected_ttc"], 165.5)

    def test_wrong_net_total_ht_is_corrected_from_line_sum(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 1,
                    "prix_unitaire": 380,
                    "prix_net": 380,
                    "taxe": "5.5%",
                    "remise": "0%",
                    "description": "A",
                },
                {
                    "ref_article": "B",
                    "qte_facturee": 1,
                    "prix_unitaire": 99.2,
                    "prix_net": 99.2,
                    "taxe": "5.5%",
                    "remise": "0%",
                    "description": "B",
                },
            ],
            {"net_total_ht": 479.6, "net_total_ttc": 505.56, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["net_total_ht"], 479.2)
        self.assertEqual(accounting_issues(result), [])

    def test_gross_total_is_not_kept_as_discount_total(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 2,
                    "prix_unitaire": 50,
                    "prix_net": 95,
                    "taxe": "20%",
                    "remise": "5%",
                    "description": "A",
                }
            ],
            {"net_total_ht": 95, "net_total_ttc": 114, "mnt_remise": 100},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["mnt_remise"], 5)
        self.assertEqual(accounting_issues(result), [])

    def test_incomplete_items_do_not_rewrite_footer_totals(self) -> None:
        doc = make_doc(
            [
                {
                    "ref_article": "A",
                    "qte_facturee": 1,
                    "prix_unitaire": 100,
                    "prix_net": 100,
                    "taxe": "20%",
                    "remise": "0%",
                    "description": "A",
                },
                {
                    "ref_article": "B",
                    "qte_facturee": None,
                    "prix_unitaire": None,
                    "prix_net": None,
                    "taxe": "20%",
                    "remise": None,
                    "description": "B",
                },
            ],
            {"net_total_ht": 250, "net_total_ttc": 300, "mnt_remise": 0},
        )

        result = postprocess_document(doc)

        self.assertEqual(result["bas_de_page"]["net_total_ht"], 250)
        self.assertEqual(result["bas_de_page"]["net_total_ttc"], 300)


if __name__ == "__main__":
    unittest.main()
