"""Auditable histology classification for the breast imaging cohort.

The primary category is mutually exclusive and is intended for Table 1.
The component labels preserve every explicitly recorded malignant component.
No histology field is used to derive Group, GroundTruth, or model scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata

import pandas as pd


IDC = "Invasive Ductal Carcinoma / NST"
ILC = "Invasive Lobular Carcinoma"
DCIS = "Ductal Carcinoma In Situ"
MICROINVASIVE = "Microinvasive Carcinoma"
INVASIVE_NOS = "Invasive Carcinoma, NOS"
PAPILLARY = "Papillary Carcinoma"
MICROPAPILLARY = "Invasive Micropapillary Carcinoma"
MUCINOUS = "Mucinous Carcinoma"
PAGET = "Paget Disease"
METAPLASTIC = "Metaplastic Carcinoma"
APOCRINE = "Apocrine Carcinoma"
TUBULAR = "Tubular Carcinoma"
CARCINOSARCOMA = "Carcinosarcoma"
OTHER_MALIGNANT = "Other Malignant Histology"
MIXED = "Mixed / Multiple Malignant Histologies"
MIXED_UNSPECIFIED = "Mixed Malignant Histology, Unspecified"
UNAVAILABLE = "Histology Unavailable / Discordant"
INDETERMINATE = "Indeterminate Histology"
UNMAPPED = "Unmapped Histology"
NOT_APPLICABLE = "Not Applicable (GroundTruth=0)"


@dataclass(frozen=True)
class HistologyResult:
    Histology_normalized: str
    Histology_primary_cat: str
    Histology_table1_cat: str
    Histology_component_labels: str
    Histology_is_multicomponent: bool
    Histology_review_flag: bool
    Histology_review_reason: str


def normalize_histology(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "", text)
    text = text.rstrip("。.;；")
    return text.replace("粘液", "黏液")


def _table1_category(primary: str) -> str:
    if primary == IDC:
        return "Invasive Ductal Carcinoma"
    if primary == DCIS:
        return "Ductal Carcinoma In Situ"
    if primary in {MIXED, MIXED_UNSPECIFIED}:
        return "Mixed / Multiple Malignant Histologies"
    if primary in {UNAVAILABLE, INDETERMINATE, UNMAPPED}:
        return "Histology Unavailable / Indeterminate"
    return "Other Malignancies"


def _result(
    normalized: str,
    primary: str,
    components: tuple[str, ...] = (),
    *,
    review: bool = False,
    reason: str = "",
) -> HistologyResult:
    unique_components = tuple(dict.fromkeys(components))
    return HistologyResult(
        Histology_normalized=normalized,
        Histology_primary_cat=primary,
        Histology_table1_cat=_table1_category(primary),
        Histology_component_labels=" | ".join(unique_components),
        Histology_is_multicomponent=len(unique_components) > 1,
        Histology_review_flag=review,
        Histology_review_reason=reason,
    )


def _single(primary: str) -> tuple[str, tuple[str, ...], bool, str]:
    return primary, (primary,), False, ""


def _review(
    primary: str, reason: str, components: tuple[str, ...] = ()
) -> tuple[str, tuple[str, ...], bool, str]:
    return primary, components, True, reason


# Exact, reviewed mappings for every malignant-cohort pathology text currently
# present in meta_data_final_sampled.csv. New text must not be guessed: it is
# classified as UNMAPPED and sent to review.
MALIGNANT_EXACT_MAP: dict[
    str, tuple[str, tuple[str, ...], bool, str]
] = {
    "浸润性导管癌": _single(IDC),
    "导管原位癌": _single(DCIS),
    "导管原位癌伴微小浸润": (MICROINVASIVE, (DCIS, MICROINVASIVE), False, ""),
    "浸润性小叶癌": _single(ILC),
    "浸润性癌": _single(INVASIVE_NOS),
    "导管原位癌伴间质浸润": (INVASIVE_NOS, (DCIS, INVASIVE_NOS), False, ""),
    "导管原位癌伴浸润性导管癌(6mm)": (MIXED, (DCIS, IDC), False, ""),
    "非特殊型浸润性癌": _single(IDC),
    "非特殊型浸润癌": _single(IDC),
    "浸润性实性乳头状癌": _single(PAPILLARY),
    "实性乳头状癌": _single(PAPILLARY),
    "以导管原位癌为主的浸润性导管癌": (DCIS, (DCIS, IDC), False, ""),
    "黏液癌": _single(MUCINOUS),
    "部分浸润性导管癌、部分浸润性小叶癌": (MIXED, (IDC, ILC), False, ""),
    "导管内乳头状癌伴浸润性导管癌": (MIXED, (PAPILLARY, IDC), False, ""),
    "导管内实性乳头状癌": _single(PAPILLARY),
    "乳头Paget病伴导管原位癌": (MIXED, (PAGET, DCIS), False, ""),
    "实性乳头状癌伴间质浸润": _single(PAPILLARY),
    "乳头Paget病": _single(PAGET),
    "纤维腺瘤及乳腺病": _review(
        UNAVAILABLE, "Benign pathology text conflicts with GroundTruth=1"
    ),
    "浸润性实性乳头状癌合并黏液癌": (MIXED, (PAPILLARY, MUCINOUS), False, ""),
    "混合型黏液癌,部分浸润性导管癌": (MIXED, (MUCINOUS, IDC), False, ""),
    "实性乳头状癌伴多灶微小浸润": _single(PAPILLARY),
    "包裹性乳头状癌": _single(PAPILLARY),
    "浸润性导管癌伴黏液分泌": _single(IDC),
    "大汗腺型导管原位癌": _single(DCIS),
    "无手术病理": _review(UNAVAILABLE, "No surgical pathology available"),
    "高级别导管原位癌": _single(DCIS),
    "浸润性大汗腺癌": _single(APOCRINE),
    "浸润性微乳头状癌": _single(MICROPAPILLARY),
    "多形性浸润性小叶癌": _single(ILC),
    "包裹性乳头状癌混合浸润性导管癌": (MIXED, (PAPILLARY, IDC), False, ""),
    "混合性腺癌": _review(
        MIXED_UNSPECIFIED, "Mixed malignancy is recorded without named components"
    ),
    "黏液腺癌": _single(MUCINOUS),
    "伴间叶分化的化生性癌": _single(METAPLASTIC),
    "混合浸润性导管癌及浸润性微乳头状癌": (MIXED, (IDC, MICROPAPILLARY), False, ""),
    "化生性癌,化生成分为鳞状细胞癌": _single(METAPLASTIC),
    "浸润性乳头状癌": _single(PAPILLARY),
    "导管内乳头状瘤伴导管上皮增生活跃": _review(
        UNAVAILABLE, "Benign pathology text conflicts with GroundTruth=1"
    ),
    "导管原位癌伴微浸润": (MICROINVASIVE, (DCIS, MICROINVASIVE), False, ""),
    "浸润性癌伴神经内分泌分化": _single(INVASIVE_NOS),
    "导管内癌": _single(DCIS),
    "伴导管原位癌的黏液腺癌": (MIXED, (MUCINOUS, DCIS), False, ""),
    "导管原位癌为主的浸润性导管癌": (DCIS, (DCIS, IDC), False, ""),
    "导管原位癌伴微小间质浸润": (MICROINVASIVE, (DCIS, MICROINVASIVE), False, ""),
    "伴大汗腺化生的浸润性癌": _single(INVASIVE_NOS),
    "浸润性小管癌": _single(TUBULAR),
    "癌肉瘤(腺癌15%+平滑肌肉瘤85%)": _single(CARCINOSARCOMA),
    "浸润性癌伴大汗腺化生": _single(INVASIVE_NOS),
    "混合性浸润性癌": _review(
        MIXED_UNSPECIFIED, "Mixed invasive malignancy is recorded without named components"
    ),
    "导管原位癌伴间质微小浸润": (MICROINVASIVE, (DCIS, MICROINVASIVE), False, ""),
    "低级别大汗腺型导管原位癌": _single(DCIS),
    "浸润性小叶癌(左)": _single(ILC),
    "低分化腺癌组织,考虑乳腺来源": _review(
        OTHER_MALIGNANT, "Breast origin is suspected rather than definitive", (OTHER_MALIGNANT,)
    ),
    "导管内乳头状病变,实性乳头状癌不能除外": _review(
        INDETERMINATE, "Malignancy is not confirmed in the pathology text"
    ),
    "原位型乳头状癌": _single(PAPILLARY),
    "低级别导管原位癌": _single(DCIS),
    "我院未手术": _review(UNAVAILABLE, "No surgical pathology available"),
    "": _review(UNAVAILABLE, "Histology text is missing"),
}


def classify_histology(value: object, group: object) -> HistologyResult:
    normalized = normalize_histology(value)
    if str(group).strip() not in {"TP", "FN"}:
        return HistologyResult(
            Histology_normalized=normalized,
            Histology_primary_cat=NOT_APPLICABLE,
            Histology_table1_cat=NOT_APPLICABLE,
            Histology_component_labels="",
            Histology_is_multicomponent=False,
            Histology_review_flag=False,
            Histology_review_reason="",
        )

    mapping = MALIGNANT_EXACT_MAP.get(normalized)
    if mapping is None:
        return _result(
            normalized,
            UNMAPPED,
            review=True,
            reason="New or unreviewed malignant-cohort histology text",
        )

    primary, components, review, reason = mapping
    return _result(
        normalized,
        primary,
        components,
        review=review,
        reason=reason,
    )


def classify_histology_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = {"HistologicalSubtype", "Group"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    rows = [
        asdict(classify_histology(histology, group))
        for histology, group in zip(data["HistologicalSubtype"], data["Group"])
    ]
    return pd.DataFrame(rows, index=data.index)


def summarize_malignant_histology(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "Group",
        "HistologicalSubtype",
        "Histology_primary_cat",
        "Histology_table1_cat",
        "Histology_component_labels",
        "Histology_review_flag",
        "Histology_review_reason",
    }
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    malignant = data[data["Group"].isin(["TP", "FN"])].copy()
    denominator = len(malignant)
    if denominator == 0:
        raise ValueError("No TP/FN cases were available for histology summary")

    if malignant["Histology_primary_cat"].eq(NOT_APPLICABLE).any():
        raise AssertionError("A TP/FN case was assigned a non-malignant histology status")

    primary = (
        malignant.groupby("Histology_table1_cat", dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    primary["N"] = denominator
    primary["percent"] = primary["n"] / denominator * 100
    primary["n_percent"] = primary.apply(
        lambda row: f"{int(row['n'])}/{int(row['N'])} ({row['percent']:.1f}%)",
        axis=1,
    )
    primary = primary.sort_values(
        ["n", "Histology_table1_cat"], ascending=[False, True]
    ).reset_index(drop=True)
    if int(primary["n"].sum()) != denominator:
        raise AssertionError("Mutually exclusive Table 1 histology counts do not reconcile")

    component_rows = malignant.loc[
        malignant["Histology_component_labels"].ne(""),
        ["Histology_component_labels"],
    ].copy()
    component_rows["Histology_component"] = component_rows[
        "Histology_component_labels"
    ].str.split(" | ", regex=False)
    component_rows = component_rows.explode("Histology_component")
    components = (
        component_rows.groupby("Histology_component", dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["n", "Histology_component"], ascending=[False, True])
        .reset_index(drop=True)
    )
    components["N"] = denominator
    components["percent"] = components["n"] / denominator * 100
    components["n_percent"] = components.apply(
        lambda row: f"{int(row['n'])}/{int(row['N'])} ({row['percent']:.1f}%)",
        axis=1,
    )

    review_columns = [
        column
        for column in [
            "PatientName",
            "Group",
            "HistologicalSubtype",
            "Histology_primary_cat",
            "Histology_review_reason",
        ]
        if column in malignant.columns
    ]
    review = malignant.loc[
        malignant["Histology_review_flag"], review_columns
    ].reset_index(drop=True)
    return primary, components, review
