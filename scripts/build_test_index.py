"""Build a small FAISS index from dummy scholarship data for local testing.

Usage:
    conda run -n llm python scripts/build_test_index.py

Output:
    /tmp/scholarship_test/faiss.index
    /tmp/scholarship_test/metadata.parquet
    /tmp/scholarship_test/schemes.csv   (the dummy source data)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure src/ is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OUT_DIR = "/tmp/scholarship_test"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Dummy scholarship schemes — 15 rows covering diverse categories
# ---------------------------------------------------------------------------
DUMMY_SCHEMES = [
    {
        "scheme_id": "NSP_PRE_MAT_SC",
        "scheme_name": "Pre-Matric Scholarship for SC Students",
        "administering_body": "Ministry of Social Justice and Empowerment",
        "level": "Central",
        "state": "",
        "eligible_categories": "SC",
        "min_income_limit": 0,
        "max_income_limit": 200000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 8, Class 10",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "Day scholar: ₹150/month + ₹750 ad hoc; Hosteller: ₹350/month + ₹1,000 ad hoc",
        "application_deadline": "October 31 annually",
        "description_text": (
            "The Pre-Matric Scholarship Scheme is for SC students studying in Classes 8 and 10 "
            "in government or government-aided schools. Family income must not exceed ₹2 lakh per annum. "
            "Awards cover monthly maintenance and ad hoc grants for books and stationery. "
            "Applications are submitted on the National Scholarship Portal (NSP)."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "NSP_POST_MAT_SC_ST",
        "scheme_name": "Post-Matric Scholarship for SC and ST Students",
        "administering_body": "Ministry of Social Justice and Empowerment",
        "level": "Central",
        "state": "",
        "eligible_categories": "SC, ST",
        "min_income_limit": 0,
        "max_income_limit": 250000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 12, Undergraduate, Postgraduate, PhD",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "Maintenance allowance ₹530–₹1,200/month depending on course level",
        "application_deadline": "November 30 annually",
        "description_text": (
            "Post-Matric Scholarship for SC and ST students covers Class 11 onwards including "
            "undergraduate, postgraduate, and doctoral programmes. The annual family income ceiling "
            "is ₹2.5 lakh. The scholarship covers tuition fees, maintenance allowance, and study "
            "tour charges. Students must apply through the NSP portal."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "PM_YASASVI_OBC",
        "scheme_name": "PM Young Achievers Scholarship Award Scheme (YASASVI) for OBC",
        "administering_body": "Ministry of Social Justice and Empowerment",
        "level": "Central",
        "state": "",
        "eligible_categories": "OBC, EWS",
        "min_income_limit": 0,
        "max_income_limit": 250000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 8, Class 10",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹75,000/year for Class 9–10; ₹1,25,000/year for Class 11–12",
        "application_deadline": "August 31 annually",
        "description_text": (
            "PM YASASVI provides scholarships to OBC, EWS, and DNT students in Classes 9 and 11 "
            "studying in top residential schools. Family income must not exceed ₹2.5 lakh per annum. "
            "Selection is based on the YASASVI entrance test conducted by NTA. "
            "The scheme covers tuition fees and hostel charges at the residential school."
        ),
        "source_url": "https://yet.nta.ac.in",
    },
    {
        "scheme_id": "BEGUM_HAZRAT_MAHAL",
        "scheme_name": "Begum Hazrat Mahal National Scholarship for Minorities",
        "administering_body": "Maulana Azad Education Foundation",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 200000,
        "eligible_gender": "Female",
        "eligible_education_levels": "Class 10, Class 12",
        "eligible_disability": "All",
        "eligible_minority": "Yes",
        "award_amount": "₹5,000 for Class 9–10; ₹6,000 for Class 11–12",
        "application_deadline": "September 30 annually",
        "description_text": (
            "Begum Hazrat Mahal National Scholarship is exclusively for meritorious girls belonging "
            "to Muslim, Christian, Sikh, Buddhist, Jain, or Zoroastrian minority communities. "
            "Family income must not exceed ₹2 lakh per annum. "
            "Students must have scored at least 50% marks in the previous qualifying examination. "
            "Applications are submitted through the NSP portal."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "NSP_DISABILITY",
        "scheme_name": "Pre and Post Matric Scholarship for Students with Disabilities",
        "administering_body": "Ministry of Social Justice and Empowerment",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 250000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 8, Class 10, Class 12, Undergraduate, Postgraduate, PhD",
        "eligible_disability": "Yes",
        "eligible_minority": "All",
        "award_amount": "₹500–₹2,000/month depending on course level",
        "application_deadline": "October 31 annually",
        "description_text": (
            "This scholarship is for students with benchmark disabilities (at least 40% disability) "
            "pursuing education from Class 1 upwards including professional and technical courses. "
            "Family income must not exceed ₹2.5 lakh per annum. "
            "The scholarship covers maintenance allowance, book grant, and reader allowance for "
            "visually impaired students. Apply on the NSP portal with disability certificate."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "CENTRAL_SECTOR_CS",
        "scheme_name": "Central Sector Scholarship Scheme for College and University Students",
        "administering_body": "Department of Higher Education",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 450000,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate, Postgraduate",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹10,000/year for UG (1st–3rd year); ₹20,000/year for PG",
        "application_deadline": "October 31 annually",
        "description_text": (
            "The Central Sector Scholarship Scheme supports meritorious students from families with "
            "annual income below ₹4.5 lakh. Students must be in the top 20 percentile in their "
            "Class 12 board examination. The scholarship is tenable for regular degree courses "
            "at accredited colleges and universities. Not applicable to professional courses like "
            "engineering and medical at private colleges."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "NSP_MINORITY_PRE_MAT",
        "scheme_name": "Pre-Matric Scholarship for Minority Communities",
        "administering_body": "Ministry of Minority Affairs",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 100000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 8, Class 10",
        "eligible_disability": "All",
        "eligible_minority": "Yes",
        "award_amount": "Day scholar: ₹1,000/year; Hosteller: ₹10,000/year",
        "application_deadline": "October 31 annually",
        "description_text": (
            "Pre-Matric Scholarship for students belonging to Muslim, Christian, Sikh, Buddhist, "
            "Jain, and Zoroastrian minority communities in Classes 1 to 10. "
            "Family income must not exceed ₹1 lakh per annum. "
            "At least 50% of available scholarships are reserved for girl students. "
            "Students must have scored at least 50% marks in the previous class examination."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "NSP_MCM_MINORITY",
        "scheme_name": "Merit-cum-Means Scholarship for Minority Communities",
        "administering_body": "Ministry of Minority Affairs",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 250000,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate, Postgraduate",
        "eligible_disability": "All",
        "eligible_minority": "Yes",
        "award_amount": "Course fee up to ₹20,000/year + maintenance ₹10,000/year",
        "application_deadline": "October 31 annually",
        "description_text": (
            "Merit-cum-Means Scholarship for Muslim, Christian, Sikh, Buddhist, Jain, and "
            "Zoroastrian minority students pursuing technical and professional undergraduate and "
            "postgraduate courses. Family income must not exceed ₹2.5 lakh per annum. "
            "Students must have scored at least 50% marks in the previous qualifying examination. "
            "30% of scholarships are reserved for girl students."
        ),
        "source_url": "https://scholarships.gov.in",
    },
    {
        "scheme_id": "MH_RAJASHRI_SHAHU",
        "scheme_name": "Rajarshi Chhatrapati Shahu Maharaj Merit Scholarship",
        "administering_body": "Social Justice Department, Government of Maharashtra",
        "level": "State",
        "state": "Maharashtra",
        "eligible_categories": "OBC, NT, SBC",
        "min_income_limit": 0,
        "max_income_limit": 800000,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate, Postgraduate",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹5,000–₹15,000/year depending on course",
        "application_deadline": "December 31 annually",
        "description_text": (
            "The Rajarshi Chhatrapati Shahu Maharaj Merit Scholarship is for OBC, NT, and SBC "
            "students domiciled in Maharashtra. Family income must not exceed ₹8 lakh per annum. "
            "Students must have scored at least 60% marks in the qualifying examination. "
            "The scholarship is available for undergraduate and postgraduate degree courses at "
            "Maharashtra government and government-aided colleges."
        ),
        "source_url": "https://mahaeschol.maharashtra.gov.in",
    },
    {
        "scheme_id": "KA_SC_ST_HOSTELLER",
        "scheme_name": "SC/ST Pre-Matric Scholarship Karnataka",
        "administering_body": "Department of Social Welfare, Government of Karnataka",
        "level": "State",
        "state": "Karnataka",
        "eligible_categories": "SC, ST",
        "min_income_limit": 0,
        "max_income_limit": 150000,
        "eligible_gender": "All",
        "eligible_education_levels": "Class 8, Class 10",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹800–₹1,500/month for hostellers; ₹400–₹600/month for day scholars",
        "application_deadline": "September 30 annually",
        "description_text": (
            "Karnataka state Pre-Matric Scholarship for SC and ST students in Classes 1 to 10 "
            "at government or government-aided schools. Family income must not exceed ₹1.5 lakh "
            "per annum. Hostellers receive higher maintenance allowance. "
            "Applications are submitted through the Karnataka Scholarship Portal (prerana.karnataka.gov.in). "
            "Students must be domiciled in Karnataka."
        ),
        "source_url": "https://prerana.karnataka.gov.in",
    },
    {
        "scheme_id": "AICTE_PG_GIRL",
        "scheme_name": "AICTE Pragati Scholarship for Girls",
        "administering_body": "All India Council for Technical Education (AICTE)",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 800000,
        "eligible_gender": "Female",
        "eligible_education_levels": "Undergraduate",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹50,000/year (tuition fee or actual, whichever is less) + ₹2,000/month incidental",
        "application_deadline": "October 31 annually",
        "description_text": (
            "AICTE Pragati Scholarship is exclusively for girl students pursuing first-year "
            "undergraduate degree or diploma in AICTE-approved technical institutions. "
            "Family income must not exceed ₹8 lakh per annum. "
            "Only one girl child per family is eligible per year. "
            "The scholarship covers tuition fees up to ₹50,000 per annum plus incidental charges."
        ),
        "source_url": "https://aicte-india.org/schemes/students-development/pragati",
    },
    {
        "scheme_id": "AICTE_SAKSHAM",
        "scheme_name": "AICTE Saksham Scholarship for Specially-Abled Students",
        "administering_body": "All India Council for Technical Education (AICTE)",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 800000,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate",
        "eligible_disability": "Yes",
        "eligible_minority": "All",
        "award_amount": "₹50,000/year (tuition fee or actual, whichever is less) + ₹2,000/month incidental",
        "application_deadline": "October 31 annually",
        "description_text": (
            "AICTE Saksham Scholarship is for students with benchmark disabilities (at least 40% "
            "disability certificate) pursuing technical undergraduate degree or diploma in "
            "AICTE-approved institutions. Family income must not exceed ₹8 lakh per annum. "
            "The scholarship covers tuition fees up to ₹50,000 per year plus ₹2,000 per month "
            "as incidental charges for books, stationery, and transport."
        ),
        "source_url": "https://aicte-india.org/schemes/students-development/saksham",
    },
    {
        "scheme_id": "INSPIRE_SCHOLARSHIP",
        "scheme_name": "INSPIRE Scholarship for Higher Education (SHE)",
        "administering_body": "Department of Science and Technology",
        "level": "Central",
        "state": "",
        "eligible_categories": "All",
        "min_income_limit": 0,
        "max_income_limit": 9999999,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate, Postgraduate",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹80,000/year scholarship + ₹20,000/year summer attachment fee",
        "application_deadline": "November 30 annually",
        "description_text": (
            "INSPIRE Scholarship for Higher Education supports students pursuing Basic and Natural "
            "Sciences at undergraduate and postgraduate levels. There is no income limit — the "
            "selection is purely merit-based: students must be in the top 1% in their Class 12 "
            "board examination or qualify through national competitive exams like JEE/NEET. "
            "The scholarship is ₹80,000 per year for 5 years covering BSc and MSc."
        ),
        "source_url": "https://online-inspire.gov.in",
    },
    {
        "scheme_id": "UP_SC_GIRLS_PHD",
        "scheme_name": "Fellowship for SC/ST Girls for PhD Research",
        "administering_body": "University Grants Commission (UGC)",
        "level": "Central",
        "state": "",
        "eligible_categories": "SC, ST",
        "min_income_limit": 0,
        "max_income_limit": 9999999,
        "eligible_gender": "Female",
        "eligible_education_levels": "PhD",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹25,000/month (JRF); ₹28,000/month (SRF) + contingency grant",
        "application_deadline": "Rolling",
        "description_text": (
            "UGC Fellowship for SC/ST Girl students pursuing PhD in any recognised university. "
            "Applicants must have qualified NET/GATE or have been admitted to a PhD programme "
            "through a national-level entrance test. The fellowship covers monthly stipend of "
            "₹25,000 for the first two years (JRF) and ₹28,000 for subsequent years (SRF), "
            "plus annual contingency grants for research materials."
        ),
        "source_url": "https://ugc.ac.in/scholarships",
    },
    {
        "scheme_id": "EWS_GENERAL_MERIT",
        "scheme_name": "Prime Minister's Scholarship Scheme for EWS Students",
        "administering_body": "Ministry of Education",
        "level": "Central",
        "state": "",
        "eligible_categories": "General, EWS",
        "min_income_limit": 0,
        "max_income_limit": 800000,
        "eligible_gender": "All",
        "eligible_education_levels": "Undergraduate, Postgraduate",
        "eligible_disability": "All",
        "eligible_minority": "All",
        "award_amount": "₹25,000/year for boys; ₹30,000/year for girls",
        "application_deadline": "October 15 annually",
        "description_text": (
            "Prime Minister's Scholarship Scheme for Economically Weaker Section (EWS) students "
            "from General category families with annual income below ₹8 lakh. Students must have "
            "secured at least 60% marks in Class 12. The scholarship supports undergraduate and "
            "postgraduate studies at recognised universities. Girl students receive a higher amount. "
            "Applications are processed through the National Scholarship Portal."
        ),
        "source_url": "https://scholarships.gov.in",
    },
]


def build_text(row: dict) -> str:
    def s(k):
        v = row.get(k, "")
        return str(v).strip() if v else ""

    return (
        f"Scheme: {s('scheme_name')}. "
        f"Administered by: {s('administering_body')} ({s('level')}). "
        f"Eligible categories: {s('eligible_categories')}. "
        f"Income limit: below \u20b9{s('max_income_limit')}. "
        f"Education: {s('eligible_education_levels')}. "
        f"Gender: {s('eligible_gender')}. "
        f"Disability: {s('eligible_disability')}. "
        f"Minority: {s('eligible_minority')}. "
        f"Award: {s('award_amount')}. "
        f"Deadline: {s('application_deadline')}. "
        f"Details: {s('description_text')}"
    )


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(DUMMY_SCHEMES)
    df["text"] = df.apply(build_text, axis=1)

    # Save the raw CSV too (useful for inspecting)
    csv_path = out / "schemes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} schemes → {csv_path}")

    # Embed
    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = df["text"].tolist()
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS IndexFlatIP (cosine sim on normalised vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # Save
    index_path = str(out / "faiss.index")
    meta_path = str(out / "metadata.parquet")
    faiss.write_index(index, index_path)
    df[["scheme_id", "scheme_name", "text"]].to_parquet(meta_path, index=False)

    print(f"\n✅ Index  → {index_path}")
    print(f"✅ Meta   → {meta_path}")
    print(f"✅ CSV    → {csv_path}")
    print(f"\nSet these env vars before running smoke_test.py:")
    print(f"  export FAISS_INDEX_PATH={index_path}")
    print(f"  export FAISS_META_PATH={meta_path}")


if __name__ == "__main__":
    main()
