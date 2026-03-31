import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer

companies_df = pd.read_json("companies.jsonl", lines=True)

def prepare_company_text(company: pd.Series) -> str:
    attributes = []

    # Get description
    if pd.notna(company.get("description")):
        description = company["description"]
        attributes.append(str(description))

    # Get primary naics
    if pd.notna(company.get("primary_naics")):
        primary_naics_data = company["primary_naics"]
        primary_naics_dict = {}
        if isinstance(primary_naics_data, str):
            try:
                primary_naics_dict = ast.literal_eval(primary_naics_data)
            except (ValueError, SyntaxError):
                pass
        elif isinstance(primary_naics_data, dict):
            primary_naics_dict = primary_naics_data

        if primary_naics_dict and "label" in primary_naics_dict:
            attributes.append(f"Industry: {primary_naics_dict['label']}")

    # Get core offerings
    core_offerings = company.get("core_offerings")
    if isinstance(core_offerings, list) and len(core_offerings) > 0:
        core_offerings_str = ", ".join(core_offerings)

        attributes.append(f"Offering: {core_offerings_str}")

    # Get target markets
    target_markets = company.get("target_markets")
    if isinstance(target_markets, list) and len(target_markets) > 0:
        target_markets_str = ", ".join(target_markets)

        attributes.append(f"Target Markets: {target_markets_str}")

    return " | ".join(attributes)

def compute_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    company_attributes = companies_df.apply(prepare_company_text, axis = 1).tolist()

    companies_embeddings = model.encode(company_attributes, show_progress_bar = False)

    return companies_embeddings


companies_embeddings = compute_embeddings()
np.save("companies_embeddings.npy", companies_embeddings)
