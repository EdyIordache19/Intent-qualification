import os
# Disable Hugging Face progress bars and symlink warnings
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# CORRECTED: Import the logging module directly from huggingface_hub.utils
from huggingface_hub.utils import logging as hf_hub_logging
hf_hub_logging.set_verbosity_error()

# Silence the transformers logger
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import ollama
import json
import re
import numpy as np
import pandas as pd
import ast
import concurrent.futures
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import io
import contextlib
import sys

class QueryParser:
    def __init__(self, model="llama3"):
        logging.info(f"Initializing the QueryParser with the {model} model")
        self.model = model
        self.system_prompt = """
            You are a highly precise data extraction API. Your ONLY job is to extract search constraints from a user query and format them into a strict JSON object.

            CRITICAL RULES FOR COMPARATORS:
            - "over", "more than", "exceeding", "greater than", "bigger than", "larger than", "above", "at least", "+" -> map to `min_revenue` or `min_employees`.
            - "under", "less than", "fewer than", "smaller than", "below", "at most", "-" -> map to `max_revenue` or `max_employees`.
            - "after", "since" -> map to `year_founded_after`.
            - "prior to", "before" -> map to `year_founded_before`.

            INSTRUCTIONS:
            1. ONLY extract a constraint if explicitly stated. Do not guess.
            2. If a constraint is not mentioned, its value MUST be `null` or `[]`.
            3. You MUST start your response with a `_reasoning` key, briefly explaining how you map the numbers based on the comparators.
            4. Start your response immediately with `{`. No markdown formatting outside the JSON.
            5. Convert abbreviations like "M" (millions) or "B" (billions) to full integers (e.g., 10M -> 10000000, 2B -> 2000000000).

            Use EXACTLY this JSON structure:
            {
            "_reasoning": "Brief explanation of how you mapped the comparators to the keys.",
            "hard_filters": {
                "location": null, // CRITICAL: MUST be a single string or null. NEVER a list [].
                "min_revenue": null, // CRITICAL: Integer only.
                "max_revenue": null,
                "min_employees": null,
                "max_employees": null,
                "year_founded_after": null,
                "year_founded_before": null,
                "is_public": null
            },
            "soft_filters": {
                "industry_or_vertical": [],
                "business_model": [],
                "core_offerings": []
            }
            }

            Example 1:
            Query: "Startups with less than 50 employees founded prior to 2015"
            Output:
            {
            "_reasoning": "'less than 50 employees' means max_employees is 50. 'prior to 2015' means year_founded_before is 2015.",
            "hard_filters": {
                "location": null, "min_revenue": null, "max_revenue": null, "min_employees": null, "max_employees": 50, "year_founded_after": null, "year_founded_before": 2015, "is_public": null
            },
            "soft_filters": { "industry_or_vertical": ["startups"], "business_model": [], "core_offerings": [] }
            }

            Example 2:
            Query: "Public software companies with more than 1,000 employees."
            Output:
            {
            "_reasoning": "'Public' means is_public is true. 'more than 1,000 employees' means min_employees is 1000.",
            "hard_filters": {
                "location": null, "min_revenue": null, "max_revenue": null, "min_employees": 1000, "max_employees": null, "year_founded_after": null, "year_founded_before": null, "is_public": true
            },
            "soft_filters": { "industry_or_vertical": ["software"], "business_model": [], "core_offerings": [] }
            }
        """
        # Critical rule for extracting only what is clear, not interpreting
        # Should use soft filtering instead of hard filtering for this reason:
        #        -> If value hard to interpret, use hard filter
        #        -> If value ambiguous, use soft filter later in cosine similarity

    def clean_query(self, raw_query):
        logging.info(f"Cleaning the query {raw_query}")
        if not raw_query:
            return ""
        # Remove extra whitespaces and newline characters
        cleaned = re.sub(r'\s+', ' ', raw_query).strip()

        return cleaned

    # Clean initial output from Llama3
    def clean_output(self, raw_output):
        logging.info("Cleaning the output")
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]

        return cleaned_output

    # Get basic JSON with None values
    def get_fallback_json(self):
        logging.info("Getting default JSON")
        return {
            "_reasoning": "Fallback used due to parsing failure.",
            "hard_filters": {
                "location": None, "min_revenue": None, "max_revenue": None,
                "min_employees": None, "max_employees": None,
                "year_founded_after": None, "year_founded_before": None, "is_public": None
            },
            "soft_filters": {
                "industry_or_vertical": [], "business_model": [], "core_offerings": []
            }
        }

    def sanitize_json(self, parsed_json):
        logging.info("Sanitizing JSON")
        if not parsed_json or "hard_filters" not in parsed_json:
            logging.warning("Hard filters missing from JSON")
            return self.get_fallback_json()

        hard_filters = parsed_json.get("hard_filters", {})

        # Type Enforcement
        filters_list = ["min_revenue", "max_revenue", "min_employees", "max_employees", "year_founded_after", "year_founded_before"]
        for filter in filters_list:
            val = hard_filters.get(filter)
            if val is not None:
                try:
                    hard_filters[filter] = int(val)
                except ValueError:
                    logging.warning(f"Invalid type for {filter}: {val}. Defaulting to None.")
                    hard_filters[filter] = None

        # Logic errors
        if hard_filters.get("min_employees") and hard_filters.get("max_employees"):
            if hard_filters["min_employees"] > hard_filters["max_employees"]:
                logging.warning("Logical error: min_employees > max_employees. Clearing both.")
                hard_filters["min_employees"] = None
                hard_filters["max_employees"] = None

        if hard_filters.get("min_revenue") and hard_filters.get("max_revenue"):
            if hard_filters["min_revenue"] > hard_filters["max_revenue"]:
                logging.warning("Logical error: min_revenue > max_revenue. Clearing both.")
                hard_filters["min_revenue"] = None
                hard_filters["max_revenue"] = None

        if hard_filters.get("year_founded_after") and hard_filters.get("year_founded_before"):
            if hard_filters["year_founded_after"] > hard_filters["year_founded_before"]:
                logging.warning("Logical error: founded_after > founded_before. Clearing both.")
                hard_filters["year_founded_after"] = None
                hard_filters["year_founded_before"] = None

        if isinstance(hard_filters.get("location"), list):
            if len(hard_filters["location"]) > 0:
                hard_filters["location"] = hard_filters["location"][0]
            else:
                hard_filters["location"] = None

        parsed_json["hard_filters"] = hard_filters
        return parsed_json


    def extract_json_from_query(self, raw_query):
        logging.info(f"Calling {self.model} model for extracting JSON from query")
        cleaned_query = self.clean_query(raw_query)

        response = ollama.chat(
            model = self.model,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f'Query: "{cleaned_query}"\nOutput:'}
            ],
            options = {
                "temperature": 0.0
            },
            format="json"
        )

        # Extract the raw output and clean from llm output
        raw_output = response["message"]["content"]
        cleaned_output = self.clean_output(raw_output)

        # Sanitize and clean JSON
        # Try to parse the output as JSON
        try:
            parsed_json = json.loads(cleaned_output.strip())
            final_json = self.sanitize_json(parsed_json)
            return final_json
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON. Raw output was: \n{raw_output}")
            return self.get_fallback_json()

    def print_json(self, json_query):
        print(json.dumps(json_query, indent = 2))

class CompaniesFilter:
    def __init__(self, df):
        logging.info("Initializing CompaniesFilter")
        self.df = df

    # Only drop if the company has specific data AND it's less/more than minimum/maximum
    # If the company's data is None, we keep it (Missing data robustness)
    def apply_filters(self, hard_filters):
        logging.info("Applying hard filters")
        filtered_df = self.df.copy()

        # Filter by Minimum Revenue
        if hard_filters.get("min_revenue") is not None:
            filtered_df = filtered_df[
                filtered_df['revenue'].isna() | (filtered_df['revenue'] >= hard_filters["min_revenue"])
            ]

        # Filter by Maximum
        if hard_filters.get("max_employees") is not None:
            filtered_df = filtered_df[
                filtered_df['employee_count'].isna() | (filtered_df['employee_count'] <= hard_filters["max_employees"])
            ]

        # Filter by Minimum Employees
        if hard_filters.get("min_employees") is not None:
            filtered_df = filtered_df[
                filtered_df['employee_count'].isna() | (filtered_df['employee_count'] >= hard_filters["min_employees"])
            ]

        # Filter by Public/Private Status
        if hard_filters.get("is_public") is not None:
            filtered_df = filtered_df[
                filtered_df['is_public'].isna() | (filtered_df['is_public'] == hard_filters["is_public"])
            ]

        # Filter by Year Founded After
        if hard_filters.get("year_founded_after") is not None:
            filtered_df = filtered_df[
                filtered_df['year_founded'].isna() | (filtered_df['year_founded'] >= hard_filters["year_founded_after"])
            ]

        # Filter by Year Founded Before
        if hard_filters.get("year_founded_before") is not None:
            filtered_df = filtered_df[
                filtered_df['year_founded'].isna() | (filtered_df['year_founded'] <= hard_filters["year_founded_before"])
            ]


        # Filter by Location
        if hard_filters.get("location") is not None:
            target_loc = str(hard_filters["location"]).lower()
            # Keep if missing, OR if the target location string is inside the company's address string
            filtered_df = filtered_df[
                filtered_df['address'].isna() |
                filtered_df['address'].astype(str).str.lower().str.contains(target_loc, na=False)
            ]

        return filtered_df

class Searcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = SentenceTransformer(model_name)

    def prepare_company_text(self, company):
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
                primary_naics_dict = ast.literal_eval(primary_naics_data)
            elif isinstance(primary_naics_data, dict):
                primary_naics_dict = primary_naics_data

            attributes.append(f"Industry: {primary_naics_dict["label"]}")

        # Get core offerings
        if len(company.get("core_offerings")) > 0:
            core_offerings = company["core_offerings"]
            core_offerings = ", ".join(core_offerings)

            attributes.append(f"Offering: {core_offerings}")

        # Get target markets
        if len(company.get("target_markets")) > 0:
            target_markets = company["target_markets"]
            target_markets = ", ".join(target_markets)

            attributes.append(target_markets)

        return " | ".join(attributes)

    def rank_companies(self, companies, query, top_k):
        logging.info("Ranking companies based on cosine similarity")
        if companies.empty:
            logging.warning("No companies left to rank")
            return companies

        logging.info(f"Preparing text and generating embeddings for {len(companies)} companies...")
        company_attributes = companies.apply(self.prepare_company_text, axis = 1).tolist()

        companies_embeddings = self.model.encode(company_attributes, show_progress_bar = False)
        query_embedding = self.model.encode([query], show_progress_bar = False)

        similarities = cosine_similarity(query_embedding, companies_embeddings)[0]

        ranked_companies = companies.copy()
        ranked_companies["score"] = similarities

        ranked_companies = ranked_companies.sort_values(by="score", ascending = False)

        return ranked_companies.head(top_k)

class IntentValidator:
    def __init__(self, model = "llama3"):
        logging.info(f"Initializing IntentValidator with {model} model")
        self.model = model
        self.system_prompt = """
        You are an expert business analyst and qualification engine. Your job is to determine if a specific company TRULY matches a user's search intent.

        You will be provided with:
        1. The User's Query.
        2. A JSON object containing the Company's profile.

        CRITICAL INSTRUCTIONS:
            - Evaluate the company's core identity, not just keyword matches.
            - PAY ATTENTION TO THE SUPPLY CHAIN: If the user asks for a "Cosmetics Company" (meaning they make makeup/skincare), a company that makes "Cosmetics Packaging" (plastic bottles) is a strict NO.
            - If the user asks for "Software", an IT consulting firm is a NO.
            - You must critically analyze if the company *is* the target, or if they just *serve* the target. If they only serve the target industry, output "is_match": false.

        You MUST output a strict JSON object with EXACTLY this schema:
        {
            "is_match": true or false,
            "confidence": "high", "medium", or "low",
            "reasoning": "1-2 sentences explaining exactly why they match or fail the specific intent of the query."
        }

        Start your response immediately with `{`. No markdown formatting. No conversational text.
        """

    def clean_json(self, raw_output):
        logging.info("Cleaning the JSON")
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]

        return cleaned_output

    def validate_company(self, query, company):
        company_name = company.get('operational_name', 'Unknown')
        logging.info(f"Validating intent for: {company_name}")
        company_data = {
            "name": company.get("operational_name"),
            "description": company.get("description"),
            "core_offerings": company.get("core_offerings"),
            "business_model": company.get("business_model"),
            "industry": company.get("primary_naics", {}).get("label") if isinstance(company.get("primary_naics"), dict) else None
        }

        prompt = f"User Query: '{query}'\n\nCompany Profile:\n{json.dumps(company_data, indent=2)}\n\nOutput strict JSON:"

        response = ollama.chat(
            model = self.model,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options = {
                "temperature": 0.0
            },
            format="json"
        )

        raw_output = response["message"]["content"]
        cleaned_output = self.clean_json(raw_output)

        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse validation JSON for {company_data['name']}")
            return {"is_match": False, "confidence": "low", "reasoning": "Parsing failed."}

    def validate_and_filter_companies(self, companies, query):
        logging.info("Extracting correct companies")
        good_companies = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures_to_row = {
                executor.submit(self.validate_company, query, company): company for _, company in companies.iterrows()
            }

            for future in concurrent.futures.as_completed(futures_to_row):
                row = futures_to_row[future]
                company_name = row.get("operational_name", "Unknown")

                try:
                    validation_result = future.result()

                    if validation_result.get("is_match") is True:
                        logging.info(f"MATCH: {company_name} | Confidence: {validation_result.get('confidence')}")
                        row_dict = row.to_dict()
                        row_dict["qualification_reasoning"] = validation_result.get("reasoning")
                        row_dict["confidence"] = validation_result.get("confidence")
                        good_companies.append(row_dict)
                    else:
                        logging.info(f"REJECTED: {company_name} | Reason: {validation_result.get('reasoning')}")

                except Exception as e:
                    logging.error(f"Validation failed for {company_name}. Error: {e}")

        logging.info(f"Intent Validation complete. {len(good_companies)} companies qualified.")
        return pd.DataFrame(good_companies)

        # for index, company in companies.iterrows():
        #     validation_result = self.validate_company(query, company)

        #     if validation_result.get("is_match") is True:
        #         company_dict = company.to_dict()

        #         company_dict["reasoning"] = validation_result.get("reasoning")
        #         company_dict["confidence"] = validation_result.get("confidence")

        #         good_companies.append(company_dict)


class SearchEngine:
    def __init__(self, companies):
        logging.info("Initializing the SearchEngine")
        self.parser = QueryParser()
        self.filter = CompaniesFilter(companies)
        self.searcher = Searcher()
        self.validator = IntentValidator()

    def run(self, query, top_k = 20):
        logging.info("Running the search engine")

        # Parse query
        json_query = self.parser.extract_json_from_query(query)
        self.parser.print_json(json_query)

        # Filter based on JSON query
        filtered_companies = self.filter.apply_filters(json_query["hard_filters"])
        print(f"Reduced dataset from {len(companies)} to {len(filtered_companies)} companies.")
        filtered_companies.to_json('filtered_companies.json', orient='records', lines=True)

        # Rank top k companies based on embeddings
        ranked_companies = self.searcher.rank_companies(filtered_companies, query, top_k)
        print(ranked_companies)

        # Validate companies
        validated_companies = self.validator.validate_and_filter_companies(ranked_companies, query)
        validated_companies.to_json('validated_companies.json', orient='records', lines=True)

        return validated_companies

if __name__ == "__main__":
    companies = pd.read_json("companies.jsonl", lines=True)
    query = input("Please enter a query: \n")

    search_engine = SearchEngine(companies)
    search_engine.run(query, 10)