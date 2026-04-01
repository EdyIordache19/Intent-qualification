import os
import sys
import io
import json
import re
import ast
import contextlib
import concurrent.futures
import argparse
import logging
import warnings

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# Must be set BEFORE importing Hugging Face libraries
# ==========================================
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ==========================================
# 2. THIRD-PARTY LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import groq
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub.utils import logging as hf_hub_logging
from transformers import logging as transformers_logging

# ==========================================
# 3. GLOBAL LOGGING & WARNING SUPPRESSION
# ==========================================
# Configure the root logger for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Silence overly verbose third-party loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_hub_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# Suppress expected deprecation and unauthenticated warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

def clean_json(raw_output: str) -> str:
    logging.info("Cleaning the output")
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output[7:]
    if cleaned_output.startswith("```"):
        cleaned_output = cleaned_output[3:]
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-3]

    return cleaned_output

def clean_query(raw_query: str) -> str:
    logging.info(f"Cleaning the user query")
    if not raw_query:
        return ""
    # Remove extra whitespaces and newline characters
    cleaned = re.sub(r'\s+', ' ', raw_query).strip()

    return cleaned

class BaseLLM:
    def __init__(self, model: str = "llama3", mode: str = "local"):
        self.model = model
        self.mode = mode

        if self.mode == "cloud":
            self.groq_client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def run_local(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = ollama.chat(
                model = self.model,
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options = {
                        "temperature": 0.0
                    },
                    format="json"
                )

            return response["message"]["content"]
        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                logging.error(f"Local model '{self.model}' not found! Please run: `ollama pull {self.model}` in your terminal.")
                sys.exit(1)
            else:
                logging.error(f"Ollama Error: {e}")
                sys.exit(1)
        except ConnectionError:
            logging.error("Could not initialize Ollama")
            sys.exit(1)

    def run_cloud(self, system_prompt: str, user_prompt: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            return chat_completion.choices[0].message.content
        except groq.AuthenticationError:
            logging.error("Invalid GROQ_API_KEY. Please check your environment variables.")
            sys.exit(1)
        except groq.RateLimitError:
            logging.error("Groq Rate Limit exceeded (Out of tokens or requests). Please wait and try again.")
            sys.exit(1)
        except groq.APIConnectionError:
            logging.error("Network error: Could not connect to the Groq API.")
            sys.exit(1)

    def run_prompt(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.mode == "local":
                result = self.run_local(system_prompt, user_prompt)
            else:
                result = self.run_cloud(system_prompt, user_prompt)

            return result
        except Exception as e:
            logging.error(f"LLM Execution Error: {e}")
            sys.exit(1)

class QueryParser(BaseLLM):
    def __init__(self, model: str = "llama3", mode: str = "local"):
        logging.info(f"Initializing the QueryParser with the {model} model")
        super().__init__(model=model, mode=mode)

        self.system_prompt = """
            You are a highly precise data extraction API. Your ONLY job is to extract search constraints from a user query and format them into a strict JSON object.

            CRITICAL RULES FOR COMPARATORS:
            - "over", "more than", "exceeding", "greater than", "bigger than", "larger than", "above", "at least", "+" -> map to `min_revenue` or `min_employees`.
            - "under", "less than", "fewer than", "smaller than", "below", "at most", "-" -> map to `max_revenue` or `max_employees`.
            - "after", "since" -> map to `year_founded_after`.
            - "prior to", "before" -> map to `year_founded_before`.
            - "not from", "outside of", "excluding", "not in" -> map to `exclude_location`.

            INSTRUCTIONS:
            1. ONLY extract a constraint if explicitly stated. Do not guess.
            2. If a constraint is not mentioned, its value MUST be `null` or `[]`.
            3. You MUST start your response with a `_reasoning` key, briefly explaining how you map the numbers based on the comparators.
            4. Start your response immediately with `{`. No markdown formatting outside the JSON.
            5. Convert abbreviations like "M" (millions) or "B" (billions) to full integers (e.g., 10M -> 10000000, 2B -> 2000000000).
            6. For `location` and `exclude_location`: If the user mentions a country, you MUST convert it to its 2-letter lowercase ISO country code (e.g., "Spain" -> "es", "United States" -> "us", "Brazil" -> "br"). If it is a city or region, leave it as is.

            Use EXACTLY this JSON structure:
            {
                "_reasoning": "Brief explanation of how you mapped the comparators to the keys.",
                "hard_filters": {
                    "location": null, // MUST be a single string or null.
                    "exclude_location": null, // MUST be a single string or null. For negative matches.
                    "min_revenue": null,
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

    # Get basic JSON with None values
    def get_fallback_json(self) -> dict:
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

    def sanitize_json(self, parsed_json: dict) -> dict:
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

        for loc_key in ["location", "exclude_location"]:
            loc_val = hard_filters.get(loc_key)
            if isinstance(loc_val, list):
                if len(loc_val) > 0:
                    hard_filters[loc_key] = loc_val[0]
                else:
                    hard_filters[loc_key] = None
            elif isinstance(loc_val, dict):
                logging.warning(f"LLM made a dictionary for {loc_key}. Clearing it.")
                hard_filters[loc_key] = None

        parsed_json["hard_filters"] = hard_filters
        return parsed_json

    def extract_json_from_query(self, raw_query: str) -> dict:
        logging.info(f"Calling {self.model} model for extracting JSON from query")
        cleaned_query = clean_query(raw_query)

        prompt = f'Query: "{cleaned_query}"\nOutput:'

        # Extract the raw output and clean from llm output
        raw_output = self.run_prompt(self.system_prompt, prompt)
        cleaned_output = clean_json(raw_output)

        # Sanitize and clean JSON
        # Try to parse the output as JSON
        try:
            parsed_json = json.loads(cleaned_output.strip())
            final_json = self.sanitize_json(parsed_json)
            return final_json
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON. Raw output was: \n{raw_output}")
            return self.get_fallback_json()

    def print_json(self, json_query: dict):
        print(json.dumps(json_query, indent = 2))

class CompaniesFilter:
    def __init__(self, df):
        logging.info("Initializing CompaniesFilter")
        self.df = df

    # Only drop if the company has specific data AND it's less/more than minimum/maximum
    # If the company's data is None, we keep it (Missing data robustness)
    def apply_filters(self, hard_filters: dict) -> pd.DataFrame:
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


        # Filter by Location (Basic string/dict matching)
        if hard_filters.get("location") is not None:
            target_loc = str(hard_filters["location"]).lower()

            def match_location(address_val):
                if pd.isna(address_val): return True # Keep missing data
                if isinstance(address_val, dict):
                    # Check if the exact target equals the country code, OR is inside any other value (like town/region)
                    country_match = str(address_val.get("country_code", "")).lower() == target_loc
                    other_match = any(target_loc in str(v).lower() for k, v in address_val.items() if k != "country_code")
                    return country_match or other_match
                return target_loc in str(address_val).lower()

            filtered_df = filtered_df[filtered_df['address'].apply(match_location)]

        # Filter by Negative Location (Exclude)
        if hard_filters.get("exclude_location") is not None:
            exclude_loc = str(hard_filters["exclude_location"]).lower()

            def exclude_match_location(address_val):
                if pd.isna(address_val): return True # Keep missing data
                if isinstance(address_val, dict):
                    # If the exact target equals the country code, we REJECT it
                    if str(address_val.get("country_code", "")).lower() == exclude_loc:
                        return False
                    # If the target is found in the town/region, we REJECT it
                    if any(exclude_loc in str(v).lower() for k, v in address_val.items() if k != "country_code"):
                        return False
                    return True
                return exclude_loc not in str(address_val).lower()

            filtered_df = filtered_df[filtered_df['address'].apply(exclude_match_location)]

        return filtered_df

class Searcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.model = SentenceTransformer(model_name)

    def prepare_company_text(self, company: pd.Series) -> str:
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

    def rank_companies(self, companies_df: pd.DataFrame, query, top_k) -> pd.DataFrame:
        logging.info("Ranking companies based on cosine similarity")
        if companies_df.empty:
            logging.warning("No companies left to rank")
            return companies_df

        if os.path.isfile("companies_embeddings.npy"):
            logging.info(f"companies_embeddings.npy found, extracting embeddings for {len(companies_df)} companies...")
            precomputed_embeddings = np.load("companies_embeddings.npy")
            companies_embeddings = precomputed_embeddings[companies_df.index.tolist()]

        else:
            logging.info(f"companies_embeddings.npy not found, preparing text and generating embeddings for {len(companies_df)} companies...")
            company_attributes = companies_df.apply(self.prepare_company_text, axis = 1).tolist()
            companies_embeddings = self.model.encode(company_attributes, show_progress_bar = False)

        query_embedding = self.model.encode([query], show_progress_bar = False)

        similarities = cosine_similarity(query_embedding, companies_embeddings)[0]

        ranked_companies = companies_df.copy()
        ranked_companies["score"] = similarities

        ranked_companies = ranked_companies.sort_values(by="score", ascending = False)

        return ranked_companies.head(top_k)

class IntentValidator(BaseLLM):
    def __init__(self, model: str = "llama3", mode: str = "local"):
        logging.info(f"Initializing IntentValidator with {model} model")
        super().__init__(model=model, mode=mode)

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

    def validate_company(self, query: str, company: pd.Series) -> dict:
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

        raw_output = self.run_prompt(self.system_prompt, prompt)
        cleaned_output = clean_json(raw_output)

        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse validation JSON for {company_data['name']}")
            return {"is_match": False, "confidence": "low", "reasoning": "Parsing failed."}

    def validate_and_filter_companies(self, companies_df: pd.DataFrame, query: str) -> pd.DataFrame:
        logging.info("Extracting correct companies")
        good_companies = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures_to_row = {
                executor.submit(self.validate_company, query, company): company for _, company in companies_df.iterrows()
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


class SearchEngine:
    def __init__(self, companies_df: pd.DataFrame, mode: str = "local", model_name: str = "llama3"):
        logging.info(f"Initializing SearchEngine | Mode: {mode.upper()} | Model: {model_name}")

        self.companies_df = companies_df
        self.parser = QueryParser(model=model_name, mode=mode)
        self.filter = CompaniesFilter(companies_df)
        self.searcher = Searcher()
        self.validator = IntentValidator(model=model_name, mode=mode)

    def run(self, query, top_k = 20):
        logging.info("Running the search engine")

        # Parse query
        json_query = self.parser.extract_json_from_query(query)
        print("\n--- Extracted Search Constraints ---")
        self.parser.print_json(json_query)
        print("------------------------------------\n")

        # Filter based on JSON query
        filtered_companies = self.filter.apply_filters(json_query["hard_filters"])
        logging.info(f"Reduced dataset from {len(self.companies_df)} to {len(filtered_companies)} companies.")
        # filtered_companies.to_json('filtered_companies.jsonl', orient='records', lines=True)

        # Rank top k companies based on embeddings
        ranked_companies = self.searcher.rank_companies(filtered_companies, query, top_k)
        # print(ranked_companies)

        # Validate companies
        validated_companies = self.validator.validate_and_filter_companies(ranked_companies, query)
        # validated_companies.to_json('validated_companies.jsonl', orient='records', lines=True)

        return validated_companies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Qualification Engine")

    parser.add_argument("--query", type=str, required=True, help="The search query to evaluate")
    parser.add_argument("--mode", type=str, choices=["local", "cloud"], default="local", help="Inference mode (local via Ollama or cloud via Groq)")
    parser.add_argument("--model", type=str, default="llama3", help="The LLM model to use (e.g., llama3, llama-3.3-70b-versatile)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of semantic matches to evaluate")

    args = parser.parse_args()
    companies_df = pd.read_json("companies.jsonl", lines=True)

    logging.info(f"Starting SearchEngine | Mode: {args.mode.upper()} | Model: {args.model}")
    search_engine = SearchEngine(companies_df, mode=args.mode, model_name=args.model)
    final_results = search_engine.run(args.query, top_k=args.top_k)

    # Print a nice summary at the end
    print("\n" + "="*50)
    print(f"FINAL QUALIFIED RESULTS: {len(final_results)} companies")
    print("="*50)

    if not final_results.empty:
        for idx, row in final_results.iterrows():
            print(f"\n[ {row['operational_name']} ] - Confidence: {row.get('confidence', 'N/A').upper()}")
            print(f"Why: {row.get('qualification_reasoning')}")
    else:
        print("\nNo companies passed the final intent qualification.")