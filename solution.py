import ollama
import json
import re
import numpy as np
import pandas as pd

class QueryParser:
    def __init__(self, model="llama3"):
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
        """Pre-processing: Clean the user's raw query."""
        if not raw_query:
            return ""
        # Remove extra whitespaces and newline characters
        cleaned = re.sub(r'\s+', ' ', raw_query).strip()

        return cleaned

    # Clean initial output from Llama3
    def clean_output(self, raw_output):
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
        if not parsed_json or "hard_filters" not in parsed_json:
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
                    print(f"[Warning] Invalid type for {filter}: {val}. Defaulting to None.")
                    hard_filters[filter] = None

        # Logic errors
        if hard_filters.get("min_employees") and hard_filters.get("max_employees"):
            if hard_filters["min_employees"] > hard_filters["max_employees"]:
                print("[Warning] Logical error: min_employees > max_employees. Clearing both.")
                hard_filters["min_employees"] = None
                hard_filters["max_employees"] = None

        if hard_filters.get("min_revenue") and hard_filters.get("max_revenue"):
            if hard_filters["min_revenue"] > hard_filters["max_revenue"]:
                print("[Warning] Logical error: min_revenue > max_revenue. Clearing both.")
                hard_filters["min_revenue"] = None
                hard_filters["max_revenue"] = None

        if hard_filters.get("year_founded_after") and hard_filters.get("year_founded_before"):
            if hard_filters["year_founded_after"] > hard_filters["year_founded_before"]:
                print("[Warning] Logical error: founded_after > founded_before. Clearing both.")
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
        cleaned_query = self.clean_query(raw_query)

        response = ollama.chat(
            model = "llama3",
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
            print(f"Failed to parse JSON. Raw output was: {raw_output}")
            return self.get_fallback_json()

    def print_json(self, json_query):
        print(json.dumps(json_query, indent = 2))

if __name__ == "__main__":
    parser = QueryParser()

    companies = pd.read_json("companies.jsonl", lines=True)
    query = input("Enter a prompt to search companies: \n")

    json_query = parser.extract_json_from_query(query)
    parser.print_json(json_query)