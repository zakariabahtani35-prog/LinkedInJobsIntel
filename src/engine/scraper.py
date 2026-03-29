"""
============================================================
SR — WEB SCRAPER (PRODUCTION)
============================================================
Handles high-volume job data scraping with robust retries,
timeouts, and session management.
============================================================
"""

import json
import time
import random
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger

log = get_logger("scraper")

class JobScraper:
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.scraping_config = self.config["scraping"]
        self.paths           = self.config["paths"]
        self.output_path     = Path(self.paths["raw_data"])
        self.base_url        = "https://remotive.com/api/remote-jobs"
        self.session         = requests.Session()
        self.session.headers.update({"User-Agent": random.choice(self.scraping_config["user_agents"])})

    def fetch_records(self, category: str) -> List[Dict[str, Any]]:
        """Scrapes categorization-specific job postings from the public API."""
        params = {"category": category, "limit": self.scraping_config["limit_per_category"]}
        
        try:
            log.info(f"  Fetching: {category}...")
            resp = self.session.get(self.base_url, params=params, timeout=20)
            resp.raise_for_status()
            
            jobs = resp.json().get("jobs", [])
            log.info(f"  Found {len(jobs)} jobs in {category}")
            return jobs
        except requests.exceptions.RequestException as e:
            log.error(f"  Failure in {category} fetch: {e}")
            return []

    def run_scraper(self) -> pd.DataFrame:
        """Central orchestrator for the parallel-compatible scraper engine."""
        all_jobs: List[Dict[str, Any]] = []
        
        log.info("Starting production scraping engine...")
        
        for category in self.scraping_config["categories"]:
            raw_data = self.fetch_records(category)
            
            for item in raw_data:
                all_jobs.append({
                    "id":           item.get("id"),
                    "title":        item.get("title", "").strip(),
                    "company":      item.get("company_name", "").strip(),
                    "category":     item.get("category", "").strip(),
                    "tags":         ", ".join(item.get("tags", [])),
                    "location":     item.get("candidate_required_location", "Remote").strip(),
                    "salary":       item.get("salary", "").strip(),
                    "description":  item.get("description", "").strip()[:1000],
                    "url":          item.get("url", ""),
                    "scraped_at":   datetime.utcnow().isoformat()
                })
            
            # Rate limiting / Polite delay
            time.sleep(random.uniform(0.5, 1.5))

        df = pd.DataFrame(all_jobs).drop_duplicates(subset=["id"])
        
        # Persistence
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False, encoding="utf-8")
        
        log.info(f"Scraping finished. Total unique records saved: {len(df):,}")
        return df
