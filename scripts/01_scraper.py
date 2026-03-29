"""
============================================================
LinkedIn Jobs Intelligence System
STEP 1 — DATA COLLECTION: Scraper Wrapper (PRODUCTION)
============================================================
Usage:
    python scripts/01_scraper.py
============================================================
"""

from src.engine.scraper import JobScraper
from src.utils.logger import get_logger

log = get_logger("script_01")

def main():
    try:
        log.info("▶️  Executing Script 01: Data Collection")
        scraper_engine = JobScraper()
        scraper_engine.run_scraper()
        log.info("✅ Script 01: Successful Termination")
    except Exception as e:
        log.error(f"❌ Script 01: Critical Failure: {e}")

if __name__ == "__main__":
    main()
