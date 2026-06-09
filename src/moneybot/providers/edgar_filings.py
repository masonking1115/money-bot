"""SEC EDGAR filings provider (free, phase-1).

Resolves a ticker to a CIK via the injected cik_map, fetches the company's
recent-submissions JSON from data.sec.gov, and normalizes it into Filing models.
"""

from __future__ import annotations

from datetime import date

import httpx

from moneybot.models import Filing

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


class EdgarFilingsProvider:
    def __init__(self, cik_map: dict[str, str], user_agent: str) -> None:
        self.cik_map = cik_map
        self.user_agent = user_agent

    def _fetch_submissions(self, cik10: str) -> dict:
        # Network seam — patched in tests so no request is made.
        resp = httpx.get(
            SUBMISSIONS_URL.format(cik10=cik10),
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_recent_filings(
        self,
        ticker: str,
        types: list[str] | None = None,
        since: date | None = None,
        as_of: date | None = None,
    ) -> list[Filing]:
        if ticker not in self.cik_map:
            raise ValueError(f"no CIK configured for {ticker} (add it to universe.yaml)")

        cik_raw = str(self.cik_map[ticker])
        cik10 = cik_raw.zfill(10)
        cik_digits = cik_raw.lstrip("0") or "0"

        data = self._fetch_submissions(cik10)
        recent = data.get("filings", {}).get("recent", {})
        # Phase-1: only filings.recent (newest ~1000) is read; older
        # filings.files batches are not paginated yet.
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accs = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])

        out: list[Filing] = []
        for form, filed_str, acc, doc in zip(forms, dates, accs, docs):
            filed = date.fromisoformat(filed_str)
            if types is not None and form not in types:
                continue
            if since is not None and filed < since:
                continue
            if as_of is not None and filed > as_of:
                continue
            url = ARCHIVES_URL.format(
                cik=cik_digits, acc_nodash=acc.replace("-", ""), doc=doc
            )
            out.append(
                Filing(
                    ticker=ticker,
                    form_type=form,
                    filed_at=filed,
                    accession_no=acc,
                    url=url,
                )
            )
        out.sort(key=lambda f: f.filed_at)
        return out
