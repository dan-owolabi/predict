import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "uefa_stats" / "raw"
OUT_PATH = BASE_DIR / "uefa_stats" / "uefa_match_stats.csv"
SUMMARY_PATH = BASE_DIR / "uefa_stats" / "summary.json"
FIXTURE_PATHS = {
    "UCL": BASE_DIR / "openfootball-europe" / "CL_all.csv",
    "UEL": BASE_DIR / "openfootball-europe" / "EL_all.csv",
}
PDF_BLOCKS = {
    "UCL": {
        2025: [(2041985, 2042065), (2044848, 2044859)],
        2026: [(2047761, 2047776), (2048047, 2048054)],
    },
    "UEL": {},
}
PDF_BASE = "https://www.uefa.com/newsfiles/{comp}/{year}/{match_id}_TS.pdf"

TEXT_REPLACEMENTS = {
    "Oﬀ": "Off",
    "ﬁ": "fi",
    "ﬀ": "ff",
    "Â»": "",
    "Total attem pts": "Total attempts",
    "Fouls com m itted": "Fouls committed",
    "Fouls suﬀered": "Fouls suffered",
    "Passes attem pted": "Passes attempted",
    "Passes com pleted": "Passes completed",
    "Pass com pletion rate": "Pass completion rate",
    "W oodwork": "Woodwork",
    "S aves": "Saves",
}

STAT_PATTERNS = {
    "home_shots": r"Total\s*attempts\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_sot": r"On\s*target\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_corners": r"Corners\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_ycards": r"Yellow\s*cards\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_rcards": r"Red\s*cards\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_fouls": r"Fouls\s*committed\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)",
    "home_possession": r"Ball\s*possession\s+\d+%\s+\d+%\s+\d+%\s+\d+%\s+(\d+)%\s+(\d+)%",
}
MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def canonical_name(name: str) -> str:
    text = str(name or "").strip().lower()
    text = (
        text.replace("ø", "o").replace("ö", "o").replace("ü", "u")
        .replace("ä", "a").replace("á", "a").replace("à", "a")
        .replace("ç", "c").replace("é", "e").replace("è", "e")
        .replace("í", "i").replace("ó", "o").replace("ñ", "n")
        .replace("å", "a").replace("ã", "a").replace("ş", "s")
    )
    text = re.sub(r"\b(fc|cf|ac|as|ssc|fk|sk|bsc|afc|ogc|rsc|pfc|gnk|vfb|club)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_fixtures():
    frames = []
    for comp, path in FIXTURE_PATHS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["Date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["FTHG"] = pd.to_numeric(df["home_goals"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["away_goals"], errors="coerce")
        df["HomeTeam"] = df["home_team"].astype(str)
        df["AwayTeam"] = df["away_team"].astype(str)
        df["comp"] = comp
        df["home_key"] = df["HomeTeam"].map(canonical_name)
        df["away_key"] = df["AwayTeam"].map(canonical_name)
        frames.append(df[["Date", "FTHG", "FTAG", "HomeTeam", "AwayTeam", "home_key", "away_key", "comp", "season_start"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_pdf(comp: str, year: int, match_id: int) -> Path | None:
    out_path = RAW_DIR / f"{comp}_{year}_{match_id}.pdf"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return out_path
    url = PDF_BASE.format(comp=comp, year=year, match_id=match_id)
    result = subprocess.run(
        ["curl.exe", "-L", "--connect-timeout", "3", "--max-time", "20", "-s", url, "-o", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
        if out_path.exists():
            out_path.unlink()
        return None
    return out_path


def normalize_pdf_text(text: str) -> str:
    for src, dst in TEXT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"(?<=[A-Za-zÀ-ÿ])\s+(?=[a-zà-ÿ])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return normalize_pdf_text(text)


def extract_date(text: str):
    long_matches = re.findall(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})", text)
    matches = long_matches or re.findall(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})", text)
    if not matches:
        return None
    day, month_txt, year = matches[0]
    month = MONTHS.get(month_txt.lower())
    if month is None:
        return None
    return datetime(int(year), month, int(day)).date()


def extract_score(text: str):
    matches = re.findall(r"(\d+)\s*-\s*(\d+)", text)
    if not matches:
        return None, None
    home_goals, away_goals = matches[0]
    return int(home_goals), int(away_goals)


def extract_stats(text: str):
    out = {}
    for key, pattern in STAT_PATTERNS.items():
        m = re.search(pattern, text)
        if not m:
            out[key] = None
            out[key.replace("home_", "away_", 1)] = None
            continue
        out[key] = float(m.group(1))
        out[key.replace("home_", "away_", 1)] = float(m.group(2))
    out["total_shots"] = _sum(out.get("home_shots"), out.get("away_shots"))
    out["total_sot"] = _sum(out.get("home_sot"), out.get("away_sot"))
    out["total_corners"] = _sum(out.get("home_corners"), out.get("away_corners"))
    out["total_ycards"] = _sum(out.get("home_ycards"), out.get("away_ycards"))
    out["total_rcards"] = _sum(out.get("home_rcards"), out.get("away_rcards"))
    out["home_bookings"] = _sum(out.get("home_ycards"), out.get("home_rcards"))
    out["away_bookings"] = _sum(out.get("away_ycards"), out.get("away_rcards"))
    out["total_bookings"] = _sum(out.get("home_bookings"), out.get("away_bookings"))
    out["total_fouls"] = _sum(out.get("home_fouls"), out.get("away_fouls"))
    return out


def _sum(a, b):
    if a is None or b is None:
        return None
    return float(a) + float(b)


def fixture_token_score(text: str, fixture_row: pd.Series) -> int:
    score = 0
    for side in ["home_key", "away_key"]:
        tokens = [tok for tok in str(fixture_row[side]).split() if len(tok) >= 3]
        score += sum(tok in text for tok in tokens)
    return score


def match_fixture(fixtures: pd.DataFrame, comp: str, match_date, home_goals: int, away_goals: int, text: str):
    candidates = fixtures[
        (fixtures["comp"] == comp)
        & (fixtures["Date"] == match_date)
        & (fixtures["FTHG"] == home_goals)
        & (fixtures["FTAG"] == away_goals)
    ].copy()
    if candidates.empty:
        return None
    if len(candidates) == 1:
        return candidates.iloc[0]
    norm_text = canonical_name(text)
    candidates["token_score"] = candidates.apply(lambda row: fixture_token_score(norm_text, row), axis=1)
    candidates = candidates.sort_values(["token_score", "HomeTeam", "AwayTeam"], ascending=[False, True, True])
    top = candidates.iloc[0]
    if len(candidates) > 1 and int(top["token_score"]) == int(candidates.iloc[1]["token_score"]):
        return None
    return top


def parse_pdf_record(fixtures: pd.DataFrame, comp: str, year: int, match_id: int, path: Path):
    text = parse_pdf_text(path)
    match_date = extract_date(text)
    home_goals, away_goals = extract_score(text)
    if match_date is None or home_goals is None:
        return None
    fixture = match_fixture(fixtures, comp, match_date, home_goals, away_goals, text)
    if fixture is None:
        return None
    stats = extract_stats(text)
    return {
        "comp": comp,
        "pdf_year": year,
        "match_id": match_id,
        "Date": str(match_date),
        "HomeTeam": fixture["HomeTeam"],
        "AwayTeam": fixture["AwayTeam"],
        "FTHG": int(home_goals),
        "FTAG": int(away_goals),
        **stats,
    }


def all_pdf_targets():
    for comp, years in PDF_BLOCKS.items():
        for year, blocks in years.items():
            for start, end in blocks:
                for match_id in range(start, end + 1):
                    yield comp, year, match_id


def main():
    ensure_dirs()
    fixtures = load_fixtures()
    rows = []
    downloaded = 0
    matched = 0
    for comp, year, match_id in all_pdf_targets():
        path = download_pdf(comp, year, match_id)
        if path is None:
            continue
        downloaded += 1
        record = parse_pdf_record(fixtures, comp, year, match_id, path)
        if record is None:
            continue
        matched += 1
        rows.append(record)
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).drop_duplicates(subset=["comp", "Date", "HomeTeam", "AwayTeam"]).to_csv(OUT_PATH, index=False)
            SUMMARY_PATH.write_text(json.dumps({
                "downloaded_pdfs": downloaded,
                "matched_rows": matched,
                "final_rows_partial": len(rows),
            }, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows).drop_duplicates(subset=["comp", "Date", "HomeTeam", "AwayTeam"]).sort_values(["Date", "HomeTeam"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    summary = {
        "downloaded_pdfs": downloaded,
        "matched_rows": matched,
        "final_rows": int(len(df)),
        "by_comp": df.groupby("comp").size().to_dict() if not df.empty else {},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
