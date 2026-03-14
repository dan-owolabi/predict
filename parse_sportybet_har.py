import argparse
import base64
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse


OUT_DIR = Path("data") / "sportybet"
HAR_EXPORTS_DIR = OUT_DIR / "har_exports"


def _safe_json_text(content):
    text = content.get("text", "")
    if not text:
        return ""
    if content.get("encoding") == "base64":
        return base64.b64decode(text).decode("utf-8", errors="replace")
    return text


def _format_event_time(ts_ms):
    if not ts_ms:
        return None
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.isoformat()


def _request_query_value(url, key):
    return parse_qs(urlparse(url).query).get(key, [None])[0]


def _selection_url(meta, row):
    params = {
        "sportId": meta.get("sport_id"),
        "productId": row.get("product_id"),
        "eventId": meta.get("event_id"),
        "marketId": row.get("market_id"),
        "outcomeId": row.get("outcome_id"),
        "selected": "0",
        "odds": row.get("odds"),
        "marketGroupsName": row.get("market_group"),
    }
    if row.get("specifier"):
        params["specifier"] = row["specifier"]
    query = "&".join(f"{k}={v}" for k, v in params.items() if v not in (None, ""))
    return f"https://www.sportybet.com/ng/lite/preMatch/detail?fromUrl=null&{query}"


def _flatten_event_payload(payload, source):
    data = payload.get("data") or {}
    sport = data.get("sport") or {}
    category = sport.get("category") or {}
    tournament = category.get("tournament") or {}
    meta = {
        "event_id": data.get("eventId"),
        "game_id": data.get("gameId"),
        "sport_id": sport.get("id"),
        "sport": sport.get("name"),
        "country": category.get("name"),
        "league": tournament.get("name"),
        "home_team": data.get("homeTeamName"),
        "away_team": data.get("awayTeamName"),
        "event_time": _format_event_time(data.get("estimateStartTime")),
        "event_url": source.get("page_url") or source.get("request_url"),
        "source_file": str(source.get("har_path")),
        "request_url": source.get("request_url"),
        "match_status": data.get("matchStatus"),
        "booking_status": data.get("bookingStatus"),
        "event_source": data.get("eventSource"),
    }

    rows = []
    groups = set()
    for market in data.get("markets") or []:
        market_title = market.get("desc") or market.get("name")
        market_group = market.get("group")
        groups.add(market_group)
        for outcome in market.get("outcomes") or []:
            row = {
                **meta,
                "market_group": market_group,
                "market_title": market_title,
                "selection": outcome.get("desc"),
                "odds": outcome.get("odds"),
                "probability": outcome.get("probability"),
                "void_probability": outcome.get("voidProbability"),
                "market_id": market.get("id"),
                "market_name": market.get("name"),
                "market_desc": market.get("desc"),
                "market_title_raw": market.get("title"),
                "market_specifier": market.get("specifier"),
                "market_group_id": market.get("groupId"),
                "outcome_id": outcome.get("id"),
                "outcome_active": outcome.get("isActive"),
                "product_id": market.get("product"),
                "source_type": market.get("sourceType"),
                "market_status": market.get("status"),
                "market_guide": market.get("marketGuide"),
                "favourite": market.get("favourite"),
                "far_near_odds": market.get("farNearOdds"),
                "last_odds_change_time": market.get("lastOddsChangeTime"),
                "selection_url": None,
            }
            row["selection_url"] = _selection_url(meta, {
                "market_id": row["market_id"],
                "outcome_id": row["outcome_id"],
                "product_id": row["product_id"],
                "odds": row["odds"],
                "market_group": row["market_group"],
                "specifier": row["market_specifier"],
            })
            rows.append(row)

    return meta, [{"name": name} for name in sorted(g for g in groups if g)], rows


def _merge_event(existing, meta, groups, rows, source):
    if not existing:
        return {
            "meta": meta,
            "groups": groups,
            "markets": rows,
            "sources": [source],
        }

    merged_groups = {g.get("name"): g for g in existing.get("groups", [])}
    for group in groups:
        merged_groups[group.get("name")] = group

    seen = set()
    merged_rows = []
    for row in existing.get("markets", []) + rows:
        key = (
            row.get("event_id"),
            row.get("market_group"),
            row.get("market_id"),
            row.get("market_specifier"),
            row.get("outcome_id"),
            row.get("selection"),
            row.get("odds"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged_rows.append(row)

    existing["groups"] = [merged_groups[k] for k in sorted(merged_groups)]
    existing["markets"] = merged_rows
    existing.setdefault("sources", []).append(source)
    return existing


def _slug(text):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "export"


def _extract_cms_exports(har, har_path):
    exports = []
    for entry in har.get("log", {}).get("entries", []):
        request = entry.get("request") or {}
        response = entry.get("response") or {}
        url = request.get("url") or ""
        if "/cms/pages/export/" not in url:
            continue
        text = _safe_json_text((response.get("content") or {}))
        try:
            payload = json.loads(text)
        except Exception:
            continue
        name = url.rstrip("/").split("/")[-1]
        exports.append(
            {
                "name": name,
                "url": url,
                "har_file": str(har_path),
                "keys_count": len((payload or {}).get("keys") or {}),
                "payload": payload,
            }
        )
    return exports


def _page_url_map(har):
    pages = {}
    for page in har.get("log", {}).get("pages", []):
        pages[page.get("id")] = page.get("title")
    return pages


def parse_har_files(paths):
    events = {}
    cms_exports = []
    summary_rows = []

    for path in paths:
        har = json.load(open(path, encoding="utf-8"))
        page_map = _page_url_map(har)
        file_event_rows = 0
        file_event_count = 0

        for entry in har.get("log", {}).get("entries", []):
            request = entry.get("request") or {}
            response = entry.get("response") or {}
            url = request.get("url") or ""
            if "factsCenter/event" not in url:
                continue
            text = _safe_json_text((response.get("content") or {}))
            try:
                payload = json.loads(text)
            except Exception:
                continue
            data = payload.get("data") or {}
            event_id = data.get("eventId") or _request_query_value(url, "eventId")
            if not event_id:
                continue

            source = {
                "har_path": str(path),
                "request_url": url,
                "page_url": page_map.get(entry.get("pageref")),
            }
            meta, groups, rows = _flatten_event_payload(payload, source)
            events[event_id] = _merge_event(events.get(event_id), meta, groups, rows, source)
            file_event_rows += len(rows)
            file_event_count += 1

        cms = _extract_cms_exports(har, path)
        cms_exports.extend(cms)
        summary_rows.append(
            {
                "har_file": str(path),
                "event_payloads": file_event_count,
                "event_rows": file_event_rows,
                "cms_exports": len(cms),
                "page_titles": " | ".join([p.get("title", "") for p in har.get("log", {}).get("pages", [])]),
            }
        )

    return events, cms_exports, summary_rows


def write_outputs(events, cms_exports, summary_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HAR_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for event_id, payload in events.items():
        safe_id = event_id.replace(":", "_")
        json_path = OUT_DIR / f"{safe_id}.json"
        csv_path = OUT_DIR / f"{safe_id}.csv"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        rows = payload.get("markets", [])
        if rows:
            headers = list(rows[0].keys())
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            catalog_rows = []
            seen = set()
            for row in rows:
                key = (row.get("market_group"), row.get("market_title"))
                if key in seen:
                    continue
                seen.add(key)
                catalog_rows.append({
                    "event_id": row.get("event_id"),
                    "market_group": row.get("market_group"),
                    "market_title": row.get("market_title"),
                })

            catalog_csv_path = OUT_DIR / f"{safe_id}_market_catalog.csv"
            with open(catalog_csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["event_id", "market_group", "market_title"])
                writer.writeheader()
                writer.writerows(sorted(catalog_rows, key=lambda x: (x["market_group"] or "", x["market_title"] or "")))

            catalog_json_path = OUT_DIR / f"{safe_id}_market_catalog.json"
            with open(catalog_json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "event_id": event_id,
                        "market_groups": sorted({row["market_group"] for row in catalog_rows if row.get("market_group")}),
                        "market_titles_by_group": {
                            group: sorted({row["market_title"] for row in catalog_rows if row.get("market_group") == group})
                            for group in sorted({row["market_group"] for row in catalog_rows if row.get("market_group")})
                        },
                        "count_market_titles": len(catalog_rows),
                    },
                    fh,
                    indent=2,
                )

    for item in cms_exports:
        name = _slug(item["name"])
        path = HAR_EXPORTS_DIR / f"{name}.json"
        out = {
            "name": item["name"],
            "url": item["url"],
            "har_file": item["har_file"],
            "keys_count": item["keys_count"],
            "payload": item["payload"],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    summary_path = OUT_DIR / "har_ingest_summary.csv"
    if summary_rows:
        headers = list(summary_rows[0].keys())
        with open(summary_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(summary_rows)

    cms_summary_path = OUT_DIR / "har_exports_summary.csv"
    if cms_exports:
        rows = []
        for item in cms_exports:
            rows.append(
                {
                    "name": item["name"],
                    "url": item["url"],
                    "har_file": item["har_file"],
                    "keys_count": item["keys_count"],
                }
            )
        headers = list(rows[0].keys())
        with open(cms_summary_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Parse SportyBet HAR files into event market snapshots.")
    parser.add_argument("har_files", nargs="+", help="One or more HAR files to ingest")
    args = parser.parse_args()

    paths = [Path(p) for p in args.har_files]
    events, cms_exports, summary_rows = parse_har_files(paths)
    write_outputs(events, cms_exports, summary_rows)

    print(f"events={len(events)}")
    print(f"cms_exports={len(cms_exports)}")
    for event_id, payload in sorted(events.items()):
        print(f"{event_id} rows={len(payload.get('markets', []))} groups={len(payload.get('groups', []))}")


if __name__ == "__main__":
    main()
