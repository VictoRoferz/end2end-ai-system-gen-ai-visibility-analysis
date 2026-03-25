"""Per-tag URL analysis: exports URL metrics with brand mentions, citation
positions, LLM output, and domain info — matching and extending the
Konsolidierung_Sources reference format."""

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from urllib.parse import urlparse

import pandas as pd

from peec_api import PeecClient

# Model ID → human-readable name (matches reference file)
MODEL_DISPLAY_NAMES = {
    "chatgpt-scraper": "GPT",
    "google-ai-overview-scraper": "AI Overviews",
    "perplexity-scraper": "Perplexity",
}

# German month abbreviations for Timeframe formatting
_DE_MONTHS = {
    1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sept", 10: "Okt", 11: "Nov", 12: "Dez",
}


def build_filter(tag_id: str) -> list[dict]:
    return [{"field": "tag_id", "operator": "in", "values": [tag_id]}]


def extract_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc
        return host.removeprefix("www.")
    except Exception:
        return ""


def is_medel_source(url: str) -> str:
    domain = extract_domain(url).lower()
    return "Yes" if "medel" in domain else "No"


def compute_anchor(all_dates: list[str]) -> date:
    """Find the earliest date and align to the Monday of that week."""
    parsed = [datetime.strptime(d, "%Y-%m-%d").date() for d in all_dates if d]
    earliest = min(parsed)
    return earliest - timedelta(days=earliest.weekday())


def _fmt_de_date(d: date) -> str:
    """Format a date as 'Mon DD' with German month abbreviations."""
    return f"{_DE_MONTHS[d.month]} {d.day:02d}"


def date_to_timeframe(date_str: str, anchor: date) -> str:
    """Convert a date string to a biweekly 'Woche N Mon DD- Mon DD' label."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    days_since = (d - anchor).days
    idx = days_since // 14 + 1
    window_start = anchor + timedelta(days=(idx - 1) * 14)
    window_end = window_start + timedelta(days=13)
    return f"Woche {idx} {_fmt_de_date(window_start)}- {_fmt_de_date(window_end)}"


def date_to_biweekly_index(date_str: str, anchor: date) -> int:
    """Return 1-based biweekly period index."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (d - anchor).days // 14 + 1


def fetch_prompt_data(client: PeecClient, project_id: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return prompt texts and prompt→tag mappings.

    Returns:
        prompt_texts: {prompt_id: prompt_text}
        prompt_tags: {prompt_id: set of tag_ids}
    """
    prompts = client.fetch_all(client.list_prompts, project_id=project_id)
    texts = {}
    tags = {}
    for p in prompts:
        pid = p["id"]
        messages = p.get("messages", [])
        texts[pid] = messages[0]["content"] if messages else ""
        tags[pid] = {t["id"] for t in p.get("tags", [])}
    return texts, tags


def fetch_chat_details(
    client: PeecClient,
    project_id: str,
    tag_id: str,
    prompt_tags: dict[str, set[str]],
    all_chats: list[dict],
    chat_cache: dict[str, dict],
) -> tuple[dict, dict, dict]:
    """Fetch chat-level brand mentions, citation positions, and LLM responses.

    Uses all_chats (from list_chats) filtered by prompt→tag association to cover
    all models (GPT, AI Overviews, Perplexity, etc.), not just GPT.

    Returns:
        chat_brands: {(prompt_id, model_id, date): set of brand names}
        url_positions: {(url, prompt_id, model_id, date): list of citationPositions}
        chat_responses: {(prompt_id, model_id, date): assistant response text}
    """
    # Filter chats belonging to this tag via prompt→tag mapping
    tag_chats = []
    for c in all_chats:
        prompt_id = c["prompt"]["id"]
        if tag_id in prompt_tags.get(prompt_id, set()):
            tag_chats.append(c)

    # Build chat_id → associations
    chat_keys: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for c in tag_chats:
        cid = c["id"]
        prompt_id = c["prompt"]["id"]
        model_id = c["model"]["id"]
        c_date = c.get("date", "")
        combo = (cid, prompt_id, model_id, c_date)
        if combo not in seen:
            seen.add(combo)
            chat_keys[cid].append((prompt_id, model_id, c_date))

    # Fetch each chat's content (with cache + throttle)
    chat_brands: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    url_positions: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    chat_responses: dict[tuple[str, str, str], str] = {}
    fetched = 0

    for i, (chat_id, associations) in enumerate(chat_keys.items()):
        if (i + 1) % 25 == 0:
            print(f"      chat {i + 1}/{len(chat_keys)}", end="\r", flush=True)

        if chat_id in chat_cache:
            content = chat_cache[chat_id]
        else:
            try:
                content = client.get_chat_content(chat_id, project_id)
                chat_cache[chat_id] = content
                fetched += 1
                if fetched % 10 == 0:
                    time.sleep(0.5)
            except Exception as e:
                print(f"      warning: chat {chat_id} failed: {e}", file=sys.stderr)
                continue

        # Extract assistant response
        assistant_text = ""
        for msg in content.get("messages", []):
            if msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                break

        for prompt_id, model_id, q_date in associations:
            key = (prompt_id, model_id, q_date)

            # Brand mentions
            for brand in content.get("brands_mentioned", []):
                chat_brands[key].add(brand["name"])

            # Citation positions per URL
            for src in content.get("sources", []):
                url_key = (src["url"], prompt_id, model_id, q_date)
                url_positions[url_key].append(src.get("citationPosition", 0))

            # LLM response
            if key not in chat_responses:
                chat_responses[key] = assistant_text

    if chat_keys:
        cached = len(chat_keys) - fetched
        print(f"      {len(chat_keys)} chats ({fetched} fetched, {cached} cached)     ")

    return dict(chat_brands), dict(url_positions), chat_responses


def main():
    parser = argparse.ArgumentParser(description="Per-tag URL analysis for Peec project")
    parser.add_argument("--start-date", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="tag_url_analysis.xlsx", help="Output file (.xlsx or .csv)")
    parser.add_argument("--tags", nargs="*", help="Filter to specific tag names (default: all)")
    args = parser.parse_args()

    client = PeecClient()

    # Get project
    projects = client.list_projects()
    project_id = projects["data"][0]["id"]
    project_name = projects["data"][0]["name"]
    print(f"Project: {project_name} ({project_id})")

    # Load prompt texts and tag associations
    print("Fetching prompts...")
    prompt_texts, prompt_tags = fetch_prompt_data(client, project_id)
    print(f"  {len(prompt_texts)} prompts loaded")

    # Load model display names
    models_raw = client.list_models(project_id)
    models_list = models_raw.get("data", []) if isinstance(models_raw, dict) else models_raw
    model_name_map = {}
    for m in models_list:
        mid = m["id"]
        model_name_map[mid] = MODEL_DISPLAY_NAMES.get(mid, mid)
    print(f"  Models: {model_name_map}")

    # Get tags
    tags = client.list_tags(project_id)
    if isinstance(tags, dict):
        tags = tags.get("data", [])

    if args.tags:
        tag_filter = {t.lower() for t in args.tags}
        tags = [t for t in tags if t["name"].lower() in tag_filter]

    print(f"Processing {len(tags)} tags: {[t['name'] for t in tags]}\n")

    # Fetch all chats for the date range (covers all models)
    print("Fetching all chats...")
    all_chats = client.fetch_all(
        client.list_chats, project_id=project_id,
        start_date=args.start_date, end_date=args.end_date,
    )
    print(f"  {len(all_chats)} chats loaded")

    all_rows = []
    all_dates: list[str] = []
    chat_cache: dict[str, dict] = {}

    # First pass: collect all URL rows and dates
    tag_data: list[tuple[str, str, list, dict, dict, dict]] = []
    for tag in tags:
        tag_name = tag["name"]
        tag_id = tag["id"]
        print(f"  [{tag_name}]")

        # Layer 1: URL report with date dimension
        print("    Fetching URL report...")
        url_rows = client.fetch_all(
            client.report_urls,
            project_id=project_id,
            start_date=args.start_date,
            end_date=args.end_date,
            filters=build_filter(tag_id),
            dimensions=["prompt_id", "model_id", "date"],
        )
        print(f"    {len(url_rows)} URL rows")

        if not url_rows:
            continue

        # Collect dates for anchor computation
        for r in url_rows:
            d = r.get("date", "")
            if d:
                all_dates.append(d)

        # Layer 2: Chat details (brand mentions + citation positions + LLM output)
        print("    Fetching chat details...")
        chat_brands, url_positions, chat_responses = fetch_chat_details(
            client, project_id, tag_id, prompt_tags, all_chats, chat_cache,
        )

        tag_data.append((tag_name, tag_id, url_rows, chat_brands, url_positions, chat_responses))

    if not all_dates:
        print("No data found.")
        return

    # Compute anchor from earliest date in dataset
    anchor = compute_anchor(all_dates)
    print(f"\nTimeframe anchor: {anchor} (Monday of earliest data week)")

    # Second pass: build rows with timeframe info
    for tag_name, tag_id, url_rows, chat_brands, url_positions, chat_responses in tag_data:
        for r in url_rows:
            prompt_id = r.get("prompt", {}).get("id", "")
            model_id = r.get("model", {}).get("id", "")
            url = r.get("url", "")
            row_date = r.get("date", "")

            key = (prompt_id, model_id, row_date)

            # Brand mentions for this prompt+model+date
            brands = chat_brands.get(key, set())
            brands_str = ", ".join(sorted(brands)) if brands else ""
            med_el_mentioned = "Yes" if any("MED-EL" in b for b in brands) else "No"

            # Citation position for this specific URL+prompt+model+date
            positions = url_positions.get((url, prompt_id, model_id, row_date), [])
            avg_position = round(sum(positions) / len(positions), 2) if positions else None

            # Query category from first token of tag name
            query_category = tag_name.split()[0] if tag_name else ""

            all_rows.append({
                "URL": url,
                "Title": r.get("title", ""),
                "Channel": r.get("classification", ""),
                "Mentioned": med_el_mentioned,
                "Mentions": brands_str,
                "Used total in Query": r.get("usage_count", 0),
                "Avg. Citations in Query": round(r.get("citation_avg", 0), 4),
                "Query Tag": tag_name,
                "Query Category": query_category,
                "Query": prompt_texts.get(prompt_id, ""),
                "Date Queried": row_date,
                "Timeframe": date_to_timeframe(row_date, anchor) if row_date else "",
                "Time frame 2 Weeks": date_to_biweekly_index(row_date, anchor) if row_date else None,
                "MEDEL Quelle/ citiert": is_medel_source(url),
                "Quellen Gesamt": "",
                "Reddit Quelle": "",
                "Wikipedia Quelle": "",
                "Improved Source": "",
                "model": model_name_map.get(model_id, model_id),
                "Kommentar": "",
                # Extra columns
                "Domain": extract_domain(url),
                "Citation Position": avg_position,
                "Citation Count": r.get("citation_count", 0),
                "LLM Output": chat_responses.get(key, ""),
            })

    if not all_rows:
        print("No data found.")
        return

    df = pd.DataFrame(all_rows)
    df.sort_values(
        ["Query Tag", "Time frame 2 Weeks", "Used total in Query"],
        ascending=[True, True, False],
        inplace=True,
    )

    # Save output
    output = args.output
    if output.endswith(".csv"):
        df.to_csv(output, index=False)
    else:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Per-model sheets
            for model_name, group_df in df.groupby("model"):
                sheet = f"Sources Raw Input {model_name}"
                group_df.to_excel(writer, index=False, sheet_name=sheet[:31])

            # Queries sheet
            queries_df = df[["Query", "Query Tag", "Query Category"]].drop_duplicates()
            queries_df.to_excel(writer, index=False, sheet_name="Queries")

    print(f"\nSaved {len(df)} rows to {output}")
    print(f"Tags: {df['Query Tag'].nunique()}")
    print(f"Unique URLs: {df['URL'].nunique()}")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Timeframes: {df['Timeframe'].nunique()}")


if __name__ == "__main__":
    main()
