"""Per-tag URL analysis: exports URL metrics broken down by tag, prompt, and model."""

import argparse
from datetime import date

import pandas as pd

from peec_api import PeecClient


def build_filter(tag_id: str) -> list[dict]:
    return [{"field": "tag_id", "operator": "in", "values": [tag_id]}]


def fetch_prompt_texts(client: PeecClient, project_id: str) -> dict[str, str]:
    """Return {prompt_id: prompt_text} for all prompts in the project."""
    prompts = client.fetch_all(client.list_prompts, project_id=project_id)
    mapping = {}
    for p in prompts:
        pid = p["id"]
        messages = p.get("messages", [])
        text = messages[0]["content"] if messages else ""
        mapping[pid] = text
    return mapping


def fetch_tag_url_data(
    client: PeecClient,
    project_id: str,
    tag_id: str,
    tag_name: str,
    start_date: str,
    end_date: str,
    prompt_texts: dict[str, str],
) -> list[dict]:
    """Fetch URL report for a single tag, broken down by prompt and model."""
    rows = client.fetch_all(
        client.report_urls,
        project_id=project_id,
        start_date=start_date,
        end_date=end_date,
        filters=build_filter(tag_id),
        dimensions=["prompt_id", "model_id"],
    )

    results = []
    for r in rows:
        prompt_id = r.get("prompt", {}).get("id", "")
        model_id = r.get("model", {}).get("id", "")
        results.append({
            "tag": tag_name,
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "classification": r.get("classification", ""),
            "prompt_id": prompt_id,
            "prompt_text": prompt_texts.get(prompt_id, ""),
            "model": model_id,
            "mentions": r.get("citation_count", 0),
            "retrievals": r.get("usage_count", 0),
            "citation_rate": round(r.get("citation_avg", 0), 4),
        })
    return results


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

    # Load prompt texts
    print("Fetching prompts...")
    prompt_texts = fetch_prompt_texts(client, project_id)
    print(f"  {len(prompt_texts)} prompts loaded")

    # Get tags
    tags = client.list_tags(project_id)
    if isinstance(tags, dict):
        tags = tags.get("data", [])

    if args.tags:
        tag_filter = {t.lower() for t in args.tags}
        tags = [t for t in tags if t["name"].lower() in tag_filter]

    print(f"Processing {len(tags)} tags: {[t['name'] for t in tags]}")

    # Collect data for each tag
    all_rows = []
    for tag in tags:
        print(f"  Fetching URLs for tag '{tag['name']}'...")
        rows = fetch_tag_url_data(
            client, project_id, tag["id"], tag["name"],
            args.start_date, args.end_date, prompt_texts,
        )
        all_rows.extend(rows)
        print(f"    {len(rows)} URL rows")

    if not all_rows:
        print("No data found.")
        return

    df = pd.DataFrame(all_rows)
    df.sort_values(["tag", "mentions"], ascending=[True, False], inplace=True)

    # Save output
    output = args.output
    if output.endswith(".csv"):
        df.to_csv(output, index=False)
    else:
        df.to_excel(output, index=False, sheet_name="Tag URL Analysis")

    print(f"\nSaved {len(df)} rows to {output}")
    print(f"Tags covered: {df['tag'].nunique()}")
    print(f"Unique URLs: {df['url'].nunique()}")


if __name__ == "__main__":
    main()
