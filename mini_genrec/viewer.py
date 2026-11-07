"""Package the completed evaluation as a self-contained, offline HTML viewer."""
import argparse
import base64
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .llm import canonical, parse_response

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/frozen-evaluation"
SNAPSHOT = ROOT / "experiments/mini-genrec-v1"
TEMPLATE = Path(__file__).with_name("viewer_assets") / "index.html"


def load(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def replay(titles, lookup, candidates, unconstrained=False):
    ranking, rejected = [], []
    seen = set()
    for title in titles:
        mid = lookup.get(canonical(title))
        reason = ("No unique catalog match" if mid is None else
                  "Outside candidate set" if not unconstrained and mid not in candidates else
                  "Duplicate" if mid in seen else None)
        if reason:
            rejected.append([title, reason])
        else:
            ranking.append(mid)
            seen.add(mid)
    return ranking, rejected


def build_data(results=RESULTS, snapshot=SNAPSHOT):
    status = load(results, "status.json")
    run = load(results, "run.json")
    if status["status"] != "complete" or status["mode"] != "api" or run["run_id"] != status["run_id"]:
        raise ValueError("Viewer requires a completed, matching API run.")
    movies = {m["movieId"]: [m["title"], m["genres"]] for m in load(snapshot, "catalog.json")["movies"]}
    groups = defaultdict(list)
    for mid, (title, _) in movies.items():
        groups[canonical(title)].append(mid)
    lookup = {title: ids[0] for title, ids in groups.items() if len(ids) == 1}
    cases = {c["user_id"]: c for c in load(snapshot, "cases.json")["cases"]
             if c["user_id"] in run["selected_users"]}
    responses = load(results, "responses.json")["responses"]
    rows = load(results, "per_user.json")
    if sum(bool(r["model"]) for r in rows) != run["logical_llm_conditions"]:
        raise ValueError("Incomplete model conditions.")
    users = {uid: dict(target=c["target_movie_id"], candidates=c["candidate_movie_ids"],
                       history=[[r["movieId"], r["rating"], (c["target_timestamp"]-r["timestamp"])//86400]
                                for r in sorted(c["history"], key=lambda r: (r["timestamp"], r["movieId"]))][-50:], rows=[])
             for uid, c in cases.items()}
    seen = set()
    for row in rows:
        uid = row["user_id"]
        case = users[uid]
        key = (uid, row["model"], row["ranker"], row["strategy"], row["history_length"])
        if key in seen:
            raise ValueError("Duplicate evaluation row.")
        seen.add(key)
        if row["ranker"] in {"random", "popularity"}:
            continue
        rejected = None
        if row["model"] and not row["error"]:
            titles = parse_response(responses[row["request_hash"]])
            ranking, rejected = replay(titles, lookup, set(case["candidates"]), row["ranker"] == "unconstrained")
            counts = Counter(reason for _, reason in rejected)
            if (ranking != row["ranking"] or len(titles) != row["suggestions"]
                    or counts["Duplicate"] != row["duplicates"]
                    or counts["No unique catalog match"] != row["outside_catalog"]
                    or (row["ranker"] != "unconstrained" and len(rejected)-row["duplicates"] != row["outside_candidates"])):
                raise ValueError("Archived response disagrees with saved validation.")
        if row["model"] and (row["budget_dropped"] or row["repeat_events_compressed"]):
            raise ValueError("This viewer requires explicit retained events for budget drops or compressed repeats.")
        window = case["history"][-row["history_length"]:]
        retained = sum(rating >= 4 or rating <= 2 for _, rating, _ in window) if row["strategy"] == "engineered_v1" else len(window)
        if len(window) != row["window_events"] or retained != row["history_events"]:
            raise ValueError("Saved history counts disagree with the viewer's reconstruction.")
        case["rows"].append({key: row[key] for key in (
            "model", "ranker", "strategy", "history_length", "ranking", "error_category", "http_status",
            "history_events", "window_events", "low_signal_dropped", "api_input_tokens", "api_output_tokens",
            "candidate_coverage")} | {"rejected": rejected})
    with (results / "metrics.csv").open(newline="", encoding="utf-8") as file:
        metrics = list(csv.DictReader(file))
    main = [r for r in metrics if r["view"] == "main"]
    context = [r for r in metrics if r["view"] == "context"]
    return dict(users=users, movies=movies, models=run["models"], main=main, context=context,
                conditions=run["logical_llm_conditions"], run_id=run["run_id"],
                sweep=base64.b64encode((results / "elbow.png").read_bytes()).decode("ascii"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "viewer.html")
    args = parser.parse_args()
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__RUN_DATA__", data, 1), encoding="utf-8")
    print(f"Open {args.output.resolve()} in a browser.")


if __name__ == "__main__":
    main()
