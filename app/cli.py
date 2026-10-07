"""Command-line entry point: `uv run python -m app <command>`.

Commands:
    process <audio> --team T --roster "A,B,C" [--json] [--db PATH]
    serve [--host H] [--port N] [--db PATH] [--no-open]

Prints a per-person task list + notes + stage timings by default; `--json` dumps
the MeetingResult as JSON instead. Exits 1 with a one-line error message (no
traceback) on missing file, missing API key, or network failure.
"""

import argparse
import json
import sys
import threading
import urllib.error
import webbrowser
from pathlib import Path

from app.feedback import FeedbackStore
from app.pipeline import DEFAULT_DB, MeetingResult, run_pipeline


def _parse_roster(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app", description="MeetNotes command line.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process", help="run the full pipeline on an audio clip")
    p.add_argument("audio", type=Path, help="path to an audio file")
    p.add_argument("--team", default="default", help="team id (default: 'default')")
    p.add_argument("--roster", default="", help='comma-separated roster, e.g. "Aarush, Kaushal"')
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"feedback DB (default: {DEFAULT_DB})")
    p.add_argument("--json", action="store_true", help="emit the MeetingResult as JSON")
    s = sub.add_parser("serve", help="serve the web UI + API (one command for the demo)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"feedback DB (default: {DEFAULT_DB})")
    s.add_argument("--no-open", action="store_true", help="don't open the browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "process":
        return _cmd_process(args)
    if args.cmd == "serve":
        return _cmd_serve(args)
    return 0  # unreachable: argparse enforces required=True


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn  # imported here so `process` doesn't pay the server import cost

    from app.api.main import DEFAULT_UI_DIST, create_app

    if not (DEFAULT_UI_DIST / "index.html").is_file():
        print(
            f"warning: no built UI at {DEFAULT_UI_DIST} (run `pnpm install && pnpm build` in ui/); "
            "serving the minimal fallback page",
            file=sys.stderr,
        )
    url = f"http://{args.host}:{args.port}"
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    print(f"MeetNotes at {url}  (Ctrl+C to stop)")
    try:
        uvicorn.run(create_app(FeedbackStore(args.db)), host=args.host, port=args.port, log_level="info")
    except OSError as exc:  # e.g. port already in use
        print(f"error: cannot serve on {url}: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_process(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    if not audio.is_file():
        print(f"error: audio file not found: {audio}", file=sys.stderr)
        return 1

    store = FeedbackStore(args.db)
    try:
        result = run_pipeline(audio, args.team, _parse_roster(args.roster), store)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:  # missing API key or similar config error
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except urllib.error.HTTPError as exc:
        print(f"error: upstream {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"error: network failure: {exc.reason}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: I/O failure: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_pretty(result)
    return 0


def _print_pretty(result: MeetingResult) -> None:
    print(f"Processed {result.audio_path.name} in {result.total_seconds:.2f}s")
    for line in result.stage_logs:
        print(f"  {line}")

    print(f"\nSummary:\n  {result.summary or '(none)'}")

    by_person: dict[str, list[dict]] = {}
    for t in result.tasks:
        by_person.setdefault(t["assignee"], []).append(t)
    # Include every roster name even without tasks, matching the UI's grouping.
    everyone = list(dict.fromkeys([*by_person.keys(), *result.roster]))
    print("\nTasks by person:")
    if not everyone:
        print("  (no roster or tasks)")
    for name in everyone:
        print(f"  {name}:")
        tasks = by_person.get(name, [])
        if not tasks:
            print("    (no tasks)")
            continue
        for task in tasks:
            deadline = f" ({task['deadline']})" if task.get("deadline") else ""
            print(f"    - {task['task']}{deadline}")
            if task.get("source_quote"):
                print(f"      quote: {task['source_quote']!r}")

    print("\nNotes:")
    if not result.notes:
        print("  (no notes)")
    for note in result.notes:
        print(f"  {note.get('topic', '')}:")
        for point in note.get("points", []):
            print(f"    - {point}")
