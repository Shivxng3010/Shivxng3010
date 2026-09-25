#!/usr/bin/env python3
"""
cards.py - render GitHub stat and repo cards as SVGs. Stdlib only.

Replaces shared public card instances with static, resilient SVGs built in your repo.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "cards.py"}

THEMES = {
    "dark": {
        "bg": "#0A101F",
        "border": "#1E293B",
        "title": "#22D3EE",
        "text": "#94A3B8",
        "muted": "#64748B",
        "value": "#F8FAFC",
        "accent": "#A78BFA",
    },
    "light": {
        "bg": "#FFFFFF",
        "border": "#CBD5E1",
        "title": "#0891B2",
        "text": "#475569",
        "muted": "#94A3B8",
        "value": "#0F172A",
        "accent": "#7C3AED",
    },
}

LANG_COLOR = {
    "Python": "#3572A5",
    "JavaScript": "#F1E05A",
    "TypeScript": "#3178C6",
    "HTML": "#E34C26",
    "CSS": "#563D7C",
    "PostgreSQL": "#336790",
    "PLpgSQL": "#336790",
    "Shell": "#89E051",
    "Docker": "#384D54",
}

FONT = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

ICON_STAR = (
    "M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 "
    "2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 "
    "01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"
)
ICON_FORK = (
    "M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75v-.878a2.25 2.25 0 "
    "111.5 0v.878a2.25 2.25 0 01-2.25 2.25h-1.5v2.128a2.251 2.251 0 11-1.5 "
    "0V8.5h-1.5A2.25 2.25 0 013.5 6.25v-.878a2.25 2.25 0 111.5 0zM5 3.25a.75.75 0 "
    "10-1.5 0 .75.75 0 001.5 0zm6.75.75a.75.75 0 100-1.5.75.75 0 000 1.5zm-3 "
    "8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"
)
ICON_REPO = (
    "M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 "
    "0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 "
    "012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 "
    "12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 "
    "00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"
)


def icon(path, x, y, size, fill):
    s = size / 16
    return f'<path transform="translate({x:.1f},{y:.1f}) scale({s:.3f})" fill="{fill}" d="{path}"/>'


def rest(path: str, token: str | None):
    req = urllib.request.Request("https://api.github.com" + path, headers=dict(UA))
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text_width(s: str, size: float) -> float:
    return len(s) * size * 0.53


def wrap(text: str, size: float, max_w: float, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if text_width(trial, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and words:
        used = len(" ".join(lines).split())
        if used < len(words):
            while lines and text_width(lines[-1] + "…", size) > max_w:
                lines[-1] = lines[-1].rsplit(" ", 1)[0]
            lines[-1] += "…"
    return lines


def frame(w, h, c, body, label):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{esc(label)}" '
        f'font-family="{FONT}">'
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" '
        f'fill="{c["bg"]}" stroke="{c["border"]}" stroke-width="1.2"/>'
        f"{body}</svg>"
    )


def render_repo(repo, theme):
    c = THEMES[theme]
    W, H = 420, 132
    pad = 18
    out = []

    out.append(icon(ICON_REPO, pad, pad, 15, c["title"]))
    out.append(
        f'<text x="{pad + 22}" y="{pad + 12}" font-size="14.5" font-weight="700" '
        f'fill="{c["title"]}">{esc(repo["name"])}</text>'
    )

    desc = repo.get("description") or "No description provided."
    for i, line in enumerate(wrap(desc, 11.5, W - 2 * pad, 3)):
        out.append(
            f'<text x="{pad}" y="{pad + 36 + i * 16}" font-size="11.5" '
            f'fill="{c["text"]}">{esc(line)}</text>'
        )

    fy = H - pad - 2
    x = pad
    if repo.get("language"):
        col = LANG_COLOR.get(repo["language"], c["muted"])
        out.append(f'<circle cx="{x + 5}" cy="{fy - 4}" r="5" fill="{col}"/>')
        out.append(
            f'<text x="{x + 15}" y="{fy}" font-size="11" fill="{c["muted"]}">'
            f'{esc(repo["language"])}</text>'
        )
        x += 15 + text_width(repo["language"], 11) + 18

    for path, count in ((ICON_STAR, repo.get("stars", 0)), (ICON_FORK, repo.get("forks", 0))):
        out.append(icon(path, x, fy - 11, 12, c["muted"]))
        out.append(
            f'<text x="{x + 17}" y="{fy}" font-size="11" fill="{c["muted"]}">'
            f'{count}</text>'
        )
        x += 17 + text_width(str(count), 11) + 18

    return frame(W, H, c, "".join(out), f'{repo["name"]} repository card')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True)
    p.add_argument("--out", type=Path, default=Path("assets"))
    p.add_argument("--projects", type=Path, default=Path("assets/projects.json"))
    args = p.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        repos = []
        page = 1
        while True:
            batch = rest(f"/users/{args.user}/repos?per_page=100&page={page}&type=owner", token)
            repos += batch
            if len(batch) < 100:
                break
            page += 1
    except Exception as e:
        print(f"Warning: could not fetch repos via API: {e}", file=sys.stderr)
        repos = []

    if not args.projects.exists():
        print(f"no {args.projects}, skipping repo cards")
        return

    wanted = json.loads(args.projects.read_text(encoding="utf-8"))["projects"]
    by_name = {r["name"].lower(): r for r in repos}

    for entry in wanted:
        repo_name = entry["repo"]
        src = by_name.get(repo_name.lower())
        card = {
            "name": repo_name,
            "description": entry.get("description") or (src.get("description") if src else ""),
            "language": (src.get("language") if src else None) or "Python",
            "stars": src.get("stargazers_count", 0) if src else 0,
            "forks": src.get("forks_count", 0) if src else 0,
        }
        for theme in ("dark", "light"):
            dest = args.out / f"card-{repo_name}-{theme}.svg"
            dest.write_text(render_repo(card, theme), encoding="utf-8")
        print(f"wrote card-{repo_name}-*.svg ({card['stars']} star, {card['language']})")


if __name__ == "__main__":
    main()
