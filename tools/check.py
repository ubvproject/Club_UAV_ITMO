#!/usr/bin/env python3
"""Автопроверка решений секции. Только стандартная библиотека Python 3.9+.

    python3 tools/check.py                    структура, размеры, синтаксис, секреты — всё в resheniya/
    python3 tools/check.py --pr-author NICK --base SHA [--assoc OWNER]
                                              плюс: pull request меняет только resheniya/<задача>/<nick>/
    python3 tools/check.py --svodka           пересобрать resheniya/README.md (кто что сдал)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESH = ROOT / "resheniya"
ZAD = ROOT / "zadachi"

MAX_MB = 5
NICK_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38})$")      # ник GitHub строчными
BAD_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".zip", ".7z", ".rar", ".tar", ".gz", ".bag", ".exe", ".dmg"}
TEXT_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".cfg", ".ini", ".sh", ".env"}
SECRETS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(password|passwd|пароль|token|api[_-]?key|secret)\b\s*[:=]\s*[\"'][^\"'\s]{4,}[\"']", re.I),
]
MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}

errors: list[str] = []
warnings: list[str] = []


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def tasks() -> list[str]:
    return sorted(d.name for d in ZAD.iterdir() if d.is_dir() and (d / "README.md").exists())


def check_file(f: Path) -> None:
    mb = f.stat().st_size / 1e6
    if f.suffix.lower() in BAD_EXT:
        errors.append(f"{rel(f)}: запрещённый тип {f.suffix} — видео и архивы ссылкой в README.md")
    elif mb > MAX_MB:
        errors.append(f"{rel(f)}: {mb:.1f} МБ, больше {MAX_MB} МБ — тяжёлое ссылкой в README.md")
    if f.suffix.lower() not in TEXT_EXT or mb > MAX_MB:
        return
    text = f.read_text(encoding="utf-8", errors="replace")
    for n, line in enumerate(text.splitlines(), 1):
        if any(r.search(line) for r in SECRETS):
            errors.append(f"{rel(f)}:{n}: похоже на секрет (пароль, токен, ключ) — уберите из кода")
    if f.suffix == ".py":
        try:
            compile(text, str(f), "exec")
        except SyntaxError as e:
            errors.append(f"{rel(f)}:{e.lineno}: синтаксическая ошибка — {e.msg}")


def check_tree() -> None:
    known = tasks()
    for d in sorted(RESH.iterdir()):
        if d.name in ("README.md", "_shablon"):
            continue
        if not d.is_dir() or d.name not in known:
            errors.append(f"{rel(d)}: нет такой задачи в zadachi/ (есть: {', '.join(known)})")
            continue
        for s in sorted(d.iterdir()):
            if s.name == ".gitkeep":
                continue
            if not s.is_dir():
                errors.append(f"{rel(s)}: файл лежит прямо в папке задачи — положите его в папку со своим ником")
                continue
            if not NICK_RE.match(s.name):
                errors.append(f"{rel(s)}: имя папки — ник GitHub строчными: латиница, цифры, «-»")
            readme = s / "README.md"
            if not readme.exists():
                errors.append(f"{rel(s)}: нет README.md — скопируйте resheniya/_shablon/README.md")
            elif not result_line(readme):
                warnings.append(f"{rel(readme)}: нет строки «Результат: …» — её читает сводка")
            for f in sorted(s.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    check_file(f)


def check_pr(author: str, base: str, assoc: str) -> None:
    if assoc.upper() in MAINTAINERS:
        print(f"Автор {author} ({assoc}) — тренер, папки не ограничены")
        return
    out = subprocess.run(["git", "-c", "core.quotePath=false", "diff", "--name-only", "--no-renames", base, "HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    nick = author.lower()
    for path in filter(None, out.splitlines()):
        parts = path.split("/")
        if not (len(parts) >= 4 and parts[0] == "resheniya" and parts[2] == nick):
            errors.append(f"{path}: файл вне вашей папки — меняйте только resheniya/<задача>/{nick}/")


def result_line(readme: Path) -> str:
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*\**Результат\**\s*:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def svodka() -> None:
    lines = ["# Сводка решений", "",
             "Собирается командой `python3 tools/check.py --svodka` — руками не править.", ""]
    for t in tasks():
        title = (ZAD / t / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        lines += [f"## [{title}](../zadachi/{t}/)", ""]
        done = sorted(s for s in (RESH / t).glob("*/") if s.is_dir()) if (RESH / t).exists() else []
        if not done:
            lines += ["_Пока никто не сдал._", ""]
            continue
        lines += ["| Участник | Результат |", "|---|---|"]
        for s in done:
            res = result_line(s / "README.md") if (s / "README.md").exists() else ""
            lines.append(f"| [{s.name}]({t}/{s.name}/) | {res.replace('|', '/') or '—'} |")
        lines.append("")
    (RESH / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Записано: {rel(RESH / 'README.md')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr-author", help="ник автора pull request")
    ap.add_argument("--base", help="коммит, от которого считать изменения")
    ap.add_argument("--assoc", default="", help="author_association из GitHub (OWNER, COLLABORATOR, …)")
    ap.add_argument("--svodka", action="store_true", help="пересобрать resheniya/README.md")
    a = ap.parse_args()

    if a.svodka:
        svodka()
        return 0
    check_tree()
    if a.pr_author:
        check_pr(a.pr_author, a.base or "origin/main", a.assoc)

    for w in warnings:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] {w}")
    for e in errors:
        print(f"[ОШИБКА] {e}")
    if errors:
        print(f"\nНайдено ошибок: {len(errors)}. Как исправить — CONTRIBUTING.md, «Частые ошибки автопроверки».")
        return 1
    print("[ ОК ] Всё в порядке")
    return 0


if __name__ == "__main__":
    sys.exit(main())
