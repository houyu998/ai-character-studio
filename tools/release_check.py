"""Pre-release self check. Run: python tools/release_check.py [repo_root]
Reports personal absolute paths, photos/images, models, archives, large files and secret-like strings.
Exit code 0 only when every count is 0. Writes nothing."""
import re, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
TEXT = {".py", ".md", ".txt", ".json", ".ps1", ".sh", ".toml", ".cfg", ".yml", ".yaml", ".gitignore", ""}
BAN_EXT = {
    "image/photo": {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".exr", ".heic", ".bmp"},
    "model": {".blend", ".blend1", ".fbx", ".obj", ".glb", ".gltf", ".abc", ".usd", ".usdz"},
    "archive": {".zip", ".7z", ".rar"},
    "video": {".mp4", ".mov", ".mkv"},
}
PATTERNS = {
    "personal path": re.compile(r"[A-Za-z]:(?:/|\\){1,2}Users(?:/|\\)|/Users/[^/\s]+/|/home/[^/\s]+/", re.I),
    "secret": re.compile(r"(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
                         r"|(api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]{8,})", re.I),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@(?!example\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
hits = {k: [] for k in list(BAN_EXT) + list(PATTERNS) + ["large file (>5 MB)", "cup preset"]}
for p in sorted(root.rglob("*")):
    if not p.is_file() or ".git" in p.parts:
        continue
    rel = p.relative_to(root).as_posix()
    ext = p.suffix.lower()
    showcase = rel.startswith("docs/images/") and ext in {".jpg", ".mp4"}  # approved renders only
    for k, exts in BAN_EXT.items():
        if ext in exts and not showcase:
            hits[k].append(rel)
    if p.stat().st_size > 5 * 2**20:
        hits["large file (>5 MB)"].append(rel)
    if re.match(r"presets/cups", rel):
        hits["cup preset"].append(rel)
    if ext in TEXT or p.name in {".gitignore", "LICENSE"}:
        if p.name == "release_check.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for k, rx in PATTERNS.items():
                if rx.search(line):
                    hits[k].append(f"{rel}:{i}")
total = 0
for k, v in hits.items():
    total += len(v)
    print(f"K {k}: {len(v)}")
    for x in v[:20]:
        print("    ", x)
print("K TOTAL", total, "-> PASS" if total == 0 else "-> FAIL")
sys.exit(0 if total == 0 else 1)
