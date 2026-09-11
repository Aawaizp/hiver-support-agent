"""Create a curated local archive; no upload and no API calls."""
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
AGENT = "e14347f8f9f185d629c453a8c973ff168d6183da3eb2af7605772c106b96e4d8"
BASELINE = "7b2d1d92e828737cea676cb483158c2898620841601e337f303b73ab2d46531c"
REVIEW = "a3b4be5445390607"


def main():
    required = ["README.md", "requirements.txt", ".env.example", ".gitignore",
                "run.py", "resume_test.py", "build_submission.py",
                "data/prepared/manifest.json", "data/prepared/duplicate_report.json",
                "data/prepared/duplicate_review.json",
                "data/prepared/retrieval_conversations_clean.jsonl"]
    files = {ROOT / name for name in required}
    for pattern in ("hiver/**/*.py", "tests/*.py", "docs/*.md",
                    "data/prepared/dev_batch_*.json", "data/prepared/test_batch_*.json",
                    "data/prepared/annotation_method_*.json", "results/*_dev_*.json"):
        files.update(ROOT.glob(pattern))
    for name in ("recovery_predictions.json", "recovery_metrics.json", "predictions.json", "metrics.json"):
        files.add(ROOT / "results/agent_test" / AGENT / name)
    for name in ("metrics.json", "simple_predictions.json", "trivial_predictions.json"):
        files.add(ROOT / "results/baselines_test" / BASELINE / name)
    for name in ("candidates.json", "human_review.json"):
        files.add(ROOT / "results/agent_review" / REVIEW / name)
    for name in ("agreement.json", "judge_results.json", "human_review_snapshot.json"):
        files.add(ROOT / "results/agent_review" / REVIEW / "judge/b0bd4a688c636ff2" / name)

    # Read configured secrets only to ensure no archive member contains their values.
    secrets = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if re.search(r"API_KEY|TOKEN|SECRET|PASSWORD", key, re.I) and len(value) >= 12:
                secrets.append(value.encode())

    contents = {}
    for path in sorted(files):
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(ROOT).as_posix()
        if resolved != path.absolute() or any(part in {".env", ".backups", "__pycache__", "api_cache"} for part in path.parts):
            raise ValueError(f"Excluded or redirected file: {relative}")
        payload = path.read_bytes()
        if any(secret in payload for secret in secrets) or re.search(rb"(?:gsk_|AIza)[A-Za-z0-9_-]{25,}", payload):
            raise ValueError(f"Possible secret in {relative}; archive not created.")
        contents[relative] = payload
    manifest = {name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()}
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    temporary = destination / "Hiver-submission.tmp"
    archive = destination / "Hiver-submission.zip"
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as bundle:
        for name, payload in contents.items():
            bundle.writestr("Hiver/" + name, payload)
        bundle.writestr("Hiver/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(temporary) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Archive integrity check failed.")
        for name, digest in manifest.items():
            if hashlib.sha256(bundle.read("Hiver/" + name)).hexdigest() != digest:
                raise RuntimeError(f"Archive content mismatch: {name}")
    temporary.replace(archive)
    print(f"Created {archive}")
    print(f"{len(contents)} files plus SHA-256 manifest; {archive.stat().st_size:,} bytes.")
    print("No .env, raw CSV, API caches or old backups included. Nothing uploaded.")


if __name__ == "__main__":
    main()
