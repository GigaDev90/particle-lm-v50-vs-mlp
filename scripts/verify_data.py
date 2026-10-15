"""Verify portable corpus/tokenizer identities without loading them into RAM."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(data_dir, checksums, recipe):
    results = []
    for filename, expected in checksums.items():
        if filename == "note":
            continue
        path = data_dir / filename
        result = {"file": filename, "expected_sha256": expected, "passed": False}
        if not path.is_file():
            result["error"] = "missing file"
        else:
            result["sha256"] = digest(path)
            result["bytes"] = path.stat().st_size
            result["passed"] = result["sha256"] == expected
            if filename.endswith(".bin"):
                expected_bytes = 2 * recipe[filename[:-4] + "_tokens"]
                result["expected_bytes"] = expected_bytes
                result["passed"] &= result["bytes"] == expected_bytes
        results.append(result)
    return {"passed": all(item["passed"] for item in results), "files": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    checksums = json.loads((ROOT / "data/checksums.json").read_text())
    recipe = json.loads((ROOT / "data/recipe.json").read_text())
    report = verify(args.data_dir, checksums, recipe)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
