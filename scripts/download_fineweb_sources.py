"""Download only the revision-pinned FineWeb shards named in a local manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    status = args.manifest.parent / 'download-progress.json'
    completed = []
    for record in manifest['files']:
        dest = Path(record['path']); dest.parent.mkdir(parents=True, exist_ok=True)
        expected = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}/{record['relative_path']}"
        if record['url'] != expected or len(manifest['revision']) != 40:
            raise ValueError('Require an exact revision-pinned source URL')
        if dest.exists():
            if dest.stat().st_size != record['bytes'] or digest(dest) != record['sha256']:
                raise ValueError('Existing source failed verification; preserved for inspection')
            completed.append(str(dest)); continue
        partial = dest.with_suffix('.parquet.part')
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > record['bytes']:
                raise ValueError('Oversized partial source; preserved for inspection')
            if offset == record['bytes']:
                break
            remaining = record['bytes'] - offset
            if shutil.disk_usage(dest.parent).free < remaining + 12 * 1024**3:
                raise RuntimeError('Insufficient launch reserve for download; partial preserved')
            headers = {'Accept-Encoding': 'identity', 'User-Agent': 'particle-lm-research/1.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            url = expected + f'?download=true&attempt={attempt}&t={int(time.time())}'
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                    if response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Unexpected range; refusing concatenation')
                        mode = 'ab'
                    elif response.status == 200:
                        mode = 'wb'; offset = 0
                    else:
                        raise ValueError(f'Unexpected status {response.status}')
                    last = 0
                    with partial.open(mode) as stream:
                        while block := response.read(4 << 20):
                            stream.write(block); offset += len(block)
                            if offset > record['bytes']:
                                raise ValueError('Source exceeded pinned size')
                            if time.monotonic() - last > 5:
                                row = dict(state='downloading', file=str(dest), bytes=offset,
                                           total_bytes=record['bytes'], completed=completed,
                                           all_source_bytes=sum(x['bytes'] for x in manifest['files']), updated_unix=time.time())
                                save(status, row); print(json.dumps(row), flush=True); last = time.monotonic()
                        stream.flush(); os.fsync(stream.fileno())
                if offset != record['bytes']:
                    raise OSError('Incomplete response')
                break
            except OSError as error:
                save(status, dict(state='retrying', file=str(dest), attempt=attempt+1,
                                  reason=str(error), completed=completed, updated_unix=time.time()))
                if attempt == 4:
                    raise
        if partial.stat().st_size != record['bytes'] or digest(partial) != record['sha256']:
            raise ValueError('Downloaded source checksum failed; partial preserved')
        partial.replace(dest); completed.append(str(dest))
    save(status, dict(state='completed', completed=completed, verified_sha256=True,
                      manifest_sha256=digest(args.manifest), updated_unix=time.time()))


if __name__ == '__main__':
    main()
