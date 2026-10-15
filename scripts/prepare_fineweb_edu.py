"""Prepare a revision-pinned FineWeb-Edu corpus from already downloaded Parquet.

No downloads or model training. Documents are outer-whitespace stripped, hashed
as exact UTF-8, assigned to one split by hash, and deduplicated with SQLite. A
new byte-level BPE sees only training documents. All selected documents are
written whole with EOS. Token windows in the native trainer may cross EOS.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time

os.environ.setdefault('RAYON_NUM_THREADS', '4')
SCHEMA = 'fineweb-edu-local-preparation-v1'
EOS = '<|endoftext|>'
SPLITS = ('train', 'val', 'test')
STOP_REQUESTED = False


class PreparationPaused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def report(output, phase, **values):
    record = dict(schema=SCHEMA, phase=phase, updated_unix=time.time(), **values)
    atomic_json(Path(output) / 'progress.json', record)
    print(json.dumps(record, allow_nan=False), flush=True)


def check_resources(output, reserve_bytes, stop_file=None, additional_bytes=0):
    if STOP_REQUESTED or (stop_file is not None and Path(stop_file).exists()):
        raise PreparationPaused('Stop requested; durable cursor is safe to resume')
    free = shutil.disk_usage(output).free
    if free < reserve_bytes + additional_bytes:
        raise PreparationPaused(f'Disk reserve would be crossed: free={free}, reserve={reserve_bytes}, pending={additional_bytes}')
    return free


def document_identity(text):
    require(isinstance(text, str), 'Parquet text must be a string')
    text = text.strip()
    content = text.encode('utf-8', errors='strict')
    digest = hashlib.sha256(content).hexdigest()
    bucket = int(digest[:16], 16) % 10_000
    split = 'train' if bucket < 9900 else 'val' if bucket < 9950 else 'test'
    return text, digest, split, len(content)


def read_source_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require(manifest.get('dataset') == 'HuggingFaceFW/fineweb-edu'
            and manifest.get('config') == 'sample-10BT', 'Use the declared official FineWeb-Edu sample-10BT pool')
    revision = manifest.get('revision')
    require(isinstance(revision, str) and len(revision) == 40 and all(c in '0123456789abcdef' for c in revision),
            'Pin a full immutable lowercase hexadecimal dataset revision')
    require(isinstance(manifest.get('files'), list) and manifest['files'], 'No local Parquet source files declared')
    sources, seen = [], set()
    for record in manifest['files']:
        source = Path(record['path']).expanduser().resolve()
        require(source not in seen, 'Source list repeats a file')
        require(source.suffix == '.parquet' and source.is_file(), f'Missing local Parquet: {source}')
        require(type(record.get('bytes')) is int and record['bytes'] > 0
                and source.stat().st_size == record['bytes'], f'Source byte count differs: {source}')
        digest = record.get('sha256', '')
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Source SHA256 is required')
        seen.add(source)
        sources.append(dict(record, path=str(source)))
    return manifest, sources


def parquet_documents(sources, cursor=(0, 0), batch_rows=256):
    """Yield bounded Arrow batches; cursor identifies the next unconsumed row."""
    import pyarrow.parquet as pq
    first_file, first_row = cursor
    for file_index, source in enumerate(sources):
        if file_index < first_file:
            continue
        parquet = pq.ParquetFile(source['path'])
        require('text' in parquet.schema_arrow.names, 'Parquet source lacks text column')
        row_number = 0
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=['text'], use_threads=False):
            if file_index == first_file and row_number + len(batch) <= first_row:
                row_number += len(batch)
                continue
            for text in batch.column(0).to_pylist():
                row = row_number
                row_number += 1
                if file_index == first_file and row < first_row:
                    continue
                yield file_index, row, text


def tokenizer_examples(sources, settings, selected_path, output, stop_file=None):
    seen, documents, byte_count = set(), 0, 0
    with Path(selected_path).open('w') as selected:
        for file_index, row, raw in parquet_documents(sources, batch_rows=settings['record_batch_rows']):
            if (row % settings['record_batch_rows']) == 0:
                check_resources(output, settings['reserve_bytes'], stop_file)
            text, digest, split, size = document_identity(raw)
            if not text or split != 'train' or digest in seen:
                continue
            if byte_count + size > settings['tokenizer_max_bytes']:
                break
            seen.add(digest)
            selected.write(json.dumps(dict(sha256=digest, split=split, source_file=file_index, row=row, utf8_bytes=size)) + '\n')
            documents += 1
            byte_count += size
            if documents % 5000 == 0:
                report(output, 'tokenizer', tokenizer_docs=documents, tokenizer_utf8_bytes=byte_count)
            yield text
            if documents >= settings['tokenizer_documents']:
                break
        require(documents > 0, 'No training document fits the tokenizer selection limits')
        selected.flush()
        os.fsync(selected.fileno())


def prepare_tokenizer(sources, settings, stage, output, stop_file=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tokenizer_path, marker = stage / 'tokenizer.json', stage / 'tokenizer-metadata.json'
    if marker.exists():
        metadata = json.loads(marker.read_text())
        require(sha256(tokenizer_path) == metadata['sha256'], 'Saved tokenizer differs from its completed record')
        require(sha256(stage / 'tokenizer-documents.jsonl') == metadata['documents_sha256'], 'Saved tokenizer selection differs')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.encode_special_tokens = True
        return tokenizer, metadata
    # The bounded tokenizer fit is deterministic and restarted if interrupted.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=settings['vocab_size'], min_frequency=2, show_progress=False,
                                  special_tokens=[EOS], initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()))
    selection = stage / 'tokenizer-documents.jsonl.tmp'
    report(output, 'tokenizer', tokenizer_docs=0, tokenizer_utf8_bytes=0)
    tokenizer.train_from_iterator(tokenizer_examples(sources, settings, selection, output, stop_file), trainer=trainer)
    require(tokenizer.get_vocab_size() == settings['vocab_size'] and tokenizer.token_to_id(EOS) == 0,
            'Tokenizer vocabulary or EOS identity differs')
    tokenizer.encode_special_tokens = True
    temporary = stage / 'tokenizer.json.tmp'
    tokenizer.save(str(temporary))
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(tokenizer_path)
    selection.replace(stage / 'tokenizer-documents.jsonl')
    fsync_directory(stage)
    count, byte_count = 0, 0
    with (stage / 'tokenizer-documents.jsonl').open() as stream:
        for line in stream:
            count += 1
            byte_count += json.loads(line)['utf8_bytes']
    metadata = dict(sha256=sha256(tokenizer_path), documents_sha256=sha256(stage / 'tokenizer-documents.jsonl'),
                    documents=count, utf8_bytes=byte_count,
                    split='train', selection='First unique training-split documents in pinned source/row order, stopping before byte cap or at document cap; whole documents only.',
                    maximum_documents=settings['tokenizer_documents'], maximum_utf8_bytes=settings['tokenizer_max_bytes'])
    atomic_json(marker, metadata)
    return tokenizer, metadata


def open_database(stage, settings):
    database = sqlite3.connect(stage / 'documents.sqlite')
    database.execute('PRAGMA journal_mode=DELETE')
    database.execute('PRAGMA synchronous=FULL')
    database.execute('CREATE TABLE IF NOT EXISTS documents (sha256 TEXT PRIMARY KEY, split TEXT NOT NULL, source_file INTEGER NOT NULL, row_number INTEGER NOT NULL, utf8_bytes INTEGER NOT NULL, selected INTEGER NOT NULL DEFAULT 0, tokens INTEGER)')
    database.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    row = database.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
    if row is None:
        state = dict(cursor=[0, 0], seen_records=0, unique_documents=0, duplicates=0, empty_documents=0,
                     maximum_document_utf8_bytes=0, peak_encode_batch_utf8_bytes=0,
                     splits={split: dict(tokens=0, documents=0) for split in SPLITS}, complete=False)
        for split in SPLITS:
            with (stage / f'{split}.bin').open('wb') as stream:
                stream.flush()
                os.fsync(stream.fileno())
            with (stage / f'{split}.boundaries.u64').open('wb') as stream:
                stream.write((0).to_bytes(8, 'little'))
                stream.flush()
                os.fsync(stream.fileno())
        fsync_directory(stage)
        database.execute('INSERT INTO metadata VALUES (?,?)', ('state', json.dumps(state)))
        database.commit()
    else:
        state = json.loads(row[0])
    # SQLite is authoritative: bytes after its last fsync-backed transaction are
    # discarded even if a process died before updating the progress mirror.
    for split in SPLITS:
        for filename, committed_bytes in ((f'{split}.bin', state['splits'][split]['tokens'] * 2),
                                          (f'{split}.boundaries.u64', (state['splits'][split]['documents'] + 1) * 8)):
            path = stage / filename
            require(path.exists() and path.stat().st_size >= committed_bytes, f'Committed token file is short: {path}')
            with path.open('r+b') as stream:
                stream.truncate(committed_bytes)
    return database, state


def completed(state, settings):
    return all(state['splits'][split]['tokens'] >= settings['targets'][split] for split in SPLITS)


def encode(sources, settings, tokenizer, stage, output, stop_file=None, max_transactions=None):
    import numpy as np
    database, state = open_database(stage, settings)
    handles = {name: (stage / name).open('ab') for split in SPLITS for name in (f'{split}.bin', f'{split}.boundaries.u64')}
    transactions = 0
    try:
        iterator = iter(parquet_documents(sources, tuple(state['cursor']), settings['record_batch_rows']))
        while not completed(state, settings):
            check_resources(output, settings['reserve_bytes'], stop_file)
            documents, exhausted, batch_bytes = [], False, 0
            for _ in range(settings['encode_batch_documents']):
                try:
                    file_index, row, raw = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                state['cursor'] = [file_index, row + 1]
                state['seen_records'] += 1
                text, digest, split, size = document_identity(raw)
                state['maximum_document_utf8_bytes'] = max(state['maximum_document_utf8_bytes'], size)
                if not text:
                    state['empty_documents'] += 1
                    continue
                try:
                    database.execute('INSERT INTO documents(sha256,split,source_file,row_number,utf8_bytes) VALUES (?,?,?,?,?)',
                                     (digest, split, file_index, row, size))
                except sqlite3.IntegrityError:
                    state['duplicates'] += 1
                    continue
                state['unique_documents'] += 1
                if state['splits'][split]['tokens'] < settings['targets'][split]:
                    documents.append((text, digest, split))
                    batch_bytes += size
                    if batch_bytes >= settings['encode_batch_bytes']:
                        break
            state['peak_encode_batch_utf8_bytes'] = max(state['peak_encode_batch_utf8_bytes'], batch_bytes)
            encoded = tokenizer.encode_batch([doc[0] for doc in documents], add_special_tokens=False)
            pending = sum((len(value.ids) + 1) * 2 + 8 for value in encoded)
            check_resources(output, settings['reserve_bytes'], stop_file, pending)
            for (_, digest, split), value in zip(documents, encoded):
                if state['splits'][split]['tokens'] >= settings['targets'][split]:
                    continue
                ids = value.ids
                require(all(0 <= token < settings['vocab_size'] for token in ids), 'Token ID outside vocabulary')
                # Text matching a special EOS spelling must remain content, not
                # an injected boundary. FineWeb source itself is not trusted.
                require(0 not in ids, 'A source document contains the reserved EOS token; use an explicit recorded policy before proceeding')
                array = np.asarray(ids + [0], dtype='<u2')
                handles[f'{split}.bin'].write(array.tobytes())
                state['splits'][split]['tokens'] += len(array)
                state['splits'][split]['documents'] += 1
                handles[f'{split}.boundaries.u64'].write(state['splits'][split]['tokens'].to_bytes(8, 'little'))
                database.execute('UPDATE documents SET selected=1,tokens=? WHERE sha256=?', (len(array), digest))
            for stream in handles.values():
                stream.flush()
                os.fsync(stream.fileno())
            state['complete'] = completed(state, settings)
            database.execute("UPDATE metadata SET value=? WHERE key='state'", (json.dumps(state),))
            database.commit()
            atomic_json(stage / 'encoding-state.json', state)
            transactions += 1
            report(output, 'encoding', **state, free_gib=shutil.disk_usage(output).free / 2**30)
            if state['complete']:
                return state
            if exhausted:
                raise RuntimeError('Pinned source pool exhausted before all whole-document token targets were met')
            if max_transactions is not None and transactions >= max_transactions:
                raise PreparationPaused('Bounded fixture stop after durable transaction')
        return state
    finally:
        database.rollback()
        for stream in handles.values():
            stream.close()
        database.close()


def build_completion(sources, manifest, settings, tokenizer_metadata, state, stage, output):
    import numpy as np
    import tokenizers
    check_resources(output, settings['reserve_bytes'])
    require(state['complete'], 'Cannot publish unfinished corpus')
    splits = {}
    for split in SPLITS:
        raw = np.memmap(stage / f'{split}.boundaries.u64', mode='r', dtype='<u8')
        tokens = np.memmap(stage / f'{split}.bin', mode='r', dtype='<u2')
        require(len(raw) == state['splits'][split]['documents'] + 1 and raw[0] == 0
                and int(raw[-1]) == len(tokens) == state['splits'][split]['tokens'], 'Boundary/token totals differ')
        require(bool(np.all(raw[1:] > raw[:-1])) and bool(np.all(tokens[raw[1:].astype(np.int64) - 1] == 0)), 'Whole-document EOS boundaries differ')
        for start in range(0, len(tokens), 1_000_000):
            require(int(tokens[start:start + 1_000_000].max()) < settings['vocab_size'], 'Published token ID exceeds vocabulary')
        boundary_path = stage / f'{split}.boundaries.npy'
        with boundary_path.open('wb') as stream:
            np.save(stream, raw, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        splits[split] = dict(file=f'{split}.bin', tokens=len(tokens), stories=state['splits'][split]['documents'],
                             documents=state['splits'][split]['documents'], requested_tokens=settings['targets'][split],
                             bytes=(stage / f'{split}.bin').stat().st_size, sha256=sha256(stage / f'{split}.bin'),
                             boundaries_file=boundary_path.name, boundaries_sha256=sha256(boundary_path),
                             selection='First distinct documents in pinned source order assigned to this content-hash split, until token target is reached; final document retained whole.')
        del raw, tokens
    database = sqlite3.connect(stage / 'documents.sqlite')
    try:
        require(database.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Document manifest failed SQLite integrity check')
        for split in SPLITS:
            count, token_count = database.execute('SELECT COUNT(*),COALESCE(SUM(tokens),0) FROM documents WHERE split=? AND selected=1', (split,)).fetchone()
            require(count == splits[split]['documents'] and token_count == splits[split]['tokens'], 'Selected-document manifest differs from token streams')
        with (stage / 'tokenizer-documents.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                row = database.execute('SELECT split,selected FROM documents WHERE sha256=?', (record['sha256'],)).fetchone()
                require(row == ('train', 1), 'Tokenizer saw a document outside the selected training corpus')
    finally:
        database.close()
    for source in sources:
        require(sha256(source['path']) == source['sha256'], 'Source bytes changed during preparation')
    metadata = dict(schema_version=2, schema=SCHEMA, status='ready', preparation_settings=settings,
                    dataset=manifest['dataset'], revision=manifest['revision'], variant=manifest['config'],
                    source_manifest_sha256=settings['source_manifest_sha256'], source_files=sources,
                    license=manifest.get('license'), attribution=manifest.get('attribution'),
                    dtype='uint16', byte_order='little', vocab_size=settings['vocab_size'], eos_token=EOS, eos_token_id=0,
                    tokenizer_sha256=tokenizer_metadata['sha256'], tokenizer_library_version=tokenizers.__version__,
                    tokenizer_training=tokenizer_metadata,
                    tokenizer_content_policy='encode_special_tokens=True: literal reserved-token spellings in source text are encoded as ordinary content bytes; one EOS0 is appended explicitly per document.',
                    train_tokens=splits['train']['tokens'], val_tokens=splits['val']['tokens'], test_tokens=splits['test']['tokens'],
                    **splits, packing='One EOS per complete document; concatenated bins allow native attention windows to cross document boundaries.',
                    split_policy='SHA256 of outer-whitespace-stripped exact UTF8 content; first64 digest bits modulo10000: [0,9900) train, [9900,9950) val, [9950,10000) test.',
                    document_manifest=dict(file='documents.sqlite', sha256=sha256(stage / 'documents.sqlite'),
                        scope='Every unique nonempty document in the consumed source prefix, including split-full documents not selected; duplicate occurrences counted in consumed_pool.'),
                    tokenizer_document_manifest=dict(file='tokenizer-documents.jsonl', sha256=tokenizer_metadata['documents_sha256']),
                    consumed_pool=state, preparation_source_sha256=settings['preparation_source_sha256'],
                    limitations=['Hash splits prevent exact-content copies crossing splits, including copies elsewhere in the source pool; near/semantic duplicates are not removed.',
                        'Training/validation/test token targets are lower bounds because the final document is never truncated.',
                        'Source-order selection is a deterministic prefix within each split, not a uniform document sample of the entire10BT pool.',
                        'Test documents are prepared but must not be used to tune architecture or training choices.'])
    atomic_json(stage / 'completion.json', metadata)
    return metadata


def publish(stage, output, metadata):
    artifacts = {record['file']: record['sha256'] for record in (metadata[split] for split in SPLITS)}
    artifacts.update({metadata[split]['boundaries_file']: metadata[split]['boundaries_sha256'] for split in SPLITS})
    artifacts.update({'tokenizer.json': metadata['tokenizer_sha256'],
                      'documents.sqlite': metadata['document_manifest']['sha256'],
                      'tokenizer-documents.jsonl': metadata['tokenizer_document_manifest']['sha256']})
    for name, digest in artifacts.items():
        source, destination = stage / name, output / name
        if destination.exists():
            require(sha256(destination) == digest, f'Interrupted publication contains changed artifact: {destination}')
        else:
            require(source.exists() and sha256(source) == digest, f'Staging artifact differs: {source}')
            source.replace(destination)
            fsync_directory(output)
            fsync_directory(stage)
    # Completion marker is always last; partial publication cannot appear ready.
    atomic_json(output / 'metadata.json', metadata)
    report(output, 'complete', splits=metadata['consumed_pool']['splits'], metadata_sha256=sha256(output / 'metadata.json'),
           source_manifest_sha256=metadata['source_manifest_sha256'])


@contextmanager
def preparation_lock(output):
    with (output / '.preparation.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another dataset preparer owns this output directory') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(source_manifest, output_dir, *, resume=False, train_tokens=1_000_000_000,
            val_tokens=2_000_000, test_tokens=2_000_000, vocab_size=8192,
            tokenizer_documents=100_000, tokenizer_max_bytes=100_000_000,
            record_batch_rows=256, encode_batch_documents=128, encode_batch_bytes=4 * 2**20, reserve_bytes=12 * 2**30,
            stop_file=None, max_transactions=None):
    import numpy
    import pyarrow
    import tokenizers
    require(all(type(value) is int and value > 0 for value in (train_tokens, val_tokens, test_tokens,
            tokenizer_documents, tokenizer_max_bytes, record_batch_rows, encode_batch_documents, encode_batch_bytes)), 'Budgets and batch sizes must be positive integers')
    require(type(vocab_size) is int and 257 <= vocab_size <= 65536, 'Vocabulary must fit uint16 including byte alphabet/EOS')
    require(type(reserve_bytes) is int and reserve_bytes >= 0, 'Disk reserve must be a nonnegative byte count')
    source_manifest, output = Path(source_manifest).resolve(), Path(output_dir).resolve()
    manifest, sources = read_source_manifest(source_manifest)
    output.mkdir(parents=True, exist_ok=True)
    with preparation_lock(output):
        settings = dict(schema=SCHEMA, source_manifest_sha256=sha256(source_manifest),
                        preparation_source_sha256=sha256(__file__), targets=dict(train=train_tokens, val=val_tokens, test=test_tokens),
                        library_versions=dict(numpy=numpy.__version__, pyarrow=pyarrow.__version__,
                                              tokenizers=tokenizers.__version__, sqlite=sqlite3.sqlite_version),
                        vocab_size=vocab_size, tokenizer_documents=tokenizer_documents, tokenizer_max_bytes=tokenizer_max_bytes,
                        record_batch_rows=record_batch_rows, encode_batch_documents=encode_batch_documents,
                        encode_batch_bytes=encode_batch_bytes, reserve_bytes=reserve_bytes)
        stage = output / '.staging'
        settings_path = output / 'preparation-settings.json'
        if settings_path.exists():
            require(resume, 'Output already has preparation state; use --resume with identical settings')
            require(json.loads(settings_path.read_text()) == settings, 'Resume settings, source identity or preparer code changed')
        else:
            allowed_inputs = {'.preparation.lock', 'SOURCE-README.md', 'download-progress.json', 'download.log', 'raw'}
            if source_manifest.parent == output:
                allowed_inputs.add(source_manifest.name)
            require(not any(path.name not in allowed_inputs for path in output.iterdir()), 'Refuse an unrelated/nonempty output directory')
            atomic_json(settings_path, settings)
        stage.mkdir(exist_ok=True)
        try:
            for index, source in enumerate(sources):
                check_resources(output, reserve_bytes, stop_file)
                report(output, 'verifying_sources', file_index=index, file_count=len(sources), file=source['path'])
                require(sha256(source['path']) == source['sha256'], f'Parquet checksum differs: {source["path"]}')
            if (output / 'metadata.json').exists():
                metadata = json.loads((output / 'metadata.json').read_text())
                require(metadata['preparation_settings'] == settings, 'Completed corpus settings differ')
                publish(stage, output, metadata)
                return metadata
            if (stage / 'completion.json').exists():
                metadata = json.loads((stage / 'completion.json').read_text())
                require(metadata['preparation_settings'] == settings, 'Pending publication settings differ')
                publish(stage, output, metadata)
                return metadata
            tokenizer, tokenizer_metadata = prepare_tokenizer(sources, settings, stage, output, stop_file)
            state = encode(sources, settings, tokenizer, stage, output, stop_file, max_transactions)
            check_resources(output, reserve_bytes, stop_file)
            metadata = build_completion(sources, manifest, settings, tokenizer_metadata, state, stage, output)
            check_resources(output, reserve_bytes, stop_file)
            require(sha256(source_manifest) == settings['source_manifest_sha256']
                    and sha256(__file__) == settings['preparation_source_sha256'], 'Preparation source or manifest changed during encoding')
            publish(stage, output, metadata)
            return metadata
        except (PreparationPaused, KeyboardInterrupt) as exc:
            report(output, 'paused', reason=str(exc), resume='Use identical arguments plus --resume; encoding uses the last durable SQLite cursor, interrupted tokenizer fitting restarts deterministically.')
            raise
        except Exception as exc:
            report(output, 'failed', reason=str(exc), resume='Inspect failure; original sources and any committed encoding state are retained.')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-tokens', type=int, default=1_000_000_000)
    parser.add_argument('--val-tokens', type=int, default=2_000_000)
    parser.add_argument('--test-tokens', type=int, default=2_000_000)
    parser.add_argument('--vocab-size', type=int, default=8192)
    parser.add_argument('--tokenizer-documents', type=int, default=100_000)
    parser.add_argument('--tokenizer-max-bytes', type=int, default=100_000_000)
    parser.add_argument('--record-batch-rows', type=int, default=256)
    parser.add_argument('--encode-batch-documents', type=int, default=128)
    parser.add_argument('--encode-batch-bytes', type=int, default=4 * 2**20,
                        help='Soft UTF8 byte cap per encoding batch; retain the whole document that reaches it')
    parser.add_argument('--min-free-gib', type=float, default=12.)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    def stop(_signal, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    values = vars(args)
    values['resume'] = args.resume
    values['reserve_bytes'] = int(values.pop('min_free_gib') * 2**30)
    try:
        prepare(**values)
    except (PreparationPaused, KeyboardInterrupt):
        raise SystemExit(75)


if __name__ == '__main__':
    main()
