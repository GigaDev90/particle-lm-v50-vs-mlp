# Dataset preparation

The recorded experiment uses a deterministic local subset of **HuggingFaceFW/fineweb-edu**, configuration `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. FineWeb-Edu is provided by Hugging Face under ODC-BY; [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md), [paper](https://arxiv.org/abs/2406.17557).

The source card links the [ODC-By 1.0 license](https://opendatacommons.org/licenses/by/1-0/) and [Common Crawl terms of use](https://commoncrawl.org/terms-of-use).

Run these commands from the repository root:

```bash
python scripts/download_fineweb_sources.py --manifest data/source-manifest.json
python scripts/prepare_fineweb_edu.py \
  --source-manifest data/source-manifest.json \
  --output-dir data/fineweb-edu-1b-v1
python scripts/verify_data.py --data-dir data/fineweb-edu-1b-v1
```

The downloader retrieves two pinned Parquet shards (approximately 4.31 GB total) and checks their exact sizes and SHA256 digests. Token files require another approximately 2.01 GB plus intermediate preparation state. The scripts require a 12 GiB free-disk reserve by default. No downloaded source text or generated token files are committed to this repository.

Preparation strips only outer whitespace and assigns each exact UTF-8 document to a split using the first 64 bits of its SHA256 digest modulo 10,000: 0–9899 train, 9900–9949 validation, 9950–9999 test. Exact-content duplicates are removed. Near or semantic duplicates are not removed.

An 8,192-entry byte-level BPE is fitted only on training-assigned documents, capped at 100,000 documents or 100,000,000 UTF-8 bytes. Its exported reference is [tokenizer.json](../data/tokenizer.json). The script independently rebuilds it; verify the resulting hashes before claiming the same tokenization. Reproducibility depends on the recorded preparation-library versions in requirements.txt.

Documents are encoded in pinned source order until each split reaches its token target, retaining the last document whole. The resulting counts are:

| Split | Tokens | Documents |
|---|---:|---:|
| Train |1,000,000,163|815,627|
| Validation |2,000,383|1,554|
| Reserved test |2,001,913|1,660|

Every document receives one EOS token. Concatenated training windows may cross document boundaries. The selection is a deterministic prefix within each content-hash split, **not** a uniformly drawn sample of all FineWeb-Edu documents. The reserved test split is prepared but unused in the published development comparison.

The native trainers sample 512-token windows with replacement from train.bin, using the same seed and microbatch order for both models. Each validation call uses 131,072 targets drawn with a separate fixed seed 12345. A training-token budget therefore measures processed targets rather than one exhaustive pass through the corpus.

The [portable checksums](../data/checksums.json) verify tokenizer and token-file contents. Metadata/source-manifest digests change when local paths change; they are not expected to equal the original workspace metadata hashes. Interruptions can be resumed by repeating the same preparation command with `--resume`.
