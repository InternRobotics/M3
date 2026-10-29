"""Download an explicitly selected release weight; no implicit latest model."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='cskrren/M3')
    parser.add_argument('--filename', required=True)
    parser.add_argument('--revision', required=True,
                        help='Published immutable commit revision (recommended)')
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{64}', args.sha256):
        parser.error('--sha256 must contain exactly 64 hexadecimal characters')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')

    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=args.repo_id, filename=args.filename,
                             revision=args.revision)
    digest = hashlib.sha256()
    with open(cached, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        if digest.hexdigest() != args.sha256.lower():
            raise RuntimeError('SHA-256 mismatch; no output checkpoint created')
        source.seek(0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as target:
            shutil.copyfileobj(source, target)
    print(f'Verified {args.output}: sha256={digest.hexdigest()}')


if __name__ == '__main__':
    main()
