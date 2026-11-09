"""V2 source binding and deterministic JSON IO."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/baseline-1.5b-20261029-01'
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def source_files():
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),
        *sorted((ROOT/'configs').glob('*.json')),ROOT/'PROJECT_REQUIREMENTS.md',ROOT/'README.md',ROOT/'V1_ARCHIVE_RECEIPT.json']
    return {p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}
def verify_files(root,files):
    for name,digest in files.items():
        path=(Path(root)/name).resolve()
        if not path.is_relative_to(Path(root).resolve()) or file_hash(path)!=digest:
            raise ValueError('file identity differs: '+name)
