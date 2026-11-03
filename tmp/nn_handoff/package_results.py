import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
handoff = ROOT / 'results' / 'NN_handoff_2026-11-02'
audit = json.loads((ROOT / 'tmp/nn_handoff/audit_data.json').read_text(encoding='utf-8'))
assert audit['passed'] and audit['validated_experiments'] == 102
raw = handoff / 'raw_reports'
raw.mkdir(exist_ok=True)
for item in audit['source_manifest']:
    source = ROOT / 'results/neural_network_skorch' / item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = raw / source.name
    shutil.copy2(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']

sources = [
    'src/network/network_train_skorch.py', 'src/network/reproduce_split.py',
    'src/network/check_results.py', 'src/network/requirements_skorch.txt',
    'src/network/README_skorch.md', 'src/datasets.py', 'src/split.py',
    'src/preprocessing.py', 'src/dataset_registry.py', 'data/dataset_registry.csv',
]
for name in sources:
    target = handoff / 'reproducibility' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
environment = {'description': 'Versions at result-check time; split versions are also in each original JSON.',
               'python': sys.version, 'packages': {name: importlib.metadata.version(name)
               for name in ['torch', 'skorch', 'numpy', 'pandas', 'scipy', 'scikit-learn']}}
(handoff / 'environment_checked.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')

# 验证汇总 CSV 仍然逐条对应原报告，避免交接文件产生单位/精度变化。
import pandas as pd
summary = pd.read_csv(handoff / 'summary_metrics.csv')
assert len(summary) == 102
for _, row in summary.iterrows():
    report = json.loads((raw / row['source_report']).read_text(encoding='utf-8'))
    for metric in ['accuracy', 'macro_f1', 'g_mean', 'ba']:
        assert abs(float(row[metric]) - report['combined_test_metrics'][metric]) < 1e-12
assert len(list(raw.glob('*.json'))) == 102
assert len(list(raw.glob('*.csv'))) == 102

destination = handoff.with_suffix('.zip')
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for file in sorted(handoff.rglob('*')):
        if file.is_file():
            archive.write(file, Path(handoff.name) / file.relative_to(handoff))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    assert len([name for name in archive.namelist() if '/raw_reports/' in name]) == 204
print(json.dumps({'zip':str(destination), 'bytes':destination.stat().st_size,
                  'files':len(list(handoff.rglob('*.*'))), 'raw_reports':204,
                  'summary_rows':102, 'validated':True}, indent=2))
