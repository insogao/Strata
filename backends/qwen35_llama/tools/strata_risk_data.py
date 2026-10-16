"""Read-only Jev registry importer. Optional PyYAML is needed only for import.

Never equate pipeline acceptance or model judges with human adjudication.
The database controls inclusion; YAML metadata is never sent to inference.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def import_jev(root):
    import yaml
    root = Path(root).resolve()
    db = root / 'crime_chat_dataset/registry/dataset_registry.sqlite'
    conn = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    # A transaction snapshot prevents mixing status revisions during import.
    conn.execute('BEGIN')
    try:
        registry = [dict(r) for r in conn.execute('SELECT * FROM cases ORDER BY case_id')]
        pairs = [dict(r) for r in conn.execute('SELECT * FROM pairs ORDER BY pair_id')]
    finally:
        conn.rollback(); conn.close()
    rows, excluded, details = [], Counter(), []
    counts, types = Counter(), Counter()
    family, lineage = Counter(), Counter()
    seen = set()
    for r in registry:
        counts[r['status']] += 1
        if r['status'] != 'accepted':
            excluded['registry_' + str(r['status'])] += 1
            continue
        p = (root / r['yaml_path']).resolve()
        if not p.is_relative_to(root):
            raise ValueError('registry path escapes source root')
        try:
            raw = p.read_bytes()
            d = yaml.safe_load(raw.decode('utf-8-sig'))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            excluded[type(exc).__name__] += 1
            details.append({'case_id': r['case_id'], 'reason': type(exc).__name__})
            continue
        if (not isinstance(d, dict) or d.get('pipeline_status') != 'accepted' or
                d.get('identity', {}).get('case_id') != r['case_id'] or
                type(d.get('labels', {}).get('risk')) is not int or d['labels']['risk'] != r['risk']):
            excluded['registry_yaml_mismatch'] += 1
            continue
        text = d.get('conversation')
        if not isinstance(text, str) or not text.strip():
            excluded['empty_conversation'] += 1
            continue
        normalized = hashlib.sha256(re.sub(r'\s+', '', text).encode()).hexdigest()
        if normalized in seen:
            excluded['duplicate_conversation'] += 1
            continue
        seen.add(normalized)
        source_type = d.get('source', {}).get('source_type', 'unknown')
        types[source_type] += 1
        family[r['family_id']] += 1; lineage[r['lineage_id']] += 1
        # Dataset origin alone cannot establish human adjudication, even if a
        # future source_type string happens to say "human". Explicit audit needed.
        rows.append({'sample_id': r['case_id'], 'conversation_id': normalized,
            'messages': [{'role': 'user', 'content': text}], 'gold_risk': r['risk'],
            'gold_status': 'provisional', 'reference_kind': 'synthetic_design_label' if str(source_type).startswith('synthetic') else 'unverified_reference',
            'scenario': r['category'], 'source_id': source_type,
            'label_policy_version': d.get('execution', {}).get('workflow_version', 'unknown'),
            'source_sha256': hashlib.sha256(raw).hexdigest(),
            'registry_status': r['status'], 'family_id': r['family_id'], 'lineage_id': r['lineage_id'],
            'source_lineage': d.get('lineage', {}), 'has_transfer': bool(r['has_transfer'])})
    # Conservative leakage grouping includes registered pairs and family/lineage.
    # Large reused placeholder families are surfaced, NOT silently ignored to
    # manufacture attractive independent test samples.
    parents = {r['sample_id']: r['sample_id'] for r in rows}
    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    def join(a, b):
        if a in parents and b in parents: parents[find(b)] = find(a)
    for p in pairs: join(p['case_a'], p['case_b'])
    identifiers = {}
    for r in rows:
        for field in ('family_id', 'lineage_id'):
            key = (field, r[field])
            if r[field]:
                join(identifiers.setdefault(key, r['sample_id']), r['sample_id'])
        for name in ('root_case_id', 'parent_case_id'):
            join(r['sample_id'], r['source_lineage'].get(name))
    for r in rows: r['leakage_group'] = digest(find(r['sample_id']))
    clusters = Counter(r['leakage_group'] for r in rows)
    summary = {'schema': 'strata-risk-import/1', 'registry_snapshot_sha256': digest(registry),
        'pair_snapshot_sha256': digest(pairs), 'source_manifest_sha256': digest([r['source_sha256'] for r in rows]),
        'registry_counts': dict(counts), 'usable': len(rows), 'excluded': dict(excluded),
        'reference_kind_counts': dict(Counter(r['reference_kind'] for r in rows)),
        'source_types': dict(types), 'risk_counts': dict(Counter(r['gold_risk'] for r in rows)),
        'scenario_counts': dict(Counter(r['scenario'] for r in rows)),
        'leakage_clusters': len(clusters), 'largest_cluster': max(clusters.values(), default=0),
        'family_sizes_top5': [n for _, n in family.most_common(5)],
        'lineage_sizes_top5': [n for _, n in lineage.most_common(5)],
        'gold_accuracy_status': 'BLOCKED_NO_GOLD_DATA',
        'note': 'Model-generated design labels; accepted is program/pipeline status, not proof of human adjudication. No formal holdout claimed.'}
    return rows, summary, details
