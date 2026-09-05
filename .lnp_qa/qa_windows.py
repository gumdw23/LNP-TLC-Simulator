from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
REPORT = Path('windows_qa_report.json').resolve()
SERVER_OUT = Path('windows_streamlit_server.log').resolve()
SERVER_ERR = Path('windows_streamlit_server.err.log').resolve()

checks: list[dict] = []

def add(name: str, ok: bool, detail: object = '') -> None:
    checks.append({'name': name, 'ok': bool(ok), 'detail': str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_report(extra: dict | None = None) -> None:
    payload = {
        'package_root': ROOT.name,
        'platform': platform.platform(),
        'python': sys.version,
        'checks': checks,
        'pass': all(x['ok'] for x in checks),
    }
    if extra:
        payload.update(extra)
    REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

try:
    add('runtime_tree_exists', ROOT.is_dir(), ROOT)
    manifest_path = ROOT / 'PACKAGE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_files = manifest['files']
    mismatches = []
    for rel, expected in expected_files.items():
        p = ROOT / Path(rel)
        if not p.is_file():
            mismatches.append(f'missing:{rel}')
        else:
            got = sha256(p)
            if got.lower() != str(expected).lower():
                mismatches.append(f'hash:{rel}:{got}')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'PACKAGE_MANIFEST.json'}
    unexpected = sorted(actual - set(expected_files))
    missing_from_tree = sorted(set(expected_files) - actual)
    add('manifest_158_of_158', len(expected_files) == 158 and not mismatches and not unexpected and not missing_from_tree,
        f"entries={len(expected_files)} mismatches={len(mismatches)} unexpected={len(unexpected)} missing={len(missing_from_tree)}")

    import rdkit
    import streamlit
    import pandas
    import numpy
    import networkx
    import sqlalchemy

    versions = {
        'rdkit': getattr(rdkit, '__version__', ''),
        'streamlit': getattr(streamlit, '__version__', ''),
        'pandas': getattr(pandas, '__version__', ''),
        'numpy': getattr(numpy, '__version__', ''),
        'networkx': getattr(networkx, '__version__', ''),
        'sqlalchemy': getattr(sqlalchemy, '__version__', ''),
    }
    expected_versions = {
        'rdkit': '2025.09.4',
        'streamlit': '1.49.1',
        'pandas': '2.2.3',
        'numpy': '2.3.5',
        'networkx': '3.6.1',
        'sqlalchemy': '2.0.50',
    }
    add('pinned_versions', versions == expected_versions, versions)
    add('python_3_12_native', sys.version_info[:2] == (3, 12), sys.version.split()[0])
    add('windows_native', sys.platform == 'win32', platform.platform())

    compile_result = subprocess.run(
        [sys.executable, '-m', 'compileall', '-q', str(ROOT)], capture_output=True, text=True
    )
    add('compileall', compile_result.returncode == 0, (compile_result.stdout + compile_result.stderr)[-2000:])

    from rdkit import Chem
    mol = Chem.MolFromSmiles('NCCNC(=O)NCCN')
    add('rdkit_native_import_and_parse', mol is not None and mol.GetNumAtoms() > 0, f'atoms={mol.GetNumAtoms() if mol else 0}')

    data_dir = Path(tempfile.mkdtemp(prefix='lnp_v100_v3ux_windows_qa_')).resolve()
    env = os.environ.copy()
    env['LNP_LIPID_LIBRARY_DATA_DIR'] = str(data_dir)
    env['PYTHONUTF8'] = '1'
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    port = 8765
    with SERVER_OUT.open('w', encoding='utf-8') as out, SERVER_ERR.open('w', encoding='utf-8') as err:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
             '--server.headless=true', f'--server.port={port}', '--browser.gatherUsageStats=false'],
            cwd=str(ROOT), env=env, stdout=out, stderr=err,
        )
        healthy = False
        health_detail = ''
        try:
            for _ in range(90):
                if proc.poll() is not None:
                    health_detail = f'server exited code={proc.returncode}'
                    break
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as r:
                        body = r.read().decode('utf-8', errors='replace')
                        if r.status == 200:
                            healthy = True
                            health_detail = f'status={r.status} body={body[:200]}'
                            break
                except Exception as exc:
                    health_detail = repr(exc)
                time.sleep(1)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    add('streamlit_health_endpoint', healthy, health_detail)

    os.environ['LNP_LIPID_LIBRARY_DATA_DIR'] = str(data_dir / 'apptest')
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120)
    at.run()
    initial_ex = [str(x) for x in at.exception]
    add('apptest_initial_boot', not initial_ex, initial_ex)

    stages = {
        'Project': 'Dashboard',
        'Library': 'My Library Workflow',
        'Novelty': 'Known-compound Novelty Gate',
        'Synthesis': 'Synthesis Planning',
        'Experiment': 'Experimental Chemistry Notebook',
        'Analyze': 'Campaign Results',
        'Next': 'Next Batch Review',
    }
    stage_results = {}
    for stage, page in stages.items():
        try:
            at.session_state['_lnp_ui_mode'] = 'Research'
            at.session_state['_lnp_research_stage'] = stage
            at.session_state['_lnp_research_page'] = page
            at.run()
            ex = [str(x) for x in at.exception]
            buttons = []
            try:
                buttons = [getattr(b, 'label', '') for b in at.button]
            except Exception:
                pass
            stage_results[stage] = {'page': page, 'exceptions': ex, 'buttons': buttons}
            add(f'apptest_stage_{stage.lower()}', not ex, f'page={page} exceptions={ex} buttons={buttons[-12:]}')
        except Exception as exc:
            stage_results[stage] = {'page': page, 'harness_error': repr(exc)}
            add(f'apptest_stage_{stage.lower()}', False, repr(exc))

    synthesis_buttons = stage_results.get('Synthesis', {}).get('buttons', [])
    add('planning_next_cta_present_once', synthesis_buttons.count('다음 단계 · 합성 실험') == 1,
        f"count={synthesis_buttons.count('다음 단계 · 합성 실험')}")

    write_report({'versions': versions, 'stage_results': stage_results, 'data_dir': str(data_dir)})
except Exception as exc:
    add('qa_harness_unhandled_exception', False, repr(exc))
    write_report({'fatal': repr(exc)})

if not all(x['ok'] for x in checks):
    raise SystemExit(1)
print('ALL WINDOWS NATIVE QA CHECKS PASS')
