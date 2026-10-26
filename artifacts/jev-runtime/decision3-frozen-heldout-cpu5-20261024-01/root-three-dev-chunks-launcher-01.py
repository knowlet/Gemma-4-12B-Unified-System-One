"""Root serial selection of exactly three original-owner dev chunks; no training."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
HERE=Path(__file__).resolve().parent
OWNER=ROOT/'artifacts/jev-runtime/llama-cpp-owner-20261024-09'
CONTROLLER_SHA='81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee'
SELECTION_SHA='72e7ae7e54f57b27caa8dacf891cc630561de67230bb74ffa1c107890ff23536'

def ref(path):
    raw=path.read_bytes()
    assert len(raw)<=1048576
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def read(path):
    assert path.stat().st_size <=1048576 and not path.is_symlink()
    return json.loads(path.read_text())

def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def current_closure(path):
    v=read(path);p=path.parent;c=read(p/'sealed-config.json');a=read(p/'admission.json')
    assert v['closure_mode']=='authenticated_stop' and not v['closure_errors']
    for key in ('whole_unit_exit_confirmed','observed_cgroup_cohort_exit_confirmed','post_stop_terminal_verified','cgroup_absent','sources_unchanged'):assert v[key] is True
    for key in ('no_escape_enforced','whole_tree_exit_confirmed','continuation_authorized','model_qualified'):assert v[key] is False
    assert len(v['two_process_scans'])==2 and all(not s['matches'] and not s['unknown'] for s in v['two_process_scans'])
    assert c['nonce']==a['nonce']==v['selection']['nonce'] and a['invocation_id']==v['invocation_id'] and a['config_sha256']==ref(p/'sealed-config.json')['sha256']
    assert c['unit']==v['selection']['unit']=='jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service'
    return ref(path)

def main():
    selection_path=HERE/'root-three-dev-chunks-selection-01.json'
    assert ref(selection_path)['sha256']==SELECTION_SHA
    selected=read(selection_path);assert selected['status']=='SELECTED_BEFORE_ANY_DEV_MODEL_INFERENCE' and len(selected['calls'])==3 and selected['training_eligible'] is False
    assert ref(OWNER/'controller.py')['sha256']==CONTROLLER_SHA
    prior=OWNER/'actual-22/external-boundary-closure.json'
    old_review=read(ROOT/'artifacts/jev-runtime/decision3-delta-browser-worker-cpu5-20261024-01/root-closed-browser-actual22-review-01.json')
    assert ref(prior)==old_review['refs']['closure']
    results=[]
    for call in selected['calls']:
        number=call['actual'];assert number==22+call['index'] and number in (23,24,25)
        spec_path=Path(call['spec']['path']);assert ref(spec_path)=={**call['spec'],'path':str(spec_path.relative_to(ROOT))}
        prior_ref=current_closure(prior)
        spec=read(spec_path)
        for row in spec['sources']:assert ref(ROOT/row['path'])==row
        for row in spec['binary_stats']:
            st=Path(row['path']).stat();assert all(getattr(st,k)==v for k,v in row['stat'].items())
        output=OWNER/f'actual-{number:02d}';terminal=OWNER/f'root-execution-dev-collector-{number:02d}.json';intent=HERE/f'root-intent-dev-collector-{number:02d}.json'
        assert not any(p.exists() or p.is_symlink() for p in (output,terminal,intent))
        argv=[str(ROOT/'.venv/bin/python'),'-B','-s',str(OWNER/'controller.py'),'--run','--expected-entry-sha256',CONTROLLER_SHA,'--selection',str(spec_path),'--selection-sha256',call['spec']['sha256'],'--output',str(output)]
        record={'profile':'root-selected-frozen-dev-cpu5-terminal-v1','index':call['index'],'ancestor_id':call['ancestor'],'argv':argv,'selection_sha256':call['spec']['sha256'],'launcher_source':ref(Path(__file__)),'fixed_three_chunk_selection':ref(selection_path),'previous_original_scoped_closure':prior_ref,'training_eligible':False,'labels_selected':False,'fresh_original_owner_admission_and_current_closure_required':True}
        save(intent,record);started=datetime.datetime.now(datetime.timezone.utc).isoformat();began=time.monotonic();result=subprocess.run(argv,cwd=ROOT,capture_output=True,check=False)
        record.update(started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-began,exit_code=result.returncode,stdout=result.stdout.decode('utf-8',errors='replace'),stderr=result.stderr.decode('utf-8',errors='replace'));save(terminal,record)
        row={'index':call['index'],'actual':number,'exit_code':result.returncode,'wall_seconds':record['wall_seconds'],'root_terminal':ref(terminal)};results.append(row);print(json.dumps(row),flush=True)
        if result.returncode:
            save(HERE/'root-three-dev-chunks-failure-01.json',{'completed':False,'results':results,'training_eligible':False});return result.returncode
        receipt=read(output/'receipt.json');assert receipt['completed'] is True and receipt['primary_error'] is None and receipt['secondary_errors']==[] and receipt['sources_unchanged'] is True
        prior=output/'external-boundary-closure.json';current_closure(prior)
    save(HERE/'root-three-dev-chunks-execution-01.json',{'completed':True,'dev_states':24,'same_forward_rows':96,'results':results,'root_closed_payload_audit_and_complete_dev_metrics_pending':True,'training_eligible':False});return 0

if __name__=='__main__':raise SystemExit(main())
