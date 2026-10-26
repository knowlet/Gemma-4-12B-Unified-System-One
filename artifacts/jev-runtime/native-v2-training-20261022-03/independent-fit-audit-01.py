import builtins,collections,hashlib,importlib.util,json,math,stat,struct,sys,time
from pathlib import Path
R=Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One');D=R/'artifacts/jev-runtime/native-v2-training-20261022-03';F=D/'fit-01';started=time.monotonic();cache={}
def need(v,label):
 if not v:raise AssertionError(label)
def read(p):
 need(p.stat().st_size<64*1024**2,'Small metadata only');return json.loads(p.read_bytes())
def pin(p):
 p=p.resolve();s=p.stat();need(stat.S_ISREG(s.st_mode) and s.st_size<64*1024**2,'Small source/adapter/capture only; no base checkpoint payload')
 key=(str(p),s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if key not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for z in iter(lambda:f.read(1024**2),b''):h.update(z)
  after=p.stat();need((s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Stable small bytes');cache[key]={'sha256':h.hexdigest(),'size_bytes':s.st_size}
 return cache[key]
def ref(p):return {'path':str(p.relative_to(R)),**pin(p)}
def rows(n):return [json.loads(x) for x in (F/n).read_bytes().splitlines() if x.strip()]
original_import=builtins.__import__
def checked_import(name,*a,**kw):
 if name.split('.')[0] in {'torch','transformers','peft','safetensors','tokenizers','modal','bitsandbytes'}:raise AssertionError('No ML imports: '+name)
 return original_import(name,*a,**kw)
builtins.__import__=checked_import
p=read(D/'launch-plan-02.json');need(pin(D/'launch-plan-02.json')['sha256']=='41435aa11dca758f91fab77cc739a5595dfe2d20ac37f62d28a98a837bb76a48','Exact selected N3 plan')
r=read(F/'receipt.json');m=read(F/'adapter/manifest.json');t=read(F/'training.json');c=read(D/'root-external-closure-01.json');strict=read(D/'root-strict-artifact-verification-01.json')
need(r['stage']=='complete' and r['steps']==r['training_updates']==142 and r['qualified'] is False,'Fresh complete142 fitting only')
need(t==r['training'] and t['completed_updates']==142 and t['development_examples']==37 and t['development_updates']==0,'Exact teacher-forced142/37 evidence')
need(r['training_quantization']=='none' and r['training_dtype']=='bfloat16' and r['deployment_quantization']=='nf4','Declared precision')
need(r['full_lm_head_unchanged'] is True and t['original_head_sha256_before']==t['original_head_sha256_after']=='9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4' and t['frozen_base_versions_unchanged'] is True,'Unchanged original head/base recorded proof')
need(m['training_receipt']['path']=='../receipt.json' and pin(F/'receipt.json')['sha256']==m['training_receipt']['sha256'],'Manifest receipt binding')
need(m['files']==r['adapter_files'],'Adapter recorded pins equality')
for n,v in m['files'].items():need(pin(F/'adapter'/n)==v,'Complete adapter payload hash')
config=read(F/'adapter/adapter_config.json')
for k,v in r['lora'].items():need(config[k]==v,'Exact saved LoRA configuration')
weight=F/'adapter/adapter_model.safetensors'
with weight.open('rb') as f:
 length=struct.unpack('<Q',f.read(8))[0];need(length<1024**2,'Bounded safetensors header');header=json.loads(f.read(length))
tensors={k:v for k,v in header.items() if k!='__metadata__'};need(len(tensors)==176,'Exact176 LoRA tensors');offsets=[];params=0
for name,v in tensors.items():
 need(v['dtype']=='F32' and '.lora_' in name and all(type(x) is int and x>0 for x in v['shape']),'Complete F32 LoRA layout')
 count=math.prod(v['shape']);begin,end=v['data_offsets'];need(end-begin==4*count,'Tensor layout byte sizes');offsets.append((begin,end));params+=count
cursor=0
for begin,end in sorted(offsets):need(begin==cursor and end>begin,'Complete gapfree safetensors layout');cursor=end
need(cursor+8+length==weight.stat().st_size and params==t['trainable_parameters']==5193728,'Exact payload geometry/parameters')
source=r['source_manifest'];need(read(F/'sources/manifest.json')==source and len(source['files'])==96,'Exact96 captured source manifest')
for label,v in source['files'].items():
 need(v['artifact_path']=='sources/'+label and pin(F/'sources'/label)=={k:v[k] for k in ['sha256','size_bytes']},'Captured source byte pin')
 current=R/('src/'+label if label.startswith('s1/') else label);need(pin(current)=={k:v[k] for k in ['sha256','size_bytes']},'Current selected source bytes')
need(len(p['sources'])==104,'Exact104 owner source population')
for v in p['sources']:need(pin(R/v['path'])=={k:v[k] for k in ['sha256','size_bytes']},'All104 selected source pins')
for split,v in r['input_manifests'].items():
 root=F/v['artifact_path'];expected=v['files'];actual={z.relative_to(root).as_posix():pin(z) for z in root.rglob('*') if z.is_file() and z.name!='input_manifest.json'}
 need(actual==expected,'Complete captured split files '+split)
for parentkey in ['initial_action_adapter','native_data','resource_components']:
 v=r[parentkey];root=F/v['artifact_path']
 if 'files' in v:
  expected=v['files'];actual={z.relative_to(root).as_posix():pin(z) for z in root.rglob('*') if z.is_file()};need(actual==expected,'Complete captured '+parentkey)
updates,dev=rows('updates.jsonl'),rows('development.jsonl');need(len(updates)==142 and len(dev)==37,'Exact ledger lengths')
records={}
for split in ['train','development']:
 compiled=rows('compiled-'+split+'.jsonl');cfg=rows('cfg-plans-'+split+'.jsonl');need(len(compiled)==len(cfg)==(142 if split=='train' else 37),'Complete compiled/CFG rows')
 need(pin(F/('compiled-'+split+'.jsonl'))['sha256']==r['compiled'][split]['sha256'],'Compiled full byte SHA')
 need(pin(F/('cfg-plans-'+split+'.jsonl'))['sha256']==r['cfg_compiled'][split]['sha256'],'CFG full byte SHA')
 need([z['case_id'] for z in compiled]==[z['case_id'] for z in cfg],'Ordered CFG/compiled alignment')
 records[split]={z['case_id']:z for z in compiled};need(len(records[split])==len(compiled),'Unique compiled examples')
need(not(set(records['train'])&set(records['development'])),'No train/dev IDs overlap')
# Reuse only the pinned actual pure ledger/resource validators under a checked-byte loader.
bpath=R/'artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py';need(pin(bpath)['sha256']=='a39e811764f83eff793f3f5ade7fb9defd7e21fdc7d86207b5e12ce7c22b8581','Fixed checked source importer')
spec=importlib.util.spec_from_file_location('_n3_audit_source_loader',bpath);b=importlib.util.module_from_spec(spec);exec(compile(bpath.read_bytes(),str(bpath),'exec',dont_inherit=True),b.__dict__)
artifact_path=R/'scripts/jev_public_action_artifact.py'
need(pin(artifact_path)=={k:source['files']['scripts/jev_public_action_artifact.py'][k] for k in ['sha256','size_bytes']},'Pinned pure phase-scope source')
artifact_spec=importlib.util.spec_from_file_location('jev_public_action_artifact',artifact_path);artifact_module=importlib.util.module_from_spec(artifact_spec);sys.modules[artifact_spec.name]=artifact_module;exec(compile(artifact_path.read_bytes(),str(artifact_path),'exec',dont_inherit=True),artifact_module.__dict__)
with b.ProjectSources({'src/'+label:v['sha256'] for label,v in source['files'].items() if label.startswith('s1/')}):
 from s1 import browser_action_adapter as a
 a._native_training(F,r,records)
 a._native_resources(F,r)
need(all(v['success'] is True for v in [r['source_verification'],r['historical_original_verification'],*r['input_verification'].values(),*r['frozen_input_verification'].values()]),'Recorded original/frozen source/input closure')
need(c['plan_sha256']==pin(D/'launch-plan-02.json')['sha256'] and c['reason']=='terminal' and c['primary_error'] is None,'External selected terminal')
need(c['cgroup_absent_after_stop'] is True and c['after_stop']['MainPID']=='0' and len(c['two_proc_scans'])==2 and all(z['matches']==z['unreadable']==[] for z in c['two_proc_scans']) and c['changed_sources']==[],'External process/cgroup/two scans/source closure')
terminal=c['terminal_before_stop'];need(all(terminal[k]==v for k,v in {'MainPID':'0','Result':'success','ExecMainCode':'1','ExecMainStatus':'0'}.items()),'Successful unit terminal')
need(terminal['InvocationID']==c['invocation_id'] and terminal['ExecMainPID']==str(c['actor']['pid']) and c['actor']==c['actor_read'],'Invocation/actor identity')
need(not (Path('/proc')/str(c['actor']['pid'])).exists() and not (Path('/sys/fs/cgroup')/p['cgroup'].lstrip('/')).exists(),'Independent current actor/cgroup absence')
host=c['host_after'];need(host['boot_id']==p['boot_id'] and host['mem_total_bytes']==p['expected_memtotal'] and host['comfy_state'] in {'http_idle','http_busy','absent_confirmed'},'Known sameboot final host')
need(strict['strict_verified'] is True and strict['model_constructors']==0 and strict['CUDA_context_created'] is False and strict['telemetry']['manifest_sha256']==pin(F/'adapter/manifest.json')['sha256'] and strict['telemetry']['weights_sha256']==pin(weight)['sha256'],'Root strict artifact verifier receipt binding')
train_mean=math.fsum(z['loss'] for z in updates)/142;dev_mean=math.fsum(z['loss'] for z in dev)/37;need(dev_mean==t['development_loss_mean'],'Exact recomputed dev loss')
result={'schema_id':'jev-native-N3-independent-cpu-fit-audit-v1','status':'complete142_fit_and_external_closure_verified','training_updates':142,'development_examples':37,'development_updates':0,'training_loss_mean':train_mean,'development_loss_mean':dev_mean,'scope':'Teacher-forced historical179 fitting diagnostics only. No browser success or model qualification inferred; interrupted119/72 attempts are not summed.','adapter':{'manifest':ref(F/'adapter/manifest.json'),'weights':ref(weight),'tensor_count':176,'parameters':params,'stored_dtype':'F32','tensor_layout_full_contiguous':True,'tensor_payload_full_SHA_verified_without_ML_imports':True},'source_counts':{'owner_population':104,'captured_runtime_sources':96,'all_full_small_file_hashes_verified':True},'pure_original_validators':{'_native_training':'PASS; complete ordered actual ledgers, forward traces and token totals','_native_resources':'PASS; pinned original resource components and complete source evidence replay'},'forward_accounting':t['execution'],'original_head_and_base_unchanged_recorded':True,'full_base_payload_read':False,'external':{'invocation_id':c['invocation_id'],'unit':p['unit'],'terminal_success':True,'two_clean_scans':True,'current_actor_and_cgroup_absent':True,'final_host_known_sameboot_total':True},'refs':{'plan':ref(D/'launch-plan-02.json'),'receipt':ref(F/'receipt.json'),'training':ref(F/'training.json'),'updates':ref(F/'updates.jsonl'),'development':ref(F/'development.jsonl'),'source_manifest':ref(F/'sources/manifest.json'),'resource_evidence':ref(F/'resource-gates.jsonl'),'external_closure':ref(D/'root-external-closure-01.json'),'root_strict_verifier':ref(D/'root-strict-artifact-verification-01.json')},'audit':{'ML_framework_or_GPU_or_model_or_browser_or_unit_mutation':False,'small_files_streamed_only':True,'max_individual_file_bytes':max(k[2] for k in cache),'elapsed_seconds':time.monotonic()-started,'auditor_path':'/tmp/audit_native_n3.py','auditor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
out=D/'independent-fit-audit-01.json';need(not out.exists(),'Fresh audit artifact');out.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n');print(json.dumps({'audit':ref(out),'training_mean':train_mean,'development_mean':dev_mean,'adapter176F32':True,'sources104':True,'scope':'No ML imports/baseweight read','seconds':time.monotonic()-started}))
