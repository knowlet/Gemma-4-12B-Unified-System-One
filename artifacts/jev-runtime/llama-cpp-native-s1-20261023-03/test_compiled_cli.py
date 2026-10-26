"""Exercise the compiled cache-option parser before any backend/model startup."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--runner',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); runner=args.runner.resolve(strict=True)
    rows=[]
    for value in ('0','1'):
        call=subprocess.run([str(runner),'--swa-full',value,'--help'],capture_output=True,text=True,timeout=5)
        assert call.returncode==0 and '--swa-full 0|1' in call.stdout
        rows.append({'kind':'valid-explicit-option-before-help','value':value,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    for value in ('2','-1','true','01','1x',''):
        call=subprocess.run([str(runner),'--model','/NONEXISTENT-S1-PARSER-MODEL','--mmproj','/NONEXISTENT-S1-PARSER-PROJECTOR','--swa-full',value],capture_output=True,text=True,timeout=5)
        assert call.returncode==1 and call.stderr.strip()=='s1-gguf: --swa-full must be exactly 0 or 1'
        rows.append({'kind':'invalid-value-before-backend','value':value,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    call=subprocess.run([str(runner),'--swa-full'],capture_output=True,text=True,timeout=5)
    assert call.returncode==1 and 'missing value for --swa-full' in call.stderr
    rows.append({'kind':'missing-value-before-backend','returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    for value in ('0','1'):
        call=subprocess.run([str(runner),'--cpu-mmap-only',value,'--gpu-layers','0','--help'],capture_output=True,text=True,timeout=5)
        assert call.returncode==0 and '--cpu-mmap-only 0|1' in call.stdout
        rows.append({'kind':'valid-cpu-mmap-before-help','value':value,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    for value in ('2','-1','true','01','1x',''):
        call=subprocess.run([str(runner),'--model','/NONEXISTENT-S1-PARSER-MODEL','--mmproj','/NONEXISTENT-S1-PARSER-PROJECTOR','--cpu-mmap-only',value],capture_output=True,text=True,timeout=5)
        assert call.returncode==1 and call.stderr.strip()=='s1-gguf: --cpu-mmap-only must be exactly 0 or 1'
        rows.append({'kind':'invalid-cpu-mmap-value-before-backend','value':value,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    call=subprocess.run([str(runner),'--cpu-mmap-only'],capture_output=True,text=True,timeout=5)
    assert call.returncode==1 and 'missing value for --cpu-mmap-only' in call.stderr
    rows.append({'kind':'missing-cpu-mmap-value-before-backend','returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    for gpu in ('1','99'):
        call=subprocess.run([str(runner),'--model','/NONEXISTENT-S1-PARSER-MODEL','--mmproj','/NONEXISTENT-S1-PARSER-PROJECTOR','--cpu-mmap-only','1','--gpu-layers',gpu],capture_output=True,text=True,timeout=5)
        assert call.returncode==1 and call.stderr.strip()=='s1-gguf: --cpu-mmap-only 1 requires --gpu-layers 0'
        rows.append({'kind':'incompatible-gpu-before-backend','gpu_layers':gpu,'returncode':call.returncode,'stdout':call.stdout,'stderr':call.stderr})
    assert not args.output.exists()
    args.output.write_text(json.dumps({'status':'PASS','compiled_cases':len(rows),'scope':'Actual C++ option parsing/help only; no backend init/model payload/context/math/quality qualification','rows':rows},indent=2)+'\n')
    print(json.dumps({'status':'PASS','compiled_cases':len(rows)}))


if __name__=='__main__': main()
