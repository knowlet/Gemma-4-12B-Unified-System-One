"""Fetch immutable public GGUF release files, hashing the streamed bytes once."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE=Path(__file__).absolute().parent
ROOT=HERE.parents[2]
REV='c418d37a17689fae554f7fc7d78db05ff7d52cfb'
BASE='https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One-GGUF/resolve/'+REV+'/'
EXPECTED={
 'artifact-manifest.json':(None,'1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0'),
 'conversion-manifest.json':(None,'464874bc51e8ee21ab56f34a4bf7d22b1fb83299a277cbcf966b42c5fc7802b7'),
 's1-boolq-Q8_0.gguf':(12669647264,'9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed'),
 'mmproj-s1-boolq-f16.gguf':(122031744,'cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1')}


def fetch(name,directory,limit):
    size,digest=EXPECTED[name]
    target=directory/name; partial=directory/(name+'.partial')
    if target.exists() or partial.exists():
        raise ValueError('Fresh download path required; partial evidence is preserved')
    count=0; h=hashlib.sha256(); started=time.monotonic(); headers={}
    request=urllib.request.Request(BASE+name+'?download=true',headers={'User-Agent':'S1-GX10-evaluation/1.0'})
    with urllib.request.urlopen(request,timeout=30) as reply:
        if reply.status!=200: raise ValueError('Expected complete HTTP200 response')
        headers={k:reply.headers[k] for k in ('Content-Length','Content-Type','ETag') if k in reply.headers}
        with partial.open('xb') as out:
            while block:=reply.read(1024**2):
                count+=len(block)
                if count>limit: raise ValueError('Download exceeded selected byte limit')
                out.write(block); h.update(block)
            out.flush(); os.fsync(out.fileno())
    if (size is not None and count!=size) or h.hexdigest()!=digest:
        raise ValueError('Immutable published file identity mismatch')
    partial.rename(target)
    row={'name':name,'revision':REV,'size_bytes':count,'sha256':h.hexdigest(),'expected_sha256':digest,'verified':True,'elapsed_seconds':time.monotonic()-started,'http_status':200,'nonsecret_response_headers':headers,'hash_scope':'Exact downloaded stream, before rename; no second full payload read'}
    (directory/(name+'.download.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:row[k] for k in ('name','size_bytes','sha256','elapsed_seconds')}),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--metadata-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.absolute()
    if out.exists(): raise ValueError('Fresh package directory required')
    if ('..' in out.parts or not out.is_relative_to(ROOT)
            or any(p.is_symlink() for p in (out,*out.parents))):
        raise ValueError('Ordinary repository output required')
    out.mkdir(parents=False,mode=0o700)
    names=['artifact-manifest.json','conversion-manifest.json'] if args.metadata_only else ['s1-boolq-Q8_0.gguf','mmproj-s1-boolq-f16.gguf']
    receipts=[]
    try:
        for name in names: receipts.append(fetch(name,out,1024**2 if args.metadata_only else EXPECTED[name][0]))
        status='PASS'
    except BaseException as error:
        (out/'download-failure.json').write_text(json.dumps({
            'type':type(error).__name__,
            'http_status':getattr(error,'code',None),
            'reason_type':type(getattr(error,'reason',None)).__name__,
            'message':'Download did not complete; exception URLs are intentionally omitted',
            'completed':receipts},indent=2)+'\n')
        raise RuntimeError('Public release download failed; see bounded failure record') from None
    (out/'package-receipt.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':status,'revision':REV,'mode':'metadata' if args.metadata_only else 'full native model/projector','files':receipts,'Torch_imported':'torch' in sys.modules},indent=2)+'\n')


if __name__=='__main__': main()
