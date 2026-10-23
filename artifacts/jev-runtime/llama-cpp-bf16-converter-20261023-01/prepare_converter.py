"""Prepare a local MIT-licensed upstream converter with an explicit BF16 byte path."""
import difflib
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).absolute().parent
SOURCE = HERE.parent / 'llama-cpp-source-184f653'
TARGET = HERE / 'runtime'
PIN = '184f6536024668e8908ab8c86033d56708e62b40'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one pinned source seam')
    return text.replace(old, new)


def main():
    if TARGET.exists():
        raise ValueError('Fresh converter directory required')
    TARGET.mkdir()
    for name in ('conversion', 'gguf-py'):
        shutil.copytree(SOURCE / name, TARGET / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in ('convert_hf_to_gguf.py', 'LICENSE'):
        shutil.copyfile(SOURCE / name, TARGET / name)
    changes = []
    p = TARGET / 'conversion/base.py'
    old = p.read_text()
    new = replace_once(old,
        '            # convert any unsupported data types to float32\n            if data_torch.dtype not in (torch.float16, torch.float32):',
        '            raw_bf16 = (getattr(self, "jev_preserve_bf16", False)\n'
        '                        and data_torch.dtype == torch.bfloat16\n'
        '                        and len(data_torch.shape) > 1\n'
        '                        and name.endswith(".weight")\n'
        '                        and not name.endswith("_norm.weight")\n'
        '                        and self.norm_shift(name) == 0.0)\n'
        '            # Keep the default conversion unchanged unless explicitly selected.\n'
        '            if data_torch.dtype not in (torch.float16, torch.float32) and not raw_bf16:')
    new = replace_once(new, '                data = data_torch.numpy()',
        '                if raw_bf16:\n'
        '                    from .jev_bf16 import raw_bf16_numpy\n'
        '                    data = raw_bf16_numpy(data_torch)\n'
        '                else:\n'
        '                    data = data_torch.numpy()')
    new = replace_once(new, '                try:\n                    data = quantize(data_qtype)',
        '                if raw_bf16 and data_qtype != gguf.GGMLQuantizationType.BF16:\n'
        '                    raise ValueError("BF16 byte path requires BF16 output for this tensor")\n'
        '                try:\n'
        '                    if not raw_bf16:\n'
        '                        data = quantize(data_qtype)')
    p.write_text(new)
    changes.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/conversion/base.py', tofile='b/conversion/base.py'))
    p = TARGET / 'convert_hf_to_gguf.py'
    old = p.read_text()
    new = replace_once(old, '    args = parser.parse_args()',
        '    parser.add_argument("--jev-preserve-bf16", action="store_true",\n'
        '                        help="Local Gemma4 Unified finite BF16 byte export; default remains unchanged")\n'
        '    args = parser.parse_args()')
    new = replace_once(new, '        model_instance = model_class(dir_model, output_type, fname_out,',
        '        if args.jev_preserve_bf16:\n'
        '            if (model_class.__name__ != "Gemma4UnifiedModel"\n'
        '                    or model_class.__module__ != "conversion.gemma"\n'
        '                    or output_type != gguf.LlamaFileType.MOSTLY_BF16\n'
        '                    or args.bigendian or args.no_lazy or args.use_temp_file\n'
        '                    or args.fuse_qkv or args.fuse_gate_up_exps or args.fp8_as_q8\n'
        '                    or hf_repo_id is not None\n'
        '                    or hparams.get("quantization_config")\n'
        '                    or hparams.get("text_config", {}).get("quantization_config")):\n'
        '                raise ValueError("BF16 byte export requires local unquantized lazy Gemma4 Unified BF16 without fusion")\n'
        '        model_instance = model_class(dir_model, output_type, fname_out,')
    new = replace_once(new, '        if args.vocab_only:',
        '        model_instance.jev_preserve_bf16 = args.jev_preserve_bf16\n\n'
        '        if args.vocab_only:')
    p.write_text(new)
    changes.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='a/convert_hf_to_gguf.py', tofile='b/convert_hf_to_gguf.py'))
    shutil.copyfile(HERE / 'jev_bf16.py', TARGET / 'conversion/jev_bf16.py')
    (HERE / 'converter.patch').write_text(''.join(changes))
    files = []
    for p in sorted(TARGET.rglob('*')):
        if p.is_file():
            raw=p.read_bytes()
            files.append({'path':p.relative_to(TARGET).as_posix(),'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    (HERE / 'source-manifest.json').write_text(json.dumps({'upstream_commit':PIN,'mode':'local experimental BF16 converter, runtime graph unchanged, no upstream submission','files':files},indent=2)+'\n')
    print(json.dumps({'file_count':len(files),'patch_bytes':(HERE/'converter.patch').stat().st_size}))


if __name__ == '__main__':
    main()
