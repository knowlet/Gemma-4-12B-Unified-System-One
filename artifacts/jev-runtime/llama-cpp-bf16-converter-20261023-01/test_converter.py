"""Synthetic actual-upstream conversion regression, with no real checkpoint payload."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
RUNTIME = HERE / 'runtime'
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / 'gguf-py'))
import numpy as np
import torch
import gguf
from conversion.base import LazyTorchTensor
from conversion.gemma import Gemma4UnifiedModel
from conversion.jev_bf16 import raw_bf16_numpy


def eager(tensor):
    return gguf.LazyNumpyTensor.to_eager(tensor)


class ByteExportRegression(unittest.TestCase):
    def test_all_finite_bf16_bit_patterns_equal_upstream(self):
        words = np.arange(65536, dtype=np.uint16)
        words = words[(words & 0x7f80) != 0x7f80].reshape(255, 256)
        tensor = torch.from_numpy(words.copy()).view(torch.bfloat16)
        raw = eager(raw_bf16_numpy(tensor))
        reference = gguf.quants.quantize(tensor.float().numpy(), gguf.GGMLQuantizationType.BF16)
        np.testing.assert_array_equal(raw, reference)
        self.assertEqual(raw.tobytes(), words.tobytes())

    def test_lazy_real_torch_byte_path(self):
        tensor = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
        raw = eager(raw_bf16_numpy(LazyTorchTensor.from_eager(tensor)))
        self.assertEqual(raw.tobytes(), tensor.view(torch.uint8).numpy().tobytes())

    def test_nonfinite_refused_before_return(self):
        for word in (0x7f80, 0xff80, 0x7f81, 0x7fc0, 0xff81):
            with self.subTest(word=word):
                tensor = torch.from_numpy(np.array([[word, 0]], dtype=np.uint16)).view(torch.bfloat16)
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    eager(raw_bf16_numpy(tensor))

    def test_noncontiguous_and_wrong_dtype_refused(self):
        with self.assertRaisesRegex(ValueError, 'Contiguous'):
            eager(raw_bf16_numpy(torch.ones((2, 3), dtype=torch.bfloat16).t()))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            raw_bf16_numpy(torch.ones((2, 3), dtype=torch.float32))

    def convert(self, folder, selected):
        folder.mkdir()
        cfg = json.loads((HERE.parent/'bf16-real-layer-cpu-20261023-01/header-selection.json').read_text())['config']['text_config']
        root_cfg = {'architectures':['Gemma4UnifiedForConditionalGeneration'], 'model_type':'gemma4_unified', 'text_config':cfg}
        (folder/'config.json').write_text(json.dumps(root_cfg))
        (folder/'tokenizer.json').write_text(json.dumps({'model':{'vocab':[['x',0]]*6},'added_tokens':[]}))
        values = {
            'model.language_model.embed_tokens.weight':torch.arange(32,dtype=torch.bfloat16).reshape(8,4),
            'model.language_model.layers.0.self_attn.q_proj.weight':torch.arange(24,dtype=torch.bfloat16).reshape(6,4),
            'model.language_model.layers.0.input_layernorm.weight':torch.tensor([0,0.5,-1,2],dtype=torch.bfloat16),
            'model.language_model.layers.0.layer_scalar':torch.tensor([0.75],dtype=torch.bfloat16),
        }
        header={}; body=b''
        for name,t in values.items():
            raw=t.view(torch.uint8).numpy().tobytes(); header[name]={'dtype':'BF16','shape':list(t.shape),'data_offsets':[len(body),len(body)+len(raw)]}; body+=raw
        encoded=json.dumps(header).encode(); encoded+=b' '*((-len(encoded))%8)
        (folder/'model.safetensors').write_bytes(struct.pack('<Q',len(encoded))+encoded+body)
        dest=folder/'synthetic.gguf'
        model=Gemma4UnifiedModel(folder,gguf.LlamaFileType.MOSTLY_BF16,dest)
        model.jev_preserve_bf16=selected
        calls=[]; original=torch.Tensor.to
        def trace(t,*args,**kwargs):
            if t.dtype==torch.bfloat16 and t.ndim>1 and (torch.float32 in args or kwargs.get('dtype')==torch.float32): calls.append(tuple(t.shape))
            return original(t,*args,**kwargs)
        with patch.object(torch.Tensor,'to',trace):
            model.prepare_tensors()
            writer=model.gguf_writer
            writer.write_header_to_file(dest); writer.write_kv_data_to_file(); writer.write_tensors_to_file(progress=False); writer.close()
        reader=gguf.GGUFReader(dest)
        outputs={t.name:{'type':int(t.tensor_type),'shape':tuple(t.shape),'bytes':t.data.tobytes()} for t in reader.tensors}
        return outputs,calls,values

    def test_actual_prepare_writer_default_and_raw_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference,casts,_=self.convert(Path(tmp)/'default',False)
            actual,no_cast,values=self.convert(Path(tmp)/'selected',True)
            self.assertEqual(reference,actual)
            self.assertTrue(casts)
            self.assertEqual(no_cast,[])
            self.assertEqual(actual['token_embd.weight']['bytes'],values['model.language_model.embed_tokens.weight'][:6].view(torch.uint8).numpy().tobytes())
            self.assertEqual(actual['blk.0.attn_norm.weight']['type'],int(gguf.GGMLQuantizationType.F32))
            self.assertEqual(actual['blk.0.layer_output_scale.weight']['type'],int(gguf.GGMLQuantizationType.F32))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'),'')


if __name__=='__main__':
    unittest.main(verbosity=2)
