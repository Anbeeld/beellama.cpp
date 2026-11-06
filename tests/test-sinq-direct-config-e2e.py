#!/usr/bin/env python3
"""Direct v2 config -> native quantization -> independently inspected companions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gguf-py'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--quantize',required=True);args=p.parse_args()
    import numpy as np
    from gguf import GGUFReader
    with tempfile.TemporaryDirectory(prefix='sinq-direct-config-') as tmp:
        d=Path(tmp)
        base={'generator':'sinq','strip_width':32,'iterations':16,'strength':1.0}
        names=['blk.0.ffn_up.weight','blk.0.ffn_down.weight']
        cfg={'schema_version':2,'defaults':base,'targets':{'include':names},'overrides':[{'selector':{'tensor':names[1]},'parameters':{'strip_width':64,'iterations':4,'strength':0.5}}]}
        config=d/'config.json';config.write_text(json.dumps(cfg),encoding='utf-8')
        def run(config_path,output,success=True):
            cmd=[args.quantize,'--pure','--sinq-config',str(config_path),args.source,str(output),'Q5_0','1']
            proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace')
            if (proc.returncode==0)!=success:raise AssertionError(proc.stdout+proc.stderr)
        output=d/'mixed.gguf';run(config,output)
        reader=GGUFReader(str(output))
        assert reader.fields['beellama.qk_sinq.version'].contents()==2
        assert reader.fields['beellama.qk_sinq.transform'].contents()=='f16_input_diagonal'
        actual=reader.fields['beellama.qk_sinq.tensors'].contents()
        recipes=reader.fields['beellama.qk_sinq.recipes'].contents()
        assert actual==sorted(names)
        expected={names[0]:base,names[1]:{**base,'strip_width':64,'iterations':4,'strength':0.5}}
        for name,text in zip(actual,recipes):assert json.loads(text)==expected[name]
        companions=[t for t in reader.tensors if t.name.endswith('.sinq_scale2')]
        assert {t.name for t in companions}=={name+'.sinq_scale2' for name in names}
        for tensor in companions:assert np.all(np.isfinite(tensor.data)&(tensor.data>0))
        reader.data._mmap.close()
        # Replay complete settings and compare serialized output, not only reports.
        replay=d/'replay.gguf';run(config,replay)
        assert output.read_bytes()==replay.read_bytes()
        bad={**cfg,'overrides':cfg['overrides']*2};config.write_text(json.dumps(bad),encoding='utf-8')
        run(config,d/'invalid.gguf',False)
        assert not (d/'invalid.gguf').exists()
    print('test-sinq-direct-config-e2e: PASS');return 0
if __name__=='__main__':raise SystemExit(main())
