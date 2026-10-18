"""Source-only CUDA operation measurements; benchmarks are never study weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import time

def measure(protocol,mae_path,output):
    import numpy as np
    import torch
    from src.mae_encoder import MAEImageEncoder
    from .hybrid_protocol import validate_hybrid_protocol,HeadSpec,RetrievalSpec
    from .hybrid_heads import CommonController,make_bank,chunk_mse
    from .hybrid_study import read_view_frames,read_actions,read_proprio,_sha_file
    from .models import attach_mae_lora
    from .source_worker import atomic
    validate_hybrid_protocol(protocol)
    task=next(t for t,s in sorted(protocol['tasks'].items()) if s['role']=='source');s=protocol['tasks'][task]
    if _sha_file(Path(s['path']))!=s['source_sha256']:raise ValueError('source benchmark dataset identity mismatch')
    demo=int(s['source']['source_validation'][0]);bank_demo=int(s['source']['source_train'][0])
    frames=read_view_frames(protocol,s['path'],demo);actions=read_actions(s['path'],demo);q=read_proprio(s['path'],demo,protocol)
    bank_frames=read_view_frames(protocol,s['path'],bank_demo)
    d=protocol['design'];chunk=d['temporal']['chunk_steps'];batch=d['training']['batch_size'];micro=d['training']['microbatch_size']
    idx=np.arange(batch)%(min(len(frames),len(bank_frames))-chunk)
    current=torch.from_numpy(np.ascontiguousarray(frames[idx]));future=torch.from_numpy(np.ascontiguousarray(frames[idx+chunk]))
    bank_current=torch.from_numpy(np.ascontiguousarray(bank_frames[idx]));bank_future=torch.from_numpy(np.ascontiguousarray(bank_frames[idx+chunk]))
    target=torch.from_numpy(np.stack([actions[t:t+chunk] for t in idx])).cuda()
    proprio=torch.from_numpy(q[idx]).float().cuda();duration=torch.full((batch,),float(chunk),device='cuda')
    torch.manual_seed(42);encoder=MAEImageEncoder.from_checkpoint(mae_path,'cuda')
    def timed(fn):
        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();return time.perf_counter()-start,result
    def encode_pair(a,b):
        return torch.cat([encoder.encode_frames(torch.cat((a[x:x+micro],b[x:x+micro]))) for x in range(0,batch,micro)])
    with torch.no_grad():
        # Use ordered current/future pairs for each microbatch.
        encode_seconds,raw=timed(lambda:encode_pair(current,future))
        fc=torch.cat([raw[x:x+micro] for x in range(0,len(raw),2*micro)])
        ff=torch.cat([raw[x+micro:x+2*micro] for x in range(0,len(raw),2*micro)])
        _,br=timed(lambda:encode_pair(bank_current,bank_future))
        bc=torch.cat([br[x:x+micro] for x in range(0,len(br),2*micro)]);bf=torch.cat([br[x+micro:x+2*micro] for x in range(0,len(br),2*micro)])
    retrieval=RetrievalSpec(**d['retrieval']);head=CommonController(chunk,spec=HeadSpec(**d['head']),retrieval=retrieval).cuda()
    bank=make_bank(bc,bf,proprio,target,duration,torch.full((batch,),bank_demo,dtype=torch.long,device='cuda'),'cuda')
    groups=torch.full((batch,),demo,dtype=torch.long,device='cuda')
    opt=torch.optim.AdamW(head.parameters(),lr=d['training']['head_lr'])
    def source_update():
        opt.zero_grad(set_to_none=True)
        a,_=head.forward_with_memory(fc,ff,proprio,duration,bank=bank,use_retrieval=True,exclude_groups=groups)
        b,_=head.forward_with_memory(fc,ff,proprio,duration,bank=None,use_retrieval=False)
        loss=(chunk_mse(a,target).mean()+chunk_mse(b,target).mean())/2;loss.backward();opt.step()
    source_update();source_seconds=[timed(source_update)[0] for _ in range(5)]
    cfg=d['inverse_model']['lora'];attach_mae_lora(encoder.mae,cfg['blocks'],cfg['rank'],cfg['alpha'],cfg['dropout'])
    opt=torch.optim.AdamW([{'params':head.parameters(),'lr':d['training']['head_lr']},{'params':[p for p in encoder.parameters() if p.requires_grad],'lr':d['training']['lora_lr']}])
    def adapt_update(use_retrieval):
        opt.zero_grad(set_to_none=True)
        for start in range(0,batch,micro):
            stop=min(start+micro,batch);features=encoder.encode_frames(torch.cat((current[start:stop],future[start:stop])))
            n=stop-start
            pred,_=head.forward_with_memory(features[:n],features[n:],proprio[start:stop],duration[start:stop],bank=bank if use_retrieval else None,use_retrieval=use_retrieval,exclude_groups=groups[start:stop],query_current=fc[start:stop],query_future=ff[start:stop])
            (chunk_mse(pred,target[start:stop]).mean()*n/batch).backward()
        opt.step()
    adaptations={}
    for condition,retrieval_on in [('lora_mlp',False),('raid_lora_mlp',True)]:
        adapt_update(retrieval_on);adaptations[condition]=[timed(lambda:adapt_update(retrieval_on))[0] for _ in range(3)]
    with torch.no_grad():
        head.eval();inference=[timed(lambda:head.forward_with_memory(fc[:1],ff[:1],proprio[:1],duration[:1],bank=bank,use_retrieval=True,exclude_groups=groups[:1]))[0] for _ in range(10)]
    report={'protocol_digest':protocol['digest'],'model_digest':protocol['model_digest'],'data_scope':'source_validation_and_source_training_only','task':task,'demo':demo,'bank_demo':bank_demo,'source_sha256':s['source_sha256'],'mae_sha256':_sha_file(Path(mae_path)),'cuda_device':torch.cuda.get_device_name(),'batch_size':batch,'microbatch_size':micro,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'reference_encode_pair_batch_seconds':encode_seconds,'source_update_seconds':source_seconds,'adaptation_update_seconds':adaptations,'controller_inference_seconds':inference,'estimated_full_source_training_seconds':statistics.median(source_seconds)*d['training']['source_updates'],'estimated_full_adaptation_seconds':{c:statistics.median(v)*d['training']['target_updates'] for c,v in adaptations.items()},'limitations':['microbenchmark uses one source task and small bank; full cache construction, full-bank retrieval and validation remain estimated','benchmark weights are discarded and cannot be reused as study checkpoints','generator and model-load costs are measured separately'],'scientific_result':False}
    atomic(output,report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mae',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    measure(json.loads(Path(a.protocol).read_text()),a.mae,a.output)
