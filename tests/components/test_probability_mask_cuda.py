"""Exact verification parity with overlapping caller scratch and poisoned tails."""
import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import PreparedStochasticVerification, SpeculativeRngState
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle

from tests.components.probability_mask_reference import FunctionalVerificationReference

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device")


def fixture(cls,alias,limited):
    batch,width,vocab=2,3,128
    device='cuda:0'
    state=DeviceRequestState.allocate(device,vocab_size=vocab,capacity=batch)
    for row in range(batch):state.admit(row,RequestAdmission(generation=7,seed=101+row,max_new_tokens=20,eos_token_ids=()),[1,2])
    generator=torch.Generator().manual_seed(73)
    q=torch.rand((batch,width-1,vocab),generator=generator);q[...,0]=-0.;q/=q.sum(-1,keepdim=True)
    p=torch.rand((batch,width,vocab),generator=generator);p[...,1]=-0.;p/=p.sum(-1,keepdim=True)
    if limited:
        q[:,1:]=float('nan');p[:,2:]=float('inf')
    rawq=torch.zeros(batch*width*vocab+128,device=device)
    rawp=torch.zeros_like(rawq)
    qview=rawq[:q.numel()].view_as(q);pview=rawp[:p.numel()].view_as(p)
    ids=torch.tensor([[4,9],[8,13]],device=device,dtype=torch.int64)
    slots=torch.arange(batch,device=device,dtype=torch.int32);epochs=torch.full((batch,),7,device=device,dtype=torch.int64)
    live=torch.tensor([batch],device=device,dtype=torch.int32)
    lengths=torch.full((batch,),2,device=device,dtype=torch.int32)if limited else None
    rng=SpeculativeRngState.allocate(device,capacity=batch);rng.generation.fill_(7);rng.target_offset.fill_(44)
    workspace={name:torch.empty(shape,device=device,dtype=dtype)for name,(shape,dtype)in cls.scratch_layout(batch,width,vocab).items()}
    if alias in ('q_into_p','cross'):workspace['safe_draft_probs']=rawp[:q.numel()].view_as(q)
    if alias in ('p_into_q','cross'):workspace['safe_target_probs']=rawq[:p.numel()].view_as(p)
    if alias=='q_shift':workspace['safe_draft_probs']=rawq[32:32+q.numel()].view_as(q)
    if alias=='p_shift':workspace['safe_target_probs']=rawp[32:32+p.numel()].view_as(p)
    op=cls(qview,ids,pview,state,slots,epochs,live,rng=rng,token_limits=lengths,workspace=workspace)
    def restore():
        rawq.zero_();rawp.zero_();qview.copy_(q);pview.copy_(p)
        if alias=='disjoint':
            op.safe_draft_probs.fill_(float('nan'));op.safe_target_probs.fill_(float('nan'))
        for tensor in vars(op.output).values():tensor.fill_(0)
    def snapshot():
        return [x.cpu().clone()for x in (op.safe_draft_probs,op.safe_target_probs,op.safe_ids,
            *vars(op.output).values(),qview,pview,*vars(rng).values(),*vars(state).values())]
    restore();op.prepare();restore();op();torch.cuda.synchronize();eager=snapshot()
    def forward():op()
    call=compile_forward(forward);restore();call();torch.cuda.synchronize();restore()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    bundle=PreparedGraphBundle(device=0,graphs={'verify':graph},buffers={},resources=(op,call,state,rng,workspace,rawq,rawp))
    worker=bundle.start_worker();results=[]
    try:
        for _ in range(3):
            restore();torch.cuda.synchronize();allocated=torch.cuda.memory_stats()['allocation.all.allocated']
            ticket=worker.submit(0);completion=worker.poll(timeout_ms=30000)
            assert completion is not None and completion.ticket==ticket and completion.error is None
            assert torch.cuda.memory_stats()['allocation.all.allocated']==allocated
            results.append(snapshot())
    finally:worker.close();graph.reset()
    if cls is PreparedStochasticVerification and alias=='disjoint':assert all(m.snapshot is None for m in op.probability_masks)
    return eager,results


@pytest.mark.parametrize('limited',[False,True])
@pytest.mark.parametrize('alias',['disjoint','q_into_p','p_into_q','cross','q_shift','p_shift'])
def test_probability_writes_preserve_reference_aliases(alias,limited):
    torch._dynamo.reset()
    expected=fixture(FunctionalVerificationReference,alias,limited)
    actual=fixture(PreparedStochasticVerification,alias,limited)
    for a,b in zip([expected[0],*expected[1]],[actual[0],*actual[1]],strict=True):
        for x,y in zip(a,b,strict=True):
            assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)), (alias,limited,x.shape,x.dtype)
