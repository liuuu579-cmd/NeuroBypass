"""Retry an overflowing FP16 batch without discarding data or advancing Adam.

The loss scale changes the numerical representation, not the intended objective.
All stochastic base/adapter dropout is disabled by the recovery protocol.
"""
import math

import torch


def matched_step(bank, optimizer, scaler, backward_batch, on_retry=None, max_retries=24):
    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        values = backward_batch(scaler)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        finite = bool(torch.isfinite(norm))
        if finite:
            # unscale_ found no nonfinite values, so this is a successful update.
            scaler.step(optimizer)
            scaler.update()
            if any(not torch.isfinite(p).all() for p in bank.parameters()):
                raise ValueError('Nonfinite adapter parameter after Adam step')
            return values, float(norm), retries
        # Do not call optimizer.step or scaler.step on a failed attempt. update
        # with an explicit scale also resets GradScaler's per-optimizer state.
        previous_scale = scaler.get_scale()
        scaler.update(new_scale=previous_scale/2)
        retries += 1
        if on_retry is not None:
            on_retry({'attempt': retries, 'old_scale': previous_scale,
                'new_scale': scaler.get_scale(), 'optimizer_advanced': False})
        if retries > max_retries or scaler.get_scale() < 2**-16:
            raise FloatingPointError('Same batch still has invalid gradients after bounded scale retries')


def forced_retry_check(bank, backward_batch, lr):
    """Actual CUDA FP16 overflow, same-batch retry, matched finite-scale replay.

Restore the initial bank after this check. The check never contributes an update
to the experimental optimizer or changes the four-arm initialization.
"""
    initial = {n:t.detach().clone() for n,t in bank.state_dict().items()}
    records = []
    optimizer = torch.optim.AdamW(bank.parameters(),lr=lr,weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda',init_scale=2**22)
    def retry(record):
        if optimizer.state:
            raise ValueError('Overflow attempt mutated Adam state')
        if any(not torch.equal(t,initial[n]) for n,t in bank.state_dict().items()):
            raise ValueError('Overflow attempt mutated adapter parameters')
        records.append(record)
    try:
        _,_,retries = matched_step(bank,optimizer,scaler,backward_batch,retry)
        if retries == 0:
            raise ValueError('Forced-overflow check did not exercise retries')
        recovered = {n:t.detach().clone() for n,t in bank.state_dict().items()}
        first_moments = {n:optimizer.state[p]['exp_avg'].detach().clone() for n,p in bank.named_parameters()}
        if {int(s['step']) for s in optimizer.state.values()} != {1}:
            raise ValueError('Retries advanced Adam more than once')
        final_scale = scaler.get_scale()
        bank.load_state_dict(initial)
        reference = torch.optim.AdamW(bank.parameters(),lr=lr,weight_decay=0.0)
        reference_scaler = torch.amp.GradScaler('cuda',init_scale=final_scale)
        _,_,reference_retries = matched_step(bank,reference,reference_scaler,backward_batch)
        error = max((t-recovered[n]).abs().max().item() for n,t in bank.state_dict().items())
        gradient_errors = {}
        for n,p in bank.named_parameters():
            a,b = first_moments[n], reference.state[p]['exp_avg']
            gradient_errors[n] = {'max_abs':float((a-b).abs().max()),
                'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-20))}
        passed = reference_retries == 0 and math.isfinite(error) and error <= 1e-7
        return {'passed':passed,'reference_retries':reference_retries,'gradient_errors':gradient_errors,'forced_initial_scale':2**22,'retry_attempts':retries,
            'successful_updates':1,'parameters_and_adam_unchanged_on_failed_attempts':True,
            'finite_scale_reference_parameter_max_abs_error':error,'final_scale':final_scale,
            'attempts':records}
    finally:
        bank.load_state_dict(initial)
        bank.zero_grad(set_to_none=True)
