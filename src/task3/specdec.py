"""Exact speculative sampling (Leviathan et al., 2023) and a plain target-only sampler.

Both models are wrapped as *engines* with the same small interface, so the identical sampler code
runs on a toy Markov chain (for exact distribution tests) and on the real Qwen models:
  engine.prefill(ids)    -> next-token probability vector after the prompt
  engine.extend(tokens)  -> probability vectors after each appended token  [n, V]
  engine.rollback(n)     -> keep only the first n processed tokens (KV-cache crop)
  engine.length          -> number of processed tokens
Bookkeeping: each engine keeps a list of emitted tokens it has not processed yet ("pending").
A round: the draft processes its pending tokens and proposes up to gamma tokens (stopping at EOS
or at the max-new-token limit); the target processes its pending tokens plus all proposals in ONE
forward pass (verification); proposals are accepted with prob min(1, p/q); the first rejected one is
replaced by a sample from the normalised positive residual (p - q)+; if every proposal is accepted
a bonus token is sampled from the target. Both caches are then cropped to exactly the returned
continuation (minus the last emitted token, which becomes pending).
"""
import time

import torch


def _sync(dev):
    if dev is not None and str(dev).startswith("cuda"):
        torch.cuda.synchronize()


class HFEngine:
    def __init__(self, model, temperature=1.0):
        self.model, self.T = model, temperature
        self.dev = next(model.parameters()).device
        self.cache, self.length, self.forward_passes = None, 0, 0

    def _probs(self, logits):
        return torch.softmax(logits.float() / self.T, dim=-1)

    @torch.no_grad()
    def prefill(self, ids):
        from transformers.cache_utils import DynamicCache
        self.cache = DynamicCache()
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.length = ids.shape[1]
        self.forward_passes = 0
        return self._probs(out.logits[0, -1])

    @torch.no_grad()
    def extend(self, tokens):
        ids = torch.tensor([tokens], device=self.dev)
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True,
                         cache_position=torch.arange(self.length, self.length + len(tokens), device=self.dev))
        self.length += len(tokens)
        self.forward_passes += 1
        return self._probs(out.logits[0])

    def rollback(self, n):
        if n < self.length:
            self.cache.crop(n)
            self.length = n


class MarkovEngine:
    """Toy first-order Markov 'language model': P[a, b] = p(next = b | last = a); start row s0."""

    def __init__(self, P, s0):
        self.P, self.s0 = P, s0
        self.toks, self.forward_passes = [], 0

    @property
    def length(self):
        return len(self.toks)

    def prefill(self, ids):
        self.toks = list(ids)
        return self.P[self.toks[-1]] if self.toks else self.s0

    def extend(self, tokens):
        self.toks += list(tokens)
        self.forward_passes += 1
        return torch.stack([self.P[t] for t in tokens])

    def rollback(self, n):
        self.toks = self.toks[:n]


def _sample(p, gen):
    """gen must live on the same device as p (CPU for the toy chain, CUDA for the real models)."""
    return int(torch.multinomial(p, 1, generator=gen))


def speculative_generate(target, draft, prompt, gamma, max_new, gen, eos=(), wrong_residual=False, timers=False):
    """Returns (tokens, stats). `wrong_residual=True` is the deliberately incorrect sampler that
    samples from p instead of the residual after a rejection (used only as a negative control)."""
    dev = getattr(target, "dev", None)
    st = {"rounds": 0, "proposed": 0, "examined": 0, "accepted": 0, "bonus": 0, "target_passes": 0,
          "draft_passes": 0, "accepted_prefix": [], "t_draft": 0.0, "t_target": 0.0, "t_accept": 0.0,
          "t_rollback": 0.0, "t_prefill": 0.0, "eos": False}
    clock = (lambda: (_sync(dev), time.perf_counter())[1]) if timers else (lambda: 0.0)
    t0 = clock()
    p_next = target.prefill(prompt)
    q_next = draft.prefill(prompt)
    st["t_prefill"] = clock() - t0
    base = prompt.shape[-1] if hasattr(prompt, "shape") else len(prompt)
    out, t_pend, d_pend = [], [], []
    while len(out) < max_new:
        st["rounds"] += 1
        # ---- draft proposals
        t0 = clock()
        g = min(gamma, max_new - len(out))
        if d_pend:
            q = draft.extend(d_pend)[-1]
            st["draft_passes"] += 1
        else:
            q = q_next
        xs, qs = [], []
        for i in range(g):
            x = _sample(q, gen)
            xs.append(x); qs.append(q)
            if x in eos or i == g - 1:
                break
            q = draft.extend([x])[-1]
            st["draft_passes"] += 1
        t1 = clock(); st["t_draft"] += t1 - t0
        # ---- one target verification pass over pending + proposals
        ps_new = target.extend(t_pend + xs)
        st["target_passes"] += 1
        ps = ps_new[len(t_pend) - 1:] if t_pend else torch.cat([p_next[None], ps_new], 0)
        t2 = clock(); st["t_target"] += t2 - t1
        # ---- accept / reject
        st["proposed"] += len(xs)
        n_acc, emitted, stop = 0, [], False
        for i, x in enumerate(xs):
            st["examined"] += 1
            px, qx = float(ps[i][x]), float(qs[i][x])
            if torch.rand(1, generator=gen, device=gen.device).item() < min(1.0, px / qx):
                n_acc += 1
                emitted.append(x)
                if x in eos:
                    stop = True
                    break
            else:
                if wrong_residual:
                    y = _sample(ps[i], gen)
                else:
                    r = (ps[i] - qs[i]).clamp(min=0)
                    y = _sample(r / r.sum(), gen)
                emitted.append(y)
                stop = y in eos
                break
        else:
            if len(out) + len(emitted) < max_new:      # bonus token only if the length cap allows it
                y = _sample(ps[len(xs)], gen)
                emitted.append(y)
                st["bonus"] += 1
                stop = y in eos
        st["accepted"] += n_acc
        st["accepted_prefix"].append(n_acc)
        t3 = clock(); st["t_accept"] += t3 - t2
        # ---- rollback / advance: caches hold everything except the last emitted token
        out += emitted
        keep = base + len(out) - 1
        target.rollback(keep)
        draft.rollback(min(keep, draft.length))
        t_pend = [out[-1]]
        d_pend = out[draft.length - base:]           # tokens the draft has not processed yet
        st["t_rollback"] += clock() - t3
        if stop:
            st["eos"] = True
            break
    st["draft_passes_total"] = st["draft_passes"]
    st["target_passes_total"] = st["target_passes"]
    return out[:max_new], st


def target_only_generate(target, prompt, max_new, gen, eos=(), timers=False):
    dev = getattr(target, "dev", None)
    clock = (lambda: (_sync(dev), time.perf_counter())[1]) if timers else (lambda: 0.0)
    t0 = clock()
    p = target.prefill(prompt)
    t_prefill = clock() - t0
    out, eos_hit = [], False
    t0 = clock()
    while len(out) < max_new:
        y = _sample(p, gen)
        out.append(y)
        if y in eos:
            eos_hit = True
            break
        if len(out) < max_new:
            p = target.extend([y])[-1]
    return out, {"t_prefill": t_prefill, "t_decode": clock() - t0, "target_passes": len(out), "eos": eos_hit}
