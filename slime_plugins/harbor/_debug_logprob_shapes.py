"""Deep debug instrumentation (monkeypatch, does NOT edit core source).

Writes per-rank files to $SLIME_DEBUG_DIR capturing the EXACT tensor numels at the
loss.py policy-loss / reducer sites, to pin down why entropy_loss / pg_loss /
train_rollout_logprob_abs_diff are ~1000x too large while tis_abs stays ~0.02.

Hypothesis to confirm: some tensor's numel != sum(response_lengths), so
x.split(response_lengths) misaligns and a group absorbs extra elements.

Activated when this module is imported (via usercustomize on PYTHONPATH).
Every rank writes SLIME_DEBUG_DIR/dbg_rank_<pid>.log so ray log suppression
cannot hide the PP-last-stage rank.
"""
import os

_INSTALLED = False
_DBG_DIR = os.environ.get("SLIME_DEBUG_DIR", "/path/to/workspace/slime_dbg_out")


def _w(msg):
    try:
        os.makedirs(_DBG_DIR, exist_ok=True)
        with open(os.path.join(_DBG_DIR, f"dbg_rank_{os.getpid()}.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import torch
    import slime.backends.megatron_utils.loss as L
    import slime.backends.megatron_utils.cp_utils as C

    # ---- 1. wrap the reducer factory: log every reducer CALL's x.numel vs sum(response_lengths)
    orig_getred = C.get_sum_of_sample_mean

    def patched_getred(total_lengths, response_lengths, loss_masks, sample_denoms=None,
                       calculate_per_token_loss=False, qkv_format="thd", max_seq_lens=None):
        red = orig_getred(total_lengths, response_lengths, loss_masks, sample_denoms,
                          calculate_per_token_loss, qkv_format, max_seq_lens)
        sum_resp = int(sum(response_lengths))
        # capture the actual denominators used
        try:
            if sample_denoms is None:
                denoms = [float(m.sum()) for m in loss_masks]
            elif hasattr(sample_denoms, "__len__"):
                denoms = [float(x) for x in sample_denoms]
            else:
                denoms = [float(sample_denoms)]
        except Exception:
            denoms = ["?"]
        masksum = [float(m.sum()) for m in loss_masks]
        tag = {"count": 0}

        def wrapped(x):
            if tag["count"] < 3:
                try:
                    import torch as _t
                    xf = x.float()
                    _w(f"[REDUCER] call#{tag['count']} x.numel={x.numel()} sum_resp={sum_resp} "
                       f"n_samples={len(response_lengths)} MATCH={x.numel()==sum_resp} "
                       f"denoms={denoms[:3]} mask_sums={masksum[:3]} "
                       f"x.abs.mean={float(xf.abs().mean()):.4f} x.abs.max={float(xf.abs().max()):.2f} "
                       f"result={float(red(x)):.4f}")
                except Exception as e:
                    _w(f"[REDUCER] err {e}")
                tag["count"] += 1
            return red(x)
        return wrapped

    C.get_sum_of_sample_mean = patched_getred
    L.get_sum_of_sample_mean = patched_getred

    # ---- 2. wrap get_log_probs_and_entropy: log logits.shape + returned list lengths
    orig_glpe = L.get_log_probs_and_entropy
    glpe_state = {"n": 0}

    def patched_glpe(logits, *, args, unconcat_tokens, total_lengths, response_lengths, **k):
        res = orig_glpe(logits, args=args, unconcat_tokens=unconcat_tokens,
                        total_lengths=total_lengths, response_lengths=response_lengths, **k)
        if glpe_state["n"] < 4:
            try:
                sum_resp = int(sum(response_lengths))
                sum_tot = int(sum(total_lengths))
                lp = res.get("log_probs")
                en = res.get("entropy")
                lp_numel = int(sum(int(t.numel()) for t in lp)) if lp else -1
                en_numel = int(sum(int(t.numel()) for t in en)) if en else -1
                _w(f"[GLPE] call#{glpe_state['n']} logits.shape={tuple(logits.shape)} "
                   f"sum_tot={sum_tot} sum_resp={sum_resp} n_samples={len(response_lengths)}")
                _w(f"[GLPE]   log_probs: n={len(lp) if lp else 0} total_numel={lp_numel} "
                   f"MATCH_sumresp={lp_numel==sum_resp} per0-3={[int(t.numel()) for t in (lp[:3] if lp else [])]}")
                if en:
                    _w(f"[GLPE]   entropy:   n={len(en)} total_numel={en_numel} "
                       f"MATCH_sumresp={en_numel==sum_resp} per0-3={[int(t.numel()) for t in en[:3]]}")
            except Exception as e:
                _w(f"[GLPE] err {e}")
            glpe_state["n"] += 1
        return res

    L.get_log_probs_and_entropy = patched_glpe

    # ---- 3. wrap policy_loss_function: log batch inputs + final reported metrics
    orig_ploss = L.policy_loss_function
    ps = {"n": 0}

    def patched_ploss(args, batch, logits, sum_of_sample_mean, *a, **k):
        if ps["n"] < 4:
            try:
                resp = batch["response_lengths"]
                sum_resp = int(sum(resp))
                lp = batch.get("log_probs")
                rlp = batch.get("rollout_log_probs")
                adv = batch.get("advantages")
                lp_n = int(sum(int(t.numel()) for t in lp)) if lp else -1
                rlp_n = int(sum(int(t.numel()) for t in rlp)) if rlp else -1
                adv_n = int(sum(int(t.numel()) for t in adv)) if adv else -1
                _w(f"[PLOSS] call#{ps['n']} logits.shape={tuple(logits.shape)} sum_resp={sum_resp} "
                   f"n_samples={len(resp)}")
                _w(f"[PLOSS]   batch['log_probs'] total_numel={lp_n} MATCH={lp_n==sum_resp}")
                _w(f"[PLOSS]   batch['rollout_log_probs'] total_numel={rlp_n} MATCH={rlp_n==sum_resp}")
                _w(f"[PLOSS]   batch['advantages'] total_numel={adv_n} MATCH={adv_n==sum_resp}")
            except Exception as e:
                _w(f"[PLOSS] pre err {e}")
        ret = orig_ploss(args, batch, logits, sum_of_sample_mean, *a, **k)
        try:
            loss, reported = ret
            if ps["n"] < 4:
                _w(f"[PLOSS]   -> abs_diff={float(reported.get('train_rollout_logprob_abs_diff', float('nan'))):.4f} "
                   f"pg_loss={float(reported.get('pg_loss', float('nan'))):.2f} "
                   f"entropy={float(reported.get('entropy_loss', float('nan'))):.2f} "
                   f"tis_abs={float(reported.get('tis_abs', float('nan'))):.6f}")
        except Exception as e:
            _w(f"[PLOSS] post err {e}")
        ps["n"] += 1
        return ret

    L.policy_loss_function = patched_ploss

    _INSTALLED = True
    _w(f"[INSTALL] deep instrumentation installed in pid {os.getpid()}")


install()
