import sys, glob, os, collections, torch

dd = sys.argv[1]
files = sorted(glob.glob(os.path.join(dd, "*.pt")), key=os.path.getmtime)
if not files:
    print("no dumps yet")
    sys.exit()
f = files[-1]
print("dump:", os.path.basename(f))
obj = torch.load(f, map_location="cpu", weights_only=False)
samples = obj if isinstance(obj, list) else (obj.get("samples") if isinstance(obj, dict) else None)
if samples and isinstance(samples[0], list):
    samples = [s for grp in samples for s in grp]
print("num samples:", len(samples) if samples else 0)

keycount = collections.Counter()
strat = collections.Counter()
chunk_modes = collections.Counter()
trim = 0
resp_trim = 0
gids = collections.Counter()
ntok = []
for s in (samples or []):
    md = getattr(s, "metadata", None) or {}
    for k in md:
        if k.startswith("harbor_trajectory") or "retokeniz" in k or "chunk" in k or k.endswith("_turns"):
            keycount[k] += 1
    gid = getattr(s, "group_id", None)
    if gid is not None:
        gids[gid] += 1
    cm = md.get("harbor_trajectory_chunk_strategy") or md.get("harbor_trajectory_mode")
    if cm:
        chunk_modes[cm] += 1
    for kk in ("harbor_trajectory_retokenized_prefix_turns", "harbor_trajectory_strict_prefix_turns",
               "harbor_trajectory_normalized_prefix_turns", "harbor_trajectory_split_after_turns"):
        v = md.get(kk)
        if isinstance(v, (int, float)) and v:
            strat[kk] += v
    if md.get("harbor_train_response_trimmed"):
        resp_trim += 1
    if md.get("harbor_train_context_trimmed"):
        trim += 1
    tok = getattr(s, "tokens", None)
    if tok is not None:
        try:
            ntok.append(len(tok))
        except Exception:
            pass

print("unique group_ids:", len(gids))
# chunks-per-trajectory = how often retoken kept a trajectory whole (1) vs split (>1)
chunks_per_gid = collections.Counter(gids.values())
print("chunks-per-group_id distribution (1 = whole trajectory, >1 = was split):", dict(sorted(chunks_per_gid.items())))
print("harbor-trajectory metadata keys seen:", dict(keycount))
print("chunk strategy/mode counts:", dict(chunk_modes))
print("transition strategy token sums:", dict(strat))
print("context-trimmed samples:", trim, " response-trimmed samples:", resp_trim)
if ntok:
    ntok.sort()
    print("tokens/sample  min/median/p95/max:", ntok[0], ntok[len(ntok) // 2], ntok[int(len(ntok) * 0.95)], ntok[-1])
# dump one sample's full harbor metadata for ground truth
for s in (samples or []):
    md = getattr(s, "metadata", None) or {}
    hk = {k: v for k, v in md.items() if k.startswith("harbor")}
    if any("traject" in k or "retok" in k or "turn" in k for k in hk):
        print("example sample harbor metadata:", {k: hk[k] for k in sorted(hk)})
        break
