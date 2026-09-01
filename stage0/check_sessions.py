import pandas as pd, numpy as np
from scipy.stats import spearmanr
d = pd.read_parquet("segments_with_session.parquet")

print("=== the one bad row ===")
print(d[d.f_spk != d.speaker_id].to_string())

d = d[d.f_spk == d.speaker_id].copy()
d["sess"] = d.f_sess

print("\n=== A. session is a SHARED protocol id ===")
print("distinct session ids globally      :", d.sess.nunique())
sps = d.groupby("sess").speaker_id.nunique()
print(f"speakers per session               : median {sps.median():.0f}  mean {sps.mean():.1f}  max {sps.max()}")
print(f"sessions per speaker               : median {d.groupby('speaker_id').sess.nunique().median():.0f}")

print("\n=== B. is age constant within (speaker, session)? ===")
g = d.groupby(["speaker_id","sess"])
na = g.age.nunique()
print(f"(speaker,session) cells            : {len(na)}")
print(f"cells with >1 distinct age         : {(na>1).sum()}  ({(na>1).mean():.6f})")
spread = g.age.agg(lambda s: s.max()-s.min())
print(f"within-cell age spread (years)     : max {spread.max():.6f}  99.9pct {spread.quantile(0.999):.6f}")
print(f"                          in days  : max {spread.max()*365.25:.2f}")

print("\n=== C. does age COLLIDE across distinct sessions? (does age alone resolve sessions?) ===")
cell = d.groupby(["speaker_id","sess"]).age.median().reset_index()
coll = cell.groupby(["speaker_id","age"]).sess.nunique()
print(f"(speaker,age) keys                 : {len(coll)}")
print(f"(speaker,session) cells            : {len(cell)}")
print(f"(speaker,age) keys covering >1 session: {(coll>1).sum()}  ({(coll>1).mean():.4f})")
print(f"sessions lost to age collisions    : {(coll-1).sum()}  of {len(cell)}  ({(coll-1).sum()/len(cell):.4f})")

print("\n=== D. is session id chronological (vs age)? ===")
rs = []
for s, grp in cell.groupby("speaker_id"):
    if len(grp) >= 5:
        rs.append(spearmanr(grp.sess, grp.age).statistic)
rs = np.array(rs); rs = rs[~np.isnan(rs)]
print(f"per-speaker spearman(session_id, age), n={len(rs)}: median {np.median(rs):.4f}  "
      f"mean {rs.mean():.4f}  min {rs.min():.4f}  frac>0.99 {(rs>0.99).mean():.4f}")

print("\n=== E. entry requirements on SESSION key ===")
sp = d.groupby("speaker_id").agg(n_seg=("filename","size"), n_sess=("sess","nunique"),
                                 hours=("duration_s", lambda s: s.sum()/3600))
ok = (sp.hours>=3)&(sp.n_sess>=8)&(sp.n_seg>=20)
print(f"speakers >=3h & >=8 sessions & >=20 seg : {ok.sum()} of {len(sp)}")
print(f"  median hours among eligible           : {sp[ok].hours.median():.1f}")
print(f"  median sessions among eligible        : {sp[ok].n_sess.median():.0f}")
sp.to_csv("speaker_session_counts.csv")
d.to_parquet("segments_with_session.parquet")
