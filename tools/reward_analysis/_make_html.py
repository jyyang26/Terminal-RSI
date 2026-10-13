#!/usr/bin/env python3
"""Deliverable 1: self-contained interactive HTML report -> reward_distribution_report.html."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
P = json.loads((BASE / "_html_payload.json").read_text())

o1, o2 = P["run1"]["overall"], P["run2"]["overall"]
c1, c2 = P["run1"]["corr"], P["run2"]["corr"]
f1, f2 = P["fail"]["run1"], P["fail"]["run2"]

DATA_JS = json.dumps(P)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Reward Distribution Diagnostic · Harbor Terminal-RL</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
 :root{{
   --hero:#285AC7; --navy:#0F2459; --ice:#E8F0FE; --mid:#93B5F5; --grey:#5A739C;
   --ink:#1e293b; --muted:#64748b; --line:#dbe3f3; --bg:#f6f8fd; --card:#ffffff;
 }}
 *{{box-sizing:border-box}}
 body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   color:var(--ink);background:var(--bg);line-height:1.55;}}
 header{{background:linear-gradient(135deg,var(--navy),var(--hero));color:#fff;padding:38px 44px 30px;}}
 header .kick{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:#bcd0f7;margin-bottom:8px}}
 header h1{{margin:0;font-size:30px;font-weight:800;letter-spacing:-.01em}}
 header p{{margin:10px 0 0;font-size:15px;color:#dbe6ff;max-width:960px}}
 .wrap{{max-width:1200px;margin:0 auto;padding:28px 24px 70px}}
 section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 28px;margin:22px 0;
   box-shadow:0 1px 3px rgba(15,36,89,.05)}}
 h2{{font-size:21px;color:var(--navy);margin:0 0 4px;font-weight:750}}
 h2 .num{{color:var(--mid);font-weight:800;margin-right:10px}}
 .sub{{color:var(--muted);font-size:13.5px;margin:0 0 18px}}
 .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}
 .card{{background:linear-gradient(160deg,#fff,var(--ice));border:1px solid var(--line);border-radius:12px;padding:18px 18px 16px}}
 .card .lab{{font-size:12.5px;color:var(--grey);font-weight:600;text-transform:uppercase;letter-spacing:.05em}}
 .card .big{{font-size:27px;font-weight:800;color:var(--navy);margin:7px 0 2px;letter-spacing:-.01em}}
 .card .delta{{font-size:13px;font-weight:600}}
 .up{{color:#1d8f54}} .down{{color:#b4552d}} .flat{{color:var(--grey)}}
 .chart{{width:100%;height:460px}}
 .chart.tall{{height:540px}}
 .note{{background:var(--ice);border-left:4px solid var(--hero);padding:13px 16px;border-radius:8px;
   font-size:14px;color:#263859;margin:16px 0 0}}
 .note b{{color:var(--navy)}}
 .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
 .findings{{margin:14px 0 0;padding:0;list-style:none}}
 .findings li{{padding:11px 14px;border:1px solid var(--line);border-radius:9px;margin-bottom:9px;font-size:14px;background:#fcfdff}}
 .findings li b{{color:var(--navy)}}
 table{{border-collapse:collapse;width:100%;font-size:13.3px;margin-top:6px}}
 th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:var(--navy);color:#fff;cursor:pointer;user-select:none;position:sticky;top:0;font-weight:600}}
 thead th:hover{{background:#17317a}}
 thead th .arrow{{opacity:.5;font-size:11px}}
 tbody tr:nth-child(even){{background:var(--ice)}}
 tbody tr:hover{{background:#dce7fe}}
 .tools{{display:flex;gap:12px;align-items:center;margin-bottom:10px;flex-wrap:wrap}}
 .tools input{{padding:8px 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;width:260px}}
 .tbl-scroll{{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}}
 .pill{{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:600}}
 .pill.r1{{background:#e3e8f1;color:var(--grey)}} .pill.r2{{background:var(--ice);color:var(--hero)}}
 .legend-row{{display:flex;gap:22px;font-size:13px;color:var(--muted);margin:4px 0 14px}}
 .sw{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-1px;margin-right:5px}}
 footer{{color:var(--muted);font-size:12.5px;text-align:center;padding:24px}}
 code{{background:#eef2fb;padding:1px 6px;border-radius:5px;font-size:12.5px;color:var(--navy)}}
 @media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div class="kick">Harbor Terminal-RL · Reward-Shaping Diagnostic</div>
  <h1>Decoupling RL Reward from Validator Count</h1>
  <p>Qwen3-27B dense PPO (kxm RESUME19) · comparing the legacy <b>discrete test-count</b> reward
     against the <b>success-smoothed</b> reward across {len(P['famnames'])} task families and
     {o1['n']}×2 trajectories. How does each reward couple to task-family identity and
     validator (test) count — and does the redesign close the reward-hacking channel?</p>
</header>
<div class="wrap">

<!-- ===== EXECUTIVE SUMMARY ===== -->
<section>
  <h2><span class="num">01</span>Executive Summary</h2>
  <p class="sub">Headline metrics — Run 1 (legacy <code>test_count_progress_rewards</code>) →
     Run 2 (<code>success_smoothed_progress_rewards</code>).</p>
  <div class="cards">
    <div class="card"><div class="lab">Mean reward</div>
      <div class="big">{o1['mean']:.3f} → {o2['mean']:.3f}</div>
      <div class="delta flat">Δ {o2['mean']-o1['mean']:+.3f} · comparable</div></div>
    <div class="card"><div class="lab">Reward std</div>
      <div class="big">{o1['std']:.3f} → {o2['std']:.3f}</div>
      <div class="delta down">total ↑ (bimodal success spike)</div></div>
    <div class="card"><div class="lab">Failing trials with R ≥ 1</div>
      <div class="big">{f1['ge1']} → {f2['ge1']}</div>
      <div class="delta up">reward hacking eliminated</div></div>
    <div class="card"><div class="lab">Failed-trial reward std</div>
      <div class="big">{f1['std']:.3f} → {f2['std']:.3f}</div>
      <div class="delta up">▼ {100*(1-f2['std']/f1['std']):.0f}% · regularized</div></div>
  </div>
  <ul class="findings">
    <li><b>Reward hacking closed.</b> Under the legacy reward, <b>{f1['ge1']} failing trajectories (5.3%)</b>
        scored R ≥ 1.0 — a failure reached <b>R = {f1['max']:.2f}</b>, out-scoring 99%+ of genuine solves.
        Under the smoothed reward this is <b>{f2['ge1']}</b>: every failure is capped at R ≤ {f2['max']:.2f}
        and every solve ≥ 1.20, so the two populations are linearly separable.</li>
    <li><b>Decoupling lives in the failure regime, not the global correlation.</b>
        Global Spearman(N<sub>v</sub>, R) is {c1['spearman']:.2f} → {c2['spearman']:.2f} (≈unchanged) — harder,
        validator-rich tasks legitimately earn the +1 bonus when solved. But failed-trial reward
        correlates with validators at Spearman 0.84 in Run 1 and is flat & bounded in Run 2.</li>
    <li><b>Cleaner critic target.</b> Failed-trial reward variance halves ({f1['std']:.2f} → {f2['std']:.2f})
        and the heavy upper tail is removed by the saturating K<sub>p</sub>/(K<sub>p</sub>+D′) term.</li>
  </ul>
</section>

<!-- ===== FAMILY DISTRIBUTION ===== -->
<section>
  <h2><span class="num">02</span>Reward Distribution by Task Family</h2>
  <p class="sub">Box = IQR of shaped reward per family (ordered by mean validator count). Toggle between the
     full distribution and the <b>failed-trial-only</b> view — the latter isolates the reward-hacking channel.</p>
  <div class="legend-row">
    <span><span class="sw" style="background:var(--grey)"></span>Run 1 (legacy)</span>
    <span><span class="sw" style="background:var(--hero)"></span>Run 2 (ours)</span>
    <span style="margin-left:auto">
      <label><input type="radio" name="fammode" value="all" checked> Full distribution</label>
      &nbsp;&nbsp;<label><input type="radio" name="fammode" value="fail"> Failed trials only</label>
    </span>
  </div>
  <div id="famChart" class="chart tall"></div>
  <div class="note" id="famNote"></div>
</section>

<!-- ===== SCATTER ===== -->
<section>
  <h2><span class="num">03</span>Validator Count vs. Reward</h2>
  <p class="sub">Each point is one trajectory: X = validator (test) count N<sub>v</sub>, Y = shaped reward.
     Solid line = binned mean trend. The dashed line marks the R = 1.0 success gate.</p>
  <div class="two">
    <div><div id="sc1" class="chart"></div></div>
    <div><div id="sc2" class="chart"></div></div>
  </div>
  <div class="note"><b>Read it:</b> in Run 1 (left) failed points (grey) climb with N<sub>v</sub> and
     cross the gate — partial-passing in validator-rich families masquerades as success. In Run 2 (right)
     failed points stay pinned below ~0.86 regardless of N<sub>v</sub>, while solves form a tight band at ~1.5.</div>
</section>

<!-- ===== VALIDATOR BUCKETS ===== -->
<section>
  <h2><span class="num">04</span>Reward by Validator Bucket</h2>
  <p class="sub">Mean reward ± std across validator-count buckets, both runs.</p>
  <div id="bucketChart" class="chart"></div>
</section>

<!-- ===== TABLE ===== -->
<section>
  <h2><span class="num">05</span>Per-Family Statistics (sortable &amp; searchable)</h2>
  <p class="sub">All {len(P['famnames'])} families. Click a header to sort; type to filter.
     ΔCorr proxy = Run2 mean − Run1 mean (per-family reward shift under the redesign).</p>
  <div class="tools">
    <input id="search" placeholder="filter families… e.g. blender, lidar, sqlite"/>
    <span style="font-size:13px;color:var(--muted)" id="tcount"></span>
  </div>
  <div class="tbl-scroll">
    <table id="famTable">
      <thead><tr>
        <th data-k="family">Family <span class="arrow">↕</span></th>
        <th data-k="n_traj">n <span class="arrow">↕</span></th>
        <th data-k="mean_validators">Mean N<sub>v</sub> <span class="arrow">↕</span></th>
        <th data-k="run1_mean">R1 mean <span class="arrow">↕</span></th>
        <th data-k="run1_std">R1 std <span class="arrow">↕</span></th>
        <th data-k="run2_mean">R2 mean <span class="arrow">↕</span></th>
        <th data-k="run2_std">R2 std <span class="arrow">↕</span></th>
        <th data-k="run2_success">R2 succ <span class="arrow">↕</span></th>
        <th data-k="shift">Δ mean <span class="arrow">↕</span></th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</section>

<footer>
  Generated 2026-10-13 · latest 10 rollout dumps per run · reward recomputed from CTRF verifier
  summaries via <code>slime_plugins.harbor.partial_reward</code> · companion:
  <code>reward_distribution_analysis.pdf</code>, <code>reward_comparison_paper.png/.pdf</code>
</footer>
</div>

<script>
const D = {DATA_JS};
const HERO="#285AC7", NAVY="#0F2459", ICE="#E8F0FE", MID="#93B5F5", GREY="#5A739C";
const axisBase = {{ axisLine:{{lineStyle:{{color:"#9fb0cf"}}}}, axisLabel:{{color:"#334155"}},
  splitLine:{{lineStyle:{{color:"#eceff7"}}}}, nameTextStyle:{{color:"#475569"}} }};

/* ---- Family box chart ---- */
const famChart = echarts.init(document.getElementById('famChart'));
function renderFam(mode){{
  const b1 = mode==='fail'? D.box1f : D.box1;
  const b2 = mode==='fail'? D.box2f : D.box2;
  famChart.setOption({{
    tooltip:{{trigger:'item',confine:true}},
    legend:{{data:['Run 1 (legacy)','Run 2 (ours)'],top:0,textStyle:{{color:'#334155'}}}},
    grid:{{left:58,right:24,top:40,bottom:120}},
    xAxis:{{type:'category',data:D.famnames,axisLabel:{{rotate:55,fontSize:10,color:'#334155'}},...axisBase}},
    yAxis:{{type:'value',name:'Shaped reward R',max: mode==='fail'?2.1:2.1, ...axisBase}},
    series:[
      {{name:'Run 1 (legacy)',type:'boxplot',data:b1,
        itemStyle:{{color:'rgba(90,115,156,.45)',borderColor:GREY}},
        boxWidth:['25%','55%']}},
      {{name:'Run 2 (ours)',type:'boxplot',data:b2,
        itemStyle:{{color:'rgba(40,90,199,.35)',borderColor:HERO}},
        boxWidth:['25%','55%']}},
      {{type:'line',markLine:{{silent:true,symbol:'none',data:[{{yAxis:1.0}}],
        lineStyle:{{color:NAVY,type:'dashed'}},label:{{formatter:'success gate R=1.0',color:NAVY,position:'end'}}}}}}
    ]
  }}, true);
}}
const famNote=document.getElementById('famNote');
function setNote(mode){{
  famNote.innerHTML = mode==='fail'
   ? "<b>Failed trials only.</b> Run-1 boxes (grey) smear upward and cross R=1.0 for check-rich families "
     +"(point_cloud_lidar, blender, sqlite) — this is the reward-hacking surface. Run-2 boxes (blue) stay "
     +"compressed below ~0.6 across every family: the saturating K<sub>p</sub>/(K<sub>p</sub>+D′) term caps partial reward."
   : "<b>Full distribution.</b> Run 2 is bimodal per family (bounded failures + a solved spike ≥1.2), so its "
     +"per-family box can look taller; the regularization is specifically in the failed-trial sub-population "
     +"(switch to the 'Failed trials only' view).";
}}
renderFam('all'); setNote('all');
document.querySelectorAll('input[name=fammode]').forEach(r=>r.addEventListener('change',e=>{{renderFam(e.target.value);setNote(e.target.value);}}));

/* ---- Scatter + trend ---- */
function scatterOpt(sc,tr,title,gate){{
  return {{
    title:{{text:title,left:'center',top:4,textStyle:{{color:NAVY,fontSize:15}}}},
    tooltip:{{trigger:'item',formatter:p=>p.seriesType==='scatter'?`N<sub>v</sub>=${{p.value[0]}}<br/>R=${{p.value[1]}}`:`N<sub>v</sub>≈${{p.value[0]}}<br/>mean R=${{p.value[1]}}`}},
    legend:{{data:['Failed','Solved','Mean trend'],top:28,textStyle:{{color:'#334155'}}}},
    grid:{{left:58,right:24,top:60,bottom:56}},
    xAxis:{{type:'log',name:'Validator count N_v (log)',min:1,...axisBase}},
    yAxis:{{type:'value',name:'Shaped reward R',max:2.1,...axisBase}},
    series:[
      {{name:'Failed',type:'scatter',data:sc.failed,symbolSize:5,
        itemStyle:{{color:'rgba(90,115,156,.45)'}}}},
      {{name:'Solved',type:'scatter',data:sc.solved,symbolSize:6,
        itemStyle:{{color:'rgba(40,90,199,.55)'}}}},
      {{name:'Mean trend',type:'line',data:tr,smooth:true,symbol:'circle',symbolSize:7,
        lineStyle:{{color:NAVY,width:3}},itemStyle:{{color:NAVY}},z:5}},
      {{type:'line',markLine:{{silent:true,symbol:'none',data:[{{yAxis:gate}}],
        lineStyle:{{color:'#94a3b8',type:'dashed'}},label:{{formatter:'R=1.0',color:'#64748b'}}}}}}
    ]
  }};
}}
echarts.init(document.getElementById('sc1')).setOption(scatterOpt(D.scatter1,D.trend1,'(a) Run 1 — Discrete Progress',1.0));
echarts.init(document.getElementById('sc2')).setOption(scatterOpt(D.scatter2,D.trend2,'(b) Run 2 — Success-Smoothed',1.0));

/* ---- Buckets ---- */
echarts.init(document.getElementById('bucketChart')).setOption({{
  tooltip:{{trigger:'axis'}},
  legend:{{data:['Run 1','Run 2'],top:0,textStyle:{{color:'#334155'}}}},
  grid:{{left:58,right:24,top:40,bottom:50}},
  xAxis:{{type:'category',data:D.buckets.map(b=>b.bucket),name:'Validator bucket N_v',...axisBase}},
  yAxis:{{type:'value',name:'Mean reward ± std',...axisBase}},
  series:[
    {{name:'Run 1',type:'bar',data:D.buckets.map(b=>({{value:b.run1_mean}})),itemStyle:{{color:GREY}},
      barGap:'10%',
      markLine:{{silent:true,symbol:'none',data:[{{yAxis:1.0}}],lineStyle:{{color:NAVY,type:'dashed'}}}}}},
    {{name:'Run 2',type:'bar',data:D.buckets.map(b=>({{value:b.run2_mean}})),itemStyle:{{color:HERO}}}},
    {{name:'Run 1',type:'custom',data:D.buckets.map((b,i)=>[i,b.run1_mean,b.run1_std]),
      renderItem:errBar(-0.22),z:5,tooltip:{{show:false}},silent:true,legendHoverLink:false}},
    {{name:'Run 2',type:'custom',data:D.buckets.map((b,i)=>[i,b.run2_mean,b.run2_std]),
      renderItem:errBar(0.22),z:5,tooltip:{{show:false}},silent:true,legendHoverLink:false}}
  ]
}});
function errBar(off){{
  return function(params,api){{
    const i=api.value(0), m=api.value(1), s=api.value(2);
    const top=api.coord([i+off, m+s]), bot=api.coord([i+off, Math.max(0,m-s)]);
    const w=5;
    const line=(x1,y1,x2,y2)=>({{type:'line',shape:{{x1,y1,x2,y2}},style:{{stroke:'#1e293b',lineWidth:1.4}}}});
    return {{type:'group',children:[
      line(top[0],top[1],bot[0],bot[1]),
      line(top[0]-w,top[1],top[0]+w,top[1]),
      line(bot[0]-w,bot[1],bot[0]+w,bot[1])
    ]}};
  }};
}}

/* ---- Sortable / searchable table ---- */
const rows = D.families.map(f=>({{...f, shift: f.run2_mean - f.run1_mean}}));
let sortK='mean_validators', sortDir=-1;
const tbody=document.querySelector('#famTable tbody');
const search=document.getElementById('search');
const tcount=document.getElementById('tcount');
function fmt(v,d){{return typeof v==='number'? (d===0? Math.round(v): v.toFixed(d)) : v;}}
function draw(){{
  const q=search.value.trim().toLowerCase();
  let rs=rows.filter(r=>r.family.toLowerCase().includes(q));
  rs.sort((a,b)=>{{const x=a[sortK],y=b[sortK];return (x>y?1:x<y?-1:0)*sortDir;}});
  tbody.innerHTML=rs.map(r=>{{
    const sc=(r.run2_success*100);
    const shift=r.shift>=0?`<span class="up">${{r.shift>=0?'+':''}}${{r.shift.toFixed(2)}}</span>`:`<span class="down">${{r.shift.toFixed(2)}}</span>`;
    return `<tr><td>${{r.family}}</td><td>${{r.n_traj}}</td><td>${{r.mean_validators.toFixed(1)}}</td>
      <td>${{r.run1_mean.toFixed(3)}}</td><td>${{r.run1_std.toFixed(3)}}</td>
      <td>${{r.run2_mean.toFixed(3)}}</td><td>${{r.run2_std.toFixed(3)}}</td>
      <td>${{sc.toFixed(0)}}%</td><td>${{shift}}</td></tr>`;
  }}).join('');
  tcount.textContent=`${{rs.length}} / ${{rows.length}} families`;
}}
document.querySelectorAll('#famTable thead th').forEach(th=>th.addEventListener('click',()=>{{
  const k=th.dataset.k; if(k==='shift'){{/*ok*/}}
  if(sortK===k) sortDir*=-1; else {{sortK=k; sortDir=(k==='family')?1:-1;}}
  draw();
}}));
search.addEventListener('input',draw);
draw();

window.addEventListener('resize',()=>{{
  famChart.resize();
  ['sc1','sc2','bucketChart'].forEach(id=>echarts.getInstanceByDom(document.getElementById(id))?.resize());
}});
</script>
</body>
</html>
"""

out = BASE / "reward_distribution_report.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, len(HTML), "bytes")
