"use client";

import { useMemo, useState } from "react";

const tabs = ["Overview", "Retrieval", "States", "Topology", "Validation", "Forecast"] as const;
type Tab = (typeof tabs)[number];

const stateColors = ["#7dd3fc", "#c4b5fd", "#f9a8d4", "#fdba74", "#fde047", "#86efac"];
const confusion = [
  [74, 9, 6, 7, 4],
  [8, 68, 12, 8, 4],
  [5, 10, 71, 10, 4],
  [3, 5, 12, 69, 11],
  [2, 3, 6, 13, 76],
];
const operations = ["Observe", "Experience", "Evaluate", "Model", "Execute"];

function Metric({ value, label, tone = "blue" }: { value: string; label: string; tone?: string }) {
  return (
    <div className={`metric ${tone}`}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function Badge({ children, good = false }: { children: React.ReactNode; good?: boolean }) {
  return <span className={good ? "badge good" : "badge"}>{children}</span>;
}

function Overview() {
  return (
    <>
      <section className="hero-grid">
        <div>
          <div className="eyebrow">PRIVACY-PRESERVING AI RESEARCH</div>
          <h1>Evaluate the system,<br /><span>not the demo.</span></h1>
          <p className="lede">
            A working research environment for testing retrieval quality, latent-state structure,
            topology, leakage, and forecast calibration across a long-running, multisource corpus.
          </p>
          <div className="actions">
            <a className="primary" href="https://github.com/lukejones3/life-topology-lab" target="_blank">View source ↗</a>
            <button className="secondary" onClick={() => document.getElementById("pipeline")?.scrollIntoView({ behavior: "smooth" })}>See evaluation pipeline</button>
          </div>
        </div>
        <div className="signal-card">
          <div className="signal-head"><span>LIVE EVALUATION SHAPE</span><Badge good>methods verified</Badge></div>
          <svg viewBox="0 0 520 265" role="img" aria-label="Synthetic latent state trajectory">
            <defs>
              <linearGradient id="line" x1="0" x2="1">
                <stop offset="0" stopColor="#7dd3fc" />
                <stop offset=".52" stopColor="#c4b5fd" />
                <stop offset="1" stopColor="#86efac" />
              </linearGradient>
              <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            {[40, 90, 140, 190, 240].map(y => <line key={y} x1="0" x2="520" y1={y} y2={y} className="grid-line" />)}
            <path className="area" d="M0 218 C38 210 52 150 89 170 S145 224 181 137 S238 44 274 105 S320 213 356 158 S410 78 448 98 S489 49 520 58 L520 265 L0 265Z" />
            <path className="trend" d="M0 218 C38 210 52 150 89 170 S145 224 181 137 S238 44 274 105 S320 213 356 158 S410 78 448 98 S489 49 520 58" />
            {[["89","170"],["181","137"],["274","105"],["356","158"],["448","98"],["520","58"]].map(([x,y], i) =>
              <circle key={i} cx={x} cy={y} r="5" fill={stateColors[i]} filter="url(#glow)" />)}
          </svg>
          <div className="signal-foot"><span>16-month analysis window</span><span>synthetic display • real pipeline</span></div>
        </div>
      </section>
      <section className="metrics">
        <Metric value="1.17M" label="normalized events" />
        <Metric value="827K" label="atomic semantic units" tone="purple" />
        <Metric value="170K+" label="embedded vectors" tone="green" />
        <Metric value="2" label="independent embedding models" tone="orange" />
      </section>
      <section className="panel pipeline" id="pipeline">
        <div className="section-head">
          <div><div className="eyebrow">EVALUATION ARCHITECTURE</div><h2>Evidence before interpretation</h2></div>
          <p>Every conclusion must survive retrieval, alternate representations, null models, and time-aware validation.</p>
        </div>
        <div className="flow">
          {[
            ["01", "Ingest", "10 source channels"],
            ["02", "Resolve", "identity + time"],
            ["03", "Represent", "multi-view vectors"],
            ["04", "Test", "nulls + decoys"],
            ["05", "Interpret", "evidence-linked"],
          ].map((item, i) => (
            <div className="flow-wrap" key={item[0]}>
              <div className="flow-card"><span>{item[0]}</span><strong>{item[1]}</strong><small>{item[2]}</small></div>
              {i < 4 && <div className="arrow">→</div>}
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function Retrieval() {
  const [model, setModel] = useState<"MiniLM" | "BGE-M3">("BGE-M3");
  const rows = model === "BGE-M3"
    ? [["1", "System architecture moved from cron to dependency-aware orchestration.", "0.928", "Relevant"], ["2", "A failure-closed quality gate stopped downstream publication.", "0.891", "Relevant"], ["3", "A weekly monitoring view exposed recurrence after deployment.", "0.847", "Relevant"], ["4", "Unrelated high-volume conversational event.", "0.312", "Decoy"]]
    : [["1", "System architecture moved from cron to dependency-aware orchestration.", "0.861", "Relevant"], ["2", "A weekly monitoring view exposed recurrence after deployment.", "0.802", "Relevant"], ["3", "A failure-closed quality gate stopped downstream publication.", "0.788", "Relevant"], ["4", "Unrelated high-volume conversational event.", "0.398", "Decoy"]];
  return (
    <section className="panel">
      <div className="section-head">
        <div><div className="eyebrow">RETRIEVAL BENCHMARK</div><h2>Two models. Same evidence. Measured disagreement.</h2></div>
        <div className="toggle">{(["MiniLM", "BGE-M3"] as const).map(x => <button key={x} className={model === x ? "active" : ""} onClick={() => setModel(x)}>{x}</button>)}</div>
      </div>
      <div className="query"><span>QUERY</span> Which evidence shows production orchestration and reliability work?</div>
      <div className="result-table">
        <div className="result-row header"><span>RANK</span><span>EVIDENCE CHUNK</span><span>SCORE</span><span>LABEL</span></div>
        {rows.map(r => <div className="result-row" key={r[0]}><span>#{r[0]}</span><span>{r[1]}</span><span className="mono">{r[2]}</span><span><Badge good={r[3] === "Relevant"}>{r[3]}</Badge></span></div>)}
      </div>
      <div className="three">
        <Metric value={model === "BGE-M3" ? "0.91" : "0.84"} label="nDCG@10" tone="green" />
        <Metric value={model === "BGE-M3" ? "92%" : "86%"} label="precision@5" tone="purple" />
        <Metric value="1,200" label="human-audit rows" tone="orange" />
      </div>
      <p className="note">The interface uses synthetic text. The benchmark design, dual-model comparison, audit workflow, and failure tests are implemented in the private research system.</p>
    </section>
  );
}

function States() {
  const [selected, setSelected] = useState(5);
  return (
    <section className="panel">
      <div className="section-head">
        <div><div className="eyebrow">BLIND VALIDATION</div><h2>Do learned states recover cognitive operations?</h2></div>
        <Badge good>label-blind test</Badge>
      </div>
      <p className="intro">A separate model assigns one operation—without seeing the learned state labels. Enrichment tests whether the states represent repeatable computation rather than attractive storytelling.</p>
      <div className="state-pills">{stateColors.map((color, i) => <button key={i} onClick={() => setSelected(i)} className={selected === i ? "selected" : ""} style={{ "--state": color } as React.CSSProperties}>S{i}</button>)}</div>
      <div className="state-detail">
        <div>
          <span className="mini-label">SELECTED LATENT STATE</span>
          <h3 style={{ color: stateColors[selected] }}>S{selected}</h3>
          <p>{["Low-activation observation", "Prospective construction", "Social experience", "Technical execution", "Relational evaluation", "Integrative modeling"][selected]}</p>
        </div>
        <div className="bars">
          {operations.map((op, i) => {
            const widths = [
              [62,14,10,8,6],[8,11,12,28,41],[18,54,14,8,6],[5,8,11,20,56],[8,18,48,20,6],[5,8,13,59,15]
            ][selected][i];
            return <div className="bar-row" key={op}><span>{op}</span><div><i style={{ width: `${widths}%`, background: stateColors[selected] }} /></div><b>{widths}%</b></div>;
          })}
        </div>
      </div>
      <div className="matrix-wrap">
        <div><div className="eyebrow">CONFUSION MATRIX</div><h3>Operation classification</h3></div>
        <div className="matrix">
          <div />
          {operations.map(x => <small key={x}>{x.slice(0,3)}</small>)}
          {confusion.map((row, i) => [<small key={`l${i}`}>{operations[i]}</small>, ...row.map((v,j) => <span key={`${i}-${j}`} style={{ opacity: .35 + v/110, background: i===j ? "#7c3aed" : "#22304a" }}>{v}</span>)])}
        </div>
      </div>
    </section>
  );
}

function Topology() {
  return (
    <section className="panel topo">
      <div className="section-head">
        <div><div className="eyebrow">CONLEY–MORSE APPROXIMATION</div><h2>Recurrent basins and one-way gateways</h2></div>
        <div><Badge>sampled dynamics</Badge> <Badge good>perturbation stable</Badge></div>
      </div>
      <div className="topo-grid">
        <svg viewBox="0 0 760 460" className="topo-map" role="img" aria-label="Synthetic Conley-Morse state graph">
          <defs>
            <marker id="tip" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker>
            <radialGradient id="basin"><stop offset="0" stopColor="#8b5cf6" stopOpacity=".28"/><stop offset="1" stopColor="#8b5cf6" stopOpacity="0"/></radialGradient>
          </defs>
          <ellipse cx="372" cy="239" rx="340" ry="190" fill="url(#basin)" />
          {[[150,110,365,84],[196,139,343,207],[392,110,382,194],[564,142,428,206],[546,329,429,264],[190,342,339,271],[371,276,371,360]].map((a,i) =>
            <line key={i} x1={a[0]} y1={a[1]} x2={a[2]} y2={a[3]} stroke="#64748b" strokeWidth="2" markerEnd="url(#tip)" />)}
          {[
            [125,90,"S0","#7dd3fc","observe"],[385,72,"S1","#c4b5fd","construct"],[592,132,"S2","#f9a8d4","experience"],
            [588,344,"S3","#fdba74","execute"],[374,390,"S4","#fde047","evaluate"],[365,235,"S5","#86efac","model"],[145,356,"G","#67e8f9","gateway"]
          ].map(([x,y,label,color,sub]) => <g key={String(label)}><circle cx={Number(x)} cy={Number(y)} r={label==="S5" ? 48 : 36} fill={String(color)} fillOpacity=".13" stroke={String(color)} strokeWidth="2"/><text x={Number(x)} y={Number(y)+2} textAnchor="middle" fill="white" fontWeight="700">{label}</text><text x={Number(x)} y={Number(y)+18} textAnchor="middle" fill="#94a3b8" fontSize="10">{sub}</text></g>)}
        </svg>
        <div className="topo-side">
          <div className="finding"><span>01</span><div><strong>Gateway structure</strong><p>Some observed transitions require an intermediate state rather than occurring directly.</p></div></div>
          <div className="finding"><span>02</span><div><strong>Recurrent basin</strong><p>S5 behaves as an integration basin with several inbound routes and fewer immediate exits.</p></div></div>
          <div className="finding"><span>03</span><div><strong>Method</strong><p>Daily vectors become a sampled multivalued map; graph decompositions are checked under perturbation.</p></div></div>
          <div className="warning">Illustration uses synthetic geometry. The source repository contains the implemented graph and topology methods.</div>
        </div>
      </div>
    </section>
  );
}

function Validation() {
  const checks = [
    ["Future leakage audit", "PASS", "No future-linked features in walk-forward folds"],
    ["High-volume decoy test", "PASS", "Activity volume does not dominate impact"],
    ["Time-shuffle null", "PASS", "Observed transition structure exceeds shuffled baseline"],
    ["Embedding agreement", "PASS", "Major structure replicated across MiniLM and BGE-M3"],
    ["Right-censor handling", "PASS", "Open regimes excluded from completed-duration estimates"],
    ["Invariant scalar search", "FAILED", "No stable conserved scalar survived alternate windows"],
  ];
  return (
    <section className="panel">
      <div className="section-head">
        <div><div className="eyebrow">FAILURE-AWARE EVALUATION</div><h2>Negative results stay in the record.</h2></div>
        <Badge good>6 checks visible</Badge>
      </div>
      <div className="checks">{checks.map(([name,status,detail]) => <div className="check" key={name}><span className={status === "PASS" ? "pass" : "fail"}>{status === "PASS" ? "✓" : "×"}</span><div><strong>{name}</strong><p>{detail}</p></div><Badge good={status === "PASS"}>{status}</Badge></div>)}</div>
      <div className="validation-foot">
        <div><span className="mini-label">CORE PRINCIPLE</span><p>A compelling interpretation is not a result until it survives a plausible alternative explanation.</p></div>
        <div><span className="mini-label">REGRESSION SURFACE</span><p>Retrieval rankings, cluster stability, leakage checks, calibration, and topology are rerunnable artifacts.</p></div>
      </div>
    </section>
  );
}

function Forecast() {
  const points = useMemo(() => [12,19,17,28,33,42,48,58,65,72], []);
  return (
    <section className="panel">
      <div className="section-head">
        <div><div className="eyebrow">WALK-FORWARD FORECASTING</div><h2>Probabilities are scored after reality arrives.</h2></div>
        <Badge good>temporal holdout</Badge>
      </div>
      <div className="forecast-grid">
        <div className="chart-card">
          <div className="chart-title"><div><span className="mini-label">CALIBRATION</span><h3>Predicted vs. observed</h3></div><span className="mono">Brier 0.164</span></div>
          <svg viewBox="0 0 520 300">
            {[45,95,145,195,245].map(y => <line key={y} x1="40" x2="500" y1={y} y2={y} className="grid-line" />)}
            <line x1="40" y1="250" x2="500" y2="25" stroke="#475569" strokeDasharray="5 6" />
            <polyline points={points.map((p,i) => `${48+i*48},${270-p*3}`).join(" ")} fill="none" stroke="#86efac" strokeWidth="4" />
            {points.map((p,i)=><circle key={i} cx={48+i*48} cy={270-p*3} r="5" fill="#0f172a" stroke="#86efac" strokeWidth="3"/>)}
          </svg>
        </div>
        <div className="forecast-cards">
          <div className="forecast-card"><span>30 DAYS</span><strong>0.37</strong><p>regime-transition probability</p><i style={{width:"37%"}} /></div>
          <div className="forecast-card"><span>90 DAYS</span><strong>0.64</strong><p>active-intention progression</p><i style={{width:"64%"}} /></div>
          <div className="forecast-card"><span>NEXT WEEK</span><strong>0.71</strong><p>top-2 state accuracy</p><i style={{width:"71%"}} /></div>
        </div>
      </div>
      <div className="note">Displayed scores are demonstrative rather than claims about a person. The implemented framework uses rolling-origin splits, simple baselines, calibration, censoring rules, and future-feature isolation.</div>
    </section>
  );
}

export default function Page() {
  const [tab, setTab] = useState<Tab>("Overview");
  const content = { Overview: <Overview />, Retrieval: <Retrieval />, States: <States />, Topology: <Topology />, Validation: <Validation />, Forecast: <Forecast /> }[tab];
  return (
    <main>
      <header>
        <button className="brand" onClick={() => setTab("Overview")}><span className="mark">LT</span><span>Life Topology Lab<small>AI EVALUATION DEMO</small></span></button>
        <nav>{tabs.map(t => <button key={t} className={tab===t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}</nav>
        <a className="github" href="https://github.com/lukejones3/life-topology-lab" target="_blank">GitHub ↗</a>
      </header>
      <div className="page">{content}</div>
      <footer><span>Life Topology Lab</span><span>Synthetic display data • Real evaluation architecture • No private corpus content</span><span>Built by Luke Jones</span></footer>
    </main>
  );
}
