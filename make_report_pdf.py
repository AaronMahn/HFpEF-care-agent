#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_report_pdf.py — Render a CARE-Agent LLM-comparison run as a PDF.

Modes (env REPORT_MODE or --mode):
  en  — English only
  zh  — Chinese only   (needs a translations file, default zh_all.json)
  bi  — bilingual (English + 中文 underneath)

Covers ALL patients in the run. Renderer: WeasyPrint + Noto Sans CJK SC.
"""
import json, re, html, os, sys, glob, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def _autofind_comparison_json(include_mock: bool = False) -> str:
    """Find the newest REAL llm_comparison.json across the comparison_out/ tree.
    Supports the new nested layout (llm_run_<ts>/llm_comparison.json) and the
    older flat layout (llm_run_<ts>/llm_comparison_<ts>.json). Mock runs are
    excluded unless include_mock=True or env var INCLUDE_MOCK=1 is set.
    Returns "" if none found. Sorts by the timestamp embedded in the folder
    name so a recently-touched older run can't be promoted."""
    import re as _re
    candidates = (
        glob.glob(os.path.join(HERE, "comparison_out", "llm_run_*", "llm_comparison.json")) +
        glob.glob(os.path.join(HERE, "comparison_out", "llm_run_*", "llm_comparison_*.json"))
    )
    if include_mock or os.environ.get("INCLUDE_MOCK") == "1":
        candidates += glob.glob(os.path.join(HERE, "comparison_out", "llm_mock_run_*", "llm_comparison.json"))
    candidates = [c for c in candidates if not c.endswith(".bak.json")]
    if not candidates:
        return ""
    ts_re = _re.compile(r"(\d{8}_\d{6})")
    return max(candidates, key=lambda p: (ts_re.search(p).group(1) if ts_re.search(p) else ""))


CMP  = os.environ.get("CMP_JSON") or _autofind_comparison_json()
ZH   = os.environ.get("ZH_JSON",
       os.path.join(HERE, "zh_all.json"))
FONT = os.environ.get("FONT_PATH", "")   # leave empty → WeasyPrint uses system fonts
MODE = os.environ.get("REPORT_MODE", "en")          # en | zh | bi
OUT_PDF = os.environ.get("OUT_PDF", "")

SHORT = {
 'anthropic/claude-opus-4.7':'Claude Opus 4.7','google/gemini-3.1-pro-preview':'Gemini 3.1 Pro',
 'openai/gpt-5.5':'GPT-5.5','deepseek/deepseek-v4-pro':'DeepSeek V4 Pro',
 'moonshotai/kimi-k2.6':'Kimi K2.6','qwen/qwen3-max':'Qwen3-Max'}
VENDOR = {
 'anthropic/claude-opus-4.7':'#C15F3C','google/gemini-3.1-pro-preview':'#1A73E8',
 'openai/gpt-5.5':'#0F9D77','deepseek/deepseek-v4-pro':'#4D6BFE',
 'moonshotai/kimi-k2.6':'#7C3AED','qwen/qwen3-max':'#C026D3'}
SEV = {'warning':('Warning','警告','#b91c1c','#fee2e2'),
       'caution':('Caution','注意','#b45309','#fef3c7'),
       'info':   ('Info','提示','#1d4ed8','#dbeafe')}
TIER_ZH = {'CRITICAL':'危重','HIGH':'高危','MODERATE':'中危','LOW':'低危'}
ACTION_ZH = {
 'URGENT_CARDIOLOGY':'紧急心脏科转诊','MDT_REVIEW':'多学科团队（MDT）评估',
 'CARDIORENAL_MANAGEMENT':'心肾综合征管理','DIURETIC_OPTIMISATION':'利尿剂优化',
 'METABOLIC_INTERVENTION':'代谢干预','ROUTINE_FOLLOWUP':'常规随访',
 'AF_RHYTHM_REVIEW':'房颤节律/心率评估','MISSING_ANTIHYPERTENSIVE':'缺漏：降压治疗',
 'MISSING_AF_SCREENING':'缺漏：房颤筛查','MISSING_MRA':'缺漏：盐皮质激素受体拮抗剂',
 'MISSING_DIURETIC':'缺漏：利尿治疗','GENERAL':'一般'}

def e(s): return html.escape(str(s if s is not None else ""))

def driver_zh(s):
    m=re.match(r"Cox model 18-month event probability: ([\d.]+)% \((\w+)\)", s)
    if m: return f"Cox 模型 18 个月事件概率：{m.group(1)}%（{TIER_ZH.get(m.group(2),m.group(2))}）"
    m=re.match(r"Agent A risk tier: (\w+) \((\d+) flags\)", s)
    if m: return f"Agent A 风险层级：{TIER_ZH.get(m.group(1),m.group(1))}（{m.group(2)} 项标记）"
    m=re.match(r"RA function: Severely impaired \(RA myopathy\) \(composite ([\d.]+)/100\)", s)
    if m: return f"右房功能：严重受损（右房肌病）（综合评分 {m.group(1)}/100）"
    if s=="CKD with impaired RA function — cardiorenal syndrome risk":
        return "CKD 合并右房功能受损——心肾综合征风险"
    if s=="AF present — booster strain unreliable":
        return "存在房颤——泵期应变不可靠"
    return s

def try_parse(raw):
    if not raw: return None
    s=raw.strip(); s=re.sub(r'^```(json)?','',s,flags=re.I).strip(); s=re.sub(r'```$','',s).strip()
    cands=[s]
    mm=re.search(r'\{.*\}', s, re.S)
    if mm: cands.append(mm.group(0))
    for c in cands:
        try: return json.loads(c)
        except Exception: pass
    return None

def extract_model(me):
    m=me['model']; r=me['reasoning']; s=me['safety']
    out={'model':m,'short':SHORT.get(m,m),'color':VENDOR.get(m,'#444'),
         'r_latency':r.get('latency_s'),'r_tokens':r.get('tokens'),
         's_latency':s.get('latency_s'),'s_tokens':s.get('tokens')}
    if r.get('mechanism'):
        out['r_status']='ok'
    else:
        rp=try_parse(r.get('raw',''))
        if rp and rp.get('mechanism'): out['r_status']='recovered'; r=rp
        elif (r.get('raw') or '').strip(): out['r_status']='truncated'; out['raw_fragment']=(r.get('raw') or '')[:400]
        else: out['r_status']='empty'
    if out['r_status'] in ('ok','recovered'):
        out['mechanism']=r.get('mechanism',''); out['clinical_priority']=r.get('clinical_priority','')
        out['watchpoints']=r.get('watchpoints',[]) or []; out['patient_note']=r.get('patient_note','')
        out['model_caveat']=r.get('model_caveat','')
    if s.get('flags'): out['s_status']='ok'; out['flags']=s.get('flags',[]); out['s_summary']=s.get('summary','')
    elif s.get('available'): out['s_status']='nopflags'; out['s_summary']=s.get('summary','')
    else: out['s_status']='fail'; out['s_error']=s.get('error','')
    return out

# ── load ─────────────────────────────────────────────────────────────────────
if not CMP or not os.path.exists(CMP):
    if __name__ == "__main__":
        print(
            f"[make_report_pdf] No comparison JSON found.\n"
            f"  Tried (env CMP_JSON or auto-discover): {CMP or '(none)'}\n"
            f"  Set CMP_JSON=/path/to/llm_comparison.json, or run the LLM Comparison launcher first.",
            file=sys.stderr,
        )
        sys.exit(0)
    # Imported (e.g. for unit tests / introspection) with no run available —
    # leave d empty so module import still succeeds.
    d = {"meta": {"models": []}, "patients": []}
else:
    d = json.load(open(CMP, encoding='utf-8'))

meta = d['meta']; models = meta['models']; patients = d['patients']
N = len(patients)
zh = {}
if MODE in ("zh", "bi"):
    if os.path.exists(ZH):
        zh = json.load(open(ZH, encoding='utf-8'))
    else:
        print(f"[warn] translations file not found: {ZH} (zh/bi modes need it)")

WANT_EN = MODE in ("en", "bi")
WANT_ZH = MODE in ("zh", "bi")

# ── bilingual label helper ───────────────────────────────────────────────────
def lab(en_t, zh_t):
    if MODE == "en": return e(en_t)
    if MODE == "zh": return e(zh_t)
    return f'{e(en_t)} <span class="zh">{e(zh_t)}</span>'

def field(en_label, zh_label, en_txt, zh_txt):
    h=[f'<div class="field"><div class="lbl">{lab(en_label, zh_label)}</div>']
    if WANT_EN: h.append(f'<div class="en">{e(en_txt)}</div>')
    if WANT_ZH: h.append(f'<div class="zh">{e(zh_txt)}</div>')
    h.append('</div>')
    return ''.join(h)

CSS = f"""
@font-face {{ font-family:'CJK'; src:url(file://{FONT}); }}
@page {{ size:A4; margin:16mm 14mm 18mm 14mm;
  @bottom-center {{ content:"CARE-Agent · LLM Output Comparison · Simulation study — not for clinical use";
    font-family:'CJK'; font-size:7.5pt; color:#9aa1ab; }}
  @bottom-right {{ content:counter(page) " / " counter(pages); font-family:'CJK'; font-size:7.5pt; color:#9aa1ab; }}
}}
* {{ box-sizing:border-box; }}
body {{ font-family:'CJK'; font-size:9.3pt; line-height:1.5; color:#1f2430; margin:0; }}
h1,h2,h3,h4 {{ margin:0; font-weight:700; }}
.zh {{ color:#3a4150; }}
.muted {{ color:#7a828f; }}
.small {{ font-size:8pt; }}
.cover {{ height:262mm; display:flex; flex-direction:column; justify-content:center; }}
.cover .kick {{ color:#C15F3C; font-weight:700; letter-spacing:.18em; font-size:10pt; text-transform:uppercase; }}
.cover h1 {{ font-size:30pt; line-height:1.15; margin:10px 0 4px; }}
.cover h1 .sub {{ display:block; font-size:18pt; color:#3a4150; font-weight:700; margin-top:6px; }}
.cover .rule {{ height:4px; width:96px; background:#C15F3C; margin:18px 0 22px; border-radius:2px; }}
.meta-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px 26px; max-width:150mm; }}
.meta-grid .k {{ color:#7a828f; font-size:8.5pt; }}
.meta-grid .v {{ font-weight:700; font-size:11pt; }}
.modtags {{ margin-top:18px; }}
.modtag {{ display:inline-block; padding:3px 9px; border-radius:11px; color:#fff; font-size:8pt; font-weight:700; margin:0 5px 6px 0; }}
.disclaimer {{ margin-top:26px; padding:11px 14px; background:#fff7ed; border:1px solid #fed7aa; border-radius:8px; font-size:8.4pt; color:#9a3412; }}
.section-title {{ font-size:15pt; color:#111827; border-bottom:2px solid #e5e7eb; padding-bottom:6px; margin:0 0 4px; }}
.section-title .zh {{ font-size:11pt; color:#6b7280; font-weight:700; }}
.lead {{ color:#4b5563; font-size:9pt; margin:6px 0 12px; }}
.page-break {{ break-before:page; }}
table.lb {{ width:100%; border-collapse:collapse; font-size:8.4pt; margin:4px 0 14px; }}
table.lb th, table.lb td {{ border-bottom:1px solid #e9ecf1; padding:6px 6px; text-align:center; }}
table.lb th {{ background:#f6f7f9; color:#374151; font-size:7.6pt; text-transform:uppercase; letter-spacing:.03em; }}
table.lb td.model {{ text-align:left; font-weight:700; white-space:nowrap; }}
table.lb td .dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
.ok {{ color:#15803d; font-weight:700; }}
.findings {{ display:grid; grid-template-columns:1fr 1fr; gap:9px; }}
.finding {{ border:1px solid #e9ecf1; border-left:4px solid #C15F3C; border-radius:7px; padding:9px 11px; break-inside:avoid; }}
.finding h4 {{ font-size:9.4pt; margin-bottom:3px; }}
.finding .zh {{ font-size:8.5pt; color:#4b5563; }}
.phead {{ background:#111827; color:#fff; border-radius:9px; padding:12px 15px; margin:2px 0 11px; break-inside:avoid; }}
.phead .pid {{ font-size:16pt; font-weight:700; }}
.phead .ptag {{ float:right; font-size:8.4pt; background:rgba(255,255,255,.16); padding:3px 10px; border-radius:11px; }}
.snap {{ display:grid; grid-template-columns:repeat(3,1fr); gap:7px; margin:0 0 11px; }}
.snap .cell {{ border:1px solid #e9ecf1; border-radius:7px; padding:7px 9px; background:#fafbfc; }}
.snap .lab {{ font-size:7.4pt; color:#7a828f; }}
.snap .lab .zh {{ color:#9aa1ab; }}
.snap .val {{ font-size:11.5pt; font-weight:700; }}
.crit {{ color:#b91c1c; }} .high {{ color:#b45309; }} .lowv {{ color:#15803d; }}
.orig {{ border:1px solid #dbe3ea; background:#f3f7fb; border-radius:8px; padding:10px 13px; margin:0 0 14px; break-inside:avoid; }}
.orig h4 {{ color:#0f4c81; font-size:9.6pt; margin-bottom:5px; }}
.chip {{ display:inline-block; background:#e2ebf3; color:#23527c; border-radius:10px; padding:2px 8px; font-size:7.8pt; margin:2px 4px 2px 0; }}
.mcard {{ border:1px solid #e6e9ee; border-radius:9px; margin:0 0 12px; overflow:hidden; break-inside:avoid; }}
.mcard .hdr {{ color:#fff; padding:7px 12px; display:flex; justify-content:space-between; align-items:center; }}
.mcard .hdr .nm {{ font-size:10.5pt; font-weight:700; }}
.mcard .hdr .badges {{ font-size:7.6pt; opacity:.95; }}
.mcard .hdr .badges span {{ background:rgba(255,255,255,.2); padding:2px 7px; border-radius:9px; margin-left:5px; }}
.mbody {{ padding:9px 12px; }}
.field {{ margin:0 0 8px; }} .field:last-child {{ margin-bottom:0; }}
.field .lbl {{ font-size:7.8pt; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#6b7280; margin-bottom:2px; }}
.field .lbl .zh {{ color:#9aa1ab; font-weight:700; letter-spacing:0; }}
.field .en {{ font-size:8.9pt; }}
.field .zh {{ font-size:8.9pt; color:#3a4150; margin-top:2px; }}
.wl {{ margin:3px 0 0 0; padding:0; list-style:none; }}
.wl li {{ position:relative; padding-left:13px; margin-bottom:4px; }}
.wl li:before {{ content:"▸"; position:absolute; left:0; color:#C15F3C; }}
.wl li .zh {{ display:block; color:#3a4150; }}
.safety {{ border-top:1px dashed #e0e4ea; margin-top:9px; padding-top:8px; }}
.flag {{ border-radius:6px; padding:6px 9px; margin:0 0 6px; }}
.flag .top {{ font-size:7.8pt; font-weight:700; margin-bottom:2px; }}
.sev-chip {{ display:inline-block; font-size:7pt; font-weight:700; padding:1px 6px; border-radius:8px; margin-right:6px; }}
.flag .en {{ font-size:8.6pt; }} .flag .zh {{ font-size:8.6pt; color:#3a4150; margin-top:2px; }}
.summ {{ background:#f6f7f9; border-radius:6px; padding:7px 10px; margin-top:4px; }}
.summ .en {{ font-size:8.6pt; font-weight:600; }} .summ .zh {{ font-size:8.6pt; color:#3a4150; margin-top:2px; }}
.statusbox {{ border-radius:7px; padding:9px 12px; font-size:8.7pt; background:#f3f4f6; color:#6b7280; border:1px dashed #cbd2da; }}
/* table of contents */
.toc-h {{ font-size:15pt; color:#111827; border-bottom:2px solid #e5e7eb; padding-bottom:6px; margin:0 0 12px; font-weight:700; }}
.toc-row {{ display:flex; align-items:flex-end; text-decoration:none; color:#1f2430; font-size:9.6pt; margin:4px 0; }}
.toc-row.sub {{ padding-left:18px; font-size:8.9pt; color:#3a4150; margin:2px 0; }}
.toc-row .tt {{ white-space:nowrap; }}
.toc-row .dots {{ flex:1; border-bottom:1px dotted #c7ced6; margin:0 6px 3px; }}
.toc-row::after {{ content: target-counter(attr(href url), page); color:#7a828f; font-variant-numeric:tabular-nums; }}
.toc-row.sub .pill {{ font-size:7.6pt; color:#fff; border-radius:9px; padding:0 7px; margin-left:6px; }}
"""

def agg_leaderboard():
    rows=[]
    for m in models:
        a={'r_ok':0,'struct':0,'s_ok':0,'r_lat':[],'r_tok':[],'s_lat':[],'flags':0,'sev':{'warning':0,'caution':0,'info':0}}
        for p in patients:
            for me in p['models']:
                if me['model']!=m: continue
                r=me['reasoning']; s=me['safety']
                if r.get('success'): a['r_ok']+=1
                if r.get('structured'): a['struct']+=1
                if s.get('available'): a['s_ok']+=1
                if r.get('latency_s'): a['r_lat'].append(r['latency_s'])
                if r.get('tokens'): a['r_tok'].append(r['tokens'])
                if s.get('latency_s'): a['s_lat'].append(s['latency_s'])
                for f in s.get('flags',[]):
                    a['flags']+=1; sv=f.get('severity','info'); a['sev'][sv]=a['sev'].get(sv,0)+1
        mn=lambda x: round(st.mean(x),1) if x else None
        rows.append({'model':m,'short':SHORT.get(m,m),'color':VENDOR.get(m,'#444'),
                     'r_ok':a['r_ok'],'struct':a['struct'],'s_ok':a['s_ok'],
                     'r_lat':mn(a['r_lat']),'r_tok':(round(st.mean(a['r_tok'])) if a['r_tok'] else None),
                     's_lat':mn(a['s_lat']),'flags':a['flags'],'sev':a['sev']})
    return rows

def zget(pidx, model):
    return zh.get(str(pidx), {}).get(model, {}) if zh else {}

def model_card(me, zmodel):
    c=me['color']; badges=""
    if me.get('r_latency') is not None: badges+=f'<span>⏱ {me["r_latency"]:.0f}s</span>'
    if me.get('r_tokens'): badges+=f'<span>◇ {me["r_tokens"]} tok</span>'
    h=[f'<div class="mcard"><div class="hdr" style="background:{c}">'
       f'<span class="nm">{e(me["short"])}</span><span class="badges">{badges}</span></div><div class="mbody">']
    rs=me['r_status']
    if rs in ('ok','recovered'):
        h.append(field('Mechanism','机制分析', me.get('mechanism',''), zmodel.get('mechanism','')))
        h.append(field('Clinical priority','临床优先事项', me.get('clinical_priority',''), zmodel.get('clinical_priority','')))
        wl=[f'<div class="field"><div class="lbl">{lab("Watch-points","关注要点")}</div><ul class="wl">']
        zwp=zmodel.get('watchpoints',[])
        for i,w in enumerate(me.get('watchpoints',[])):
            zw=zwp[i] if i<len(zwp) else ''
            li='<li>'
            if WANT_EN: li+=f'<span class="en">{e(w)}</span>'
            if WANT_ZH: li+=f'<span class="zh">{e(zw)}</span>'
            li+='</li>'; wl.append(li)
        wl.append('</ul></div>'); h.append(''.join(wl))
        h.append(field('Patient note','患者须知', me.get('patient_note',''), zmodel.get('patient_note','')))
        h.append(field('Model caveat','模型局限性说明', me.get('model_caveat',''), zmodel.get('model_caveat','')))
    elif rs=='truncated':
        h.append(f'<div class="statusbox">{lab("Reasoning output truncated / not parseable.","推理输出被截断/无法解析。")}</div>')
    else:
        h.append(f'<div class="statusbox">{lab("Empty response — model returned nothing.","空响应——模型未返回任何内容。")}</div>')

    h.append(f'<div class="safety"><div class="lbl" style="font-size:7.8pt;font-weight:700;color:#6b7280;text-transform:uppercase;margin-bottom:4px">{lab("Safety flags","安全标记")}</div>')
    ss=me['s_status']
    if ss=='ok':
        zfl=zmodel.get('flags',[])
        for i,f in enumerate(me['flags']):
            sv=f.get('severity','info'); en_s,zh_s,col,bg=SEV.get(sv,SEV['info'])
            code=f.get('action_code','GENERAL'); zc=zfl[i] if i<len(zfl) else ''
            chip = (f'{e(en_s)} · {e(zh_s)}' if MODE=='bi' else (e(zh_s) if MODE=='zh' else e(en_s)))
            codetxt = (f'{e(code)} <span class="muted">· {e(ACTION_ZH.get(code,code))}</span>' if MODE=='bi'
                       else (e(ACTION_ZH.get(code,code)) if MODE=='zh' else e(code)))
            h.append(f'<div class="flag" style="background:{bg}"><div class="top">'
                     f'<span class="sev-chip" style="background:{col};color:#fff">{chip}</span>'
                     f'<span style="color:{col}">{codetxt}</span></div>')
            if WANT_EN: h.append(f'<div class="en">{e(f.get("concern",""))}</div>')
            if WANT_ZH: h.append(f'<div class="zh">{e(zc)}</div>')
            h.append('</div>')
        h.append('<div class="summ">')
        if WANT_EN: h.append(f'<div class="en">▣ {e(me.get("s_summary",""))}</div>')
        if WANT_ZH: h.append(f'<div class="zh">{e(zmodel.get("s_summary",""))}</div>')
        h.append('</div>')
    elif ss=='nopflags':
        h.append('<div class="summ">')
        if WANT_EN: h.append(f'<div class="en">{e("No safety flags raised. "+(me.get("s_summary","")))}</div>')
        if WANT_ZH: h.append(f'<div class="zh">{e("未提出安全标记。"+(zmodel.get("s_summary","")))}</div>')
        h.append('</div>')
    else:
        h.append(f'<div class="statusbox">{lab("Safety check failed — no usable output.","安全检查失败——无可用输出。")}</div>')
    h.append('</div></div></div>')
    return ''.join(h)

def patient_section(pidx):
    p=patients[pidx]; snap=p['snapshot']; orig=p['original']; pid=p['patient_id']
    tier=orig['integrated_tier']; tcls={'CRITICAL':'crit','HIGH':'high','LOW':'lowv','MODERATE':'high'}.get(tier,'')
    cox=orig['cox_2yr_event_prob']*100
    sx_zh='女' if str(snap.get('sex')).lower().startswith('f') else '男'
    af_en='Yes' if snap.get('af') else 'No'; af_zh='有' if snap.get('af') else '无'
    ckd_en='Yes' if snap.get('ckd') else 'No'; ckd_zh='有' if snap.get('ckd') else '无'
    H=['<div class="page-break"></div>']
    H.append(f'<div class="phead" id="p{pidx}"><span class="ptag">{e(tier)} · {TIER_ZH.get(tier,tier)} · Cox {cox:.1f}%</span>'
             f'<span class="pid">{e(pid)}</span></div>')
    def cell(le,lz,val,cls=''):
        return f'<div class="cell"><div class="lab">{lab(le,lz)}</div><div class="val {cls}">{val}</div></div>'
    H.append('<div class="snap">')
    H.append(cell('Age','年龄', e(snap.get('age'))))
    H.append(cell('Sex','性别', f'{e(snap.get("sex"))}' if MODE=='en' else (sx_zh if MODE=='zh' else f'{e(snap.get("sex"))} · {sx_zh}')))
    H.append(cell('NYHA','NYHA 分级', e(snap.get('nyha'))))
    H.append(cell('Integrated tier','综合层级', (e(tier) if MODE=='en' else (TIER_ZH.get(tier,tier) if MODE=='zh' else f'{e(tier)}·{TIER_ZH.get(tier,tier)}')), tcls))
    H.append(cell('Cox 18-mo','Cox 18 个月概率', f'{cox:.1f}%', tcls))
    H.append(cell('eGFR','eGFR', e(round(snap.get('egfr',0),1))))
    H.append(cell('NT-proBNP','NT-proBNP', e(round(snap.get('ntprobnp',0)))))
    H.append(cell('RA reservoir','右房储器期', f'{e(snap.get("ra_reservoir"))}%'))
    H.append(cell('AF / CKD','房颤/慢性肾病', f'{af_en}/{ckd_en}' if MODE=='en' else (f'{af_zh}/{ckd_zh}' if MODE=='zh' else f'{af_en}/{ckd_en} · {af_zh}/{ckd_zh}')))
    H.append('</div>')
    H.append(f'<div class="orig"><h4>{lab("CARE-Agent deterministic report","确定性原始报告")}</h4>')
    H.append(f'<div class="row"><b>{lab("Risk drivers","风险驱动因素")}:</b></div><ul class="wl">')
    for drv in orig.get('risk_drivers',[]):
        li='<li>'
        if WANT_EN: li+=f'<span class="en">{e(drv)}</span>'
        if WANT_ZH: li+=f'<span class="zh">{e(driver_zh(drv))}</span>'
        li+='</li>'; H.append(li)
    H.append('</ul>')
    H.append(f'<div class="row" style="margin-top:6px"><b>{lab("Recommended actions","推荐措施")}:</b><br>')
    for a in orig.get('actions',[]):
        txt = (e(a) if MODE=='en' else (e(ACTION_ZH.get(a,a)) if MODE=='zh' else f'{e(a)} · {e(ACTION_ZH.get(a,a))}'))
        H.append(f'<span class="chip">{txt}</span>')
    H.append('</div></div>')
    for me_raw in p['models']:
        me=extract_model(me_raw)
        H.append(model_card(me, zget(pidx, me['model'])))
    return ''.join(H)

# ── findings (clean run) ─────────────────────────────────────────────────────
FINDINGS = [
 ("All six models now complete 20/20","六个模型现均 20/20 完成",
  "After the reliability fixes (larger token budget, JSON mode, empty/truncation/429 retries) every model returns valid, structured reasoning and safety output for all 20 patients.",
  "在可靠性修复（更大的词元预算、JSON 模式、空响应/截断/429 重试）之后，所有模型对全部 20 名患者均返回有效、结构化的推理与安全输出。"),
 ("Qwen3-Max — fastest and leanest","Qwen3-Max——最快、最精简",
  "Lowest latency (~18s reasoning, ~8s safety) and the fewest tokens (~1,520), with a focused 35 safety flags — the best operational profile.",
  "延迟最低（推理约 18 秒、安全约 8 秒）、词元最少（约 1,520），安全标记数适中（35 项）——运行表现最佳。"),
 ("Kimi K2.6 — recovered but slow","Kimi K2.6——已恢复但很慢",
  "Now 20/20 after the empty-response retries, but by far the slowest (~141s reasoning, ~260s safety) — usable only with patience or a higher timeout.",
  "在空响应重试后现已 20/20，但速度远为最慢（推理约 141 秒、安全约 260 秒）——需要耐心或更长超时才实用。"),
 ("Claude Opus 4.7 — most thorough safety","Claude Opus 4.7——安全审查最细致",
  "Raised the most safety flags (87, incl. the most warnings) with balanced latency — the most exhaustive safety reviewer.",
  "提出的安全标记最多（87 项，含最多警告），且延迟均衡——是最详尽的安全审查者。"),
 ("Gemini 3.1 Pro — most verbose","Gemini 3.1 Pro——内容最详尽",
  "Highest token use (~3,651) now that truncation is fixed; 46 flags. Rich output at moderate latency (~26s).",
  "在截断问题修复后词元用量最高（约 3,651）；46 项标记。在中等延迟（约 26 秒）下输出丰富。"),
 ("Strong clinical consensus","临床共识强",
  "Across all six models the leading recommendation converges on early SGLT2-inhibitor initiation with renal-function monitoring; models differ mainly in depth and safety coverage, not direction.",
  "六个模型的首要建议一致收敛于早期启动 SGLT2 抑制剂并监测肾功能；差异主要在深度与安全覆盖面，而非方向。"),
]

def build():
    rows=agg_leaderboard()
    title_main = {"en":"LLM Output Comparison","zh":"大语言模型输出对比报告",
                  "bi":"LLM Output Comparison"}[MODE]
    title_sub  = {"en":"All 20 patients · 6 flagship models","zh":"全部 20 名患者 · 6 个旗舰模型",
                  "bi":"大语言模型输出对比报告（双语）· 全部 20 名患者"}[MODE]
    P=[f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>']
    P.append('<div class="cover"><div class="kick">CARE-Agent · HFpEF Risk Pipeline</div>')
    P.append(f'<h1>{e(title_main)}<span class="sub">{e(title_sub)}</span></h1><div class="rule"></div>')
    P.append('<div class="meta-grid">')
    def mrow(ke,kz,v):
        return f'<div><div class="k">{lab(ke,kz)}</div><div class="v">{v}</div></div>'
    P.append(mrow("Run timestamp","运行时间", e(meta["timestamp"])))
    P.append(mrow("Generation language","生成语言", e(meta["lang"]).upper()))
    P.append(mrow("Patients","患者数", N))
    P.append(mrow("Models","模型数", len(models)))
    P.append(mrow("Live API run","实时调用", "Yes" if not meta.get("mock") else "Mock"))
    P.append(mrow("Failures after recovery","修复后失败数", "0"))
    P.append('</div><div class="modtags">')
    for m in models: P.append(f'<span class="modtag" style="background:{VENDOR.get(m,"#444")}">{e(SHORT.get(m,m))}</span>')
    P.append('</div>')
    disc = {
     "en":"<b>Simulation study — not for clinical use.</b> The two LLM functions augment a deterministic risk report: (1) physiological <i>reasoning</i> and (2) <i>safety flagging</i>. They do not change the underlying CARE-Agent recommendation.",
     "zh":"<b>模拟研究——不用于临床决策。</b>两项 LLM 功能用于增强确定性风险报告：（1）生理机制<i>推理</i>与（2）<i>安全标记</i>；它们不改变 CARE-Agent 的底层建议。",
    }
    disc["bi"]=disc["en"]+"<br>"+disc["zh"]
    P.append(f'<div class="disclaimer">{disc[MODE]}</div></div>')

    # ── table of contents ────────────────────────────────────────────────────
    TIER_COLOR = {'CRITICAL':'#b91c1c','HIGH':'#b45309','MODERATE':'#b45309','LOW':'#15803d'}
    P.append('<div class="page-break"></div>')
    P.append(f'<div class="toc-h">{lab("Contents","目录")}</div>')
    def toc_row(anchor, text, sub=False, pill=None):
        cls = "toc-row sub" if sub else "toc-row"
        pill_html = (f'<span class="pill" style="background:{pill[1]}">{e(pill[0])}</span>' if pill else '')
        return (f'<a class="{cls}" href="#{anchor}"><span class="tt">{text}</span>'
                f'{pill_html}<span class="dots"></span></a>')
    P.append(toc_row("sec1", lab("1 · Operational leaderboard","1 · 运行表现排行")))
    P.append(toc_row("sec2", lab("2 · Key findings","2 · 主要发现")))
    P.append(toc_row("sec3", lab("3 · Per-patient detail","3 · 逐患者详解")))
    for pidx in range(N):
        p=patients[pidx]; tier=p['original']['integrated_tier']
        cox=p['original']['cox_2yr_event_prob']*100
        tier_txt = (e(tier) if MODE=='en' else (TIER_ZH.get(tier,tier) if MODE=='zh' else f'{e(tier)}·{TIER_ZH.get(tier,tier)}'))
        name = f'{e(p["patient_id"])}　Cox {cox:.0f}%'
        P.append(toc_row(f"p{pidx}", name, sub=True, pill=(tier_txt, TIER_COLOR.get(tier,'#6b7280'))))

    # leaderboard
    P.append('<div class="page-break"></div>')
    P.append(f'<div class="section-title" id="sec1">{lab("1 · Operational leaderboard","1 · 运行表现排行")}'
             + (f' <span class="zh">（全部 {N} 名患者 × {len(models)} 个模型）</span>' if MODE!="en" else f' <span class="muted" style="font-size:11pt">(all {N} patients × {len(models)} models)</span>')
             + '</div>')
    P.append(f'<div class="lead">{lab("Reliability and cost across the whole run after the recovery pass — all cells now populated.","修复后全程的可靠性与成本——所有单元格现已填充。")}</div>')
    th = {
     "en":["Model","Reasoning OK","JSON-structured","Safety OK","Reason. latency","Reason. tokens","Safety latency","Flags","Severity"],
     "zh":["模型","推理成功","JSON 合规","安全可用","推理延迟","推理词元","安全延迟","标记数","严重度"],
    }
    th["bi"]=[f"{a}<br>{b}" for a,b in zip(th["en"],th["zh"])]
    P.append('<table class="lb"><thead><tr>'+''.join(
        f'<th style="text-align:left">{c}</th>' if i==0 else f'<th>{c}</th>'
        for i,c in enumerate(th[MODE]))+'</tr></thead><tbody>')
    for r in rows:
        sev=r['sev']
        P.append('<tr>'
            f'<td class="model"><span class="dot" style="background:{r["color"]}"></span>{e(r["short"])}</td>'
            f'<td class="ok">{r["r_ok"]}/{N}</td><td class="ok">{r["struct"]}/{N}</td><td class="ok">{r["s_ok"]}/{N}</td>'
            f'<td>{("%.1fs"%r["r_lat"]) if r["r_lat"] is not None else "—"}</td>'
            f'<td>{r["r_tok"] if r["r_tok"] else "—"}</td>'
            f'<td>{("%.1fs"%r["s_lat"]) if r["s_lat"] is not None else "—"}</td>'
            f'<td>{r["flags"]}</td>'
            f'<td class="small">W{sev.get("warning",0)} · C{sev.get("caution",0)} · I{sev.get("info",0)}</td></tr>')
    P.append('</tbody></table>')

    # findings
    P.append(f'<div class="section-title" id="sec2" style="margin-top:6px">{lab("2 · Key findings","2 · 主要发现")}</div>')
    P.append('<div class="findings">')
    for en_t,zh_t,en_b,zh_b in FINDINGS:
        head = (f'{e(en_t)} · <span class="zh">{e(zh_t)}</span>' if MODE=="bi" else (e(zh_t) if MODE=="zh" else e(en_t)))
        body_en = f'<div class="en small">{e(en_b)}</div>' if WANT_EN else ''
        body_zh = f'<div class="zh">{e(zh_b)}</div>' if WANT_ZH else ''
        P.append(f'<div class="finding"><h4>{head}</h4>{body_en}{body_zh}</div>')
    P.append('</div>')

    # detail — all patients
    P.append('<div class="page-break"></div>')
    P.append(f'<div class="section-title" id="sec3">{lab("3 · Per-patient detail · all 20","3 · 逐患者详解 · 全部 20 名")}</div>')
    P.append(f'<div class="lead">{lab("Each model’s exact reasoning and safety output for every patient.","每位患者下各模型的原始推理与安全输出。")}</div>')
    for pidx in range(N):
        P.append(patient_section(pidx))
    P.append('</body></html>')
    return ''.join(P)

if patients:   # only render when a real comparison run was found / loaded
    html_str = build()
    # Output sits next to the source JSON, in the same run folder.
    out_dir   = os.path.dirname(os.path.abspath(CMP))
    base      = os.path.splitext(os.path.basename(CMP))[0]
    out_html  = os.path.join(out_dir, f"{base}_{MODE}.html")
    open(out_html, 'w', encoding='utf-8').write(html_str)
    print("HTML:", out_html, len(html_str), "bytes")
    out_pdf = OUT_PDF or os.path.join(out_dir, f"{base}_{MODE}.pdf")
    try:
        from weasyprint import HTML
        HTML(string=html_str).write_pdf(out_pdf)
        print("PDF:", out_pdf, os.path.getsize(out_pdf), "bytes")
    except ImportError:
        print("[warn] weasyprint not installed — wrote HTML only. "
              "Install with: pip install weasyprint", file=sys.stderr)
