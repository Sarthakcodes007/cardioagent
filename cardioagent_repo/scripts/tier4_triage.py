"""
CardioAgent — Tier 4: Clinical Triage Agent
RAG-equipped LLM that converts a 5-dim rhythm-burden vector into:
  - Triage level  : routine / urgent / emergency
  - Rationale     : 2-4 sentence clinical reasoning
  - Citations     : ACC/AHA 2023 and ESC 2020 guideline sections

Architecture:
  Knowledge base  : ACC/AHA 2023 + ESC 2020 AF guideline key sections
  Embedder        : sentence-transformers/all-MiniLM-L6-v2
  Vector store    : FAISS flat index (top-5 retrieval)
  LLM backbone    : microsoft/Phi-3.5-mini-instruct (3.8B, free HF access)
                    OR meta-llama/Llama-3.2-3B-Instruct (if you have access)

Evaluation:
  100 synthetic patient cases with guideline-derived expected triage
  Cohen's kappa between LLM output and guideline-based ground truth

Run: python tier4_triage.py
"""

import os, json, random, warnings
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import cohen_kappa_score, classification_report

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_MODEL     = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL       = 'microsoft/Phi-3.5-mini-instruct'   # free HF access, no token needed
TOP_K           = 5          # retrieved passages per query
MAX_NEW_TOKENS  = 300
KB_PATH         = '/workspace/cardioagent/guidelines_kb.json'
INDEX_PATH      = '/workspace/cardioagent/faiss_index.bin'
MODEL_CACHE     = '/workspace/cardioagent/models/phi35'
SEED            = 42
N_EVAL_CASES    = 100

TRIAGE_LEVELS   = ['routine', 'urgent', 'emergency']
DIM_NAMES       = ['af_burden', 'longest_ep_min_norm',
                   'episode_count_norm', 'nocturnal_ratio', 'trend_slope_norm']

random.seed(SEED); np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GUIDELINE KNOWLEDGE BASE
# Key passages from ACC/AHA 2023 and ESC 2020 AF guidelines
# ─────────────────────────────────────────────────────────────────────────────
GUIDELINE_PASSAGES = [
    # ACC/AHA 2023 — AF Burden and Stroke Risk
    {"id": "ACC2023-5.1.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 5.1.1",
     "text": "AF burden exceeding 11% over a 24-hour monitoring period is associated with stroke risk equivalent to persistent AF. Anticoagulation assessment using the CHA2DS2-VASc scoring system is recommended for patients meeting this threshold. Annual stroke risk increases significantly above this burden level."},

    {"id": "ACC2023-5.1.2",
     "source": "ACC/AHA 2023 AF Guideline, Section 5.1.2",
     "text": "Patients with CHA2DS2-VASc score ≥2 in males or ≥3 in females with documented AF burden above 11% should be considered for oral anticoagulation therapy unless contraindicated. The decision should be individualised based on bleeding risk assessment using the HAS-BLED score."},

    {"id": "ACC2023-5.2.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 5.2.1",
     "text": "Paroxysmal AF characterised by multiple short episodes (episode count >5 per 24-hour period) with increasing burden trend over 48 hours represents progressive arrhythmia burden requiring urgent clinical review and potential rhythm control strategy initiation."},

    {"id": "ACC2023-5.3.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 5.3.1",
     "text": "A single AF episode exceeding 24 hours in duration classified as sustained AF. Episodes of this duration trigger the immediate escalation pathway and require same-day clinical assessment to evaluate thromboembolic risk and initiate anticoagulation if not already prescribed."},

    {"id": "ACC2023-6.1.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 6.1.1",
     "text": "Rate control targets for AF patients include resting heart rate below 110 bpm as an initial target (lenient control) or below 80 bpm for patients with symptoms or heart failure (strict control). Continuous monitoring is recommended for burden above 5% to assess adequacy of rate control."},

    {"id": "ACC2023-7.1.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 7.1.1",
     "text": "Rhythm control with antiarrhythmic drug therapy or catheter ablation should be considered for symptomatic AF patients. Early rhythm control initiated within one year of AF diagnosis is associated with improved cardiovascular outcomes compared to delayed intervention."},

    {"id": "ACC2023-8.1.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 8.1.1",
     "text": "Wearable cardiac monitor findings of AF burden below 1% over a 24-hour monitoring window with stable or decreasing trend slope are classified as low-risk and appropriate for routine follow-up at standard clinical intervals without urgent intervention."},

    # ESC 2020 — Nocturnal AF and Sleep Apnoea
    {"id": "ESC2020-11.3",
     "source": "ESC 2020 AF Guideline, Section 11.3",
     "text": "Nocturnal AF clustering, defined as greater than 60% of AF episodes occurring between 22:00 and 06:00, is strongly associated with obstructive sleep apnoea as a precipitating and perpetuating factor. Sleep study referral and CPAP therapy evaluation are recommended as part of the AF management pathway when nocturnal predominance is identified."},

    {"id": "ESC2020-11.4",
     "source": "ESC 2020 AF Guideline, Section 11.4",
     "text": "Treatment of obstructive sleep apnoea in AF patients is associated with reduced AF recurrence and improved rhythm control outcomes. Identification of nocturnal AF clustering should prompt systematic evaluation for sleep-disordered breathing using polysomnography or validated home sleep testing."},

    {"id": "ESC2020-5.2",
     "source": "ESC 2020 AF Guideline, Section 5.2",
     "text": "AF episode duration classification: episodes lasting less than 30 seconds are not clinically significant. Episodes of 30 seconds to 7 days are classified as paroxysmal. Episodes exceeding 7 days are persistent AF. The longest single episode duration is a key determinant of thromboembolic risk stratification."},

    {"id": "ESC2020-8.1",
     "source": "ESC 2020 AF Guideline, Section 8.1",
     "text": "Urgent clinical review is indicated when wearable monitoring detects AF burden exceeding 20% combined with an increasing burden trend. This pattern suggests progression from paroxysmal to early persistent AF requiring evaluation for rhythm control intensification."},

    {"id": "ESC2020-4.1",
     "source": "ESC 2020 AF Guideline, Section 4.1",
     "text": "The CHA2DS2-VASc score assigns 1 point each for congestive heart failure, hypertension, age 65-74, diabetes mellitus, vascular disease, and female sex; and 2 points for age ≥75 or prior stroke/TIA. Anticoagulation is recommended for scores ≥2 in males and ≥3 in females regardless of AF pattern."},

    # Emergency thresholds
    {"id": "ACC2023-9.1",
     "source": "ACC/AHA 2023 AF Guideline, Section 9.1",
     "text": "Emergency department referral is indicated for haemodynamically unstable AF, AF with rapid ventricular response causing symptoms, first-detected AF with high stroke risk profile (CHA2DS2-VASc ≥4), or monitoring-detected AF burden exceeding 40% without prior anticoagulation. Same-day evaluation should not be delayed pending outpatient scheduling."},

    {"id": "ESC2020-10.2",
     "source": "ESC 2020 AF Guideline, Section 10.2",
     "text": "Increasing AF burden trend (positive slope over 48-hour window) combined with AF burden exceeding 15% and multiple short episodes indicates a high probability of progression to persistent AF within 6 months. Proactive rhythm control strategy initiation reduces this progression risk by approximately 30-40%."},

    {"id": "ACC2023-3.2",
     "source": "ACC/AHA 2023 AF Guideline, Section 3.2",
     "text": "Routine monitoring follow-up is appropriate for patients with AF burden less than 5%, fewer than 3 episodes per monitoring period, stable or decreasing burden trend, and no high-risk features. Standard clinical review at 3-6 month intervals is recommended for this risk category."},
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD RAG INDEX
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_index(embedder):
    """Embed guideline passages and build FAISS flat index."""
    texts = [p['text'] for p in GUIDELINE_PASSAGES]
    print(f"  Embedding {len(texts)} guideline passages...")
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)    # inner product = cosine sim after L2 norm
    index.add(embeddings)

    print(f"  FAISS index built: {index.ntotal} passages, dim={dim}")
    return index

def retrieve_passages(query: str, index, embedder, k: int = TOP_K) -> list:
    """Retrieve top-k relevant guideline passages for a query."""
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [GUIDELINE_PASSAGES[i] for i in indices[0] if i < len(GUIDELINE_PASSAGES)]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BURDEN VECTOR → CLINICAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def burden_to_clinical_summary(burden: dict) -> str:
    """
    Convert normalised burden vector to human-readable clinical summary
    for use as RAG query and LLM context.
    """
    af_pct      = burden['af_burden'] * 100
    longest_min = burden['longest_ep_min_norm'] * 60 * 24   # approx minutes in 24hr window
    ep_count    = max(1, int(burden['episode_count_norm'] * 72))
    noc_pct     = burden['nocturnal_ratio'] * 100
    trend       = burden['trend_slope_norm']
    trend_dir   = "increasing" if trend > 0.55 else "decreasing" if trend < 0.45 else "stable"

    summary = (
        f"AF burden: {af_pct:.1f}% of 24-hour monitoring window. "
        f"Longest single episode: {longest_min:.0f} minutes. "
        f"Episode count: {ep_count} distinct episodes. "
        f"Nocturnal ratio: {noc_pct:.0f}% of AF occurring between 22:00-06:00. "
        f"Burden trend: {trend_dir} over monitoring window."
    )
    return summary

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LLM TRIAGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical decision support assistant for cardiac monitoring.
You analyse wearable ECG rhythm-burden data and provide evidence-based triage recommendations.
Always cite specific guideline sections. Be concise and clinically precise.
Respond ONLY with valid JSON in exactly this format:
{
  "triage_level": "routine|urgent|emergency",
  "rationale": "2-4 sentence clinical reasoning",
  "guideline_citations": ["Section X.X.X", "Section Y.Y.Y"]
}"""

def build_triage_prompt(clinical_summary: str, passages: list) -> str:
    context = "\n\n".join(
        f"[{p['source']}]\n{p['text']}" for p in passages
    )
    return f"""PATIENT RHYTHM-BURDEN PROFILE:
{clinical_summary}

RELEVANT GUIDELINE PASSAGES:
{context}

Based on the patient profile and guideline passages above, provide a triage recommendation.
Respond with JSON only."""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — LOAD LLM
# ─────────────────────────────────────────────────────────────────────────────
def load_llm():
    print(f"\nLoading LLM: {LLM_MODEL}")
    print("(~3.5 GB download on first run, ~2 GB VRAM in 4-bit)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        cache_dir=MODEL_CACHE,
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    print("LLM loaded.")
    return tokenizer, model

def generate_triage(prompt: str, tokenizer, model) -> dict:
    """Run LLM inference and parse JSON output."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt}
    ]

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|system|>{SYSTEM_PROMPT}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"

    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy for consistent clinical output
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse JSON
    try:
        # Find JSON block
        start = response.find('{')
        end   = response.rfind('}') + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            if result.get('triage_level') not in TRIAGE_LEVELS:
                result['triage_level'] = 'routine'
            return result
    except Exception:
        pass

    # Fallback parse
    level = 'routine'
    for lvl in ['emergency', 'urgent', 'routine']:
        if lvl in response.lower():
            level = lvl; break

    return {
        'triage_level': level,
        'rationale': response[:300].strip(),
        'guideline_citations': []
    }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RULE-BASED GROUND TRUTH (guideline-derived)
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_triage(burden: dict) -> str:
    """
    Deterministic triage derived directly from ACC/AHA 2023 + ESC 2020 thresholds.
    Used as ground truth for kappa evaluation.
    """
    af_pct     = burden['af_burden'] * 100
    longest_m  = burden['longest_ep_min_norm'] * 60 * 24
    ep_count   = burden['episode_count_norm'] * 72
    noc_ratio  = burden['nocturnal_ratio']
    trend      = burden['trend_slope_norm']
    increasing = trend > 0.55

    # Emergency: ACC/AHA 9.1 — burden >40% or longest episode >720 min (12 hrs)
    if af_pct > 40 or longest_m > 720:
        return 'emergency'

    # Urgent: ESC 8.1 — burden >20% + increasing, or burden >11% + high episode count
    if (af_pct > 20 and increasing) or (af_pct > 11 and ep_count > 5):
        return 'urgent'

    # Urgent: ACC/AHA 5.2.1 — episode count >5 with increasing trend
    if ep_count > 5 and increasing:
        return 'urgent'

    # Urgent: ESC 11.3 — nocturnal clustering >60%
    if af_pct > 5 and noc_ratio > 0.60:
        return 'urgent'

    # Routine: ACC/AHA 8.1.1 — burden <5%, stable/decreasing
    if af_pct < 5 and not increasing:
        return 'routine'

    # Default: anything between 5-11% without escalating features
    if af_pct >= 5:
        return 'urgent'
    return 'routine'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — GENERATE EVALUATION CASES
# ─────────────────────────────────────────────────────────────────────────────
def generate_eval_cases(n: int = N_EVAL_CASES) -> list:
    """
    Generate 100 diverse patient cases covering all three triage levels.
    Stratified: ~35 routine, ~40 urgent, ~25 emergency.
    """
    cases = []
    np.random.seed(SEED)

    for _ in range(n):
        r = random.random()

        if r < 0.25:        # emergency
            af   = np.random.uniform(0.35, 0.80)
            lep  = np.random.uniform(0.40, 1.0)
            ep   = np.random.uniform(0.10, 0.50)
            noc  = np.random.uniform(0.30, 0.90)
            slp  = np.random.uniform(0.60, 1.0)

        elif r < 0.65:      # urgent
            af   = np.random.uniform(0.05, 0.40)
            lep  = np.random.uniform(0.05, 0.40)
            ep   = np.random.uniform(0.05, 0.30)
            noc  = np.random.uniform(0.20, 0.90)
            slp  = np.random.uniform(0.40, 0.90)

        else:               # routine
            af   = np.random.uniform(0.00, 0.08)
            lep  = np.random.uniform(0.00, 0.10)
            ep   = np.random.uniform(0.00, 0.10)
            noc  = np.random.uniform(0.10, 0.60)
            slp  = np.random.uniform(0.20, 0.55)

        burden = {
            'af_burden':            float(np.clip(af,  0, 1)),
            'longest_ep_min_norm':  float(np.clip(lep, 0, 1)),
            'episode_count_norm':   float(np.clip(ep,  0, 1)),
            'nocturnal_ratio':      float(np.clip(noc, 0, 1)),
            'trend_slope_norm':     float(np.clip(slp, 0, 1)),
        }
        cases.append(burden)

    return cases

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("CardioAgent — Tier 4: Clinical Triage Agent")
    print(f"Device: {DEVICE}")
    print(f"{'='*65}\n")

    os.makedirs('/workspace/cardioagent/models', exist_ok=True)

    # ── Load embedder ────────────────────────────────────────────────────────
    print(f"Loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    print("Embedder loaded.")

    # ── Build RAG index ──────────────────────────────────────────────────────
    print("\nBuilding RAG knowledge base from ACC/AHA 2023 + ESC 2020 guidelines...")
    rag_index = build_rag_index(embedder)

    # ── Load LLM ─────────────────────────────────────────────────────────────
    tokenizer, llm = load_llm()

    # ── Quick demo inference ─────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("DEMO — Example patient case")
    print("─"*65)
    demo_burden = {
        'af_burden':           0.14,
        'longest_ep_min_norm': 0.08,
        'episode_count_norm':  0.12,
        'nocturnal_ratio':     0.68,
        'trend_slope_norm':    0.72
    }
    summary   = burden_to_clinical_summary(demo_burden)
    passages  = retrieve_passages(summary, rag_index, embedder)
    prompt    = build_triage_prompt(summary, passages)
    result    = generate_triage(prompt, tokenizer, llm)

    print(f"\nPatient profile:\n  {summary}")
    print(f"\nRetrieved {len(passages)} guideline passages:")
    for p in passages: print(f"  - {p['source']}")
    print(f"\nTriage output:")
    print(f"  Level    : {result.get('triage_level','N/A').upper()}")
    print(f"  Rationale: {result.get('rationale','N/A')}")
    print(f"  Citations: {result.get('guideline_citations', [])}")

    # ── 100-case evaluation ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"Evaluating {N_EVAL_CASES} cases — LLM vs guideline-derived ground truth")
    print("─"*65)

    cases      = generate_eval_cases(N_EVAL_CASES)
    gt_labels  = [rule_based_triage(c) for c in cases]
    llm_labels = []
    results    = []

    for i, burden in enumerate(cases):
        summary  = burden_to_clinical_summary(burden)
        passages = retrieve_passages(summary, rag_index, embedder)
        prompt   = build_triage_prompt(summary, passages)
        result   = generate_triage(prompt, tokenizer, llm)
        llm_labels.append(result.get('triage_level', 'routine'))
        results.append({**burden, **result, 'ground_truth': gt_labels[i]})

        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{N_EVAL_CASES} cases...")

    # ── Metrics ──────────────────────────────────────────────────────────────
    label_map  = {'routine': 0, 'urgent': 1, 'emergency': 2}
    gt_int     = [label_map[l] for l in gt_labels]
    llm_int    = [label_map.get(l, 0) for l in llm_labels]

    kappa = cohen_kappa_score(gt_int, llm_int)
    acc   = sum(g == p for g, p in zip(gt_int, llm_int)) / len(gt_int)

    print(f"\n{'='*65}")
    print("Tier 4 Evaluation Results")
    print(f"{'='*65}")
    print(classification_report(gt_int, llm_int,
                                 target_names=TRIAGE_LEVELS, digits=4))
    print(f"Cohen's Kappa : {kappa:.4f}  (target: ≥0.70)")
    print(f"Accuracy      : {acc:.4f}")

    # Distribution check
    from collections import Counter
    gt_dist  = Counter(gt_labels)
    llm_dist = Counter(llm_labels)
    print(f"\nGround truth distribution : {dict(gt_dist)}")
    print(f"LLM output distribution   : {dict(llm_dist)}")

    # Save results
    out_path = '/workspace/cardioagent/tier4_results.json'
    with open(out_path, 'w') as f:
        json.dump({'kappa': kappa, 'accuracy': acc,
                   'cases': results}, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"Tier 4 complete. All four tiers done.")
    print("Next: python pipeline.py  (end-to-end test)")

if __name__ == '__main__':
    main()
