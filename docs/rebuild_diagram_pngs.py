#!/usr/bin/env python3
"""
Rebuild all 11 Mermaid diagrams with a modern, polished design system.
Uses per-diagram theme frontmatter, semantic classDefs, and styled edges.
Renders via mmdc (Puppeteer) for correct foreignObject/text rendering.
"""

import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "mermaid_svgs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Modern Design System ──────────────────────────────────────────────
# Inspired by Linear/Vercel design language — clean, professional, readable

THEME_FRONTMATTER = """%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#EEF2FF',
  'primaryTextColor': '#3730A3',
  'primaryBorderColor': '#818CF8',
  'secondaryColor': '#F0F9FF',
  'secondaryTextColor': '#0369A1',
  'secondaryBorderColor': '#38BDF8',
  'tertiaryColor': '#ECFDF5',
  'tertiaryTextColor': '#065F46',
  'tertiaryBorderColor': '#34D399',
  'lineColor': '#64748B',
  'textColor': '#1E293B',
  'background': '#FFFFFF',
  'mainBkg': '#EEF2FF',
  'nodeBorder': '#818CF8',
  'clusterBkg': '#F8FAFC',
  'clusterBorder': '#CBD5E1',
  'titleColor': '#1E293B',
  'edgeLabelBackground': '#FFFFFF',
  'fontFamily': 'Inter, system-ui, -apple-system, sans-serif',
  'fontSize': '15px'
}}}%%
"""

CLASS_DEFS = """classDef process fill:#EEF2FF,stroke:#818CF8,stroke-width:2px,color:#3730A3,rx:8,ry:8
classDef data fill:#F0F9FF,stroke:#38BDF8,stroke-width:2px,color:#0369A1,rx:8,ry:8
classDef output fill:#ECFDF5,stroke:#34D399,stroke-width:2px,color:#065F46,rx:8,ry:8
classDef accent fill:#FFFBEB,stroke:#FBBF24,stroke-width:2px,color:#92400E,rx:8,ry:8
classDef highlight fill:#F5F3FF,stroke:#A78BFA,stroke-width:2px,color:#6D28D9,rx:8,ry:8
classDef danger fill:#FEF2F2,stroke:#F87171,stroke-width:2px,color:#991B1B,rx:8,ry:8
classDef subtle fill:#F8FAFC,stroke:#CBD5E1,stroke-width:1.5px,color:#475569,rx:6,ry:6
"""

# ── Diagram Sources ────────────────────────────────────────────────────
# Structure: THEME_FRONTMATTER + "flowchart TD\n" + CLASS_DEFS + diagram_body
# classDef must come AFTER the diagram type declaration

DIAGRAMS = {}

# Diagram 01 — End-to-End Pipeline
DIAGRAMS[1] = THEME_FRONTMATTER + "flowchart TD\n" + CLASS_DEFS + """
    subgraph S1["① Chemistry Input"]
        direction TB
        Mol["Molecular Geometry<br/>PySCF / OpenFermion"]
        Ham["Electronic Hamiltonian<br/>H = Σ hₗ Pₗ  (Jordan-Wigner)"]
        Graph["Atom Graph<br/>Nodes: Z, hybridization · Edges: bond, Rᵢⱼ"]
        Mol --> Ham --> Graph
    end

    subgraph S2["② Encoder Stack"]
        direction TB
        GNN["Chemistry GNN<br/>3-layer Edge-Aware MPNN → Soft Prefix"]
        HEnc["Hamiltonian Encoder<br/>4-layer Transformer → Cross-Attn K,V"]
    end

    subgraph S3["③ Decoder & Output"]
        direction TB
        Pool["UCCSD Operator Pool<br/>Fermionic excitations · 0% Z-only"]
        Dec["Decoder<br/>6-layer Transformer → Autoregressive Tokens"]
        Seq["Operator Sequence<br/>A₁, A₂, ..., Aₖ"]
        Pool --> Dec --> Seq
    end

    S1 ==> S2
    GNN -.->|"prefix conditioning"| Dec
    HEnc -.->|"cross-attention K,V"| Dec
    S2 ==> S3

    class Mol,Ham,Graph data
    class GNN,HEnc process
    class Pool,Dec,Seq output
"""

# Diagram 02 — Transformer Architecture
DIAGRAMS[2] = THEME_FRONTMATTER + "flowchart TD\n" + CLASS_DEFS + """
    subgraph Input_Layer["Input Processing"]
        direction TB
        Atoms["Atom Features<br/>(Z, charge, hybrid,<br/>valence, aromaticity)"]
        Bonds["Bond Features<br/>(bond order, R_ij,<br/>conjugation)"]
        Globals["Global Features<br/>(N_q, N_e, spin 2S+1,<br/>active space size)"]
    end

    subgraph GNN_Layer["Chemistry GNN (3-layer MPNN)"]
        direction TB
        MP1["Layer 1: Edge-weighted message passing<br/>h_v ← h_v + Σ σ(W·[h_w, e_vw, g])"]
        MP2["Layer 2: Residual + LayerNorm + Dropout(0.1)"]
        MP3["Layer 3: Residual + LayerNorm + Dropout(0.1)"]
        MP1 --> MP2 --> MP3
    end

    subgraph Encoder_Layer["Hamiltonian Encoder (4-layer Transformer)"]
        direction TB
        Emb["Hamiltonian Embedding<br/>Pauli terms → token embeddings<br/>+ positional encoding"]
        SA["Multi-Head Self-Attention (8 heads)<br/>Q,K,V from Hamiltonian tokens"]
        FFN["Feed-Forward Network<br/>Linear → GELU → Linear<br/>+ residual + LayerNorm"]
        Emb --> SA --> FFN
    end

    subgraph Decoder_Layer["Decoder (6-layer Transformer)"]
        direction TB
        TokEmb["Token Embedding<br/>Operator tokens → learned embeddings"]
        CA["Cross-Attention (8 heads)<br/>Q from decoder, K,V from encoder"]
        DecSA["Causal Self-Attention<br/>Autoregressive mask"]
        DecFFN["Feed-Forward + LM Head<br/>→ softmax over operator vocab"]
        TokEmb --> DecSA --> CA --> DecFFN
    end

    Input_Layer ==> GNN_Layer
    GNN_Layer -.->|"soft prefix tokens"| Decoder_Layer
    Encoder_Layer -.->|"cross-attention K,V"| Decoder_Layer
    Input_Layer --> Encoder_Layer

    class Atoms,Bonds,Globals data
    class MP1,MP2,MP3,Emb,SA,FFN process
    class TokEmb,CA,DecSA,DecFFN output
"""

# Diagram 03 — RL Training Loop
DIAGRAMS[3] = THEME_FRONTMATTER + "flowchart TD\n" + CLASS_DEFS + """
    Start(("Start Epoch"))
    Sample["Sample N=16 circuits<br/>per molecule via<br/>autoregressive decoder"]
    Eval["Evaluate Energies<br/>via Cache or L-BFGS-B"]
    Reward["Compute Multi-Component Reward<br/>R = w₁·(-E/|E_ref|) + w₂·entanglement<br/>+ w₃·(-depth/max_len) + w₄·non_commute"]
    Advantage["GRPO Advantage Normalization<br/>A_i = (R_i - mean(R)) / (std(R) + ε)"]
    Loss["DAPO Loss (Asymmetric Clipping)<br/>L = -min(r·A, clip(r, 1-ε_low, 1+ε_high)·A)<br/>ε_low=0.2, ε_high=0.28"]
    Update["Policy Gradient Update<br/>θ ← θ - η·∇L"]
    MAP["MAP-Elites Archive Update<br/>Insert into 10×10 grid<br/>(entanglement × depth)"]
    Replay["Replay Buffer Mixing<br/>60% elites + 30% novelty<br/>+ 10% random exploration"]
    Converge{"Converged?"}
    Done(("Done"))

    Start --> Sample --> Eval --> Reward --> Advantage --> Loss --> Update --> MAP --> Replay --> Converge
    Converge -->|"no"| Sample
    Converge -->|"yes"| Done

    class Start,Done accent
    class Sample,Eval process
    class Reward,Advantage,Loss,Update highlight
    class MAP,Replay output
    class Converge danger
"""

# Diagram 04 — VQE vs C-GQE
DIAGRAMS[4] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph VQE["Traditional VQE"]
        direction TB
        VAns["Fixed Ansatz<br/>(UCCSD / HEA)<br/>Human-designed"]
        VParam["k continuous params<br/>θ ∈ R^k on quantum device"]
        VOpt["Classical Optimizer<br/>(COBYLA / SPSA)<br/>θ ← θ - η·∇E"]
        VLimited["Limitations<br/>• Barren plateaus<br/>• Manual design per molecule<br/>• No transfer learning"]
        VAns --> VParam --> VOpt --> VLimited
    end

    subgraph CGQE["H-cGQE (Our Method)"]
        direction TB
        CGraph["Molecular Graph Input<br/>GNN encodes structure<br/>→ soft prefix tokens"]
        CGen["Autoregressive Generation<br/>Transformer decoder samples<br/>operator sequence A₁...Aₖ"]
        CClassical["L-BFGS-B Angle Tuning<br/>θ* = argmin ⟨ψ₀|U†HU|ψ₀⟩<br/>200 iters, ftol=10⁻¹⁰"]
        CAdvantage["Advantages<br/>• Amortized across molecules<br/>• Transferable GNN embeddings<br/>• QD-GRPO diversity"]
        CGraph --> CGen --> CClassical --> CAdvantage
    end

    VQE -.->|"replaced by"| CGQE

    class VAns,VParam,VOpt,VLimited danger
    class CGraph,CGen,CClassical,CAdvantage output
"""

# Diagram 05 — Chemistry GNN Encoder
DIAGRAMS[5] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph Graph["Molecular Graph Input"]
        direction TB
        N["Nodes (Atoms)<br/>• Atomic number Z<br/>• Hybridization (sp/sp²/sp³)<br/>• Formal charge<br/>• Valence electrons<br/>• Aromaticity flag"]
        E["Edges (Bonds)<br/>• Bond order (1/2/3)<br/>• 3D distance R_ij (Å)<br/>• Conjugation<br/>• Ring membership"]
        G["Globals<br/>• N_q (qubit count)<br/>• N_e (electron count)<br/>• Spin 2S+1<br/>• Active space (n_occ, n_virt)"]
    end

    subgraph MPNN["Edge-Aware Message Passing (3 layers)"]
        direction TB
        H0["h_v⁽⁰⁾ = Linear(node_feat)<br/>e_vw⁽⁰⁾ = Linear(edge_feat)<br/>g⁽⁰⁾ = Linear(global_feat)"]
        H1["Layer ℓ+1:<br/>h_v⁽ℓ⁺¹⁾ = h_v⁽ℓ⁾ + Σ σ(W·[h_w⁽ℓ⁾, e_vw⁽ℓ⁾, g⁽ℓ⁾])<br/>+ LayerNorm + Dropout(0.1)"]
        H2["Layer ℓ+2: same structure"]
        H3["Layer ℓ+3: same structure"]
        H0 --> H1 --> H2 --> H3
    end

    subgraph Readout["Graph Readout → Conditioning"]
        direction TB
        Pool["Pooled = concat[mean(h_v), max(h_v),<br/>sum(h_v), g_final]<br/>→ Linear → GELU → Dropout<br/>→ Linear → latent_dim=128"]
        Norm["LayerNorm(latent)"]
        Proj["Prefix Projection:<br/>Linear(128→128) → GELU<br/>→ Linear(128→128) → LayerNorm<br/>→ Soft Prompt Tokens"]
        Pool --> Norm --> Proj
    end

    Graph ==> MPNN ==> Readout
    Proj -->|"prefix conditioning"| Decoder["Operator Pool Decoder"]

    class N,E,G data
    class H0,H1,H2,H3 process
    class Pool,Norm,Proj output
    class Decoder accent
"""

# Diagram 06 — Jordan-Wigner Mapping
DIAGRAMS[6] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph JW["Jordan-Wigner Mapping"]
        direction TB
        Fermion["Fermionic Excitation<br/>a⁺_p a_q (single)<br/>a⁺_p a⁺_q a_r a_s (double)"]
        JWMap["JW Transform:<br/>a⁺_p → (X_p − iY_p)/2 ⊗ Z_{p−1}...Z₀"]
        Pauli["Pauli Words:<br/>YZXI, XZYI, IYZX, ...<br/>(always contain X/Y)"]
        Fermion --> JWMap --> Pauli
    end

    subgraph Pool["Operator Pool Construction"]
        direction TB
        Singles["Single Excitations:<br/>τ_pq = a⁺_p a_q − h.c.<br/>→ 2 Pauli words each"]
        Doubles["Double Excitations:<br/>τ_pqrs = a⁺_p a⁺_q a_r a_s − h.c.<br/>→ 8 Pauli words each"]
        Scale["Scale Factors:<br/>θ_scale = 1/√N_excitations<br/>(normalization)"]
        Singles --> Pool2["Combined Pool<br/>0% Z-only guaranteed"]
        Doubles --> Pool2
        Pool2 --> Scale
    end

    JW ==> Pool

    class Fermion,JWMap,Pauli process
    class Singles,Doubles,Pool2 data
    class Scale output
"""

# Diagram 07 — MAP-Elites Archive
DIAGRAMS[7] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph Archive["MAP-Elites Archive — Per-Molecule Elite Library"]
        direction TB
        Grid["10×10 Grid<br/>Axis 1: Entanglement Density<br/>  (frac of X/Y multi-qubit ops)<br/>Axis 2: Circuit Depth<br/>  (gate count / max_seq_len)"]
        Cell["Each Cell stores:<br/>• Best operator sequence [A₁,...,Aₖ]<br/>• Optimized angles θ*<br/>• Energy E* = ⟨ψ₀|U†HU|ψ₀⟩<br/>• Generation (epoch discovered)<br/>• Visit count"]
        Coverage["Coverage Tracking:<br/>• λ = 1.0 when coverage < 25%<br/>• λ = 0.5 when coverage 25–50%<br/>• λ = 0.1 when coverage > 50%<br/>• λ = 0.0 when coverage > 80%"]
        Grid --> Cell --> Coverage
    end

    subgraph Selection["Elite Selection for Replay"]
        direction TB
        Sample2["Sample from archive:<br/>• 60% from filled cells (energy-weighted)<br/>• 30% from empty cells (novelty-driven)<br/>• 10% random exploration"]
        Inject["Inject into replay buffer<br/>alongside online samples"]
        Sample2 --> Inject
    end

    Archive --> Selection
    Selection -->|"pretrain mixing"| Buffer["Replay Buffer"]

    class Grid,Cell,Coverage process
    class Sample2,Inject output
    class Buffer accent
"""

# Diagram 08 — L-BFGS-B Fine-Tuning (sequence diagram, no classDefs)
DIAGRAMS[8] = """%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#EEF2FF',
  'primaryTextColor': '#3730A3',
  'primaryBorderColor': '#818CF8',
  'secondaryColor': '#F0F9FF',
  'secondaryTextColor': '#0369A1',
  'secondaryBorderColor': '#38BDF8',
  'tertiaryColor': '#ECFDF5',
  'tertiaryTextColor': '#065F46',
  'tertiaryBorderColor': '#34D399',
  'lineColor': '#64748B',
  'textColor': '#1E293B',
  'background': '#FFFFFF',
  'actorBkg': '#EEF2FF',
  'actorBorder': '#818CF8',
  'actorTextColor': '#3730A3',
  'actorLineColor': '#94A3B8',
  'signalColor': '#475569',
  'signalTextColor': '#1E293B',
  'labelBoxBkgColor': '#F8FAFC',
  'labelBoxBorderColor': '#CBD5E1',
  'labelTextColor': '#475569',
  'loopTextColor': '#475569',
  'activationBorderColor': '#94A3B8',
  'activationBkgColor': '#F1F5F9',
  'sequenceNumberColor': '#FFFFFF',
  'fontFamily': 'Inter, system-ui, -apple-system, sans-serif',
  'fontSize': '14px'
}}}%%
sequenceDiagram
    participant Policy as 🧠 Transformer Policy
    participant Eval as ⚡ Energy Evaluator
    participant CUDAQ as 💻 CUDA-Q Simulator
    participant LBFGS as 🎯 L-BFGS-B Optimizer

    Policy->>Eval: Generate operator seq [A₁, A₂, ..., Aₖ]
    Eval->>Eval: Check DedupCache (MD5 hash of ops)
    alt Cache Hit
        Eval-->>Policy: Return cached E* (μs)
    else Cache Miss
        Eval->>LBFGS: Initialize θ₀ = (0.01, ..., 0.01)
        loop Iterations 1..max_iters
            LBFGS->>CUDAQ: observe(kernel, H, n_qubits, n_e, pauli_words, θ)
            CUDAQ-->>LBFGS: E = ⟨ψ|H|ψ⟩
            LBFGS->>LBFGS: Approximate inverse Hessian
            LBFGS->>LBFGS: Line search + Wolfe conditions
            LBFGS->>LBFGS: Update θ ← θ + Δθ
        end
        LBFGS-->>Eval: Return E* (optimized)
        Eval->>Eval: Store in DedupCache
        Eval-->>Policy: Return E* (ms–s depending on n_qubits)
    end
"""

# Diagram 09 — B200 Energy Cache
DIAGRAMS[9] = THEME_FRONTMATTER + "flowchart TD\n" + CLASS_DEFS + """
    subgraph Cache_Build["Stage 1: Cache Precompute (B200 GPU)"]
        direction TB
        Mols1["35 GIC Molecules<br/>(4–28 qubits)"]
        Gen["For each molecule:<br/>• Build UCCSD operator pool<br/>• Sample 500–2000 random sequences<br/>• L-BFGS-B optimize angles<br/>• CUDA-Q observe → E*"]
        Store[("SQLite: (MD5_hash, energy,<br/>molecule, n_qubits, operators)<br/>24,000+ entries")]
        Mols1 --> Gen --> Store
    end

    subgraph Recovery["Stage 2: Cache → Pretrain JSON"]
        direction TB
        Load["Load SQLite cache"]
        Replay2["Replay deterministic<br/>circuit generation<br/>(same seed = same ops)"]
        Match["Match hash to recover<br/>(operators, energy) pairs"]
        Export["Export JSON:<br/>17,408 samples across 34 molecules"]
        Load --> Replay2 --> Match --> Export
    end

    subgraph Offline["Stage 3: Offline RL Training (Any GPU)"]
        direction TB
        Prefill["Pre-fill replay buffer<br/>80% pretrain fraction<br/>→ 1,600 cached samples"]
        Train["DAPO policy updates<br/>using cached energies<br/>NO CUDA-Q needed"]
        Decay["Pretrain fraction decays<br/>80% → 0% over 100 epochs<br/>→ smooth online transition"]
        Prefill --> Train --> Decay
    end

    Cache_Build ==> Recovery ==> Offline

    class Mols1,Gen data
    class Store accent
    class Load,Replay2,Match,Export process
    class Prefill,Train,Decay output
"""

# Diagram 10 — QSCI & FMO2 Scaling
DIAGRAMS[10] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph Brute["Brute-Force SV (Infeasible)"]
        SV["2⁴⁰ = 1.1×10¹²<br/>amplitudes<br/>~1 TB GPU memory<br/>→ OOM on any GPU"]
    end

    subgraph QSCI_P["QSCI (28–40q)"]
        direction TB
        Sample3["Sample quantum circuit<br/>→ bitstring distribution"]
        Select["Select top-K determinants<br/>by probability (K ~ 1000)"]
        Subspace["Build subspace Hamiltonian<br/>H_sub ∈ ℂ^(K×K)"]
        Diag["Classical diagonalization<br/>E = eigmin(H_sub)"]
        Sample3 --> Select --> Subspace --> Diag
    end

    subgraph FMO2_P["FMO2 (Macromolecules)"]
        direction TB
        Frag["Fragment molecule into<br/>monomers + dimers<br/>(8–12 qubits each)"]
        EvalF["Evaluate each fragment<br/>on GPU or QPU"]
        Reassemble["Reassemble parent energy:<br/>E_FMO2 = Σ E_i − Σ E_ij<br/>(pairwise correction)"]
        Frag --> EvalF --> Reassemble
    end

    Brute -.->|"replaced by"| QSCI_P
    Brute -.->|"replaced by"| FMO2_P

    class SV danger
    class Sample3,Select,Subspace,Diag process
    class Frag,EvalF,Reassemble output
"""

# Diagram 11 — Phase 3 Pipeline
DIAGRAMS[11] = THEME_FRONTMATTER + "flowchart LR\n" + CLASS_DEFS + """
    subgraph S1["Stage 1: Precompute (B200)"]
        direction TB
        H1["Generate Hamiltonians<br/>(PySCF → JW mapping)"]
        H2["H-cGQE Inference<br/>(sample 500–2000 circuits<br/>per molecule)"]
        H3["L-BFGS-B + CUDA-Q<br/>observe → energy"]
        H4[("SQLite Cache<br/>24k+ entries")]
        H1 --> H2 --> H3 --> H4
    end

    subgraph S2["Stage 2: Offline RL (L40S)"]
        direction TB
        R1["Load cache →<br/>pretrain JSON"]
        R2["Pre-fill replay buffer<br/>(80% pretrain fraction)"]
        R3["QD-GRPO training<br/>DAPO + MAP-Elites<br/>+ novelty bonus"]
        R4["RL-tuned checkpoint<br/>h_cgqe_rl_dapo_phase3.pt"]
        R1 --> R2 --> R3 --> R4
    end

    subgraph S3["Stage 3: QPU Validation"]
        direction TB
        Q1["Generate QWC manifests<br/>(QASM 2.0 export)"]
        Q2["qBraid submission<br/>→ Rigetti Cepheus<br/>(108q superconducting)"]
        Q3["Retrieve + parse<br/>shot counts → expectations"]
        Q4["Compare: QPU vs SV<br/>vs exact FCI"]
        Q1 --> Q2 --> Q3 --> Q4
    end

    S1 ==>|"rl_energy_cache.sqlite"| S2
    S2 ==>|"RL checkpoint<br/>+ MAP-Elites archive"| S3

    class H1,H2,H3 data
    class H4 accent
    class R1,R2,R3,R4 process
    class Q1,Q2,Q3,Q4 output
"""

# ── Render ────────────────────────────────────────────────────────────

puppeteer_config = {
    "args": ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
}
pup_cfg_path = Path("/tmp/puppeteer-config.json")
pup_cfg_path.write_text(json.dumps(puppeteer_config))

for idx in range(1, 12):
    content = DIAGRAMS[idx]
    mmd_file = OUT_DIR / f"diagram_{idx:02d}.mmd"
    svg_file = OUT_DIR / f"diagram_{idx:02d}.svg"
    png_file = OUT_DIR / f"diagram_{idx:02d}.png"

    mmd_file.write_text(content, encoding="utf-8")

    # Render PNG
    cmd_png = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-p", str(pup_cfg_path),
        "-i", str(mmd_file),
        "-o", str(png_file),
        "-w", "2400",
        "-b", "white",
        "-s", "2"
    ]
    print(f"Rendering Diagram {idx:02d} to PNG...")
    res_png = subprocess.run(cmd_png, capture_output=True, text=True)
    if res_png.returncode != 0:
        print(f"  PNG Error: {res_png.stderr[:500]}")
    else:
        sz_kb = png_file.stat().st_size / 1024
        print(f"  ✓ {png_file.name} ({sz_kb:.0f} KB)")

    # Render SVG
    cmd_svg = [
        "npx", "-y", "@mermaid-js/mermaid-cli",
        "-p", str(pup_cfg_path),
        "-i", str(mmd_file),
        "-o", str(svg_file),
        "-w", "2400",
        "-b", "white"
    ]
    subprocess.run(cmd_svg, capture_output=True, text=True)

print("\nAll 11 diagrams rendered with modern design system!")
