# LinkedIn Post — Ryoushi | Quantum Buddies (GIC 2026 Journey)

**Character count: ~1,780 (within the 1,300–2,000 engagement sweet spot; LinkedIn hard limit is 3,000)**

---

🧬 What if a transformer could *write* quantum circuits?

5 months. 3 phases. 35 molecules. 1 model. Real quantum hardware.

We're Ryoushi | Quantum Buddies, and this was our journey through the GIC 2026 — the Generative Quantum Eigensolver Innovation Challenge, hosted by Mitsubishi Chemical Group & AIST, with QPU access via qBraid and GPU acceleration through NVIDIA CUDA-Q.

**Phase 1 — Team Formation (Mar–Apr)**
We proposed H-cGQE: a Hierarchical Conditional Generative Quantum Eigensolver. Treat circuit synthesis like language generation — operators are vocabulary, circuits are sentences, and a transformer learns the grammar of good ansätze. No variational parameters in the circuit itself. No barren plateaus.

**Phase 2 — Conceptual Design (Apr–May)**
Built the pipeline on AIRE HPC (3× L40S). Hit our first real wall: diagonal sequence collapse — the model learned to generate Z-only operators that commute, producing zero gradients and freezing energy at the Hartree-Fock baseline. We proposed three fixes: symmetry-preserving masking, curriculum learning for entanglement, and RL with dense energy feedback.

Advanced to finalists.

**Phase 3 — Applied Execution (Jun–Jul)**
This is where it got real:

✅ UCCSD operator pool — diagonal collapse now structurally impossible
✅ DAPO reinforcement learning — asymmetric clipping + dense reward (energy, entanglement, depth, non-commuting fraction)
✅ One 7.85M-parameter model across 35 molecules, 4–28 qubits
✅ QWC grouping — 3–5× fewer measurement circuits
✅ Multi-vendor QPU access via qBraid — Rigetti, IonQ, IQM
✅ Manifest-based async pipeline with a SQLite ledger for budget + idempotency
✅ Model open-sourced on HuggingFace with a one-click qBraid launch button

Small model. Big chemistry. Real hardware.

Whatever the final outcome, going from a concept doc to circuits running on superconducting and trapped-ion qubits in five months is a milestone we're proud of.

🔗 Code: github.com/Quantum-Buddies/Conditional-GQE_materials
🔗 Model: huggingface.co/Ryukijano/h-cgqe-gic2026

#QuantumComputing #GenerativeAI #QuantumChemistry #GIC2026 #CUDAQ #qBraid #Ryoushi

---

## Visual asset — "GIC 2026 journey" loop

Anthropic-style design language applied to a quantum-chemistry subject: warm putty canvas, one clay accent, flat geometry, no gradients/shadows/glow, editorial serif for labels, hairline strokes only.

**Palette**
- Canvas: `#f0eee6` (putty)
- Card / line art: `#ffffff` fill, `#191919` (ink) 1px stroke
- Muted text: `#737373`
- Accent (used once per scene, sparingly): `#cc785c` (clay)
- Font: serif display (`Tiempos`/`Georgia` fallback) for the scene label, sans (`Inter`/system-ui) for captions

**6-scene loop (~1.8s per scene, infinite loop)** — expanded to reflect the actual architecture depth in the README, not just the surface pitch:
1. **Graph** — benzene-ring line art; a clay message-pulse travels node-to-node around the ring. Caption: "A molecule becomes a graph." *(Chemistry GNN — edge-aware message passing)*
2. **Encoder → Decoder** — a boxed "Hamiltonian encoder" with bar-chart Pauli-term glyphs, dashed cross-attention line into a row of decoder token boxes filling in one at a time. Caption: "Encoder + decoder write the sequence." *(Conditional encoder-decoder transformer)*
3. **Collapse, avoided** — top row: four greyed-out `Z` tokens struck through with a clay diagonal line (the failure mode). Bottom row: `X Y Y X` tokens filling in with a clay checkmark at the end. Caption: "Collapse, avoided." *(UCCSD operator pool — 0% Z-only by construction)*
4. **MAP-Elites** — a 6×6 grid (paper: 10×10) of hairline cells progressively filling with solid clay squares in a scattered, non-linear order; live coverage % ticks up. Caption: "Quality-diversity, not mode collapse." *(QD-GRPO / MAP-Elites archive)*
5. **Multi-vendor QPU** — a "Circuit" box with a dashed line + traveling clay dot dispatching to one of three chip icons (Rigetti / IonQ / IQM), each rendered as a hairline card with a 3×2 dot grid. Caption: "One workspace. Any hardware." *(qBraid dispatch)*
6. **Convergence** — thin ink line graph draws left to right, descending and flattening; a clay dot lands where it flattens. Caption: "Energy finds its floor. ≤ 1.6 mHa — chemical accuracy."

Loop back to scene 1.

**Deliverables**:
- `docs/gic2026_journey_loop.svg` — 4-scene source animation (SMIL `<animate>` tags), plays natively in any browser (kept as the lightweight/simple variant)
- `docs/gic2026_journey_loop.gif` — **ready to upload directly to LinkedIn** — the full 6-scene version (800×450, 42 frames / ~10.9s loop, ~420 KB)
- `docs/build_gif.py` — regenerates the GIF from static per-frame SVGs via `rsvg-convert` + Pillow; each scene is its own function (`scene1`…`scene6`) if you want to reorder, retime, or add a 7th panel (e.g. QSCI/FMO2 scaling, 40q)
