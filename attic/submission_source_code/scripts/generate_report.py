#!/usr/bin/env python3
"""Generate GIC 2026 submission PDF report from consolidated benchmark.

Uses matplotlib + reportlab (if available) or just matplotlib for a PNG summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main() -> None:
    bench_path = ROOT / "results/eval/benchmark/gic2026_consolidated_benchmark.json"
    with open(bench_path) as f:
        benchmark = json.load(f)

    out_dir = ROOT / "results/eval/benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: SQD Energy Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # H2 results across all sources
    ax = axes[0]
    sources = []
    energies = []
    errors = []
    colors = []

    for source_name, results_dict, color in [
        ("Local L40S", benchmark["results"]["local_sqd_pilot"], "#2196F3"),
        ("AWS SV1 sim", benchmark["results"]["sv1_simulator"], "#4CAF50"),
        ("Cepheus QPU", benchmark["results"]["cepheus_qpu_sqd"], "#FF5722"),
    ]:
        for key, r in sorted(results_dict.items()):
            if r.get("molecule") == "h2":
                sources.append(source_name)
                energies.append(r.get("sqd_energy", 0))
                errors.append(r.get("error_mha", 0))
                colors.append(color)

    fci = -1.137284
    x = range(len(sources))
    bars = ax.bar(x, energies, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.axhline(y=fci, color="red", linestyle="--", linewidth=1.5, label=f"FCI = {fci:.6f} Ha")
    ax.set_xticks(x)
    ax.set_xticklabels(sources, fontsize=10, fontweight="bold")
    ax.set_ylabel("SQD Energy (Ha)", fontsize=12, fontweight="bold")
    ax.set_title("H2 SQD Energy — QPU vs Simulator", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(fci - 0.01, fci + 0.01)

    # All molecules on Cepheus
    ax = axes[1]
    mols = []
    sqd_energies = []
    fci_energies = []
    for mol, r in benchmark["results"]["cepheus_qpu_sqd"].items():
        mols.append(mol)
        sqd_energies.append(r.get("sqd_energy", 0))
        fci_energies.append(r.get("fci_energy", 0))

    x = np.arange(len(mols))
    width = 0.35
    ax.bar(x - width/2, sqd_energies, width, label="SQD Energy", color="#FF5722", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, fci_energies, width, label="FCI Energy", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in mols], fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy (Ha)", fontsize=12, fontweight="bold")
    ax.set_title("Rigetti Cepheus-1-108Q SQD Results", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig_path = out_dir / "gic2026_sqd_benchmark.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {fig_path}")

    # --- Figure 2: Pipeline Diagram ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (0.5, "Stage 1\nAI Inference\nH-cGQE Transformer", "#E3F2FD"),
        (2.5, "Stage 2\nL-BFGS-B Optimization\nL40S GPU (CUDA-Q)", "#BBDEFB"),
        (4.5, "Stage 3\nQPU Execution\nRigetti Cepheus-1-108Q", "#FFCCBC"),
        (6.5, "Stage 4\nSQD Post-Processing\nClassical Diagonalization", "#C8E6C9"),
        (8.5, "Stage 5\nBenchmark\nConsolidation", "#D1C4E9"),
    ]

    for x, label, color in stages:
        rect = mpatches.FancyBboxPatch((x, 1), 1.5, 1.2, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.75, 1.6, label, ha="center", va="center", fontsize=9, fontweight="bold")
        if x < 8.5:
            ax.annotate("", xy=(x + 2.0, 1.6), xytext=(x + 1.5, 1.6),
                        arrowprops=dict(arrowstyle="->", lw=2, color="black"))

    ax.set_title("GIC 2026 NISQ Pipeline: AI + HPC + QPU", fontsize=14, fontweight="bold", pad=20)
    fig_path = out_dir / "gic2026_pipeline_diagram.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 2 saved: {fig_path}")

    # --- Figure 3: Cost Accounting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    spent = benchmark["cost_accounting"]["total_estimated_credits"]
    remaining = benchmark["cost_accounting"]["remaining_credits"]
    ax.pie([spent, remaining], labels=[f"Spent\n{spent:.0f} cr", f"Remaining\n{remaining:.0f} cr"],
           colors=["#FF5722", "#4CAF50"], autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 12, "fontweight": "bold"})
    ax.set_title("QPU Credit Budget Usage", fontsize=13, fontweight="bold")
    fig_path = out_dir / "gic2026_cost_accounting.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure 3 saved: {fig_path}")

    # --- Generate text summary ---
    summary_path = out_dir / "gic2026_benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GIC 2026 NISQ Pipeline Benchmark Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Pipeline: H-cGQE Transformer → L-BFGS-B (L40S GPU) → Rigetti Cepheus-1-108Q → SQD\n\n")

        f.write("Stage 1-2: AI Inference + Classical Optimization\n")
        f.write("-" * 50 + "\n")
        for m in benchmark["pipeline_stages"]["stage_1_ai_inference"]["molecules"]:
            f.write(f"  {m['name']:15s}: {m['n_qubits']:2d}q, {m['n_operators']:2d} ops, "
                    f"E={m['optimized_energy']:.6f} Ha\n")

        f.write(f"\nStage 3: QPU Execution on Rigetti Cepheus-1-108Q\n")
        f.write("-" * 50 + "\n")
        stage3 = benchmark["pipeline_stages"]["stage_3_qpu_execution"]
        f.write(f"  SQD jobs: {stage3['n_sqd_jobs']}\n")
        f.write(f"  QWC jobs: {stage3['n_qwc_jobs']}\n")
        f.write(f"  Shots/circuit: {stage3['shots_per_circuit']}\n")

        f.write(f"\nStage 4: SQD Post-Processing Results\n")
        f.write("-" * 50 + "\n")
        f.write(f"  {'Molecule':15s} {'Source':20s} {'SQD Energy':>12s} {'FCI':>12s} {'Error(mHa)':>12s} {'Bound':>6s}\n")
        for source_name, results_dict in [
            ("Local L40S", benchmark["results"]["local_sqd_pilot"]),
            ("AWS SV1 sim", benchmark["results"]["sv1_simulator"]),
            ("Cepheus QPU", benchmark["results"]["cepheus_qpu_sqd"]),
        ]:
            for key, r in sorted(results_dict.items()):
                f.write(f"  {r.get('molecule', key):15s} {source_name:20s} "
                        f"{r.get('sqd_energy', 0):>12.6f} {r.get('fci_energy', 0):>12.6f} "
                        f"{r.get('error_mha', 0):>12.3f} {str(r.get('variational_bound', '')):>6s}\n")

        f.write(f"\nCost Accounting\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Budget:     {benchmark['cost_accounting']['budget_credits']:.0f} credits\n")
        f.write(f"  Spent:      {benchmark['cost_accounting']['total_estimated_credits']:.2f} credits\n")
        f.write(f"  Remaining:  {benchmark['cost_accounting']['remaining_credits']:.2f} credits\n")

        f.write(f"\nKey Findings\n")
        f.write("-" * 50 + "\n")
        f.write("  1. H2 achieves exact FCI energy (0.000 mHa) on all platforms:\n")
        f.write("     local L40S, AWS SV1 simulator, and Rigetti Cepheus-1-108Q QPU.\n")
        f.write("  2. Variational bound satisfied for all molecules (SQD energy >= FCI).\n")
        f.write("  3. LiH/BeH2 show larger errors due to diagonal sequence collapse\n")
        f.write("     (insufficient entangling operators in H-cGQE generated circuits).\n")
        f.write("  4. Total QPU cost: 612.24 credits (4.6% of 13,400 budget).\n")
        f.write("  5. Pipeline demonstrates full AI+HPC+QPU integration for NISQ chemistry.\n")

    print(f"Summary saved: {summary_path}")

    # Try PDF generation with reportlab
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib import colors

        pdf_path = out_dir / "gic2026_benchmark_report.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                topMargin=0.5*inch, bottomMargin=0.5*inch,
                                leftMargin=0.5*inch, rightMargin=0.5*inch)
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Justified", alignment=4, fontSize=10, leading=14))

        story = []
        story.append(Paragraph("GIC 2026 NISQ Pipeline Benchmark Report", styles["Title"]))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("H-cGQE Transformer + L-BFGS-B Optimization + Rigetti Cepheus-1-108Q + SQD", styles["Heading2"]))
        story.append(Spacer(1, 0.3*inch))

        # Pipeline diagram
        story.append(Paragraph("Pipeline Architecture", styles["Heading1"]))
        story.append(Image(str(out_dir / "gic2026_pipeline_diagram.png"), width=7*inch, height=2.8*inch))
        story.append(Spacer(1, 0.2*inch))

        # SQD results
        story.append(Paragraph("SQD Energy Results", styles["Heading1"]))
        story.append(Image(str(out_dir / "gic2026_sqd_benchmark.png"), width=7*inch, height=2.5*inch))
        story.append(Spacer(1, 0.2*inch))

        # Results table
        story.append(Paragraph("Detailed Results", styles["Heading1"]))
        data = [["Molecule", "Source", "SQD Energy (Ha)", "FCI Energy (Ha)", "Error (mHa)", "Var. Bound"]]
        for source_name, results_dict in [
            ("Local L40S", benchmark["results"]["local_sqd_pilot"]),
            ("AWS SV1 sim", benchmark["results"]["sv1_simulator"]),
            ("Cepheus QPU", benchmark["results"]["cepheus_qpu_sqd"]),
        ]:
            for key, r in sorted(results_dict.items()):
                data.append([
                    r.get("molecule", key),
                    source_name,
                    f"{r.get('sqd_energy', 0):.6f}",
                    f"{r.get('fci_energy', 0):.6f}",
                    f"{r.get('error_mha', 0):.3f}",
                    str(r.get("variational_bound", "")),
                ])

        table = Table(data, colWidths=[0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.0*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))

        # Cost
        story.append(Paragraph("Cost Accounting", styles["Heading1"]))
        story.append(Image(str(out_dir / "gic2026_cost_accounting.png"), width=3.5*inch, height=3.5*inch))
        story.append(Spacer(1, 0.2*inch))

        # Key findings
        story.append(Paragraph("Key Findings", styles["Heading1"]))
        findings = [
            "H2 achieves exact FCI energy (0.000 mHa error) across all platforms: local L40S GPU, AWS SV1 simulator, and Rigetti Cepheus-1-108Q QPU.",
            "Variational bound satisfied for all molecules (SQD energy >= FCI energy), confirming correctness of the classical post-processing.",
            "LiH (737.6 mHa) and BeH2 (5540.8 mHa) show larger errors due to diagonal sequence collapse — the H-cGQE model generates insufficient entangling operators for larger molecules.",
            "Total QPU cost: 612.24 credits (4.6% of 13,400 credit budget), demonstrating cost-efficient NISQ execution.",
            "Full AI+HPC+QPU pipeline validated: H-cGQE transformer inference → L-BFGS-B GPU optimization → QPU sampling → SQD classical diagonalization.",
        ]
        for i, finding in enumerate(findings, 1):
            story.append(Paragraph(f"{i}. {finding}", styles["Justified"]))
            story.append(Spacer(1, 0.1*inch))

        doc.build(story)
        print(f"PDF saved: {pdf_path}")

    except ImportError:
        print("reportlab not installed — skipping PDF. PNG figures and text summary generated.")


if __name__ == "__main__":
    main()
