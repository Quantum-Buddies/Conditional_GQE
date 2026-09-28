#!/usr/bin/env python3
"""Generate the official cover plus the first two Phase 3 technical pages.

This is deliberately a partial submission draft. It contains:
  1. the unmodified official Phase 3 cover from the existing submission;
  2. an executive summary and end-to-end workflow page;
  3. an architecture, hybrid-compute partition, and evaluation-design page.

Results are intentionally deferred until the final benchmark artifacts have
been validated. All diagrams are vector graphics drawn directly into the PDF.
"""
from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COVER = ROOT / "proposals/Ryoushi_Quantum_Buddies_Phase3_Version1.pdf"
OUTPUT = ROOT / "proposals/Ryoushi_Quantum_Buddies_Phase3_Front_Matter_Draft.pdf"

NAVY = colors.HexColor("#15324A")
BLUE = colors.HexColor("#2878A5")
TEAL = colors.HexColor("#2A8C82")
GOLD = colors.HexColor("#C6922E")
INK = colors.HexColor("#202A33")
MUTED = colors.HexColor("#536370")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_TEAL = colors.HexColor("#E8F4F1")
PALE_GOLD = colors.HexColor("#F7F0DF")
PALE_GREY = colors.HexColor("#F3F5F6")
RULE = colors.HexColor("#C7D0D6")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Phase3Title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=17,
            leading=19,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=2.5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Phase3Subtitle",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=11,
            leading=11.8,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "h1": ParagraphStyle(
            "Phase3H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=12.5,
            leading=14,
            textColor=NAVY,
            spaceBefore=3.2 * mm,
            spaceAfter=1.8 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Phase3H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=12,
            textColor=BLUE,
            spaceBefore=2.4 * mm,
            spaceAfter=1.2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Phase3Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=11,
            leading=11.8,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=2.1 * mm,
        ),
        "small": ParagraphStyle(
            "Phase3Small",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.2,
            leading=10.2,
            textColor=INK,
        ),
        "small_center": ParagraphStyle(
            "Phase3SmallCenter",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.8,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Phase3Callout",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10.4,
            leading=11.3,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "Phase3TableHead",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=9.2,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "Phase3Table",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.9,
            leading=9.7,
            textColor=INK,
        ),
    }


class PipelineDiagram(Flowable):
    """Five-stage workflow with a visible quantum-feedback loop."""

    def __init__(self, width: float, height: float = 47 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        c = self.canv
        stages = [
            ("1", "CHEMISTRY", "Molecular graph\n+ Pauli Hamiltonian", PALE_GREY, NAVY),
            ("2", "GENERATE", "Conditional Transformer\nsamples operator sequences", PALE_BLUE, BLUE),
            ("3", "REFINE", "L-BFGS-B optimizes\ncontinuous angles", PALE_GOLD, GOLD),
            ("4", "EXECUTE", "CUDA-Q simulator\nor qBraid QPU", PALE_TEAL, TEAL),
            ("5", "RECOVER", "SQD / expectation\npost-processing", PALE_BLUE, NAVY),
        ]
        gap = 4 * mm
        box_w = (self.width - gap * 4) / 5
        box_h = 28 * mm
        y = 13 * mm

        for idx, (number, heading, body, fill, accent) in enumerate(stages):
            x = idx * (box_w + gap)
            c.setFillColor(fill)
            c.setStrokeColor(accent)
            c.setLineWidth(1)
            c.roundRect(x, y, box_w, box_h, 2.2 * mm, fill=1, stroke=1)
            c.setFillColor(accent)
            c.circle(x + 5 * mm, y + box_h - 5 * mm, 3.1 * mm, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Times-Bold", 8)
            c.drawCentredString(x + 5 * mm, y + box_h - 6 * mm, number)
            c.setFillColor(accent)
            c.setFont("Times-Bold", 8.4)
            c.drawString(x + 9.5 * mm, y + box_h - 6.2 * mm, heading)
            c.setFillColor(INK)
            c.setFont("Times-Roman", 7.7)
            for line_no, line in enumerate(body.splitlines()):
                c.drawCentredString(x + box_w / 2, y + 10.7 * mm - line_no * 3.3 * mm, line)

            if idx < len(stages) - 1:
                start_x = x + box_w + 0.6 * mm
                end_x = x + box_w + gap - 0.6 * mm
                arrow_y = y + box_h / 2
                c.setStrokeColor(MUTED)
                c.setFillColor(MUTED)
                c.setLineWidth(1.1)
                c.line(start_x, arrow_y, end_x, arrow_y)
                c.line(end_x, arrow_y, end_x - 1.8 * mm, arrow_y + 1.2 * mm)
                c.line(end_x, arrow_y, end_x - 1.8 * mm, arrow_y - 1.2 * mm)

        # Feedback line: measured energy is the training signal.
        left = box_w + gap + box_w / 2
        right = 3 * (box_w + gap) + box_w / 2
        feedback_y = 6 * mm
        c.setStrokeColor(TEAL)
        c.setFillColor(TEAL)
        c.setLineWidth(1.2)
        c.line(right, y, right, feedback_y)
        c.line(right, feedback_y, left, feedback_y)
        c.line(left, feedback_y, left, y)
        c.line(left, y, left - 1.2 * mm, y - 1.8 * mm)
        c.line(left, y, left + 1.2 * mm, y - 1.8 * mm)
        label = "quantum energy feedback updates the policy"
        c.setFillColor(colors.white)
        label_w = stringWidth(label, "Times-Italic", 7.8) + 4 * mm
        c.rect((left + right - label_w) / 2, feedback_y - 1.8 * mm, label_w, 3.8 * mm, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.setFont("Times-Italic", 7.8)
        c.drawCentredString((left + right) / 2, feedback_y - 0.5 * mm, label)


class ArchitectureDiagram(Flowable):
    """Conditional model architecture and constrained decoder."""

    def __init__(self, width: float, height: float = 61 * mm) -> None:
        super().__init__()
        self.width = width
        self.height = height

    @staticmethod
    def _box(c, x, y, w, h, title, lines, fill, stroke) -> None:
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 2 * mm, fill=1, stroke=1)
        c.setFillColor(stroke)
        c.setFont("Times-Bold", 8.5)
        c.drawCentredString(x + w / 2, y + h - 5 * mm, title)
        c.setFillColor(INK)
        c.setFont("Times-Roman", 7.6)
        for i, line in enumerate(lines):
            c.drawCentredString(x + w / 2, y + h - 10 * mm - i * 3.2 * mm, line)

    @staticmethod
    def _arrow(c, x1, y1, x2, y2, color=MUTED) -> None:
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(1.1)
        c.line(x1, y1, x2, y2)
        angle = 1 if x2 >= x1 else -1
        c.line(x2, y2, x2 - angle * 1.8 * mm, y2 + 1.2 * mm)
        c.line(x2, y2, x2 - angle * 1.8 * mm, y2 - 1.2 * mm)

    def draw(self) -> None:
        c = self.canv
        left_w = 34 * mm
        encoder_w = 43 * mm
        decoder_w = 51 * mm
        output_w = 35 * mm
        gap = (self.width - left_w - encoder_w - decoder_w - output_w) / 3
        y_mid = 24 * mm

        self._box(
            c, 0, y_mid + 20 * mm, left_w, 16 * mm,
            "MOLECULAR GRAPH",
            ["atoms, bonds, geometry", "charge and active space"],
            PALE_TEAL, TEAL,
        )
        self._box(
            c, 0, y_mid - 2 * mm, left_w, 16 * mm,
            "HAMILTONIAN",
            ["Pauli words P_l", "coefficients h_l"],
            PALE_BLUE, BLUE,
        )

        encoder_x = left_w + gap
        self._box(
            c, encoder_x, y_mid + 8 * mm, encoder_w, 28 * mm,
            "CONDITIONING ENCODERS",
            ["Chemistry GNN", "Hamiltonian Transformer", "cross-molecule context"],
            PALE_GREY, NAVY,
        )

        decoder_x = encoder_x + encoder_w + gap
        self._box(
            c, decoder_x, y_mid + 5 * mm, decoder_w, 34 * mm,
            "AUTOREGRESSIVE DECODER",
            ["causal self-attention", "cross-attention to H", "operator token distribution", "variable-length sequence"],
            PALE_BLUE, BLUE,
        )
        c.setFillColor(PALE_GOLD)
        c.setStrokeColor(GOLD)
        c.roundRect(decoder_x + 4 * mm, y_mid - 10 * mm, decoder_w - 8 * mm, 11 * mm, 1.8 * mm, fill=1, stroke=1)
        c.setFillColor(GOLD)
        c.setFont("Times-Bold", 7.8)
        c.drawCentredString(decoder_x + decoder_w / 2, y_mid - 4.2 * mm, "PHYSICS-AWARE CONSTRAINTS")
        c.setFillColor(INK)
        c.setFont("Times-Roman", 7.1)
        c.drawCentredString(
            decoder_x + decoder_w / 2,
            y_mid - 7.7 * mm,
            "UCCSD pool  |  entanglement mask  |  symmetry checks",
        )

        output_x = decoder_x + decoder_w + gap
        self._box(
            c, output_x, y_mid + 8 * mm, output_w, 28 * mm,
            "COMPACT ANSATZ",
            ["P_1, P_2, ..., P_k", "continuous theta refined", "after generation"],
            PALE_TEAL, TEAL,
        )

        self._arrow(c, left_w, y_mid + 28 * mm, encoder_x, y_mid + 28 * mm)
        self._arrow(c, left_w, y_mid + 6 * mm, encoder_x, y_mid + 17 * mm)
        self._arrow(c, encoder_x + encoder_w, y_mid + 22 * mm, decoder_x, y_mid + 22 * mm)
        self._arrow(c, decoder_x + decoder_w, y_mid + 22 * mm, output_x, y_mid + 22 * mm)

        # Stage labels across the bottom.
        c.setStrokeColor(RULE)
        c.line(0, 2 * mm, self.width, 2 * mm)
        labels = [
            (0, "CHEMISTRY INPUT"),
            (encoder_x, "CONDITION"),
            (decoder_x, "GENERATE"),
            (output_x, "REFINE + EXECUTE"),
        ]
        c.setFillColor(MUTED)
        c.setFont("Times-Bold", 7)
        for x, text in labels:
            c.drawString(x, -1.5 * mm, text)


def _callout(text: str, style: ParagraphStyle, width: float, fill=PALE_GOLD) -> Table:
    table = Table([[Paragraph(text, style)]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.8, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4 * mm),
            ]
        )
    )
    return table


def _page_number(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 12 * mm, width - doc.rightMargin, 12 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawString(doc.leftMargin, 8 * mm, "Global Industry Challenge 2026 — Phase 3")
    canvas.drawRightString(width - doc.rightMargin, 8 * mm, f"Technical page {doc.page}")
    canvas.restoreState()


def _technical_pdf() -> bytes:
    styles = _styles()
    stream = io.BytesIO()
    doc = SimpleDocTemplate(
        stream,
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=14 * mm,
        bottomMargin=17 * mm,
        title="H-cGQE Phase 3 Front Matter",
        author="Ryoushi | Quantum Buddies",
    )
    content_w = A4[0] - doc.leftMargin - doc.rightMargin

    story = [
        Paragraph(
            "Scaling Generative Quantum Eigensolvers for EUV Materials",
            styles["title"],
        ),
        Paragraph(
            "H-cGQE: chemistry-conditioned circuit generation, quantum-feedback training, "
            "and reproducible hybrid execution",
            styles["subtitle"],
        ),
        Paragraph("1. Executive Summary", styles["h1"]),
        Paragraph(
            "<b>Industrial motivation.</b> Accurate electronic-structure calculations are central "
            "to screening halogenated aromatic materials used in extreme-ultraviolet (EUV) "
            "lithography. The difficult regime is not merely producing one low-energy circuit; "
            "it is repeatedly designing compact ansätze across related molecules while controlling "
            "simulation cost, circuit depth, and hardware noise. Our prototype treats ansatz design "
            "as a learned, reusable mapping from molecular context and a qubit Hamiltonian to an "
            "operator sequence.",
            styles["body"],
        ),
        Paragraph(
            "<b>Phase 2 finding.</b> Supervised generation alone exhibited <i>diagonal sequence "
            "collapse</i>: on larger systems the decoder preferred commuting I/Z operators that "
            "leave a Hartree–Fock reference unchanged apart from phase. This produced a flat "
            "continuous optimization landscape and exposed a general failure mode of generative "
            "eigensolvers—not simply a tuning issue.",
            styles["body"],
        ),
        Paragraph(
            "<b>Phase 3 prototype.</b> We implement a supervised warm start followed by DAPO-style "
            "reinforcement learning from CUDA-Q energy feedback. A UCCSD-derived operator pool, "
            "entanglement-aware decoding, and energy-gated auxiliary rewards keep exploration "
            "within physically useful circuit families. Generated topology and continuous "
            "coefficients are deliberately separated: the Transformer proposes discrete Pauli "
            "operators, while bounded L-BFGS-B refines rotation angles. Selected circuits are then "
            "executed on a simulator or QPU and post-processed with Hamiltonian-aware estimators.",
            styles["body"],
        ),
        Spacer(1, 1.5 * mm),
        PipelineDiagram(content_w),
        Paragraph(
            "Figure 1. End-to-end hybrid workflow. Quantum execution supplies both evaluation "
            "evidence and the energy signal used to improve the generative policy.",
            styles["small_center"],
        ),
        Spacer(1, 2 * mm),
        _callout(
            "<b>Design principle.</b> Expensive quantum work is reserved for state preparation and "
            "sampling. Molecular construction, conditional generation, coefficient optimization, "
            "subspace recovery, and provenance tracking remain classical and parallelizable.",
            styles["callout"],
            content_w,
        ),
        Paragraph("1.1 Phase 3 evaluation questions", styles["h2"]),
    ]

    metric_data = [
        [
            Paragraph("Question", styles["table_head"]),
            Paragraph("Measured evidence", styles["table_head"]),
            Paragraph("Primary metric", styles["table_head"]),
        ],
        [
            Paragraph("Does quantum feedback prevent diagonal collapse?", styles["table"]),
            Paragraph("Supervised and RL circuits evaluated on identical Hamiltonians", styles["table"]),
            Paragraph("Correlation energy and entangling-operator fraction", styles["table"]),
        ],
        [
            Paragraph("Is the workflow practical on available platforms?", styles["table"]),
            Paragraph("CUDA-Q GPU execution and qBraid-managed QPU jobs", styles["table"]),
            Paragraph("Qubits, depth, shots, wall-clock, and credits", styles["table"]),
        ],
        [
            Paragraph("Can each headline value be independently checked?", styles["table"]),
            Paragraph("Versioned JSON artifacts, manifests, and deterministic post-processing", styles["table"]),
            Paragraph("Energy error versus a same-instance classical reference", styles["table"]),
        ],
    ]
    metrics = Table(metric_data, colWidths=[0.29 * content_w, 0.44 * content_w, 0.27 * content_w])
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.6 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * mm),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GREY]),
            ]
        )
    )
    story.extend([metrics, PageBreak()])

    story.extend(
        [
            Paragraph("2. Technical Approach and Hybrid Architecture", styles["h1"]),
            Paragraph(
                "For a molecular Hamiltonian H = Σ_l h_l P_l, H-cGQE models a conditional "
                "distribution over variable-length operator sequences. Conditioning combines "
                "molecular graph information with Pauli-term structure; the decoder predicts a "
                "compact ordered ansatz rather than regressing the energy directly. This allows "
                "one policy to serve multiple molecules and active spaces while preserving an "
                "explicit, inspectable circuit representation.",
                styles["body"],
            ),
            ArchitectureDiagram(content_w),
            Paragraph(
                "Figure 2. H-cGQE separates chemistry conditioning, discrete circuit topology, "
                "and continuous coefficient refinement. Physics-aware constraints act during "
                "decoding rather than attempting to repair invalid sequences afterward.",
                styles["small_center"],
            ),
            Paragraph("2.1 Training strategy: valid prior, then physical feedback", styles["h2"]),
            Paragraph(
                "<b>Supervised warm start.</b> Teacher-forced sequence learning establishes the "
                "operator vocabulary, termination behavior, and a non-collapsed prior. "
                "<b>Quantum-feedback fine-tuning.</b> Groups of sampled circuits are ranked by "
                "their measured or simulated energy after short coefficient refinement. "
                "Group-relative advantages, asymmetric policy clipping, entropy control, and "
                "persistent circuit-to-energy caching stabilize updates. Structural rewards are "
                "gated on improvement over Hartree–Fock so that shallow or unusual circuits are "
                "not rewarded unless they also improve the physical objective.",
                styles["body"],
            ),
            Paragraph("2.2 Division of work across the hybrid system", styles["h2"]),
        ]
    )

    partition_data = [
        [
            Paragraph("Component", styles["table_head"]),
            Paragraph("Responsibility", styles["table_head"]),
            Paragraph("Why it is placed there", styles["table_head"]),
        ],
        [
            Paragraph("<b>CPU chemistry</b>", styles["table"]),
            Paragraph("PySCF integrals, active spaces, Jordan–Wigner mapping, references", styles["table"]),
            Paragraph("Deterministic preprocessing; parallel across molecules/fragments", styles["table"]),
        ],
        [
            Paragraph("<b>GPU AI</b>", styles["table"]),
            Paragraph("Conditional generation, policy updates, candidate ranking", styles["table"]),
            Paragraph("Batched autoregressive inference and mixed-precision training", styles["table"]),
        ],
        [
            Paragraph("<b>GPU simulator</b>", styles["table"]),
            Paragraph("Expectation values and bounded coefficient refinement", styles["table"]),
            Paragraph("Fast exact feedback for tractable active spaces", styles["table"]),
        ],
        [
            Paragraph("<b>QPU via qBraid</b>", styles["table"]),
            Paragraph("Selected circuit sampling on a concrete hardware backend", styles["table"]),
            Paragraph("Tests physical realizability and hardware-facing resource costs", styles["table"]),
        ],
        [
            Paragraph("<b>Classical recovery</b>", styles["table"]),
            Paragraph("SQD/subspace diagonalization, aggregation, uncertainty, manifests", styles["table"]),
            Paragraph("Converts sampled configurations into auditable energy estimates", styles["table"]),
        ],
    ]
    partition = Table(partition_data, colWidths=[0.19 * content_w, 0.40 * content_w, 0.41 * content_w])
    partition.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.45, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_GREY]),
            ]
        )
    )
    story.append(partition)
    story.extend(
        [
            Paragraph("2.3 Scalability and scientific controls", styles["h2"]),
            Paragraph(
                "Scaling claims are separated into measured circuit width and effective parent-system "
                "size. Exact statevector evaluation is restricted to validated hardware limits; larger "
                "systems use tensor-network or fragment/subspace paths only when their approximation "
                "error can be reported independently. Every energy comparison uses the same Hamiltonian "
                "instance and active space. FCI/CASCI is used where tractable; correlated classical "
                "references such as CCSD(T) are required when exact diagonalization is unavailable. "
                "The final results section will report qubit count, transpiled depth, two-qubit gates, "
                "shots, wall-clock time, and error for each headline experiment.",
                styles["body"],
            ),
            KeepTogether(
                [
                    _callout(
                        "<b>Reproducibility boundary.</b> Training is not required to verify the "
                        "hardware-facing claims. The submission package will include the exact circuit "
                        "manifests, Hamiltonians, raw-count summaries, deterministic recovery code, "
                        "dependency versions, and expected numeric outputs used in the final tables.",
                        styles["callout"],
                        content_w,
                        fill=PALE_TEAL,
                    )
                ]
            ),
        ]
    )

    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    return stream.getvalue()


def main() -> None:
    if not SOURCE_COVER.exists():
        raise FileNotFoundError(f"Official Phase 3 cover source not found: {SOURCE_COVER}")

    technical_reader = PdfReader(io.BytesIO(_technical_pdf()))
    cover_reader = PdfReader(str(SOURCE_COVER))

    writer = PdfWriter()
    writer.add_page(cover_reader.pages[0])
    for page in technical_reader.pages:
        writer.add_page(page)

    writer.add_metadata(
        {
            "/Title": "Ryoushi | Quantum Buddies — Phase 3 Front Matter Draft",
            "/Author": "Ryoushi | Quantum Buddies",
            "/Subject": "GIC 2026 Quantum Materials Discovery Challenge",
        }
    )
    with OUTPUT.open("wb") as handle:
        writer.write(handle)

    print(f"Created: {OUTPUT}")
    print(f"Pages: {len(writer.pages)} (official cover + 2 technical pages)")
    print("Results pages intentionally deferred.")


if __name__ == "__main__":
    main()
