"""Hand Cartesian starting guesses for labelled-ours R–Sn(OH)₃.

These coordinates are **not** MatGen-Q Table 1 / SI. They are textbook-length
tetrahedral / sp² templates so PySCF can opt without RDKit.

Bond lengths (Å), generic organotin / organic values — not a paper SI:
  Sn–C alkyl 2.15, Sn–C vinyl 2.12, Sn–C aryl 2.13, Sn–C CF3 2.18
  Sn–O 1.96, O–H 0.96, C–H 1.09, C–C 1.54, C=C 1.34, C–C(Ar) 1.39, C–F 1.33

CXRO f2 at 13.5 nm (91.84 eV) is a **proxy** (Henke atomic scattering factors),
not an ab initio σ(92 eV).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# --- labelled-ours metric constants ------------------------------------------
R_SN_C_ALKYL = 2.15
R_SN_C_VINYL = 2.12
R_SN_C_ARYL = 2.13
R_SN_C_CF3 = 2.18
R_SN_O = 1.96
R_OH = 0.96
R_CH = 1.09
R_CH_AR = 1.08
R_CC = 1.54
R_CDOUBLE = 1.34
R_CC_AR = 1.39
R_CF = 1.33
TETRA_COS = -1.0 / 3.0  # cos(109.471°)

# 13.5 nm = 91.841 eV. Linear interpolation on CXRO / Henke f2 grids
# (https://henke.lbl.gov/optical_constants/sf/<el>.nff) between 91.5395 and
# 93.0201 eV. PROXY only — not ab initio photoabsorption.
CXRO_E_EV = 91.841
CXRO_WAVELENGTH_NM = 13.5
CXRO_F2_13P5NM: dict[str, float] = {
    "Sn": 23.9718,
    "I": 30.4673,
    "F": 4.2731,
    "O": 2.7652,
    "C": 0.7659,
    "H": 0.03296,
}
CXRO_CITATION = (
    "Henke, Gullikson, Davis, At. Data Nucl. Data Tables 54, 181 (1993); "
    "CXRO atomic scattering factors f2 interpolated at 91.84 eV (13.5 nm). "
    "This is a CXRO-style atomic **proxy**, not ab initio σ(92 eV)."
)

FIRST_WAVE: tuple[str, ...] = ("me", "et", "nbu", "vinyl", "ph")
UNEVALUATED_POOL: tuple[str, ...] = ("ipr", "npr", "allyl", "cf3")
ALL_LIGAND_IDS: tuple[str, ...] = FIRST_WAVE + UNEVALUATED_POOL

LIGAND_SMILES: dict[str, str] = {
    "me": "C[Sn](O)(O)O",
    "et": "CC[Sn](O)(O)O",
    "nbu": "CCCC[Sn](O)(O)O",
    "vinyl": "C=C[Sn](O)(O)O",
    "ph": "c1ccccc1[Sn](O)(O)O",
    "ipr": "CC(C)[Sn](O)(O)O",
    "npr": "CCC[Sn](O)(O)O",
    "allyl": "C=CC[Sn](O)(O)O",
    "cf3": "FC(F)(F)[Sn](O)(O)O",
}

LIGAND_NAMES: dict[str, str] = {
    "me": "methyltin_ours",
    "et": "ethyltin_ours",
    "nbu": "nbutyltin_ours",
    "vinyl": "vinyltin_ours",
    "ph": "phenyltin_ours",
    "ipr": "isopropyltin_ours",
    "npr": "npropyltin_ours",
    "allyl": "allyltin_ours",
    "cf3": "trifluoromethyltin_ours",
}

# Neutral closed-shell formulae for the parent RSn(OH)3 (sanity checks).
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "me": {"C": 1, "H": 6, "O": 3, "Sn": 1},
    "et": {"C": 2, "H": 8, "O": 3, "Sn": 1},
    "nbu": {"C": 4, "H": 12, "O": 3, "Sn": 1},
    "vinyl": {"C": 2, "H": 6, "O": 3, "Sn": 1},
    "ph": {"C": 6, "H": 8, "O": 3, "Sn": 1},
    "ipr": {"C": 3, "H": 10, "O": 3, "Sn": 1},
    "npr": {"C": 3, "H": 10, "O": 3, "Sn": 1},
    "allyl": {"C": 3, "H": 8, "O": 3, "Sn": 1},
    "cf3": {"C": 1, "F": 3, "H": 3, "O": 3, "Sn": 1},
}

ROLE_SN = "sn"
ROLE_OH_O = "oh_o"
ROLE_OH_H = "oh_h"
ROLE_LIGAND = "ligand"

Vec = np.ndarray
AtomTriple = tuple[str, tuple[float, float, float], str]


def _as_xyz(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector in template builder")
    return v / n


def _rot_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    k = _norm(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def _frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = _norm(axis)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _norm(np.cross(z, tmp))
    y = np.cross(z, x)
    return x, y, z


def tetrahedral_dirs(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four unit vectors; the first is +axis, the rest complete a tetrahedron."""
    x, y, z = _frame(axis)
    rest: list[np.ndarray] = []
    for ang in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        vec = -z / 3.0 + np.sqrt(8.0 / 9.0) * (np.cos(ang) * x + np.sin(ang) * y)
        rest.append(_norm(vec))
    return z, rest[0], rest[1], rest[2]


def _dihedral_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = _norm(n1)
    n2 = _norm(n2)
    m1 = np.cross(n1, _norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def place_trans(a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float) -> np.ndarray:
    """Place D so A–B–C–D is anti (dihedral ~180°) with a tetrahedral B–C–D angle."""
    incoming = c - b
    n = np.cross(b - a, c - b)
    n = _norm(n)
    rot = np.deg2rad(180.0 - np.degrees(np.arccos(TETRA_COS)))
    cand_plus = c + bond * _norm(_rot_axis(incoming, n, rot))
    cand_minus = c + bond * _norm(_rot_axis(incoming, n, -rot))
    d_plus = abs(abs(_dihedral_deg(a, b, c, cand_plus)) - 180.0)
    d_minus = abs(abs(_dihedral_deg(a, b, c, cand_minus)) - 180.0)
    return cand_plus if d_plus <= d_minus else cand_minus


def ch2_hydrogens(center: np.ndarray, nbr_a: np.ndarray, nbr_b: np.ndarray, r_ch: float) -> list[np.ndarray]:
    v1 = _norm(nbr_a - center)
    v2 = _norm(nbr_b - center)
    s = v1 + v2
    n = _norm(np.cross(v1, v2))
    half = 0.5 * s
    w = np.sqrt(2.0 / 3.0) * n
    h1 = _norm(-half + w)
    h2 = _norm(-half - w)
    return [center + r_ch * h1, center + r_ch * h2]


def terminal_ch3_hydrogens(center: np.ndarray, prev: np.ndarray, r_ch: float) -> list[np.ndarray]:
    _d0, d1, d2, d3 = tetrahedral_dirs(prev - center)
    return [center + r_ch * d for d in (d1, d2, d3)]


def _atom(sym: str, xyz: np.ndarray, role: str) -> AtomTriple:
    return (sym, _as_xyz(xyz), role)


def sn_oh3_atoms(sn: np.ndarray | None = None) -> list[AtomTriple]:
    """Sn(OH)₃ with Sn at `sn` (default origin); ligand along +z is left vacant."""
    origin = np.zeros(3) if sn is None else np.asarray(sn, dtype=float)
    _along, o1, o2, o3 = tetrahedral_dirs(np.array([0.0, 0.0, 1.0]))
    atoms: list[AtomTriple] = [_atom("Sn", origin, ROLE_SN)]
    for i, d in enumerate((o1, o2, o3)):
        o = origin + R_SN_O * d
        twist = i * (2.0 * np.pi / 3.0)
        n = _norm(o - origin)
        tmp = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        perp = _norm(np.cross(n, tmp))
        w = _rot_axis(perp, n, twist)
        u_oh = _norm((1.0 / 3.0) * n + np.sqrt(8.0 / 9.0) * w)
        h = o + R_OH * u_oh
        atoms.append(_atom("O", o, ROLE_OH_O))
        atoms.append(_atom("H", h, ROLE_OH_H))
    return atoms


def _methyl_on_axis(c: np.ndarray, toward_sn: np.ndarray, r_xh: float, x_sym: str) -> list[AtomTriple]:
    hs = terminal_ch3_hydrogens(c, toward_sn, r_xh)
    return [_atom(x_sym, h, ROLE_LIGAND) for h in hs]


def _alkyl_chain(n_carbon: int) -> list[AtomTriple]:
    """All-trans n-alkyl C1…Cn attached to Sn at the origin along +z. C1 is alpha."""
    if n_carbon < 1:
        raise ValueError("alkyl chain needs ≥1 carbon")
    sn = np.zeros(3)
    carbons: list[np.ndarray] = [np.array([0.0, 0.0, R_SN_C_ALKYL])]
    if n_carbon >= 2:
        _d0, d_next, _h1, _h2 = tetrahedral_dirs(sn - carbons[0])
        carbons.append(carbons[0] + R_CC * d_next)
    for i in range(2, n_carbon):
        carbons.append(place_trans(carbons[i - 3] if i >= 3 else sn, carbons[i - 2], carbons[i - 1], R_CC))
    atoms: list[AtomTriple] = [_atom("C", c, ROLE_LIGAND) for c in carbons]
    neighbors: list[list[np.ndarray]] = [[] for _ in carbons]
    neighbors[0].append(sn)
    for i in range(n_carbon - 1):
        neighbors[i].append(carbons[i + 1])
        neighbors[i + 1].append(carbons[i])
    for i, c in enumerate(carbons):
        n_bonds = len(neighbors[i])
        if n_bonds == 1:
            for h in terminal_ch3_hydrogens(c, neighbors[i][0], R_CH):
                atoms.append(_atom("H", h, ROLE_LIGAND))
        elif n_bonds == 2:
            for h in ch2_hydrogens(c, neighbors[i][0], neighbors[i][1], R_CH):
                atoms.append(_atom("H", h, ROLE_LIGAND))
        else:
            raise ValueError(f"unexpected coordination {n_bonds} on alkyl C{i}")
    return atoms


def _vinyl_ligand() -> list[AtomTriple]:
    ca = np.array([0.0, 0.0, R_SN_C_VINYL])
    u_cb = np.array([np.sqrt(0.75), 0.0, 0.5])
    cb = ca + R_CDOUBLE * u_cb
    u_ca_h = np.array([-np.sqrt(0.75), 0.0, 0.5])
    ha = ca + R_CH * u_ca_h
    u_to_ca = -u_cb
    hb_cis = cb + R_CH * _norm(_rot_axis(u_to_ca, np.array([0.0, 1.0, 0.0]), np.deg2rad(120.0)))
    hb_trans = cb + R_CH * _norm(_rot_axis(u_to_ca, np.array([0.0, 1.0, 0.0]), np.deg2rad(-120.0)))
    return [
        _atom("C", ca, ROLE_LIGAND),
        _atom("C", cb, ROLE_LIGAND),
        _atom("H", ha, ROLE_LIGAND),
        _atom("H", hb_cis, ROLE_LIGAND),
        _atom("H", hb_trans, ROLE_LIGAND),
    ]


def _allyl_ligand() -> list[AtomTriple]:
    """Sn–CH2–CH=CH2, CH2 tetrahedral, vinyl plane xz."""
    sn = np.zeros(3)
    ca = np.array([0.0, 0.0, R_SN_C_ALKYL])
    _d0, d_next, _h1, _h2 = tetrahedral_dirs(sn - ca)
    cb = ca + R_CC * d_next
    # Vinyl on Cb–Cc, trans-ish in the Sn–Ca–Cb plane.
    cc = place_trans(sn, ca, cb, R_CDOUBLE)
    # Rebuild Cc at C=C length along the same direction.
    cc = cb + R_CDOUBLE * _norm(cc - cb)
    atoms = [
        _atom("C", ca, ROLE_LIGAND),
        _atom("C", cb, ROLE_LIGAND),
        _atom("C", cc, ROLE_LIGAND),
    ]
    for h in ch2_hydrogens(ca, sn, cb, R_CH):
        atoms.append(_atom("H", h, ROLE_LIGAND))
    # sp2 H on Cb (one) and two on Cc, in the Sn–Ca–Cb plane.
    n = _norm(np.cross(ca - sn, cb - ca))
    u_bc = _norm(ca - cb)
    u_cb = _norm(cc - cb)
    # Remaining sp2 direction at Cb: opposite the bisector of Ca and Cc, in plane.
    bis = _norm(u_bc + u_cb)
    h_b = cb - R_CH * bis
    atoms.append(_atom("H", h_b, ROLE_LIGAND))
    u_to_cb = _norm(cb - cc)
    h_c1 = cc + R_CH * _norm(_rot_axis(u_to_cb, n, np.deg2rad(120.0)))
    h_c2 = cc + R_CH * _norm(_rot_axis(u_to_cb, n, np.deg2rad(-120.0)))
    atoms.append(_atom("H", h_c1, ROLE_LIGAND))
    atoms.append(_atom("H", h_c2, ROLE_LIGAND))
    return atoms


def _phenyl_ligand() -> list[AtomTriple]:
    ipso = np.array([0.0, 0.0, R_SN_C_ARYL])
    center = ipso + np.array([0.0, 0.0, R_CC_AR])
    atoms: list[AtomTriple] = []
    carbons: list[np.ndarray] = []
    for k in range(6):
        ang = k * np.pi / 3.0
        c = center + R_CC_AR * np.array([np.sin(ang), 0.0, -np.cos(ang)])
        carbons.append(c)
        atoms.append(_atom("C", c, ROLE_LIGAND))
    for k, c in enumerate(carbons):
        if k == 0:
            continue
        radial = _norm(c - center)
        atoms.append(_atom("H", c + R_CH_AR * radial, ROLE_LIGAND))
    return atoms


def _isopropyl_ligand() -> list[AtomTriple]:
    sn = np.zeros(3)
    ca = np.array([0.0, 0.0, R_SN_C_ALKYL])
    _to_sn, d_me1, d_me2, d_h = tetrahedral_dirs(sn - ca)
    me1 = ca + R_CC * d_me1
    me2 = ca + R_CC * d_me2
    h_a = ca + R_CH * d_h
    atoms = [
        _atom("C", ca, ROLE_LIGAND),
        _atom("C", me1, ROLE_LIGAND),
        _atom("C", me2, ROLE_LIGAND),
        _atom("H", h_a, ROLE_LIGAND),
    ]
    for me in (me1, me2):
        atoms.extend(_methyl_on_axis(me, ca, R_CH, "H"))
    return atoms


def _cf3_ligand() -> list[AtomTriple]:
    c = np.array([0.0, 0.0, R_SN_C_CF3])
    atoms = [_atom("C", c, ROLE_LIGAND)]
    atoms.extend(_methyl_on_axis(c, np.zeros(3), R_CF, "F"))
    return atoms


def _ligand_atoms(ligand_id: str) -> list[AtomTriple]:
    key = ligand_id.lower()
    if key == "me":
        c = np.array([0.0, 0.0, R_SN_C_ALKYL])
        return [_atom("C", c, ROLE_LIGAND), *_methyl_on_axis(c, np.zeros(3), R_CH, "H")]
    if key == "et":
        return _alkyl_chain(2)
    if key == "npr":
        return _alkyl_chain(3)
    if key == "nbu":
        return _alkyl_chain(4)
    if key == "ipr":
        return _isopropyl_ligand()
    if key == "vinyl":
        return _vinyl_ligand()
    if key == "allyl":
        return _allyl_ligand()
    if key == "ph":
        return _phenyl_ligand()
    if key == "cf3":
        return _cf3_ligand()
    raise KeyError(f"unknown ligand {ligand_id!r}; known: {', '.join(ALL_LIGAND_IDS)}")


@dataclass(frozen=True)
class MoleculeGuess:
    ligand_id: str
    smiles: str
    name: str
    atoms: tuple[AtomTriple, ...]
    provenance: str

    def cartesian(self) -> list[list[object]]:
        return [[sym, list(xyz)] for sym, xyz, _role in self.atoms]

    def pyscf_atom(self) -> str:
        lines = [f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}" for sym, xyz, _r in self.atoms]
        return "; ".join(lines)

    def by_role(self, *roles: str) -> list[AtomTriple]:
        wanted = set(roles)
        return [a for a in self.atoms if a[2] in wanted]

    def fragment_sn_oh3(self) -> list[AtomTriple]:
        return self.by_role(ROLE_SN, ROLE_OH_O, ROLE_OH_H)

    def fragment_r(self) -> list[AtomTriple]:
        return self.by_role(ROLE_LIGAND)

    def element_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sym, _xyz, _role in self.atoms:
            counts[sym] = counts.get(sym, 0) + 1
        return dict(sorted(counts.items()))

    def xyz_block(self, comment: str) -> str:
        lines = [str(len(self.atoms)), comment]
        for sym, xyz, _role in self.atoms:
            lines.append(f"{sym:2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
        return "\n".join(lines) + "\n"


def build_rsn_oh3(ligand_id: str) -> MoleculeGuess:
    key = ligand_id.lower().strip()
    if key not in LIGAND_SMILES:
        raise KeyError(f"unknown ligand {ligand_id!r}; known: {', '.join(ALL_LIGAND_IDS)}")
    atoms = tuple(sn_oh3_atoms() + _ligand_atoms(key))
    guess = MoleculeGuess(
        ligand_id=key,
        smiles=LIGAND_SMILES[key],
        name=LIGAND_NAMES[key],
        atoms=atoms,
        provenance=(
            "labelled-ours hand Cartesian template; NOT MatGen-Q Table 1; "
            f"ligand={key}; SMILES={LIGAND_SMILES[key]}; "
            "src/gqe/data/rsn_oh3_templates.py"
        ),
    )
    expected = EXPECTED_COUNTS[key]
    got = guess.element_counts()
    if got != expected:
        raise ValueError(f"{key} formula mismatch: got {got} expected {expected}")
    _assert_no_clashes(guess)
    return guess


def _assert_no_clashes(guess: MoleculeGuess, min_dist: float = 0.65) -> None:
    xyz = np.array([atom[1] for atom in guess.atoms], dtype=float)
    n = len(xyz)
    for i in range(n):
        d = np.linalg.norm(xyz[i + 1 :] - xyz[i], axis=1)
        if d.size and float(d.min()) < min_dist:
            j = i + 1 + int(np.argmin(d))
            a = guess.atoms[i]
            b = guess.atoms[j]
            raise ValueError(
                f"{guess.ligand_id} clash {a[0]}-{b[0]} = {float(d.min()):.3f} Å "
                f"(min {min_dist} Å)"
            )


def build_sn_oh3_fragment() -> MoleculeGuess:
    atoms = tuple(sn_oh3_atoms())
    return MoleculeGuess(
        ligand_id="sn_oh3",
        smiles="[Sn](O)(O)O",
        name="sn_oh3_ours",
        atoms=atoms,
        provenance="labelled-ours •Sn(OH)₃ / Sn(OH)₃ fragment template; NOT MatGen-Q",
    )


def element_counts_from_atoms(atoms: Iterable[AtomTriple | Sequence[object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in atoms:
        sym = item[0]
        counts[str(sym)] = counts.get(str(sym), 0) + 1
    return dict(sorted(counts.items()))


def atomic_13p5nm_proxy(counts: Mapping[str, int]) -> dict[str, object]:
    """Weighted element-count proxy: sum n_Z * f2_Z(13.5 nm). Not σ(92 eV)."""
    weighted = 0.0
    used: dict[str, float] = {}
    unknown: list[str] = []
    for el, n in sorted(counts.items()):
        f2 = CXRO_F2_13P5NM.get(el)
        if f2 is None:
            unknown.append(el)
            continue
        used[el] = float(f2)
        weighted += int(n) * float(f2)
    return {
        "label": "ours",
        "kind": "cxro_atomic_f2_proxy",
        "not_ab_initio_sigma_92eV": True,
        "citation": CXRO_CITATION,
        "energy_eV": CXRO_E_EV,
        "wavelength_nm": CXRO_WAVELENGTH_NM,
        "f2_weights": dict(CXRO_F2_13P5NM),
        "element_counts": dict(counts),
        "elements_used": used,
        "unknown_elements": unknown,
        "weighted_sum_f2": float(weighted),
        "note": (
            "Sn (and I if present) dominate 13.5 nm absorption in this CXRO f2 "
            "sum; F > O > C > H. Do not report as ab initio σ(92 eV)."
        ),
    }


def parse_ligand_list(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        return list(FIRST_WAVE)
    if isinstance(spec, str):
        tokens = [t.strip().lower() for t in spec.replace(";", ",").split(",") if t.strip()]
    else:
        tokens = [str(t).strip().lower() for t in spec if str(t).strip()]
    aliases = {
        "methyl": "me",
        "ch3": "me",
        "ethyl": "et",
        "n-bu": "nbu",
        "nbutyl": "nbu",
        "n-butyl": "nbu",
        "bu": "nbu",
        "phenyl": "ph",
        "i-pr": "ipr",
        "ipr": "ipr",
        "isopropyl": "ipr",
        "n-pr": "npr",
        "npropyl": "npr",
        "n-propyl": "npr",
        "cf3": "cf3",
        "trifluoromethyl": "cf3",
    }
    out: list[str] = []
    for tok in tokens:
        key = aliases.get(tok, tok)
        if key not in LIGAND_SMILES:
            raise KeyError(f"unknown ligand {tok!r}; known: {', '.join(ALL_LIGAND_IDS)}")
        if key not in out:
            out.append(key)
    return out
