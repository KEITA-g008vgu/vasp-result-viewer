# -*- coding: utf-8 -*-
"""
VASP Result Viewer  for VASP 6.6
-----------------------------------------------------
VASP の計算結果を可視化する。
  ・電荷密度分布      : CHGCAR の 3D 等値面（原子・結合・単位胞と重ねて表示）
  ・DOS / BAND        : vasprun.xml。チェックボックスで PDOS / PBAND を切り替え
  ・フェルミ面        : VASPKIT 263/264 と同じ FermiSurfer 形式 (.frmsf) に変換して表示。
                        元素ごとの寄与をチェックボックスで切り替え
  ・輸送特性          : vaspout.h5（電子フォノン輸送）。ゼーベック係数・電気伝導率・電子熱伝導率の
                        平均と a, b, c 軸方向の値

描画の設定・画面まわり（高 DPI 対応、スクロールパネル、OpenGL 3.3 描画、VESTA の色と半径、
マウス操作）は VASP Input Generator v1.8.8 から引き継いでいる。

依存: numpy, spglib, pillow, matplotlib, h5py, PyOpenGL, pyopengltk (+ 任意で tkinterdnd2)
"""

import os
import sys

# 行列計算ライブラリ（BLAS）の並列は 1 本にする。このアプリでは、重い計算（WAVECAR の軌道分解など）を
# 自分で複数のスレッドに分けるので、BLAS も並列にすると取り合いになって遅くなる（numpy を読む前に設定する）
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

APP_NAME = "VASP Result Viewer"
APP_VERSION = "2.6.8"


# ######################################################################
#  ここから下は VASP Input Generator v1.8.8 から引き継いだ部分
#  （POSCAR 読み込み・BZ（推奨パス・描画）・VESTA の色/半径/結合表・シェーダー・高 DPI・スクロールパネル）
# ######################################################################
ELEMENTS = """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe
Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr""".split()
Z_OF = {el: i + 1 for i, el in enumerate(ELEMENTS)}


# ======================================================================
#  POSCAR 読み込み
# ======================================================================
class Structure:
    def __init__(self, comment, lattice, species, counts, frac):
        self.comment = comment
        self.lattice = np.array(lattice, dtype=float)   # 行ベクトル a1,a2,a3 [Å]
        self.species = species                          # ['Ta','Pd','Se']
        self.counts = counts                            # [4, 2, 12]
        self.frac = np.array(frac, dtype=float)         # 分率座標 (N,3)

    @property
    def symbols(self):
        out = []
        for s, n in zip(self.species, self.counts):
            out += [s] * n
        return out

    @property
    def natoms(self):
        return int(sum(self.counts))

    @property
    def formula(self):
        return "".join(f"{s}{n}" for s, n in zip(self.species, self.counts))

    def spglib_cell(self):
        nums = [Z_OF.get(s, 0) for s in self.symbols]
        # 未知元素名は種ごとに別番号を割り当てる
        if 0 in nums:
            mapping = {s: 200 + i for i, s in enumerate(self.species)}
            nums = [mapping[s] for s in self.symbols]
        return (self.lattice.tolist(), self.frac.tolist(), nums)


def _clean_symbol(tok):
    # "Ta_pv" や "Se/xxxx" のような表記から元素記号だけ取り出す
    tok = tok.split("/")[0].split("_")[0]
    return tok.strip()


def parse_poscar(text):
    lines = [l.rstrip() for l in text.replace("\r", "").split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 8:
        raise ValueError("POSCAR の行数が足りません。")

    comment = lines[0].strip()
    scale_tokens = lines[1].split()
    vecs = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])

    # スケーリング係数（負なら体積指定、3つなら軸ごと）
    if len(scale_tokens) == 3:
        lattice = vecs * np.array([float(x) for x in scale_tokens])
    else:
        s = float(scale_tokens[0])
        if s < 0:
            vol0 = abs(np.linalg.det(vecs))
            lattice = vecs * (abs(s) / vol0) ** (1.0 / 3.0)
        else:
            lattice = vecs * s

    idx = 5
    tokens = lines[idx].split()
    if all(t.lstrip("-").isdigit() for t in tokens):
        # VASP4 形式（元素名行なし）→ コメント行から元素名を推定
        counts = [int(t) for t in tokens]
        guess = [_clean_symbol(t) for t in comment.split()]
        if len(guess) >= len(counts) and all(g in Z_OF for g in guess[:len(counts)]):
            species = guess[:len(counts)]
        else:
            raise ValueError("元素名の行がありません（VASP5 以降の形式にしてください）。")
        idx += 1
    else:
        species = [_clean_symbol(t) for t in tokens]
        counts = [int(t) for t in lines[idx + 1].split()[:len(species)]]
        idx += 2

    if lines[idx].strip()[:1] in "sS":     # Selective dynamics
        idx += 1
    mode = lines[idx].strip()[:1]
    idx += 1
    n = sum(counts)
    coords = []
    for i in range(n):
        if idx + i >= len(lines):
            raise ValueError("原子座標の行数が原子数と一致しません。")
        coords.append([float(x) for x in lines[idx + i].split()[:3]])
    coords = np.array(coords)

    if mode in "cCkK":   # Cartesian
        if len(scale_tokens) == 1 and float(scale_tokens[0]) > 0:
            coords = coords * float(scale_tokens[0])
        frac = coords @ np.linalg.inv(lattice)
    else:
        frac = coords
    return Structure(comment, lattice, species, counts, frac)


def lattice_params(st):
    a = st.lattice
    la = np.linalg.norm(a, axis=1)
    def ang(u, v):
        return np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
    return la, (ang(a[1], a[2]), ang(a[0], a[2]), ang(a[0], a[1])), abs(np.linalg.det(a))


# ======================================================================
#  ブリルアンゾーン（Wigner-Seitz セル）の計算と描画
# ======================================================================
def reciprocal_2pi(lattice):
    """行ベクトル b1,b2,b3 （2π を含む, 1/Å）"""
    return 2.0 * np.pi * np.linalg.inv(np.array(lattice, dtype=float)).T


def brillouin_zone(B, nmax=4, tol=1e-8):
    """逆格子 B（行ベクトル）の第一ブリルアンゾーンを求める。
    戻り値: (頂点 (M,3), 面のリスト [頂点インデックスを外周順に並べたもの])"""
    rng = np.arange(-nmax, nmax + 1)
    n = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    n = n[np.any(n != 0, axis=1)]
    G = n @ B
    G2 = np.einsum("ij,ij->i", G, G)
    order = np.argsort(G2)
    G, G2 = G[order], G2[order]
    scale = np.sqrt(G2[0])

    # Voronoi に関係する G だけを残す（G/2 が他のすべての半空間の内側にあるもの）
    half = 0.5 * G
    lhs = half @ G.T                       # (i, j): (G_i/2)·G_j
    rhs = 0.5 * G2[None, :]
    inside = lhs <= rhs + tol * scale ** 2
    np.fill_diagonal(inside, True)
    rel = np.where(inside.all(axis=1))[0]
    P, P2 = G[rel], G2[rel]

    # 3 平面の交点を全探索し、すべての半空間の内側にあるものを頂点とする
    verts = []
    m = len(P)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                A = np.array([P[i], P[j], P[k]])
                if abs(np.linalg.det(A)) < 1e-10 * scale ** 3:
                    continue
                x = np.linalg.solve(A, 0.5 * np.array([P2[i], P2[j], P2[k]]))
                if np.all(P @ x <= 0.5 * P2 + 1e-7 * scale ** 2):
                    verts.append(x)
    V = []
    for v in verts:
        if not any(np.linalg.norm(v - u) < 1e-6 * scale for u in V):
            V.append(v)
    V = np.array(V)

    faces = []
    for g, g2 in zip(P, P2):
        on = np.where(np.abs(V @ g - 0.5 * g2) < 1e-6 * scale ** 2)[0]
        if len(on) < 3:
            continue
        c = V[on].mean(axis=0)
        nrm = g / np.linalg.norm(g)
        e1 = V[on[0]] - c
        e1 -= nrm * np.dot(e1, nrm)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nrm, e1)
        ang = [np.arctan2(np.dot(V[i] - c, e2), np.dot(V[i] - c, e1)) for i in on]
        faces.append([int(on[i]) for i in np.argsort(ang)])
    return V, faces


def bz_datasets(st):
    """元の構造と標準化構造について、BZ・高対称点・推奨パスをまとめて返す"""
    import seekpath
    import warnings as _w
    _w.filterwarnings("ignore", module="seekpath")
    cell = st.spglib_cell()
    out = []

    r0 = seekpath.get_path_orig_cell(cell, with_time_reversal=True, symprec=1e-3)
    B0 = reciprocal_2pi(st.lattice)
    note0 = "入力はスーパーセル：高対称点はプリミティブセルの BZ 上" if r0.get("is_supercell") else ""
    out.append({"title": "元の構造 (入力 POSCAR)", "B": B0, "A": np.array(st.lattice, float), "res": r0,
                "note": note0})

    r1 = seekpath.get_path(cell, with_time_reversal=True, symprec=1e-3)
    B1 = reciprocal_2pi(r1["primitive_lattice"])
    out.append({"title": "標準構造 (seekpath 標準化プリミティブ)", "B": B1,
                "A": np.array(r1["primitive_lattice"], float), "res": r1, "note": ""})

    for d in out:
        d["V"], d["faces"] = brillouin_zone(d["B"])
        d["cart"] = {lb: np.array(c) @ d["B"] for lb, c in d["res"]["point_coords"].items()}
        d["label"] = (f"{d['res']['bravais_lattice_extended']}  "
                      f"{d['res']['spacegroup_international']} (#{d['res']['spacegroup_number']})")
    return out


_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def kpt_label(lb):
    """seekpath のラベルを表示用に（GAMMA → Γ, X_2 → X₂）"""
    if lb.upper() == "GAMMA":
        return "Γ"
    if "_" in lb:
        a, b = lb.split("_", 1)
        return a + b.translate(_SUB)
    return lb


def label_parts(lb):
    """表示用ラベルを (本体, 下付き) に分ける（GAMMA → ("Γ", ""), X_2 → ("X", "2"), b1 → ("b", "1")）。
    下付き文字は Unicode の ₂ などを使うとフォントによって潰れるので、小さい文字で別に描く"""
    if lb.upper() == "GAMMA":
        return "Γ", ""
    if "_" in lb:
        a, b = lb.split("_", 1)
        return a, b
    if len(lb) >= 2 and lb[0].isalpha() and lb[1:].isdigit():
        return lb[0], lb[1:]
    return lb, ""


def bz_edges(faces):
    """面のリストから辺（頂点対）と、その辺を共有する面の番号を求める"""
    em = {}
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            key = (min(a, b), max(a, b))
            em.setdefault(key, []).append(fi)
    return em


BZ_AXIS_STYLE = {"recip": ("#009600", None), "abc": ("#e07000", None), "xyz": ("#555555", (6, 4))}
# 矢印の長さ（BZ の大きさに対する倍率）。高対称点のラベルや互いのラベルと重ならないよう段違いにする
BZ_AXIS_LEN = {"recip": 1.35, "abc": 1.65, "xyz": 1.95}


def bz_primitives(d, R, W, H, scale, pan=(0.0, 0.0), show=("recip",)):
    """ブリルアンゾーンの描画要素（画面座標）を作る。
    手前を向いた面は塗らず実線、奥の面は薄く塗って辺を破線（教科書風の表示）。
    戻り値: list of (kind, data)。描く順に並んでいる"""
    V, faces, B = d["V"], d["faces"], d["B"]
    P = V @ R.T
    cx, cy = pan
    sx = lambda p: (W / 2 + (p[0] - cx) * scale, H / 2 - (p[1] - cy) * scale)
    face_front = []
    for f in faces:
        n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
        if np.dot(n, V[f].mean(axis=0)) < 0:      # 外向きにそろえる
            n = -n
        face_front.append((n @ R.T)[2] > 0)
    out = []
    # 1) 奥の面（薄い塗り）
    for f, fr in zip(faces, face_front):
        if not fr:
            out.append(("poly", [sx(P[i]) for i in f]))
    em = bz_edges(faces)
    # 2) 奥の辺（破線）
    for (a, b), fl in em.items():
        if not any(face_front[k] for k in fl):
            out.append(("edge_back", (sx(P[a]), sx(P[b]))))
    # 3) 軸（向きのみ。長さはそろえる）: 逆格子 b1-b3、実空間 a,b,c、直交座標 x,y,z
    R0 = np.abs(V).max()
    axes = []
    if "recip" in show:
        axes += [("recip", b, f"b{i + 1}", BZ_AXIS_LEN["recip"]) for i, b in enumerate(B)]
    if "abc" in show:
        axes += [("abc", a, name, BZ_AXIS_LEN["abc"]) for a, name in zip(d["A"], "abc")]
    if "xyz" in show:
        axes += [("xyz", e, name, BZ_AXIS_LEN["xyz"]) for e, name in zip(np.eye(3), "xyz")]
    for kind, v, name, ln in axes:
        pv = (v / np.linalg.norm(v) * R0 * ln) @ R.T
        out.append(("arrow", (sx(np.zeros(3)), sx(pv), name, kind)))
    # 4) 推奨パスと高対称点（赤）
    path = d["res"]["path"]
    for a, b in path:
        out.append(("path", (sx(d["cart"][a] @ R.T), sx(d["cart"][b] @ R.T))))
    used = sorted({lb for seg in path for lb in seg})
    for lb in used:
        out.append(("kpt", (sx(d["cart"][lb] @ R.T), lb)))
    # 5) 手前の辺（実線）
    for (a, b), fl in em.items():
        if any(face_front[k] for k in fl):
            out.append(("edge_front", (sx(P[a]), sx(P[b]))))
    return out


def bz_fit_scale(d, W, H, show=("recip",)):
    """表示する矢印の先端まで収まる倍率"""
    f = max([1.25] + [BZ_AXIS_LEN[k] for k in show]) + 0.15
    r = np.abs(d["V"]).max() * f
    return min(W, H) / (2 * r) * 0.95


def render_bz_pil(prims, W, H, title_lines, font_px=16, ss=2):
    """画像保存用: 描画要素を PIL 画像にする。
    タイトルと推奨パスの文字列は図の上の専用の帯に書き、図（BZ・軸・高対称点）と重ならないようにする"""
    from PIL import Image, ImageDraw, ImageFont
    head = int(10 + font_px * 1.5 * len(title_lines) + 8)      # 見出し帯の高さ [px]
    img = Image.new("RGB", (W * ss, (H + head) * ss), "white")
    dr = ImageDraw.Draw(img)

    def font(sz, names):
        for fp in names:
            try:
                return ImageFont.truetype(fp, int(sz * ss))
            except Exception:
                continue
        return ImageFont.load_default()
    # ラベル（Γ や下付き数字を含む）は Segoe UI、タイトル（日本語）はメイリオ等
    lab_fonts = ("segoeuib.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
    jp_fonts = ("meiryo.ttc", "YuGothM.ttc", "msgothic.ttc", "NotoSansCJK-Regular.ttc", "DejaVuSans.ttf")
    f = font(font_px, jp_fonts)
    fb = font(font_px * 1.1, lab_fonts)
    ft = font(font_px * 1.1, jp_fonts)
    S = lambda p: (p[0] * ss, (p[1] + head) * ss)      # 図は見出し帯の下に描く
    fsub = font(font_px * 0.8, lab_fonts)

    def text_sub(xy, name, fill, stroke=False):
        base, sub = label_parts(name)
        kw = {"stroke_width": 2 * ss, "stroke_fill": "white"} if stroke else {}
        dr.text(S(xy), base, fill=fill, font=fb, anchor="ls", **kw)
        if sub:
            w = dr.textlength(base, font=fb)
            dr.text((S(xy)[0] + w + ss, S(xy)[1] + 4 * ss), sub, fill=fill, font=fsub, anchor="ls", **kw)
    for kind, dd in prims:
        if kind == "poly":
            dr.polygon([S(p) for p in dd], fill=(221, 232, 243))
        elif kind == "edge_back":
            p0, p1 = np.array(S(dd[0])), np.array(S(dd[1]))
            L = np.linalg.norm(p1 - p0)
            n = max(1, int(L / (8 * ss)))
            for k in range(0, n, 2):
                a = p0 + (p1 - p0) * k / n
                b = p0 + (p1 - p0) * min(k + 1, n) / n
                dr.line([tuple(a), tuple(b)], fill=(140, 150, 165), width=ss)
        elif kind == "edge_front":
            dr.line([S(dd[0]), S(dd[1])], fill=(60, 80, 110), width=2 * ss)
        elif kind == "arrow":
            p0, p1, name, ak = dd
            hexc = BZ_AXIS_STYLE[ak][0]
            rgb = tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5))
            if BZ_AXIS_STYLE[ak][1]:
                q0, q1 = np.array(S(p0)), np.array(S(p1))
                Lq = np.linalg.norm(q1 - q0)
                nq = max(1, int(Lq / (10 * ss)))
                for k in range(0, nq, 2):
                    dr.line([tuple(q0 + (q1 - q0) * k / nq), tuple(q0 + (q1 - q0) * min(k + 1, nq) / nq)],
                            fill=rgb, width=2 * ss)
            else:
                dr.line([S(p0), S(p1)], fill=rgb, width=2 * ss)
            v = np.array(p1) - np.array(p0)
            L = np.linalg.norm(v)
            if L > 1:
                u = v / L
                nrm = np.array([-u[1], u[0]])
                tip = np.array(p1)
                pts = [tip, tip - u * 12 + nrm * 5, tip - u * 12 - nrm * 5]
                dr.polygon([S(p) for p in pts], fill=rgb)
            text_sub((p1[0] + 8, p1[1] - 4), name, rgb)
        elif kind == "path":
            dr.line([S(dd[0]), S(dd[1])], fill=(214, 39, 40), width=3 * ss)
        elif kind == "kpt":
            (x, y), name = dd
            r = 4.5
            dr.ellipse([S((x - r, y - r)), S((x + r, y + r))], fill=(214, 39, 40))
            text_sub((x + 7, y - 4), name, "black", stroke=True)
    y = 8
    for k, t in enumerate(title_lines):
        dr.text((10 * ss, y * ss), t, fill="black", font=ft if k == 0 else (f if k < len(title_lines) - 1 else fb))
        y += font_px * 1.5
    dr.line([(0, (head - 3) * ss), (W * ss, (head - 3) * ss)], fill=(210, 210, 210), width=ss)
    return img.resize((W, H + head), Image.LANCZOS)


# ======================================================================
#  結晶構造の表示
# ======================================================================
# 共有結合半径 [Å]（Cordero et al., Dalton Trans. 2008, DOI:10.1039/b801115j）と
# 元素色（Jmol 配色）。値は ASE (ase.data) と同じ。
ATOM_STYLE = {
    "H": (0.31, "#ffffff"), "He": (0.28, "#d9ffff"), "Li": (1.28, "#cc80ff"),
    "Be": (0.96, "#c2ff00"), "B": (0.84, "#ffb5b5"), "C": (0.76, "#909090"),
    "N": (0.71, "#3050f8"), "O": (0.66, "#ff0d0d"), "F": (0.57, "#90e050"),
    "Ne": (0.58, "#b3e3f5"), "Na": (1.66, "#ab5cf2"), "Mg": (1.41, "#8aff00"),
    "Al": (1.21, "#bfa6a6"), "Si": (1.11, "#f0c8a0"), "P": (1.07, "#ff8000"),
    "S": (1.05, "#ffff30"), "Cl": (1.02, "#1ff01f"), "Ar": (1.06, "#80d1e3"),
    "K": (2.03, "#8f40d4"), "Ca": (1.76, "#3dff00"), "Sc": (1.70, "#e6e6e6"),
    "Ti": (1.60, "#bfc2c7"), "V": (1.53, "#a6a6ab"), "Cr": (1.39, "#8a99c7"),
    "Mn": (1.39, "#9c7ac7"), "Fe": (1.32, "#e06633"), "Co": (1.26, "#f090a0"),
    "Ni": (1.24, "#50d050"), "Cu": (1.32, "#c88033"), "Zn": (1.22, "#7d80b0"),
    "Ga": (1.22, "#c28f8f"), "Ge": (1.20, "#668f8f"), "As": (1.19, "#bd80e3"),
    "Se": (1.20, "#ffa100"), "Br": (1.20, "#a62929"), "Kr": (1.16, "#5cb8d1"),
    "Rb": (2.20, "#702eb0"), "Sr": (1.95, "#00ff00"), "Y": (1.90, "#94ffff"),
    "Zr": (1.75, "#94e0e0"), "Nb": (1.64, "#73c2c9"), "Mo": (1.54, "#54b5b5"),
    "Tc": (1.47, "#3b9e9e"), "Ru": (1.46, "#248f8f"), "Rh": (1.42, "#0a7d8c"),
    "Pd": (1.39, "#006985"), "Ag": (1.45, "#c0c0c0"), "Cd": (1.44, "#ffd98f"),
    "In": (1.42, "#a67573"), "Sn": (1.39, "#668080"), "Sb": (1.39, "#9e63b5"),
    "Te": (1.38, "#d47a00"), "I": (1.39, "#940094"), "Xe": (1.40, "#429eb0"),
    "Cs": (2.44, "#57178f"), "Ba": (2.15, "#00c900"), "La": (2.07, "#70d4ff"),
    "Ce": (2.04, "#ffffc7"), "Pr": (2.03, "#d9ffc7"), "Nd": (2.01, "#c7ffc7"),
    "Pm": (1.99, "#a3ffc7"), "Sm": (1.98, "#8fffc7"), "Eu": (1.98, "#61ffc7"),
    "Gd": (1.96, "#45ffc7"), "Tb": (1.94, "#30ffc7"), "Dy": (1.92, "#1fffc7"),
    "Ho": (1.92, "#00ff9c"), "Er": (1.89, "#00e675"), "Tm": (1.90, "#00d452"),
    "Yb": (1.87, "#00bf38"), "Lu": (1.87, "#00ab24"), "Hf": (1.75, "#4dc2ff"),
    "Ta": (1.70, "#4da6ff"), "W": (1.62, "#2194d6"), "Re": (1.51, "#267dab"),
    "Os": (1.44, "#266696"), "Ir": (1.41, "#175487"), "Pt": (1.36, "#d0d0e0"),
    "Au": (1.36, "#ffd123"), "Hg": (1.32, "#b8b8d0"), "Tl": (1.45, "#a6544d"),
    "Pb": (1.46, "#575961"), "Bi": (1.48, "#9e4fb5"), "Po": (1.40, "#ab5c00"),
    "At": (1.50, "#754f45"), "Rn": (1.50, "#428296"), "Fr": (2.60, "#420066"),
    "Ra": (2.21, "#007d00"), "Ac": (2.15, "#70abfa"), "Th": (2.06, "#00baff"),
    "Pa": (2.00, "#00a1ff"), "U": (1.96, "#008fff"), "Np": (1.90, "#0080ff"),
    "Pu": (1.87, "#006bff"), "Am": (1.80, "#545cf2"), "Cm": (1.69, "#785ce3"),
    "Bk": (2.00, "#8a4fe3"), "Cf": (2.00, "#a136d4"), "Es": (2.00, "#b31fd4"),
    "Fm": (2.00, "#b31fba"), "Md": (2.00, "#b30da6"), "No": (2.00, "#bd0d87"),
    "Lr": (2.00, "#c70066"),
}

def build_supercell_atoms(st, n, packed=False, tol=1e-3):
    """表示用の原子リストを作る。
    戻り値: (デカルト座標 (N,3), 元素記号リスト, サイトラベルリスト)
    packed=True なら、境界（分率座標 = 0）上の原子を反対側の面にも複製する（VESTA の境界表示に相当）"""
    n = [max(1, int(x)) for x in n]
    labels_site = []
    cnt = {}
    for s in st.symbols:
        cnt[s] = cnt.get(s, 0) + 1
        labels_site.append(f"{s}{cnt[s]}")
    frac0 = np.mod(st.frac, 1.0)
    frac0[np.abs(frac0 - 1.0) < tol] = 0.0
    fr, sy, lb = [], [], []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                shift = np.array([i, j, k], dtype=float)
                for p, s, l in zip(frac0, st.symbols, labels_site):
                    fr.append(p + shift)
                    sy.append(s)
                    lb.append(l)
    fr = np.array(fr)
    if packed:
        extra_f, extra_s, extra_l = [], [], []
        for p, s, l in zip(fr, sy, lb):
            # 0 の面にある座標を N の面にも複製（全組み合わせ）
            opts = [[c] + ([c + n[d]] if abs(c) < tol else []) for d, c in enumerate(p)]
            for a in opts[0]:
                for b in opts[1]:
                    for c in opts[2]:
                        q = np.array([a, b, c])
                        if np.any(np.abs(q - p) > 0.5):
                            extra_f.append(q)
                            extra_s.append(s)
                            extra_l.append(l)
        if extra_f:
            fr = np.vstack([fr, np.array(extra_f)])
            sy += extra_s
            lb += extra_l
    cart = fr @ st.lattice
    return cart, sy, lb


# VESTA の既定の原子半径・色（VESTA 付属 elements.ini の値: 番号, 記号, 原子半径, vdW半径, イオン半径, R, G, B）
VESTA_ELEMENTS_INI = """  1  H  0.46  1.20  0.200    1.00000    0.80000    0.80000
  2 He  1.22  1.40  1.220    0.98907    0.91312    0.81091
  3 Li  1.57  1.40  0.590    0.52731    0.87953    0.45670
  4 Be  1.12  1.40  0.270    0.37147    0.84590    0.48292
  5  B  0.81  1.40  0.110    0.12490    0.63612    0.05948
  6  C  0.77  1.70  0.150    0.50430    0.28659    0.16236
  7  N  0.74  1.55  1.460    0.69139    0.72934    0.90280
  8  O  0.74  1.52  1.400    0.99997    0.01328    0.00000
  9  F  0.72  1.47  1.330    0.69139    0.72934    0.90280
 10 Ne  1.60  1.54  1.600    0.99954    0.21788    0.71035
 11 Na  1.91  1.54  1.020    0.97955    0.86618    0.23787
 12 Mg  1.60  1.54  0.720    0.98773    0.48452    0.08470
 13 Al  1.43  1.54  0.390    0.50718    0.70056    0.84062
 14 Si  1.18  2.10  0.260    0.10596    0.23226    0.98096
 15  P  1.10  1.80  0.170    0.75557    0.61256    0.76425
 16  S  1.04  1.80  1.840    1.00000    0.98071    0.00000
 17 Cl  0.99  1.75  1.810    0.19583    0.98828    0.01167
 18 Ar  1.92  1.88  1.920    0.81349    0.99731    0.77075
 19  K  2.35  1.88  1.510    0.63255    0.13281    0.96858
 20 Ca  1.97  1.88  1.120    0.35642    0.58863    0.74498
 21 Sc  1.64  1.88  0.745    0.71209    0.38930    0.67279
 22 Ti  1.47  1.88  0.605    0.47237    0.79393    1.00000
 23  V  1.35  1.88  0.580    0.90000    0.10000    0.00000
 24 Cr  1.29  1.88  0.615    0.00000    0.00000    0.62000
 25 Mn  1.37  1.88  0.830    0.66148    0.03412    0.62036
 26 Fe  1.26  1.88  0.780    0.71051    0.44662    0.00136
 27 Co  1.25  1.88  0.745    0.00000    0.00000    0.68666
 28 Ni  1.25  1.88  0.690    0.72032    0.73631    0.74339
 29 Cu  1.28  1.88  0.730    0.13390    0.28022    0.86606
 30 Zn  1.37  1.88  0.740    0.56123    0.56445    0.50799
 31 Ga  1.53  1.88  0.620    0.62292    0.89293    0.45486
 32 Ge  1.22  1.88  0.530    0.49557    0.43499    0.65193
 33 As  1.21  1.85  0.335    0.45814    0.81694    0.34249
 34 Se  1.04  1.90  1.980    0.60420    0.93874    0.06122
 35 Br  1.14  1.85  1.960    0.49645    0.19333    0.01076
 36 Kr  1.98  2.02  1.980    0.98102    0.75805    0.95413
 37 Rb  2.50  2.02  1.610    1.00000    0.00000    0.60000
 38 Sr  2.15  2.02  1.260    0.00000    1.00000    0.15259
 39  Y  1.82  2.02  1.019    0.40259    0.59739    0.55813
 40 Zr  1.60  2.02  0.720    0.00000    1.00000    0.00000
 41 Nb  1.47  2.02  0.640    0.29992    0.70007    0.46459
 42 Mo  1.40  2.02  0.590    0.70584    0.52602    0.68925
 43 Tc  1.35  2.02  0.560    0.80574    0.68699    0.79478
 44 Ru  1.34  2.02  0.620    0.81184    0.72113    0.68089
 45 Rh  1.34  2.02  0.665    0.80748    0.82205    0.67068
 46 Pd  1.37  2.02  0.860    0.75978    0.76818    0.72454
 47 Ag  1.44  2.02  1.150    0.72032    0.73631    0.74339
 48 Cd  1.52  2.02  0.950    0.95145    0.12102    0.86354
 49 In  1.67  2.02  0.800    0.84378    0.50401    0.73483
 50 Sn  1.58  2.02  0.690    0.60764    0.56052    0.72926
 51 Sb  1.41  2.00  0.760    0.84627    0.51498    0.31315
 52 Te  1.37  2.06  2.210    0.67958    0.63586    0.32038
 53  I  1.33  1.98  2.200    0.55914    0.12200    0.54453
 54 Xe  2.18  2.16  0.480    0.60662    0.63218    0.97305
 55 Cs  2.72  2.16  1.740    0.05872    0.99922    0.72578
 56 Ba  2.24  2.16  1.420    0.11835    0.93959    0.17565
 57 La  1.88  2.16  1.160    0.35340    0.77057    0.28737
 58 Ce  1.82  2.16  0.970    0.82055    0.99071    0.02374
 59 Pr  1.82  2.16  1.126    0.99130    0.88559    0.02315
 60 Nd  1.82  2.16  1.109    0.98701    0.55560    0.02744
 61 Pm  1.81  2.16  1.093    0.00000    0.00000    0.96000
 62 Sm  1.81  2.16  1.270    0.99042    0.02403    0.49195
 63 Eu  2.06  2.16  1.066    0.98367    0.03078    0.83615
 64 Gd  1.79  2.16  1.053    0.75325    0.01445    1.00000
 65 Tb  1.77  2.16  1.040    0.44315    0.01663    0.99782
 66 Dy  1.77  2.16  1.027    0.19390    0.02374    0.99071
 67 Ho  1.76  2.16  1.015    0.02837    0.25876    0.98608
 68 Er  1.75  2.16  1.004    0.28688    0.45071    0.23043
 69 Tm  1.00  2.16  0.994    0.00000    0.00000    0.88000
 70 Yb  1.94  2.16  0.985    0.15323    0.99165    0.95836
 71 Lu  1.72  2.16  0.977    0.15097    0.99391    0.71032
 72 Hf  1.59  2.16  0.710    0.70704    0.70552    0.35090
 73 Ta  1.47  2.16  0.640    0.71952    0.60694    0.33841
 74  W  1.41  2.16  0.600    0.55616    0.54257    0.50178
 75 Re  1.37  2.16  0.530    0.70294    0.69401    0.55789
 76 Os  1.35  2.16  0.630    0.78703    0.69512    0.47379
 77 Ir  1.36  2.16  0.625    0.78975    0.81033    0.45049
 78 Pt  1.39  2.16  0.625    0.79997    0.77511    0.75068
 79 Au  1.44  2.16  1.370    0.99628    0.70149    0.22106
 80 Hg  1.55  2.16  1.020    0.82940    0.72125    0.79823
 81 Tl  1.71  2.16  0.885    0.58798    0.53854    0.42649
 82 Pb  1.75  2.16  1.190    0.32386    0.32592    0.35729
 83 Bi  1.82  2.16  1.030    0.82428    0.18732    0.97211
 84 Po  1.77  2.16  0.940    0.00000    0.00000    1.00000
 85 At  0.62  2.16  0.620    0.00000    0.00000    1.00000
 86 Rn  0.80  2.16  0.800    1.00000    1.00000    0.00000
 87 Fr  1.00  2.16  1.800    0.00000    0.00000    0.00000
 88 Ra  2.35  2.16  1.480    0.42959    0.66659    0.34786
 89 Ac  2.03  2.16  1.120    0.39344    0.62101    0.45034
 90 Th  1.80  2.16  1.050    0.14893    0.99596    0.47106
 91 Pa  1.63  2.16  0.780    0.16101    0.98387    0.20855
 92  U  1.56  2.16  0.730    0.47774    0.63362    0.66714
 93 Np  1.56  2.16  0.750    0.30000    0.30000    0.30000
 94 Pu  1.64  2.16  0.860    0.30000    0.30000    0.30000
 95 Am  1.73  2.16  0.975    0.30000    0.30000    0.30000
"""

VESTA_ELEMENTS = {}
for _l in VESTA_ELEMENTS_INI.strip().splitlines():
    _t = _l.split()
    VESTA_ELEMENTS[_t[1]] = (float(_t[2]), (float(_t[5]), float(_t[6]), float(_t[7])))

# VESTA の既定の結合距離上限 [Å]（元素対ごと。pymatgen の vesta_cutoffs.yaml の値。
# 同じ元素対に向きの違う 2 つの値がある場合（H–O など）は、小さい方を採用）
VESTA_BOND_CUTOFFS = {
    ("Ac","Br"):3.22726, ("Ac","Cl"):3.08646, ("Ac","F"):2.58646, ("Ac","O"):2.7326,
    ("Ag","As"):2.45642, ("Ag","Br"):2.37642, ("Ag","Cl"):3.05939, ("Ag","F"):2.76939,
    ("Ag","H"):1.65642, ("Ag","I"):2.53642, ("Ag","N"):2.00642, ("Ag","O"):2.81139,
    ("Ag","P"):2.37642, ("Ag","S"):3.08839, ("Ag","Se"):2.41642, ("Ag","Te"):2.66642,
    ("Al","As"):2.75646, ("Al","Br"):2.65646, ("Al","Cl"):2.48846, ("Al","F"):2.00146,
    ("Al","H"):1.90646, ("Al","I"):2.86646, ("Al","N"):2.24646, ("Al","O"):2.1074,
    ("Al","P"):2.69646, ("Al","S"):2.66646, ("Al","Se"):2.72646, ("Al","Te"):2.93646,
    ("Am","Br"):3.22944, ("Am","Cl"):3.08945, ("Am","F"):2.61945, ("Am","O"):2.71649,
    ("As","Au"):2.26998, ("As","B"):2.42646, ("As","Ba"):3.69, ("As","Be"):2.60649,
    ("As","Bi"):2.87642, ("As","Br"):2.80646, ("As","C"):2.38646, ("As","Ca"):3.48295,
    ("As","Cd"):3.29295, ("As","Ce"):3.08644, ("As","Cl"):2.61646, ("As","Co"):2.43642,
    ("As","Cr"):2.49642, ("As","Cs"):4.15942, ("As","Cu"):2.71895, ("As","Dy"):3.14,
    ("As","Er"):3.18, ("As","Eu"):2.93898, ("As","F"):2.15646, ("As","Fe"):2.50642,
    ("As","Ga"):2.38998, ("As","Gd"):3.18, ("As","Ge"):2.47998, ("As","Hf"):2.71642,
    ("As","Hg"):2.65642, ("As","Ho"):2.87898, ("As","I"):3.03646, ("As","In"):2.66642,
    ("As","Ir"):2.58998, ("As","K"):3.94942, ("As","La"):3.51293, ("As","Lu"):3.19649,
    ("As","Mg"):3.09293, ("As","Mn"):2.51642, ("As","Mo"):2.62701, ("As","Na"):3.64942,
    ("As","Nb"):2.69642, ("As","Ni"):2.28998, ("As","O"):2.24546, ("As","P"):2.29998,
    ("As","Pb"):3.02644, ("As","Pd"):2.34998, ("As","Pr"):3.35649, ("As","Pt"):2.30998,
    ("As","Rb"):4.04645, ("As","Re"):2.61998, ("As","Rh"):2.41998, ("As","Ru"):2.40998,
    ("As","S"):2.84649, ("As","Sb"):2.64998, ("As","Sc"):3.04291, ("As","Se"):2.98649,
    ("As","Si"):2.66002, ("As","Sm"):3.30649, ("As","Sn"):2.77642, ("As","Sr"):3.62295,
    ("As","Ta"):2.70642, ("As","Tb"):3.26649, ("As","Te"):3.10646, ("As","Th"):3.40649,
    ("As","Ti"):2.87642, ("As","Tl"):3.09644, ("As","Tm"):3.22649, ("As","U"):3.02644,
    ("As","V"):2.54642, ("As","W"):2.58998, ("As","Y"):3.24649, ("As","Yb"):3.19649,
    ("As","Zn"):2.95293, ("As","Zr"):3.07004, ("Au","Br"):2.77646, ("Au","Cl"):2.88295,
    ("Au","F"):2.34646, ("Au","H"):1.41998, ("Au","I"):3.21295, ("Au","N"):2.3826,
    ("Au","O"):2.34646, ("Au","P"):2.18998, ("Au","S"):2.8326, ("Au","Se"):2.22998,
    ("Au","Te"):2.45998, ("B","B"):1.85846, ("B","Br"):2.33646, ("B","Cl"):2.19646,
    ("B","F"):1.76646, ("B","H"):1.59646, ("B","I"):2.55646, ("B","N"):1.93846, ("B","O"):1.82746,
    ("B","P"):2.37646, ("B","S"):2.27646, ("B","Se"):2.40646, ("B","Te"):2.65646,
    ("Ba","Br"):3.74295, ("Ba","Cl"):3.55295, ("Ba","F"):3.05095, ("Ba","H"):3.08295,
    ("Ba","I"):3.99295, ("Ba","N"):3.33295, ("Ba","O"):3.14795, ("Ba","P"):3.48295,
    ("Ba","S"):3.66195, ("Ba","Se"):3.74295, ("Ba","Te"):3.94295, ("Be","Br"):2.50649,
    ("Be","Cl"):2.36649, ("Be","F"):1.88749, ("Be","H"):1.71649, ("Be","I"):2.70649,
    ("Be","N"):2.10649, ("Be","O"):1.98749, ("Be","P"):2.55649, ("Be","S"):2.43649,
    ("Be","Se"):2.57649, ("Be","Te"):2.81649, ("Bi","Br"):3.17993, ("Bi","Cl"):3.04291,
    ("Bi","F"):2.55291, ("Bi","H"):2.12642, ("Bi","I"):3.38291, ("Bi","N"):2.56329,
    ("Bi","O"):2.6608, ("Bi","P"):2.78642, ("Bi","S"):3.13291, ("Bi","Se"):3.24329,
    ("Bi","Te"):3.02642, ("Bk","Br"):3.15233, ("Bk","Cl"):3.02291, ("Bk","F"):2.54233,
    ("Bk","O"):2.64329, ("Br","C"):2.26002, ("Br","Ca"):3.36995, ("Br","Cd"):3.21295,
    ("Br","Ce"):3.40452, ("Br","Cf"):3.14233, ("Br","Cl"):2.33296, ("Br","Co"):2.33642,
    ("Br","Cr"):2.97293, ("Br","Cs"):4.06942, ("Br","Cu"):2.89295, ("Br","Dy"):3.16945,
    ("Br","Er"):3.14945, ("Br","Eu"):3.46549, ("Br","F"):2.20646, ("Br","Fe"):2.8952,
    ("Br","Ga"):2.6426, ("Br","Gd"):3.19945, ("Br","Ge"):2.34998, ("Br","Hf"):2.62642,
    ("Br","Hg"):3.09293, ("Br","Ho"):3.20159, ("Br","In"):3.05329, ("Br","Ir"):2.49998,
    ("Br","K"):3.8513, ("Br","La"):3.43293, ("Br","Li"):3.11654, ("Br","Lu"):3.11945,
    ("Br","Mg"):2.99293, ("Br","Mn"):3.05293, ("Br","Mo"):2.8535, ("Br","Na"):3.57715,
    ("Br","Nb"):3.07646, ("Br","Nd"):3.37293, ("Br","Ni"):2.80649, ("Br","Np"):3.21233,
    ("Br","O"):2.35646, ("Br","Os"):2.72002, ("Br","P"):2.44293, ("Br","Pa"):3.18437,
    ("Br","Pb"):3.62451, ("Br","Pd"):2.80649, ("Br","Pm"):3.18233, ("Br","Pr"):3.27649,
    ("Br","Pt"):2.94191, ("Br","Pu"):3.19233, ("Br","Rb"):4.05498, ("Br","Re"):2.70002,
    ("Br","Rh"):2.7126, ("Br","Ru"):2.30998, ("Br","S"):2.21998, ("Br","Sb"):2.96646,
    ("Br","Sc"):2.94291, ("Br","Se"):2.78002, ("Br","Si"):2.55002, ("Br","Sm"):3.26649,
    ("Br","Sn"):3.2152, ("Br","Sr"):3.54295, ("Br","Ta"):2.60642, ("Br","Tb"):3.18649,
    ("Br","Te"):2.90002, ("Br","Th"):3.31945, ("Br","Ti"):3.20293, ("Br","Tl"):3.80942,
    ("Br","Tm"):3.13945, ("Br","U"):3.39452, ("Br","V"):2.87329, ("Br","W"):2.49998,
    ("Br","Y"):3.15649, ("Br","Yb"):3.12945, ("Br","Zn"):2.86293, ("Br","Zr"):2.98004,
    ("C","C"):1.89002, ("C","Cl"):2.11002, ("C","Co"):2.19691, ("C","Cu"):2.32649,
    ("C","F"):1.76002, ("C","Fe"):2.25191, ("C","H"):1.2, ("C","I"):2.16998, ("C","N"):1.79202,
    ("C","O"):1.97249, ("C","P"):1.93998, ("C","Pd"):2.33649, ("C","Pt"):2.36649,
    ("C","S"):2.15002, ("C","Se"):2.01998, ("C","Si"):2.23302, ("C","Te"):2.25998,
    ("Ca","Cl"):3.23295, ("Ca","F"):2.70495, ("Ca","H"):2.69295, ("Ca","I"):3.58295,
    ("Ca","N"):3.00295, ("Ca","O"):2.83062, ("Ca","P"):3.41295, ("Ca","S"):3.31295,
    ("Ca","Se"):3.42295, ("Ca","Te"):3.62295, ("Cd","Cl"):3.09295, ("Cd","F"):2.67395,
    ("Cd","H"):2.52295, ("Cd","I"):3.46295, ("Cd","N"):2.82295, ("Cd","O"):2.76695,
    ("Cd","P"):3.20295, ("Cd","S"):3.16695, ("Cd","Se"):3.26295, ("Cd","Te"):3.45295,
    ("Ce","Cl"):3.25293, ("Ce","F"):2.75452, ("Ce","H"):2.34644, ("Ce","I"):3.62452,
    ("Ce","N"):2.78549, ("Ce","O"):2.86393, ("Ce","P"):3.00644, ("Ce","S"):3.36293,
    ("Ce","Se"):3.04644, ("Ce","Te"):3.22644, ("Cf","Cl"):3.01291, ("Cf","F"):2.53233,
    ("Cf","O"):2.63291, ("Cl","Cl"):2.14296, ("Cl","Cm"):3.18291, ("Cl","Co"):2.74593,
    ("Cl","Cr"):2.80293, ("Cl","Cs"):3.91042, ("Cl","Cu"):2.75295, ("Cl","D"):1.5,
    ("Cl","Dy"):3.01945, ("Cl","Er"):2.99944, ("Cl","Eu"):3.32549, ("Cl","F"):2.14646,
    ("Cl","Fe"):2.86293, ("Cl","Ga"):2.52646, ("Cl","Gd"):3.06349, ("Cl","Ge"):2.49002,
    ("Cl","H"):1.5, ("Cl","Hf"):2.75646, ("Cl","Hg"):3.24939, ("Cl","Ho"):3.05159,
    ("Cl","I"):3.33295, ("Cl","In"):3.52939, ("Cl","Ir"):2.56746, ("Cl","K"):3.65976,
    ("Cl","La"):3.33452, ("Cl","Li"):2.91814, ("Cl","Lu"):2.96944, ("Cl","Mg"):2.79293,
    ("Cl","Mn"):2.84593, ("Cl","Mo"):2.80447, ("Cl","N"):2.20646, ("Cl","NH"):3.48195,
    ("Cl","Na"):3.39412, ("Cl","Nb"):2.76291, ("Cl","Nd"):3.22493, ("Cl","Ni"):2.62649,
    ("Cl","Np"):3.07233, ("Cl","O"):2.16646, ("Cl","Os"):2.54002, ("Cl","P"):2.28746,
    ("Cl","Pa"):3.01437, ("Cl","Pb"):3.39295, ("Cl","Pd"):2.65649, ("Cl","Pm"):3.41233,
    ("Cl","Pr"):3.12749, ("Cl","Pt"):2.75646, ("Cl","Pu"):3.05233, ("Cl","Rb"):3.86664,
    ("Cl","Re"):3.44712, ("Cl","Rh"):2.62646, ("Cl","Ru"):2.70646, ("Cl","S"):2.37002,
    ("Cl","Sb"):2.80646, ("Cl","Sc"):2.92291, ("Cl","Se"):2.57002, ("Cl","Si"):2.38002,
    ("Cl","Sm"):3.08749, ("Cl","Sn"):3.13111, ("Cl","Sr"):3.37295, ("Cl","Ta"):2.6739,
    ("Cl","Tb"):3.04349, ("Cl","Tc"):2.56002, ("Cl","Te"):2.73906, ("Cl","Th"):3.15945,
    ("Cl","Ti"):3.02293, ("Cl","Tl"):3.72942, ("Cl","Tm"):2.98944, ("Cl","U"):3.24452,
    ("Cl","V"):3.15293, ("Cl","W"):2.47, ("Cl","Y"):3.00649, ("Cl","Yb"):2.98249,
    ("Cl","Zn"):2.72293, ("Cl","Zr"):3.33651, ("Cm","F"):2.68291, ("Cm","O"):2.79291,
    ("Co","F"):2.35293, ("Co","H"):1.9278, ("Co","I"):2.52878, ("Co","In"):3.13629,
    ("Co","N"):2.36293, ("Co","O"):2.40493, ("Co","P"):2.36642, ("Co","S"):2.65293,
    ("Co","Se"):2.39642, ("Co","Te"):2.61642, ("Cr","F"):2.45293, ("Cr","H"):1.67642,
    ("Cr","I"):3.19293, ("Cr","N"):2.5152, ("Cr","O"):2.44293, ("Cr","P"):2.42642,
    ("Cr","S"):2.72491, ("Cr","Se"):2.44642, ("Cr","Te"):2.67642, ("Cs","F"):3.49942,
    ("Cs","H"):3.55942, ("Cs","I"):4.40942, ("Cs","N"):3.94942, ("Cs","O"):3.53642,
    ("Cs","P"):3.64194, ("Cs","S"):4.24942, ("Cs","Se"):4.21655, ("Cs","Te"):4.4631,
    ("Cu","F"):2.46295, ("Cu","H"):1.81649, ("Cu","I"):3.01795, ("Cu","N"):2.49295,
    ("Cu","O"):2.47295, ("Cu","P"):2.65649, ("Cu","S"):2.76095, ("Cu","Se"):2.76295,
    ("Cu","Te"):2.87649, ("D","F"):1.1, ("D","N"):1.2, ("D","O"):1.2, ("Dy","F"):2.52944,
    ("Dy","H"):2.09, ("Dy","I"):3.39945, ("Dy","N"):2.38, ("Dy","O"):2.65651, ("Dy","P"):2.77,
    ("Dy","S"):2.823, ("Dy","Se"):2.81, ("Dy","Te"):3.22, ("Er","F"):2.51049, ("Er","H"):2.06,
    ("Er","I"):3.38945, ("Er","N"):2.36, ("Er","O"):2.63651, ("Er","P"):2.75, ("Er","S"):3.27651,
    ("Er","Se"):3.18649, ("Er","Te"):3.22, ("Es","O"):2.70139, ("Eu","F"):2.83549,
    ("Eu","H"):2.18898, ("Eu","I"):3.69549, ("Eu","N"):2.95549, ("Eu","O"):2.94249,
    ("Eu","P"):2.85898, ("Eu","S"):3.37949, ("Eu","Se"):2.89898, ("Eu","Te"):3.08898,
    ("F","Fe"):2.36293, ("F","Ga"):2.14646, ("F","Gd"):3.15651, ("F","Ge"):2.01002, ("F","H"):1.1,
    ("F","Hf"):3.18291, ("F","Hg"):2.88293, ("F","Ho"):2.56159, ("F","I"):3.18295,
    ("F","In"):2.35491, ("F","Ir"):2.15002, ("F","K"):3.11142, ("F","Kr"):2.67549,
    ("F","La"):2.79293, ("F","Li"):2.34276, ("F","Lu"):2.48249, ("F","Mg"):2.29093,
    ("F","Mn"):2.41093, ("F","Mo"):2.2998, ("F","N"):1.82646, ("F","NH"):2.99195,
    ("F","Na"):2.80398, ("F","Nb"):2.35646, ("F","Nd"):2.73452, ("F","Ni"):2.20249,
    ("F","Np"):2.59233, ("F","Os"):2.07746, ("F","P"):2.01002, ("F","Pa"):2.54437,
    ("F","Pb"):2.92045, ("F","Pd"):2.34649, ("F","Pm"):2.55233, ("F","Po"):2.83646,
    ("F","Pr"):2.62945, ("F","Pt"):2.54002, ("F","Pu"):2.58233, ("F","Rb"):3.37645,
    ("F","Re"):2.16002, ("F","Rh"):2.16646, ("F","Ru"):2.57646, ("F","S"):1.95002,
    ("F","Sb"):2.35646, ("F","Sc"):2.32291, ("F","Se"):2.08002, ("F","Si"):1.93002,
    ("F","Sm"):2.60649, ("F","Sn"):2.63793, ("F","Sr"):2.88195, ("F","Ta"):2.2539,
    ("F","Tb"):2.54249, ("F","Tc"):2.24219, ("F","Te"):2.22002, ("F","Th"):2.68945,
    ("F","Ti"):2.86293, ("F","Tl"):3.26942, ("F","Tm"):2.51649, ("F","U"):2.80293,
    ("F","V"):2.87293, ("F","W"):2.03, ("F","Xe"):2.62649, ("F","Y"):2.51049, ("F","Yb"):2.50649,
    ("F","Zn"):2.38293, ("F","Zr"):2.99651, ("Fe","H"):1.68642, ("Fe","I"):3.1552,
    ("Fe","N"):2.48193, ("Fe","O"):2.44693, ("Fe","P"):2.42642, ("Fe","S"):2.83793,
    ("Fe","Se"):2.43642, ("Fe","Te"):2.68642, ("Ga","H"):1.55998, ("Ga","I"):2.91646,
    ("Ga","N"):1.96, ("Ga","O"):2.18646, ("Ga","P"):2.46998, ("Ga","S"):2.61946,
    ("Ga","Se"):3.41295, ("Ga","Te"):2.58998, ("Gd","H"):2.13, ("Gd","I"):3.41945, ("Gd","N"):2.42,
    ("Gd","O"):2.76651, ("Gd","P"):2.81, ("Gd","S"):3.13649, ("Gd","Se"):2.85, ("Gd","Te"):3.24,
    ("Ge","Ge"):2.6, ("Ge","H"):1.59998, ("Ge","I"):2.54998, ("Ge","N"):1.92998,
    ("Ge","O"):2.09802, ("Ge","P"):2.36998, ("Ge","S"):2.56702, ("Ge","Se"):2.70002,
    ("Ge","Te"):2.60998, ("H","Hf"):1.93642, ("H","Hg"):1.86642, ("H","Ho"):2.11898,
    ("H","In"):1.87642, ("H","Ir"):1.80998, ("H","K"):3.21942, ("H","La"):2.77293,
    ("H","Lu"):2.42649, ("H","Mg"):2.24293, ("H","Mn"):1.70642, ("H","Mo"):1.83701, ("H","N"):1.2,
    ("H","Na"):2.79942, ("H","Nb"):1.90642, ("H","Ni"):1.44998, ("H","O"):1.2, ("H","P"):1.45998,
    ("H","Pb"):2.27644, ("H","Pd"):1.51998, ("H","Pr"):2.62649, ("H","Pt"):1.44998,
    ("H","Rb"):3.43645, ("H","Re"):1.79998, ("H","Rh"):1.59998, ("H","Ru"):1.65998,
    ("H","S"):1.42998, ("H","Sb"):2.81998, ("H","Sc"):2.24291, ("H","Se"):1.58998,
    ("H","Si"):1.82002, ("H","Sm"):2.56649, ("H","Sn"):2.00642, ("H","Sr"):2.87295,
    ("H","Ta"):1.91642, ("H","Tb"):2.51649, ("H","Te"):1.87998, ("H","Th"):2.67649,
    ("H","Ti"):1.76642, ("H","Tl"):2.35644, ("H","Tm"):2.45649, ("H","U"):2.27644,
    ("H","V"):1.73642, ("H","W"):1.80998, ("H","Y"):2.46649, ("H","Yb"):2.42649,
    ("H","Zn"):2.13293, ("H","Zr"):2.29004, ("Hf","I"):2.83642, ("Hf","N"):2.24642,
    ("Hf","O"):2.37946, ("Hf","P"):2.63642, ("Hf","S"):2.64642, ("Hf","Se"):2.67642,
    ("Hf","Te"):2.87642, ("Hg","Hg"):3.1952, ("Hg","I"):3.33293, ("Hg","N"):2.17642,
    ("Hg","O"):2.86939, ("Hg","P"):2.57642, ("Hg","S"):3.42093, ("Hg","Se"):2.82642,
    ("Hg","Te"):2.76642, ("Ho","I"):3.44159, ("Ho","N"):2.41898, ("Ho","O"):2.67047,
    ("Ho","P"):2.79898, ("Ho","S"):3.13547, ("Ho","Se"):2.84898, ("Ho","Te"):3.03898,
    ("I","I"):2.4, ("I","In"):3.19291, ("I","Ir"):2.70998, ("I","K"):4.11717, ("I","La"):3.64293,
    ("I","Li"):3.37676, ("I","Lu"):3.36945, ("I","Mg"):3.17293, ("I","Mn"):3.23293,
    ("I","Mo"):2.74701, ("I","Na"):3.88251, ("I","Nb"):3.1439, ("I","Nd"):3.59452,
    ("I","Ni"):3.00649, ("I","Np"):3.44233, ("I","O"):2.47646, ("I","P"):2.44998,
    ("I","Pb"):3.69562, ("I","Pd"):2.96649, ("I","Pr"):3.49649, ("I","Pt"):2.41998,
    ("I","Pu"):3.43233, ("I","Rb"):4.33462, ("I","Re"):2.65998, ("I","Rh"):2.52998,
    ("I","Ru"):2.52998, ("I","S"):2.40998, ("I","Sb"):3.21646, ("I","Sc"):3.15291,
    ("I","Se"):2.58998, ("I","Si"):2.76002, ("I","Sm"):3.44649, ("I","Sn"):3.52293,
    ("I","Sr"):3.74295, ("I","Ta"):2.81642, ("I","Tb"):3.40945, ("I","Te"):3.13702,
    ("I","Th"):3.56649, ("I","Ti"):3.08291, ("I","Tl"):3.94142, ("I","Tm"):3.37945,
    ("I","U"):3.62452, ("I","V"):2.66642, ("I","W"):2.70998, ("I","Y"):3.37649, ("I","Yb"):3.37945,
    ("I","Zn"):3.07293, ("I","Zr"):3.19004, ("In","Mn"):3.14729, ("In","N"):2.18642,
    ("In","O"):2.46491, ("In","P"):2.68642, ("In","S"):2.93291, ("In","Se"):2.62642,
    ("In","Te"):2.84642, ("Ir","N"):2.10998, ("Ir","O"):2.27746, ("Ir","P"):2.50998,
    ("Ir","S"):2.42998, ("Ir","Se"):2.55998, ("Ir","Te"):2.75998, ("K","N"):3.41942,
    ("K","O"):3.25142, ("K","P"):3.44942, ("K","S"):3.79285, ("K","Se"):4.00186,
    ("K","Te"):4.23284, ("La","N"):3.05293, ("La","O"):2.90983, ("La","P"):3.32293,
    ("La","S"):3.35593, ("La","Se"):3.45293, ("La","Te"):3.65293, ("Li","N"):2.66213,
    ("Li","O"):2.60087, ("Li","S"):3.02481, ("Li","Se"):3.2433, ("Li","Te"):3.42496,
    ("Lu","N"):2.71649, ("Lu","O"):2.57749, ("Lu","P"):3.11649, ("Lu","S"):3.03649,
    ("Lu","Se"):3.16649, ("Lu","Te"):3.35649, ("Mg","N"):2.56293, ("Mg","O"):2.41824,
    ("Mg","P"):3.00293, ("Mg","S"):2.89293, ("Mg","Se"):3.03293, ("Mg","Te"):3.24293,
    ("Mn","N"):2.56193, ("Mn","O"):2.51652, ("Mn","P"):2.39642, ("Mn","S"):2.93293,
    ("Mn","Se"):2.47642, ("Mn","Te"):2.70642, ("Mo","N"):2.4735, ("Mo","O"):2.3475,
    ("Mo","P"):2.54701, ("Mo","S"):2.80067, ("Mo","Se"):2.59701, ("Mo","Te"):2.79701,
    ("N","N"):1.8826, ("N","Na"):3.12942, ("N","Nb"):2.46046, ("N","Nd"):3.01293,
    ("N","Ni"):2.30649, ("N","O"):1.81746, ("N","P"):1.97146, ("N","Pb"):3.0967,
    ("N","Pd"):2.40451, ("N","Pr"):2.90649, ("N","Pt"):2.41649, ("N","Rb"):3.79645,
    ("N","Re"):2.10998, ("N","Rh"):2.2626, ("N","Ru"):2.2626, ("N","S"):2.28849,
    ("N","Sb"):2.56446, ("N","Sc"):2.54291, ("N","Se"):2.1, ("N","Si"):2.12002, ("N","Sm"):3.02351,
    ("N","Sn"):2.7152, ("N","Sr"):3.09295, ("N","Ta"):2.16642, ("N","Tb"):2.80649,
    ("N","Te"):2.16998, ("N","Th"):2.94649, ("N","Ti"):2.08642, ("N","Tl"):2.59644,
    ("N","Tm"):2.74649, ("N","U"):2.78649, ("N","V"):2.38329, ("N","W"):2.10998, ("N","Y"):2.77649,
    ("N","Yb"):2.84851, ("N","Zn"):2.43293, ("N","Zr"):2.65004, ("NH","O"):3.08895,
    ("Na","O"):2.95693, ("Na","P"):3.47942, ("Na","S"):3.57685, ("Na","Se"):3.71593,
    ("Na","Te"):3.95459, ("Nb","O"):2.45329, ("Nb","P"):2.61642, ("Nb","S"):2.74,
    ("Nb","Se"):2.66642, ("Nb","Te"):2.85642, ("Nd","O"):2.8587, ("Nd","S"):3.42712,
    ("Nd","Se"):3.42293, ("Nd","Te"):3.60293, ("Ni","O"):2.28149, ("Ni","P"):2.31998,
    ("Ni","S"):2.58649, ("Ni","Se"):2.18998, ("Ni","Te"):2.47998, ("Np","O"):2.63646,
    ("Np","S"):3.2, ("O","O"):1.7, ("O","Os"):2.23, ("O","P"):2.08646, ("O","Pa"):2.63383,
    ("O","Pb"):3.04096, ("O","Pd"):2.39849, ("O","Po"):2.64646, ("O","Pr"):2.74449,
    ("O","Pt"):2.40649, ("O","Pu"):2.68329, ("O","Rb"):3.43945, ("O","Re"):2.3426,
    ("O","Rh"):2.24946, ("O","Ru"):2.22646, ("O","S"):1.8, ("O","Sb"):2.45237, ("O","Sc"):2.42029,
    ("O","Se"):2.16102, ("O","Si"):1.99002, ("O","Sm"):2.88251, ("O","Sn"):2.82146,
    ("O","Sr"):2.98095, ("O","Ta"):2.74646, ("O","Tb"):2.65549, ("O","Tc"):2.22446,
    ("O","Te"):2.3334, ("O","Th"):2.77349, ("O","Ti"):2.35391, ("O","Tl"):3.36945,
    ("O","Tm"):2.60649, ("O","U"):2.94295, ("O","V"):2.84939, ("O","W"):2.20102,
    ("O","Xe"):2.63451, ("O","Y"):2.62549, ("O","Yb"):2.74551, ("O","Zn"):2.41693,
    ("O","Zr"):3.09651, ("Os","S"):2.56002, ("P","P"):2.48381, ("P","Pb"):2.94644,
    ("P","Pd"):2.46998, ("P","Pr"):3.28649, ("P","Pt"):2.23998, ("P","Rb"):3.50645,
    ("P","Re"):2.50998, ("P","Rh"):2.43998, ("P","Ru"):2.33998, ("P","S"):2.57646,
    ("P","Sb"):2.56998, ("P","Sc"):2.96291, ("P","Se"):2.69646, ("P","Si"):2.58002,
    ("P","Sm"):3.23649, ("P","Sn"):2.60642, ("P","Sr"):3.43295, ("P","Ta"):2.62642,
    ("P","Tb"):3.19649, ("P","Te"):2.56998, ("P","Th"):3.33649, ("P","Ti"):2.51642,
    ("P","Tl"):3.01644, ("P","Tm"):3.13649, ("P","U"):2.94644, ("P","V"):2.46642,
    ("P","W"):2.50998, ("P","Y"):3.17649, ("P","Yb"):3.13649, ("P","Zn"):2.86293,
    ("P","Zr"):3.02004, ("Pb","S"):3.40395, ("Pb","Se"):3.55295, ("Pb","Te"):3.14644,
    ("Pd","S"):2.69649, ("Pd","Se"):2.26998, ("Pd","Te"):2.52998, ("Pr","S"):3.20649,
    ("Pr","Se"):3.32649, ("Pr","Te"):3.50649, ("Pt","S"):2.76649, ("Pt","Se"):2.23998,
    ("Pt","Te"):2.49998, ("Pu","S"):3.3, ("Rb","S"):3.97645, ("Rb","Se"):4.13773,
    ("Rb","Te"):4.28802, ("Re","S"):2.61998, ("Re","Se"):2.54998, ("Re","Te"):2.74998,
    ("Rh","S"):2.19998, ("Rh","Se"):2.37998, ("Rh","Te"):2.59998, ("Ru","S"):2.6426,
    ("Ru","Se"):2.69451, ("Ru","Te"):2.58998, ("S","S"):2.2, ("S","Sb"):3.05, ("S","Sc"):2.88391,
    ("S","Se"):2.81649, ("S","Si"):2.47602, ("S","Sm"):3.15649, ("S","Sn"):3.15293,
    ("S","Sr"):3.51295, ("S","Ta"):2.8439, ("S","Tb"):3.11649, ("S","Te"):2.79002,
    ("S","Th"):3.24649, ("S","Ti"):2.74646, ("S","Tl"):3.66442, ("S","Tm"):3.05649,
    ("S","U"):3.25293, ("S","V"):2.82293, ("S","W"):2.43998, ("S","Y"):3.08649, ("S","Yb"):3.03649,
    ("S","Zn"):2.80293, ("S","Zr"):2.91004, ("Sb","Se"):3.05646, ("Sb","Te"):2.82998,
    ("Sc","Se"):3.00291, ("Sc","Te"):3.20291, ("Se","Se"):2.93649, ("Se","Si"):2.61002,
    ("Se","Sm"):3.27649, ("Se","Sn"):2.96646, ("Se","Sr"):3.58295, ("Se","Ta"):2.66642,
    ("Se","Tb"):3.23649, ("Se","Te"):2.57998, ("Se","Th"):3.36649, ("Se","Ti"):2.53642,
    ("Se","Tl"):3.00644, ("Se","Tm"):3.18649, ("Se","U"):3.00644, ("Se","V"):2.48642,
    ("Se","W"):2.55998, ("Se","Y"):3.21649, ("Se","Yb"):3.16649, ("Se","Zn"):2.93293,
    ("Se","Zr"):3.03004, ("Si","Si"):2.6, ("Si","Te"):2.84002, ("Sm","Te"):3.46649,
    ("Sn","Te"):2.91642, ("Sr","Te"):3.73295, ("Ta","Te"):2.85642, ("Tb","Te"):3.42649,
    ("Te","Te"):2.80998, ("Te","Th"):3.54649, ("Te","Ti"):2.95642, ("Te","Tl"):3.23644,
    ("Te","Tm"):3.37649, ("Te","U"):3.16644, ("Te","V"):2.72642, ("Te","W"):2.75998,
    ("Te","Y"):3.40649, ("Te","Yb"):3.36649, ("Te","Zn"):3.16293, ("Te","Zr"):3.17004,
}


# Pauling の電気陰性度（境界外の結合原子を探すときの向き A1→A2 の判定に使う）
PAULING_EN = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82,
    "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.6, "Mo": 2.16, "Tc": 1.9,
    "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.1, "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Sm": 1.17, "Eu": 1.2, "Gd": 1.2, "Tb": 1.1, "Dy": 1.22, "Ho": 1.23, "Er": 1.24,
    "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3, "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2,
    "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0,
    "At": 2.2, "Fr": 0.7, "Ra": 0.9, "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36,
    "Pu": 1.28, "Am": 1.13,
}


BALL_SCALE = 0.3         # 球の半径 = VESTA 原子半径 × BALL_SCALE
BOND_RADIUS = 0.10       # 結合（円柱）の半径 [Å]
LABEL_SIZE = 11          # ラベルの文字サイズ [pt]
UI_SCALE = 1.0           # 画面の拡大率（Windows の表示スケール 150% なら 1.5）。起動時に設定


def element_style(sym):
    """(VESTA 原子半径 [Å], (r, g, b) 0–1) を返す。VESTA 表にない元素は共有結合半径と Jmol 色"""
    if sym in VESTA_ELEMENTS:
        return VESTA_ELEMENTS[sym]
    r, hexc = ATOM_STYLE.get(sym, (1.5, "#ff1493"))
    return r, tuple(int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))


P_BLOCK_FRAMEWORK = set("B Al Ga In Tl C Si Ge Sn Pb N P As Sb Bi O S Se Te".split())


def default_bond_rules(species, shortest=None):
    """構造に含まれる元素対ごとの結合ルール {(A,B): [有効, 最大距離]}。
    ・VESTA の既定表にある対 → 有効（VESTA の距離）
    ・表にない対で、両方が 13–16 族（クラスレートの Ga–Sn, Sn–Sn など骨格）かつ
      最短距離が共有結合半径の和×1.15 以内 → 有効（距離 = 共有結合半径の和×1.15）
    ・それ以外 → 無効（距離の目安 = 共有結合半径の和×1.15）"""
    rules = {}
    sp = sorted(set(species))
    for i, a in enumerate(sp):
        for b in sp[i:]:
            key = tuple(sorted((a, b)))
            if key in VESTA_BOND_CUTOFFS:
                rules[key] = [True, VESTA_BOND_CUTOFFS[key]]
                continue
            ra = ATOM_STYLE.get(a, (1.5, ""))[0]
            rb = ATOM_STYLE.get(b, (1.5, ""))[0]
            dmax = round((ra + rb) * 1.15, 3)
            on = (a in P_BLOCK_FRAMEWORK and b in P_BLOCK_FRAMEWORK and shortest is not None
                  and shortest.get(key, np.inf) <= dmax)
            rules[key] = [bool(on), dmax]
    return rules


def shortest_pair_distances(st):
    """元素対ごとの最短原子間距離（周期境界込み）"""
    out = {}
    L = st.lattice
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ L
    cart = np.mod(st.frac, 1.0) @ L
    sy = st.symbols
    for i in range(len(cart)):
        d = np.linalg.norm(cart[None, None, i] - (cart[:, None, :] + shifts[None, :, :]), axis=2)
        d[d < 0.1] = np.inf
        dm = d.min(axis=1)
        for j in range(len(cart)):
            key = tuple(sorted((sy[i], sy[j])))
            out[key] = min(out.get(key, np.inf), dm[j])
    return out


def find_bonds(cart, symbols, rules, lattice=None, max_elems=2_000_000):
    """元素対ごとの最大距離（rules）以下の原子対を結合とする。
    戻り値: (表示範囲内の結合 (M,2), 範囲外の相手への半結合 [(i, 相手の座標)])"""
    sp = sorted(set(symbols))
    idx = {s: k for k, s in enumerate(sp)}
    e = np.array([idx[s] for s in symbols])
    C = np.zeros((len(sp), len(sp)))
    for (a, b), (on, dmax) in rules.items():
        if on and a in idx and b in idx:
            C[idx[a], idx[b]] = C[idx[b], idx[a]] = dmax
    N = len(cart)
    pairs = []
    chunk = max(1, max_elems // max(1, N))
    for i0 in range(0, N, chunk):
        a = cart[i0:i0 + chunk]
        d = np.sqrt(((a[:, None, :] - cart[None, :, :]) ** 2).sum(axis=2))
        cut = C[e[i0:i0 + chunk][:, None], e[None, :]]
        ii, jj = np.where((d < cut) & (d > 0.1))
        ii = ii + i0
        m = ii < jj
        pairs.append(np.column_stack([ii[m], jj[m]]))
    inner = np.vstack(pairs) if pairs else np.zeros((0, 2), int)
    stubs = []
    if lattice is not None and N and C.max() > 0:
        rng = (-1, 0, 1)
        disp = {tuple(np.round(p, 3)) for p in cart}
        for sh in [np.array([i, j, k]) for i in rng for j in rng for k in rng]:
            if not sh.any():
                continue
            q = cart + sh @ lattice
            d = np.sqrt(((cart[:, None, :] - q[None, :, :]) ** 2).sum(axis=2))
            ii, jj = np.where((d < C[e[:, None], e[None, :]]) & (d > 0.1))
            for i, j in zip(ii, jj):
                if tuple(np.round(q[j], 3)) not in disp:
                    stubs.append((i, q[j]))
    return inner, stubs


def view_matrix(view_dir, up):
    """視線を view_dir（画面手前向き）、上方向を up とする回転行列（行 = 画面の x, y, z）"""
    z = np.asarray(view_dir, float)
    z = z / np.linalg.norm(z)
    y = np.asarray(up, float) - np.dot(up, z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.array([x, y, z])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _shade(c, f):
    return tuple(min(1.0, max(0.0, x * f)) for x in c)


def _tint(c, t):
    return tuple(x + (1 - x) * t for x in c)


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(min(1, max(0, x)) * 255)) for x in c)


def make_sphere_sprite(rgb, r_px, ss=4):
    """陰影付きの球の画像（RGBA, PIL）。VESTA 風に左上から光を当て、ハイライトと縁取りを付ける"""
    from PIL import Image
    r_px = max(2, int(round(r_px)))
    n = (2 * r_px + 2) * ss
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - c) / (r_px * ss)
    y = -(yy - c) / (r_px * ss)
    rr = x * x + y * y
    inside = rr <= 1.0
    z = np.sqrt(np.clip(1 - rr, 0, 1))
    L = np.array([-0.45, 0.55, 0.70])
    L = L / np.linalg.norm(L)
    diff = np.clip(x * L[0] + y * L[1] + z * L[2], 0, 1)
    H = L + np.array([0, 0, 1.0])
    H = H / np.linalg.norm(H)
    spec = np.clip(x * H[0] + y * H[1] + z * H[2], 0, 1) ** 40
    base = np.array(rgb, float)
    col = base[None, None, :] * (0.30 + 0.75 * diff[..., None]) + 0.55 * spec[..., None]
    rim = np.clip((np.sqrt(rr) - 0.93) / 0.07, 0, 1)                 # 縁を少し暗く
    col = col * (1 - 0.45 * rim[..., None])
    col = np.clip(col, 0, 1)
    alpha = np.clip((1.0 - np.sqrt(rr)) * r_px * ss / 1.5 + 0.5, 0, 1) * inside
    img = np.dstack([col * 255, alpha * 255]).astype(np.uint8)
    im = Image.fromarray(img, "RGBA")
    return im.resize((n // ss, n // ss), Image.LANCZOS)


# ----------------------------------------------------------------------
#  OpenGL による高速描画（回転・拡大を GPU で行う。使えない環境では Tk キャンバス描画に切り替え）
# ----------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def make_label_atlas(labels, size_px, font_path=None):
    """ラベル文字列の画像をまとめた 1 枚のテクスチャ（2 のべき乗サイズ）を作る。
    戻り値: (RGBA 画像の numpy 配列, {文字列: (u0, v0, u1, v1, w_px, h_px)})"""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for fp in ([font_path] if font_path else []) + ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"]:
        try:
            font = ImageFont.truetype(fp, max(6, int(size_px)))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    pad = 3
    tiles = []
    for s in sorted(set(labels)):
        bb = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), s, font=font, stroke_width=2)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
        im = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), s, font=font, fill=(0, 0, 0, 255),
                                stroke_width=2, stroke_fill=(255, 255, 255, 255))
        tiles.append((s, im))
    # 単純な行詰め
    W = _next_pow2(max([512] + [t[1].width for t in tiles]))
    x = y = rowh = 0
    pos = {}
    for s, im in tiles:
        if x + im.width > W:
            x, y = 0, y + rowh
            rowh = 0
        pos[s] = (x, y)
        x += im.width
        rowh = max(rowh, im.height)
    H = _next_pow2(max(16, y + rowh))
    atlas = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    info = {}
    for s, im in tiles:
        px, py = pos[s]
        atlas.paste(im, (px, py))
        info[s] = (px / W, py / H, (px + im.width) / W, (py + im.height) / H, im.width, im.height)
    return np.asarray(atlas, dtype=np.uint8), info


# ----------------------------------------------------------------------
#  新しい OpenGL（3.3 以上）による描画: 球 = インポスター（ピクセルごとの光線計算）,
#  結合 = 三角形の円柱のインスタンス描画, 4x MSAA でアンチエイリアス
# ----------------------------------------------------------------------
_VS_SPHERE = """
#version 330
layout(location=0) in vec2 corner;
layout(location=1) in vec3 center;
layout(location=2) in float radius;
layout(location=3) in vec3 color;
uniform mat4 view;
uniform mat4 proj;
out vec2 v_corner;
out vec3 v_center;
out float v_radius;
out vec3 v_color;
void main() {
    vec3 c = (view * vec4(center, 1.0)).xyz;
    v_corner = corner * 1.05;
    v_center = c;
    v_radius = radius;
    v_color = color;
    gl_Position = proj * vec4(c + vec3(v_corner * radius, 0.0), 1.0);
}
"""
_FS_SPHERE = """
#version 330
in vec2 v_corner;
in vec3 v_center;
in float v_radius;
in vec3 v_color;
uniform mat4 proj;
uniform vec3 light;
out vec4 frag;
void main() {
    float d2 = dot(v_corner, v_corner);
    if (d2 > 1.0) discard;
    float z = sqrt(1.0 - d2);
    vec3 n = vec3(v_corner, z);
    vec3 p = v_center + n * v_radius;
    vec4 clip = proj * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    vec3 L = normalize(light);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 40.0);
    vec3 col = v_color * (0.30 + 0.72 * diff) + vec3(0.50) * spec;
    float rim = clamp((sqrt(d2) - 0.92) / 0.08, 0.0, 1.0);
    col *= 1.0 - 0.40 * rim;
    frag = vec4(col, 1.0);
}
"""
_VS_CYL = """
#version 330
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
layout(location=2) in vec3 pa;
layout(location=3) in vec3 pb;
layout(location=4) in float radius;
layout(location=5) in vec3 color;
uniform mat4 view;
uniform mat4 proj;
out vec3 v_n;
out vec3 v_color;
void main() {
    vec3 d = pb - pa;
    vec3 z = normalize(d);
    vec3 t = abs(z.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 x = normalize(cross(t, z));
    vec3 y = cross(z, x);
    vec3 w = pa + (x * pos.x + y * pos.y) * radius + d * pos.z;
    vec3 nw = normalize(x * nrm.x + y * nrm.y + z * nrm.z);
    v_n = mat3(view) * nw;
    v_color = color;
    gl_Position = proj * view * vec4(w, 1.0);
}
"""
_FS_CYL = """
#version 330
in vec3 v_n;
in vec3 v_color;
uniform vec3 light;
out vec4 frag;
void main() {
    vec3 n = normalize(v_n);
    vec3 L = normalize(light);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 40.0);
    frag = vec4(v_color * (0.30 + 0.72 * diff) + vec3(0.45) * spec, 1.0);
}
"""
_VS_LINE = """
#version 330
layout(location=0) in vec3 pos;
uniform mat4 view;
uniform mat4 proj;
void main() { gl_Position = proj * view * vec4(pos, 1.0); }
"""
_FS_LINE = """
#version 330
uniform vec3 color;
out vec4 frag;
void main() { frag = vec4(color, 1.0); }
"""
_VS_LABEL = """
#version 330
layout(location=0) in vec2 corner;
layout(location=1) in vec3 center;
layout(location=2) in float lift;
layout(location=3) in vec4 uv;
layout(location=4) in vec2 size_px;
uniform mat4 view;
uniform mat4 proj;
uniform float px_per_unit;
out vec2 v_uv;
void main() {
    vec3 c = (view * vec4(center, 1.0)).xyz;
    c.z += lift;
    vec2 half_size = size_px * 0.5 / px_per_unit;
    gl_Position = proj * vec4(c + vec3(corner * half_size, 0.0), 1.0);
    vec2 t = corner * 0.5 + 0.5;
    v_uv = vec2(mix(uv.x, uv.z, t.x), mix(uv.w, uv.y, t.y));
}
"""
_FS_LABEL = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;
out vec4 frag;
void main() {
    vec4 c = texture(tex, v_uv);
    if (c.a < 0.05) discard;
    frag = c;
}
"""


def _unit_cylinder_mesh(nseg=16):
    """z = 0..1、半径 1 の円柱の側面（三角形）。頂点ごとに (位置, 法線)"""
    v = []
    for j in range(nseg):
        a0, a1 = 2 * np.pi * j / nseg, 2 * np.pi * (j + 1) / nseg
        p00 = (np.cos(a0), np.sin(a0), 0.0)
        p01 = (np.cos(a0), np.sin(a0), 1.0)
        p10 = (np.cos(a1), np.sin(a1), 0.0)
        p11 = (np.cos(a1), np.sin(a1), 1.0)
        n0 = (np.cos(a0), np.sin(a0), 0.0)
        n1 = (np.cos(a1), np.sin(a1), 0.0)
        for p, n in ((p00, n0), (p10, n1), (p11, n1), (p00, n0), (p11, n1), (p01, n0)):
            v.append(p + n)
    return np.array(v, np.float32)


def _unit_cone_mesh(nseg=16):
    v = []
    k = 1 / np.sqrt(2)
    for j in range(nseg):
        a0, a1 = 2 * np.pi * j / nseg, 2 * np.pi * (j + 1) / nseg
        am = (a0 + a1) / 2
        for (p, n) in (((np.cos(a0), np.sin(a0), 0.0), (np.cos(a0) * k, np.sin(a0) * k, k)),
                       ((np.cos(a1), np.sin(a1), 0.0), (np.cos(a1) * k, np.sin(a1) * k, k)),
                       ((0.0, 0.0, 1.0), (np.cos(am) * k, np.sin(am) * k, k))):
            v.append(p + n)
    return np.array(v, np.float32)


def add_bonded_outside(st, n, cart, sy, lb, rules):
    """VESTA の既定（境界内の A1 に結合した A2 を境界外でも表示）に相当。
    A1 = 電気陰性度の小さい側（陽イオン側）。同種元素の対や電気陰性度が同じ対では探さない。1 層のみ。"""
    sites_c = np.mod(st.frac, 1.0) @ st.lattice
    labels_site = []
    cnt = {}
    for s in st.symbols:
        cnt[s] = cnt.get(s, 0) + 1
        labels_site.append(f"{s}{cnt[s]}")
    have = {tuple(np.round(p, 3)) for p in cart}
    new_c, new_s, new_l = [], [], []
    rng = [range(-1, k + 1) for k in n]
    shifts = np.array([[i, j, k] for i in rng[0] for j in rng[1] for k in rng[2]]) @ st.lattice
    for (a, b), (on, dmax) in rules.items():
        if not on or a == b:
            continue
        ea, eb = PAULING_EN.get(a, 0), PAULING_EN.get(b, 0)
        if ea == eb:
            continue
        A1, A2 = (a, b) if ea < eb else (b, a)
        idx1 = [i for i, s in enumerate(sy) if s == A1]
        idx2 = [k for k, s in enumerate(st.symbols) if s == A2]
        if not idx1 or not idx2:
            continue
        cand = (sites_c[idx2][:, None, :] + shifts[None, :, :]).reshape(-1, 3)
        cand_l = [labels_site[k] for k in idx2 for _ in range(len(shifts))]
        P1 = cart[idx1]
        hit = set()
        chunk = max(1, 2_000_000 // max(1, len(cand)))
        for i0 in range(0, len(P1), chunk):
            d = np.sqrt(((P1[i0:i0 + chunk, None, :] - cand[None, :, :]) ** 2).sum(axis=2))
            hit.update(np.where((d < dmax) & (d > 0.1))[1].tolist())
        for j in sorted(hit):
            key = tuple(np.round(cand[j], 3))
            if key not in have:
                have.add(key)
                new_c.append(cand[j])
                new_s.append(A2)
                new_l.append(cand_l[j])
    if new_c:
        cart = np.vstack([cart, np.array(new_c)])
        sy = sy + new_s
        lb = lb + new_l
    return cart, sy, lb, len(new_c)


def build_structure_model(st, n, packed, rules, ball_scale, want_stubs, bonded_outside=True):
    cart, sy, lb = build_supercell_atoms(st, n, packed)
    n_out = 0
    if bonded_outside:
        cart, sy, lb, n_out = add_bonded_outside(st, n, cart, sy, lb, rules)
    sty = [element_style(s) for s in sy]
    rad = np.array([r for r, _ in sty]) * ball_scale
    col = [c for _, c in sty]
    bonds, stubs = find_bonds(cart, sy, rules, st.lattice if want_stubs else None)
    edges = []
    o = np.zeros(3)
    a, b, c = st.lattice
    corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
    E = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    for i, j in E:
        edges.append((corners[i], corners[j], 0.014))
    return dict(cart=cart, sy=sy, lb=lb, rad=rad, col=col, bonds=bonds, stubs=stubs, cell_edges=edges,
                n_outside=n_out)


# ======================================================================
#  GUI
# ======================================================================
def enable_high_dpi():
    """Windows に「高 DPI 対応」を宣言する（Tk のウィンドウを作る前に呼ぶ）。
    宣言しないと、表示スケール 125%/150% などで Windows が画面を画像として引き伸ばし、文字がにじむ。"""
    if not sys.platform.startswith("win"):
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # システム DPI 対応（Windows 8.1 以降）
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()       # 古い Windows 用
        except Exception:
            pass


def fit_geometry(win, w, h):
    """96 dpi 基準の大きさ w×h を拡大率に合わせ、画面からはみ出さないように設定する"""
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    W = min(int(w * UI_SCALE), int(sw * 0.96))
    H = min(int(h * UI_SCALE), int(sh * 0.88))
    win.geometry(f"{W}x{H}")
    return W, H


_SCROLL_PANELS = []


def make_scroll_panel(parent, tk, ttk, padding=8, resizable=False):
    """縦スクロールできる左パネル。中身が画面の高さに収まらないときだけスクロールバーを出す。
    戻り値 (outer, inner): outer を pack し、部品は inner に置く。"""
    outer = ttk.Frame(parent)
    try:
        bg = ttk.Style().lookup("TFrame", "background") or parent.cget("bg")
    except Exception:
        bg = None
    canvas = tk.Canvas(outer, highlightthickness=0, bd=0, yscrollincrement=max(1, int(20 * UI_SCALE)))
    if bg:
        try:
            canvas.configure(bg=bg)
        except Exception:
            pass
    vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    inner = ttk.Frame(canvas, padding=padding)
    canvas.create_window(0, 0, window=inner, anchor="nw")
    shown = {"v": False}

    def update(_=None):
        rw, rh = inner.winfo_reqwidth(), inner.winfo_reqheight()
        if resizable:
            if not shown.get("w0"):
                shown["w0"] = True
                canvas.configure(width=rw)         # 幅は最初だけ合わせ、あとは境界線のドラッグに任せる
            canvas.configure(scrollregion=(0, 0, rw, rh))
        else:
            canvas.configure(width=rw, scrollregion=(0, 0, rw, rh))
        ch = canvas.winfo_height()
        need = ch > 1 and rh > ch + 1
        if need and not shown["v"]:
            vsb.pack(side="right", fill="y", before=canvas)   # 幅を狭めてもスクロールバーが残るよう先に配置
            shown["v"] = True
        elif not need and shown["v"]:
            vsb.pack_forget()
            shown["v"] = False
            canvas.yview_moveto(0)

    _SCROLL_PANELS.append((outer, canvas, shown))

    # ホイール: パネル内のすべての部品に専用のバインドタグを先頭に付け、どこにカーソルがあってもスクロール
    tag = f"ScrollPanel{len(_SCROLL_PANELS)}_{id(outer)}"

    def on_wheel(ev, step=None):
        if not shown["v"]:
            return None
        if step is None:
            step = -3 if ev.delta > 0 else 3
        canvas.yview_scroll(step, "units")
        return "break"

    outer.bind_class(tag, "<MouseWheel>", on_wheel)                          # Windows / macOS
    outer.bind_class(tag, "<Button-4>", lambda e: on_wheel(e, -3))           # Linux
    outer.bind_class(tag, "<Button-5>", lambda e: on_wheel(e, 3))

    # トラックパッドの 2 本指スクロール（Tk 9 の <TouchpadScroll>。Windows / macOS）
    # %D に Δx（上位 16 bit）と Δy（下位 16 bit）が符号付きで入っている（TIP 684）
    acc = {"y": 0.0}

    def on_touchpad(ev):
        if not shown["v"]:
            return None
        try:
            d = int(ev.delta)
        except Exception:
            return "break"
        dy = d & 0xFFFF
        if dy >= 0x8000:
            dy -= 0x10000
        acc["y"] -= dy
        step = float(canvas.cget("yscrollincrement")) or 20.0
        n = int(acc["y"] / step)
        if n:
            canvas.yview_scroll(n, "units")
            acc["y"] -= n * step
        return "break"

    try:
        outer.bind_class(tag, "<TouchpadScroll>", on_touchpad)
    except Exception:
        pass      # Tk 8.6 以前には <TouchpadScroll> がない（その場合タッチパッドは <MouseWheel> として届く）

    def tag_all(w):
        try:
            t = w.bindtags()
            if tag not in t:
                w.bindtags((tag,) + tuple(t))
            for c in w.winfo_children():
                tag_all(c)
        except Exception:
            pass

    def update_and_tag(_=None):
        update()
        tag_all(outer)

    inner.bind("<Configure>", update_and_tag)
    canvas.bind("<Configure>", update_and_tag)
    outer.after_idle(lambda: tag_all(outer))
    return outer, inner


# ######################################################################
#  ここから下は VASP Result Viewer で新しく書いた部分
# ######################################################################

# ======================================================================
#  設定値（必要に応じてここだけ書き換えればよい）
# ======================================================================
ISO_SIGMA_N = 2.0            # 等値面の既定値 d = <|ρ|> + n σ(|ρ|) の n（VESTA の既定値 2）
ISO_COLOR_POS = (1.0, 1.0, 0.0)      # 正の等値面の色（VESTA の既定: 黄）
ISO_COLOR_NEG = (0.0, 1.0, 1.0)      # 負の等値面の色（VESTA の既定: 水色）
ISO_OPACITY = 0.6            # 等値面の不透明度の初期値
CHG_AUTO_MAXGRID = 160       # 解像度「自動」のとき、1 辺がこの点数以下になるよう間引く
CHG_MAX_TRIANGLES = 4_000_000    # 簡易描画（OpenGL なし）で、スーパーセル全体の三角形の上限
CHG_MAX_TRIANGLES_GPU = 100_000_000   # OpenGL（インスタンス描画）で、スーパーセル全体の三角形の上限
FS_INTERP_TARGET = 48        # フェルミ面の補間の初期値: k点メッシュの最小の辺がこの点数以上になる倍率
FS_INTERP_MAX = 6            # 補間の倍率の上限（FermiSurfer の Interpolation に相当）
FS_LABEL_SIZE = 14           # フェルミ面の軸ラベルの文字サイズ [pt]
ENERGY_WINDOW = (-4.0, 4.0)  # DOS / BAND の縦軸の初期範囲 [eV]（E_F 基準）
FATBAND_SCALE = 60.0         # PBAND の丸の大きさ（重み 1 のときの面積 [pt^2]）
# グラフの系列の色（matplotlib の既定の 10 色）
PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
AXIS_COLORS = {"avg": "#000000", "a": "#d00000", "b": "#00a000", "c": "#0000d0"}   # 結晶軸と同じ配色
# FermiSurfer のカラーバー（BGR: 青→水→緑→黄→赤）
FS_BAR = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
EVANG_TO_MS = 1.602176634e-19 * 1e-10 / 1.054571817e-34      # |∇_k E| [eV Å] → 速度 [m/s]
ANG3_TO_BOHR3 = 0.529177210903 ** 3                          # e/Å^3 → e/bohr^3 の係数


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _floats(s, dtype=np.float64):
    """空白区切りの数値文字列 → numpy 配列。
    Fortran が桁あふれで書いた '********' や、E の落ちた指数表記は 1 つずつ変換する（'***' は NaN）"""
    t = s.split()
    try:
        return np.array(t, dtype=dtype)
    except ValueError:
        return np.array([_ffloat(x) for x in t], dtype=dtype)


def _ffloat(t):
    """'0.123-100'（E が落ちた指数表記）や '*****' にも耐える変換"""
    try:
        return float(t)
    except ValueError:
        import re
        m = re.match(r"^([+-]?\d*\.?\d+)([+-]\d+)$", t)
        if m:
            return float(m.group(1) + "E" + m.group(2))
        return float("nan")


# ======================================================================
#  CHGCAR（電荷密度）
# ======================================================================
def read_chgcar(path):
    """CHGCAR / CHG / PARCHG を読む。戻り値: (Structure, rho[ix, iy, iz] (e/Å^3), 情報 dict)
    ファイルには ρ(r)×体積 が x が最も速く回る順で書かれている（VASP Wiki: CHGCAR）。
    スピン分極・非共線計算では最初のブロック（全電荷）だけを読む。"""
    lines = _read_text(path).split("\n")
    i = 7
    while i < len(lines) and lines[i].strip():
        i += 1
    if i >= len(lines) - 2:
        raise ValueError("CHGCAR の形式ではありません（格子点数の行が見つかりません）。")
    st = parse_poscar("\n".join(lines[:i]))
    dims = lines[i + 1].split()
    if len(dims) < 3:
        raise ValueError("CHGCAR の格子点数 (NGXF NGYF NGZF) を読み取れませんでした。")
    nx, ny, nz = (int(t) for t in dims[:3])
    n = nx * ny * nz
    per = max(1, len(lines[i + 2].split()))
    nl = -(-n // per)
    vals = _floats(" ".join(lines[i + 2:i + 2 + nl]))
    if vals.size < n:
        raise ValueError(f"電荷密度のデータ数が足りません（{vals.size} / {n}）。ファイルが途中で切れていませんか。")
    vol = abs(np.linalg.det(st.lattice))
    rho = (vals[:n].reshape((nz, ny, nx)).transpose(2, 1, 0) / vol).astype(np.float32)
    info = {"grid": (nx, ny, nz), "volume": vol, "nelect": float(vals[:n].sum() / n),
            "min": float(rho.min()), "max": float(rho.max())}
    return st, np.ascontiguousarray(rho), info


def default_iso_level(rho, n_sigma=ISO_SIGMA_N):
    """VESTA の既定の等値面レベル d = <|ρ|> + n σ(|ρ|)（VESTA マニュアル 16 章）"""
    a = np.abs(rho.astype(np.float64))
    d = float(a.mean() + n_sigma * a.std())
    if not (a.min() < d < a.max()):          # ほぼ一様な密度などで式の値が範囲外になる場合は中央の値
        d = float(0.5 * (a.min() + a.max()))
    return d


# ======================================================================
#  等値面（四面体法）と平面による切り取り
# ======================================================================
# 立方体の頂点番号 c = 4*d0 + 2*d1 + d2 と、6 個の四面体への分割は FermiSurfer と同じ
# （fermi_patch.cpp / initialize.cpp の corner1[0]: 主対角線 0–7 を共有する分割）
_CUBE = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])
_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])


def isosurface_tets(F, level, mat, attrs=None, max_cells=6_000_000, src=False):
    """周期的な格子上のスカラー場 F[i0, i1, i2] の等値面 F = level を四面体法で三角形にする。
    mat: 分率座標 → デカルト座標の行列（行ベクトル。実空間なら格子、逆空間なら逆格子）
    attrs: 一緒に補間する量 (n0, n1, n2, K)（色に使う重みなど）
    戻り値 D: (T, 3, 6 + K)。各頂点 [x, y, z, ∇F のデカルト成分 3 個, 補間した attrs K 個]
    src=True なら、さらに 12 列を付ける: 頂点を作った四面体の 4 頂点への重み λ (4) と、
    その 4 頂点の格子番号（i0*n1*n2 + i1*n2 + i2）を 4096 で割った商 (4)・余り (4)。
    これがあれば、格子上の任意の量 W の頂点での値を Σ λ_c W[c] で後から求められる（vertex_values）。
    クリップや平行移動でも λ は線形に補間され、格子番号は三角形の中で共通なので壊れない。"""
    F = np.asarray(F, dtype=np.float32)
    n = np.array(F.shape)
    K = 0 if attrs is None else attrs.shape[3]
    empty = np.zeros((0, 3, 6 + K + (12 if src else 0)), np.float32)
    G = F - np.float32(level)
    Ge = np.pad(G, ((0, 1), (0, 1), (0, 1)), mode="wrap")
    neg = Ge < 0                                        # 8 頂点の符号がそろっていない立方体だけを処理する
    any_neg = all_neg = None
    for d0, d1, d2 in _CUBE:
        c = neg[d0:d0 + n[0], d1:d1 + n[1], d2:d2 + n[2]]
        any_neg = c if any_neg is None else (any_neg | c)
        all_neg = c if all_neg is None else (all_neg & c)
    I = np.argwhere(any_neg & ~all_neg)
    del neg, any_neg, all_neg
    if len(I) == 0:
        return empty
    if len(I) > max_cells:
        raise MemoryError(f"等値面が通る格子が多すぎます（{len(I)} 個）。レベルを上げるか解像度を下げてください。")
    C = I[:, None, :] + _CUBE[None, :, :]               # (M, 8, 3) 頂点の格子番号（周期の外に 1 だけ出る）
    Cm = C % n
    i0, i1, i2 = Cm[..., 0], Cm[..., 1], Cm[..., 2]
    V = G[i0, i1, i2]                                   # (M, 8)
    # 勾配（周期境界つき中心差分）: 分率座標での微分 → デカルト座標
    g = np.empty(C.shape, np.float32)
    g[..., 0] = (G[(i0 + 1) % n[0], i1, i2] - G[(i0 - 1) % n[0], i1, i2]) * (n[0] / 2.0)
    g[..., 1] = (G[i0, (i1 + 1) % n[1], i2] - G[i0, (i1 - 1) % n[1], i2]) * (n[1] / 2.0)
    g[..., 2] = (G[i0, i1, (i2 + 1) % n[2]] - G[i0, i1, (i2 - 1) % n[2]]) * (n[2] / 2.0)
    g = g @ np.linalg.inv(np.asarray(mat, float)).T.astype(np.float32)
    X = (C / n).astype(np.float32)                      # 分率座標
    cols = [X, g]
    if K:
        cols.append(attrs[i0, i1, i2].astype(np.float32))
    Dc = np.concatenate(cols, axis=2)                   # (M, 8, 6+K)

    out = []
    for tet in _TETS:
        v = V[:, tet]
        nb = (v < 0).sum(axis=1)
        m = (nb > 0) & (nb < 4)
        if not m.any():
            continue
        v = v[m]
        d = Dc[m][:, tet]
        order = np.argsort(v, axis=1)
        v = np.take_along_axis(v, order, axis=1)
        d = np.take_along_axis(d, order[:, :, None], axis=1)
        nb = nb[m]
        if src:
            ci = np.take_along_axis(Cm[m][:, tet], order[:, :, None], axis=1)            # (行, 4, 3)
            flat = (ci[..., 0] * n[1] + ci[..., 1]) * n[2] + ci[..., 2]                   # (行, 4)
            ids = np.concatenate([flat // 4096, flat % 4096], axis=1).astype(np.float32)   # (行, 8)
            lam = np.broadcast_to(np.eye(4, dtype=np.float32), (len(v), 4, 4))
            d = np.concatenate([d, lam, np.broadcast_to(ids[:, None, :], (len(v), 4, 8))], axis=2)

        def cut(rows, i, j):
            vi, vj = v[rows, i], v[rows, j]
            t = (-vi / (vj - vi))[:, None]
            return d[rows, i] * (1 - t) + d[rows, j] * t

        r = np.where(nb == 1)[0]
        if len(r):
            out.append(np.stack([cut(r, 0, 1), cut(r, 0, 2), cut(r, 0, 3)], axis=1))
        r = np.where(nb == 2)[0]
        if len(r):
            p02, p03, p12, p13 = cut(r, 0, 2), cut(r, 0, 3), cut(r, 1, 2), cut(r, 1, 3)
            out.append(np.stack([p02, p03, p12], axis=1))
            out.append(np.stack([p13, p03, p12], axis=1))
        r = np.where(nb == 3)[0]
        if len(r):
            out.append(np.stack([cut(r, 0, 3), cut(r, 1, 3), cut(r, 2, 3)], axis=1))
    if not out:
        return empty
    D = np.concatenate(out, axis=0)
    D[:, :, :3] = D[:, :, :3] @ np.asarray(mat, np.float32)
    return D


def clip_halfspace(D, nrm, dist):
    """三角形 D (T, 3, C)（先頭 3 成分が位置）のうち nrm·x <= dist の側だけを残す（境界で切る）"""
    if len(D) == 0:
        return D
    s = D[:, :, :3] @ np.asarray(nrm, np.float32) - np.float32(dist)
    inside = s <= 0
    cnt = inside.sum(axis=1)
    if (cnt == 3).all():
        return D
    out = [D[cnt == 3]]
    for r in range(3):
        i, j, k = r, (r + 1) % 3, (r + 2) % 3
        m = (cnt == 1) & inside[:, i]                    # 頂点 i だけが内側
        if m.any():
            A, sa = D[m, i], s[m, i]
            tj = (sa / (sa - s[m, j]))[:, None]
            tk = (sa / (sa - s[m, k]))[:, None]
            out.append(np.stack([A, A + (D[m, j] - A) * tj, A + (D[m, k] - A) * tk], axis=1))
        m = (cnt == 2) & ~inside[:, i]                   # 頂点 i だけが外側
        if m.any():
            O, J, Kk, so = D[m, i], D[m, j], D[m, k], s[m, i]
            tj = (s[m, j] / (s[m, j] - so))[:, None]
            tk = (s[m, k] / (s[m, k] - so))[:, None]
            PJ = J + (O - J) * tj
            PK = Kk + (O - Kk) * tk
            out.append(np.stack([J, Kk, PK], axis=1))
            out.append(np.stack([J, PK, PJ], axis=1))
    return np.concatenate(out, axis=0)


def bz_planes(B, nmax=3):
    """第一ブリルアンゾーンを囲む Bragg 面の逆格子ベクトル G の一覧（面は G·k = |G|^2 / 2）"""
    rng = np.arange(-nmax, nmax + 1)
    nn = np.array(np.meshgrid(rng, rng, rng, indexing="ij")).reshape(3, -1).T
    nn = nn[np.any(nn != 0, axis=1)]
    G = nn @ B
    G2 = np.einsum("ij,ij->i", G, G)
    order = np.argsort(G2)
    G, G2 = G[order], G2[order]
    lhs = (0.5 * G) @ G.T
    inside = lhs <= 0.5 * G2[None, :] + 1e-8 * G2[0]
    np.fill_diagonal(inside, True)
    return G[inside.all(axis=1)]


def fold_into_first_bz(D, B):
    """逆格子の単位胞 [0,1)^3 で作った三角形を第一ブリルアンゾーンに移す
    （FermiSurfer の First Brillouin zone 表示に相当）。
    1) 三角形の重心が原点に最も近くなる逆格子ベクトルだけ平行移動する
    2) 移動後に 3 頂点ともゾーンの中にある三角形は、それで終わり（大部分はこれ）
    3) ゾーンの境界をまたぐ三角形（全体の数 %）だけを、Bragg 面で切って振り分ける"""
    if len(D) == 0:
        return D
    rng = (-2, -1, 0, 1)
    S = (np.array([[a, b, c] for a in rng for b in rng for c in rng], float) @ B).astype(np.float32)
    S2 = np.einsum("ij,ij->i", S, S)
    P = bz_planes(B).astype(np.float32)
    P2h = 0.5 * np.einsum("ij,ij->i", P, P)
    tol = 1e-5 * float(P2h.min())

    def nearest(V):                            # 各点について |v + S|^2 が最小になる S の番号
        g = np.empty(len(V), np.intp)
        for i0 in range(0, len(V), 400_000):
            g[i0:i0 + 400_000] = np.argmin(2.0 * (V[i0:i0 + 400_000] @ S.T) + S2[None, :], axis=1)
        return g

    out = D.copy()
    out[:, :, :3] += S[nearest(D[:, :, :3].mean(axis=1))][:, None, :]
    inside = np.ones(len(D), bool)
    for i0 in range(0, len(D), 300_000):
        v = out[i0:i0 + 300_000, :, :3]
        inside[i0:i0 + 300_000] = ((v @ P.T) <= P2h + tol).all(axis=(1, 2))
    if inside.all():
        return out
    rest = D[~inside]
    g = nearest(rest[:, :, :3].reshape(-1, 3)).reshape(-1, 3)
    pieces = [out[inside]]
    for si in np.unique(g):
        Dk = rest[(g == si).any(axis=1)].copy()
        Dk[:, :, :3] += S[si]
        for gv, g2h in zip(P, P2h):
            Dk = clip_halfspace(Dk, gv, g2h)
            if len(Dk) == 0:
                break
        if len(Dk):
            pieces.append(Dk)
    return np.concatenate(pieces, axis=0)


def bgr_colors(t):
    """0–1 の値 → FermiSurfer の BGR カラースケール (…, 3)"""
    x = np.clip(np.nan_to_num(np.asarray(t, np.float32)), 0.0, 1.0) * np.float32(4.0)
    seg = np.minimum(x.astype(np.int8), 3)
    f = (x - seg)[..., None]
    bar = np.asarray(FS_BAR, np.float32)
    return bar[seg] * (1 - f) + bar[seg + 1] * f


def box_edges(vecs, origin=None):
    """平行六面体の 12 本の辺 → (24, 3) の頂点列（線分ごとに 2 点）"""
    o = np.zeros(3) if origin is None else np.asarray(origin, float)
    a, b, c = np.asarray(vecs, float)
    cn = [o, o + a, o + b, o + c, o + a + b, o + a + c, o + b + c, o + a + b + c]
    E = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    return np.array([cn[k] for e in E for k in e], np.float32)


# ======================================================================
#  EIGENVAL / DOSCAR / OUTCAR / KPOINTS / PROCAR
# ======================================================================
def read_eigenval(path):
    """戻り値: dict(kpts (nk,3), weights (nk,), eig (nspin, nk, nb), nelect)"""
    lines = _read_text(path).split("\n")
    try:
        ispin = int(lines[0].split()[3])
        nelect, nk, nb = (int(float(t)) for t in lines[5].split()[:3])
    except Exception:
        raise ValueError("EIGENVAL のヘッダを読み取れませんでした。")
    nspin = 2 if ispin == 2 else 1
    kpts = np.zeros((nk, 3))
    wts = np.zeros(nk)
    eig = np.zeros((nspin, nk, nb))
    i = 6
    for k in range(nk):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i + nb >= len(lines):
            raise ValueError("EIGENVAL が途中で切れています。")
        t = _floats(lines[i])
        kpts[k], wts[k] = t[:3], t[3]
        blk = _floats(" ".join(lines[i + 1:i + 1 + nb])).reshape(nb, -1)
        eig[0, k] = blk[:, 1]
        if nspin == 2:
            eig[1, k] = blk[:, 2]
        i += 1 + nb
    return {"kpts": kpts, "weights": wts, "eig": eig, "nelect": nelect}


def read_doscar_efermi(path):
    """DOSCAR の 6 行目 4 番目の値（フェルミエネルギー）。VASPKIT 262/263 と同じ読み方"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(5):
            f.readline()
        return float(f.readline().split()[3])


def read_outcar_efermi(path):
    ef = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "E-fermi" in line:
                try:
                    ef = float(line.split(":")[1].split()[0])
                except Exception:
                    pass
    if ef is None:
        raise ValueError("OUTCAR に E-fermi が見つかりません。")
    return ef


def _kpt_label_text(s):
    s = s.replace("!", " ").replace("#", " ").strip()
    return s.split()[0] if s else ""


def read_kpoints_file(path):
    """KPOINTS を読む。戻り値 dict:
      kind = "mesh"   : mesh=(n1,n2,n3), gamma=True/False
      kind = "line"   : npts, segments=[(ラベル始, ラベル終), ...]
      kind = "list"   : 明示的な k点リスト（mesh は 1 行目に書かれていれば入る）"""
    lines = [l for l in _read_text(path).replace("\r", "").split("\n")]
    out = {"kind": "list", "mesh": None, "gamma": True, "segments": []}
    t0 = lines[0].split()
    # VASPKIT 261（および VASP Input Generator）が 1 行目に書くメッシュ: "G  n1 n2 n3  分解能  # ..."
    if len(t0) >= 4 and t0[0].upper() in ("G", "M"):
        try:
            out["mesh"] = tuple(int(x) for x in t0[1:4])
            out["gamma"] = t0[0].upper() == "G"
        except ValueError:
            pass
    try:
        n = int(lines[1].split()[0])
    except Exception:
        return out
    mode = lines[2].strip()[:1].lower() if len(lines) > 2 else ""
    if mode == "l":
        out["kind"] = "line"
        out["npts"] = n
        pts = []
        for l in lines[4:]:
            t = l.split()
            if len(t) >= 3:
                try:
                    [float(x) for x in t[:3]]
                except ValueError:
                    continue
                pts.append(_kpt_label_text(" ".join(t[3:])))
        out["segments"] = [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
    elif n == 0 and mode in ("g", "m") and len(lines) > 3:
        out["kind"] = "mesh"
        out["mesh"] = tuple(int(float(x)) for x in lines[3].split()[:3])
        out["gamma"] = mode == "g"
    elif n > 0:
        # 明示的な k点リスト: "x y z 重み ラベル" の形でラベルが付いていれば読む
        labs = []
        for l in lines[3:3 + n]:
            t = l.split()
            labs.append(_kpt_label_text(" ".join(t[4:])) if len(t) > 4 else "")
        if len(labs) == n:
            out["point_labels"] = labs
    return out


def _procar_fast(buf, pos0, nk, nb, nion, key):
    """PROCAR は固定書式なので、k点 1 個分のうち最初の band 行から先は、どの k点でも同じバイト数・同じ並びになる。
    そこで各原子の tot 欄のバイト位置を最初の k点で調べておき、他の k点では同じ位置のバイトだけを取り出して
    まとめて数値にする（行に分けて 1 つずつ変換するより 1 桁速い）。
    書式が一定でない・桁あふれがあるなどで使えないときは None（呼び出し側が通常の方法で読む）。
    戻り値: tot (nk, nb, nion) float32"""
    try:
        if pos0 < 0:
            return None
        total = len(buf)
        pos1 = buf.find(key, pos0 + 8)
        end0 = pos1 if pos1 >= 0 else total
        b0 = buf.find(b"band", pos0, end0)
        if b0 < 0:
            return None
        lines = buf[b0:end0].split(b"\n")
        bands = [i for i, l in enumerate(lines) if l.startswith(b"band")]
        if len(bands) != nb or bands[0] != 0:
            return None
        lpb = (bands[1] - bands[0]) if nb > 1 else len(lines)
        off = next((i + 1 for i in range(0, lpb) if lines[i].startswith(b"ion")), None)
        if off is None:
            return None
        starts = np.concatenate([[0], np.cumsum([len(l) + 1 for l in lines])])
        li = (off + np.arange(nb)[:, None] * lpb + np.arange(nion)[None, :]).ravel()      # (バンド, 原子) の順
        first = lines[li[0]].rstrip()
        width = len(first) - (first.rfind(b" ") + 1) + 1              # tot 欄の幅（前の空白 1 文字を含む）
        if any(len(lines[i].rstrip()) != len(first) for i in li):
            return None
        col0 = starts[li] + len(first) - width
        cols = (col0[:, None] + np.arange(width)[None, :]).ravel()
        region = int(cols.max()) + 1                                   # band 行の先頭から必要な範囲のバイト数
        last_band = int(starts[bands[-1]])                             # 最後の band 行の位置（位置合わせの確認用）
        blen = end0 - pos0
        raw = np.empty((nk, len(cols)), np.uint8)
        pos = pos0
        for ik in range(nk):
            if ik > 0:
                # 次の k点の塊は、ほぼ blen バイト先にある（k点の行の長さが少し違うことがある）
                lo = max(0, pos + blen - 256)
                q = buf.find(key, lo, min(total, pos + blen + 256))
                if q < 0:
                    q = buf.find(key, pos + 8)
                    if q < 0:
                        return None
                pos = q
            bk = buf.find(b"band", pos, min(total, pos + 2000))
            if bk < 0 or bk + region > total or buf[bk + last_band:bk + last_band + 4] != b"band":
                return None
            raw[ik] = np.frombuffer(buf, dtype=np.uint8, count=region, offset=bk)[cols]
        out = raw.view(f"S{width}").astype(np.float32).reshape(nk, nb, nion)
        # 念のため、最初の k点を通常の読み方と比べる
        chk = np.array([float(lines[i].rsplit(None, 1)[1]) for i in li[:min(len(li), 300)]], np.float32)
        if not np.allclose(out[0].ravel()[:len(chk)], chk, atol=1e-6):
            return None
        return out
    except Exception:
        return None


def read_procar_groups(path, groups, ngroups, nk_expected=None, progress=None):
    """PROCAR (LORBIT = 10/11) から、原子のグループ（元素）ごとに合計した重み tot を読む。
    groups[ion] = グループ番号。戻り値: w (nk, nb, ngroups) float32
    非共線計算では最初のブロック（全体）だけ、スピン分極計算では最初のスピンだけを使う。
    数 GB になるファイルなので、メモリに載せずに（mmap）k点ごとの塊を取り出し、数値の変換は
    numpy にまとめて任せる。"""
    import mmap
    import re
    groups = np.asarray(groups)
    nion = len(groups)
    G = np.zeros((nion, ngroups), np.float32)
    G[np.arange(nion), groups] = 1.0
    f = open(path, "rb")
    try:
        try:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            buf = f.read()                   # mmap できない場所（一部のネットワークドライブ）では全部読む
        m = re.search(rb"k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)", buf[:2000])
        if not m:
            raise ValueError("PROCAR のヘッダ（# of k-points …）を読み取れませんでした。")
        nk, nb, ni = (int(x) for x in m.groups())
        if ni != nion:
            raise ValueError(f"PROCAR の原子数 ({ni}) が POSCAR ({nion}) と一致しません。")
        if nk_expected is not None and nk != nk_expected:
            raise ValueError(f"PROCAR の k点数 ({nk}) が EIGENVAL ({nk_expected}) と一致しません。")
        key = b"k-point "
        w = np.empty((nk, nb, ngroups), np.float32)
        lay = None
        pos = buf.find(key, m.end())
        fast = _procar_fast(buf, pos, nk, nb, nion, key)
        if fast is not None:
            return fast @ G
        for ik in range(nk):
            if pos < 0:
                raise ValueError("PROCAR が途中で切れています。")
            nxt = buf.find(key, pos + 8)
            end = nxt if nxt >= 0 else len(buf)
            lines = buf[pos:end].split(b"\n")
            if lay is None:
                bands = [i for i, l in enumerate(lines) if l.startswith(b"band")]
                if not bands:
                    raise ValueError("PROCAR の形式を解釈できませんでした（band 行がありません）。")
                lpb = (bands[1] - bands[0]) if len(bands) > 1 else (len(lines) - bands[0])
                off = next((i - bands[0] + 1 for i in range(bands[0], bands[0] + lpb)
                            if lines[i].startswith(b"ion")), None)
                if off is None:
                    raise ValueError("PROCAR に ion の見出し行がありません（LORBIT = 11 で計算してください）。")
                lay = (bands[0], lpb, off, len(lines[bands[0] + off].split()))
            j0, lpb, off, ncol = lay
            blk = lines[j0:j0 + nb * lpb]
            if len(blk) < (nb - 1) * lpb + off + nion:
                raise ValueError("PROCAR が途中で切れています。")
            tot = np.empty((nion, nb), np.float32)
            for a in range(nion):
                rows = blk[off + a::lpb][:nb]
                try:
                    tot[a] = [float(r.rsplit(None, 1)[1]) for r in rows]
                except ValueError:           # 桁あふれ（'****'）など
                    tot[a] = [_ffloat(r.rsplit(None, 1)[1].decode()) for r in rows]
            tot = np.nan_to_num(tot)
            w[ik] = tot.T @ G
            pos = nxt
            if progress and ik % 100 == 0:
                progress(ik, nk)
        return w
    finally:
        try:
            buf.close()
        except Exception:
            pass
        f.close()


# ======================================================================
#  vasprun.xml（DOS / BAND）
# ======================================================================
def _rows_to_array(rows, dtype=np.float64):
    """数値の行のリスト → 配列。桁あふれ（'****'）は NaN。列の数がずれないよう行ごとにそろえる"""
    try:
        return np.array(" ".join(rows).split(), dtype=dtype)
    except ValueError:
        pass
    import re
    from collections import Counter
    cnt = Counter(len(r.split()) for r in rows[:2000] if "*" not in r)
    ncol = cnt.most_common(1)[0][0] if cnt else None
    toks = []
    for r in rows:
        if "*" in r:
            t = re.sub(r"\*+", " nan ", r).split()
            if ncol:
                t = (t + ["nan"] * ncol)[:ncol]
        else:
            t = r.split()
        toks.extend(t)
    try:
        return np.array(toks, dtype=dtype)
    except ValueError:
        return np.array([_ffloat(x) for x in toks], dtype=dtype)


def _xml_numbers(block, dtype=np.float64):
    """vasprun.xml の <set>…<r> 数値 </r>…</set> の塊から、数値だけを 1 次元配列で取り出す。
    タグを空白に置き換えて numpy でまとめて変換する（要素を 1 つずつたどるより桁違いに速い）。
    桁あふれ（'****'）があるときだけ、行ごとの変換に切り替える。"""
    import re
    import warnings
    i = block.find(b"<set")
    if i > 0:
        block = block[i:]
    if b"*" not in block:
        txt = re.sub(rb"<[^>]*>", b" ", block)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return np.fromstring(txt, dtype=dtype, sep=" ")
            except Exception:
                pass
    rows = [r.decode("ascii", "replace") for r in re.findall(rb"<r>([^<]*)</r>", block)]
    return _rows_to_array(rows, dtype)


def _between(data, a, b, start=0):
    """data の start 以降で、タグ a と b のあいだの部分（なければ None。b がなければ最後まで）"""
    i = data.find(a, start)
    if i < 0:
        return None
    j = data.find(b, i)
    return data[i + len(a):(j if j >= 0 else len(data))]


def read_vasprun(path, want=None, progress=None):
    """vasprun.xml から DOS・固有値・射影を読む。
    want: 読むものの組。"eig"（固有値）, "dos"（全 DOS）, "pdos"（部分 DOS）, "proj"（バンドの射影）。
          None なら全部。使わない部分（数百 MB になる射影など）は読み飛ばせるので、
          画面では最初に必要なものだけ読み、PDOS / PBAND はチェックを入れたときに読む。
    射影（PDOS / PBAND）は最初のスピン成分だけを使う（非共線計算では全体、非磁性では唯一の成分）。"""
    import re
    import xml.etree.ElementTree as ET
    out = {"symbols": [], "lattice": None, "frac": None, "efermi": None, "kpts": None, "weights": None,
           "kgen": {}, "eig": None, "occ": None, "dos_e": None, "dos_total": None, "dos_partial": None,
           "dos_orbitals": [], "proj": None, "proj_orbitals": [], "incar": {}, "has_pdos": False,
           "has_proj": False}
    with open(path, "rb") as f:
        data = f.read()
    c0 = data.find(b"<calculation>")
    if c0 < 0:
        raise ValueError("vasprun.xml に計算結果（<calculation>）がありません。計算が始まる前のファイルではありませんか。")
    calc = data.rfind(b"<calculation>")

    # ---- 見出し部分（INCAR, k点, 原子, 構造）: ここは小さいので XML として読む ----
    try:
        head = ET.fromstring(data[:c0] + b"</modeling>")
    except ET.ParseError as e:
        raise ValueError(f"vasprun.xml の先頭部分を解釈できませんでした: {e}")
    wanted = {"ISPIN", "LSORBIT", "LNONCOLLINEAR", "NELECT", "NBANDS", "LORBIT", "NEDOS", "SYSTEM"}
    for el in head.iter("i"):
        nm = el.get("name")
        if nm in wanted:
            out["incar"][nm] = (el.text or "").strip()
    for rc in head.findall("atominfo/array[@name='atoms']/set/rc"):
        out["symbols"].append(rc.find("c").text.strip())
    kp = head.find("kpoints")
    if kp is not None:
        gen = kp.find("generation")
        if gen is not None:
            out["kgen"]["param"] = gen.get("param", "")
            d = gen.find("*[@name='divisions']")
            if d is not None and d.text:
                out["kgen"]["divisions"] = [int(float(x)) for x in d.text.split()]
        kl = kp.find("varray[@name='kpointlist']")
        if kl is not None:
            out["kpts"] = np.array([[float(x) for x in v.text.split()] for v in kl.findall("v")])
        wl = kp.find("varray[@name='weights']")
        if wl is not None:
            out["weights"] = np.array([float(v.text.split()[0]) for v in wl.findall("v")])

    def structure(el):
        b = el.find("crystal/varray[@name='basis']")
        p = el.find("varray[@name='positions']")
        if b is not None and p is not None:
            out["lattice"] = np.array([[float(x) for x in v.text.split()] for v in b.findall("v")])
            out["frac"] = np.array([[float(x) for x in v.text.split()] for v in p.findall("v")])

    for el in head.findall("structure"):
        structure(el)
    i = data.rfind(b'<structure name="finalpos"')
    if i >= 0:
        j = data.find(b"</structure>", i)
        if j >= 0:
            try:
                structure(ET.fromstring(data[i:j + 12]))
            except Exception:
                pass
    nion = len(out["symbols"])
    nk = 0 if out["kpts"] is None else len(out["kpts"])
    if want is None:
        want = ("eig", "dos", "pdos", "proj")
    warn = []

    i = data.rfind(b'<i name="efermi">')
    if i >= 0:
        v = _ffloat(data[i + 17:i + 80].split(b"<")[0].decode().strip() or "nan")
        out["efermi"] = None if np.isnan(v) else v

    # ---- 固有値 ----
    blk = _between(data, b"<eigenvalues>", b"</eigenvalues>", calc)
    if "eig" in want and blk is not None and nk:
        a = _xml_numbers(blk)
        nspin = max(1, blk.count(b'comment="spin'))
        if a.size % (nspin * nk * 2) == 0 and a.size:
            a = a.reshape(nspin, nk, -1, 2)
            out["eig"], out["occ"] = a[..., 0], a[..., 1]
        else:
            warn.append("固有値のデータ数が k点数と合いません（計算が途中で止まっていませんか）。")
    if progress:
        progress("固有値")

    # ---- DOS ----
    dos = _between(data, b"<dos>", b"</dos>", calc) if ("dos" in want or "pdos" in want) else None
    if dos is not None:
        tot = _between(dos, b"<total>", b"</total>")
        if tot is not None:
            a = _xml_numbers(tot)
            ns = max(1, tot.count(b'comment="spin'))
            if a.size % (ns * 3) == 0 and a.size:
                a = a.reshape(ns, -1, 3)                       # (スピン, NEDOS, [E, DOS, 積分])
                out["dos_e"], out["dos_total"] = a[0, :, 0], a[:, :, 1]
        par = _between(dos, b"<partial>", b"</partial>") if "pdos" in want else None
        if par is not None and nion and out["dos_e"] is not None:
            hdr = par[:max(0, par.find(b"<set"))]
            names = [x.decode().strip() for x in re.findall(rb"<field>([^<]*)</field>", hdr)]
            names = [x for x in names if x and x != "energy"]
            # 非共線計算（spin 1–4 = 全体, mx, my, mz）: spin 1（全体）だけ。
            # ISPIN = 2（spin 1, 2 = 上向き, 下向き）: 2 つの和（全 DOS と同じく 2 スピンの和にそろえる）
            pieces = re.findall(rb'<set comment="spin\s*1">(.*?)</set>', par, re.S)
            a = _xml_numbers(b"<set>" + b" ".join(pieces), np.float32)
            ncl_p = re.search(rb'<set comment="spin\s*4">', par) is not None
            if not ncl_p and re.search(rb'<set comment="spin\s*2">', par) is not None:
                pieces2 = re.findall(rb'<set comment="spin\s*2">(.*?)</set>', par, re.S)
                a2 = _xml_numbers(b"<set>" + b" ".join(pieces2), np.float32)
                if a2.size == a.size:
                    nedos_, ncol_ = len(out["dos_e"]), 1 + len(names)
                    if a.size == nion * nedos_ * ncol_:
                        a = a.reshape(nion, nedos_, ncol_)
                        a2 = a2.reshape(nion, nedos_, ncol_)
                        a = np.concatenate([a[:, :, :1], a[:, :, 1:] + a2[:, :, 1:]], axis=2).ravel()
                else:
                    warn.append("部分状態密度の 2 つのスピンのデータ数が合いません。")
            nedos, ncol = len(out["dos_e"]), 1 + len(names)
            if a.size == nion * nedos * ncol and a.size:
                a = a.reshape(nion, nedos, ncol)               # (原子, NEDOS, 列)
                out["dos_partial"] = np.ascontiguousarray(a[:, :, 1:])
                out["dos_orbitals"] = names
            else:
                warn.append("部分状態密度のデータ数が合わないため、PDOS は読み込めませんでした。")
    if progress:
        progress("DOS")

    # ---- 射影（PBAND 用）----
    out["has_proj"] = data.find(b"<projected>", calc) >= 0
    out["has_pdos"] = data.find(b"<partial>", calc) >= 0
    pj = _between(data, b"<projected>", b"</projected>", calc) if "proj" in want else None
    if pj is not None and out["eig"] is not None and nion:
        e = pj.find(b"</eigenvalues>")
        arr = pj[e + 14:] if e >= 0 else pj
        hdr = arr[:max(0, arr.find(b"<set"))]
        names = [x.decode().strip() for x in re.findall(rb"<field>([^<]*)</field>", hdr)]
        m1 = re.search(rb'<set comment="spin\s*1">', arr)
        m2 = re.search(rb'<set comment="spin\s*2">', arr)
        if m1 is not None and names:
            sub = arr[m1.end():(m2.start() if m2 is not None else len(arr))]
            del arr
            a = _xml_numbers(b"<set>" + sub, np.float32)
            nb = out["eig"].shape[2]
            if a.size == nk * nb * nion * len(names):
                out["proj"] = a.reshape(nk, nb, nion, len(names))
                out["proj_orbitals"] = names
            else:
                warn.append("射影のデータ数が合わないため、PBAND は読み込めませんでした。")
    # 桁あふれ（'***'）で読めなかった数値の個数
    out["n_overflow"] = {k: int(np.isnan(out[k]).sum()) for k in ("eig", "dos_total", "dos_partial", "proj")
                         if out.get(k) is not None and np.isnan(out[k]).any()}
    if not data[-200:].rstrip().endswith(b"</modeling>"):
        warn.append("vasprun.xml が途中で終わっています。読めた部分だけを使います。")
    if warn:
        out["warning"] = " ".join(warn)
    return out


def orbital_l(name):
    """軌道名（s, py, dxy, x2-y2, fxyz, …）→ 's' / 'p' / 'd' / 'f'"""
    n = name.strip().lower()
    if n.startswith("x2") or n.startswith("dx"):
        return "d"
    return n[0] if n and n[0] in "spdf" else "?"


def species_groups(symbols):
    """原子ごとの元素記号 → (元素のリスト（出現順）, 原子ごとのグループ番号)"""
    sp = []
    for s in symbols:
        if s not in sp:
            sp.append(s)
    return sp, np.array([sp.index(s) for s in symbols], int)


def math_label(lb):
    """k点ラベル → matplotlib 用（GAMMA / \\Gamma / G → Γ、X_2 → 下付き）"""
    s = lb.strip().replace("$", "").lstrip("\\")
    if not s:
        return ""
    if s.upper() in ("GAMMA", "G", "Γ"):
        return "Γ"
    base, sub = label_parts(s)
    if base.upper() == "GAMMA":
        base = "Γ"
    return f"{base}$_{{{sub}}}$" if sub else base


def band_path_axis(kpts, lattice, kgen, kfile=None, point_labels=None):
    """バンド図の横軸を作る。戻り値: (距離 x (nk,), 目盛り位置, 目盛りラベル, 区間 [(始, 終), ...])
    ・line-mode（vasprun.xml の listgenerated）: 1 区間あたり divisions 点。名前は KPOINTS から
    ・明示的な k点リスト: point_labels（k点ごとのラベル）があればその位置に目盛り
    区間のつなぎ目が不連続なら距離は進めず、ラベルを "A|B" とする。"""
    B = reciprocal_2pi(lattice)
    kc = np.asarray(kpts) @ B
    nk = len(kc)
    div = None
    if kgen.get("param") == "listgenerated" and kgen.get("divisions"):
        div = int(kgen["divisions"][0])
    if div and (nk % div != 0 or div < 2):
        div = None
    breaks, marks = set(), []
    if div:
        starts = list(range(0, nk, div))
        breaks = set(starts[1:])
        names = kfile["segments"] if (kfile and kfile.get("kind") == "line"
                                      and len(kfile.get("segments", [])) == len(starts)) else None
        for j, s0 in enumerate(starts):
            marks.append((s0, math_label(names[j][0]) if names else ""))
            marks.append((s0 + div - 1, math_label(names[j][1]) if names else ""))
    else:
        pl = point_labels if (point_labels is not None and len(point_labels) == nk) else None
        for i in range(1, nk):
            if np.linalg.norm(kc[i] - kc[i - 1]) < 1e-9 or (pl and pl[i - 1] and pl[i]):
                breaks.add(i)
        if pl:
            marks = [(i, math_label(l)) for i, l in enumerate(pl) if l]
        else:
            b = sorted(breaks)
            marks = [(0, "")] + [(i, "") for j in b for i in (j - 1, j)] + [(nk - 1, "")]
    x = np.zeros(nk)
    for i in range(1, nk):
        x[i] = x[i - 1] + (0.0 if i in breaks else np.linalg.norm(kc[i] - kc[i - 1]))
    b = [0] + sorted(breaks) + [nk]
    segs = [(b[j], b[j + 1] - 1) for j in range(len(b) - 1)]
    ticks, labels = [], []
    for i, lab in marks:
        if ticks and abs(ticks[-1] - x[i]) < 1e-9:
            if lab and labels[-1] and lab != labels[-1]:
                labels[-1] = f"{labels[-1]}|{lab}"
            elif lab and not labels[-1]:
                labels[-1] = lab
        else:
            ticks.append(float(x[i]))
            labels.append(lab)
    return x, ticks, labels, segs


def total_dos(vr):
    """全状態密度 (nedos,)。非共線計算（4 成分: 全体, mx, my, mz）は全体だけ、ISPIN = 2 は 2 スピンの和"""
    t = vr["dos_total"]
    ncl = vr["incar"].get("LNONCOLLINEAR", "F").strip().upper().startswith("T") or len(t) == 4
    return t[0] if (ncl or len(t) == 1) else t[:2].sum(axis=0)


# ======================================================================
#  フェルミ面: VASPKIT 263/264 と同じ FermiSurfer 形式への変換
# ======================================================================
def unfold_kmesh_atoms(st, mesh, kpts):
    """既約 k点（分率座標）を Γ中心メッシュ全体に展開する対応表を作る。原子の移り先も記録する。
    戻り値 (mapping, opidx, pinv, rots):
      mapping (n1*n2*n3,) … 格子点 (i1, i2, i3)（i3 が最も速く回る）→ 既約 k点の番号
      opidx   (n1*n2*n3,) … その格子点を作るのに使った対称操作の番号
      pinv    (操作数, 原子数) … pinv[op, b] = 操作 op で原子 b に移ってくる元の原子の番号
      rots    (操作数, 3, 3) … 操作 op の回転（デカルト座標。r' = rots[op] r）。軌道ごとの重みの展開に使う
    結晶の対称操作 {R|t}（spglib）と時間反転 k → -k を使う。
    {R|t} は原子 a を原子 p(a) に移し、k を k' = R^{-T} k に移す。このとき、状態の原子ごとの重みは
        w_{p(a)}(k') = w_a(k)
    なので、展開先の格子点での原子 b の重みは、既約 k点での原子 pinv[op, b] の重みになる。
    （元素全体の和はこの入れ替えで変わらないので、元素単位なら入れ替えを無視しても正しい。
      原子 1 個ずつを見るときは、この入れ替えが必要。）時間反転は原子を動かさない。"""
    import spglib
    N = np.array(mesh, int)
    ntot = int(N.prod())
    kp = np.asarray(kpts, float)
    frac = np.mod(st.frac, 1.0)
    nat = len(frac)
    nums = np.array(st.spglib_cell()[2])
    last = None
    for symprec in (1e-5, 1e-3, 1e-2):
        sym = spglib.get_symmetry(st.spglib_cell(), symprec=symprec)
        if sym is None:
            continue
        rots = np.asarray(sym.rotations if hasattr(sym, "rotations") else sym["rotations"])
        trans = np.asarray(sym.translations if hasattr(sym, "translations") else sym["translations"])
        tol = max(10 * symprec, 1e-3)                    # 原子位置の一致の許容値 [Å]
        mapping = -np.ones(ntot, int)
        opidx = np.zeros(ntot, int)
        pinv, rcs, ok_all = [], [], True
        LT = np.asarray(st.lattice, float).T
        LTi = np.linalg.inv(LT)
        idx = np.arange(len(kp))
        for R, t in zip(rots, trans):
            # 原子の移り先 p(a)
            img = frac @ np.asarray(R, float).T + t
            dlt = img[:, None, :] - frac[None, :, :]
            dlt = (dlt - np.rint(dlt)) @ st.lattice
            dist = np.linalg.norm(dlt, axis=2)
            dist[nums[:, None] != nums[None, :]] = np.inf
            p_ = dist.argmin(axis=1)
            if dist[np.arange(nat), p_].max() > tol or len(set(p_.tolist())) != nat:
                ok_all = False
                break
            inv = np.empty(nat, int)
            inv[p_] = np.arange(nat)
            S = np.rint(np.linalg.inv(np.asarray(R, float))).astype(int)     # 行ベクトル k に対して k' = k S
            for sign in (1, -1):
                op = len(pinv)
                pinv.append(inv)
                rcs.append(LT @ np.asarray(R, float) @ LTi)
                g = (kp @ (sign * S)) * N
                gi = np.rint(g)
                ok = np.abs(g - gi).max(axis=1) < 2e-3
                gi = gi[ok].astype(int) % N
                flat = (gi[:, 0] * N[1] + gi[:, 1]) * N[2] + gi[:, 2]
                new = mapping[flat] < 0
                mapping[flat[new]] = idx[ok][new]
                opidx[flat[new]] = op
        if not ok_all:
            continue
        last = mapping
        if (mapping >= 0).all():
            return mapping, opidx, np.array(pinv), np.array(rcs)
    missing = int((last < 0).sum()) if last is not None else ntot
    raise ValueError(f"k点メッシュ {tuple(N)} のうち {missing} 点を、計算された k点から対称操作で再現できませんでした。"
                     "KPOINTS のメッシュ、POSCAR（計算に使った構造）、Γ中心メッシュかどうかを確認してください。")


def unfold_kmesh(st, mesh, kpts):
    """既約 k点 → メッシュ全体の対応表だけを返す（unfold_kmesh_atoms の 1 番目）"""
    return unfold_kmesh_atoms(st, mesh, kpts)[0]


def site_labels(st):
    """原子ごとの名前（元素ごとに POSCAR の順で番号を付ける: Cu1, Cu2, …。VASPKIT の ATOM_LABEL と同じ）"""
    cnt, out = {}, []
    for sy in st.symbols:
        cnt[sy] = cnt.get(sy, 0) + 1
        out.append(f"{sy}{cnt[sy]}")
    return out


# ---- 軌道（PROCAR の lm 成分）ごとの重み ----
# VASP の PROCAR (LORBIT = 11/12) の列の並び。実球面調和関数で、軸は計算セルのデカルト座標 x, y, z
ORBITAL_SETS = {1: ("py", "pz", "px"),
                2: ("dxy", "dyz", "dz2", "dxz", "x2-y2"),
                3: ("fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3")}
_ORB_ALIAS = {"dx2-y2": "x2-y2", "dx2": "x2-y2"}


def orbital_l_of(name):
    """PROCAR の列名 → 方位量子数 l（s, p, d, f の 0–3）。分からなければ None"""
    n = _ORB_ALIAS.get(name, name)
    if n in ("s", "p", "d", "f"):
        return "spdf".index(n)
    for l, names in ORBITAL_SETS.items():
        if n in names:
            return l
    return None


def _real_harmonics(l, r):
    """実球面調和関数（共通の規格化定数を除く）を VASP の PROCAR の列の順で返す。r (n, 3) → (n, 2l+1)"""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r2 = x * x + y * y + z * z
    if l == 1:
        return np.stack([y, z, x], axis=1)
    if l == 2:
        s3 = np.sqrt(3.0)
        return np.stack([s3 * x * y, s3 * y * z, 0.5 * (3 * z * z - r2), s3 * x * z, 0.5 * s3 * (x * x - y * y)], axis=1)
    if l == 3:
        a, b, c = np.sqrt(5.0 / 8.0), np.sqrt(15.0), np.sqrt(3.0 / 8.0)
        return np.stack([a * y * (3 * x * x - y * y), b * x * y * z, c * y * (5 * z * z - r2),
                         0.5 * z * (5 * z * z - 3 * r2), c * x * (5 * z * z - r2),
                         0.5 * b * z * (x * x - y * y), a * x * (x * x - 3 * y * y)], axis=1)
    raise ValueError(l)


def orbital_rotation(Rc, l):
    """回転 Rc（デカルト, 3×3。反転を含んでよい）に対する実球面調和関数の変換行列 A (2l+1, 2l+1):
        Y_m(Rc u) = Σ_m' A[m, m'] Y_m'(u)"""
    if l == 0:
        return np.ones((1, 1))
    rng = np.random.default_rng(12345)
    u = rng.normal(size=(60, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    X, *_ = np.linalg.lstsq(_real_harmonics(l, u), _real_harmonics(l, u @ np.asarray(Rc, float).T), rcond=None)
    return X.T


def procar_orbital_names(path):
    """PROCAR の列名（ion と tot を除く）。例: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'x2-y2']"""
    with open(path, "rb") as f:
        head = f.read(20000)
    for line in head.split(b"\n"):
        if line.startswith(b"ion"):
            t = line.decode("ascii", "replace").split()
            return t[1:-1] if t[-1] == "tot" else t[1:]
    return []


def orbital_matrices(names, rots):
    """対称操作ごとの、軌道の列についての変換行列 A (操作数, 列数, 列数)。
    l の全成分（p なら py, pz, px）がそろっている組は回転で互いに混ざる。そろっていない列や
    LORBIT = 10 の列（s, p, d, f）は混ざらないものとして扱う（単位行列）。"""
    n = len(names)
    canon = [_ORB_ALIAS.get(x, x) for x in names]
    A = np.tile(np.eye(n), (len(rots), 1, 1))
    for l, group in ORBITAL_SETS.items():
        if all(g in canon for g in group):
            ix = [canon.index(g) for g in group]
            for o, Rc in enumerate(rots):
                A[o][np.ix_(ix, ix)] = orbital_rotation(Rc, l)
    return A


def orbital_unfold_coef(A, mask, ops=None):
    """選んだ軌道の組 S（mask）の重みを、既約 k点から対称操作で展開するときの係数。
    操作 {R|t} で k → k' に移した状態の、軌道 m の重みは
        w'_m = Σ_{m', m''} A[m, m'] A[m, m''] ρ[m', m'']      （ρ は既約 k点での軌道の密度行列）
    PROCAR にあるのは対角成分 ρ[m', m'] = w_m' だけなので、
        Σ_{m∈S} w'_m ≈ Σ_m' coef[m'] w_m' ,   coef[m'] = Σ_{m∈S} A[m, m']²
    とする。行列 Aᵀ P_S A の非対角成分が 0 なら（S が回転で閉じている、または軌道が入れ替わるだけ）厳密。
    戻り値 (coef (操作数, 列数), 非対角成分の最大値)。ops を渡すと、その操作だけで最大値を調べる"""
    mask = np.asarray(mask, bool)
    As = A[:, mask, :]
    M = np.einsum("omi,omj->oij", As, As)
    coef = np.einsum("oii->oi", M).copy()
    off = M - np.einsum("oi,ij->oij", coef, np.eye(M.shape[1]))
    sel = slice(None) if ops is None else np.asarray(ops, int)
    return coef.astype(np.float32), float(np.abs(off[sel]).max()) if off[sel].size else 0.0


def orbital_label(names, mask):
    """選んだ軌道の表示名。l の全成分がそろっていれば p, d のようにまとめる"""
    mask = np.asarray(mask, bool)
    canon = [_ORB_ALIAS.get(x, x) for x in names]
    out, used = [], set()
    for i, nm in enumerate(canon):
        if not mask[i] or i in used:
            continue
        l = orbital_l_of(nm)
        grp = ORBITAL_SETS.get(l, ())
        if nm in grp and all(g in canon and mask[canon.index(g)] for g in grp):
            out.append("spdf"[l])
            used.update(canon.index(g) for g in grp)
        else:
            out.append({"x2-y2": "dx2-y2"}.get(nm, nm))
    return "+".join(out)


def _procar_fast_orb(buf, pos0, nk, nb, nion, key, progress=None, sel=None, out=None):
    """_procar_fast と同じ考え方で、全部の列（軌道の列と末尾の tot）を読む。
    戻り値 (nk, nb, nion, 軌道の列数 + 1) int16（値 × 1000。PROCAR は小数 3 桁なので情報は落ちない）。
    最後の列が tot。使えなければ None
    sel（バンド番号のリスト）を渡すと、そのバンドの行だけを読む（ファイルの位置を計算して必要な所だけ読むので、
    ディスクから読む量も減る）。out（(nk, nb, nion, 列数 + 1) int16）を渡すと、そこに書き込む"""
    import re
    try:
        if pos0 < 0:
            return None
        total = len(buf)
        pos1 = buf.find(key, pos0 + 8)
        end0 = pos1 if pos1 >= 0 else total
        b0 = buf.find(b"band", pos0, end0)
        if b0 < 0:
            return None
        lines = buf[b0:end0].split(b"\n")
        bands = [i for i, l in enumerate(lines) if l.startswith(b"band")]
        if len(bands) != nb or bands[0] != 0:
            return None
        lpb = (bands[1] - bands[0]) if nb > 1 else len(lines)
        off = next((i + 1 for i in range(0, lpb) if lines[i].startswith(b"ion")), None)
        if off is None:
            return None
        starts = np.concatenate([[0], np.cumsum([len(l) + 1 for l in lines])])
        sel = np.arange(nb) if sel is None else np.asarray(sorted(sel), int)
        li = (off + sel[:, None] * lpb + np.arange(nion)[None, :]).ravel()
        first = lines[li[0]].rstrip()
        ends = [m.end() for m in re.finditer(rb"\S+", first)]
        ncol = len(ends) - 1                                           # 先頭の原子番号を除く（末尾の tot を含む）
        if ncol < 2:
            return None
        width = ends[1] - ends[0]
        if any(ends[j + 1] - ends[j] != width for j in range(ncol)):
            return None
        if any(len(lines[i].rstrip()) != len(first) for i in li):
            return None
        fld = ends[0] + np.arange(ncol) * width
        cols = (starts[li][:, None, None] + fld[None, :, None] + np.arange(width)[None, None, :]).ravel()
        region = int(cols.max()) + 1
        last_band = int(starts[bands[-1]])
        blen = end0 - pos0
        if out is None:
            out = np.zeros((nk, nb, nion, ncol), np.int16)
        elif out.shape != (nk, nb, nion, ncol):
            return None
        pos = pos0
        for ik in range(nk):
            if ik > 0:
                lo = max(0, pos + blen - 256)
                q = buf.find(key, lo, min(total, pos + blen + 256))
                if q < 0:
                    q = buf.find(key, pos + 8)
                    if q < 0:
                        return None
                pos = q
            bk = buf.find(b"band", pos, min(total, pos + 2000))
            if bk < 0 or bk + region > total or buf[bk + last_band:bk + last_band + 4] != b"band":
                return None
            raw = np.frombuffer(buf, dtype=np.uint8, count=region, offset=bk)[cols]
            v = raw.view(f"S{width}").astype(np.float32)
            out[ik, sel] = np.rint(v * 1000.0).astype(np.int16).reshape(len(sel), nion, ncol)
            if progress and ik % 50 == 0:
                progress(ik, nk)
        chk = np.array([[float(t) for t in lines[i].split()[1:]] for i in li[:min(len(li), 100)]], np.float32)
        if not np.allclose(out[0, sel].reshape(-1, ncol)[:len(chk)] * 1e-3, chk, atol=1e-6):
            return None
        return out
    except Exception:
        return None


def procar_header(path):
    """PROCAR の (k点数, バンド数, 原子数)"""
    import re
    with open(path, "rb") as f:
        head = f.read(2000)
    m = re.search(rb"k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)", head)
    if not m:
        raise ValueError("PROCAR のヘッダ（# of k-points …）を読み取れませんでした。")
    return tuple(int(x) for x in m.groups())


def read_procar_orbitals(path, nion, nk_expected=None, progress=None, bands=None, out=None):
    """PROCAR から、原子ごと・軌道（列）ごとの重みと tot を 1 回で読む。
    戻り値 (列名のリスト, w (nk, nb, nion, 列数 + 1) int16)。w は値 × 1000、最後の列が tot。
    非共線計算では最初のブロック（全体）、スピン分極計算では最初のスピンだけを使う。
    bands（0 始まりの番号のリスト）を渡すとそのバンドだけ読み、out を渡すとそこに書く（他のバンドはそのまま）"""
    import mmap
    import re
    names = procar_orbital_names(path)
    if not names:
        raise ValueError("PROCAR に ion の見出し行がありません（LORBIT = 11 で計算してください）。")
    ncol = len(names)
    f = open(path, "rb")
    try:
        try:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            buf = f.read()
        m = re.search(rb"k-points:\s*(\d+)\s+# of bands:\s*(\d+)\s+# of ions:\s*(\d+)", buf[:2000])
        if not m:
            raise ValueError("PROCAR のヘッダ（# of k-points …）を読み取れませんでした。")
        nk, nb, ni = (int(x) for x in m.groups())
        if ni != nion:
            raise ValueError(f"PROCAR の原子数 ({ni}) が POSCAR ({nion}) と一致しません。")
        if nk_expected is not None and nk != nk_expected:
            raise ValueError(f"PROCAR の k点数 ({nk}) が EIGENVAL ({nk_expected}) と一致しません。")
        key = b"k-point "
        pos = buf.find(key, m.end())
        fast = _procar_fast_orb(buf, pos, nk, nb, nion, key, progress, sel=bands, out=out)
        if fast is not None:
            return names, fast
        w = np.zeros((nk, nb, nion, ncol + 1), np.int16) if out is None else out
        blist = range(nb) if bands is None else sorted(bands)
        lay = None
        for ik in range(nk):
            if pos < 0:
                raise ValueError("PROCAR が途中で切れています。")
            nxt = buf.find(key, pos + 8)
            end = nxt if nxt >= 0 else len(buf)
            lines = buf[pos:end].split(b"\n")
            if lay is None:
                bands = [i for i, l in enumerate(lines) if l.startswith(b"band")]
                if not bands:
                    raise ValueError("PROCAR の形式を解釈できませんでした（band 行がありません）。")
                lpb = (bands[1] - bands[0]) if len(bands) > 1 else (len(lines) - bands[0])
                off = next((i - bands[0] + 1 for i in range(bands[0], bands[0] + lpb)
                            if lines[i].startswith(b"ion")), None)
                if off is None:
                    raise ValueError("PROCAR に ion の見出し行がありません（LORBIT = 11 で計算してください）。")
                lay = (bands[0], lpb, off)
            j0, lpb, off = lay
            for b in blist:
                rows = lines[j0 + b * lpb + off:j0 + b * lpb + off + nion]
                if len(rows) < nion:
                    raise ValueError("PROCAR が途中で切れています。")
                try:
                    a = np.array(b" ".join(rows).split(), np.float32).reshape(nion, -1)
                except ValueError:           # 桁あふれ（'****'）など
                    a = np.array([[_ffloat(t.decode()) for t in r.split()] for r in rows], np.float32)
                w[ik, b] = np.rint(np.nan_to_num(a[:, 1:2 + ncol]) * 1000.0).astype(np.int16)
            pos = nxt
            if progress and ik % 50 == 0:
                progress(ik, nk)
        return names, w
    finally:
        try:
            buf.close()
        except Exception:
            pass
        f.close()


class FermiWeights:
    """フェルミ面に付ける重み。バンド番号 b を指定すると、メッシュ全体での重み (n1, n2, n3, チャンネル数) を返す。
    チャンネル = 原子の組（元素全体や、原子 1 個）。cmat (原子数, チャンネル数) の各列が 1 つの組。
    既約 k点での原子ごとの重み（PROCAR）を、対称操作による原子の入れ替えを含めて展開する。
    orb（列ごとの True/False）を渡すと、選んだ軌道だけの重みにする。そのときは軌道の混ざり方も
    対称操作ごとに考慮する（orbital_unfold_coef）。self.exact が False なら、選んだ軌道の組は
    対称操作で他の軌道と混ざるので、展開した値は近似。"""

    def __init__(self, d, cmat, orb=None):
        self.d = d
        self.cmat = np.asarray(cmat, np.float32)
        self.orb = None if orb is None else np.asarray(orb, bool)
        self.exact, self.offdiag = True, 0.0
        if self.orb is not None:
            self.coef, self.offdiag = orbital_unfold_coef(d["orb_A"], self.orb, np.unique(d["opidx"]))
            self.exact = self.offdiag < 1e-6

    def __getitem__(self, b):
        d = self.d
        if self.orb is None:
            wa = d["Wir"][d["mapping"], b]                               # (格子点, 原子)
        else:
            mp, op = d["mapping"], d["opidx"]
            wa = np.empty((len(mp), d["Worb"].shape[2]), np.float32)
            for i0 in range(0, len(mp), 20000):                          # メモリを使いすぎないよう分けて計算
                sl = slice(i0, i0 + 20000)
                wo = d["Worb"][mp[sl], b].astype(np.float32)             # (格子点, 原子, 列)
                wa[sl] = np.einsum("gam,gm->ga", wo, self.coef[op[sl]]) * np.float32(1e-3)
        wa = np.take_along_axis(wa, d["pinv"][d["opidx"]], axis=1)       # 原子の入れ替え
        return (wa @ self.cmat).reshape(*d["mesh"], self.cmat.shape[1])


def load_fermi_orbitals(d, progress=None):
    """軌道ごとの重み d["Worb"]（load_fermi_weights で tot と一緒に読む）"""
    if d.get("Worb") is None:
        load_fermi_weights(d, progress)
    return d["Worb"]


def read_frmsf(path):
    """FermiSurfer 形式 (.frmsf) を読む。VASPKIT 263/264/265 やこのアプリが書いたファイル。
    戻り値は build_fermi_data と同じ形の dict（構造や PROCAR の情報はない）。
    E はファイルの値そのまま（E − E_F）なので efermi = 0。色のデータがあれば frmsf_W (nb, n1, n2, n3, K)"""
    with open(path, "rb") as f:
        head = [f.readline() for _ in range(6)]
        rest = f.read()
    try:
        mesh = tuple(int(x) for x in head[0].split()[:3])
        gtype = int(head[1].split()[0])
        nb = int(head[2].split()[0])
        B = np.array([[float(x) for x in h.split()[:3]] for h in head[3:6]])
    except Exception:
        raise ValueError("frmsf のヘッダ（メッシュ・バンド数・逆格子ベクトル）を読み取れませんでした。")
    if gtype != 1:
        raise ValueError(f"この frmsf は k点メッシュの種類が {gtype} です。Γ を含むメッシュ（2 行目が 1。"
                         "VASPKIT やこのアプリが書く形式）にだけ対応しています。")
    if min(mesh) < 1 or nb < 1 or B.shape != (3, 3) or abs(np.linalg.det(B)) < 1e-12:
        raise ValueError("frmsf のヘッダの値が正しくありません。")
    try:
        a = np.array(rest.split(), np.float32)
    except ValueError:
        a = np.array([_ffloat(t.decode("ascii", "replace")) for t in rest.split()], np.float32)
        a = np.nan_to_num(a)
    n = int(np.prod(mesh))
    blocks = len(a) / (nb * n)
    if len(a) < nb * n or abs(blocks - round(blocks)) > 1e-9 or round(blocks) > 4:
        raise ValueError(f"frmsf のデータの数 ({len(a)}) が、メッシュ {mesh} × バンド数 {nb} と合いません。")
    blocks = int(round(blocks))
    a = a.reshape(blocks, nb, *mesh)
    W = None
    if blocks > 1:
        W = np.ascontiguousarray(np.moveaxis(a[1:], 0, -1))        # (nb, n1, n2, n3, K)
    return {"st": None, "mesh": mesh, "B": B, "efermi": 0.0, "efermi_src": "frmsf は E − E_F で書かれている",
            "E": np.ascontiguousarray(a[0]), "W": None, "species": [], "nk_ir": n, "nspin": 1, "note": "",
            "structure_file": "", "mapping": None, "has_procar": False, "folder": os.path.dirname(os.path.abspath(path)),
            "kpts_ir": None, "Wir": None, "frmsf_W": W, "frmsf_path": path}


def build_fermi_data(folder, efermi=None, use_procar=True, progress=None):
    """計算フォルダから FermiSurfer 形式に相当するデータを作る（VASPKIT 263 / 264 と同じ入力ファイル）。
      POSCAR（なければ CONTCAR）, KPOINTS（1 行目にメッシュ）, EIGENVAL, DOSCAR（E_F。なければ OUTCAR）,
      PROCAR（元素ごとの寄与。なくてもよい）
    戻り値 dict: mesh, B（2π を含む逆格子, 1/Å）, efermi, E (nb, n1, n2, n3)（生の固有値）,
                 W (nb, n1, n2, n3, 元素数) または None, species, 情報"""
    def p(name):
        return os.path.join(folder, name)

    def need(names, what):
        for n in names:
            if os.path.isfile(p(n)):
                return p(n)
        raise FileNotFoundError(f"{what} が見つかりません（{' / '.join(names)}）。")

    pos = need(["POSCAR", "CONTCAR"], "構造ファイル")
    st = parse_poscar(_read_text(pos))
    kfile = read_kpoints_file(need(["KPOINTS"], "KPOINTS"))
    if not kfile.get("mesh"):
        raise ValueError("KPOINTS から k点メッシュを読み取れませんでした。\n"
                         "VASPKIT 261 または VASP Input Generator の「フェルミ面 (FERMI)」で作った KPOINTS"
                         "（1 行目にメッシュが書かれている）か、Γ中心の自動メッシュを使ってください。")
    if not kfile.get("gamma", True):
        raise ValueError("Monkhorst-Pack（Γ からずれた）メッシュには対応していません。Γ中心メッシュで計算してください。")
    mesh = tuple(int(x) for x in kfile["mesh"])
    ev = read_eigenval(need(["EIGENVAL"], "EIGENVAL"))
    src = ""
    if efermi is None:
        if os.path.isfile(p("DOSCAR")):
            try:
                efermi, src = read_doscar_efermi(p("DOSCAR")), "DOSCAR"
            except Exception:
                efermi = None
        if efermi is None and os.path.isfile(p("OUTCAR")):
            efermi, src = read_outcar_efermi(p("OUTCAR")), "OUTCAR"
        if efermi is None:
            raise FileNotFoundError("フェルミエネルギーを読めませんでした（DOSCAR または OUTCAR が必要です）。")
    else:
        src = "手入力"
    mapping = unfold_kmesh(st, mesh, ev["kpts"])
    nb = ev["eig"].shape[2]
    E = ev["eig"][0][mapping].T.reshape(nb, *mesh)            # (nb, n1, n2, n3)
    W = None
    note = ""
    if use_procar and os.path.isfile(p("PROCAR")):
        grp = np.array([st.species.index(s) for s in st.symbols], int)
        w = read_procar_groups(p("PROCAR"), grp, len(st.species), nk_expected=len(ev["kpts"]),
                               progress=progress)
        if w.shape[1] != nb:
            raise ValueError(f"PROCAR のバンド数 ({w.shape[1]}) が EIGENVAL ({nb}) と一致しません。")
        W = np.ascontiguousarray(w[mapping].transpose(1, 0, 2).reshape(nb, *mesh, len(st.species)))
    if not os.path.isfile(p("PROCAR")):
        note = "PROCAR がないため、元素ごとの寄与は表示できません（LORBIT = 11 で計算すると出力されます）。"
    return {"st": st, "mesh": mesh, "B": reciprocal_2pi(st.lattice), "efermi": float(efermi),
            "efermi_src": src, "E": E, "W": W, "species": list(st.species), "nk_ir": len(ev["kpts"]),
            "nspin": ev["eig"].shape[0], "note": note, "structure_file": os.path.basename(pos),
            "mapping": mapping, "has_procar": os.path.isfile(p("PROCAR")), "folder": folder,
            "kpts_ir": ev["kpts"], "Wir": None, "Worb": None, "frmsf_W": None,
            "orb_names": procar_orbital_names(p("PROCAR")) if os.path.isfile(p("PROCAR")) else []}


def load_fermi_weights(d, progress=None, bands=None):
    """build_fermi_data(…, use_procar=False) で読んだデータに、PROCAR の原子ごと・軌道ごとの重みを追加する。
    bands を渡すとそのバンドだけを読む（まだ読んでいないものだけ）。何度かに分けて呼べば、最初に E_F 付近の
    バンドだけを読んで表示し、残りを後から読む、という使い方ができる。
    d["Worb"] (既約 k点, バンド, 原子, 軌道), d["Wir"] (既約 k点, バンド, 原子) = tot、d["w_loaded"]（読んだバンド）"""
    st = d["st"]
    nb = d["E"].shape[0]
    path = os.path.join(d["folder"], "PROCAR")
    if d.get("_wbuf") is None:
        nk, nbp, ni = procar_header(path)
        if ni != st.natoms:
            raise ValueError(f"PROCAR の原子数 ({ni}) が POSCAR ({st.natoms}) と一致しません。")
        if nk != d["nk_ir"]:
            raise ValueError(f"PROCAR の k点数 ({nk}) が EIGENVAL ({d['nk_ir']}) と一致しません。")
        if nbp != nb:
            raise ValueError(f"PROCAR のバンド数 ({nbp}) が EIGENVAL ({nb}) と一致しません。")
        names = procar_orbital_names(path)
        if not names:
            raise ValueError("PROCAR に ion の見出し行がありません（LORBIT = 11 で計算してください）。")
        mapping, opidx, pinv, rots = unfold_kmesh_atoms(st, d["mesh"], d["kpts_ir"])
        d["_wbuf"] = np.zeros((nk, nb, ni, len(names) + 1), np.int16)
        d["w_loaded"] = np.zeros(nb, bool)
        d["mapping"], d["opidx"], d["pinv"], d["rots"] = mapping, opidx, pinv, rots
        d["orb_names"] = names
        d["orb_A"] = orbital_matrices(names, rots)
    w = d["_wbuf"]
    todo = [b for b in (range(nb) if bands is None else bands) if 0 <= b < nb and not d["w_loaded"][b]]
    if todo:
        read_procar_orbitals(path, st.natoms, nk_expected=d["nk_ir"], progress=progress, bands=todo, out=w)
        d["w_loaded"][todo] = True
    d["Worb"] = w[..., :-1]
    d["Wir"] = w[..., -1].astype(np.float32) / np.float32(1000.0)     # 最後に入れる（読み込み済みの印）
    return d["Wir"]


def fermi_weights_bundle(d):
    """PROCAR から読んだ重みと展開の表（キャッシュ・ライブラリに保存する分）"""
    return {k: d[k] for k in ("Worb", "Wir", "w_loaded", "mapping", "opidx", "pinv", "rots", "orb_names", "orb_A")}


def apply_fermi_weights_bundle(d, b):
    for k, v in b.items():
        d[k] = v
    d["w_loaded"] = np.ones(d["E"].shape[0], bool)


def fermi_source_files(folder):
    """フェルミ面のデータの元になるファイル（キャッシュの判定に使う）"""
    return [os.path.join(folder, n) for n in ("POSCAR", "CONTCAR", "KPOINTS", "EIGENVAL", "DOSCAR", "OUTCAR")]


def crossing_bands(E, ef):
    """E_F を横切るバンドの番号（0 始まり）"""
    mn = E.reshape(E.shape[0], -1).min(axis=1)
    mx = E.reshape(E.shape[0], -1).max(axis=1)
    return [int(b) for b in np.where((mn < ef) & (mx > ef))[0]]


def write_frmsf(path, mesh, B, E_rel, weights=None, bands=None):
    """VASPKIT 263（weights=None）/ 264・265（weights あり）と同じ書式の .frmsf を書く。
      1 行目: メッシュ (3I6) / 2 行目: 1（Γ中心: x_i = (i-1)/N）/ 3 行目: バンド数 (I6)
      4–6 行目: 逆格子ベクトル（2π を含む, 1/Å, 3F15.9）
      以降: E - E_F [eV] (F12.6) をバンド → k1 → k2 → k3 の順（k3 が最も速い）、続けて重み
    bands を指定すると、そのバンドだけを書く（VASPKIT の REDUCE_FERMISURFACE_FILE = .TRUE. に相当）"""
    sel = list(range(E_rel.shape[0])) if bands is None else list(bands)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("%6d%6d%6d\n" % tuple(mesh))
        f.write(" 1\n")
        f.write("%6d\n" % len(sel))
        for v in B:
            f.write("%15.9f%15.9f%15.9f\n" % tuple(v))
        for arr in ([E_rel] if weights is None else [E_rel, weights]):
            for b in sel:
                a = np.asarray(arr[b], float).ravel() + 0.0      # -0.0 は 0.0 として書く（VASPKIT と同じ）
                for i0 in range(0, len(a), 200000):
                    f.write("\n".join(["%12.6f" % x for x in a[i0:i0 + 200000]]) + "\n")


def kumo_interpolate(F, ratio):
    """周期的な格子上の量 F (n1, n2, n3[, K]) を各方向 ratio 倍の細かさに補間する。
    FermiSurfer の Interpolation と同じ 3 次の補間（kumo.cpp の kumo_coef。前後 4 点を使う局所的な補間）"""
    ratio = int(ratio)
    if ratio <= 1:
        return F
    F = np.asarray(F, np.float32)
    for ax in range(3):
        sh = list(F.shape)
        sh[ax] *= ratio
        out = np.empty(sh, np.float32)
        nb4 = [np.roll(F, -m, axis=ax) for m in (-1, 0, 1, 2)]       # i-1, i, i+1, i+2
        for j in range(ratio):
            x = j / ratio
            mx = 1.0 - x
            c = (-0.5 * x * mx * mx, mx * (mx * mx + 3.0 * x * mx + 0.5 * x * x),
                 x * (x * x + 3.0 * mx * x + 0.5 * mx * mx), -0.5 * x * x * mx)
            sl = [slice(None)] * F.ndim
            sl[ax] = slice(j, None, ratio)
            out[tuple(sl)] = c[0] * nb4[0] + c[1] * nb4[1] + c[2] * nb4[2] + c[3] * nb4[3]
        F = out
    return F


def _kumo_coef(j, ratio):
    x = j / ratio
    mx = 1.0 - x
    return (-0.5 * x * mx * mx, mx * (mx * mx + 3.0 * x * mx + 0.5 * x * x),
            x * (x * x + 3.0 * mx * x + 0.5 * mx * mx), -0.5 * x * x * mx)


def interpolate_bands(E, ratio, W=None, keep_plan=False):
    """エネルギー順に並んだ m 本のバンド E (m, n1, n2, n3) を、各方向 ratio 倍の細かさに補間する。
    VASP のバンド番号はエネルギーの順なので、バンドが交差する所で E_n(k) は折れ曲がっている。
    そのまま補間すると折れ目の近くで面が波打つため、次の手順で補間する:
      1) 格子の 1 方向の線ごとに、傾きが連続するようにバンドをつなぎ直す（交差をまたいでつなぐ）
      2) つなぎ直したバンドを 3 次補間（FermiSurfer と同じ kumo 補間）
      3) 補間後の各点でエネルギー順に並べ直す
    これを 3 方向について順に行う。格子点上の値は変わらない。
    W (m, n1, n2, n3, K) を渡すと、同じつなぎ方で重みも補間する。戻り値 (E, W)
    keep_plan=True なら (E, W, plan) を返す。plan を interpolate_weights に渡すと、後から別の重みを
    同じつなぎ方で補間できる（エネルギーの補間をやり直さずに済む）"""
    ratio = int(ratio)
    E = np.asarray(E, np.float32)
    plan = []
    if ratio <= 1:
        return (E, W, plan) if keep_plan else (E, W)
    m = E.shape[0]
    pad = 3
    for ax in (1, 2, 3):
        n = E.shape[ax]
        A = np.moveaxis(E, ax, -1)
        shp = A.shape                                   # (m, p, q, n)
        A = A.reshape(m, -1, n)
        idx = np.arange(-pad, n + pad) % n
        Ae = A[:, :, idx]                               # 周期的に前後 pad 点ずつ延長 (m, L, P)
        P = Ae.shape[2]
        D = np.empty_like(Ae)
        rank = np.empty(Ae.shape, np.intp)
        rank[:, :, 0] = rank[:, :, 1] = np.arange(m)[:, None]
        D[:, :, 0], D[:, :, 1] = Ae[:, :, 0], Ae[:, :, 1]
        for i in range(2, P):
            # 直前 2 点からの直線外挿値の順位どおりに、次の点のバンドを割り当てる
            # （1 次元で |外挿値 − 実際の値| の和を最小にする割り当ては、順位を合わせたもの）
            pred = 2.0 * D[:, :, i - 1] - D[:, :, i - 2]
            r = np.argsort(np.argsort(pred, axis=0, kind="stable"), axis=0, kind="stable")
            rank[:, :, i] = r
            D[:, :, i] = np.take_along_axis(Ae[:, :, i], r, axis=0)
        out = np.empty((m, A.shape[1], n * ratio), np.float32)
        if W is not None:
            Wl = np.moveaxis(W, ax, 3)                  # (m, p, q, n, K)
            wshp = Wl.shape
            Wl = Wl.reshape(m, -1, n, wshp[-1])[:, :, idx]
            WD = np.take_along_axis(Wl, rank[..., None], axis=0)
            wout = np.empty((m, A.shape[1], n * ratio, wshp[-1]), np.float32)
        for j in range(ratio):
            c = _kumo_coef(j, ratio)
            out[:, :, j::ratio] = sum(c[k] * D[:, :, pad - 1 + k:pad - 1 + k + n] for k in range(4))
            if W is not None:
                wout[:, :, j::ratio] = sum(c[k] * WD[:, :, pad - 1 + k:pad - 1 + k + n] for k in range(4))
        order = np.argsort(out, axis=0, kind="stable")
        out = np.take_along_axis(out, order, axis=0)
        E = np.moveaxis(out.reshape(shp[:-1] + (n * ratio,)), -1, ax)
        if keep_plan:
            it = np.int8 if m <= 127 else np.int16
            plan.append((ax, n, idx, rank.astype(it), order.astype(it)))
        if W is not None:
            wout = np.take_along_axis(wout, order[..., None], axis=0)
            W = np.moveaxis(wout.reshape(wshp[:3] + (n * ratio, wshp[-1])), 3, ax)
    if W is not None:
        W = np.clip(W, 0.0, None)
    E = np.ascontiguousarray(E)
    W = None if W is None else np.ascontiguousarray(W)
    return (E, W, plan) if keep_plan else (E, W)


def interpolate_weights(plan, W, ratio):
    """interpolate_bands(…, keep_plan=True) の手順で、重み W (m, n1, n2, n3) を補間する（1 成分）"""
    W = np.asarray(W, np.float32)
    pad = 3
    for ax, n, idx, rank, order in plan:
        Wl = np.moveaxis(W, ax, -1)
        shp = Wl.shape
        m = shp[0]
        Wl = Wl.reshape(m, -1, n)[:, :, idx]
        WD = np.take_along_axis(Wl, rank.astype(np.intp), axis=0)
        out = np.empty((m, WD.shape[1], n * ratio), np.float32)
        for j in range(ratio):
            c = _kumo_coef(j, ratio)
            out[:, :, j::ratio] = sum(c[k] * WD[:, :, pad - 1 + k:pad - 1 + k + n] for k in range(4))
        out = np.take_along_axis(out, order.astype(np.intp), axis=0)
        W = np.moveaxis(out.reshape(shp[:-1] + (n * ratio,)), -1, ax)
    return np.ascontiguousarray(np.clip(W, 0.0, None))


def refine_bands_plan(E, bands, ratio, tol=1e-4):
    """フェルミ面を作るバンドを補間する（重みなし）。重みは後から refine_weights で同じ手順で補間する。
    戻り値 (fine, runs)
      fine {バンド番号: E_fine}（縮退したバンドは同じ配列を共有する）
      runs [{"members": [[b, …], …], "plan": plan, "out": {b: グループ内の番号}}]"""
    fine, runs = {}, []
    bands = sorted(bands)
    if not bands:
        return fine, runs
    nb = E.shape[0]
    groups, cur = [], [bands[0]]
    for b in bands[1:]:
        if b <= cur[-1] + 3:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    for run in groups:
        lo, hi = (max(0, run[0] - 2), min(nb - 1, run[-1] + 2)) if ratio > 1 else (run[0], run[-1])
        reps, members = [], []
        for b in range(lo, hi + 1):
            if reps and float(np.abs(E[b] - E[reps[-1]]).max()) < tol:
                members[-1].append(b)
            else:
                reps.append(b)
                members.append([b])
        if ratio > 1:
            Ef, _, plan = interpolate_bands(E[reps], ratio, None, keep_plan=True)
        else:
            Ef, plan = np.ascontiguousarray(E[reps], np.float32), []
        where = {}
        arrs = [Ef[k] for k in range(len(members))]     # 縮退したバンドは同じ配列オブジェクトを共有させる
        for k, mem in enumerate(members):
            for b in mem:
                if b in run:
                    fine[b] = arrs[k]
                    where[b] = k
        runs.append({"members": members, "plan": plan, "out": where})
    return fine, runs


def refine_weights(runs, W, ratio):
    """refine_bands_plan の手順で、選んだ重み W を補間する。W[b] → (n1, n2, n3)（1 成分）。
    縮退したバンドは平均を使う。戻り値 {バンド番号: 細かい格子の重み（1 次元）}"""
    out = {}
    for r in runs:
        Wg = np.stack([np.mean([np.asarray(W[b], np.float32) for b in mem], axis=0) for mem in r["members"]], axis=0)
        Wf = interpolate_weights(r["plan"], Wg, ratio) if r["plan"] else Wg
        for b, k in r["out"].items():
            out[b] = Wf[k].ravel()
    return out


def refine_bands(E, W, bands, ratio, tol=1e-4):
    """フェルミ面を作るバンド（番号のリスト bands）を補間する。戻り値 {バンド番号: (E_fine, W_fine)}
    ・番号が連続するバンドをまとめ、上下 2 本ずつを加えたグループで interpolate_bands を行う
      （グループ内の交差を正しくつなぐため）
    ・どの k点でもエネルギーが同じバンド（反転対称 + SOC での 2 重縮退など）は 1 本として扱い、
      重みは縮退したバンドの平均を使う"""
    out = {}
    bands = sorted(bands)
    if not bands:
        return out
    if ratio <= 1:
        # 補間なし: 縮退して重なるバンドは同じ配列を共有させる（面の計算と描画を 1 回で済ませるため）
        for b in bands:
            if b - 1 in out and float(np.abs(E[b] - E[b - 1]).max()) < tol:
                out[b] = out[b - 1]
            else:
                out[b] = (np.ascontiguousarray(E[b], np.float32), None if W is None else W[b])
        return out
    nb = E.shape[0]
    runs, cur = [], [bands[0]]
    for b in bands[1:]:
        if b <= cur[-1] + 3:                 # 近い番号のバンドは同じグループにする
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    for run in runs:
        lo, hi = max(0, run[0] - 2), min(nb - 1, run[-1] + 2)
        reps, members = [], []
        for b in range(lo, hi + 1):
            if reps and float(np.abs(E[b] - E[reps[-1]]).max()) < tol:
                members[-1].append(b)
            else:
                reps.append(b)
                members.append([b])
        Wg = None
        if W is not None:
            Wg = np.stack([np.mean([W[b] for b in mem], axis=0) for mem in members], axis=0)
        Ef, Wf = interpolate_bands(E[reps], ratio, Wg)
        for k, mem in enumerate(members):
            item = (Ef[k], None if Wf is None else Wf[k])       # 縮退したバンドは同じ配列を共有する
            for b in mem:
                if b in run:
                    out[b] = item
    return out


def vertex_src(D):
    """isosurface_tets(…, src=True) の三角形 D から、頂点の (λ (T, 3, 4), 格子番号 (T, 3, 4)) を取り出す"""
    c = D.shape[2] - 12
    lam = np.ascontiguousarray(D[:, :, c:c + 4])
    idx = (np.rint(D[:, :, c + 4:c + 8]).astype(np.int32) * 4096 + np.rint(D[:, :, c + 8:c + 12]).astype(np.int32))
    return lam, idx


def vertex_values(D, Wflat, src=None):
    """isosurface_tets(…, src=True) の三角形 D の各頂点での、格子上の量の値 Σ λ_c W[c]。
    Wflat: 格子の値を 1 次元にしたもの（等値面を作った格子と同じ並び）。
    src: vertex_src(D) の結果（何度も使うときは先に作っておくと速い）。戻り値 (T, 3) float32"""
    if len(D) == 0:
        return np.zeros((0, 3), np.float32)
    lam, idx = vertex_src(D) if src is None else src
    W = np.asarray(Wflat, np.float32)
    return np.einsum("tvc,tvc->tv", lam, W[idx]).astype(np.float32)


def fermi_band_mesh(E_b, level, B, W_b=None, first_bz=True, interpol=1, src=False):
    """1 本のバンドのフェルミ面（E = level）の三角形。
    interpol > 1 なら、k点メッシュを各方向 interpol 倍に補間してから面を作る。
    戻り値 D (T, 3, 6 + 元素数): [k (デカルト, 1/Å), ∇_k E (eV Å), 元素ごとの重み]"""
    if interpol > 1:
        E_b = kumo_interpolate(E_b, interpol)
        if W_b is not None:
            W_b = np.clip(kumo_interpolate(W_b, interpol), 0.0, None)
    D = isosurface_tets(E_b, level, B, attrs=W_b, src=src)
    if first_bz:
        D = fold_into_first_bz(D, B)
    return D


# ======================================================================
#  WAVECAR: 選んだバンドの電荷密度を、原子のまわりで s, p, d, f 成分に分ける
# ======================================================================
WAVE_C = 0.262465831          # 2m/ħ² [1/(eV Å²)]。|k+G|² / WAVE_C = エネルギー [eV]（WaveTrans・pymatgen と同じ値）


class WavecarFile:
    """VASP の WAVECAR（Fortran の直接アクセスファイル）を読む。
    書式は WaveTrans / pymatgen.io.vasp.outputs.Wavecar と同じ解釈:
      レコード 1: レコード長, スピン数, 精度のタグ（45200/53300: 単精度, 45210/53310: 倍精度）
      レコード 2: k点数, バンド数, ENCUT, 格子ベクトル (3×3), E_F
      以降、スピン → k点ごとに [平面波の数, k点, (固有値の実部・虚部, 占有数) × バンド数] のレコードと、
      バンドごとの係数のレコード
    必要なレコードだけを位置を指定して読むので、数 GB のファイルでも全体は読まない。
    平面波 G の並び（i3 → i2 → i1 の順で |k+G|²/C < ENCUT のもの）も WaveTrans と同じ。
    vasp_std / vasp_gam / vasp_ncl は、平面波の数から判定する。"""

    def __init__(self, path, progress=None):
        self.path = path
        with open(path, "rb") as f:
            h = np.frombuffer(f.read(24), np.float64)
            if len(h) < 3:
                raise ValueError("WAVECAR が空か、壊れています。")
            if not (np.all(np.isfinite(h)) and 0 < h[0] < 1e10 and h[1] in (1.0, 2.0)
                    and round(h[2]) in (45200, 53300, 45210, 53310)):
                raise ValueError("WAVECAR の形式として読めません（ファイルが壊れているか、WAVECAR ではありません。"
                                 "vasp_std / vasp_gam / vasp_ncl の WAVECAR に対応しています）。")
            self.recl, self.nspin, self.rtag = int(h[0]), int(round(h[1])), int(round(h[2]))
            if self.rtag in (45200, 53300):
                self.cdtype = np.complex64
            elif self.rtag in (45210, 53310):
                self.cdtype = np.complex128
            else:
                raise ValueError(f"WAVECAR の形式（タグ {self.rtag}）に対応していません。")
            f.seek(self.recl)
            h2 = np.frombuffer(f.read(13 * 8), np.float64)
            if len(h2) < 13:
                raise ValueError("WAVECAR が空か、壊れています。")
        self.nk, self.nb = int(round(h2[0])), int(round(h2[1]))
        self.encut = float(h2[2])
        self.lattice = h2[3:12].reshape(3, 3)
        self.efermi = float(h2[12])
        self.vol = abs(float(np.linalg.det(self.lattice)))
        self.B = 2.0 * np.pi * np.linalg.inv(self.lattice).T
        self._nbmax()
        self.kpts = np.zeros((self.nk, 3))
        self.nplw = np.zeros((self.nspin, self.nk), int)
        self.eig = np.zeros((self.nspin, self.nk, self.nb))
        self.occ = np.zeros((self.nspin, self.nk, self.nb))
        # k点ごとの情報のレコード（ファイル全体に散らばっている）を読む。バッファなし・readinto で 1 回ずつ読む
        nrd = 4 + 3 * self.nb
        a = np.empty(nrd, np.float64)
        mv = memoryview(a).cast("B")
        ntot = self.nspin * self.nk
        with open(path, "rb", buffering=0) as f:
            for s in range(self.nspin):
                for k in range(self.nk):
                    f.seek(self._rec(s, k, None) * self.recl)
                    got = f.readinto(mv)
                    if got != nrd * 8:
                        raise ValueError(f"WAVECAR が途中で切れています（スピン {s + 1}, k点 {k + 1}）。")
                    if progress is not None and k % 20 == 0:
                        progress(s * self.nk + k, ntot)
                    self.nplw[s, k] = int(round(a[0]))
                    self.kpts[k] = a[1:4]
                    eo = a[4:].reshape(self.nb, 3)
                    self.eig[s, k], self.occ[s, k] = eo[:, 0], eo[:, 2]
        g_std = self.gvectors(self.kpts[0], gamma=False)
        if self.nplw[0, 0] == len(g_std):
            self.vtype = "std"
        elif self.nplw[0, 0] == 2 * len(g_std):
            self.vtype = "ncl"
        elif self.nplw[0, 0] == len(self.gvectors(self.kpts[0], gamma=True)):
            self.vtype = "gam"
        else:
            raise ValueError("WAVECAR の平面波の数が、ENCUT と格子から決まる数と合いません（対応していない形式）。")

    def _rec(self, s, k, n):
        """レコード番号（0 始まり）。n = None は k点の情報のレコード。
        k点の情報（4 + 3×バンド数 個の倍精度）がレコード長に収まらないときは、複数のレコードにまたがる"""
        ninfo = -(-(4 + 3 * self.nb) * 8 // self.recl)
        return 2 + (s * self.nk + k) * (self.nb + ninfo) + (0 if n is None else ninfo + n)

    def _nbmax(self):
        """各方向の G の最大値（WaveTrans の方法）"""
        b = self.B
        bm = np.linalg.norm(b, axis=1)
        gm = np.sqrt(self.encut * WAVE_C)

        def ang(i, j):
            return np.arccos(np.dot(b[i], b[j]) / (bm[i] * bm[j]))

        def sph(i, j, k):
            c = np.cross(b[i], b[j])
            return np.dot(b[k], c) / (bm[k] * np.linalg.norm(c))
        A = gm / bm
        A[0] /= abs(np.sin(ang(0, 1)))
        A[1] /= abs(np.sin(ang(0, 1)))
        A[2] /= abs(sph(0, 1, 2))
        Bv = gm / bm
        Bv[0] /= abs(np.sin(ang(0, 2)))
        Bv[1] /= abs(sph(0, 2, 1))
        Bv[2] /= abs(np.sin(ang(0, 2)))
        C = gm / bm
        C[0] /= abs(sph(1, 2, 0))
        C[1] /= abs(np.sin(ang(1, 2)))
        C[2] /= abs(np.sin(ang(1, 2)))
        self.nbmax = (np.max([A + 1, Bv + 1, C + 1], axis=0)).astype(int)

    def gvectors(self, kpt, gamma=False):
        """k点 kpt（分率座標）の平面波の G（整数, (npw, 3)）。WAVECAR の係数と同じ順"""
        nb = self.nbmax
        r1 = np.r_[0:nb[0] + 1] if gamma else np.r_[0:nb[0] + 1, -nb[0]:0]
        r2 = np.r_[0:nb[1] + 1, -nb[1]:0]
        r3 = np.r_[0:nb[2] + 1, -nb[2]:0]
        g3, g2, g1 = np.meshgrid(r3, r2, r1, indexing="ij")
        G = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=1)
        if gamma:
            keep = ~((G[:, 0] == 0) & (G[:, 1] < 0)) & ~((G[:, 0] == 0) & (G[:, 1] == 0) & (G[:, 2] < 0))
            G = G[keep]
        q = (np.asarray(kpt, float)[None, :] + G) @ self.B
        E = np.einsum("ij,ij->i", q, q) / WAVE_C
        return G[E < self.encut]

    def band(self, s, k, n):
        """スピン s, k点 k, バンド n（すべて 0 始まり）の平面波係数。
        戻り値 (G (npw, 3) 整数, c (成分数, npw) 複素数)。成分数は vasp_ncl なら 2、それ以外は 1。
        vasp_gam の WAVECAR は G の半分しかないので、c(−G) = c(G)* で残りを補う（pymatgen と同じく
        G ≠ 0 の係数を √2 で割る）"""
        npl = int(self.nplw[s, k])
        # np.fromfile は Windows で 1 回ごとに時間がかかることがあるので、readinto で読む
        c = np.empty(npl, self.cdtype)
        with open(self.path, "rb", buffering=0) as f:
            f.seek(self._rec(s, k, n) * self.recl)
            got = f.readinto(memoryview(c).cast("B"))
        if got != c.nbytes:
            raise ValueError(f"WAVECAR が途中で切れています（スピン {s + 1}, k点 {k + 1}, バンド {n + 1}）。")
        c = c.astype(np.complex128)
        if self.vtype == "ncl":
            G = self.gvectors(self.kpts[k], gamma=False)
            return G, c.reshape(2, npl // 2)
        if self.vtype == "gam":
            G = self.gvectors(self.kpts[k], gamma=True)
            nz = np.any(G != 0, axis=1)
            c = c.copy()
            c[nz] /= np.sqrt(2.0)
            return np.vstack([G, -G[nz]]), np.concatenate([c, np.conj(c[nz])])[None, :]
        return self.gvectors(self.kpts[k], gamma=False), c[None, :]


def real_harmonics_normalized(l, u):
    """規格化した実球面調和関数（VASP の PROCAR の列の順: s / py pz px / dxy dyz dz2 dxz dx2-y2 / f…）。
    u: 単位ベクトル (n, 3)。戻り値 (n, 2l+1)"""
    if l == 0:
        return np.full((len(u), 1), np.sqrt(1.0 / (4.0 * np.pi)))
    return _real_harmonics(l, u) * np.sqrt((2 * l + 1) / (4.0 * np.pi))


def spherical_jl(l, x):
    """球ベッセル関数 j_l(x)（l = 0–3）。x が小さいところは級数で計算する"""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = x < 0.05 + 0.25 * l
    xs = x[small]
    fact = [1.0, 3.0, 15.0, 105.0][l]                       # (2l+1)!!
    x2 = xs * xs
    out[small] = xs ** l / fact * (1.0 - x2 / (2 * (2 * l + 3)) + x2 * x2 / (8 * (2 * l + 3) * (2 * l + 5)))
    xb = x[~small]
    s, c = np.sin(xb), np.cos(xb)
    if l == 0:
        out[~small] = s / xb
    elif l == 1:
        out[~small] = s / xb ** 2 - c / xb
    elif l == 2:
        out[~small] = (3.0 / xb ** 3 - 1.0 / xb) * s - 3.0 * c / xb ** 2
    else:
        out[~small] = (15.0 / xb ** 4 - 6.0 / xb ** 2) * s - (15.0 / xb ** 3 - 1.0 / xb) * c
    return out


def default_sphere_radii(st, factor=0.5):
    """原子ごとの球の半径 = 最も近い原子までの距離 × factor（0.5 以下なら球は重ならない）"""
    frac = np.mod(st.frac, 1.0)
    L = np.asarray(st.lattice, float)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    out = np.empty(len(frac))
    for a in range(len(frac)):
        d = (frac[None, :, :] + shifts[:, None, :] - frac[a]).reshape(-1, 3) @ L
        n = np.linalg.norm(d, axis=1)
        out[a] = n[n > 1e-6].min() * factor
    return out


def symmetric_grid(st, spacing=0.07):
    """出力の格子の大きさ。結晶の対称操作で格子点が格子点に移るように選ぶ
    （回転で混ざる軸は同じ分割数、並進が格子点に乗るように分割数を倍数にする）"""
    import spglib
    L = np.asarray(st.lattice, float)
    N = np.maximum(8, np.ceil(np.linalg.norm(L, axis=1) / spacing)).astype(int)
    sym = spglib.get_symmetry(st.spglib_cell(), symprec=1e-3)
    rots = np.asarray(sym["rotations"] if isinstance(sym, dict) else sym.rotations)
    trans = np.asarray(sym["translations"] if isinstance(sym, dict) else sym.translations)
    group = [0, 1, 2]
    for R in rots:
        for i in range(3):
            for j in range(3):
                if i != j and R[i, j] != 0:
                    gi, gj = group[i], group[j]
                    group = [gi if g == gj else g for g in group]
    den = 1
    for t in trans:
        for x in np.mod(t, 1.0):
            for q in range(1, 13):
                if abs(x * q - round(x * q)) < 1e-3:
                    den = int(np.lcm(den, q))
                    break
    den = int(np.lcm(den, 2))
    for g in set(group):
        ax = [i for i in range(3) if group[i] == g]
        n = int(max(N[i] for i in ax))
        n = int(np.ceil(n / den) * den)
        for i in ax:
            N[i] = n
    return tuple(int(x) for x in N), rots, trans


def symmetrize_grid(F, rots, trans, points=None):
    """格子上の量 F (N1, N2, N3)（または同じ大きさの配列のリスト）を対称操作で平均する:
    F_sym(x) = 1/|G| Σ F(R⁻¹(x − t))。points（格子番号 (n, 3)）を渡すと、その点だけ計算して (n,) を返す。
    格子は symmetric_grid で選んだもの（回転で混ざる軸は同じ分割数、並進は格子点に乗る）を前提に、
    整数の計算だけで移り先を求める"""
    many = isinstance(F, (list, tuple))
    Fs = list(F) if many else [F]
    N = np.array(Fs[0].shape)
    idx = np.indices(Fs[0].shape).reshape(3, -1).T if points is None else np.asarray(points)
    flats = [np.ascontiguousarray(f).ravel() for f in Fs]
    acc = [np.zeros(len(idx), np.float64) for _ in Fs]
    for R, t in zip(rots, trans):
        Ri = np.rint(np.linalg.inv(np.asarray(R, float))).astype(np.int64)
        T = np.rint(np.mod(t, 1.0) * N).astype(np.int64)
        si = ((idx - T) @ Ri.T) % N
        fl = (si[:, 0] * N[1] + si[:, 1]) * N[2] + si[:, 2]
        for a, f in zip(acc, flats):
            a += f[fl]
    out = []
    for a in acc:
        a /= len(rots)
        out.append(a.astype(np.float32) if points is not None else a.reshape(Fs[0].shape).astype(np.float32))
    return out if many else out[0]


def sphere_points(st, radii, N):
    """各原子の球の中にある格子点。戻り値 [(原子の番号, 格子番号 (n, 3), 中心からのベクトル (n, 3) Å)]"""
    L = np.asarray(st.lattice, float)
    Li = np.linalg.inv(L)
    N = np.array(N)
    out = []
    for a, (x0, R) in enumerate(zip(np.mod(st.frac, 1.0), radii)):
        ext = np.ceil(R * np.linalg.norm(Li, axis=0) * N).astype(int) + 1   # 各方向の格子点の範囲
        c = np.floor(x0 * N).astype(int)
        r = [np.arange(c[i] - ext[i], c[i] + ext[i] + 2) for i in range(3)]
        g = np.stack(np.meshgrid(*r, indexing="ij"), axis=-1).reshape(-1, 3)
        v = (g / N - x0) @ L
        dist = np.linalg.norm(v, axis=1)
        m = dist <= R
        out.append((a, g[m] % N, v[m]))
    return out


class _JlTable:
    """球ベッセル関数 j_l(x) の表（細かい間隔で 1 回だけ作り、線形補間で引く）。
    間隔 dx = 5e-4 での線形補間の誤差は |j_l''| dx² / 8 ≲ 3×10⁻⁸（|j_l''| ≤ 1）"""
    DX = 5e-4

    def __init__(self, lmax, xmax):
        self.lmax = lmax
        self.build(xmax)

    def build(self, xmax):
        self.xmax = float(xmax)
        self.x = np.arange(0.0, self.xmax + 2 * self.DX, self.DX)
        self.tab = [spherical_jl(l, self.x).astype(np.float64) for l in range(self.lmax + 1)]

    def __call__(self, l, x):
        """x（任意の形）での j_l(x)。表の範囲外は正確な式で計算する"""
        if x.size and float(x.max()) > self.xmax:
            return spherical_jl(l, x)
        t = x * (1.0 / self.DX)
        i = np.minimum(t.astype(np.int64), len(self.x) - 2)
        f = t - i
        tb = self.tab[l]
        return tb[i] * (1.0 - f) + tb[i + 1] * f

    def all_l(self, x, ls):
        """同じ x について、いくつかの l の j_l(x) をまとめて引く（位置の計算を 1 回で済ませる）。単精度で返す"""
        if x.size and float(x.max()) > self.xmax:
            return {l: spherical_jl(l, x).astype(np.float32) for l in ls}
        t = x * (1.0 / self.DX)
        i = np.minimum(t.astype(np.int64), len(self.x) - 2)
        f = (t - i).astype(np.float32)
        g = 1.0 - f
        out = {}
        for l in ls:
            tb = self.tab[l].astype(np.float32)
            out[l] = tb[i] * g + tb[i + 1] * f
        return out


def thin_kweights(st, kpts, mesh, factor):
    """k点の間引き: Γ中心メッシュ mesh の各方向 factor 個おきの点（= 粗い Γ中心メッシュ）だけを使うときの、
    既約 k点 kpts の重み。粗いメッシュの点は細かいメッシュの点に含まれるので、それぞれを対称操作で
    既約 k点に対応させて数える（最後に対称操作で平均するので、どの等価な点を使っても同じ）。
    割り切れない方向は間引かない。戻り値 (重み (既約 k点数,), 粗いメッシュ, 使う既約 k点の数)"""
    N = np.array(mesh, int)
    mapping = unfold_kmesh(st, tuple(N), kpts)
    f = np.array([factor if n % factor == 0 else 1 for n in N])
    idx = np.indices(tuple(N)).reshape(3, -1).T
    sel = np.all(idx % f == 0, axis=1)
    w = np.bincount(mapping[sel], minlength=len(kpts)).astype(float)
    w /= w.sum()
    return w, tuple(int(x) for x in N // f), int((w > 0).sum())


def _orbital_density_matrices_fast(wc, tau, selection, kweights, rr, ls, fac, njobs, progress, cancel, nthreads):
    """orbital_band_density の中心部分（密度行列 M_l の足し合わせ）の速い版。結果は元の方法と同じ式で、
    (1) j_l は表から引く、(2) 行列計算は単精度（WAVECAR の係数ももともと単精度）、足し合わせは倍精度、
    (3) j_l(qr) と Y_lm(q̂) の積の大きな表を作らず、(c_G e^{iqτ} Y_lm(q̂)) @ j_l(q r) の形で計算する、
    (4) k点ごとに複数のスレッドで並列に計算する（numpy の計算中は Python の制約が外れる）"""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    nat, nr = len(tau), len(rr)
    pref = np.float32(4.0 * np.pi / np.sqrt(wc.vol))
    by_k = {}
    for (s, k), bands in selection.items():
        by_k.setdefault(k, []).extend((s, n) for n in bands)
    ks = sorted(by_k)
    # j_l の表の範囲: 最初の k点の G から |q| の最大を見積もり、余裕を持たせる（超えたら正確な式で計算する）
    if ks:
        G0, _ = wc.band(*by_k[ks[0]][0][:1], ks[0], by_k[ks[0]][0][1])
        qmax = float(np.linalg.norm((wc.kpts[ks[0]][None, :] + G0) @ wc.B, axis=1).max())
        kmax = float(np.linalg.norm(np.asarray(wc.kpts) @ wc.B, axis=1).max())
        table = _JlTable(max(ls), (qmax + 2 * kmax + 0.5) * rr[-1] * 1.02)
    lock = threading.Lock()
    state = {"done": 0, "norm": 0.0}

    def work(chunk):
        Ml = {l: np.zeros((nat, nr, 2 * l + 1, 2 * l + 1)) for l in ls}
        norm = 0.0
        for k in chunk:
            if cancel is not None and cancel():
                raise RuntimeError("中止しました。")
            items = by_k[k]
            G = None
            for s, n in items:
                Gb, C = wc.band(s, k, n)
                if G is None:                                         # k点ごとの表
                    G = Gb
                    q = (wc.kpts[k][None, :] + G) @ wc.B
                    qn = np.linalg.norm(q, axis=1)
                    qh = q / np.where(qn > 1e-12, qn, 1.0)[:, None]
                    qh[qn <= 1e-12] = [0.0, 0.0, 1.0]
                    phase = np.exp(1j * (tau @ q.T)).astype(np.complex64)                 # (原子, G)
                    x = qn[:, None] * rr[None, :]
                    J = table.all_l(x, ls)                                               # (G, r)
                    Y = {l: real_harmonics_normalized(l, qh).T.astype(np.float32) for l in ls}   # (m, G)
                w = kweights[k] * fac
                for comp in C:
                    comp = comp.astype(np.complex64)
                    norm += w * float(np.vdot(comp, comp).real)
                    A = phase * comp[None, :]                                           # (原子, G)
                    for l in ls:
                        m = 2 * l + 1
                        Bm = (A[:, None, :] * Y[l][None, :, :]).reshape(nat * m, -1)      # (原子·m, G)
                        cl = (Bm.real @ J[l]) + 1j * (Bm.imag @ J[l])                    # (原子·m, r)
                        cl = (cl.reshape(nat, m, nr).transpose(0, 2, 1) * (pref * (1j ** l))).astype(np.complex128)
                        Ml[l] += w * np.einsum("arm,arn->armn", cl, np.conj(cl)).real
                with lock:
                    state["done"] += 1
                    if progress:
                        progress(state["done"], njobs)
        return Ml, norm

    if nthreads is None:
        nthreads = max(1, min(8, (os.cpu_count() or 2) - 1))
    nthreads = max(1, min(nthreads, len(ks)))
    chunks = [ks[i::nthreads] for i in range(nthreads)] if ks else []
    M = {l: np.zeros((nat, nr, 2 * l + 1, 2 * l + 1)) for l in ls}
    total = 0.0
    if nthreads <= 1:
        results = [work(c) for c in chunks]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            results = list(ex.map(work, chunks))
    for Ml, norm in results:
        for l in ls:
            M[l] += Ml[l]
        total += norm
    return M, total


ORBITAL_M_NAMES = {0: ["s"], 1: ["py", "pz", "px"], 2: ["dxy", "dyz", "dz2", "dxz", "dx2-y2"],
                   3: ["fy(3x2-y2)", "fxyz", "fyz2", "fz3", "fxz2", "fz(x2-y2)", "fx(x2-3y2)"]}


# 画面に出す成分の名前（ORBITAL_M_NAMES と同じ順）
ORBITAL_M_LABELS = {0: ["s"], 1: ["py", "pz", "px"], 2: ["dxy", "dyz", "dz²", "dzx", "dx²−y²"],
                    3: ["y(3x²−y²)", "xyz", "yz²", "z³", "xz²", "z(x²−y²)", "x(x²−3y²)"]}


# 画面に並べる順（m の番号。中身の順は PROCAR と同じのまま）: p は px, py, pz、d は dxy, dyz, dzx, dz², dx²−y²
ORBITAL_M_ORDER = {0: [0], 1: [2, 0, 1], 2: [0, 1, 3, 2, 4], 3: [0, 1, 2, 3, 4, 5, 6]}
# 説明（CHGCAR の 1 行目・ライブラリの名前）に使う名前（画面と同じく dzx）
ORBITAL_M_TEXT = {0: ["s"], 1: ["py", "pz", "px"], 2: ["dxy", "dyz", "dz2", "dzx", "dx2-y2"],
                  3: ["fy(3x2-y2)", "fxyz", "fyz2", "fz3", "fxz2", "fz(x2-y2)", "fx(x2-3y2)"]}


def orbital_subset_name(subset):
    """{l: [m…]} → 表示用の名前（l を全部選んだら "d"、一部なら "dxy+dxz"、f は "f:xyz" など）"""
    parts = []
    for l in sorted(subset):
        ms = list(subset[l])
        if not ms:
            continue
        if len(ms) == 2 * l + 1:
            parts.append("spdf"[l])
        else:
            parts += [("f:" if l == 3 else "") + ORBITAL_M_TEXT[l][m] for m in ORBITAL_M_ORDER[l] if m in ms]
    return "+".join(parts) or "(なし)"


def read_potcar_lchannels(path):
    """POTCAR から元素ごとの軌道の種類 l を決める。戻り値 [(元素記号, [l…], 価電子配置の文字列)]（POTCAR の順）。
    価電子配置（VRHFIN =Se: s2p4 など）から、価電子の殻として
      s, p は常に含める（典型元素は ns np、遷移金属も ns np (n−1)d と数える）
      d, f は価電子配置に書かれているときだけ（Ta_pv: 5p6s5d → s p d、Pd: d9 s1 → s p d、Se: s2p4 → s p）
    PAW の射影子の l（"Non local Part"）は、価電子に無い d（Se など）も含むので使わない"""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "VRHFIN" in line and "=" in line:
                body = line.split("=", 1)[1]
                sym, _, conf = body.partition(":")
                ls = {0, 1}
                for ch in conf.lower():
                    if ch in "df":
                        ls.add("spdf".index(ch))
                out.append((sym.strip(), sorted(ls), " ".join(conf.split())))
    return out


def element_lchannels(species, entries):
    """元素 → [l…]（POTCAR の元素記号で対応づける。合わなければ順番で）。決められなければ None"""
    if not entries:
        return None
    by_sym = {sym: ls for sym, ls, _ in entries}
    if all(el in by_sym for el in species):
        return {el: list(by_sym[el]) for el in species}
    if len(entries) == len(species):
        return {el: list(e[1]) for el, e in zip(species, entries)}
    return None


def element_valence_text(species, entries):
    """元素 → 価電子配置の文字列（表示用）"""
    if not entries:
        return {}
    by_sym = {sym: conf for sym, _, conf in entries}
    if all(el in by_sym for el in species):
        return {el: by_sym[el] for el in species}
    if len(entries) == len(species):
        return {el: e[2] for el, e in zip(species, entries)}
    return {}


class AtomSubsets(dict):
    """原子ごとに違う成分を選ぶとき: {原子の番号: {l: [m…]}}（元素ごとの軌道の設定）。
    orbital_points_density / orbital_field_ext / sphere_caps の subset に、共通の {l: [m…]} の代わりに渡せる"""


def subset_of(subset, a):
    """原子 a に使う成分 {l: [m…]}"""
    return subset.get(a, {}) if isinstance(subset, AtomSubsets) else subset


def orbital_m_breakdown_text(res):
    """m ごとの球の中の電子数（全原子）の表示用の文字列。値が小さい l は省く"""
    em = res.get("electrons_m")
    if not em:
        return ""
    rows = []
    tot = max(sum(res["electrons"].values()), 1e-12)
    for l in (1, 2, 3):
        vals = [em.get(f"{l},{m}", 0.0) for m in range(2 * l + 1)]
        if sum(vals) < 1e-3 * tot:
            continue
        rows.append(f"  {'spdf'[l]}: " + "  ".join(f"{ORBITAL_M_LABELS[l][m]} {vals[m]:.4f}" for m in ORBITAL_M_ORDER[l]))
    return ("成分ごと（全原子）:\n" + "\n".join(rows) + "\n") if rows else ""


def symmetry_atom_maps(st, rots, trans):
    """対称操作 {R|t}（分率座標。x' = R x + t）ごとの、原子の移り先 p[op, a] とデカルト座標での回転 Rc[op]。
    原子の位置の一致は 1e-2 Å 以内（symmetric_grid と同じ symprec = 1e-3 の操作に対して十分）"""
    frac = np.mod(np.asarray(st.frac, float), 1.0)
    L = np.asarray(st.lattice, float)
    LT, LTi = L.T, np.linalg.inv(L.T)
    nums = np.array(st.spglib_cell()[2])
    nat = len(frac)
    perm, Rcs = [], []
    for R, t in zip(rots, trans):
        img = frac @ np.asarray(R, float).T + np.asarray(t, float)
        dlt = img[:, None, :] - frac[None, :, :]
        dist = np.linalg.norm((dlt - np.rint(dlt)) @ L, axis=2)
        dist[nums[:, None] != nums[None, :]] = np.inf
        p = dist.argmin(axis=1)
        if dist[np.arange(nat), p].max() > 1e-2 or len(set(p.tolist())) != nat:
            raise ValueError("対称操作で原子の移り先を決められませんでした（POSCAR の原子位置が対称からずれている可能性）。")
        perm.append(p)
        Rcs.append(LT @ np.asarray(R, float) @ LTi)
    return np.array(perm), np.array(Rcs)


def symmetrize_density_matrices(M, st, rots, trans):
    """原子のまわりの密度行列 M[l] (原子, r, m, m') を結晶の対称操作で平均する（全 k点で計算したものと同じになる）。
    操作 g = {R|t} が原子 a を原子 b = p_g(a) に移すとき、対称化した密度は
        ρ_sym(τ_b + r) = 1/|G| Σ_g ρ_a(R⁻¹ r)
    で、実球面調和関数は Y(R⁻¹u) = A_gᵀ Y(u)（A_g = orbital_rotation(R) の転置）と変わるので
        M_sym[b] = 1/|G| Σ_g A_g M[p_g⁻¹(b)] A_gᵀ
    m ごと（px だけ、dxy だけ、など）の密度は、この M_sym から作る。電荷密度の格子での平均と違い、
    回転で px が py に混ざることを正しく扱える"""
    perm, Rcs = symmetry_atom_maps(st, rots, trans)
    out = {}
    for l, Ml in M.items():
        acc = np.zeros_like(Ml)
        for p, Rc in zip(perm, Rcs):
            A = orbital_rotation(Rc, l)
            inv = np.empty_like(p)
            inv[p] = np.arange(len(p))                   # inv[b] = 操作で b に移ってくる元の原子
            acc += np.einsum("mi,arij,nj->armn", A, Ml[inv], A)
        out[l] = acc / len(perm)
    return out


def orbital_points_density(M, st, radii, N, dr, subset, atoms=None, diag_only=False):
    """密度行列 M（対称化したもの）から、格子 N の上の電荷密度を作る。
    subset = {l: [m の番号, …]}: 選んだ成分の張る空間への射影
        ρ_S(τ_a + r) = Σ_l Σ_{m,m'∈S_l} M_l[a](|r|)_{mm'} Y_lm(r̂) Y_lm'(r̂)
      （l の全成分を選ぶと成分 l の密度そのもの。m を 1 つだけなら |c_lm|² Y_lm² ）
    atoms: 使う原子の番号（None なら全部）。diag_only=True のときは m ごとの電子数 {(l, m): 値} を返す"""
    N = np.array(N)
    L = np.asarray(st.lattice, float)
    dv = abs(np.linalg.det(L)) / float(np.prod(N))
    rho = None if diag_only else np.zeros(tuple(N), np.float32)
    elec = {}
    nr = next(iter(M.values())).shape[1]
    use = None if atoms is None else set(int(a) for a in atoms)
    for a, gi, v in sphere_points(st, radii, N):
        if len(gi) == 0 or (use is not None and a not in use):
            continue
        dist = np.linalg.norm(v, axis=1)
        u = v / np.where(dist > 1e-12, dist, 1.0)[:, None]
        u[dist <= 1e-12] = [0.0, 0.0, 1.0]
        x = dist / dr
        i0 = np.minimum(np.floor(x).astype(int), nr - 2)
        f = (x - i0)
        val = np.zeros(len(gi))
        for l, ms in subset_of(subset, a).items():
            ms = list(ms)
            if not ms:
                continue
            Y = real_harmonics_normalized(l, u)[:, ms]                       # (点, |S|)
            Ma = M[l][a][:, ms][:, :, ms]                                    # (r, |S|, |S|)
            if diag_only:
                d = np.einsum("rmm->rm", Ma)
                dd = d[i0] * (1 - f)[:, None] + d[i0 + 1] * f[:, None]       # (点, |S|)
                s = (dd * Y * Y).sum(axis=0) * dv
                for j, m in enumerate(ms):
                    elec[(l, m)] = elec.get((l, m), 0.0) + float(s[j])
                continue
            Mi = Ma[i0] * (1 - f)[:, None, None] + Ma[i0 + 1] * f[:, None, None]
            val += np.einsum("pm,pmn,pn->p", Y, Mi, Y)
        if not diag_only:
            rho[gi[:, 0], gi[:, 1], gi[:, 2]] = val
    return elec if diag_only else rho


def orbital_band_density(wc, st, selection, kweights, radii, lmax=3, grid=None, dr=0.03,
                         progress=None, cancel=None, symmetrize=True, fast=True, nthreads=None, sym_method="matrix"):
    """選んだバンドの電荷密度を、原子のまわりの球の中で角運動量 l = 0..lmax の成分に分ける。
      wc        : WavecarFile（または band(s, k, n) と kpts, B, vol, nspin, vtype を持つもの）
      st        : 構造（POSCAR。原子の位置と対称操作に使う）
      selection : {(s, k): [バンド番号 (0 始まり), …]}
      kweights  : k点の重み（合計 1）
      radii     : 原子ごとの球の半径 [Å]
    方法: 球の中心 τ のまわりで ψ(τ + r) を実球面調和関数で展開する。平面波の展開
          e^{i q·r} = 4π Σ_lm i^l j_l(qr) Y_lm(q̂) Y_lm(r̂)  （q = k + G）
      を使うと、c_lm(r) = (4π/√V) i^l Σ_G c_G e^{i q·τ} j_l(qr) Y_lm(q̂) を補間なしで計算できる。
      成分 l の密度 ρ_l(τ + r) = Σ_nk w_k f |Σ_m c_lm(|r|) Y_lm(r̂)|²（f = 2: スピンなし, 1: スピン分極・SOC）
    既約 k点だけの WAVECAR でも正しくなるように、最後に結晶の対称操作で平均する
    （原子 a のまわりの電荷は、等価な原子のまわりに移る）。
    戻り値 dict: rho {l: (N1, N2, N3) float32 [e/Å³]}, label (N1, N2, N3) int16（その点がある球の原子の番号、外は -1）,
                 grid, electrons {l: 球の中の電子数}, total（全空間の擬波動関数の電子数）, …"""
    if grid is None:
        grid, rots, trans = symmetric_grid(st)
    else:
        _, rots, trans = symmetric_grid(st)
    N = np.array(grid)
    L = np.asarray(st.lattice, float)
    tau = np.mod(st.frac, 1.0) @ L
    nat = len(tau)
    rmax = float(np.max(radii))
    # 球の外側の薄い殻（格子の対角線の 1.5 倍）まで展開を延ばしておく。等値面を球の境界でなめらかに切るために使う
    shell = 1.5 * grid_cell_diag(L, N)
    rr = np.arange(0.0, rmax + shell + 2 * dr, dr)
    nr = len(rr)
    ls = list(range(lmax + 1))
    M = {l: np.zeros((nat, nr, 2 * l + 1, 2 * l + 1)) for l in ls}     # 密度行列 Re Σ w c_m c_m'*
    fac = 2.0 if (wc.nspin == 1 and wc.vtype != "ncl") else 1.0
    # 重み 0 の k点（間引きで使わない点）は計算しない
    selection = {sk: bands for sk, bands in selection.items() if bands and kweights[sk[1]] > 0}
    jobs = [(s, k, n) for (s, k), bands in sorted(selection.items()) for n in bands]
    total_norm = 0.0
    done = 0
    cache_k = None
    if fast:
        M, total_norm = _orbital_density_matrices_fast(wc, tau, selection, kweights, rr, ls, fac, len(jobs),
                                                       progress, cancel, nthreads)
        selection = {}
    for (s, k), bands in sorted(selection.items()):
        if not bands:
            continue
        for n in bands:
            if cancel is not None and cancel():
                raise RuntimeError("中止しました。")
            G, C = wc.band(s, k, n)
            if cache_k != k:                                            # k点ごとの表（j_l と Y_lm(q̂)）
                q = (wc.kpts[k][None, :] + G) @ wc.B
                qn = np.linalg.norm(q, axis=1)
                qh = q / np.where(qn > 1e-12, qn, 1.0)[:, None]
                qh[qn <= 1e-12] = [0.0, 0.0, 1.0]
                phase = np.exp(1j * (tau @ q.T))                          # (原子, G)
                K = {}
                for l in ls:
                    jl = spherical_jl(l, qn[:, None] * rr[None, :])         # (G, r)
                    Y = real_harmonics_normalized(l, qh)                   # (G, m)
                    K[l] = (jl[:, :, None] * Y[:, None, :]).reshape(len(G), -1) * (4.0 * np.pi / np.sqrt(wc.vol))
                cache_k = k
            w = kweights[k] * fac
            for comp in C:
                total_norm += w * float(np.vdot(comp, comp).real)
                A = phase * comp[None, :]                                  # (原子, G)
                for l in ls:
                    cl = (A.real @ K[l] + 1j * (A.imag @ K[l])) * (1j ** l)   # (原子, r·m)
                    cl = cl.reshape(nat, nr, 2 * l + 1)
                    M[l] += w * np.einsum("arm,arn->armn", cl, np.conj(cl)).real
            done += 1
            if progress:
                progress(done, len(jobs))
    # 対称化: 密度行列の段階で行う（m ごとの成分も正しくなる）。sym_method="grid" は以前の方法（格子での平均。確認用）
    if symmetrize and sym_method == "matrix":
        M = symmetrize_density_matrices(M, st, rots, trans)
    # 格子点での値
    rho = {l: np.zeros(tuple(N), np.float32) for l in ls}
    label = -np.ones(tuple(N), np.int16)
    for a, gi, v in sphere_points(st, radii, N):
        if len(gi) == 0:
            continue
        dist = np.linalg.norm(v, axis=1)
        u = v / np.where(dist > 1e-12, dist, 1.0)[:, None]
        u[dist <= 1e-12] = [0.0, 0.0, 1.0]
        x = dist / dr
        i0 = np.minimum(np.floor(x).astype(int), nr - 2)
        f = (x - i0)[:, None, None]
        for l in ls:
            Ml = M[l][a, i0] * (1 - f) + M[l][a, i0 + 1] * f                # (点, m, m')
            Y = real_harmonics_normalized(l, u)
            rho[l][gi[:, 0], gi[:, 1], gi[:, 2]] = np.einsum("pm,pmn,pn->p", Y, Ml, Y)
        label[gi[:, 0], gi[:, 1], gi[:, 2]] = a
    # 対称化（球の中の点だけ。以前の方法）
    if symmetrize and sym_method == "grid":
        pts = np.argwhere(label >= 0)
        sym = symmetrize_grid([rho[l] for l in ls], rots, trans, pts)
        for l, v in zip(ls, sym):
            rho[l][pts[:, 0], pts[:, 1], pts[:, 2]] = v
    dv = abs(np.linalg.det(L)) / float(np.prod(N))
    electrons = {l: float(rho[l].sum() * dv) for l in ls}
    out = {"rho": rho, "label": label, "grid": tuple(int(x) for x in N), "electrons": electrons,
           "total": total_norm, "nbands_k": len(jobs), "radii": np.asarray(radii, float), "lmax": lmax,
           "nops": len(rots), "fac": fac, "shell": float(shell)}
    if not symmetrize or sym_method == "matrix":
        # m ごとの表示用: 対称化した密度行列と、m ごとの球の中の電子数
        out["M"] = {l: np.ascontiguousarray(M[l]) for l in ls}
        out["dr"] = float(dr)
        out["electrons_m"] = {f"{l},{m}": v for (l, m), v in orbital_points_density(
            M, st, radii, N, dr, {l: range(2 * l + 1) for l in ls}, diag_only=True).items()}
    return out



# ======================================================================
#  断面（VESTA の Sections）: 等値面で囲まれた部分を、描画範囲の境界（セルの面・原子の球）で切った切り口
# ======================================================================
def grid_cell_diag(L, N):
    """格子の 1 つの小さな平行六面体の最も長い対角線 [Å]"""
    a = np.asarray(L, float) / np.asarray(N, float)[:, None]
    return float(max(np.linalg.norm(s0 * a[0] + s1 * a[1] + a[2]) for s0 in (1, -1) for s1 in (1, -1)))


def clip_tris(T, vals):
    """三角形 T (n, 3, C)（列は位置・法線・値など、線形に補間してよい量）を、頂点の値 vals (n, 3) ≥ 0 の
    部分だけに切る。境界は辺の上で線形補間（三角形 1 つが 0〜2 個になる）"""
    T = np.asarray(T, np.float32)
    vals = np.asarray(vals, np.float32)
    if len(T) == 0:
        return T
    pos = vals >= 0
    npos = pos.sum(axis=1)
    out = [T[npos == 3]]

    def cut(Ti, vi, i, j):
        a, b = vi[:, i], vi[:, j]
        t = (a / np.where(a - b != 0, a - b, 1.0))[:, None]
        return Ti[:, i] * (1 - t) + Ti[:, j] * t

    for k in (1, 2):
        m = npos == k
        if not m.any():
            continue
        Ti, vi, pi = T[m], vals[m], pos[m]
        # 回して、頂点 0 が「1 個だけの側」になるようにする（k = 1: 正が 1 個、k = 2: 負が 1 個）
        lone = pi if k == 1 else ~pi
        r = np.argmax(lone, axis=1)
        idx = (r[:, None] + np.arange(3)[None, :]) % 3
        Ti = np.take_along_axis(Ti, idx[:, :, None], axis=1)
        vi = np.take_along_axis(vi, idx, axis=1)
        p01, p02 = cut(Ti, vi, 0, 1), cut(Ti, vi, 0, 2)
        if k == 1:
            out.append(np.stack([Ti[:, 0], p01, p02], axis=1))
        else:
            out.append(np.stack([p01, Ti[:, 1], Ti[:, 2]], axis=1))
            out.append(np.stack([p01, Ti[:, 2], p02], axis=1))
    return np.concatenate(out, axis=0)


def sphere_signed(P, st, radii, atoms):
    """点 P (n, 3) [Å] の、選んだ原子の球に対する符号付きの距離 max_a (R_a − |P − τ_a|)（球の中で正）。
    周期境界の一番近い像を使う"""
    L = np.asarray(st.lattice, float)
    Li = np.linalg.inv(L)
    f = np.asarray(P, float) @ Li
    s = np.full(len(f), -np.inf)
    frac = np.mod(np.asarray(st.frac, float), 1.0)
    for a in atoms:
        d = f - frac[a]
        d -= np.rint(d)
        s = np.maximum(s, radii[a] - np.linalg.norm(d @ L, axis=1))
    return s


def sphere_signed_grid(st, radii, atoms, N):
    """格子点での sphere_signed（GPU レイキャスティング用）。各原子の球のまわり（境界から margin 以内）だけを
    計算し、それより遠い点は −margin にする（遠い点で要るのは「球の外で、境界から十分離れている」ことだけ。
    margin = 4 × 格子間隔の最大 + 格子の対角線 で、描画で正確な判定に切り替える幅 2 × 格子間隔より広い）"""
    N = np.array(N)
    L = np.asarray(st.lattice, float)
    Li = np.linalg.inv(L)
    margin = 4.0 * float(max(np.linalg.norm(L, axis=1) / N)) + grid_cell_diag(L, N)
    S = np.full(int(np.prod(N)), -margin, np.float32)
    frac = np.mod(np.asarray(st.frac, float), 1.0)
    for a in atoms:
        R = float(radii[a])
        Re = R + margin
        x0 = frac[a]
        ext = np.ceil(Re * np.linalg.norm(Li, axis=0) * N).astype(int) + 1
        c = np.floor(x0 * N).astype(int)
        rg = [np.arange(c[i] - ext[i], c[i] + ext[i] + 2) for i in range(3)]
        g = np.stack(np.meshgrid(*rg, indexing="ij"), axis=-1).reshape(-1, 3)
        dist = np.linalg.norm((g / N - x0) @ L, axis=1)
        m = dist <= Re
        flat = np.ravel_multi_index((g[m] % N).T, tuple(N))
        np.maximum.at(S, flat, (R - dist[m]).astype(np.float32))
    return S.reshape(tuple(N))


def orbital_field_ext(M, st, radii, N, dr, subset, atoms, shell):
    """選んだ原子・成分の密度を、球の中に加えて球の外側の薄い殻（厚さ shell）でも計算した格子データ。
    殻の値は球の中と同じ式（展開）をそのまま外へ延ばしたもので、等値面を球の境界でなめらかに切るためだけに使う
    （表示する値・電子数・保存する CHGCAR は球の中だけの値）。殻が重なるところは、表面に近い方の原子の値"""
    N = np.array(N)
    L = np.asarray(st.lattice, float)
    Li = np.linalg.inv(L)
    nr = next(iter(M.values())).shape[1]
    rlast = (nr - 2) * dr
    F = np.zeros(tuple(N), np.float32)
    best = np.full(tuple(N), np.inf, np.float32)
    frac = np.mod(np.asarray(st.frac, float), 1.0)
    for a in atoms:
        R = float(radii[a])
        Re = min(R + shell, rlast)
        x0 = frac[a]
        ext = np.ceil(Re * np.linalg.norm(Li, axis=0) * N).astype(int) + 1
        c = np.floor(x0 * N).astype(int)
        rg = [np.arange(c[i] - ext[i], c[i] + ext[i] + 2) for i in range(3)]
        g = np.stack(np.meshgrid(*rg, indexing="ij"), axis=-1).reshape(-1, 3)
        v = (g / N - x0) @ L
        dist = np.linalg.norm(v, axis=1)
        m = dist <= Re
        g, v, dist = g[m] % N, v[m], dist[m]
        key = (dist - R).astype(np.float32)
        cur = best[g[:, 0], g[:, 1], g[:, 2]]
        take = key < cur
        if not take.any():
            continue
        g, v, dist, key = g[take], v[take], dist[take], key[take]
        u = v / np.where(dist > 1e-12, dist, 1.0)[:, None]
        u[dist <= 1e-12] = [0.0, 0.0, 1.0]
        x = dist / dr
        i0 = np.minimum(np.floor(x).astype(int), nr - 2)
        fr = x - i0
        val = np.zeros(len(g))
        for l, ms in subset_of(subset, a).items():
            ms = list(ms)
            if not ms:
                continue
            Y = real_harmonics_normalized(l, u)[:, ms]
            Ma = M[l][a][:, ms][:, :, ms]
            Mi = Ma[i0] * (1 - fr)[:, None, None] + Ma[i0 + 1] * fr[:, None, None]
            val += np.einsum("pm,pmn,pn->p", Y, Mi, Y)
        F[g[:, 0], g[:, 1], g[:, 2]] = val
        best[g[:, 0], g[:, 1], g[:, 2]] = key
    return F


class OrbitalGridCache:
    """軌道の選択を切り替えるたびの計算をなくすための、原子ごとの前計算（計算結果ごとに 1 回だけ作る）。
    原子 a の球と外側の薄い殻（orbital_field_ext と同じ範囲）の格子点で、成分の組 (m, m') ごとの値
        P_l[a](m, m') = (2 − δ_mm') M_l[a](|r|)_mm' Y_lm(r̂) Y_lm'(r̂)      (m ≤ m')
    を持つ。成分の集合 S を選んだときの値は Σ_{m ≤ m' ∈ S} P_l[a](m, m')（= Σ_{m,m'∈S} M Y Y と同じ）。
    lsets = {元素: [l…]} を渡すと、その元素にない l は作らない（使わないのでメモリの節約）"""

    def __init__(self, M, st, radii, N, dr, shell, label, lsets=None):
        self.N = tuple(int(x) for x in N)
        N = np.array(N)
        L = np.asarray(st.lattice, float)
        Li = np.linalg.inv(L)
        nr = next(iter(M.values())).shape[1]
        rlast = (nr - 2) * dr
        frac = np.mod(np.asarray(st.frac, float), 1.0)
        lab = np.asarray(label).ravel()
        self.atoms = {}
        for a in range(len(frac)):
            R = float(radii[a])
            Re = min(R + shell, rlast)
            x0 = frac[a]
            ext = np.ceil(Re * np.linalg.norm(Li, axis=0) * N).astype(int) + 1
            c = np.floor(x0 * N).astype(int)
            rg = [np.arange(c[i] - ext[i], c[i] + ext[i] + 2) for i in range(3)]
            g = np.stack(np.meshgrid(*rg, indexing="ij"), axis=-1).reshape(-1, 3)
            v = (g / N - x0) @ L
            dist = np.linalg.norm(v, axis=1)
            m = dist <= Re
            g, v, dist = g[m] % N, v[m], dist[m]
            # 同じ格子点が周期的な像で 2 回出ることはない（球 + 殻はセルより小さい）が、念のため 1 回に
            flat = np.ravel_multi_index(g.T, tuple(N))
            flat, first = np.unique(flat, return_index=True)
            v, dist = v[first], dist[first]
            u = v / np.where(dist > 1e-12, dist, 1.0)[:, None]
            u[dist <= 1e-12] = [0.0, 0.0, 1.0]
            x = dist / dr
            i0 = np.minimum(np.floor(x).astype(int), nr - 2)
            fr = x - i0
            ls = range(4) if lsets is None else lsets.get(st.symbols[a], range(4))
            pairs = {}
            for l in ls:
                if l not in M:
                    continue
                Y = real_harmonics_normalized(l, u)                          # (点, 2l+1)
                Ma = M[l][a]
                Mi = Ma[i0] * (1 - fr)[:, None, None] + Ma[i0 + 1] * fr[:, None, None]
                idx, arr = [], []
                for m1 in range(2 * l + 1):
                    for m2 in range(m1, 2 * l + 1):
                        f2 = 1.0 if m1 == m2 else 2.0
                        idx.append((m1, m2))
                        arr.append((f2 * Mi[:, m1, m2] * Y[:, m1] * Y[:, m2]).astype(np.float32))
                pairs[l] = (idx, np.array(arr, np.float32))
            self.atoms[a] = {"flat": flat.astype(np.int64 if np.prod(N) > 2**31 - 1 else np.int32), "key": (dist - R).astype(np.float32),
                             "inside": lab[flat] == a, "pairs": pairs}
        self._own = {}

    def values(self, a, sub):
        """原子 a の格子点での、成分 sub = {l: [m…]} の値（float64）"""
        A = self.atoms[a]
        val = np.zeros(len(A["flat"]))
        for l, ms in sub.items():
            if not ms or l not in A["pairs"]:
                continue
            idx, arr = A["pairs"][l]
            sel = set(int(m) for m in ms)
            rows = [k for k, (m1, m2) in enumerate(idx) if m1 in sel and m2 in sel]
            if rows:
                val += arr[rows].sum(axis=0, dtype=np.float64)
        return val

    def owners(self, atoms):
        """選んだ原子の殻が重なるところで、表面に近い方の原子を使う（orbital_field_ext と同じ規則）。
        戻り値 {原子: その原子の点のうち使う点の bool}（原子の組ごとに覚えておく）"""
        key = tuple(sorted(int(a) for a in atoms))
        if key in self._own:
            return self._own[key]
        best = np.full(int(np.prod(self.N)), np.inf, np.float32)
        own = np.full(int(np.prod(self.N)), -1, np.int32)
        for a in key:
            A = self.atoms[a]
            take = A["key"] < best[A["flat"]]
            best[A["flat"][take]] = A["key"][take]
            own[A["flat"][take]] = a
        out = {a: own[self.atoms[a]["flat"]] == a for a in key}
        if len(self._own) > 16:
            self._own.clear()
        self._own[key] = out
        return out

    def field_ext(self, sub, atoms):
        """orbital_field_ext と同じ値（球 + 殻）。sub は AtomSubsets または共通の {l: [m…]}"""
        F = np.zeros(int(np.prod(self.N)), np.float32)
        own = self.owners(atoms)
        for a in own:
            w = own[a]
            F[self.atoms[a]["flat"][w]] = self.values(a, subset_of(sub, a))[w]
        return F.reshape(self.N)

    def inside_density(self, sub, atoms):
        """orbital_points_density と同じ値（球の中だけ。球の外は 0）"""
        rho = np.zeros(int(np.prod(self.N)), np.float32)
        for a in atoms:
            A = self.atoms[a]
            w = A["inside"]
            rho[A["flat"][w]] = self.values(a, subset_of(sub, a))[w]
        return rho.reshape(self.N)

    def nbytes(self):
        return sum(A["flat"].nbytes + A["key"].nbytes + A["inside"].nbytes
                   + sum(arr.nbytes for _, arr in A["pairs"].values()) for A in self.atoms.values())


_ICO = {}


def icosphere(level=4):
    """単位球の三角形分割（正二十面体を level 回細分）。戻り値 (頂点 (n, 3), 三角形 (m, 3))"""
    if level in _ICO:
        return _ICO[level]
    t = (1 + 5 ** 0.5) / 2
    V = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    Fc = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
          [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
          [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(level):
        mid = {}
        nf = []

        def m(i, j):
            k = (min(i, j), max(i, j))
            if k not in mid:
                p = V[i] + V[j]
                V.append(p / np.linalg.norm(p))
                mid[k] = len(V) - 1
            return mid[k]
        for a, b, c in Fc:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        Fc = nf
    _ICO[level] = (np.array(V), np.array(Fc))
    return _ICO[level]


def _box_clip(T, L, ncell):
    """三角形を描画範囲（分率座標 0 ≤ f_i ≤ n_i）の中だけに切る"""
    Li = np.linalg.inv(np.asarray(L, float))
    for i in range(3):
        if not len(T):
            break
        f = (T[:, :, :3].reshape(-1, 3) @ Li)[:, i].reshape(-1, 3)
        T = clip_tris(T, f + 1e-6)
        f = (T[:, :, :3].reshape(-1, 3) @ Li)[:, i].reshape(-1, 3)
        T = clip_tris(T, ncell[i] - f + 1e-6)
    return T


def section_faces(F, L, ncell, level, absval=False, st=None, radii=None, atoms=None):
    """描画範囲（セル ncell[0]×ncell[1]×ncell[2]）の 6 つの面のうち、値 ≥ level の部分（VESTA の Sections）。
    F: 1 セル分の周期的な格子データ。absval=True なら |値| ≥ level（負の等値面の断面）。
    st, radii, atoms を渡すと、選んだ原子の球の中だけにする（WAVECAR の軌道分解）。
    戻り値 (n, 3, 7): 各頂点 [x, y, z, 外向き法線 3, 値]"""
    F = np.asarray(F, np.float32)
    N = np.array(F.shape)
    L = np.asarray(L, float)
    out = []
    for ax in range(3):
        b, c = [i for i in range(3) if i != ax]
        face = np.take(F, 0, axis=ax)                      # (N_b, N_c): f_ax = 0（= n_ax、周期）
        if absval:
            face = np.abs(face)
        fe = np.pad(face, ((0, 1), (0, 1)), mode="wrap")   # 正方形の 4 隅
        q = np.stack([fe[:-1, :-1], fe[1:, :-1], fe[1:, 1:], fe[:-1, 1:]], axis=-1)   # (N_b, N_c, 4)
        sel = np.argwhere(q.max(axis=-1) >= level)
        if len(sel) == 0:
            continue
        nrm = np.cross(L[b], L[c])
        nrm = nrm / np.linalg.norm(nrm)
        if nrm @ L[ax] < 0:
            nrm = -nrm
        for side, sgn, fax in ((0, -1.0, 0.0), (1, 1.0, float(ncell[ax]))):
            # 面の上のセルの並び（n_b × n_c 枚）
            tiles = np.array([[tb, tc] for tb in range(ncell[b]) for tc in range(ncell[c])])
            jb = (sel[None, :, 0] + tiles[:, None, 0] * N[b]).reshape(-1)
            jc = (sel[None, :, 1] + tiles[:, None, 1] * N[c]).reshape(-1)
            qv = np.tile(q[sel[:, 0], sel[:, 1]], (len(tiles), 1))           # (s, 4)
            corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            fr = np.zeros((len(jb), 4, 3))
            fr[:, :, ax] = fax
            fr[:, :, b] = (jb[:, None] + corner[None, :, 0]) / N[b]
            fr[:, :, c] = (jc[:, None] + corner[None, :, 1]) / N[c]
            P = fr @ L
            Tq = np.concatenate([P, np.broadcast_to(sgn * nrm, P.shape), qv[:, :, None]], axis=2)
            T = np.concatenate([Tq[:, [0, 1, 2]], Tq[:, [0, 2, 3]]], axis=0).astype(np.float32)
            T = clip_tris(T, T[:, :, 6] - level)
            if st is not None and len(T):
                s = sphere_signed(T[:, :, :3].reshape(-1, 3), st, radii, atoms).reshape(-1, 3)
                T = clip_tris(T, s)
            if len(T):
                out.append(T)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3, 7), np.float32)


def sphere_caps(M, st, radii, dr, subset, atoms, level, ncell, ico_level=5):
    """選んだ原子の球の表面のうち、値 ≥ level の部分（球の境界での切り口）。描画範囲の外は切る。
    球の表面の値は、密度行列 M から r = R（球の半径）で直接計算する。戻り値 (n, 3, 7)"""
    Vu, Fu = icosphere(ico_level)
    L = np.asarray(st.lattice, float)
    nr = next(iter(M.values())).shape[1]
    frac = np.mod(np.asarray(st.frac, float), 1.0)
    out = []
    for a in atoms:
        R = float(radii[a])
        x = R / dr
        i0 = min(int(np.floor(x)), nr - 2)
        fr = x - i0
        val = np.zeros(len(Vu))
        for l, ms in subset_of(subset, a).items():
            ms = list(ms)
            if not ms:
                continue
            Y = real_harmonics_normalized(l, Vu)[:, ms]
            Ma = M[l][a][:, ms][:, :, ms]
            Mi = Ma[i0] * (1 - fr) + Ma[i0 + 1] * fr
            val += np.einsum("pm,mn,pn->p", Y, Mi, Y)
        if val.max() < level:
            continue
        keep = (val[Fu] >= level).any(axis=1)
        if not keep.any():
            continue
        Fk = Fu[keep]
        base = np.concatenate([Vu[Fk] * R, Vu[Fk], val[Fk][:, :, None]], axis=2)     # (m, 3, 7)、中心が原点
        base = clip_tris(base.astype(np.float32), base[:, :, 6] - level)
        if not len(base):
            continue
        # 描画範囲に入る像（中心が範囲の外でも、球が範囲にかかるもの）
        Li = np.linalg.inv(L)
        rf = R * np.linalg.norm(Li, axis=0)                 # 球の分率座標での広がり
        imgs = []
        for i in range(-1, ncell[0] + 1):
            for j in range(-1, ncell[1] + 1):
                for k in range(-1, ncell[2] + 1):
                    c = frac[a] + (i, j, k)
                    if np.all(c + rf >= 0) and np.all(c - rf <= np.asarray(ncell)):
                        imgs.append(c @ L)
        for cpos in imgs:
            T = base.copy()
            T[:, :, :3] += cpos.astype(np.float32)
            T = _box_clip(T, L, ncell)
            if len(T):
                out.append(T)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 3, 7), np.float32)


SECTION_MODES = ("B-G-R", "R-G-B", "グレースケール", "等値面と同じ色")


def section_colors(v, vmin, vmax, mode, iso_rgb=(1.0, 1.0, 0.0)):
    """断面の色（VESTA の Sections と同じ考え方: 飽和レベル vmin〜vmax の間の位置 T で色を決める。
    範囲の外は端の色。B-G-R は T = 0 で青、0.5 で緑、1 で赤）"""
    v = np.asarray(v, float)
    if mode == "等値面と同じ色":
        return np.broadcast_to(np.asarray(iso_rgb, float), v.shape + (3,)).copy()
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-30), 0.0, 1.0)
    if mode == "グレースケール":
        return np.repeat(t[..., None], 3, axis=-1)
    if mode == "R-G-B":
        t = 1.0 - t
    r = np.clip(2 * t - 1, 0, 1)
    g = 1 - np.abs(2 * t - 1)
    b = np.clip(1 - 2 * t, 0, 1)
    return np.stack([r, g, b], axis=-1)


def section_vertices(T, vmin, vmax, mode, alpha, iso_rgb=(1.0, 1.0, 0.0)):
    """断面の三角形 (n, 3, 7) → 描画用の頂点配列 (3n, 10)"""
    out = np.empty((len(T), 3, 10), np.float32)
    out[:, :, :6] = T[:, :, :6]
    out[:, :, 6:9] = section_colors(T[:, :, 6], vmin, vmax, mode, iso_rgb)
    out[:, :, 9] = alpha
    return out.reshape(-1, 10)


def ascii_comment(s):
    """CHGCAR の 1 行目用: 日本語の説明を英語の表記にする（ファイルは ASCII で書くので、日本語は「?」になってしまう）"""
    import re
    s = str(s).replace("\n", " ")
    s = re.sub(r"（k点 (\d+) 倍間引き）", r" (k-mesh thinned x\1)", s)
    s = re.sub(r"（(\d+) / (\d+) 状態）", r" (\1 of \2 states)", s)
    for a, b in (("バンド", "bands"), ("E − E_F", "E-E_F"), ("〜", " to "), ("–", "-"), ("、", ", "), ("（", " ("),
                 ("）", ")"), ("−", "-"), ("²", "2"), ("³", "3"), ("orbital", "orbital"), ("なし", "none")):
        s = s.replace(a, b)
    s = "".join(c if ord(c) < 128 else "_" for c in s)
    return " ".join(s.split())


def write_chgcar(path, st, rho, comment="VASP Result Viewer"):
    """CHGCAR 形式で書く（ρ × 体積 を x が最も速い順に 5 個ずつ。VESTA やこのアプリで開ける）"""
    L = np.asarray(st.lattice, float)
    vol = abs(np.linalg.det(L))
    N = rho.shape
    vals = (np.asarray(rho, np.float64) * vol).transpose(2, 1, 0).ravel()
    with open(path, "w", encoding="ascii", errors="replace", newline="\n") as f:
        f.write(ascii_comment(comment) + "\n   1.00000000000000\n")
        for v in L:
            f.write("  %20.14f %20.14f %20.14f\n" % tuple(v))
        f.write("   " + "   ".join(st.species) + "\n   " + "   ".join(str(c) for c in st.counts) + "\nDirect\n")
        for x in np.asarray(st.frac, float):
            f.write("  %.10f  %.10f  %.10f\n" % tuple(x))
        f.write("\n%5d%5d%5d\n" % tuple(N))
        for i0 in range(0, len(vals), 500000):
            chunk = vals[i0:i0 + 500000]
            txt = [" ".join("%.11E" % x for x in chunk[j:j + 5]) for j in range(0, len(chunk), 5)]
            f.write("\n".join(" " + t for t in txt) + "\n")


# ======================================================================
#  読み込んだデータの保存（キャッシュとライブラリ）
# ======================================================================
STORE_MMAP_BYTES = 64 * 1024 * 1024       # これより大きい配列は、開くときにメモリに全部は読まない（必要な所だけ読む）
CACHE_LIMIT_GB = 20.0                     # キャッシュの上限（超えたら古いものから消す）


def app_data_dir():
    """アプリのデータを置くフォルダ（Windows: %LOCALAPPDATA%\\VASPResultViewer）。
    環境変数 VASP_VIEWER_DATA で変えられる"""
    if os.environ.get("VASP_VIEWER_DATA"):
        return os.environ["VASP_VIEWER_DATA"]
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "VASPResultViewer")


def _bundle_encode(obj, arrays):
    """保存できる形（JSON と配列）に直す。保存できないもの（ファイルの読み手など）は捨てる"""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return {"__list__": [_bundle_encode(x, arrays) for x in obj.tolist()]}
        name = f"a{len(arrays):05d}.npy"
        arrays[name] = obj
        return {"__nd__": name}
    if isinstance(obj, Structure):
        return {"__st__": {"comment": obj.comment, "lattice": obj.lattice.tolist(), "species": list(obj.species),
                           "counts": [int(c) for c in obj.counts], "frac": obj.frac.tolist()}}
    if isinstance(obj, dict):
        return {"__dict__": [[_bundle_encode(k, arrays), _bundle_encode(v, arrays)] for k, v in obj.items()
                             if not (isinstance(k, str) and k.startswith("_"))]}
    if isinstance(obj, tuple):
        return {"__tuple__": [_bundle_encode(x, arrays) for x in obj]}
    if isinstance(obj, list):
        return [_bundle_encode(x, arrays) for x in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return {"__f__": repr(float(obj))}
    if isinstance(obj, (complex, np.complexfloating)):
        return {"__c__": [float(obj.real), float(obj.imag)]}
    if obj is None or isinstance(obj, str):
        return obj
    return {"__skip__": type(obj).__name__}


def _bundle_decode(obj, folder):
    if isinstance(obj, list):
        return [_bundle_decode(x, folder) for x in obj]
    if not isinstance(obj, dict):
        return obj
    if "__nd__" in obj:
        p = os.path.join(folder, obj["__nd__"])
        big = os.path.getsize(p) > STORE_MMAP_BYTES
        return np.load(p, mmap_mode="c" if big else None, allow_pickle=False)
    if "__st__" in obj:
        s = obj["__st__"]
        return Structure(s["comment"], s["lattice"], s["species"], s["counts"], s["frac"])
    if "__dict__" in obj:
        return {_bundle_decode(k, folder): _bundle_decode(v, folder) for k, v in obj["__dict__"]}
    if "__tuple__" in obj:
        return tuple(_bundle_decode(x, folder) for x in obj["__tuple__"])
    if "__list__" in obj:
        return [_bundle_decode(x, folder) for x in obj["__list__"]]
    if "__f__" in obj:
        return float(obj["__f__"])
    if "__c__" in obj:
        return complex(*obj["__c__"])
    if "__skip__" in obj:
        return None
    return obj


def _hashable(k):
    if isinstance(k, list):
        return tuple(_hashable(x) for x in k)
    return k


def save_bundle(folder, obj, meta):
    """obj（辞書・配列・構造など）を folder に保存する。途中で失敗しても壊れたものが残らないよう、
    一時フォルダに書いてから名前を変える"""
    import json
    import shutil
    tmp = folder + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    arrays = {}
    enc = _bundle_encode(obj, arrays)
    size = 0
    for name, a in arrays.items():
        np.save(os.path.join(tmp, name), np.ascontiguousarray(a), allow_pickle=False)
        size += os.path.getsize(os.path.join(tmp, name))
    with open(os.path.join(tmp, "data.json"), "w", encoding="utf-8") as f:
        json.dump(enc, f, ensure_ascii=False)
    meta = dict(meta)
    meta["bytes"] = size + os.path.getsize(os.path.join(tmp, "data.json"))
    meta["app_version"] = APP_VERSION
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.replace(tmp, folder)
    return meta


def load_bundle(folder):
    import json
    with open(os.path.join(folder, "data.json"), encoding="utf-8") as f:
        enc = json.load(f)
    obj = _bundle_decode(enc, folder)
    return obj, read_bundle_meta(folder)


def read_bundle_meta(folder):
    import json
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def file_signature(paths):
    """ファイルの (絶対パス, 大きさ, 更新時刻) の一覧。ないファイルは飛ばす"""
    out = []
    for p in paths:
        if p and os.path.isfile(p):
            s = os.stat(p)
            out.append([os.path.abspath(p), int(s.st_size), int(s.st_mtime_ns)])
    return out


class DataStore:
    """キャッシュ（読み込んだファイルの中身を自動で保存し、同じファイルなら次からそれを読む）と
    ライブラリ（利用者が名前を付けて保存し、元のファイルがなくても開ける）"""

    def __init__(self, root=None):
        self.root = root or app_data_dir()
        self.cache_dir = os.path.join(self.root, "cache")
        self.lib_dir = os.path.join(self.root, "library")
        self.limit_gb = CACHE_LIMIT_GB
        self.enabled = True
        self._lock = None
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.lib_dir, exist_ok=True)
            self._load_settings()
        except Exception:
            self.enabled = False

    # ---- 設定 ----
    def _load_settings(self):
        import json
        p = os.path.join(self.root, "settings.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                s = json.load(f)
            self.limit_gb = float(s.get("cache_limit_gb", self.limit_gb))
            self.enabled = bool(s.get("cache_enabled", True))

    def save_settings(self):
        import json
        with open(os.path.join(self.root, "settings.json"), "w", encoding="utf-8") as f:
            json.dump({"cache_limit_gb": self.limit_gb, "cache_enabled": self.enabled}, f)

    # ---- キャッシュ ----
    @staticmethod
    def key(kind, paths, params=None):
        import hashlib
        import json
        sig = file_signature(paths)
        if not sig:
            return None
        txt = json.dumps([kind, sig, params], sort_keys=True, default=str)
        return kind + "_" + hashlib.sha1(txt.encode("utf-8")).hexdigest()[:24]

    def cache_get(self, kind, paths, params=None):
        """キャッシュにあれば中身を返す（なければ None）。元のファイルが変わっていれば別のキーになるので使われない"""
        if not self.enabled:
            return None
        k = self.key(kind, paths, params)
        if k is None:
            return None
        folder = os.path.join(self.cache_dir, k)
        if not os.path.isfile(os.path.join(folder, "meta.json")):
            return None
        try:
            obj, meta = load_bundle(folder)
            os.utime(os.path.join(folder, "meta.json"))          # 最後に使った時刻（古いものから消すため）
            return obj
        except Exception:
            return None

    def cache_put(self, kind, paths, params, obj, background=True):
        """読み込んだ中身をキャッシュに保存する（既定では裏で書く）"""
        if not self.enabled:
            return
        k = self.key(kind, paths, params)
        if k is None:
            return
        folder = os.path.join(self.cache_dir, k)
        sig = file_signature(paths)

        def work():
            try:
                save_bundle(folder, obj, {"kind": kind, "sources": sig, "params": str(params),
                                          "created": _now_text()})
                self.prune()
            except Exception:
                pass
        if background:
            import threading
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def entries(self, where):
        """[(フォルダ, meta)]（新しい順）"""
        base = self.cache_dir if where == "cache" else self.lib_dir
        out = []
        if not os.path.isdir(base):
            return out
        for name in os.listdir(base):
            folder = os.path.join(base, name)
            if name.endswith(".tmp") or not os.path.isfile(os.path.join(folder, "meta.json")):
                continue
            try:
                meta = read_bundle_meta(folder)
            except Exception:
                continue
            meta["_used"] = os.path.getmtime(os.path.join(folder, "meta.json"))
            out.append((folder, meta))
        out.sort(key=lambda fm: fm[1]["_used"], reverse=True)
        return out

    def cache_bytes(self):
        return sum(int(m.get("bytes", 0)) for _, m in self.entries("cache"))

    def prune(self):
        """キャッシュが上限を超えていたら、最後に使ったのが古いものから消す"""
        import shutil
        ent = self.entries("cache")
        total = sum(int(m.get("bytes", 0)) for _, m in ent)
        limit = self.limit_gb * 1024 ** 3
        for folder, m in reversed(ent):
            if total <= limit:
                break
            shutil.rmtree(folder, ignore_errors=True)
            total -= int(m.get("bytes", 0))

    def clear_cache(self):
        import shutil
        for folder, _ in self.entries("cache"):
            shutil.rmtree(folder, ignore_errors=True)

    # ---- ライブラリ ----
    def lib_save(self, name, kind, obj, info):
        import time as _t
        eid = f"{_t.strftime('%Y%m%d_%H%M%S')}_{abs(hash((name, _t.time()))) % 10 ** 6:06d}"
        meta = {"name": name, "kind": kind, "created": _now_text()}
        meta.update(info)
        return save_bundle(os.path.join(self.lib_dir, eid), obj, meta)

    def lib_load(self, folder):
        return load_bundle(folder)

    def lib_rename(self, folder, name):
        import json
        meta = read_bundle_meta(folder)
        meta["name"] = name
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    def lib_delete(self, folder):
        import shutil
        shutil.rmtree(folder)


def _now_text():
    import time as _t
    return _t.strftime("%Y-%m-%d %H:%M")



# ======================================================================
#  輸送特性（VASP 6.5 以降の電子フォノン輸送: vaspout.h5）
# ======================================================================
TRANSPORT_GROUP = "results/electron_phonon/electrons"      # py4vasp の定義（_raw/definition.py）と同じ場所
TRANSPORT_QUANTITIES = {
    # 名前: (h5 のデータセット名, 表示名, 単位, 表示用の倍率)
    "seebeck": ("seebeck", "Seebeck coefficient", "μV/K", 1.0),
    "sigma": ("e_conductivity", "Electrical conductivity", "S/m", 1.0),
    "kappa_e": ("e_t_conductivity", "Electronic thermal conductivity", "W/(m K)", 1.0),
    "mobility": ("mobility", "Mobility", "cm$^2$/(V s)", 1.0),
}


def _h5_scalar(x):
    x = np.asarray(x)
    v = x.reshape(-1)[0] if x.size else None
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode(errors="replace").strip()
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_transport_h5(path):
    """vaspout.h5 から電子フォノン輸送の結果を読む。
    アキュムレータ（散乱近似・キャリア密度などの組み合わせ）ごとに transport_N グループがあり、
    各テンソルは (温度, 3, 3)（デカルト座標 x, y, z）。"""
    import h5py
    with h5py.File(path, "r") as f:
        if TRANSPORT_GROUP not in f:
            raise ValueError("この vaspout.h5 には電子フォノン輸送の結果 "
                             f"（{TRANSPORT_GROUP}）がありません。\nELPH_MODE = TRANSPORT の計算の出力を選んでください。")
        g = f[TRANSPORT_GROUP]
        lattice = None
        for base in ("results/positions", "intermediate/ion_dynamics"):
            if f"{base}/lattice_vectors" in f:
                lat = np.array(f[f"{base}/lattice_vectors"], float)
                if lat.ndim == 3:
                    lat = lat[-1]
                sc = float(np.asarray(f[f"{base}/scale"]).reshape(-1)[0]) if f"{base}/scale" in f else 1.0
                lattice = lat * sc
                break
        species = []
        if "results/positions/ion_types" in f:
            ty = [(_t.decode() if isinstance(_t, bytes) else str(_t)).strip()
                  for _t in np.asarray(f["results/positions/ion_types"]).reshape(-1)]
            nt = np.asarray(f["results/positions/number_ion_types"]).reshape(-1)
            species = list(zip(ty, [int(x) for x in nt]))
        # 化学ポテンシャルの指定方法（キャリア密度 / 化学ポテンシャル / セルあたりのキャリア数）
        mu_tag, mu_vals, mu_label = None, None, ""
        cp = g["chemical_potential"] if "chemical_potential" in g else None
        chem = {}
        if cp is not None:
            for key, tag, lab in (("selfen_carrier_den", "selfen_carrier_den", "Carrier density (cm$^{-3}$)"),
                                  ("selfen_mu", "selfen_mu", "Chemical potential (eV)"),
                                  ("selfen_carrier_per_cell", "selfen_carrier_per_cell", "Carriers per cell")):
                if key in cp and mu_tag is None:
                    mu_tag, mu_vals, mu_label = tag, np.asarray(cp[key], float).reshape(-1), lab
            for key in ("efermi", "muij", "nij", "temperatures"):
                if key in cp:
                    chem[key] = np.asarray(cp[key], float)
        ids = sorted(int(k.split("_")[1]) for k in g.keys()
                     if k.startswith("transport_") and k.split("_")[1].isdigit())
        inst = []
        for n in ids:
            name = f"transport_{n}"
            if name not in g:
                continue
            t = g[name]
            d = {"id": n, "temps": np.asarray(t["temps"], float).reshape(-1)}
            for key, (ds, _, _, _) in TRANSPORT_QUANTITIES.items():
                if ds in t:
                    a = np.asarray(t[ds], float)
                    d[key] = a.reshape(-1, 3, 3)
            d["approx"] = _h5_scalar(t["scattering_approximation"]) if "scattering_approximation" in t else ""
            d["mu_index"] = None
            if "id_idx" in t:
                ii = np.asarray(t["id_idx"]).reshape(-1)
                if len(ii) >= 3:
                    d["mu_index"] = int(ii[2]) - 1          # py4vasp と同じ: 3 番目が化学ポテンシャルの番号
            d["mu_value"] = (float(mu_vals[d["mu_index"]]) if (mu_vals is not None and d["mu_index"] is not None
                                                               and 0 <= d["mu_index"] < len(mu_vals)) else None)
            se = f"self_energy_{n}"
            for key, ds in (("nbands_sum", "nbands_sum"), ("delta", "delta")):
                d[key] = _h5_scalar(g[se][ds]) if (se in g and ds in g[se]) else None
            inst.append(d)
    if not inst:
        raise ValueError("輸送のアキュムレータ（transport_N）が 1 つもありません。")
    # ゼーベック係数の単位: py4vasp は V/K としている。値の大きさから μV/K で書かれていると
    # 判断できる場合（|S| の中央値 > 0.01）は換算しない。
    allS = np.concatenate([np.abs(np.diagonal(d["seebeck"], axis1=1, axis2=2)).ravel()
                           for d in inst if "seebeck" in d] or [np.zeros(1)])
    med = float(np.median(allS[np.isfinite(allS)])) if np.isfinite(allS).any() else 0.0
    s_unit_in = "μV/K" if med > 1e-2 else "V/K"
    if s_unit_in == "V/K":
        for d in inst:
            if "seebeck" in d:
                d["seebeck"] = d["seebeck"] * 1e6
    return {"lattice": lattice, "species": species, "instances": inst, "mu_tag": mu_tag,
            "mu_values": mu_vals, "mu_label": mu_label, "chem": chem, "seebeck_unit_in_file": s_unit_in}


def tensor_along(T, n):
    """テンソル T (…, 3, 3) の、単位ベクトル n 方向の成分 n·T·n"""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return np.einsum("i,...ij,j->...", n, T, n)


def tensor_average(T):
    """等方平均 (T_xx + T_yy + T_zz) / 3（py4vasp の isotropic と同じ）"""
    return np.trace(T, axis1=-2, axis2=-1) / 3.0


def transport_label(d, mu_tag):
    """アキュムレータの表示名"""
    s = [f"#{d['id']}", str(d.get("approx") or "")]
    if d.get("mu_value") is not None:
        v = d["mu_value"]
        if mu_tag == "selfen_carrier_den":
            s.append(("n = %.2e cm^-3 (hole)" % abs(v)) if v < 0 else ("n = %.2e cm^-3 (electron)" % v))
        elif mu_tag == "selfen_mu":
            s.append("μ = %.4f eV" % v)
        else:
            s.append("carriers/cell = %g" % v)
    if d.get("nbands_sum"):
        s.append(f"nbands_sum = {d['nbands_sum']}")
    if d.get("delta"):
        s.append(f"δ = {d['delta']}")
    return "  ".join(x for x in s if x)


# ======================================================================
#  3D 描画（電荷密度・フェルミ面で共通）
#  シェーダー・光源・正射影・4x MSAA・マウス操作は VASP Input Generator の結晶構造表示と同じ。
#  三角形メッシュ（等値面）の描画を追加している。
#
#  scene（描画する内容）は dict:
#    bg       : 背景色 (r, g, b)
#    spheres  : (N, 7) [x, y, z, 半径, r, g, b] または None
#    cyls     : (M, 10) [始点 3, 終点 3, 半径, r, g, b] または None
#    lines    : [(頂点 (K, 3)（線分ごとに 2 点）, (r, g, b), 線幅), ...]
#    mesh     : (V, 10) [x, y, z, 法線 3, r, g, b, a]（不透明な三角形。3 頂点ごとに 1 枚）
#    mesh_t   : 同じ形式（半透明な三角形）
#    mesh_offsets : (K, 3) mesh / mesh_t をずらして K 回描く（スーパーセル。インスタンス描画）。省略で 1 回
#    mesh_s / mesh_s_t : mesh と同じ形式で、ずらさずに 1 回だけ描く三角形（断面など）
#    volume   : GPU レイキャスティングで描く等値面（dict。ChargeTab._volume_scene を参照）
#    labels   : (文字列のリスト, 位置 (N, 3), 手前へのずらし量 (N,)) または None（原子ラベルなど）
#    triad    : (3 本のベクトル, 3 個の名前) または None（左下の軸表示）
#    center   : 回転の中心 (3,)     radius : 全体を囲む半径
#    fit      : 全体表示のときに収める点 (P, 3)
# ======================================================================
_VS_MESH = """
#version 330
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
layout(location=2) in vec4 color;
layout(location=3) in vec3 offs;            // スーパーセルの各セルのずれ（インスタンス描画。なければ 0）
uniform mat4 view;
uniform mat4 proj;
out vec3 v_n;
out vec4 v_color;
void main() {
    v_n = mat3(view) * nrm;
    v_color = color;
    gl_Position = proj * view * vec4(pos + offs, 1.0);
}
"""
_FS_MESH = """
#version 330
in vec3 v_n;
in vec4 v_color;
uniform vec3 light;
out vec4 frag;
void main() {
    vec3 n = normalize(v_n);
    if (n.z < 0.0) n = -n;                      // 両面を同じように照らす（正射影なので視線は +z）
    vec3 L = normalize(light);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 40.0);
    frag = vec4(v_color.rgb * (0.30 + 0.72 * diff) + vec3(0.30) * spec, v_color.a);
}
"""
LIGHT_DIR = (-0.45, 0.55, 0.70)        # 左上手前からの光（結晶構造表示と同じ）


# ---- GPU レイキャスティング（電荷密度の等値面）----
# 描画範囲（セル・スーパーセルの平行六面体）の奥側の面を描き、画素ごとに視線を手前から奥へたどって、
# 「値 ≥ レベル（かつ軌道分解の球の中）」の領域に最初に入る位置を求める。
#   ・データは 1 セル分の 3D テクスチャ（周期的にくり返して読む）。スーパーセルでもデータは増えない
#   ・値は格子点の値の 3 重線形補間（三角形の方法と同じ補間）。法線は中心差分の勾配
#   ・工夫 1（空の部分を飛ばす）: 4×4×4 格子のブロックごとに値の最大・最小と球の中かどうかを用意し、
#     面がありえないブロックは一気に飛ばす
#   ・工夫 2（大まかに調べてから絞る）: 半格子ごとに調べ、入ったら二分法で位置を決める
#   ・入った所が描画範囲の面・球の境界なら、その切り口を値に応じた色で塗る（VESTA の Sections）
#   ・奥行き（gl_FragDepth）を書くので、原子・結合との前後関係は奥行きの判定でそのまま正しくなる
_VS_VOL = """
#version 330
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
uniform mat4 view;
uniform mat4 proj;
out vec3 w_pos;
out vec3 w_nrm;
void main() { w_pos = pos; w_nrm = nrm; gl_Position = proj * view * vec4(pos, 1.0); }
"""
_FS_VOL = """
#version 330
in vec3 w_pos;
in vec3 w_nrm;
uniform mat4 view;
uniform mat4 proj;
uniform sampler3D vol;          // r: 値, g: 選んだ原子の球に対する符号付き距離（球がなければ 1）
uniform sampler3D blk;          // ブロックごとの (値の最大, 値の最小, 距離の最大, 0)
uniform mat3 Li;                // ワールド座標 → 分率座標
uniform vec3 N;                 // 格子の数 (N0, N1, N2)
uniform vec3 ncell;             // 描画範囲のセルの数
uniform vec3 NB;                // ブロックの数
uniform float B;                // ブロックの大きさ（格子）
uniform float level;
uniform int neg;                // 1: 負の等値面（値 ≤ −レベル）も描く
uniform vec3 isoPos;
uniform vec3 isoNeg;
uniform float alpha;
uniform int secOn;
uniform float vmin;
uniform float vmax;
uniform int cmode;              // 0: B-G-R, 1: R-G-B, 2: グレー, 3: 等値面と同じ色
uniform vec3 light;
uniform vec3 vdir;              // 視線の向き（ワールド座標。手前 → 奥）
uniform int nat;                // 球の数（0: 球なし、または多すぎてテクスチャの距離だけを使う）
uniform vec4 atoms[256];        // 選んだ原子の球 (分率座標 xyz, 半径 [Å])
uniform mat3 Lm;                // 分率座標 → ワールド座標
uniform float sband;            // テクスチャの距離がこれより境界に近いときだけ、球との距離を正確に計算する [Å]
out vec4 frag;

vec2 samp(vec3 u) { return texture(vol, (u.zyx + 0.5) / N.zyx).rg; }
// 選んだ原子の球に対する符号付き距離（球の中で正）。周期境界の一番近い像で、式から正確に求める
float sdExact(vec3 u) {
    vec3 f = u / N;
    float s = -1e30;
    for (int a = 0; a < nat; a++) {
        vec3 d = f - atoms[a].xyz;
        d -= floor(d + 0.5);
        s = max(s, atoms[a].w - length(Lm * d));
    }
    return s;
}
// 球の中か: テクスチャの距離（3 重線形補間）で判定し、境界に近いときだけ正確な式で判定し直す
// （GPU のテクスチャの補間は精度が限られるので、球の境界の位置は式で決める）
float sdist(vec3 u, float g) { return (nat > 0 && abs(g) < sband) ? sdExact(u) : g; }
bool insideAt(vec3 u) {
    vec2 v = samp(u);
    if (!(v.r >= level || (neg == 1 && v.r <= -level))) return false;
    return sdist(u, v.g) >= 0.0;
}
// 点 u のブロック（周期のどこか base と、その中のブロックの番号 b）。
// 飛ばすかどうかの判定と、ブロックの出口の計算に必ず同じ b を使う（GPU の割り算の丸めで、
// 2 つが別のブロックを見て、調べていないブロックを飛ばしてしまうのを防ぐ）
void blockOf(vec3 u, out vec3 base, out vec3 b) {
    base = floor(u / N) * N;
    b = clamp(floor((u - base) / B), vec3(0.0), NB - 1.0);
}
bool skipBlock(vec3 b) {
    vec4 m = texelFetch(blk, ivec3(b).zyx, 0);
    if (m.b < -sband) return true;            // ブロック全体が球の外（境界の近くは正確な判定に任せる）
    return (m.r < level) && (neg == 0 || m.g > -level);
}
float blockExit(vec3 u, vec3 du, vec3 base, vec3 b) {
    vec3 lo = base + b * B;
    vec3 hi = base + min((b + 1.0) * B, N);
    float te = 1e30;
    for (int i = 0; i < 3; i++) {
        if (du[i] > 1e-12) te = min(te, (hi[i] - u[i]) / du[i]);
        else if (du[i] < -1e-12) te = min(te, (lo[i] - u[i]) / du[i]);
    }
    return max(te, 0.0);
}
vec3 cmap(float v) {
    float t = clamp((v - vmin) / max(vmax - vmin, 1e-30), 0.0, 1.0);
    if (cmode == 2) return vec3(t);
    if (cmode == 1) t = 1.0 - t;
    return vec3(clamp(2.0 * t - 1.0, 0.0, 1.0), 1.0 - abs(2.0 * t - 1.0), clamp(1.0 - 2.0 * t, 0.0, 1.0));
}
vec3 gradW(vec3 u, int ch) {                   // ワールド座標での勾配（中心差分、1 格子）
    vec3 g;
    for (int i = 0; i < 3; i++) {
        vec3 e = vec3(0.0); e[i] = 1.0;
        vec2 a = samp(u + e), b = samp(u - e);
        g[i] = 0.5 * ((ch == 0 ? a.r : a.g) - (ch == 0 ? b.r : b.g)) * N[i];
    }
    return transpose(Li) * g;
}
float depthOf(vec3 p) { vec4 c = proj * view * vec4(p, 1.0); return 0.5 * c.z / c.w + 0.5; }

void main() {
    vec3 nE = mat3(view) * w_nrm;
    if (nE.z > -1e-6) discard;                 // 奥側の面だけを使う（画素ごとに 1 回）
    // この画素の視線と描画範囲の交わり: 手前の入口 S から奥の出口まで
    vec3 f0 = Li * w_pos, df = Li * vdir;
    float smax = 1e30, smin = -1e30;
    int ent = -1;
    for (int i = 0; i < 3; i++) {
        if (abs(df[i]) < 1e-12) continue;
        float s1 = f0[i] / df[i], s2 = (f0[i] - ncell[i]) / df[i];
        float a = min(s1, s2), b = max(s1, s2);
        if (b < smax) { smax = b; ent = i; }
        smin = max(smin, a);
    }
    float T = smax - smin;
    if (ent < 0 || T <= 0.0) discard;
    vec3 S = w_pos - smax * vdir;
    vec3 uS = (Li * S) * N;
    vec3 du = df * N;
    float h = 0.5 / max(max(abs(du.x), abs(du.y)), abs(du.z));
    int kind = 0;                               // 1: 範囲の面の切り口, 2: 球の境界の切り口, 3: 等値面
    float th = -1.0;
    if (insideAt(uS)) { th = 0.0; kind = 1; }
    else {
        float t = 0.0;
        for (int it = 0; it < 16384 && t < T; it++) {
            vec3 u = uS + t * du;
            vec3 base, bb;
            blockOf(u, base, bb);
            if (skipBlock(bb)) { t += blockExit(u, du, base, bb) + 1e-3 * h; continue; }
            float tn = min(t + h, T);
            if (insideAt(uS + tn * du)) {
                float a = t, b = tn;
                for (int k = 0; k < 14; k++) { float m = 0.5 * (a + b); if (insideAt(uS + m * du)) b = m; else a = m; }
                th = b;
                vec3 ua = uS + a * du;
                kind = sdist(ua, samp(ua).g) < 0.0 ? 2 : 3;
                break;
            }
            if (tn >= T) break;
            t = tn;
        }
    }
    if (th < 0.0) discard;
    if (secOn == 0 && kind != 3) {
        // 切り口を描かないとき: 領域の中を進み、等値面から出る所（面の裏側）を探す
        float t = th; th = -1.0;
        for (int it = 0; it < 16384 && t < T; it++) {
            float tn = min(t + h, T);
            vec3 un = uS + tn * du;
            if (!insideAt(un)) {
                if (sdist(un, samp(un).g) < 0.0) break;   // 球の外へ出た（切り口）
                float a = t, b = tn;
                for (int k = 0; k < 14; k++) { float m = 0.5 * (a + b); if (insideAt(uS + m * du)) a = m; else b = m; }
                th = a; kind = 3;
                break;
            }
            if (tn >= T) break;
            t = tn;
        }
        if (th < 0.0) discard;
    }
    vec3 u = uS + th * du;
    vec3 p = S + th * vdir;
    vec2 v = samp(u);
    vec3 col, nW;
    if (kind == 3) {
        nW = gradW(u, 0);
        col = v.r >= 0.0 ? isoPos : isoNeg;
    } else {
        if (kind == 2) {
            if (nat > 0) {                               // 球の表面の法線: 正確な距離の勾配
                vec3 g;
                for (int i = 0; i < 3; i++) { vec3 e = vec3(0.0); e[i] = 0.25; g[i] = (sdExact(u + e) - sdExact(u - e)) * 2.0 * N[i]; }
                nW = transpose(Li) * g;
            } else nW = gradW(u, 1);
        }
        else { nW = vec3(Li[0][ent], Li[1][ent], Li[2][ent]); }
        col = cmode == 3 ? (v.r >= 0.0 ? isoPos : isoNeg) : cmap(abs(v.r));
    }
    vec3 n = mat3(view) * nW;
    float ln = length(n);
    n = ln > 1e-20 ? n / ln : vec3(0.0, 0.0, 1.0);
    if (n.z < 0.0) n = -n;
    vec3 L = normalize(light);
    vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), 40.0);
    frag = vec4(col * (0.30 + 0.72 * diff) + vec3(0.30) * spec, alpha);
    gl_FragDepth = depthOf(p);
}
"""
VOL_BLOCK = 4          # 空の部分を飛ばすブロックの大きさ（格子の数）


def volume_blocks(F, S=None, B=VOL_BLOCK):
    """ブロック（B×B×B の格子のセル）ごとの (値の最大, 値の最小, 球に対する距離の最大)。
    セルの 8 隅の値の範囲を使うので、3 重線形補間の値はブロックの中で必ずこの範囲に入る（飛ばしても見落とさない）"""
    F = np.asarray(F, np.float32)
    N = np.array(F.shape)

    def corner_ext(A, fn):
        out = A.copy()
        for d0 in (0, 1):
            for d1 in (0, 1):
                for d2 in (0, 1):
                    if d0 or d1 or d2:
                        out = fn(out, np.roll(A, (-d0, -d1, -d2), axis=(0, 1, 2)))
        return out
    NB = -(-N // B)

    def block(A, fn, fill):
        P = np.full(tuple(NB * B), fill, np.float32)
        P[:N[0], :N[1], :N[2]] = A
        P = P.reshape(NB[0], B, NB[1], B, NB[2], B)
        return fn.reduce(fn.reduce(fn.reduce(P, axis=5), axis=3), axis=1)
    mx = block(corner_ext(F, np.maximum), np.maximum, -np.inf)
    mn = block(corner_ext(F, np.minimum), np.minimum, np.inf)
    if S is None:
        sm = np.ones(tuple(NB), np.float32)
    else:
        sm = block(corner_ext(np.asarray(S, np.float32), np.maximum), np.maximum, -np.inf)
    return np.ascontiguousarray(np.stack([mx, mn, sm], axis=-1).astype(np.float32))


def box_triangles(L, ncell):
    """描画範囲（分率座標 [0, n0]×[0, n1]×[0, n2]）の 12 枚の三角形 (36, 6) [位置, 外向き法線]"""
    L = np.asarray(L, float)
    n = np.asarray(ncell, float)
    Li = np.linalg.inv(L)
    out = []
    for ax in range(3):
        b, c = [i for i in range(3) if i != ax]
        for side in (0.0, 1.0):
            q = []
            for fb, fc in ((0, 0), (1, 0), (1, 1), (0, 1)):
                f = np.zeros(3)
                f[ax], f[b], f[c] = side * n[ax], fb * n[b], fc * n[c]
                q.append(f @ L)
            nrm = Li[:, ax] * (1.0 if side else -1.0)
            nrm = nrm / np.linalg.norm(nrm)
            for tri in ((0, 1, 2), (0, 2, 3)):
                for k in tri:
                    out.append(np.r_[q[k], nrm])
    return np.array(out, np.float32)


# 線の描画（元の定義を置き換える）: 奥にある線ほど背景の色に近づけて、手前と奥を見分けられるようにする。
# 平行投影の線画（ブリルアンゾーンの枠）は手前と奥の区別がつかず、回転の向きが逆に見えることがあるため。
DEPTH_CUE_MIN = 0.12        # いちばん奥の線の濃さ（1 = 手前と同じ）
_VS_LINE = """
#version 330
layout(location=0) in vec3 pos;
uniform mat4 view;
uniform mat4 proj;
out float vz;
void main() { vec4 e = view * vec4(pos, 1.0); vz = e.z; gl_Position = proj * e; }
"""
_FS_LINE = """
#version 330
uniform vec3 color;
uniform vec3 bgcol;
uniform vec3 cue;       // (いちばん奥の z, いちばん手前の z, 奥の濃さ)。奥の濃さ 1 = 奥行きで暗くしない
in float vz;
out vec4 frag;
void main() {
    float f = clamp((vz - cue.x) / max(cue.y - cue.x, 1e-20), 0.0, 1.0);
    frag = vec4(mix(bgcol, color, mix(cue.z, 1.0, f * f)), 1.0);
}
"""


# ---- 凸多面体の枠（ブリルアンゾーン・逆格子の単位胞）: 手前の辺は実線、奥の辺は破線 ----
HIDDEN_DASH_PX = (7.0, 5.0)     # 破線の線の長さ・すき間 [ピクセル]
HIDDEN_DIM = 0.8                # 破線の濃さ（1 = 実線と同じ）


def edge_face_normals(edges):
    """凸多面体の辺 edges (E, 2, 3) について、各辺をはさむ 2 つの面の外向き法線 (E, 2, 3) を求める。
    面は「1 つの頂点を共有する 2 辺が張る平面のうち、全部の頂点がその片側にあるもの」として見つける。"""
    E = np.asarray(edges, float).reshape(-1, 2, 3)
    P = E.reshape(-1, 3)
    c = P.mean(axis=0)
    size = float(np.abs(P - c).max()) or 1.0
    tol = 1e-6 * size
    planes = []                                        # (外向き単位法線, 定数 d)。面上の点 p で n·p = d
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            for a in (0, 1):
                for b in (0, 1):
                    if np.linalg.norm(E[i, a] - E[j, b]) > tol:
                        continue
                    n = np.cross(E[i, 1 - a] - E[i, a], E[j, 1 - b] - E[j, b])
                    ln = np.linalg.norm(n)
                    if ln < 1e-9 * size * size:
                        continue
                    n = n / ln
                    dist = (P - E[i, a]) @ n
                    if dist.max() > tol and dist.min() < -tol:
                        continue                       # 頂点が両側にある → 面ではない
                    if dist.max() > tol:
                        n = -n
                    dpl = float(n @ E[i, a])
                    if not any(np.abs(n - q).max() < 1e-6 and abs(dpl - dq) < tol for q, dq in planes):
                        planes.append((n, dpl))
    out = np.zeros((len(E), 2, 3))
    for i in range(len(E)):
        adj = [n for n, dpl in planes if abs(n @ E[i, 0] - dpl) < tol and abs(n @ E[i, 1] - dpl) < tol]
        if adj:
            out[i, 0] = adj[0]
            out[i, 1] = adj[1] if len(adj) > 1 else adj[0]
    return out


def split_hidden_edges(edges, normals, R, scale, dash_px, gap_px):
    """辺を「手前（実線）」と「奥（破線）」に分ける。奥 = 辺をはさむ面がどちらも向こうを向いている辺
    （中身が詰まった多面体なら隠れて見えない辺）。破線は画面上の長さで刻む。
    戻り値 (実線の頂点 (N, 3), 破線の頂点 (M, 3))。どちらも線分ごとに 2 点"""
    E = np.asarray(edges, float).reshape(-1, 2, 3)
    R = np.asarray(R, float)
    front = (np.asarray(normals, float) @ R[2]) > 1e-6            # 面が手前を向いているか (E, 2)
    hidden = ~front.any(axis=1)
    solid = E[~hidden].reshape(-1, 3)
    dashes = []
    period = dash_px + gap_px
    for p0, p1 in E[hidden]:
        L = float(np.linalg.norm((p1 - p0) @ R[:2].T)) * scale    # 画面上の長さ [ピクセル]
        if L < 1.0:
            continue                                               # 視線と平行で点になっている辺
        n = int(max(1, round((L + gap_px) / period)))
        if n == 1:
            dashes += [p0, p1]
            continue
        step = (L - dash_px) / (n - 1)                             # 両端が線で終わるように間隔を調整
        for k in range(n):
            t0 = k * step / L
            t1 = min(1.0, t0 + dash_px / L)
            dashes += [p0 + (p1 - p0) * t0, p0 + (p1 - p0) * t1]
    dash = np.array(dashes, float).reshape(-1, 3)
    return solid.astype(np.float32), dash.astype(np.float32)


def _dim_color(color, bg, k):
    return tuple(b + (c - b) * k for c, b in zip(color[:3], bg[:3]))


def empty_scene(bg=(1.0, 1.0, 1.0)):
    return {"bg": bg, "spheres": None, "cyls": None, "lines": [], "mesh": None, "mesh_t": None,
            "labels": None, "triad": None, "center": np.zeros(3), "radius": 1.0, "fit": np.zeros((1, 3))}


def mesh_vertices(D, rgba):
    """isosurface_tets の出力 D (T, 3, 6+K) と色 → 描画用の頂点配列 (3T, 10)
    rgba: (4,) で全体に 1 色、または (T, 3, 4) / (T, 3, 3) で頂点ごとの色"""
    T = len(D)
    out = np.empty((T, 3, 10), np.float32)
    out[:, :, :3] = D[:, :, :3]
    g = D[:, :, 3:6]
    ln = np.linalg.norm(g, axis=2, keepdims=True)
    # 勾配が 0 の頂点は三角形の面法線で代用する
    fn = np.cross(D[:, 1, :3] - D[:, 0, :3], D[:, 2, :3] - D[:, 0, :3])
    fl = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.where(fl > 0, fl, 1.0)
    out[:, :, 3:6] = np.where(ln > 1e-12, g / np.where(ln > 1e-12, ln, 1.0), fn[:, None, :])
    c = np.asarray(rgba, np.float32)
    if c.ndim == 1:
        out[:, :, 6:10] = c if len(c) == 4 else np.r_[c, 1.0]
    else:
        out[:, :, 6:6 + c.shape[2]] = c
        if c.shape[2] == 3:
            out[:, :, 9] = 1.0
    return out.reshape(-1, 10)


def make_scene_renderer():
    """OpenGL 3.3 の描画本体（ウィンドウの種類に依存しない。現在のコンテキストに描く）"""
    import ctypes
    import OpenGL
    OpenGL.ERROR_CHECKING = False
    from OpenGL import GL
    from OpenGL.GL import shaders

    F = 4

    def _mat(M):
        return np.ascontiguousarray(np.asarray(M, np.float32).T)    # GLSL は列優先

    class SceneRenderer:
        def __init__(self):
            self.ready = False
            self.msaa = None
            self.msaa_size = None
            self.width = self.height = 1
            self.scene = None
            self.dirty = True
            self.gl_info = ""
            self.n_sph = self.n_cyl = self.n_mesh = self.n_mesh_t = 0
            self.line_groups = []
            self.atlas_key = None
            self.atlas_info = {}
            self.n_lab = 0
            self.label_sig = None

        # ---------- 初期化 ----------
        def init(self):
            ver = (GL.glGetString(GL.GL_VERSION) or b"0").decode(errors="ignore")
            self.gl_info = (GL.glGetString(GL.GL_RENDERER) or b"").decode(errors="ignore")
            mm = ver.split()[0].split(".")
            if (int(mm[0]), int(mm[1])) < (3, 3) or "GDI Generic" in self.gl_info:
                raise RuntimeError(f"OpenGL 3.3 以上が必要です（この環境: {ver}, {self.gl_info}）")
            comp = lambda vs, fs: shaders.compileProgram(shaders.compileShader(vs, GL.GL_VERTEX_SHADER),
                                                         shaders.compileShader(fs, GL.GL_FRAGMENT_SHADER),
                                                         validate=False)
            self.p_sph = comp(_VS_SPHERE, _FS_SPHERE)
            self.p_cyl = comp(_VS_CYL, _FS_CYL)
            self.p_line = comp(_VS_LINE, _FS_LINE)
            self.p_lab = comp(_VS_LABEL, _FS_LABEL)
            self.p_mesh = comp(_VS_MESH, _FS_MESH)
            self.vol_error = None
            try:
                self.p_vol = comp(_VS_VOL, _FS_VOL)
            except Exception as e:                 # レイキャスティングだけ使えない（三角形で描く）
                self.p_vol = None
                self.vol_error = str(e)[:300]
            self.vol_key = None
            self.n_volbox = 0
            quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], np.float32)
            self.vbo_quad = self._buf(quad)
            self.vbo_cylmesh = self._buf(_unit_cylinder_mesh(16))
            self.n_cylmesh = 16 * 6
            self.vbo_conemesh = self._buf(_unit_cone_mesh(16))
            self.n_conemesh = 16 * 3
            names = ("sph", "cyl", "line", "axcyl", "axcone", "axlab", "mesh", "mesh_t", "lab", "moff", "mesh_s",
                     "mesh_s_t", "volbox")
            for nm in names:
                setattr(self, "vbo_" + nm, GL.glGenBuffers(1))
            self.vao_sph = self._vao_instanced(self.vbo_quad, [(0, 2)], self.vbo_sph, [(1, 3), (2, 1), (3, 3)])
            cyl_attr = [(2, 3), (3, 3), (4, 1), (5, 3)]
            self.vao_cyl = self._vao_instanced(self.vbo_cylmesh, [(0, 3), (1, 3)], self.vbo_cyl, cyl_attr)
            self.vao_axcyl = self._vao_instanced(self.vbo_cylmesh, [(0, 3), (1, 3)], self.vbo_axcyl, cyl_attr)
            self.vao_axcone = self._vao_instanced(self.vbo_conemesh, [(0, 3), (1, 3)], self.vbo_axcone, cyl_attr)
            self.vao_axlab = self._vao_instanced(self.vbo_quad, [(0, 2)], self.vbo_axlab,
                                                 [(1, 3), (2, 1), (3, 4), (4, 2)])
            self.vao_lab = self._vao_instanced(self.vbo_quad, [(0, 2)], self.vbo_lab,
                                               [(1, 3), (2, 1), (3, 4), (4, 2)])
            self.vao_line = self._vao_plain(self.vbo_line, [(0, 3)])
            mattr = [(0, 3), (1, 3), (2, 4)]
            self.vao_mesh = self._vao_instanced(self.vbo_mesh, mattr, self.vbo_moff, [(3, 3)])
            self.vao_mesh_t = self._vao_instanced(self.vbo_mesh_t, mattr, self.vbo_moff, [(3, 3)])
            # 断面もずれ 0 を 1 回分だけ読む（location 3 を無効のままにすると、球・結合・ラベルの描画で使った
            # 値が残ることがある（OpenGL 3.3 では未定義。NVIDIA で断面がずれて描かれた原因））
            self.vbo_zoff = self._buf(np.zeros((1, 3), np.float32))
            self.vao_mesh_s = self._vao_instanced(self.vbo_mesh_s, mattr, self.vbo_zoff, [(3, 3)])
            self.vao_mesh_s_t = self._vao_instanced(self.vbo_mesh_s_t, mattr, self.vbo_zoff, [(3, 3)])
            self.vao_volbox = self._vao_plain(self.vbo_volbox, [(0, 3), (1, 3)])
            self.tex_vol, self.tex_blk = GL.glGenTextures(2)
            self.n_moff = 1
            self.tex = GL.glGenTextures(1)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            self.ready = True
            self._make_msaa()

        def _buf(self, arr):
            b = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, b)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            return b

        def _vao_plain(self, vbo, attrs):
            vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            stride = sum(n for _, n in attrs) * F
            off = 0
            for loc, n in attrs:
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribPointer(loc, n, GL.GL_FLOAT, False, stride, ctypes.c_void_p(off))
                off += n * F
            GL.glBindVertexArray(0)
            return vao

        def _vao_instanced(self, mesh_vbo, mesh_attrs, inst_vbo, inst_attrs):
            vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, mesh_vbo)
            stride = sum(n for _, n in mesh_attrs) * F
            off = 0
            for loc, n in mesh_attrs:
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribPointer(loc, n, GL.GL_FLOAT, False, stride, ctypes.c_void_p(off))
                off += n * F
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, inst_vbo)
            stride = sum(n for _, n in inst_attrs) * F
            off = 0
            for loc, n in inst_attrs:
                GL.glEnableVertexAttribArray(loc)
                GL.glVertexAttribPointer(loc, n, GL.GL_FLOAT, False, stride, ctypes.c_void_p(off))
                GL.glVertexAttribDivisor(loc, 1)
                off += n * F
            GL.glBindVertexArray(0)
            return vao

        def _make_msaa(self):
            """4x マルチサンプリング用のオフスクリーン描画先（作れなければ直接描画）"""
            try:
                if self.msaa:
                    fbo, rb_c, rb_d = self.msaa
                    GL.glDeleteFramebuffers(1, [fbo])
                    GL.glDeleteRenderbuffers(2, [rb_c, rb_d])
                    self.msaa = None
                samples = min(4, int(GL.glGetIntegerv(GL.GL_MAX_SAMPLES)))
                if samples < 2:
                    return
                W, H = max(1, self.width), max(1, self.height)
                fbo = GL.glGenFramebuffers(1)
                rb_c, rb_d = GL.glGenRenderbuffers(2)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb_c)
                GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, samples, GL.GL_RGBA8, W, H)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb_d)
                GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, samples, GL.GL_DEPTH_COMPONENT24, W, H)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb_c)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rb_d)
                ok = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
                if ok:
                    self.msaa = (fbo, rb_c, rb_d)
                    self.msaa_size = (W, H)
                else:
                    GL.glDeleteFramebuffers(1, [fbo])
                    GL.glDeleteRenderbuffers(2, [rb_c, rb_d])
            except Exception:
                self.msaa = None

        def resize(self, w, h):
            self.width, self.height = max(1, int(w)), max(1, int(h))
            if self.ready:
                GL.glViewport(0, 0, self.width, self.height)
                self._make_msaa()

        # ---------- データ転送 ----------
        def _upload(self, vbo, arr):
            arr = np.ascontiguousarray(arr, np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, max(arr.nbytes, 4), arr if arr.size else None,
                            GL.GL_DYNAMIC_DRAW)

        def set_scene(self, scene):
            self.scene = scene
            self.dirty = True

        def _rebuild(self):
            sc = self.scene
            z = np.zeros((0, 1), np.float32)
            sph = sc.get("spheres")
            sph = z.reshape(0, 7) if sph is None else np.asarray(sph, np.float32)
            self._upload(self.vbo_sph, sph)
            self.n_sph = len(sph)
            cyl = sc.get("cyls")
            cyl = z.reshape(0, 10) if cyl is None else np.asarray(cyl, np.float32)
            self._upload(self.vbo_cyl, cyl)
            self.n_cyl = len(cyl)
            self.line_groups = []
            allv, start = [], 0
            for verts, color, width in sc.get("lines") or []:
                v = np.asarray(verts, np.float32).reshape(-1, 3)
                if len(v):
                    allv.append(v)
                    self.line_groups.append((start, len(v), tuple(color), float(width)))
                    start += len(v)
            self.line_all = np.vstack(allv) if allv else z.reshape(0, 3)
            self._upload(self.vbo_line, self.line_all)
            for key in ("mesh", "mesh_t", "mesh_s", "mesh_s_t"):
                m = sc.get(key)
                m = z.reshape(0, 10) if m is None else np.asarray(m, np.float32)
                self._upload(getattr(self, "vbo_" + key), m)
                setattr(self, "n_" + key, len(m))
            off = sc.get("mesh_offsets")
            off = np.zeros((1, 3), np.float32) if off is None or len(off) == 0 else np.asarray(off, np.float32)
            self._upload(self.vbo_moff, off)
            self.n_moff = len(off)
            self._upload_volume(sc.get("volume"))
            self.label_sig = None
            self.dirty = False

        def _upload_volume(self, vol):
            """レイキャスティングのデータ（3D テクスチャ）。データが変わったときだけ送り直す"""
            if vol is None or self.p_vol is None:
                self.n_volbox = 0
                return
            if vol.get("key") != self.vol_key:
                F = np.asarray(vol["data"], np.float32)
                S = vol.get("sdist")
                RG = np.empty(F.shape + (2,), np.float32)
                RG[..., 0] = F
                RG[..., 1] = 1.0 if S is None else np.asarray(S, np.float32)
                blk = volume_blocks(F, S)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
                GL.glBindTexture(GL.GL_TEXTURE_3D, self.tex_vol)
                for pn, pv in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                               (GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT), (GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT),
                               (GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT)):
                    GL.glTexParameteri(GL.GL_TEXTURE_3D, pn, pv)
                n0, n1, n2 = F.shape
                fmt = GL.GL_RG32F if F.size <= 24_000_000 else GL.GL_RG16F     # 大きな格子は半精度でメモリを節約
                GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, fmt, n2, n1, n0, 0, GL.GL_RG, GL.GL_FLOAT,
                                np.ascontiguousarray(RG))
                GL.glBindTexture(GL.GL_TEXTURE_3D, self.tex_blk)
                for pn, pv in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)):
                    GL.glTexParameteri(GL.GL_TEXTURE_3D, pn, pv)
                b0, b1, b2 = blk.shape[:3]
                blk4 = np.zeros((b0, b1, b2, 4), np.float32)
                blk4[..., :3] = blk
                GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA32F, b2, b1, b0, 0, GL.GL_RGBA, GL.GL_FLOAT, blk4)
                GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
                self.vol_key = vol.get("key")
                self.vol_shape = F.shape
                self.vol_nb = blk.shape[:3]
            box = box_triangles(vol["L"], vol["ncell"])
            self._upload(self.vbo_volbox, box)
            self.n_volbox = len(box)

        def _draw_volume(self, view, proj, R):
            vol = self.scene.get("volume")
            p = self.p_vol
            GL.glUseProgram(p)
            u = lambda nm: GL.glGetUniformLocation(p, nm)
            GL.glUniformMatrix4fv(u("view"), 1, False, _mat(view))
            GL.glUniformMatrix4fv(u("proj"), 1, False, _mat(proj))
            L = np.asarray(vol["L"], float)
            GL.glUniformMatrix3fv(u("Li"), 1, False, _mat(np.linalg.inv(L).T))
            GL.glUniform3f(u("N"), *[float(x) for x in self.vol_shape])
            GL.glUniform3f(u("ncell"), *[float(x) for x in vol["ncell"]])
            GL.glUniform3f(u("NB"), *[float(x) for x in self.vol_nb])
            GL.glUniform1f(u("B"), float(VOL_BLOCK))
            GL.glUniform1f(u("level"), float(vol["level"]))
            GL.glUniform1i(u("neg"), 1 if vol.get("neg") else 0)
            GL.glUniform3f(u("isoPos"), *vol.get("iso_rgb", (1.0, 1.0, 0.0)))
            GL.glUniform3f(u("isoNeg"), *vol.get("neg_rgb", (0.0, 1.0, 1.0)))
            GL.glUniform1f(u("alpha"), float(vol.get("alpha", 1.0)))
            GL.glUniform1i(u("secOn"), 1 if vol.get("sections", True) else 0)
            GL.glUniform1f(u("vmin"), float(vol.get("vmin", 0.0)))
            GL.glUniform1f(u("vmax"), float(vol.get("vmax", 1.0)))
            GL.glUniform1i(u("cmode"), int(vol.get("cmode", 0)))
            GL.glUniform3f(u("light"), *LIGHT_DIR)
            GL.glUniform3f(u("vdir"), *[float(x) for x in -np.asarray(R, float)[2]])
            GL.glUniformMatrix3fv(u("Lm"), 1, False, _mat(L.T))
            sph = vol.get("spheres")
            nat = 0 if sph is None or len(sph) > 256 else len(sph)
            GL.glUniform1i(u("nat"), nat)
            if nat:
                GL.glUniform4fv(u("atoms"), nat, np.ascontiguousarray(np.asarray(sph, np.float32)))
            # 正確な計算をする帯の幅: 格子 2 個分（テクスチャの補間の誤差より十分広い）
            GL.glUniform1f(u("sband"), float(2.0 * max(np.linalg.norm(L, axis=1) / np.asarray(self.vol_shape, float))))
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_3D, self.tex_vol)
            GL.glUniform1i(u("vol"), 1)
            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_3D, self.tex_blk)
            GL.glUniform1i(u("blk"), 2)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            translucent = float(vol.get("alpha", 1.0)) < 0.999
            if translucent:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glBindVertexArray(self.vao_volbox)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.n_volbox)
            if translucent:
                GL.glDisable(GL.GL_BLEND)

        def _prepare_labels(self, label_px):
            """ラベル（原子ラベルと軸の名前）の文字画像と配置データを用意する"""
            sc = self.scene
            labs = sc.get("labels")
            texts = list(labs[0]) if labs is not None else []
            names = list(sc["triad"][1]) if sc.get("triad") is not None else []
            self._atlas(texts + names, label_px)
            sig = (len(texts), self.atlas_key)
            if sig == self.label_sig:
                return
            rows = np.zeros((len(texts), 10), np.float32)
            if texts:
                rows[:, 0:3] = np.asarray(labs[1], np.float32)
                rows[:, 3] = np.asarray(labs[2], np.float32)
                rows[:, 4:10] = np.array([self.atlas_info[t] for t in texts], np.float32)
            self._upload(self.vbo_lab, rows)
            self.n_lab = len(texts)
            self.label_sig = sig

        def _atlas(self, strings, size_px):
            key = (tuple(sorted(set(strings))), int(size_px))
            if key == self.atlas_key:
                return
            arr, info = make_label_atlas(list(key[0]), size_px)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, arr.shape[1], arr.shape[0], 0,
                            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, arr)
            self.atlas_info = info
            self.atlas_key = key

        # ---------- 描画 ----------
        @staticmethod
        def _ortho(l, r, b, t, n, f):
            M = np.eye(4)
            M[0, 0], M[1, 1], M[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
            M[0, 3], M[1, 3], M[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
            return M

        def _set_common(self, prog, view, proj):
            GL.glUseProgram(prog)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(prog, "view"), 1, False, _mat(view))
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(prog, "proj"), 1, False, _mat(proj))
            loc = GL.glGetUniformLocation(prog, "light")
            if loc >= 0:
                GL.glUniform3f(loc, *LIGHT_DIR)

        def draw(self, R, scale, pan, label_px=14.0, ui_scale=1.0, default_fbo=0):
            """R: 回転行列, scale: 1 単位あたりのピクセル数, pan: 画面内の平行移動（単位は scene と同じ）"""
            sc = self.scene
            target = self.msaa[0] if self.msaa and self.msaa_size == (self.width, self.height) else default_fbo
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target)
            GL.glViewport(0, 0, self.width, self.height)
            bg = (1.0, 1.0, 1.0) if sc is None else sc.get("bg", (1.0, 1.0, 1.0))
            GL.glClearColor(bg[0], bg[1], bg[2], 1.0)
            GL.glDepthMask(True)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            if sc is None or scale is None or not self.ready:
                self._finish(target, default_fbo)
                return
            if self.dirty:
                self._rebuild()
            self._prepare_labels(label_px)
            W, H = self.width, self.height
            ce = np.asarray(sc["center"], float) @ R.T
            cx, cy = ce[0] + pan[0], ce[1] + pan[1]
            ext = float(sc["radius"]) * 3.0            # 奥行きの範囲（scene の単位に比例させる: Å でも 1/Å でもよい）
            view = np.eye(4)
            view[:3, :3] = R
            proj = self._ortho(cx - W / 2 / scale, cx + W / 2 / scale, cy - H / 2 / scale, cy + H / 2 / scale,
                               -(ce[2] + ext), -(ce[2] - ext))
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_BLEND)
            if self.n_sph:
                self._set_common(self.p_sph, view, proj)
                GL.glBindVertexArray(self.vao_sph)
                GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, self.n_sph)
            if self.n_cyl:
                self._set_common(self.p_cyl, view, proj)
                GL.glBindVertexArray(self.vao_cyl)
                GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_cylmesh, self.n_cyl)
            if self.n_mesh:
                self._set_common(self.p_mesh, view, proj)
                GL.glBindVertexArray(self.vao_mesh)
                GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_mesh, self.n_moff)
            if self.n_mesh_s:
                self._set_common(self.p_mesh, view, proj)
                GL.glBindVertexArray(self.vao_mesh_s)
                GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_mesh_s, 1)
            hl = sc.get("edge_faces")
            if self.line_groups or hl is not None:
                self._set_common(self.p_line, view, proj)
                GL.glBindVertexArray(self.vao_line)
                loc = GL.glGetUniformLocation(self.p_line, "color")
                GL.glUniform3f(GL.glGetUniformLocation(self.p_line, "bgcol"), *sc["bg"][:3])
                zl = self.line_all @ np.asarray(R, float)[2] if len(self.line_all) else np.zeros(1)
                GL.glUniform3f(GL.glGetUniformLocation(self.p_line, "cue"), float(zl.min()), float(zl.max()),
                               DEPTH_CUE_MIN if sc.get("depth_cue") else 1.0)
                groups = list(self.line_groups)
                if hl is not None:
                    # 凸多面体の枠: 向きが変わるたびに、手前の辺（実線）と奥の辺（破線）を作り直す
                    solid, dash = split_hidden_edges(hl[0], hl[1], R, scale, HIDDEN_DASH_PX[0] * ui_scale,
                                                     HIDDEN_DASH_PX[1] * ui_scale)
                    self._upload(self.vbo_line, np.vstack([self.line_all, solid, dash]))
                    n0 = len(self.line_all)
                    groups.append((n0 + len(solid), len(dash), _dim_color(hl[2], sc["bg"], HIDDEN_DIM), float(hl[3])))
                    groups.append((n0, len(solid), tuple(hl[2]), float(hl[3])))
                for start, n, color, width in groups:
                    if n <= 0:
                        continue
                    GL.glUniform3f(loc, *color)
                    try:
                        GL.glLineWidth(max(1.0, width * ui_scale))
                    except Exception:
                        pass
                    GL.glDrawArrays(GL.GL_LINES, start, n)
            if self.n_volbox:
                self._draw_volume(view, proj, R)
            if self.n_lab:
                # ラベルは原子の手前側の表面に置く（奥行きの判定で、隠れた原子のラベルは見えない）
                self._draw_labels(self.vao_lab, self.n_lab, view, proj, scale)
            if self.n_mesh_t or self.n_mesh_s_t:
                # 半透明の面: まず奥行きだけを書き、次に最も手前の 1 枚だけを色を混ぜて描く。
                # （三角形を奥から順に並べ替えなくても、描く順番によるむらが出ない。
                #   原子・結合・単位胞は先に描いてあるので、面の向こうに透けて見える）
                self._set_common(self.p_mesh, view, proj)

                def draw_t():
                    if self.n_mesh_t:
                        GL.glBindVertexArray(self.vao_mesh_t)
                        GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_mesh_t, self.n_moff)
                    if self.n_mesh_s_t:
                        GL.glBindVertexArray(self.vao_mesh_s_t)
                        GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_mesh_s_t, 1)
                GL.glColorMask(False, False, False, False)
                draw_t()
                GL.glColorMask(True, True, True, True)
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
                GL.glDepthMask(False)
                draw_t()
                GL.glDepthMask(True)
                GL.glDisable(GL.GL_BLEND)
            if sc.get("triad") is not None:
                self._draw_triad(sc["triad"], R, W, H, label_px, ui_scale)
            GL.glBindVertexArray(0)
            self._finish(target, default_fbo)

        def _draw_labels(self, vao, n, view, proj, px_per_unit):
            GL.glUseProgram(self.p_lab)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.p_lab, "view"), 1, False, _mat(view))
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.p_lab, "proj"), 1, False, _mat(proj))
            GL.glUniform1f(GL.glGetUniformLocation(self.p_lab, "px_per_unit"), px_per_unit)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
            GL.glUniform1i(GL.glGetUniformLocation(self.p_lab, "tex"), 0)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glBindVertexArray(vao)
            GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, 6, n)
            GL.glDisable(GL.GL_BLEND)

        def _draw_triad(self, triad, R, W, H, label_px, ui_scale):
            vecs, names = triad
            sz = int(160 * ui_scale)
            GL.glViewport(int(8 * ui_scale), int(8 * ui_scale), sz, sz)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
            view = np.eye(4)
            view[:3, :3] = R
            proj = self._ortho(-1.7, 1.7, -1.7, 1.7, -5, 5)
            cols = ((0.85, 0, 0), (0, 0.65, 0), (0, 0, 0.85))
            cyl, cone, lab = [], [], np.zeros((3, 10), np.float32)
            for k, (v, c) in enumerate(zip(vecs, cols)):
                u = np.asarray(v, float) / np.linalg.norm(v)
                cyl.append(np.r_[0, 0, 0, u * 0.85, 0.09, c])
                cone.append(np.r_[u * 0.85, u * 1.2, 0.2, c])
                u0, v0, u1, v1, w, h = self.atlas_info[names[k]]
                lab[k] = np.r_[u * 1.45, 0.3, u0, v0, u1, v1, w, h]
            self._upload(self.vbo_axcyl, np.array(cyl, np.float32))
            self._upload(self.vbo_axcone, np.array(cone, np.float32))
            self._upload(self.vbo_axlab, lab)
            self._set_common(self.p_cyl, view, proj)
            GL.glBindVertexArray(self.vao_axcyl)
            GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_cylmesh, 3)
            GL.glBindVertexArray(self.vao_axcone)
            GL.glDrawArraysInstanced(GL.GL_TRIANGLES, 0, self.n_conemesh, 3)
            GL.glDisable(GL.GL_DEPTH_TEST)
            self._draw_labels(self.vao_axlab, 3, view, proj, sz / 3.4)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glViewport(0, 0, W, H)

        def _finish(self, target, default_fbo=0):
            if target != default_fbo:
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target)
                GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, default_fbo)
                GL.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                                     GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, default_fbo)

        def read_pixels(self, default_fbo=0, back=True):
            """描画済みの画面を (H, W, 3) の配列で返す"""
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, default_fbo)
            if back and default_fbo == 0:
                try:
                    GL.glReadBuffer(GL.GL_BACK)
                except Exception:
                    pass
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            return np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)[::-1].copy()

    return SceneRenderer


def make_scene_gl_view_class():
    """OpenGL 3.3 の描画ウィジェット（PyOpenGL + pyopengltk）。使えなければ例外（呼び出し側で簡易描画に切り替え）"""
    from pyopengltk import OpenGLFrame
    Renderer = make_scene_renderer()

    class SceneGLView(OpenGLFrame):
        def __init__(self, master, owner, **kw):
            super().__init__(master, **kw)
            self.owner = owner          # dict: scene, R, scale, zoom, pan, label_px, on_gl_fail, on_resize
            self.ren = Renderer()
            self.failed = None
            self.pending = False
            self.width = self.height = 1
            self.gl_info = ""

        def tkMap(self, evt):
            try:
                super().tkMap(evt)
            except Exception as e:
                self._fail(e)
                return
            # 表示された直後（タブを初めて開いたときなど）に、大きさに合わせて描き直す
            if self.owner.get("on_resize"):
                self.after(30, self.owner["on_resize"])
            self.request()

        def _fail(self, e):
            self.failed = e
            if self.owner.get("on_gl_fail"):
                self.after(10, lambda: self.owner["on_gl_fail"](e))

        def initgl(self):
            if not self.ren.ready:
                self.ren.width, self.ren.height = self.width, self.height
                self.ren.init()
                self.gl_info = self.ren.gl_info
            else:
                self.ren.resize(self.width, self.height)

        def tkExpose(self, evt):
            if self.failed is None and self.context_created:
                self.request()

        def tkResize(self, evt):
            self.width, self.height = max(1, evt.width), max(1, evt.height)
            if self.failed is None and self.context_created and self.winfo_ismapped() and self.ren.ready:
                self.tkMakeCurrent()
                self.ren.resize(self.width, self.height)
                if self.owner.get("on_resize"):
                    self.owner["on_resize"]()
                self.request()

        def set_scene(self, scene):
            self.ren.set_scene(scene)

        def redraw(self):
            self.pending = False
            ow = self.owner
            s = None if ow["scale"] is None else ow["scale"] * ow["zoom"]
            self.ren.draw(ow["R"], s, ow["pan"], ow["label_px"](), UI_SCALE)

        def request(self):
            """アイドル時にまとめて 1 回描く"""
            if not self.pending and self.failed is None:
                self.pending = True
                self.after_idle(self._display_safe)

        def _display_safe(self):
            try:
                if self.winfo_ismapped() and self.context_created and self.ren.ready:
                    import time as _t
                    self._display()
                    ts = getattr(self, "_times", [])
                    ts.append(_t.perf_counter())
                    self._times = ts[-30:]
                else:
                    self.pending = False
            except Exception as e:
                self.pending = False
                self._fail(e)

        def grab(self):
            from PIL import Image
            self.tkMakeCurrent()
            self.redraw()
            arr = self.ren.read_pixels()
            self.tkSwapBuffers()
            return Image.fromarray(arr)

    return SceneGLView


# ----------------------------------------------------------------------
#  OpenGL が使えない環境のための簡易描画（Pillow。奥から順に塗る）
# ----------------------------------------------------------------------
def render_scene_pil(scene, R, scale, pan, W, H, ui_scale=1.0, sprite_cache=None, max_tris=None,
                     label_px=18.0):
    """max_tris を指定すると、三角形がそれより多いときに間引いて描く（ドラッグ中の応答用）"""
    from PIL import Image, ImageDraw, ImageFont
    bg = tuple(int(round(255 * x)) for x in scene.get("bg", (1, 1, 1)))
    img = Image.new("RGBA", (W, H), bg + (255,))
    dr = ImageDraw.Draw(img, "RGBA")
    ce = np.asarray(scene["center"], float) @ R.T
    cx, cy = ce[0] + pan[0], ce[1] + pan[1]
    L = np.array(LIGHT_DIR) / np.linalg.norm(LIGHT_DIR)

    def proj(P):
        E = np.asarray(P, float) @ R.T
        return W / 2 + (E[..., 0] - cx) * scale, H / 2 - (E[..., 1] - cy) * scale, E[..., 2]

    items = []          # (奥行き, 種類, データ)
    offs = scene.get("mesh_offsets")
    for key in ("mesh", "mesh_t", "mesh_s", "mesh_s_t"):
        m = scene.get(key)
        if m is None or len(m) == 0:
            continue
        tri = np.asarray(m, np.float32).reshape(-1, 3, 10)
        if key in ("mesh", "mesh_t") and offs is not None and len(offs) > 1:
            tri = np.concatenate([tri + np.r_[o, np.zeros(7)].astype(np.float32) for o in np.asarray(offs, np.float32)])
        if max_tris and len(tri) > max_tris:
            tri = tri[::int(np.ceil(len(tri) / max_tris))]
        x, y, z = proj(tri[:, :, :3])
        n = tri[:, :, 3:6].mean(axis=1) @ R.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[n[:, 2] < 0] *= -1
        diff = np.clip(n @ L, 0, 1)
        col = tri[:, :, 6:9].mean(axis=1) * (0.30 + 0.72 * diff[:, None])
        col = np.clip(col * 255, 0, 255).astype(int)
        alpha = np.clip(tri[:, :, 9].mean(axis=1) * 255, 0, 255).astype(int)
        zc = z.mean(axis=1)
        for i in range(len(tri)):
            items.append((zc[i], 0, (x[i, 0], y[i, 0], x[i, 1], y[i, 1], x[i, 2], y[i, 2],
                                     (col[i, 0], col[i, 1], col[i, 2], alpha[i]))))
    for verts, color, width in scene.get("lines") or []:
        v = np.asarray(verts, float).reshape(-1, 2, 3)
        if len(v) == 0:
            continue
        x, y, z = proj(v)
        c = tuple(int(round(255 * t)) for t in color) + (255,)
        for i in range(len(v)):
            ci = c
            if scene.get("depth_cue"):        # 奥にある線ほど背景の色に近づける（GL の描画と同じ）
                f = min(1.0, max(0.0, (z[i].mean() - z.min()) / max(float(z.max() - z.min()), 1e-20)))
                k = DEPTH_CUE_MIN + (1.0 - DEPTH_CUE_MIN) * f * f
                ci = tuple(int(round(255 * (b_ + (t - b_) * k))) for t, b_ in zip(color, scene["bg"][:3])) + (255,)
            items.append((z[i].mean(), 1, (x[i, 0], y[i, 0], x[i, 1], y[i, 1], ci, max(1, int(round(width * ui_scale))))))
    hl = scene.get("edge_faces")
    if hl is not None:
        solid, dash = split_hidden_edges(hl[0], hl[1], R, scale, HIDDEN_DASH_PX[0] * ui_scale,
                                         HIDDEN_DASH_PX[1] * ui_scale)
        for verts, color in ((dash, _dim_color(hl[2], scene.get("bg", (1, 1, 1)), HIDDEN_DIM)), (solid, hl[2])):
            v = np.asarray(verts, float).reshape(-1, 2, 3)
            if len(v) == 0:
                continue
            x, y, z = proj(v)
            c = tuple(int(round(255 * t)) for t in color[:3]) + (255,)
            for i in range(len(v)):
                items.append((z[i].mean(), 1, (x[i, 0], y[i, 0], x[i, 1], y[i, 1], c,
                                               max(1, int(round(hl[3] * ui_scale))))))
    cyl = scene.get("cyls")
    if cyl is not None and len(cyl):
        cyl = np.asarray(cyl, float)
        x0, y0, z0 = proj(cyl[:, 0:3])
        x1, y1, z1 = proj(cyl[:, 3:6])
        for i in range(len(cyl)):
            c = tuple(int(round(255 * min(1, t * 0.8))) for t in cyl[i, 7:10]) + (255,)
            items.append(((z0[i] + z1[i]) / 2, 1, (x0[i], y0[i], x1[i], y1[i], c,
                                                   max(1, int(round(2 * cyl[i, 6] * scale))))))
    sph = scene.get("spheres")
    if sph is not None and len(sph):
        sph = np.asarray(sph, float)
        x, y, z = proj(sph[:, 0:3])
        for i in range(len(sph)):
            items.append((z[i], 2, (x[i], y[i], sph[i, 3] * scale, tuple(sph[i, 4:7]))))
    labs = scene.get("labels")
    if labs is not None and len(labs[0]):
        x, y, z = proj(np.asarray(labs[1], float))
        for i, t in enumerate(labs[0]):
            items.append((z[i] + float(labs[2][i]), 3, (x[i], y[i], t)))
    lfont = None
    for fp in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            lfont = ImageFont.truetype(fp, max(6, int(label_px)))
            break
        except Exception:
            continue
    if lfont is None:
        lfont = ImageFont.load_default()
    items.sort(key=lambda t: t[0])
    cache = {} if sprite_cache is None else sprite_cache
    for _, kind, d in items:
        if kind == 0:
            dr.polygon(d[:6], fill=d[6])
        elif kind == 1:
            dr.line(d[:4], fill=d[4], width=d[5])
        elif kind == 3:
            dr.text((d[0], d[1]), d[2], fill=(0, 0, 0, 255), font=lfont, anchor="mm",
                    stroke_width=2, stroke_fill=(255, 255, 255, 255))
        else:
            x, y, r, c = d
            key = (c, int(round(r)))
            if key not in cache:
                cache[key] = make_sphere_sprite(c, r)
            spr = cache[key]
            img.alpha_composite(spr, (int(round(x - spr.width / 2)), int(round(y - spr.height / 2))))
            dr = ImageDraw.Draw(img, "RGBA")
    if scene.get("triad") is not None:
        vecs, names = scene["triad"]
        u = ui_scale
        ox, oy, size = 60 * u, H - 60 * u, 38 * u
        try:
            font = ImageFont.truetype("arialbd.ttf", int(14 * u))
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(14 * u))
            except Exception:
                font = ImageFont.load_default()
        fg = (0, 0, 0, 255) if sum(bg) > 380 else (255, 255, 255, 255)
        its = []
        for v, name, c in zip(vecs, names, ((217, 0, 0, 255), (0, 166, 0, 255), (0, 0, 217, 255))):
            p = (np.asarray(v, float) / np.linalg.norm(v)) @ R.T
            its.append((p[2], name, c, ox + p[0] * size, oy - p[1] * size))
        for _, name, c, x, y in sorted(its, key=lambda t: t[0]):
            dr.line([ox, oy, x, y], fill=c, width=max(2, int(3 * u)))
            dr.text((ox + (x - ox) * 1.35, oy + (y - oy) * 1.35), name, fill=fg, font=font, anchor="mm")
    return img.convert("RGB")


def colorbar_image(width, height, vmin, vmax, title="", bg=(0, 0, 0), ui_scale=1.0):
    """FermiSurfer と同じ BGR カラーバー（縦）。上が最大"""
    from PIL import Image, ImageDraw, ImageFont
    bgc = tuple(int(round(255 * x)) for x in bg)
    fg = (0, 0, 0) if sum(bgc) > 380 else (255, 255, 255)
    img = Image.new("RGB", (width, height), bgc)
    dr = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", int(12 * ui_scale))
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", int(12 * ui_scale))
        except Exception:
            font = ImageFont.load_default()
    pad_t, pad_b = int(34 * ui_scale), int(26 * ui_scale)
    bw = int(18 * ui_scale)
    x0 = int(10 * ui_scale)
    hbar = max(10, height - pad_t - pad_b)
    t = np.linspace(1, 0, hbar)
    cols = (bgr_colors(t) * 255).astype(np.uint8)
    bar = np.repeat(cols[:, None, :], bw, axis=1)
    img.paste(Image.fromarray(bar, "RGB"), (x0, pad_t))
    dr.rectangle([x0, pad_t, x0 + bw, pad_t + hbar], outline=fg)
    if title:
        dr.text((x0, int(6 * ui_scale)), title, fill=fg, font=font)
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + hbar * (1 - f)
        dr.line([x0 + bw, y, x0 + bw + int(4 * ui_scale), y], fill=fg)
        dr.text((x0 + bw + int(7 * ui_scale), y), "%.4g" % (vmin + (vmax - vmin) * f), fill=fg, font=font,
                anchor="lm")
    return img


class Viewer3D:
    """3D 表示領域（OpenGL 3.3 → 使えなければ Pillow の簡易描画）。
    左ドラッグ: 回転 / 右ドラッグ: 移動 / ホイール: 拡大縮小（結晶構造表示と同じ操作）"""

    def __init__(self, tk, parent, on_status=None):
        self.tk = tk
        self.holder = parent
        self.on_status = on_status
        self.scene = None
        self.view = {"R": np.eye(3), "zoom": 1.0, "pan": [0.0, 0.0], "scale": None, "auto": True,
                     "drag": None, "pending": False, "scene": None}
        self.label_pt = FS_LABEL_SIZE          # ラベルの文字サイズ [pt]
        self.view.update(label_px=lambda: self.label_pt * 1.33 * UI_SCALE, on_gl_fail=self._on_gl_fail,
                         on_resize=self._on_gl_resize)
        self.mode = None
        self.w = None
        self.gl_error = None
        self._photo = None
        self._sprites = {}
        try:
            if os.environ.get("VASP_VIEWER_NO_GL"):          # 環境変数で OpenGL を使わない設定にできる
                raise RuntimeError("環境変数 VASP_VIEWER_NO_GL が設定されています")
            View = make_scene_gl_view_class()
            w = View(parent, self.view, width=800, height=700)
            w.pack(fill="both", expand=True)
            self.w, self.mode = w, "gl2"
            self._bind_mouse(w)
        except Exception as e:
            self._use_pil(e)

    # ---- 描画方式 ----
    def engine_text(self):
        if self.mode == "gl2":
            return "OpenGL 3.3（GPU、4x MSAA）"
        return "簡易描画（OpenGL 3.3 が使えないため。回転は遅くなります）"

    def _use_pil(self, reason=None):
        if self.w is not None:
            try:
                self.w.destroy()
            except Exception:
                pass
        cv = self.tk.Canvas(self.holder, background="white", highlightthickness=0)
        cv.pack(fill="both", expand=True)
        self.w, self.mode = cv, "pil"
        self.gl_error = None if reason is None else str(reason)
        self._bind_mouse(cv)
        cv.bind("<Configure>", lambda e: self.fit(False) if (self.view["auto"] and self.scene is not None)
                else self.request())
        if self.scene is not None:
            self.holder.after(50, lambda: self.fit(self.view["auto"]))
        if self.on_status:
            self.on_status()

    def _on_gl_fail(self, e, kind=None):
        if self.mode != "pil":
            self._use_pil(e)
            if getattr(self, "on_mode_change", None):
                self.holder.after(80, self.on_mode_change)

    def volume_ok(self):
        """GPU レイキャスティングが使えるか（True / False。まだ OpenGL を初期化していなければ None）"""
        if self.mode != "gl2":
            return False
        ren = getattr(self.w, "ren", None)
        if ren is None or not ren.ready:
            return None
        return ren.p_vol is not None

    def _on_gl_resize(self):
        if self.view["auto"] and self.scene is not None:
            self.fit(False)

    # ---- 内容・視点 ----
    def size(self):
        if self.w is None:
            return 800, 700
        return max(self.w.winfo_width(), 200), max(self.w.winfo_height(), 200)

    def set_scene(self, scene, refit=None):
        """scene を差し替える。refit=True なら全体表示に戻す（None: 最初だけ）"""
        first = self.scene is None
        self.scene = scene
        if self.mode == "gl2":
            self.w.set_scene(scene)
        if refit or (refit is None and first) or self.view["scale"] is None:
            self.fit(first or bool(refit))
        else:
            self.redraw()

    def fit(self, reset=False):
        sc = self.scene
        if sc is None:
            return
        v = self.view
        if reset:
            v["zoom"], v["pan"], v["auto"] = 1.0, [0.0, 0.0], True
        self.holder.update_idletasks()
        W, H = self.size()
        P = np.asarray(sc["fit"], float) @ v["R"].T
        c = np.asarray(sc["center"], float) @ v["R"].T
        half_w = max(np.abs(P[:, 0] - c[0]).max(), 1e-6)
        half_h = max(np.abs(P[:, 1] - c[1]).max(), 1e-6)
        v["scale"] = min(W / (2 * half_w), H / (2 * half_h)) * 0.90
        self.redraw()

    def set_view(self, R):
        self.view["R"] = np.asarray(R, float)
        self.view["pan"] = [0.0, 0.0]
        self.fit(False)

    def redraw(self):
        v = self.view
        v["pending"] = False
        if self.scene is None or v["scale"] is None or self.w is None:
            return
        if self.mode == "gl2":
            self.w.request()
        else:
            from PIL import ImageTk
            W, H = self.size()
            img = render_scene_pil(self.scene, v["R"], v["scale"] * v["zoom"], v["pan"], W, H, UI_SCALE,
                                   self._sprites, max_tris=40000 if v["drag"] is not None else None,
                                   label_px=v["label_px"]())
            self._photo = ImageTk.PhotoImage(img, master=self.w)
            self.w.delete("all")
            self.w.create_image(0, 0, image=self._photo, anchor="nw")
            if len(self._sprites) > 300:
                self._sprites.clear()

    def request(self):
        """マウス移動のたびに描かず、アイドル時にまとめて 1 回描く"""
        if self.mode == "gl2":
            self.redraw()
        elif not self.view["pending"]:
            self.view["pending"] = True
            self.holder.after_idle(self.redraw)

    def grab(self):
        """現在の表示を PIL 画像で返す"""
        if self.scene is None:
            return None
        if self.mode == "gl2":
            return self.w.grab()
        v = self.view
        W, H = self.size()
        k = 2
        return render_scene_pil(self.scene, v["R"], v["scale"] * v["zoom"] * k, v["pan"], W * k, H * k,
                                UI_SCALE * k, label_px=v["label_px"]() * k)

    def fps_text(self):
        ts = getattr(self.w, "_times", [])
        if self.mode == "gl2" and len(ts) >= 5 and ts[-1] > ts[0]:
            return (f"回転中の表示 約 {(len(ts) - 1) / (ts[-1] - ts[0]):.0f} fps  "
                    f"GPU: {getattr(self.w, 'gl_info', '')}")
        return ""

    # ---- マウス操作 ----
    def _bind_mouse(self, w):
        v = self.view

        def press(ev, btn):
            v["drag"] = (btn, ev.x, ev.y, v["R"].copy(), list(v["pan"]))

        def move(ev):
            if v["drag"] is None or v["scale"] is None:
                return
            btn, x0, y0, R0, p0 = v["drag"]
            dx, dy = ev.x - x0, ev.y - y0
            if btn == 1:
                v["R"] = rot_x(np.radians(dy * 0.4)) @ rot_y(np.radians(dx * 0.4)) @ R0
            else:
                s = v["scale"] * v["zoom"]
                v["pan"] = [p0[0] - dx / s, p0[1] + dy / s]
                v["auto"] = False
            self.request()

        def release(ev):
            v["drag"] = None
            if self.on_status:
                self.on_status()
            if self.mode == "gl2":
                self.w._times = []
            else:
                self.request()                 # 簡易描画: ドラッグ中は間引いているので、離したら全部描く

        def wheel(ev, delta=None):
            d = delta if delta is not None else ev.delta
            v["zoom"] *= 1.12 if d > 0 else 1 / 1.12
            v["auto"] = False
            self.request()

        w.bind("<ButtonPress-1>", lambda e: press(e, 1))
        w.bind("<ButtonPress-3>", lambda e: press(e, 3))
        w.bind("<B1-Motion>", move)
        w.bind("<B3-Motion>", move)
        w.bind("<ButtonRelease-1>", release)
        w.bind("<ButtonRelease-3>", release)
        w.bind("<MouseWheel>", wheel)                       # Windows
        w.bind("<Button-4>", lambda e: wheel(e, 1))         # Linux
        w.bind("<Button-5>", lambda e: wheel(e, -1))


# ----------------------------------------------------------------------
#  タブ 5, 6: 結晶構造 / ブリルアンゾーン（VASP Input Generator の「結晶構造を表示」「ブリルアンゾーンを表示」と同じ内容）
#  計算フォルダ・CHGCAR・vasprun.xml などを開くと、その構造が自動でこの 2 つのタブにも入る
#  （タブを開いたときに描く。開いていないタブの描画で待たされないようにするため）
# ----------------------------------------------------------------------
STRUCT_FILETYPES = [("構造ファイル", ("POSCAR*", "CONTCAR*", "*.vasp", "*.poscar", "*POSCAR", "*CONTCAR")),
                    ("All files", "*")]


def structure_from_symbols(comment, lattice, symbols, frac):
    """原子ごとの元素記号の列 → Structure（同じ元素が続く部分を 1 つの組にまとめる）"""
    species, counts = [], []
    for s in symbols:
        if species and species[-1] == s:
            counts[-1] += 1
        else:
            species.append(s)
            counts.append(1)
    return Structure(comment, np.asarray(lattice, float), species, counts, np.asarray(frac, float))


def find_structure_file(folder):
    """フォルダの構造ファイル: CONTCAR（中身があるとき）→ POSCAR の順。見つからなければ None"""
    for n in ("CONTCAR", "POSCAR"):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            try:
                parse_poscar(_read_text(p))
                return p
            except Exception:
                continue
    return None


def spacegroup_text(st, symprec=1e-3):
    try:
        import spglib
        ds = spglib.get_symmetry_dataset(st.spglib_cell(), symprec=symprec)
        if ds is None:
            return "—"
        num = int(ds.number if hasattr(ds, "number") else ds["number"])
        sym = ds.international if hasattr(ds, "international") else ds["international"]
        return f"{sym} (#{num})"
    except Exception:
        return "—"


def close_pair_warning(dmin):
    """物理的にありえないほど近い原子対（共有結合半径の和の 0.5 倍未満）"""
    return ", ".join(f"{a}–{b} {d:.2f} Å" for (a, b), d in sorted(dmin.items())
                     if d < 0.5 * (ATOM_STYLE.get(a, (1.5, ""))[0] + ATOM_STYLE.get(b, (1.5, ""))[0]))


def bz_datasets_safe(st):
    """bz_datasets（seekpath を使う）。seekpath がないときや失敗したときは、元の構造の BZ だけ（高対称点なし）"""
    try:
        return bz_datasets(st), ""
    except Exception as e:
        B = reciprocal_2pi(st.lattice)
        d = {"title": "元の構造 (入力 POSCAR)", "B": B, "A": np.array(st.lattice, float),
             "res": {"path": [], "point_coords": {}}, "note": "",
             "label": f"空間群 {spacegroup_text(st)}"}
        d["V"], d["faces"] = brillouin_zone(B)
        d["cart"] = {}
        why = ("seekpath がインストールされていません" if isinstance(e, ImportError) else f"{type(e).__name__}: {e}")
        return [d], f"推奨パスと高対称点は表示できません（{why}）。"


class _StructureSource:
    """結晶構造・BZ タブ共通: 構造の受け取り（自動の場合はタブを開いたときに描く）"""

    def _src_init(self):
        self.st = None
        self.src = ""
        self.pending = None

    def set_structure(self, st, src, origin="", show=False):
        """構造を渡す。show=True ならこのタブを開いて描く。False なら、タブが表示中のときだけすぐ描く"""
        self.pending = (st, src, origin)
        if show:
            self.app.select_tab(self)
        if show or self.app.current_tab() is self:
            self.app.root.after(30, self._apply_pending)

    def on_show(self):
        if self.pending is not None:
            self.app.root.after(30, self._apply_pending)

    def _apply_pending(self):
        if self.pending is None:
            return
        st, src, origin = self.pending
        self.pending = None
        remember_view(self.app, self)
        try:
            self._install_structure(st, src, origin)
        except Exception as e:
            self.app.messagebox.showerror(self.title, f"表示できませんでした。\n\n{type(e).__name__}: {e}")
            return
        restore_view(self.app, self, f"{self.KEY}:" + (os.path.abspath(src) if src else st.formula))

    def choose(self):
        p = self.app.filedialog.askopenfilename(title="構造ファイル（POSCAR / CONTCAR）を選択",
                                                filetypes=STRUCT_FILETYPES)
        if p:
            self.app.load_structure(p)


class StructureTab(_StructureSource):
    title = "結晶構造"
    KEY = "struct"
    STYLE = ("bond_var", "outside_var", "packed_var", "stub_var", "cell_var", "axes_var", "ball_var", "bondr_var",
             "lsize_var")

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self._src_init()
        self.d = None
        side, holder = make_side_layout(app, frame, on_toggle=lambda: self.viewer.fit(self.viewer.view["auto"]))
        self.side = side
        bar = ttk.Frame(holder, padding=(4, 3, 4, 3))      # 描画領域の上: 視点・画像の保存（電荷密度タブと同じ配置）
        bar.pack(side="top", fill="x")
        self.viewer = Viewer3D(tk, holder, on_status=self._perf)
        L = lambda i: (lambda: self.d["st"].lattice[i])
        Rc = lambda i: (lambda: np.linalg.inv(self.d["st"].lattice).T[i])
        view_toolbar(app, bar, self.viewer,
                     [("a", L(0), Rc(2)), ("b", L(1), Rc(2)), ("c", L(2), Rc(1)),
                      ("a*", Rc(0), L(2)), ("b*", Rc(1), L(2)), ("c*", Rc(2), L(1))], self._home, self.save_image)

        section(app, side, "構造")
        ttk.Button(side, text="POSCAR / CONTCAR を開く…", command=self.choose).pack(anchor="w")
        ttk.Label(side, text="計算フォルダ・CHGCAR・vasprun.xml などを開くと、その構造も自動でここに入ります",
                  foreground="#666", wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w")
        self.put_info = info_text(app, side, height=8)

        section(app, side, "スーパーセル")
        fs = ttk.Frame(side)
        fs.pack(anchor="w")
        self.nvars = [tk.StringVar(value="1") for _ in range(3)]
        for lab, v in zip("abc", self.nvars):
            ttk.Label(fs, text=lab).pack(side="left")
            en = ttk.Entry(fs, textvariable=v, width=3)
            en.pack(side="left", padx=(1, 5))
            en.bind("<Return>", lambda ev: self.rebuild())
        ttk.Button(fs, text="更新", command=self.rebuild).pack(side="left")
        self.cell_lbl = ttk.Label(side, text="", foreground="#333", wraplength=int(330 * UI_SCALE), justify="left")
        self.cell_lbl.pack(anchor="w", pady=(4, 0))
        self.perf = ttk.Label(side, text="", foreground="#666", wraplength=int(330 * UI_SCALE), justify="left")
        self.perf.pack(anchor="w")

        section(app, side, "表示")
        self.bond_var = tk.BooleanVar(value=True)
        self.outside_var = tk.BooleanVar(value=True)
        self.packed_var = tk.BooleanVar(value=True)
        self.stub_var = tk.BooleanVar(value=False)
        self.cell_var = tk.BooleanVar(value=True)
        self.axes_var = tk.BooleanVar(value=True)
        for text, var, fn in (("結合", self.bond_var, self.make_scene),
                              ("結合相手を境界外にも表示（VESTA 既定）", self.outside_var, self.rebuild),
                              ("境界上の原子（面・辺・頂点）", self.packed_var, self.rebuild),
                              ("範囲外への結合を半分だけ表示", self.stub_var, self.rebuild),
                              ("単位胞", self.cell_var, self.make_scene),
                              ("結晶軸 a, b, c", self.axes_var, self.make_scene)):
            ttk.Checkbutton(side, text=text, variable=var, command=fn).pack(anchor="w")
        fsz = ttk.Frame(side)
        fsz.pack(anchor="w", pady=(4, 0))
        self.ball_var = tk.StringVar(value=str(BALL_SCALE))
        self.bondr_var = tk.StringVar(value=str(BOND_RADIUS))
        self.lsize_var = tk.StringVar(value=str(LABEL_SIZE))
        for r_, (lab, var) in enumerate((("球の大きさ（×原子半径）", self.ball_var), ("結合の半径 [Å]", self.bondr_var),
                                         ("ラベル文字サイズ [pt]", self.lsize_var))):
            ttk.Label(fsz, text=lab).grid(row=r_, column=0, sticky="w")
            en = ttk.Entry(fsz, textvariable=var, width=6)
            en.grid(row=r_, column=1, padx=4)
            en.bind("<Return>", lambda ev: self.rebuild())
        ttk.Button(fsz, text="反映", command=self.rebuild).grid(row=0, column=2, rowspan=3)

        section(app, side, "ラベル表示（チェックした元素のみ）")
        self.lab_frame = ttk.Frame(side)
        self.lab_frame.pack(anchor="w")
        ttk.Label(self.lab_frame, text="（読み込むと表示されます）", foreground="#777").pack()
        self.lab_vars = {}
        fla = ttk.Frame(side)
        fla.pack(anchor="w")
        ttk.Button(fla, text="全部", command=lambda: self._labels_all(True)).pack(side="left")
        ttk.Button(fla, text="なし", command=lambda: self._labels_all(False)).pack(side="left", padx=4)

        section(app, side, "結合の設定（元素対ごと）")
        self.bond_frame = ttk.Frame(side)
        self.bond_frame.pack(anchor="w")
        ttk.Label(self.bond_frame, text="（読み込むと表示されます）", foreground="#777").grid()
        self.rule_vars = {}
        fbb = ttk.Frame(side)
        fbb.pack(anchor="w", pady=(4, 0))
        ttk.Button(fbb, text="結合を再計算", command=self._rules_changed).pack(side="left")
        ttk.Button(fbb, text="既定値に戻す", command=self._rules_default).pack(side="left", padx=6)
        ttk.Label(side, text="既定値は VESTA の結合距離表。表にない対は、\n13–16族どうし（骨格）以外はオフ",
                  foreground="#666", justify="left").pack(anchor="w")
        self.viewer.w.after(200, self._placeholder)

    def _placeholder(self):
        if self.d is None and self.viewer.mode == "pil":
            try:
                self.viewer.w.create_text(20, 20, anchor="nw", fill="#777",
                                          text="POSCAR / CONTCAR を開くか、計算フォルダを開くと表示されます")
            except Exception:
                pass

    # 結合の表・スーパーセル・ラベル・画像の保存は電荷密度タブと同じ処理
    def _build_bond_rows(self, dmin):
        return ChargeTab._build_bond_rows(self, dmin)

    def _current_rules(self):
        return ChargeTab._current_rules(self)

    def _rules_changed(self):
        return ChargeTab._rules_changed(self)

    def _rules_default(self):
        return ChargeTab._rules_default(self)

    def _read_n(self):
        return ChargeTab._read_n(self)

    def _labels_all(self, on):
        return ChargeTab._labels_all(self, on)

    def save_image(self):
        return ChargeTab.save_image(self)

    def _home(self):
        st = self.d["st"]
        return view_matrix(st.lattice[0], np.linalg.inv(st.lattice).T[2])

    def _perf(self):
        if not hasattr(self, "perf") or not hasattr(self, "viewer"):
            return
        self.perf.configure(text=f"描画: {self.viewer.engine_text()}\n{self.viewer.fps_text()}")

    def _install_structure(self, st, src, origin):
        app = self.app
        tk, ttk = app.tk, app.ttk
        dmin = shortest_pair_distances(st)
        rules = default_bond_rules(st.species, dmin)
        old_rules = self._current_rules() if self.d is not None and self.rule_vars else {}
        for key in rules:                     # 同じ元素対の設定は、読み込み直しても引き継ぐ
            if key in old_rules:
                rules[key] = list(old_rules[key])
        self.st, self.src = st, src
        self.d = {"st": st, "rules": rules, "n": [1, 1, 1], "dmin": dmin, "model": None,
                  "warn": close_pair_warning(dmin)}
        old = {k: v.get() for k, v in self.lab_vars.items()}
        for w in self.lab_frame.winfo_children():
            w.destroy()
        self.lab_vars = {}
        for k, el in enumerate(st.species):
            if el in self.lab_vars:
                continue
            v = tk.BooleanVar(value=old.get(el, False))
            self.lab_vars[el] = v
            kk = len(self.lab_vars) - 1
            tk.Label(self.lab_frame, text="●", fg=_hex(element_style(el)[1]), font=("", 13)).grid(
                row=kk // 4, column=2 * (kk % 4))
            ttk.Checkbutton(self.lab_frame, text=el, variable=v, command=self.make_scene).grid(
                row=kk // 4, column=2 * (kk % 4) + 1, sticky="w", padx=(0, 8))
        self._build_bond_rows(dmin)
        self.side.fit_width()
        la, ang, vol = lattice_params(st)
        self.put_info(f"{origin or 'ファイル'}: {os.path.basename(src) if src else '—'}\n"
                      f"組成  : {st.formula}  ({st.natoms} atoms)\n"
                      f"空間群: {spacegroup_text(st)}\n"
                      f"a,b,c : {la[0]:.4f} {la[1]:.4f} {la[2]:.4f} Å\n"
                      f"α,β,γ : {ang[0]:.2f} {ang[1]:.2f} {ang[2]:.2f}°\n"
                      f"体積  : {vol:.3f} Å³\n"
                      + (f"場所  : {os.path.dirname(os.path.abspath(src))}" if src else ""))
        for v in self.nvars:
            v.set("1")
        self.viewer.view["R"] = self._home()
        self.viewer.scene = None
        app.root.update_idletasks()
        self.rebuild(refit=True, quiet=True)

    def _build_model(self, n):
        d = self.d
        d["n"] = list(n)
        d["model"] = build_structure_model(d["st"], n, self.packed_var.get(), d["rules"],
                                           fnum(self.ball_var, BALL_SCALE, 0.05, 2.0),
                                           self.stub_var.get(), self.outside_var.get())

    def rebuild(self, refit=None, quiet=False):
        """スーパーセル・境界・球の大きさなどを反映して原子と結合を作り直す"""
        d = self.d
        if d is None:
            return
        n = self._read_n()
        if n is None:
            self.app.messagebox.showwarning("スーパーセル", "1 以上の整数を入力してください。")
            return
        est = d["st"].natoms * n[0] * n[1] * n[2]
        if not quiet and n != d["n"] and est > 5000 and not self.app.messagebox.askyesno(
                "スーパーセル", f"原子数が約 {est} 個になり、表示が重くなります。続けますか？"):
            return
        if refit is None:
            refit = n != d["n"]
        with _Busy(self.app, "結晶構造を作成中…"):
            self._build_model(n)
            self.make_scene(refit=refit)

    def make_scene(self, refit=None):
        d = self.d
        if d is None or d.get("model") is None:
            return
        st, m, n = d["st"], d["model"], d["n"]
        sc = empty_scene()
        cart, rad = m["cart"], m["rad"]
        col = np.array(m["col"], float)
        sc["spheres"] = np.hstack([cart, rad[:, None], col]).astype(np.float32)
        br = fnum(self.bondr_var, BOND_RADIUS, 0.01, 1.0)
        cy = []
        if self.bond_var.get() and len(m["bonds"]):
            i, j = np.asarray(m["bonds"], int).T
            mid = 0.5 * (cart[i] + cart[j])
            r = np.full((len(i), 1), br)
            cy.append(np.hstack([cart[i], mid, r, col[i]]))
            cy.append(np.hstack([mid, cart[j], r, col[j]]))
        if self.bond_var.get() and self.stub_var.get() and m.get("stubs"):
            i = np.array([s[0] for s in m["stubs"]], int)
            q = np.array([s[1] for s in m["stubs"]], float)
            cy.append(np.hstack([cart[i], 0.5 * (cart[i] + q), np.full((len(i), 1), br), col[i]]))
        if cy:
            sc["cyls"] = np.vstack(cy).astype(np.float32)
        if self.cell_var.get():
            sc["lines"] = [(box_edges(st.lattice), (0, 0, 0), 1.3)]
        on = {el for el, v in self.lab_vars.items() if v.get()}
        idx = [k for k, s in enumerate(m["sy"]) if s in on]
        if idx:
            sc["labels"] = ([m["lb"][k] for k in idx], cart[idx], rad[idx] * 1.02)
        self.viewer.label_pt = fnum(self.lsize_var, LABEL_SIZE, 4, 40)
        if self.axes_var.get():
            sc["triad"] = (st.lattice, ("a", "b", "c"))
        a, b, c = st.lattice * np.array(n, float)[:, None]
        corners = np.array([[0, 0, 0], a, b, c, a + b, a + c, b + c, a + b + c])
        sc["center"] = corners.mean(axis=0)
        sc["radius"] = float(np.linalg.norm(np.vstack([corners, cart]) - sc["center"], axis=1).max()) + 3.0
        sc["fit"] = np.vstack([corners, cart + rad[:, None], cart - rad[:, None]])
        self.cell_lbl.configure(
            text=f"{n[0]}×{n[1]}×{n[2]}  表示原子 {len(cart)} 個（うち境界外の結合相手 {m['n_outside']} 個）  "
                 f"結合 {len(m['bonds'])} 本\n色・原子半径: VESTA の elements.ini 既定値 / 正射影"
                 + (f"\n【注意】異常に近い原子対: {d['warn']}" if d["warn"] else ""),
            foreground="#b00000" if d["warn"] else "#333")
        self.viewer.set_scene(sc, refit=refit)
        self._perf()

    # ---- 表示の状態・ライブラリ ----
    def view_state(self):
        if self.d is None:
            return None
        st = {"vars": capture_vars(self, skip=("rule_vars",)), "view": viewer_state(self.viewer)}
        try:
            st["rules"] = [[a, b, bool(r[0]), float(r[1])] for (a, b), r in self._current_rules().items()]
        except Exception:
            pass
        return st

    def apply_view_state(self, stt):
        d = self.d
        if d is None or not stt:
            return
        restore_vars(self, stt.get("vars"))
        if stt.get("rules"):
            for a, b, on, dmax in stt["rules"]:
                if (a, b) in self.rule_vars:
                    self.rule_vars[(a, b)][0].set(on)
                    self.rule_vars[(a, b)][1].set(f"{dmax:.3f}")
            d["rules"] = self._current_rules()
        if self._read_n() is None:
            for v in self.nvars:
                v.set("1")
        self.rebuild(refit=False, quiet=True)
        apply_viewer_state(self.viewer, stt.get("view"))

    def library_payload(self):
        if self.d is None:
            return None
        obj = {"st": self.d["st"], "path": self.src, "view_state": self.view_state(), "tab": "struct"}
        return "structure", obj, f"{self.d['st'].formula} {os.path.basename(self.src)}", \
            {"formula": self.d["st"].formula, "source": self.src}

    def library_open(self, obj, meta):
        self.pending = None
        remember_view(self.app, self)
        self._install_structure(obj["st"], obj.get("path", ""), f"ライブラリ「{meta.get('name', '')}」")
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), obj.get("view_state"))


class BZTab(_StructureSource):
    title = "ブリルアンゾーン"
    KEY = "bz"
    STYLE = ("ax_vars", "sync_var")

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self._src_init()
        self.panes = []
        bar = ttk.Frame(frame, padding=6)
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="POSCAR / CONTCAR を開く…", command=self.choose).pack(side="left")
        ttk.Button(bar, text="初期位置に戻す", command=self.reset_all).pack(side="left", padx=(10, 0))
        self.sync_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="左右の回転を連動", variable=self.sync_var).pack(side="left", padx=10)
        ttk.Button(bar, text="画像を保存…", command=self.save_png).pack(side="left", padx=4)
        self.ax_vars = {"recip": tk.BooleanVar(value=True), "abc": tk.BooleanVar(value=False),
                        "xyz": tk.BooleanVar(value=False)}
        ttk.Label(bar, text="   表示する軸:").pack(side="left")
        for key, text in (("recip", "逆格子 b₁b₂b₃（緑）"), ("abc", "実空間 a,b,c（橙）"),
                          ("xyz", "直交座標 x,y,z（灰・破線）")):
            ttk.Checkbutton(bar, text=text, variable=self.ax_vars[key],
                            command=lambda: [self.request(p) for p in self.panes]).pack(side="left", padx=4)
        bar2 = ttk.Frame(frame, padding=(6, 0, 6, 4))
        bar2.pack(side="top", fill="x")
        self.src_lbl = ttk.Label(bar2, text="POSCAR / CONTCAR を開くか、計算フォルダを開くと表示されます",
                                 foreground="#333")
        self.src_lbl.pack(side="top", anchor="w")
        ttk.Label(bar2, text="左ドラッグ: 回転 / 右ドラッグ: 移動 / ホイール: 拡大縮小    "
                             "赤: 推奨パス (HPKOT)   軸の矢印は向きのみ（長さはそろえて表示）   破線の辺: 奥の辺   "
                             "x,y,z は各セル自身の直交座標系",
                  foreground="#555").pack(side="top", anchor="w")
        self.body = ttk.Frame(frame)
        self.body.pack(fill="both", expand=True)
        import tkinter.font as tkfont
        self.f_lab = tkfont.Font(frame, family="Arial", size=13, weight="bold")
        self.f_lab_sub = tkfont.Font(frame, family="Arial", size=9, weight="bold")
        self.f_vec = tkfont.Font(frame, family="Arial", size=12, weight="bold")
        self.f_vec_sub = tkfont.Font(frame, family="Arial", size=9, weight="bold")

    @staticmethod
    def default_R():
        """初期の向き: 少し斜め上から（kz が上）"""
        e, a = np.radians(22), np.radians(-60)
        return view_matrix([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], [0, 0, 1.0])

    def _install_structure(self, st, src, origin):
        tk, ttk = self.app.tk, self.app.ttk
        with _Busy(self.app, "ブリルアンゾーンを作成中…"):
            datasets, note = bz_datasets_safe(st)
        self.st, self.src = st, src
        for w in self.body.winfo_children():
            w.destroy()
        self.panes = []
        for d in datasets:
            fr = ttk.Frame(self.body)
            fr.pack(side="left", fill="both", expand=True)
            title = [d["title"], d["label"]] + ([d["note"]] if d["note"] else [])
            ttk.Label(fr, text="\n".join(title), font=("", 10, "bold"), justify="center").pack(pady=(4, 0))
            cv = tk.Canvas(fr, background="white", highlightthickness=0)
            cv.pack(fill="both", expand=True)
            path_str = "  |  ".join("–".join(kpt_label(x) for x in seg) for seg in d["res"]["path"])
            ttk.Label(fr, text=path_str, foreground="#333", wraplength=int(620 * UI_SCALE)).pack(pady=(0, 6))
            p = {"d": d, "cv": cv, "R": self.default_R(), "zoom": 1.0, "pan": [0.0, 0.0],
                 "drag": None, "pending": False, "title": title, "path": path_str}
            self.panes.append(p)
            self._bind(p)
        self.src_lbl.configure(
            text=f"{origin or 'ファイル'}: {os.path.basename(src) if src else '—'}  —  {st.formula}"
                 + (f"   （{os.path.dirname(os.path.abspath(src))}）" if src else "")
            + (f"\n【注意】{note}" if note else ""),
            foreground="#b00000" if note else "#333")

    # ---- 描画（VASP Input Generator と同じ）----
    def _text_sub(self, cv, x, y, name, fill, fmain, fsub, halo=False):
        base, sub = label_parts(name)
        offs = ((-1, 0), (1, 0), (0, -1), (0, 1)) if halo else ()
        w = fmain.measure(base)
        for ox, oy in offs:
            cv.create_text(x + ox, y + oy, text=base, fill="white", font=fmain, anchor="sw")
            if sub:
                cv.create_text(x + w + 1 + ox, y + 4 + oy, text=sub, fill="white", font=fsub, anchor="sw")
        cv.create_text(x, y, text=base, fill=fill, font=fmain, anchor="sw")
        if sub:
            cv.create_text(x + w + 1, y + 4, text=sub, fill=fill, font=fsub, anchor="sw")

    def shown_axes(self):
        return tuple(k for k, v in self.ax_vars.items() if v.get())

    def draw(self, p):
        p["pending"] = False
        cv = p["cv"]
        try:
            W, H = max(cv.winfo_width(), 100), max(cv.winfo_height(), 100)
        except Exception:
            return
        s = bz_fit_scale(p["d"], W, H, self.shown_axes()) * p["zoom"]
        prims = bz_primitives(p["d"], p["R"], W, H, s, p["pan"], self.shown_axes())
        cv.delete("all")
        for kind, dd in prims:
            if kind == "poly":
                cv.create_polygon([c for xy in dd for c in xy], fill="#dde8f3", outline="")
            elif kind == "edge_back":
                cv.create_line(*dd[0], *dd[1], fill="#8c96a5", dash=(4, 4))
            elif kind == "edge_front":
                cv.create_line(*dd[0], *dd[1], fill="#3c506e", width=2)
            elif kind == "arrow":
                p0, p1, name, ak = dd
                colr, dash = BZ_AXIS_STYLE[ak]
                kw = {"dash": dash} if dash else {}
                cv.create_line(*p0, *p1, fill=colr, width=2, arrow="last", arrowshape=(12, 14, 5), **kw)
                self._text_sub(cv, p1[0] + 8, p1[1] - 2, name, colr, self.f_vec, self.f_vec_sub)
            elif kind == "path":
                cv.create_line(*dd[0], *dd[1], fill="#d62728", width=3, capstyle="round")
            elif kind == "kpt":
                (x, y), name = dd
                cv.create_oval(x - 4.5, y - 4.5, x + 4.5, y + 4.5, fill="#d62728", outline="")
                self._text_sub(cv, x + 7, y - 2, name, "black", self.f_lab, self.f_lab_sub, halo=True)

    def request(self, p):
        if not p["pending"]:
            p["pending"] = True
            p["cv"].after_idle(lambda: self.draw(p))

    def reset_all(self):
        for p in self.panes:
            p["R"], p["zoom"], p["pan"] = self.default_R(), 1.0, [0.0, 0.0]
            self.request(p)

    def _bind(self, p):
        cv = p["cv"]

        def press(ev, btn):
            p["drag"] = (btn, ev.x, ev.y, p["R"].copy(), list(p["pan"]))

        def move(ev):
            if p["drag"] is None:
                return
            btn, x0, y0, R0, p0 = p["drag"]
            dx, dy = ev.x - x0, ev.y - y0
            if btn == 1:
                Rn = rot_x(np.radians(dy * 0.4)) @ rot_y(np.radians(dx * 0.4)) @ R0
                if self.sync_var.get():
                    dR = Rn @ p["R"].T             # 連動: もう片方にも同じ回転（差分）を加える
                    for q in self.panes:
                        if q is not p:
                            q["R"] = dR @ q["R"]
                            self.request(q)
                p["R"] = Rn
            else:
                s = bz_fit_scale(p["d"], cv.winfo_width(), cv.winfo_height(), self.shown_axes()) * p["zoom"]
                p["pan"] = [p0[0] - dx / s, p0[1] + dy / s]
            self.request(p)

        def release(ev):
            p["drag"] = None

        def wheel(ev, delta=None):
            dlt = delta if delta is not None else ev.delta
            p["zoom"] *= 1.12 if dlt > 0 else 1 / 1.12
            self.request(p)

        cv.bind("<ButtonPress-1>", lambda e: press(e, 1))
        cv.bind("<ButtonPress-3>", lambda e: press(e, 3))
        cv.bind("<B1-Motion>", move)
        cv.bind("<B3-Motion>", move)
        cv.bind("<ButtonRelease-1>", release)
        cv.bind("<ButtonRelease-3>", release)
        cv.bind("<MouseWheel>", wheel)
        cv.bind("<Button-4>", lambda e: wheel(e, 1))
        cv.bind("<Button-5>", lambda e: wheel(e, -1))
        cv.bind("<Configure>", lambda e: self.request(p))

    def render_image(self):
        """表示中の BZ を 1 枚の画像に（見出し帯付き。左右に並べる）"""
        from PIL import Image
        imgs = []
        for p in self.panes:
            cv = p["cv"]
            W, H = max(cv.winfo_width(), 100), max(cv.winfo_height(), 100)
            s = bz_fit_scale(p["d"], W, H, self.shown_axes()) * p["zoom"]
            prims = bz_primitives(p["d"], p["R"], W, H, s, p["pan"], self.shown_axes())
            imgs.append(render_bz_pil(prims, W, H, p["title"] + [p["path"]], font_px=14))
        out = Image.new("RGB", (sum(i.width for i in imgs), max(i.height for i in imgs)), "white")
        x = 0
        for i in imgs:
            out.paste(i, (x, 0))
            x += i.width
        return out

    def save_png(self):
        if not self.panes:
            return
        fn = self.app.filedialog.asksaveasfilename(title="画像を保存", defaultextension=".png",
                                                   filetypes=[("PNG", "*.png")])
        if not fn:
            return
        self.render_image().save(fn)
        self.app.set_msg(f"画像を保存しました: {fn}")

    # ---- 表示の状態・ライブラリ ----
    def view_state(self):
        if not self.panes:
            return None
        return {"vars": capture_vars(self),
                "panes": [{"R": np.asarray(p["R"], float).tolist(), "zoom": float(p["zoom"]),
                           "pan": [float(x) for x in p["pan"]]} for p in self.panes]}

    def apply_view_state(self, stt):
        if not stt or not self.panes:
            return
        restore_vars(self, stt.get("vars"))
        for p, q in zip(self.panes, stt.get("panes", [])):
            p["R"], p["zoom"], p["pan"] = np.asarray(q["R"], float), float(q["zoom"]), list(q["pan"])
        for p in self.panes:
            self.request(p)

    def library_payload(self):
        if self.st is None:
            return None
        obj = {"st": self.st, "path": self.src, "view_state": None, "bz_state": self.view_state(), "tab": "bz"}
        st_tab = self.app.struct_tab
        if st_tab.d is not None and st_tab.st is self.st:
            obj["view_state"] = st_tab.view_state()
        return "structure", obj, f"{self.st.formula} {os.path.basename(self.src)}", \
            {"formula": self.st.formula, "source": self.src}

    def library_open(self, obj, meta):
        self.pending = None
        remember_view(self.app, self)
        self._install_structure(obj["st"], obj.get("path", ""), f"ライブラリ「{meta.get('name', '')}」")
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), obj.get("bz_state"))


# ======================================================================
#  GUI
# ======================================================================
class _Busy:
    """時間のかかる処理のあいだ、カーソルを砂時計にしてメッセージを出す"""

    def __init__(self, app, text):
        self.app, self.text = app, text

    def __enter__(self):
        self.app.set_msg(self.text)
        try:
            self.app.root.configure(cursor="watch")
        except Exception:
            pass
        self.app.root.update_idletasks()
        self.app.root.update()
        return self

    def __exit__(self, *a):
        try:
            self.app.root.configure(cursor="")
        except Exception:
            pass
        return False


BACKGROUND_LOAD = True      # False: 読み込みを画面のスレッドで行う（テスト用）


def run_in_background(app, text, work, done, fail=None):
    """時間のかかる読み込みを裏のスレッドで行う（画面は止まらない）。
    work(progress) を裏で実行し、終わったら done(結果) を画面のスレッドで呼ぶ。
    progress(文字列) で進み具合を伝えると、メッセージ欄に経過時間と一緒に表示する"""
    import threading
    import time as _t
    job = {"done": False, "res": None, "err": None, "prog": "", "t0": _t.time()}

    def finish():
        try:
            app.root.configure(cursor="")
        except Exception:
            pass
        if job["err"] is not None:
            if fail is not None:
                fail(job["err"])
            else:
                app.messagebox.showerror(text, f"{type(job['err']).__name__}: {job['err']}")
                app.set_msg("")
        else:
            done(job["res"])

    if not BACKGROUND_LOAD:
        try:
            job["res"] = work(lambda s: None)
        except Exception as e:
            job["err"] = e
        job["done"] = True
        finish()
        return job

    def runner():
        try:
            job["res"] = work(lambda s: job.__setitem__("prog", s))
        except BaseException as e:
            job["err"] = e
        job["done"] = True

    threading.Thread(target=runner, daemon=True).start()

    def poll():
        if not job["done"]:
            el = _t.time() - job["t0"]
            app.set_msg(f"{text}" + (f"  {job['prog']}" if job["prog"] else "")
                        + f"\n（経過 {el:.0f} 秒。読み込み中もほかの操作はできます）")
            app.root.after(250, poll)
            return
        job["elapsed"] = _t.time() - job["t0"]
        finish()
    app.root.after(50, poll)
    return job


def make_side_layout(app, parent, on_toggle=None):
    """左: メニュー（縦スクロール・境界線ドラッグで幅変更・◀ で非表示）、右: 描画領域。
    結晶構造ウィンドウ（VASP Input Generator）と同じ構成。戻り値 (side, holder)"""
    tk, ttk = app.tk, app.ttk
    tbar = ttk.Frame(parent)
    tbar.pack(side="left", fill="y")
    paned = tk.PanedWindow(parent, orient="horizontal", sashwidth=max(5, int(6 * UI_SCALE)),
                           sashrelief="raised", showhandle=False, bd=0)
    paned.pack(side="left", fill="both", expand=True)
    side_outer, side = make_scroll_panel(paned, tk, ttk, padding=8, resizable=True)
    paned.add(side_outer, minsize=int(120 * UI_SCALE), stretch="never")
    holder = ttk.Frame(paned)
    paned.add(holder, stretch="always")
    st = {"shown": True}

    def toggle():
        if st["shown"]:
            paned.forget(side_outer)
            btn.configure(text="▶")
        else:
            paned.add(side_outer, before=holder, minsize=int(120 * UI_SCALE), stretch="never")
            btn.configure(text="◀")
        st["shown"] = not st["shown"]
        if on_toggle:
            parent.after(50, on_toggle)

    btn = ttk.Button(tbar, text="◀", width=2, command=toggle)
    btn.pack(side="top", pady=2)

    def fit_width(retry=3):
        """メニューの中身が収まる幅まで広げる（狭めることはしない。ウィンドウ幅の 45% まで）"""
        try:
            if not st["shown"] or not side_outer.winfo_ismapped():
                return
            rootw, cur = app.root.winfo_width(), side_outer.winfo_width()
            if rootw < 400 or cur <= 1:              # まだ画面の配置が終わっていない → 少し待つ
                if retry > 0:
                    parent.after(250, lambda: fit_width(retry - 1))
                return
            need = min(side.winfo_reqwidth() + int(24 * UI_SCALE), int(rootw * 0.45))
            if need > cur:
                paned.paneconfigure(side_outer, width=need)
        except Exception:
            pass

    side.fit_width = lambda: parent.after(60, fit_width)
    side_outer.bind("<Map>", lambda e: parent.after(150, fit_width), add="+")
    return side, holder


def section(app, side, text):
    app.ttk.Label(side, text=text, font=("", 10, "bold")).pack(anchor="w", pady=(8, 2))


def fnum(var, default, lo=-np.inf, hi=np.inf):
    """入力欄の数値を読む。不正なら既定値に戻す"""
    try:
        x = float(var.get())
        if not lo <= x <= hi:
            raise ValueError
        return x
    except (ValueError, TypeError):
        var.set(str(default))
        return default


def info_text(app, side, height=7, width=44):
    tk = app.tk
    t = tk.Text(side, width=width, height=height, font=("Consolas", 9), relief="flat",
                bg=app.root.cget("bg"), wrap="word")
    t.pack(fill="x")
    t.insert("1.0", "未読み込み")
    t.configure(state="disabled")

    def put(s):
        t.configure(state="normal")
        t.delete("1.0", "end")
        t.insert("1.0", s)
        t.configure(state="disabled", height=max(2, min(16, s.count("\n") + 1 + len(s) // 60)))
    return put


class PlotArea:
    """matplotlib のグラフ領域（ツールバーで拡大・移動・画像保存ができる）"""

    def __init__(self, app, parent):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        matplotlib.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "xtick.direction": "in",
                                    "ytick.direction": "in", "legend.frameon": False,
                                    "font.family": "sans-serif",
                                    "font.sans-serif": ["Arial", "Yu Gothic", "Meiryo", "MS Gothic",
                                                        "Noto Sans CJK JP", "DejaVu Sans"],
                                    "mathtext.fontset": "dejavusans", "savefig.dpi": 300})
        # constrained レイアウト: ウィンドウの大きさが変わるたびに余白を取り直す
        self.fig = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        tb = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def message(self, text):
        self.fig.clear()
        self.fig.text(0.5, 0.5, text, ha="center", va="center", color="#666")
        self.canvas.draw_idle()

    def draw(self):
        self.canvas.draw_idle()


# フェルミ面タブの「斜め」の向き: b1 を手前、b2 を右、b3 を上にして、少し右・少し上から見る
FS_OBLIQUE = rot_x(np.radians(1.5)) @ rot_y(np.radians(-21.0))


def view_toolbar(app, bar, viewer, items, home, save_cmd, oblique=None, extra=()):
    """描画領域の上に横 1 列で並べる視点ボタン（VESTA と同じ配置）:
    a, b, c, a*, b*, c*（または b1, b2, b3）, 斜め, 全体表示, 初期位置, [extra = [(名前, 関数)]], 画像を保存"""
    if oblique is None:
        oblique = rot_x(np.radians(-20)) @ rot_y(np.radians(25))
    ttk = app.ttk
    for name, vdir, up in items:
        ttk.Button(bar, text=name, width=3,
                   command=lambda d=vdir, u=up: viewer.set_view(view_matrix(d(), u()))).pack(side="left", padx=1)
    ttk.Button(bar, text="斜め", width=5, command=lambda: viewer.set_view(oblique @ home())).pack(
        side="left", padx=(6, 1))
    ttk.Button(bar, text="全体表示", command=lambda: viewer.fit(True)).pack(side="left", padx=1)
    ttk.Button(bar, text="初期位置", command=lambda: (viewer.view.update(R=home()), viewer.fit(True))).pack(
        side="left", padx=1)
    for k, (name, cmd) in enumerate(extra):
        ttk.Button(bar, text=name, command=cmd).pack(side="left", padx=(6 if k == 0 else 1, 1))
    ttk.Button(bar, text="画像を保存…", command=save_cmd).pack(side="left", padx=(6 if not extra else 1, 1))


# ----------------------------------------------------------------------
#  タブ 1: 電荷密度
# ----------------------------------------------------------------------
# ---- 表示の状態（視点・レベル・チェックボックスなど）の保存と復元 ----
def capture_vars(obj, names=None, skip=()):
    """obj の tk 変数（と、tk 変数の辞書・リスト）の値をまとめて取り出す"""
    import tkinter as tk
    out = {}
    for k, v in list(vars(obj).items()):
        if k in skip or (names is not None and k not in names):
            continue
        if isinstance(v, tk.Variable):
            try:
                out[k] = v.get()
            except Exception:
                pass
        elif isinstance(v, dict) and v and all(isinstance(x, tk.Variable) for x in v.values()):
            out[k] = {"#d": {str(kk): x.get() for kk, x in v.items()}}
        elif isinstance(v, list) and v and all(isinstance(x, tk.Variable) for x in v):
            out[k] = {"#l": [x.get() for x in v]}
    return out


def restore_vars(obj, state, only=None, skip=()):
    """capture_vars で取り出した値を戻す（いまある変数だけ。コマンドは呼ばない）"""
    import tkinter as tk
    for k, val in (state or {}).items():
        if k in skip or (only is not None and k not in only):
            continue
        cur = getattr(obj, k, None)
        try:
            if isinstance(val, dict) and "#d" in val and isinstance(cur, dict):
                for kk, x in cur.items():
                    if str(kk) in val["#d"]:
                        x.set(val["#d"][str(kk)])
            elif isinstance(val, dict) and "#l" in val and isinstance(cur, list):
                for x, y in zip(cur, val["#l"]):
                    x.set(y)
            elif isinstance(cur, tk.Variable):
                cur.set(val)
        except Exception:
            pass


def viewer_state(viewer):
    v = viewer.view
    return {"R": np.asarray(v["R"], float).tolist(), "zoom": float(v["zoom"]), "pan": [float(x) for x in v["pan"]],
            "auto": bool(v["auto"])}


def apply_viewer_state(viewer, st):
    if not st or viewer.scene is None:
        return
    v = viewer.view
    v["R"] = np.asarray(st["R"], float)
    v["zoom"], v["pan"], v["auto"] = float(st["zoom"]), list(st["pan"]), bool(st["auto"])
    viewer.fit(False)


class ViewMemory:
    """データごとの表示の状態（キー → 状態）をアプリの保存フォルダの views.json に覚えておく"""
    MAX = 300

    def __init__(self, store):
        self.path = os.path.join(store.root, "views.json") if store is not None else None
        self.data = {}
        if self.path and os.path.isfile(self.path):
            try:
                import json
                with open(self.path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key):
        return self.data.get(key, {}).get("state") if key else None

    def put(self, key, state):
        import time as _t
        if not key or state is None:
            return
        self.data[key] = {"state": state, "time": _t.time()}
        if len(self.data) > self.MAX:
            for k in sorted(self.data, key=lambda k: self.data[k].get("time", 0))[:len(self.data) - self.MAX]:
                self.data.pop(k, None)

    def save(self):
        if not self.path:
            return
        import json
        try:
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, default=float)
            os.replace(self.path + ".tmp", self.path)
        except Exception:
            pass


def remember_view(app, tab):
    """タブで今表示しているデータの表示状態を覚える"""
    key = getattr(tab, "view_key", None)
    if not key or getattr(app, "views", None) is None:
        return
    try:
        app.views.put(key, tab.view_state())
    except Exception:
        pass


def restore_view(app, tab, key, embedded=None):
    """前に覚えた表示状態があれば戻す（ライブラリに入っている状態は、覚えたものがなければ使う）"""
    tab.view_key = key
    if not getattr(app, "restore_var", None) or not app.restore_var.get():
        return False
    st = app.views.get(key) if getattr(app, "views", None) is not None else None
    st = st or embedded
    if not st:
        return False
    try:
        tab.apply_view_state(st)
        app.add_msg("（前回の表示の状態を復元しました）")
        return True
    except Exception:
        return False


def cached_read(app, kind, paths, params, fn):
    """キャッシュにあればそれを、なければ fn() で読んでキャッシュに入れる（書き込みは裏で行う）。
    戻り値 (中身, キャッシュから読んだか)"""
    store = getattr(app, "store", None)
    if store is not None:
        obj = store.cache_get(kind, paths, params)
        if obj is not None:
            return obj, True
    obj = fn()
    if store is not None:
        snap = dict(obj) if isinstance(obj, dict) else obj      # 後で書き換えられても困らないよう浅いコピー
        store.cache_put(kind, paths, params, snap)
    return obj, False


WV_ALL_ELEM = "全元素"


class ChargeTab:
    title = "電荷密度"

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.d = None                      # 読み込んだデータ
        side, holder = make_side_layout(app, frame, on_toggle=lambda: self.viewer.fit(self.viewer.view["auto"]))
        self.side = side
        # 描画領域の上: 1 行目 = 視点・画像の保存、2 行目 = 等値面のレベル（VESTA と同じ配置）
        bar1 = ttk.Frame(holder, padding=(4, 3, 4, 0))
        bar1.pack(side="top", fill="x")
        bar2 = ttk.Frame(holder, padding=(4, 3, 4, 3))
        bar2.pack(side="top", fill="x")
        self.viewer = Viewer3D(tk, holder, on_status=self._perf)
        self.viewer.on_mode_change = lambda: self.rebuild() if self.d is not None else None
        L = lambda i: (lambda: self.d["st"].lattice[i])
        Rc = lambda i: (lambda: np.linalg.inv(self.d["st"].lattice).T[i])
        view_toolbar(app, bar1, self.viewer,
                     [("a", L(0), Rc(2)), ("b", L(1), Rc(2)), ("c", L(2), Rc(1)),
                      ("a*", Rc(0), L(2)), ("b*", Rc(1), L(2)), ("c*", Rc(2), L(1))], self._home, self.save_image)
        ttk.Label(bar2, text="等値面のレベル [e/Å³]:").pack(side="left")
        self.level_var = tk.StringVar(value="")
        e = ttk.Entry(bar2, textvariable=self.level_var, width=12)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self.rebuild())
        ttk.Button(bar2, text="反映", command=self.rebuild).pack(side="left")
        ttk.Button(bar2, text="既定値", command=self.reset_level).pack(side="left", padx=(4, 0))

        W = int(330 * UI_SCALE)

        # ---- よく使うもの（開く・等値面・視点）を上に、細かい設定を下に ----
        section(app, side, "① CHGCAR")
        ttk.Button(side, text="CHGCAR を開く…", command=self.choose).pack(anchor="w")
        self.put_info = info_text(app, side, height=7)

        section(app, side, "等値面")
        f2 = ttk.Frame(side)
        f2.pack(anchor="w")
        ttk.Label(f2, text="不透明度 (0–1):").grid(row=0, column=0, sticky="w")
        self.opac_var = tk.StringVar(value=str(ISO_OPACITY))
        e2 = ttk.Entry(f2, textvariable=self.opac_var, width=6)
        e2.grid(row=0, column=1, padx=4)
        e2.bind("<Return>", lambda ev: self.make_scene())
        ttk.Button(f2, text="反映", command=self.make_scene).grid(row=0, column=2)
        self.iso_lbl = ttk.Label(side, text="", foreground="#333", wraplength=W, justify="left")
        self.iso_lbl.pack(anchor="w", pady=(2, 0))

        section(app, side, "② WAVECAR から軌道分解")
        self.wv = None
        ttk.Button(side, text="計算フォルダを開く…", command=self.wv_choose).pack(anchor="w")
        self.wv_info = ttk.Label(side, text="", foreground="#333", wraplength=W, justify="left")
        self.wv_info.pack(anchor="w")
        fwb = ttk.Frame(side)
        fwb.pack(anchor="w", pady=(2, 0))
        self.wv_mode = tk.StringVar(value="band")
        self.wv_bands = tk.StringVar(value="")
        self.wv_emin = tk.StringVar(value="-0.1")
        self.wv_emax = tk.StringVar(value="0.1")
        ttk.Radiobutton(fwb, text="バンド番号:", value="band", variable=self.wv_mode).grid(row=0, column=0, sticky="w")
        ttk.Entry(fwb, textvariable=self.wv_bands, width=16).grid(row=0, column=1, columnspan=3, sticky="w", padx=2)
        ttk.Radiobutton(fwb, text="E − E_F [eV]:", value="energy", variable=self.wv_mode).grid(row=1, column=0, sticky="w")
        ttk.Entry(fwb, textvariable=self.wv_emin, width=6).grid(row=1, column=1, padx=2)
        ttk.Label(fwb, text="〜").grid(row=1, column=2)
        ttk.Entry(fwb, textvariable=self.wv_emax, width=6).grid(row=1, column=3, padx=2)
        ttk.Radiobutton(fwb, text="バンド番号 かつ E − E_F", value="both",
                        variable=self.wv_mode).grid(row=2, column=0, columnspan=4, sticky="w")
        fwr = ttk.Frame(side)
        fwr.pack(anchor="w", pady=(2, 0))
        ttk.Label(fwr, text="球の半径の比:").pack(side="left")
        self.wv_rfac = tk.StringVar(value="0.5")
        ttk.Entry(fwr, textvariable=self.wv_rfac, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(fwr, text="k点の間引き:").pack(side="left")
        self.wv_thin = tk.StringVar(value="なし")
        self.wv_thin_box = ttk.Combobox(fwr, textvariable=self.wv_thin, width=6, state="readonly", values=["なし"])
        self.wv_thin_box.pack(side="left", padx=2)
        self.wv_thin_box.bind("<<ComboboxSelected>>", lambda ev: self._wv_thin_text())
        self.wv_thin_lbl = ttk.Label(side, text="", foreground="#666", wraplength=W, justify="left")
        self.wv_thin_lbl.pack(anchor="w")
        fwc = ttk.Frame(side)
        fwc.pack(anchor="w", pady=(2, 0))
        self.wv_btn = ttk.Button(fwc, text="計算", command=self.wv_compute)
        self.wv_btn.pack(side="left")
        ttk.Button(fwc, text="中止", command=self.wv_cancel).pack(side="left", padx=4)
        self.wv_prog = ttk.Label(side, text="", foreground="#333", wraplength=W, justify="left")
        self.wv_prog.pack(anchor="w")
        self.wv_sel = ttk.Frame(side)
        self.wv_sel.pack(anchor="w", fill="x")

        section(app, side, "表示")
        self.atom_var = tk.BooleanVar(value=True)
        self.bond_var = tk.BooleanVar(value=True)
        self.cell_var = tk.BooleanVar(value=True)
        self.axes_var = tk.BooleanVar(value=True)
        fdv = ttk.Frame(side)
        fdv.pack(anchor="w")
        for k, (text, var) in enumerate((("原子", self.atom_var), ("結合", self.bond_var), ("単位胞", self.cell_var),
                                         ("結晶軸", self.axes_var))):
            ttk.Checkbutton(fdv, text=text, variable=var, command=self.make_scene).grid(row=0, column=k, sticky="w",
                                                                                         padx=(0, 8))

        section(app, side, "スーパーセル")
        fs = ttk.Frame(side)
        fs.pack(anchor="w")
        self.nvars = [tk.StringVar(value="1") for _ in range(3)]
        for lab, v in zip("abc", self.nvars):
            ttk.Label(fs, text=lab).pack(side="left")
            en = ttk.Entry(fs, textvariable=v, width=3)
            en.pack(side="left", padx=(1, 5))
            en.bind("<Return>", lambda ev: self.apply_supercell())
        ttk.Button(fs, text="更新", command=self.apply_supercell).pack(side="left")
        self.cell_lbl = ttk.Label(side, text="", foreground="#333", wraplength=W, justify="left")
        self.cell_lbl.pack(anchor="w", pady=(2, 0))

        section(app, side, "ラベル")
        self.lab_frame = ttk.Frame(side)
        self.lab_frame.pack(anchor="w")
        self.lab_vars = {}
        fla = ttk.Frame(side)
        fla.pack(anchor="w")
        ttk.Button(fla, text="全部", command=lambda: self._labels_all(True)).pack(side="left")
        ttk.Button(fla, text="なし", command=lambda: self._labels_all(False)).pack(side="left", padx=4)
        ttk.Label(fla, text="文字 [pt]:").pack(side="left", padx=(8, 2))
        self.lsize_var = tk.StringVar(value=str(LABEL_SIZE))
        en = ttk.Entry(fla, textvariable=self.lsize_var, width=5)
        en.pack(side="left")
        en.bind("<Return>", lambda ev: self.make_scene())
        ttk.Button(fla, text="反映", command=self.make_scene).pack(side="left", padx=4)

        section(app, side, "結合（元素対ごと）")
        self.bond_frame = ttk.Frame(side)
        self.bond_frame.pack(anchor="w")
        self.rule_vars = {}
        fbb = ttk.Frame(side)
        fbb.pack(anchor="w", pady=(4, 0))
        ttk.Button(fbb, text="結合を再計算", command=self._rules_changed).pack(side="left")
        ttk.Button(fbb, text="既定値に戻す", command=self._rules_default).pack(side="left", padx=6)

        section(app, side, "断面")
        self.sec_var = tk.BooleanVar(value=True)
        fsc = ttk.Frame(side)
        fsc.pack(anchor="w")
        ttk.Checkbutton(fsc, text="表示", variable=self.sec_var, command=self.make_scene).grid(row=0, column=0, sticky="w")
        ttk.Label(fsc, text="色:").grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.sec_mode = tk.StringVar(value=SECTION_MODES[0])
        cbm = ttk.Combobox(fsc, textvariable=self.sec_mode, values=list(SECTION_MODES), width=14, state="readonly")
        cbm.grid(row=0, column=2, columnspan=2, sticky="w", padx=4)
        cbm.bind("<<ComboboxSelected>>", lambda ev: self.make_scene())
        self.sec_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(fsc, text="色の範囲を自動", variable=self.sec_auto, command=self.make_scene).grid(
            row=1, column=0, columnspan=4, sticky="w")
        ttk.Label(fsc, text="最小 / 最大:").grid(row=2, column=0, sticky="w")
        self.sec_min = tk.StringVar(value="")
        self.sec_max = tk.StringVar(value="")
        for col, var in ((1, self.sec_min), (2, self.sec_max)):
            en = ttk.Entry(fsc, textvariable=var, width=9)
            en.grid(row=2, column=col, padx=2)
            en.bind("<Return>", lambda ev: (self.sec_auto.set(False), self.make_scene()))
        ttk.Button(fsc, text="反映", command=lambda: (self.sec_auto.set(False), self.make_scene())).grid(row=2, column=3)

        section(app, side, "描画方式")
        fdm = ttk.Frame(side)
        fdm.pack(anchor="w")
        self.draw_var = tk.StringVar(value="ray")
        ttk.Radiobutton(fdm, text="GPU レイキャスティング", value="ray",
                        variable=self.draw_var, command=self.rebuild).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(fdm, text="三角形", value="tri",
                        variable=self.draw_var, command=self.rebuild).grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.res_var = tk.StringVar(value="auto")
        f3 = ttk.Frame(side)
        f3.pack(anchor="w", pady=(2, 0))
        ttk.Label(f3, text="三角形の解像度:").pack(side="left")
        ttk.Radiobutton(f3, text=f"自動（1 辺 {CHG_AUTO_MAXGRID} 点以下）", value="auto", variable=self.res_var,
                        command=self.rebuild).pack(side="left")
        ttk.Radiobutton(f3, text="全点", value="full", variable=self.res_var,
                        command=self.rebuild).pack(side="left", padx=4)
        self.perf = ttk.Label(side, text="", foreground="#666", wraplength=W, justify="left")
        self.perf.pack(anchor="w", pady=(4, 0))

    def _home(self):
        st = self.d["st"]
        return view_matrix(st.lattice[0], np.linalg.inv(st.lattice).T[2])

    def _perf(self):
        if not hasattr(self, "perf") or not hasattr(self, "viewer"):
            return
        self.perf.configure(text=f"描画: {self.viewer.engine_text()}\n{self.viewer.fps_text()}")

    def choose(self):
        p = self.app.filedialog.askopenfilename(title="CHGCAR を選択", filetypes=[("All files", "*.*")])
        if p:
            self.load(p)

    def load(self, path):
        app = self.app
        if getattr(self, "_chg_job", None) is not None and not self._chg_job["done"]:
            app.messagebox.showinfo("CHGCAR", "前の CHGCAR を読み込み中です。終わってから開いてください。")
            return

        def work(progress):
            return cached_read(app, "chgcar", [path], None,
                               lambda: dict(zip(("st", "rho", "info"), read_chgcar(path))))

        def fail(e):
            app.messagebox.showerror("CHGCAR 読み込みエラー", f"{path}\n\n{e}")
            app.set_msg("")
        self._chg_job = run_in_background(app, f"CHGCAR を読み込み中… {path}", work,
                                          lambda r: self._load_done(path, *r), fail)

    def _load_done(self, path, c, hit):
        app = self.app
        remember_view(app, self)
        st, rho, info = c["st"], c["rho"], c["info"]
        self._install(st, rho, info, path, head=f"ファイル: {os.path.basename(path)}",
                      msg=f"電荷密度を読み込みました: {path}" + ("（前回読んだ内容をキャッシュから読みました）" if hit else "") + "\n"
                          "等値面レベルは描画領域の上の入力欄で変更できます。値は ρ(r) [e/Å³]（CHGCAR の値 ÷ セル体積）。"
                          "セル（スーパーセル）の境界での切り口は、値に応じた色の断面として描きます（VESTA と同じ）。")
        app.add_recent("chgcar", path)
        restore_view(app, self, "chg:" + os.path.abspath(path))

    # ---- 表示の状態 ----
    STYLE = ("opac_var", "res_var", "atom_var", "bond_var", "cell_var", "axes_var", "lsize_var", "wv_rfac",
             "sec_var", "sec_mode", "draw_var")

    def view_state(self):
        if self.d is None:
            return None
        st = {"vars": capture_vars(self, skip=("wv_mode", "wv_bands", "wv_emin", "wv_emax", "wv_orb", "wv_el",
                                               "wv_m", "wv_target", "rule_vars")),
              "view": viewer_state(self.viewer)}
        try:
            st["rules"] = [[a, b, bool(r[0]), float(r[1])] for (a, b), r in self._current_rules().items()]
        except Exception:
            pass
        return st

    def apply_view_state(self, stt):
        d = self.d
        if d is None or not stt:
            return
        restore_vars(self, stt.get("vars"), skip=("nvars",))
        if stt.get("rules"):
            for a, b, on, dmax in stt["rules"]:
                if (a, b) in self.rule_vars:
                    self.rule_vars[(a, b)][0].set(on)
                    self.rule_vars[(a, b)][1].set(f"{dmax:.3f}")
            d["rules"] = self._current_rules()
        restore_vars(self, stt.get("vars"), only=("nvars",))
        n = self._read_n() or [1, 1, 1]
        self._build_model(n)
        self.rebuild()
        apply_viewer_state(self.viewer, stt.get("view"))

    def _install(self, st, rho, info, path, head, msg, keep_view=False, keep_level=False, default=None, orb=None):
        """読み込んだ（または計算した）電荷密度を表示する"""
        app = self.app
        old_level = self.level_var.get()
        dmin = shortest_pair_distances(st)
        rules = default_bond_rules(st.species, dmin)
        old_rules = self._current_rules() if self.d is not None else {}
        for key in rules:                     # 同じ元素対の設定は、読み込み直しても引き継ぐ
            if key in old_rules:
                rules[key] = list(old_rules[key])
        self.d = {"st": st, "rho": rho, "info": info, "path": path, "rules": rules, "D": None, "Dn": None,
                  "n": [1, 1, 1], "orb": orb}
        self._build_model(self._read_n() or [1, 1, 1])
        tk, ttk = app.tk, app.ttk
        old = {k: v.get() for k, v in self.lab_vars.items()}
        for w in self.lab_frame.winfo_children():
            w.destroy()
        self.lab_vars = {}
        for k, el in enumerate(st.species):
            v = tk.BooleanVar(value=old.get(el, False))
            self.lab_vars[el] = v
            tk.Label(self.lab_frame, text="●", fg=_hex(element_style(el)[1]), font=("", 13)).grid(
                row=k // 4, column=2 * (k % 4))
            ttk.Checkbutton(self.lab_frame, text=el, variable=v, command=self.make_scene).grid(
                row=k // 4, column=2 * (k % 4) + 1, sticky="w", padx=(0, 8))
        self._build_bond_rows(dmin)
        self.side.fit_width()
        self.d["default"] = default_iso_level(rho) if default is None else float(default)
        la, ang, vol = lattice_params(st)
        self.put_info(f"{head}\n"
                      f"組成  : {st.formula}  ({st.natoms} atoms)\n"
                      f"a,b,c : {la[0]:.4f} {la[1]:.4f} {la[2]:.4f} Å\n"
                      f"α,β,γ : {ang[0]:.2f} {ang[1]:.2f} {ang[2]:.2f}°\n"
                      f"格子点: {info['grid'][0]} × {info['grid'][1]} × {info['grid'][2]}\n"
                      f"電子数: {info['nelect']:.3f}（格子点の平均 × 体積）\n"
                      f"ρ 最小/最大: {info['min']:.4g} / {info['max']:.4g} e/Å³")
        if not keep_view:
            self.viewer.view["R"] = self._home()
            self.viewer.scene = None
        app.select_tab(self)
        app.root.update_idletasks()
        if keep_level and old_level.strip():
            self.level_var.set(old_level)
            self.rebuild()
        else:
            self.reset_level()
        app.set_msg(msg)
        app.show_structure(st, path, "電荷密度タブで開いたデータ")

    # ---- ② WAVECAR から軌道分解 ----
    def wv_choose(self):
        # Windows のフォルダ選択画面はファイルを表示しないので、フォルダ内のファイルを選んでもらい、その場所を開く
        p = self.app.filedialog.askopenfilename(
            title="計算フォルダの WAVECAR（またはフォルダ内のどれか 1 つのファイル）を選択",
            filetypes=[("WAVECAR", ("WAVECAR", "WAVECAR*")),
                       ("VASP の出力", ("WAVECAR", "EIGENVAL", "POSCAR", "CONTCAR", "OUTCAR", "DOSCAR", "KPOINTS",
                                       "INCAR", "PROCAR", "vasprun.xml")),
                       ("All files", "*")])
        if p:
            self.wv_open(os.path.dirname(p))

    def wv_open(self, folder):
        """WAVECAR のあるフォルダを確認する（WAVECAR の k点ごとの情報・EIGENVAL・E_F・k点の間引きの候補）。
        数十 GB の WAVECAR でも画面が止まらないよう、裏のスレッドで読む"""
        app = self.app
        if getattr(self, "_wv_open_job", None) is not None and not self._wv_open_job["done"]:
            app.messagebox.showinfo("WAVECAR", "前の WAVECAR を確認中です。終わってから開いてください。")
            return
        self.wv_info.configure(text=f"確認中… {folder}", foreground="#333")

        def fail(e):
            self.wv_info.configure(text="")
            app.messagebox.showerror("WAVECAR 読み込みエラー", f"{folder}\n\n{e}")
            app.set_msg("")
        self._wv_open_job = run_in_background(app, f"WAVECAR を確認中… {folder}",
                                              lambda progress: self._wv_probe(folder, progress),
                                              lambda r: self._wv_open_done(folder, r), fail)

    def _wv_probe(self, folder, progress):
        """（裏のスレッド）フォルダのファイルを読んで確認する。画面には触らない"""
        import time as _t
        tm = {}

        def need(names, what):
            for n in names:
                q = os.path.join(folder, n)
                if os.path.isfile(q):
                    return q
            raise FileNotFoundError(f"{what} が見つかりません（{' / '.join(names)}）。")
        t0 = _t.time()
        wc = WavecarFile(need(["WAVECAR"], "WAVECAR"),
                         progress=lambda i, n: progress(f"WAVECAR の k点の情報 {i} / {n}"))
        tm["WAVECAR"] = _t.time() - t0
        progress("POSCAR・EIGENVAL・E_F")
        t0 = _t.time()
        pos = need(["POSCAR", "CONTCAR"], "構造ファイル")
        st = parse_poscar(_read_text(pos))
        if not np.allclose(np.asarray(st.lattice, float), wc.lattice, atol=1e-3):
            raise ValueError(f"{os.path.basename(pos)} の格子が WAVECAR と一致しません（計算に使った構造を置いてください）。")
        ev = read_eigenval(need(["EIGENVAL"], "EIGENVAL（k点の重みに使います）"))
        if len(ev["kpts"]) != wc.nk:
            raise ValueError(f"EIGENVAL の k点数 ({len(ev['kpts'])}) が WAVECAR ({wc.nk}) と一致しません。")
        dk = ev["kpts"] - wc.kpts
        if np.abs(dk - np.rint(dk)).max() > 1e-4:
            raise ValueError("EIGENVAL の k点が WAVECAR と一致しません。")
        tm["POSCAR・EIGENVAL"] = _t.time() - t0
        t0 = _t.time()
        kw = np.asarray(ev["weights"], float)
        kw = kw / kw.sum()
        ef, src = wc.efermi, "WAVECAR"
        for name, fn in (("DOSCAR", read_doscar_efermi), ("OUTCAR", read_outcar_efermi)):
            q = os.path.join(folder, name)
            if os.path.isfile(q):
                try:
                    v = fn(q)
                    if v is not None:
                        ef, src = float(v), name
                        break
                except Exception:
                    pass
        tm["E_F"] = _t.time() - t0
        # 元素ごとの軌道の種類（POTCAR の価電子配置と PAW の射影子）。POTCAR がなければ s〜f を全部
        lsets = None
        q = os.path.join(folder, "POTCAR")
        if os.path.isfile(q):
            try:
                ent = read_potcar_lchannels(q)
                lsets = element_lchannels(st.species, ent)
                if lsets:
                    lsets["#conf"] = element_valence_text(st.species, ent)
            except Exception:
                lsets = None
        wv = {"folder": folder, "wc": wc, "st": st, "kw": kw, "ef": ef, "src": src, "res": None, "lsets": lsets,
              "struct": os.path.basename(pos), "pos": pos,
              "files": [os.path.join(folder, "WAVECAR"), pos, os.path.join(folder, "EIGENVAL"),
                        os.path.join(folder, "KPOINTS")], "thin": {}, "hist": {}, "times": tm}
        # k点の間引きの候補（KPOINTS に Γ中心のメッシュが書かれていて、WAVECAR の k点がその既約 k点のとき）
        progress("k点の間引きの候補")
        t0 = _t.time()
        thin = {}
        try:
            kf = read_kpoints_file(os.path.join(folder, "KPOINTS"))
            if kf.get("mesh") and kf.get("gamma", True):
                mesh = tuple(int(x) for x in kf["mesh"])
                for f in (2, 3, 4):
                    if any(n % f == 0 and n // f >= 1 for n in mesh) and f <= max(mesh):
                        w, cm, nused = thin_kweights(st, wc.kpts, mesh, f)
                        if cm != mesh:
                            thin[f] = (w, cm, nused)
                wv["mesh"] = mesh
        except Exception:
            thin = {}
        wv["thin"] = thin
        tm["間引きの候補"] = _t.time() - t0
        return wv

    def _wv_open_done(self, folder, wv):
        """（画面のスレッド）確認の結果を画面に出す"""
        app = self.app
        self.wv = wv
        wc, st, ef, src, pos = wv["wc"], wv["st"], wv["ef"], wv["src"], wv["pos"]
        thin = wv["thin"]
        self.wv_thin_box.configure(values=["なし"] + [f"{f} 倍" for f in sorted(thin)])
        self.wv_thin.set("なし")
        self._wv_thin_text()
        emin = wc.eig.min(axis=(0, 1)) - ef
        emax = wc.eig.max(axis=(0, 1)) - ef
        cross = [n + 1 for n in range(wc.nb) if emin[n] < 0 < emax[n]]
        near = [n + 1 for n in range(wc.nb) if emax[n] > -0.5 and emin[n] < 0.5]
        kind = {"std": "vasp_std", "gam": "vasp_gam", "ncl": "vasp_ncl（SOC）"}[wc.vtype]
        self.wv_info.configure(
            text=f"フォルダ: {os.path.basename(os.path.normpath(folder))}（{kind}）\n"
                 f"k点 {wc.nk}（既約）  バンド {wc.nb}  スピン {wc.nspin}  ENCUT {wc.encut:g} eV\n"
                 f"E_F = {ef:.4f} eV（{src}）\n"
                 f"E_F を横切るバンド: {', '.join(map(str, cross)) or 'なし'}\n"
                 f"E_F ± 0.5 eV にかかるバンド: {', '.join(map(str, near[:20])) or 'なし'}"
                 + (" …" if len(near) > 20 else ""), foreground="#333")
        if not self.wv_bands.get().strip() and cross:
            self.wv_bands.set(" ".join(map(str, cross)))
        for w in self.wv_sel.winfo_children():
            w.destroy()
        self.wv_prog.configure(text="")
        self.side.fit_width()
        app.add_recent("wavecar", folder)
        app.show_structure(st, pos, "電荷密度タブ（WAVECAR）のフォルダ")
        tt = "、".join(f"{k} {v:.1f} 秒" for k, v in wv["times"].items())
        app.set_msg(f"WAVECAR を確認しました: {folder}\nバンドを指定して「計算」を押してください。\n"
                    f"（確認にかかった時間: {tt}）")

    def _wv_thin_factor(self):
        t = self.wv_thin.get()
        return 1 if t == "なし" else int(t.split()[0])

    def _wv_thin_text(self):
        wv = self.wv
        if wv is None:
            return
        f = self._wv_thin_factor()
        n_all = int((np.asarray(wv["kw"]) > 0).sum())
        if f == 1:
            txt = "" if wv["thin"] else "（KPOINTS が Γ中心のメッシュでないため、間引きは使えません）"
        else:
            w, cm, nused = wv["thin"][f]
            txt = (f"メッシュ {wv['mesh'][0]}×{wv['mesh'][1]}×{wv['mesh'][2]} → {cm[0]}×{cm[1]}×{cm[2]}"
                   f"（k点 {n_all} → {nused} 個、近似）")
        self.wv_thin_lbl.configure(text=txt)

    def _wv_selection(self):
        """指定 → {(スピン, k点): [バンド番号 (0 始まり)]} と説明"""
        wv = self.wv
        wc = wv["wc"]
        sel = {}
        mode = self.wv_mode.get()
        nums, win = None, None
        if mode in ("band", "both"):
            nums = []
            try:
                for tok in self.wv_bands.get().replace(",", " ").replace("–", "-").replace("〜", "-").split():
                    if "-" in tok[1:]:
                        a, b = tok.split("-", 1)
                        if int(b) < int(a):
                            raise ValueError
                        nums += list(range(int(a), int(b) + 1))
                    else:
                        nums.append(int(tok))
            except ValueError:
                raise ValueError("バンド番号は整数で指定してください（例: 241 242 / 241-244）。")
            nums = sorted(set(nums))
            if not nums or min(nums) < 1 or max(nums) > wc.nb:
                raise ValueError(f"バンド番号は 1〜{wc.nb} で指定してください。")
            bdesc = (f"{nums[0]}–{nums[-1]}" if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1))
                     else " ".join(map(str, nums)))
        if mode in ("energy", "both"):
            try:
                e0 = float(self.wv_emin.get())
                e1 = float(self.wv_emax.get())
            except ValueError:
                raise ValueError("エネルギー範囲は数値で指定してください（例: -0.1 〜 0.1）。")
            if e1 <= e0:
                raise ValueError("エネルギー範囲は 下限 < 上限 で指定してください。")
            win = (e0, e1)
        for s in range(wc.nspin):
            for k in range(wc.nk):
                ok = np.ones(wc.nb, bool)
                if nums is not None:
                    ok &= np.isin(np.arange(1, wc.nb + 1), nums)
                if win is not None:
                    de = wc.eig[s, k] - wv["ef"]
                    ok &= (de >= win[0]) & (de <= win[1])
                sel[(s, k)] = [int(n) for n in np.where(ok)[0]]
        if not any(sel.values()):
            raise ValueError("この範囲に入る固有値がありません。" if nums is None else
                             "指定したバンドに、この E − E_F の範囲に入る固有値がありません。")
        parts = []
        if nums is not None:
            parts.append("バンド " + bdesc)
        if win is not None:
            parts.append(f"E − E_F = {win[0]:g}〜{win[1]:g} eV")
        desc = "、".join(parts)
        if nums is not None and win is not None:
            nst = sum(len(v) for v in sel.values())
            desc += f"（{nst} / {len(nums) * wc.nk * wc.nspin} 状態）"
        return sel, desc

    def wv_compute(self):
        import threading
        app = self.app
        wv = self.wv
        if wv is None:
            app.messagebox.showinfo("WAVECAR", "先に「計算フォルダを開く…」で WAVECAR のあるフォルダを選んでください。")
            return
        if wv.get("wc") is None:
            app.messagebox.showinfo("WAVECAR", "ライブラリから開いた結果です。軌道・元素・原子の切り替えはできますが、"
                                    "別のバンドで計算し直すには「計算フォルダを開く…」で元の WAVECAR のあるフォルダを開いてください。")
            return
        if wv.get("job") is not None and not wv["job"]["done"]:
            return
        try:
            sel, desc = self._wv_selection()
            fac = float(self.wv_rfac.get())
            if not 0.05 <= fac <= 0.5:
                raise ValueError("球の半径の比は 0.05〜0.5 にしてください（0.5 を超えると隣の原子の球と重なります）。")
        except ValueError as e:
            app.messagebox.showwarning("WAVECAR", str(e))
            return
        radii = default_sphere_radii(wv["st"], fac)
        thin = self._wv_thin_factor()
        kw = np.asarray(wv["kw"]) if thin == 1 else wv["thin"][thin][0]
        tdesc = "" if thin == 1 else f"（k点 {thin} 倍間引き）"
        job = {"n": 0, "tot": 0, "done": False, "err": None, "res": None, "cancel": False, "desc": desc + tdesc,
               "radii": radii, "thin": thin, "rfac": fac, "bdesc": desc, "cached": False, "t0": 0.0}
        wv["job"] = job
        store = getattr(app, "store", None)
        # v: 2 から、m ごとの表示に使う密度行列（対称化済み）も結果に入れる
        ckey = ("wave_orb", wv["files"], {"sel": desc, "rfac": round(fac, 6), "thin": thin, "lmax": 3, "v": 3})

        def work():
            import time as _t
            job["t0"] = _t.time()
            try:
                if store is not None:
                    hit = store.cache_get(*ckey)
                    if hit is not None:
                        job["res"], job["cached"] = hit, True
                        return
                job["res"] = orbital_band_density(
                    wv["wc"], wv["st"], sel, kw, radii,
                    progress=lambda n, t: job.update(n=n, tot=t), cancel=lambda: job["cancel"])
                if store is not None:
                    store.cache_put(*ckey, dict(job["res"]), background=False)
            except Exception as e:
                job["err"] = e
            finally:
                job["t1"] = _t.time()
                job["done"] = True

        self.wv_btn.state(["disabled"])
        threading.Thread(target=work, daemon=True).start()
        app.root.after(300, self._wv_poll)

    def wv_cancel(self):
        if self.wv and self.wv.get("job") is not None:
            self.wv["job"]["cancel"] = True

    def _wv_poll(self):
        wv = self.wv
        job = wv["job"]
        if not job["done"]:
            import time as _t
            msg = "計算中…"
            if job["tot"]:
                msg = f"計算中… バンド × k点 {job['n']} / {job['tot']}"
                el = _t.time() - job["t0"]
                if job["n"] >= 3 and el > 2:
                    rest = el / job["n"] * (job["tot"] - job["n"])
                    msg += f"（残り 約 {rest / 60:.1f} 分）" if rest > 90 else f"（残り 約 {rest:.0f} 秒）"
            self.wv_prog.configure(text=msg + "\n裏で計算中。ほかの操作もできます")
            self.app.root.after(300, self._wv_poll)
            return
        self.wv_btn.state(["!disabled"])
        if job["err"] is not None:
            self.wv_prog.configure(text="")
            self.app.messagebox.showerror("WAVECAR", str(job["err"]))
            return
        res = job["res"]
        res["desc"] = job["desc"]
        wv["res"] = res
        names = "spdf"
        tot_l = sum(res["electrons"].values())
        # 間引きの違いの比較（同じバンド・同じ球の大きさで計算したもの）
        h = wv["hist"].setdefault((job["bdesc"], round(job["rfac"], 6)), {})
        h[job["thin"]] = dict(res["electrons"])
        comp = ""
        if len(h) >= 2:
            rows = []
            for f in sorted(h):
                e = h[f]
                rows.append(("なし" if f == 1 else f"{f} 倍") + ": " + " ".join(f"{names[l]} {e[l]:.4f}" for l in sorted(e)))
            comp = "\n間引きの比較（球の中の電子数）\n  " + "\n  ".join(rows)
            if 1 in h:
                base = h[1]
                worst = max(abs(h[f][l] - base[l]) / max(sum(base.values()), 1e-12)
                            for f in h if f != 1 for l in base)
                comp += f"\n  間引きなしとの差: 最大 {100 * worst:.2f} %（全電子数に対して）"
        took = "キャッシュから" if job["cached"] else f"計算時間 {job.get('t1', 0) - job['t0']:.0f} 秒"
        self.wv_prog.configure(text=(
            f"{job['desc']}: 球の中の電子数 " + "  ".join(f"{names[l]} {v:.4f}" for l, v in res["electrons"].items())
            + f"（計 {tot_l:.4f}、セル全体 {res['total']:.4f}）\n" + orbital_m_breakdown_text(res)
            + took + comp))
        self._wv_build_selector()
        self._wv_apply(first=True)

    def _wv_build_selector(self):
        tk, ttk = self.app.tk, self.app.ttk
        for w in self.wv_sel.winfo_children():
            w.destroy()
        st = self.wv["st"]
        self.wv_orb = {}
        self.wv_m = {}
        has_m = self.wv.get("res") is not None and self.wv["res"].get("M") is not None
        # 元素ごとの軌道の設定 {元素: {l: [m…]}}（最初はその元素にある軌道を全部）。
        # チェックボックスは「編集する対象」の設定を表示する
        self.wv_esub = {el: {l: list(range(2 * l + 1)) if l in self._wv_lset(el) else [] for l in range(4)}
                        for el in st.species}
        self.wv_rows = {}
        self.wv_target = tk.StringVar(value=WV_ALL_ELEM)
        ft = ttk.Frame(self.wv_sel)
        ft.pack(anchor="w", pady=(2, 0))
        ttk.Label(ft, text="軌道の設定:").pack(side="left")
        for name in [WV_ALL_ELEM] + list(st.species):
            ttk.Radiobutton(ft, text=name, value=name, variable=self.wv_target,
                            command=self._wv_target_changed).pack(side="left", padx=(4, 0))
        fo = ttk.Frame(self.wv_sel)
        fo.pack(anchor="w", pady=(2, 0))
        for l, nm in enumerate("spdf"):
            v = tk.BooleanVar(value=True)
            self.wv_orb[l] = v
            if not has_m:
                cb = ttk.Checkbutton(fo, text=nm, variable=v, command=self._wv_edited)
                cb.grid(row=0, column=l, sticky="w")
                self.wv_rows[l] = [cb]
                continue
            row = 2 * l             # f は 2 行使うので、行の番号は l ごとに 2 つずつ空ける
            cb = ttk.Checkbutton(fo, text=nm, variable=v, command=lambda l=l: self._wv_l_toggled(l))
            cb.grid(row=row, column=0, sticky="w", padx=(0, 6))
            self.wv_rows[l] = [cb]
            if l == 0:
                continue
            for k, m in enumerate(ORBITAL_M_ORDER[l]):          # 並び: px, py, pz / dxy, dyz, dzx, dz², dx²−y²
                lab = ORBITAL_M_LABELS[l][m]
                vm = tk.BooleanVar(value=True)
                self.wv_m[(l, m)] = vm
                # f は 7 成分あるので 2 行に分ける（4 + 3）
                rr_, cc_ = (row, 1 + k) if l < 3 else (row + k // 4, 1 + k % 4)
                cm = ttk.Checkbutton(fo, text=lab, variable=vm, command=lambda l=l: self._wv_m_toggled(l))
                cm.grid(row=rr_, column=cc_, sticky="w", padx=(0, 4))
                self.wv_rows[l].append(cm)
        self.wv_target_note = ttk.Label(self.wv_sel, text="", foreground="#666",
                                        wraplength=int(330 * UI_SCALE), justify="left")
        self.wv_target_note.pack(anchor="w")
        if self.wv.get("res") is not None and not has_m:
            ttk.Label(self.wv_sel, text="（この結果は以前の版で計算したので、px, py, … の成分には分けられません。"
                                        "分けるには計算し直してください）", foreground="#666",
                      wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w")
        fe = ttk.Frame(self.wv_sel)
        fe.pack(anchor="w")
        ttk.Label(fe, text="表示する元素:").grid(row=0, column=0, sticky="w")
        self.wv_el = {}
        for k, el in enumerate(st.species):
            v = tk.BooleanVar(value=True)
            self.wv_el[el] = v
            ttk.Checkbutton(fe, text=el, variable=v, command=self._wv_apply).grid(row=0, column=k + 1, sticky="w",
                                                                                 padx=(0, 6))
        self.wv_sel_lbl = ttk.Label(self.wv_sel, text="", foreground="#333", wraplength=int(330 * UI_SCALE),
                                    justify="left")
        self.wv_sel_lbl.pack(anchor="w")
        lsets = self.wv.get("lsets")
        if lsets:
            conf = lsets.get("#conf", {})
            src = "  ".join(f"{el}: {' '.join('spdf'[l] for l in lsets[el])}"
                            + (f"（{conf[el]}）" if conf.get(el) else "") for el in st.species)
            txt = f"元素ごとの軌道（POTCAR の価電子配置から）: {src}"
        else:
            txt = "POTCAR がないので、どの元素も s〜f を表示しています"
        ttk.Label(self.wv_sel, text=txt, foreground="#666", wraplength=int(330 * UI_SCALE),
                  justify="left").pack(anchor="w")
        fb = ttk.Frame(self.wv_sel)
        fb.pack(anchor="w", pady=(2, 0))
        ttk.Button(fb, text="CHGCAR 形式で保存…", command=self.wv_save).pack(side="left")
        self._wv_target_changed()
        self.side.fit_width()

    def _wv_lset(self, el):
        """元素 el にある軌道 l のリスト（POTCAR から。分からなければ s〜f）"""
        ls = (self.wv.get("lsets") or {}).get(el)
        return list(ls) if ls else [0, 1, 2, 3]

    def _wv_l_toggled(self, l):
        """l のチェック → その l の成分（m）を全部同じにする"""
        on = self.wv_orb[l].get()
        for (ll, m), v in self.wv_m.items():
            if ll == l:
                v.set(on)
        self._wv_edited()

    def _wv_m_toggled(self, l):
        """成分（m）のチェック → l のチェックは「全部選んでいるとき」だけ付ける"""
        ms = [v.get() for (ll, m), v in self.wv_m.items() if ll == l]
        self.wv_orb[l].set(all(ms))
        self._wv_edited()

    def _wv_subset(self):
        """チェックボックスに表示している成分 {l: [m, …]}（m の番号は PROCAR と同じ順）"""
        out = {}
        for l, v in self.wv_orb.items():
            ms_all = list(range(2 * l + 1))
            if l == 0 or not self.wv_m:
                out[l] = ms_all if v.get() else []
            else:
                out[l] = [m for m in ms_all if self.wv_m[(l, m)].get()]
        return out

    def _wv_show_subset(self, sub):
        """成分 {l: [m…]} をチェックボックスに表示する（コマンドは呼ばない）"""
        for l, v in self.wv_orb.items():
            v.set(len(sub.get(l, [])) == 2 * l + 1)
        for (l, m), v in self.wv_m.items():
            v.set(m in sub.get(l, []))

    def _wv_edited(self):
        """チェックボックスを変えた → 編集している対象（全元素、または 1 つの元素）に書き込んで表示し直す"""
        sub = self._wv_subset()
        tgt = self.wv_target.get()
        for el in self.wv_esub:
            if tgt == WV_ALL_ELEM or el == tgt:
                ok = self._wv_lset(el)
                self.wv_esub[el] = {l: list(ms) if l in ok else [] for l, ms in sub.items()}
        self._wv_target_note()
        self._wv_apply()

    def _wv_target_changed(self):
        """編集する対象を切り替えた → その対象の設定をチェックボックスに表示（全元素なら最初の元素の設定）"""
        tgt = self.wv_target.get()
        if tgt in self.wv_esub:
            shown = self._wv_lset(tgt)
            self._wv_show_subset(self.wv_esub[tgt])
        else:
            # 全元素: どれかの元素にある軌道を表示。チェックは「その軌道のある元素で全部選んでいる」とき
            shown = sorted({l for el in self.wv_esub for l in self._wv_lset(el)})
            sub = {}
            for l in range(4):
                els = [el for el in self.wv_esub if l in self._wv_lset(el)]
                sub[l] = [m for m in range(2 * l + 1) if els and all(m in self.wv_esub[el].get(l, []) for el in els)]
            self._wv_show_subset(sub)
        for l, ws in self.wv_rows.items():
            for w in ws:
                if l in shown:
                    w.grid()
                else:
                    w.grid_remove()
        self._wv_target_note()

    def _wv_target_note(self):
        same = len({repr(v) for v in self.wv_esub.values()}) == 1
        txt = ""
        if self.wv_target.get() == WV_ALL_ELEM and not same:
            txt = "元素ごとに違う設定になっています（ここでチェックを変えると、全元素が同じ設定になります）"
        self.wv_target_note.configure(text=txt)

    def _wv_choice(self):
        """選んでいる (原子ごとの成分 AtomSubsets, 原子の番号のリスト, 元素の説明, 軌道の説明)"""
        st = self.wv["st"]
        els = [el for el in st.species if self.wv_el[el].get() and any(self.wv_esub[el].values())]
        atoms = [i for i, sy in enumerate(st.symbols) if sy in els]
        sub = AtomSubsets({a: {l: list(ms) for l, ms in self.wv_esub[st.symbols[a]].items() if ms} for a in atoms})
        names = {el: orbital_subset_name(self.wv_esub[el]) for el in els}
        if len(set(names.values())) <= 1:
            lname = next(iter(names.values())) if names else "(なし)"
        else:
            lname = " ".join(f"{el}({names[el]})" for el in els)
        return sub, atoms, "+".join(els) if els else "(なし)", lname

    WV_MEMO = 8          # 直前に表示した組み合わせをいくつ覚えておくか

    def _wv_gcache(self):
        """原子ごと・成分の組ごとの前計算（計算結果ごとに 1 回だけ作る。密度行列のない以前の結果では None）"""
        wv = self.wv
        res = wv["res"]
        if res.get("M") is None:
            return None
        if wv.get("gcache_of") is not res:
            st = wv["st"]
            shell = float(res.get("shell", 1.5 * grid_cell_diag(st.lattice, res["grid"])))
            lsets = wv.get("lsets")
            with _Busy(self.app, "切り替えを速くするための前計算中…（計算結果ごとに 1 回だけ）"):
                wv["gcache"] = OrbitalGridCache(res["M"], st, res["radii"], res["grid"], res["dr"], shell,
                                                res["label"], {k: v for k, v in lsets.items() if k != "#conf"}
                                                if lsets else None)
            wv["gcache_of"] = res
            wv["memo"] = {}            # 組み合わせごとの表示（覚えておく）
            wv["scache"] = {}          # 原子の組ごとの球の境界までの距離（全格子点）
        return wv["gcache"]

    def _wv_apply(self, first=False):
        wv = self.wv
        if wv is None or wv.get("res") is None:
            return
        res = wv["res"]
        st = wv["st"]
        sub, atoms, aname, lname = self._wv_choice()
        gc = self._wv_gcache()
        memo = wv.setdefault("memo", {}) if gc is not None else {}
        mkey = repr((aname, sorted((a, sorted(s_.items())) for a, s_ in sub.items())))
        hit = memo.pop(mkey, None)
        if hit is None:
            rho = np.zeros(res["grid"], np.float32)
            part = AtomSubsets()
            for l in range(4):
                # l の成分を全部選んだ原子: 計算済みの密度そのもの（その原子の球の中だけ）
                full = [a for a in atoms if l in sub[a] and (len(sub[a][l]) == 2 * l + 1 or res.get("M") is None)]
                if full:
                    rho += res["rho"][l] * np.isin(res["label"], np.array(full, np.int16))
                for a in atoms:
                    if l in sub[a] and a not in full:
                        part.setdefault(a, {})[l] = sub[a][l]     # 一部の成分だけ: 選んだ成分の空間への射影
            if part:
                if gc is not None:
                    rho += gc.inside_density(part, sorted(part))
                else:
                    with _Busy(self.app, "選んだ成分の電荷密度を作成中…"):
                        rho += orbital_points_density(res["M"], st, res["radii"], res["grid"], res["dr"], part,
                                                      atoms=sorted(part))
            vol = abs(np.linalg.det(np.asarray(st.lattice, float)))
            info = {"grid": res["grid"], "volume": vol, "nelect": float(rho.sum() * vol / rho.size),
                    "min": float(rho.min()), "max": float(rho.max())}
            # 元素ごとの電子数（球の中）
            dv = vol / rho.size
            rows = []
            for el in aname.split("+") if atoms else []:
                ids = [a for a in atoms if st.symbols[a] == el]
                ne = float(rho[np.isin(res["label"], np.array(ids, np.int16))].sum() * dv)
                rows.append(f"{el}({orbital_subset_name(self.wv_esub[el])}) {ne:.4f}")
            inside = rho[(res["label"] >= 0) & (rho != 0)]
            # 既定のレベルは球の中の値だけで決める（球の外は 0 なので、全体で決めると低すぎて球の境界が面になる）
            dflt = default_iso_level(inside) if inside.size else None
            F = gc.field_ext(sub, atoms) if gc is not None else None
            hit = {"rho": rho, "info": info, "rows": rows, "dflt": dflt, "F": F}
        if gc is not None:
            memo[mkey] = hit                     # いちばん新しいものを最後に（古いものから捨てる）
            while len(memo) > self.WV_MEMO:
                memo.pop(next(iter(memo)))
        rows = hit["rows"]
        self.wv_sel_lbl.configure(text=("選んだものの球の中の電子数: " + "  ".join(rows)) if rows else "")
        wv["label"] = (lname, aname)
        if first:
            remember_view(self.app, self)
            self.view_key = None
        orb = self._wv_orb(res, st, sub, atoms)
        if orb is not None:
            orb["F"] = hit["F"]
            sc = wv.setdefault("scache", {})
            orb["skey"], orb["scache"] = tuple(atoms), sc
            orb["S"] = sc.get(tuple(atoms))
        self._install(st, hit["rho"], hit["info"], os.path.join(wv["folder"], "WAVECAR"), default=hit["dflt"],
                      orb=orb,
                      head=f"WAVECAR から計算: {res['desc']}\n軌道 {lname} / {aname}",
                      msg=(f"WAVECAR から軌道分解した電荷密度を表示しています: {res['desc']}、軌道 {lname}、{aname}\n"
                           "球の外（原子と原子の間）の電荷は含みません。原子核のすぐ近くは、WAVECAR に PAW の補正が"
                           "入っていないので正確ではありません（VASP の PARCHG と同じ）。"),
                      keep_view=not first, keep_level=not first)

    @staticmethod
    def _wv_orb(res, st, subset, atoms):
        """等値面を球の境界でなめらかに切り、断面を描くための情報（密度行列のない以前の結果では None）"""
        if res.get("M") is None:
            return None
        shell = float(res.get("shell", 1.5 * grid_cell_diag(st.lattice, res["grid"])))
        return {"M": res["M"], "radii": res["radii"], "dr": res["dr"], "atoms": list(atoms), "shell": shell,
                "subset": subset, "F": None}

    def _wv_state(self):
        return {"esub": {el: {str(l): list(ms) for l, ms in sub.items()} for el, sub in self.wv_esub.items()},
                "target": self.wv_target.get(),
                "el": {k: v.get() for k, v in self.wv_el.items()}}

    def _wv_set_state(self, stt):
        if "esub" in stt:
            for el, sub in stt["esub"].items():
                if el in self.wv_esub:
                    ok = self._wv_lset(el)
                    self.wv_esub[el] = {int(l): [int(m) for m in ms] if int(l) in ok else []
                                        for l, ms in sub.items()}
        elif "orb" in stt:
            # v2.6.4 までの保存（全元素共通の設定）→ 全元素に同じ設定
            old = {}
            for l in range(4):
                on = bool(stt["orb"].get(str(l), True))
                old[l] = [m for m in range(2 * l + 1) if bool(stt.get("m", {}).get(f"{l},{m}", on))] if l else \
                    ([0] if on else [])
            for el in self.wv_esub:
                ok = self._wv_lset(el)
                self.wv_esub[el] = {l: list(ms) if l in ok else [] for l, ms in old.items()}
        for k, v in self.wv_el.items():
            v.set(bool(stt.get("el", {}).get(k, True)))
        tgt = stt.get("target", WV_ALL_ELEM)
        self.wv_target.set(tgt if tgt in self.wv_esub else WV_ALL_ELEM)
        self._wv_target_changed()

    # ---- ライブラリ ----
    def library_payload(self):
        d = self.d
        if d is None:
            return None
        obj = {"st": d["st"], "rho": d["rho"], "info": d["info"], "path": d.get("path", "")}
        obj["view_state"] = self.view_state()
        wv = self.wv
        from_wave = bool(wv and wv.get("res") is not None and str(d.get("path", "")).endswith("WAVECAR"))
        if from_wave:
            obj["wv"] = {k: wv[k] for k in ("folder", "st", "kw", "ef", "res", "struct", "lsets") if k in wv}
            obj["wv_state"] = self._wv_state()
            lname, aname = wv.get("label", ("", ""))
            name = f"{d['st'].formula} {wv['res']['desc']} {lname} {aname}"
            kind = "wavecar"
        else:
            name = f"{d['st'].formula} {os.path.basename(str(d.get('path', '')))}"
            kind = "chgcar"
        return kind, obj, name, {"formula": d["st"].formula, "source": d.get("path", "")}

    def library_open(self, obj, meta):
        remember_view(self.app, self)
        self._library_open(obj, meta)
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), obj.get("view_state"))

    def _library_open(self, obj, meta):
        st, rho, info = obj["st"], obj["rho"], obj["info"]
        head = f"ライブラリ: {meta.get('name', '')}"
        if obj.get("wv"):
            self.wv = dict(obj["wv"], wc=None, job=None)
            res = self.wv["res"]
            self.wv_info.configure(text=f"ライブラリから開いた結果: {res.get('desc', '')}\n"
                                        "（軌道・元素・原子の切り替えはできます。計算し直すには元のフォルダを開いてください）",
                                   foreground="#333")
            self._wv_build_selector()
            self._wv_set_state(obj.get("wv_state", {}))
            sub, atoms, aname, lname = self._wv_choice()
            self.wv["label"] = (lname, aname)
            inside = rho[(res["label"] >= 0) & (rho != 0)]
            self._install(st, rho, info, os.path.join(self.wv.get("folder", ""), "WAVECAR"), head=head,
                          msg=f"ライブラリから開きました: {meta.get('name', '')}",
                          default=default_iso_level(inside) if inside.size else None,
                          orb=self._wv_orb(res, st, sub, atoms))
        else:
            self._install(st, rho, info, obj.get("path", ""), head=head,
                          msg=f"ライブラリから開きました: {meta.get('name', '')}")

    def wv_save(self):
        wv = self.wv
        if wv is None or wv.get("res") is None or self.d is None:
            return
        lname, aname = wv.get("label", ("", ""))
        safe = "".join(c if c.isalnum() or c in "+-_" else "_" for c in f"{lname}_{aname}")
        p = self.app.filedialog.asksaveasfilename(title="CHGCAR 形式で保存", initialdir=wv["folder"],
                                                  initialfile=f"CHGCAR_orb_{safe}")
        if not p:
            return
        try:
            with _Busy(self.app, "書き出し中…"):
                write_chgcar(p, wv["st"], self.d["rho"],
                             comment=f"{wv['res']['desc']} orbital {lname} {aname} (VASP Result Viewer)")
        except Exception as e:
            self.app.messagebox.showerror("保存", str(e))
            return
        self.app.set_msg(f"保存しました: {p}\nVESTA でも開けます（値は ρ × セル体積。CHGCAR と同じ）。")

    # ---- 結合の設定（元素対ごと）: 入力ファイル作成アプリと同じ表 ----
    def _build_bond_rows(self, dmin):
        tk, ttk = self.app.tk, self.app.ttk
        for w in self.bond_frame.winfo_children():
            w.destroy()
        self.rule_vars = {}
        self.d["dmin"] = dmin
        ttk.Label(self.bond_frame, text="対", foreground="#555").grid(row=0, column=1, sticky="w")
        ttk.Label(self.bond_frame, text="最大距離[Å]", foreground="#555").grid(row=0, column=2)
        ttk.Label(self.bond_frame, text="最短距離", foreground="#555").grid(row=0, column=3)
        for r_, (key, (on, dmax)) in enumerate(sorted(self.d["rules"].items()), start=1):
            ov = tk.BooleanVar(value=bool(on))
            dv = tk.StringVar(value=f"{dmax:.3f}")
            self.rule_vars[key] = (ov, dv)
            ttk.Checkbutton(self.bond_frame, variable=ov, command=self._rules_changed).grid(row=r_, column=0)
            ttk.Label(self.bond_frame, text=f"{key[0]}–{key[1]}").grid(row=r_, column=1, sticky="w")
            en = ttk.Entry(self.bond_frame, textvariable=dv, width=7)
            en.grid(row=r_, column=2, padx=3)
            en.bind("<Return>", lambda ev: self._rules_changed())
            dm = dmin.get(key, np.inf)
            ttk.Label(self.bond_frame, text=f"{dm:.3f}" if np.isfinite(dm) else "—", foreground="#555").grid(
                row=r_, column=3)

    def _current_rules(self):
        """表の値 → {(A, B): [有効, 最大距離]}。不正な距離は前の値に戻す"""
        out = {}
        for key, (ov, dv) in self.rule_vars.items():
            try:
                dmax = float(dv.get())
                if not 0.0 < dmax < 50.0:
                    raise ValueError
            except ValueError:
                dmax = float(self.d["rules"][key][1])
                dv.set(f"{dmax:.3f}")
            out[key] = [ov.get(), dmax]
        return out

    def _rules_changed(self):
        d = self.d
        if d is None or not self.rule_vars:
            return
        d["rules"] = self._current_rules()
        self._build_model(d["n"])
        self.make_scene()
        nb = len(d["model"]["bonds"])
        on = [f"{a}–{b} ≤ {r[1]:.3f} Å" for (a, b), r in sorted(d["rules"].items()) if r[0]]
        self.app.set_msg(f"結合を計算し直しました: {nb} 本\n" + ("、".join(on) if on else "（結合を描く対はありません）"))

    def _rules_default(self):
        d = self.d
        if d is None:
            return
        d["rules"] = default_bond_rules(d["st"].species, d["dmin"])
        self._build_bond_rows(d["dmin"])
        self._rules_changed()

    def _read_n(self):
        """スーパーセルの入力欄 → [na, nb, nc]（不正なら None）"""
        try:
            n = [int(v.get()) for v in self.nvars]
            if min(n) < 1:
                raise ValueError
            return n
        except ValueError:
            return None

    def _build_model(self, n):
        d = self.d
        d["n"] = list(n)
        d["model"] = build_structure_model(d["st"], n, True, d["rules"], BALL_SCALE, False, bonded_outside=False)

    def apply_supercell(self):
        d = self.d
        if d is None:
            return
        n = self._read_n()
        if n is None:
            self.app.messagebox.showwarning("スーパーセル", "1 以上の整数を入力してください。")
            return
        est = d["st"].natoms * n[0] * n[1] * n[2]
        if est > 5000 and not self.app.messagebox.askyesno(
                "スーパーセル", f"原子数が約 {est} 個になり、表示が重くなります。続けますか？"):
            return
        with _Busy(self.app, "スーパーセルを作成中…"):
            self._build_model(n)
            self.make_scene(refit=True)
        self.app.set_msg(f"スーパーセル {n[0]}×{n[1]}×{n[2]} を表示しました。")

    def _labels_all(self, on):
        for v in self.lab_vars.values():
            v.set(on)
        self.make_scene()

    def reset_level(self):
        if self.d is None:
            return
        self.level_var.set(f"{self.d['default']:.5g}")
        self.rebuild()

    def rebuild(self):
        """等値面を作り直す"""
        d = self.d
        if d is None:
            return
        lv = fnum(self.level_var, round(d["default"], 6), 1e-12, 1e12)
        rho = d["rho"]
        orb = d.get("orb")
        stride = [1, 1, 1]
        if self.res_var.get() == "auto":
            for ax, n in enumerate(rho.shape):
                need = int(np.ceil(n / CHG_AUTO_MAXGRID))
                stride[ax] = next((s for s in range(need, need + 8) if n % s == 0), need)
        ray = self._use_ray()
        try:
            with _Busy(self.app, "等値面を計算中…"):
                if orb is not None:
                    # 軌道分解: 球の外側の薄い殻まで延ばした値で等値面を作り、球の境界で切る（段々にならない）
                    if orb.get("F") is None:
                        orb["F"] = orbital_field_ext(orb["M"], d["st"], orb["radii"], rho.shape, orb["dr"],
                                                     orb["subset"], orb["atoms"], orb["shell"])
                    field = orb["F"]
                else:
                    field = rho
                if ray:
                    # GPU レイキャスティング: 格子をそのまま GPU に送る（間引かない。レベルを変えても送り直さない）
                    sdist = None
                    if orb is not None:
                        if orb.get("S") is None:
                            orb["S"] = sphere_signed_grid(d["st"], orb["radii"], orb["atoms"], rho.shape)
                            sc = orb.get("scache")
                            if sc is not None:           # 原子の組ごとに覚えておく（軌道を変えても同じ）
                                if len(sc) > 8:
                                    sc.clear()
                                sc[orb["skey"]] = orb["S"]
                        sdist = orb["S"]
                    if d.get("vol_key") is None:
                        ChargeTab._vol_count += 1
                        d["vol_key"] = ChargeTab._vol_count
                    d.update(D=None, Dn=None, ray=True, field=field, sdist=sdist, sub=field, level=lv, sec_key=None)
                    self.iso_lbl.configure(
                        text=f"= {lv * ANG3_TO_BOHR3:.5g} e/bohr³  （既定値 {d['default']:.5g} e/Å³）")
                    self.make_scene()
                    self.app.set_msg(f"等値面を更新しました（レベル {lv:.5g} e/Å³、GPU レイキャスティング）。")
                    return
                d["ray"] = False
                sub = field[::stride[0], ::stride[1], ::stride[2]]
                D = isosurface_tets(sub, lv, d["st"].lattice)
                if orb is not None and len(D):
                    s = sphere_signed(D[:, :, :3].reshape(-1, 3), d["st"], orb["radii"], orb["atoms"]).reshape(-1, 3)
                    D = clip_tris(D, s)
                d["D"] = D
                d["Dn"] = isosurface_tets(sub, -lv, d["st"].lattice) if (orb is None and d["info"]["min"] < -lv) else None
                d["sub"], d["level"], d["sec_key"] = sub, lv, None
        except MemoryError as e:
            self.app.messagebox.showwarning("等値面", str(e))
            return
        nt = len(d["D"]) + (0 if d["Dn"] is None else len(d["Dn"]))
        self.iso_lbl.configure(
            text=f"= {lv * ANG3_TO_BOHR3:.5g} e/bohr³  （既定値 {d['default']:.5g} e/Å³）  三角形 {nt} 枚"
                 + ("" if stride == [1, 1, 1] else f"（格子 {sub.shape[0]}×{sub.shape[1]}×{sub.shape[2]} に間引き）"))
        self.make_scene()
        self.app.set_msg(f"等値面を更新しました（レベル {lv:.5g} e/Å³、三角形 {nt} 枚）。")

    _vol_count = 0

    def _use_ray(self):
        """GPU レイキャスティングで描くか（OpenGL 3.3 でレイキャスティングのシェーダーが使えるときだけ）"""
        return self.draw_var.get() == "ray" and self.viewer.mode == "gl2" and self.viewer.volume_ok() is not False

    def _check_ray(self):
        """OpenGL の初期化のあとで、レイキャスティングが使えないと分かったら三角形で描き直す"""
        d = self.d
        if d is not None and d.get("ray") and self.viewer.volume_ok() is False:
            err = getattr(getattr(self.viewer.w, "ren", None), "vol_error", "")
            self.rebuild()
            self.app.add_msg("【注意】この PC では GPU レイキャスティングを使えないため、三角形で描いています。"
                             + (f"（{err}）" if err else ""))

    def _sec_range(self, lv, field):
        """断面の色の飽和レベル（自動: 最小 = レベル、最大 = データの最大値）"""
        vmax_data = float(max(np.abs(field).max(), lv))
        if self.sec_auto.get():
            vmin, vmax = lv, vmax_data
            self.sec_min.set(f"{vmin:.5g}")
            self.sec_max.set(f"{vmax:.5g}")
        else:
            vmin = fnum(self.sec_min, round(lv, 6), 0.0, 1e12)
            vmax = fnum(self.sec_max, round(vmax_data, 6), 0.0, 1e12)
        return vmin, vmax

    def make_scene(self, refit=None):
        d = self.d
        if d is None or (d["D"] is None and not d.get("ray")):
            return
        st, m = d["st"], d["model"]
        n = d["n"]
        op = fnum(self.opac_var, ISO_OPACITY, 0.05, 1.0)
        sc = empty_scene()
        if self.atom_var.get():
            sc["spheres"] = np.hstack([m["cart"], m["rad"][:, None], np.array(m["col"])])
        if self.bond_var.get() and len(m["bonds"]):
            cart, col = m["cart"], m["col"]
            cy = []
            for i, j in m["bonds"]:
                mid = 0.5 * (cart[i] + cart[j])
                cy.append(np.r_[cart[i], mid, BOND_RADIUS, col[i]])
                cy.append(np.r_[mid, cart[j], BOND_RADIUS, col[j]])
            sc["cyls"] = np.array(cy, np.float32)
        if self.cell_var.get():
            sc["lines"] = [(box_edges(st.lattice), (0, 0, 0), 1.3)]
        ncell = n[0] * n[1] * n[2]
        ntri = 0
        iso_all = True
        if d.get("ray"):
            # GPU レイキャスティング: 1 セル分のデータを周期的にくり返して読む（スーパーセルでもデータは増えない）
            lv = d["level"]
            vmin, vmax = self._sec_range(lv, d["field"])
            orb = d.get("orb")
            sph = None
            if orb is not None:
                fr = np.mod(np.asarray(st.frac, float), 1.0)
                sph = np.array([[*fr[a], orb["radii"][a]] for a in orb["atoms"]], np.float32)
            sc["volume"] = {"key": d["vol_key"], "data": d["field"], "sdist": d.get("sdist"), "L": st.lattice,
                            "spheres": sph,
                            "ncell": list(n), "level": lv, "alpha": op,
                            "neg": d.get("orb") is None and d["info"]["min"] < -lv,
                            "iso_rgb": ISO_COLOR_POS, "neg_rgb": ISO_COLOR_NEG, "sections": self.sec_var.get(),
                            "vmin": vmin, "vmax": vmax, "cmode": SECTION_MODES.index(self.sec_mode.get())}
            self.app.root.after(400, self._check_ray)
        else:
            parts = [mesh_vertices(d["D"], ISO_COLOR_POS + (op,))]
            if d["Dn"] is not None and len(d["Dn"]):
                parts.append(mesh_vertices(d["Dn"], ISO_COLOR_NEG + (op,)))
            mesh = np.vstack(parts)
            # スーパーセル: 1 セル分の等値面を、格子ベクトルの分だけずらして何回も描く（GPU ではインスタンス描画）
            ntri = len(mesh) // 3
            limit = CHG_MAX_TRIANGLES_GPU if self.viewer.mode == "gl2" else CHG_MAX_TRIANGLES
            iso_all = ntri * ncell <= limit
            if ncell > 1 and iso_all:
                sc["mesh_offsets"] = (np.array([[i, j, k] for i in range(n[0]) for j in range(n[1])
                                                for k in range(n[2])], float) @ st.lattice).astype(np.float32)
            sc["mesh" if op >= 0.999 else "mesh_t"] = mesh
            # 断面（VESTA の Sections）: 描画範囲の境界（セルの面）と、軌道分解の球の境界での切り口
            if self.sec_var.get() and d.get("sub") is not None:
                sec = self._sections(n if iso_all else [1, 1, 1], op)
                if sec is not None and len(sec):
                    sc["mesh_s" if op >= 0.999 else "mesh_s_t"] = sec
        # 元素ごとのラベル（サイトの名前 Ta1, Se3, … を原子の手前側の表面に表示）
        on = {el for el, v in self.lab_vars.items() if v.get()}
        idx = [i for i, s in enumerate(m["sy"]) if s in on] if self.atom_var.get() else []
        if idx:
            sc["labels"] = ([m["lb"][i] for i in idx], m["cart"][idx], m["rad"][idx] * 1.02)
        self.viewer.label_pt = fnum(self.lsize_var, LABEL_SIZE, 4, 40)
        if self.axes_var.get():
            sc["triad"] = (st.lattice, ("a", "b", "c"))
        a, b, c = st.lattice * np.array(n, float)[:, None]
        corners = np.array([[0, 0, 0], a, b, c, a + b, a + c, b + c, a + b + c])
        sc["center"] = corners.mean(axis=0)
        sc["radius"] = float(np.linalg.norm(corners - sc["center"], axis=1).max()) + 3.0
        sc["fit"] = np.vstack([corners, m["cart"]])
        self.cell_lbl.configure(
            text=f"{n[0]}×{n[1]}×{n[2]}  表示原子 {len(m['cart'])} 個  結合 {len(m['bonds'])} 本"
                 + ("" if iso_all else f"\n【注意】等値面の三角形が {ntri * ncell} 枚になるため、等値面は最初の 1 セルだけに"
                                       "描いています（レベルを上げるか、スーパーセルを小さくしてください）。"))
        self.viewer.set_scene(sc, refit=refit)
        self._perf()

    def _sections(self, ncell, alpha):
        """断面の頂点配列（色付き）。形は等値面のレベル・セルの数・解像度が変わったときだけ作り直す"""
        d = self.d
        orb = d.get("orb")
        lv, sub, st = d["level"], d["sub"], d["st"]
        key = (lv, tuple(ncell), sub.shape)
        if d.get("sec_key") != key:
            with _Busy(self.app, "断面を作成中…"):
                empty = np.zeros((0, 3, 7), np.float32)
                if orb is not None:
                    S = section_faces(sub, st.lattice, ncell, lv, st=st, radii=orb["radii"], atoms=orb["atoms"])
                    C = sphere_caps(orb["M"], st, orb["radii"], orb["dr"], orb["subset"], orb["atoms"], lv, ncell)
                    S = np.concatenate([S, C], axis=0)
                    Sn = empty
                else:
                    S = section_faces(sub, st.lattice, ncell, lv)
                    Sn = section_faces(-sub, st.lattice, ncell, lv) if d["info"]["min"] < -lv else empty
            d["sec"], d["sec_n"], d["sec_key"] = S, Sn, key
        S, Sn = d["sec"], d["sec_n"]
        vmin, vmax = self._sec_range(lv, sub)
        mode = self.sec_mode.get()
        parts = []
        if len(S):
            parts.append(section_vertices(S, vmin, vmax, mode, alpha, ISO_COLOR_POS))
        if len(Sn):
            parts.append(section_vertices(Sn, vmin, vmax, mode, alpha, ISO_COLOR_NEG))
        return np.vstack(parts) if parts else None

    def save_image(self):
        img = self.viewer.grab()
        if img is None:
            return
        p = self.app.filedialog.asksaveasfilename(title="画像を保存", defaultextension=".png",
                                                  filetypes=[("PNG", "*.png")])
        if p:
            img.convert("RGB").save(p)
            self.app.set_msg(f"画像を保存しました: {p}")


# ----------------------------------------------------------------------
#  タブ 2: DOS / BAND
# ----------------------------------------------------------------------
class DosBandTab:
    title = "DOS / BAND"

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.band = None
        self.dos = None
        self.elems = []
        self.elem_vars = {}
        side, holder = make_side_layout(app, frame)
        self.side = side
        self.plot = PlotArea(app, holder)
        self.plot.message("vasprun.xml を読み込んでください")

        section(app, side, "① vasprun.xml")
        ttk.Button(side, text="BAND 計算の vasprun.xml を開く…", command=lambda: self.choose("band")).pack(anchor="w")
        self.band_lbl = ttk.Label(side, text="BAND: 未読み込み", foreground="#777",
                                  wraplength=int(330 * UI_SCALE), justify="left")
        self.band_lbl.pack(anchor="w")
        ttk.Button(side, text="DOS 計算の vasprun.xml を開く…", command=lambda: self.choose("dos")).pack(
            anchor="w", pady=(4, 0))
        self.dos_lbl = ttk.Label(side, text="DOS: 未読み込み", foreground="#777",
                                 wraplength=int(330 * UI_SCALE), justify="left")
        self.dos_lbl.pack(anchor="w")

        section(app, side, "表示")
        self.view_var = tk.StringVar(value="both")
        fv = ttk.Frame(side)
        fv.pack(anchor="w")
        for text, val in (("BAND + DOS", "both"), ("BAND", "band"), ("DOS", "dos")):
            ttk.Radiobutton(fv, text=text, value=val, variable=self.view_var, command=self.draw).pack(
                side="left", padx=(0, 8))
        self.pband_var = tk.BooleanVar(value=False)
        self.pdos_var = tk.BooleanVar(value=False)
        self.orb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="PBAND（元素ごとの寄与を丸の大きさで表示）", variable=self.pband_var,
                        command=self.draw).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(side, text="PDOS（元素ごとの状態密度）", variable=self.pdos_var,
                        command=self.draw).pack(anchor="w")
        ttk.Checkbutton(side, text="PDOS を軌道別 (s, p, d, f) に分ける", variable=self.orb_var,
                        command=self.draw).pack(anchor="w", padx=(18, 0))

        section(app, side, "元素（PBAND / PDOS に表示するもの）")
        self.elem_frame = ttk.Frame(side)
        self.elem_frame.pack(anchor="w")
        ttk.Label(self.elem_frame, text="（読み込むと表示されます）", foreground="#777").pack()
        fa = ttk.Frame(side)
        fa.pack(anchor="w")
        ttk.Button(fa, text="全部", command=lambda: self._all(True)).pack(side="left")
        ttk.Button(fa, text="なし", command=lambda: self._all(False)).pack(side="left", padx=4)

        section(app, side, "エネルギー")
        fe = ttk.Frame(side)
        fe.pack(anchor="w")
        self.efb_var = tk.StringVar(value="")       # BAND の E_F
        self.efd_var = tk.StringVar(value="")       # DOS の E_F
        self.emin_var = tk.StringVar(value=str(ENERGY_WINDOW[0]))
        self.emax_var = tk.StringVar(value=str(ENERGY_WINDOW[1]))
        self.dmax_var = tk.StringVar(value="")
        self.fat_var = tk.StringVar(value=str(FATBAND_SCALE))
        rows = (("BAND の E_F [eV]:", self.efb_var), ("DOS の E_F [eV]:", self.efd_var),
                ("表示範囲 下限 E−E_F [eV]:", self.emin_var),
                ("表示範囲 上限 E−E_F [eV]:", self.emax_var), ("DOS の上限（空欄 = 自動）:", self.dmax_var),
                ("PBAND の丸の大きさ:", self.fat_var))
        for r, (lab, var) in enumerate(rows):
            ttk.Label(fe, text=lab).grid(row=r, column=0, sticky="w")
            en = ttk.Entry(fe, textvariable=var, width=10)
            en.grid(row=r, column=1, padx=4, pady=1)
            en.bind("<Return>", lambda ev: self.draw())
        fb = ttk.Frame(side)
        fb.pack(anchor="w", pady=(4, 0))
        ttk.Button(fb, text="再描画", command=self.draw).pack(side="left")
        ttk.Button(fb, text="E_F を読み直す", command=lambda: self._default_ef(True)).pack(side="left", padx=4)
        fb2 = ttk.Frame(side)
        fb2.pack(anchor="w", pady=(2, 0))
        ttk.Button(fb2, text="BAND: VBM を 0 にする", command=self.set_vbm).pack(side="left")
        ttk.Button(fb2, text="BAND に DOS の E_F を使う", command=self.use_dos_ef).pack(side="left", padx=4)
        self.ef_lbl = ttk.Label(side, text="", foreground="#555", wraplength=int(330 * UI_SCALE), justify="left")
        self.ef_lbl.pack(anchor="w", pady=(4, 0))
        ttk.Label(side, text="画像の保存・拡大・移動は、グラフ下のツールバーで行います。",
                  foreground="#666", wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w", pady=(6, 0))

    # ---- 読み込み ----
    def choose(self, kind):
        p = self.app.filedialog.askopenfilename(
            title=f"{kind.upper()} 計算の vasprun.xml を選択",
            filetypes=[("vasprun.xml", "*.xml"), ("All files", "*.*")])
        if p:
            self.load(p, kind)

    STYLE = ("view_var", "orb_var", "emin_var", "emax_var", "fat_var")

    def _key(self):
        parts = [os.path.abspath(d["path"]) for d in (self.band, self.dos) if d is not None and d.get("path")]
        return "dos:" + "|".join(parts) if parts else None

    def view_state(self):
        if self.band is None and self.dos is None:
            return None
        return {"vars": capture_vars(self)}

    def apply_view_state(self, stt):
        restore_vars(self, stt.get("vars"))
        self.draw()

    def load(self, path, kind=None):
        """最初は固有値と全 DOS だけを読む（速い）。PDOS / PBAND 用のデータは、チェックを入れたときに読む"""
        app = self.app
        remember_view(app, self)
        try:
            with _Busy(app, f"vasprun.xml を読み込み中… {path}"):
                want = ("eig",) if kind == "band" else ("eig", "dos")
                vr, hit = cached_read(app, "vasprun", [path], {"want": list(want), "v": 2},
                                      lambda: read_vasprun(path, want=want))
        except Exception as e:
            app.messagebox.showerror("vasprun.xml 読み込みエラー", f"{path}\n\n{e}")
            app.set_msg("")
            return
        is_line = vr["kgen"].get("param") == "listgenerated"
        warn = []
        if vr.get("warning"):
            warn.append(vr["warning"])
        if kind is None:
            kind = "band" if (is_line or vr["dos_total"] is None) else "dos"
        if kind == "band" and vr["eig"] is None:
            app.messagebox.showerror("BAND", "この vasprun.xml には固有値がありません。")
            return
        if kind == "dos" and vr["dos_total"] is None:
            app.messagebox.showerror("DOS", "この vasprun.xml には DOS がありません。")
            return
        vr["path"] = path
        sp, grp = species_groups(vr["symbols"])
        vr["species"], vr["groups"] = sp, grp
        soc = vr["incar"].get("LSORBIT", "F").strip().upper().startswith("T")
        tag = "  SOC" if soc else ""
        if vr["incar"].get("ISPIN", "1").strip() == "2":
            warn.append("ISPIN = 2 の計算です。このアプリは非磁性を前提にしており、BAND・PBAND は最初のスピン、"
                        "DOS・PDOS は 2 スピンの和を表示します。")
        if kind == "band":
            kfile = None
            kp = os.path.join(os.path.dirname(path), "KPOINTS")
            if os.path.isfile(kp):
                try:
                    kfile = read_kpoints_file(kp)
                except Exception:
                    kfile = None
            plabels = kfile.get("point_labels") if kfile else None
            if plabels is not None and len(plabels) != len(vr["kpts"]):
                plabels = None
            w = vr["weights"]
            vr["keep"] = None
            if not is_line and w is not None and (w == 0).any() and (w > 0).any():
                # 重み 0 の k点だけがバンド経路（ハイブリッド汎関数などの計算方法）
                keep = np.where(w == 0)[0]
                vr["keep"] = keep
                vr["kpts"], vr["eig"], vr["occ"] = vr["kpts"][keep], vr["eig"][:, keep], vr["occ"][:, keep]
                if plabels is not None:
                    plabels = [plabels[i] for i in keep]
                warn.append(f"重み 0 の k点 {len(keep)} 個をバンド経路として表示します。")
            x, ticks, labels, segs = band_path_axis(vr["kpts"], vr["lattice"], vr["kgen"], kfile, plabels)
            vr.update(x=x, ticks=ticks, labels=labels, segs=segs)
            if not is_line and not any(labels):
                warn.append("この vasprun.xml は line-mode の BAND 計算ではありません（k点を計算した順に並べて表示します）。")
            elif not any(labels):
                warn.append("同じフォルダに KPOINTS がない（または区間数が合わない）ため、高対称点の名前を表示できません。")
            self.band = vr
            self.band_lbl.configure(
                text=f"BAND: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}\n"
                     f"k点 {len(vr['kpts'])}、バンド {vr['eig'].shape[2]}、"
                     f"射影 {'あり' if vr['has_proj'] else 'なし'}{tag}", foreground="#222")
            if not vr["has_proj"]:
                warn.append("BAND の vasprun.xml に射影がないため PBAND は表示できません（LORBIT = 11 で計算してください）。")
        else:
            self.dos = vr
            self.dos_lbl.configure(
                text=f"DOS: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}\n"
                     f"NEDOS {len(vr['dos_e'])}、"
                     f"PDOS {'あり' if vr['has_pdos'] else 'なし'}{tag}", foreground="#222")
            if not vr["has_pdos"]:
                warn.append("DOS の vasprun.xml に部分状態密度がないため PDOS は表示できません（LORBIT = 11 で計算してください）。")
        self._overflow_warn(vr, warn)
        self._rebuild_elems()
        self._default_ef(False, only=kind)
        app.select_tab(self)
        app.set_msg(f"{kind.upper()} を読み込みました: {path}" + ("（キャッシュから）" if hit else "")
                    + ("\n" + "\n".join("【注意】" + w for w in warn) if warn else ""))
        self.draw()
        app.add_recent("vasprun_" + kind, path)
        restore_view(app, self, self._key())
        try:
            if vr.get("lattice") is not None and vr.get("frac") is not None:
                app.show_structure(structure_from_symbols(f"vasprun.xml ({kind})", vr["lattice"], vr["symbols"],
                                                          vr["frac"]), path, f"DOS / BAND タブの vasprun.xml（{kind.upper()}）")
        except Exception:
            pass

    # ---- ライブラリ ----
    def library_payload(self):
        if self.band is None and self.dos is None:
            return None
        # PDOS / PBAND のデータも入れておく（ライブラリから開いたときは元の vasprun.xml がないかもしれない）
        if self.band is not None:
            self._lazy(self.band, "proj", ("eig", "proj"), "PBAND")
        if self.dos is not None:
            self._lazy(self.dos, "dos_partial", ("eig", "dos", "pdos"), "PDOS")
        obj = {"band": self.band, "dos": self.dos, "view_state": self.view_state()}
        any_d = self.band if self.band is not None else self.dos
        form = "".join(f"{s}{list(any_d['symbols']).count(s)}" for s in any_d["species"])
        parts = [k for k, v in (("BAND", self.band), ("DOS", self.dos)) if v is not None]
        src = os.path.dirname(any_d.get("path", ""))
        return "dosband", obj, f"{form} {'+'.join(parts)} {os.path.basename(src)}", {"formula": form, "source": src}

    def library_open(self, obj, meta):
        remember_view(self.app, self)
        self._library_open(obj, meta)
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), obj.get("view_state"))

    def _library_open(self, obj, meta):
        self.band, self.dos = obj.get("band"), obj.get("dos")
        for d in (self.band, self.dos):
            if d is not None:
                d["tried_proj"] = d["tried_dos_partial"] = True      # 足りないデータを元のファイルから読みに行かない
        if self.band is not None:
            vr = self.band
            self.band_lbl.configure(text=f"BAND（ライブラリ）: {meta.get('name', '')}\n"
                                         f"k点 {len(vr['kpts'])}、バンド {vr['eig'].shape[2]}、"
                                         f"射影 {'あり' if vr.get('proj') is not None else 'なし'}", foreground="#222")
        else:
            self.band_lbl.configure(text="BAND: （なし）", foreground="#777")
        if self.dos is not None:
            vr = self.dos
            self.dos_lbl.configure(text=f"DOS（ライブラリ）: {meta.get('name', '')}\n"
                                        f"NEDOS {len(vr['dos_e'])}、"
                                        f"PDOS {'あり' if vr.get('dos_partial') is not None else 'なし'}", foreground="#222")
        else:
            self.dos_lbl.configure(text="DOS: （なし）", foreground="#777")
        self._rebuild_elems()
        for kind in ("band", "dos"):
            if getattr(self, kind) is not None:
                self._default_ef(False, only=kind)
        self.app.select_tab(self)
        self.app.set_msg(f"ライブラリから開きました: {meta.get('name', '')}")
        self.draw()

    @staticmethod
    def _overflow_warn(vr, warn):
        if vr.get("n_overflow"):
            names = {"eig": "固有値", "dos_total": "全 DOS", "dos_partial": "PDOS", "proj": "射影"}
            warn.append("vasprun.xml に桁あふれした数値（'****'）があり、欠損として読みました: "
                        + "、".join(f"{names[k]} {n} 個" for k, n in vr["n_overflow"].items())
                        + "。その部分は線が途切れます。")

    def _lazy(self, d, key, want, what):
        """PDOS / PBAND 用のデータを、必要になったときに 1 度だけ読む。読めたら True"""
        if d is None:
            return False
        if d.get(key) is not None:
            return True
        if d.get("tried_" + key) or not d.get("has_pdos" if key == "dos_partial" else "has_proj"):
            return False
        d["tried_" + key] = True
        app = self.app
        try:
            with _Busy(app, f"{what} のデータを読み込み中…（大きなファイルは時間がかかります） {d['path']}"):
                vr, _hit = cached_read(app, "vasprun", [d["path"]], {"want": list(want), "v": 2},
                                       lambda: read_vasprun(d["path"], want=want))
        except Exception as e:
            app.messagebox.showerror("vasprun.xml 読み込みエラー", f"{d['path']}\n\n{e}")
            return False
        warn = [vr["warning"]] if vr.get("warning") else []
        if key == "proj" and vr["proj"] is not None:
            d["proj"] = vr["proj"] if d.get("keep") is None else vr["proj"][d["keep"]]
            d["proj_orbitals"] = vr["proj_orbitals"]
        elif key == "dos_partial" and vr["dos_partial"] is not None:
            d["dos_partial"], d["dos_orbitals"] = vr["dos_partial"], vr["dos_orbitals"]
        self._overflow_warn(vr, warn)
        app.set_msg(f"{what} のデータを読み込みました。" + ("\n" + "\n".join("【注意】" + w for w in warn) if warn else ""))
        return d.get(key) is not None

    def _rebuild_elems(self):
        tk, ttk = self.app.tk, self.app.ttk
        sp = []
        for d in (self.band, self.dos):
            if d is not None:
                sp += [s for s in d["species"] if s not in sp]
        old = {k: v.get() for k, v in self.elem_vars.items()}
        for w in self.elem_frame.winfo_children():
            w.destroy()
        self.elems, self.elem_vars = sp, {}
        for k, el in enumerate(sp):
            v = tk.BooleanVar(value=old.get(el, True))
            self.elem_vars[el] = v
            tk.Label(self.elem_frame, text="●", fg=PLOT_COLORS[k % len(PLOT_COLORS)], font=("", 13)).grid(
                row=k // 4, column=2 * (k % 4))
            ttk.Checkbutton(self.elem_frame, text=el, variable=v, command=self.draw).grid(
                row=k // 4, column=2 * (k % 4) + 1, sticky="w", padx=(0, 8))
        self.side.fit_width()

    def _all(self, on):
        for v in self.elem_vars.values():
            v.set(on)
        self.draw()

    def _ef_label(self):
        t = []
        for name, d, var in (("BAND", self.band, self.efb_var), ("DOS", self.dos, self.efd_var)):
            if d is not None:
                f = d["efermi"]
                t.append(f"{name}: 原点 {var.get()} eV（ファイルの E_F = " + ("—" if f is None else f"{f:.4f}") + " eV）")
        self.ef_lbl.configure(text="\n".join(t))

    def _default_ef(self, redraw, only=None):
        """BAND と DOS の原点を、それぞれのファイルに書かれている E_F にする（互いに独立）"""
        for name, d, var in (("band", self.band, self.efb_var), ("dos", self.dos, self.efd_var)):
            if d is not None and d["efermi"] is not None and only in (None, name):
                var.set(f"{d['efermi']:.6f}")
        self._ef_label()
        if redraw:
            self.draw()

    def use_dos_ef(self):
        if self.dos is None or not self.efd_var.get().strip():
            self.app.messagebox.showinfo("E_F", "先に DOS の vasprun.xml を読み込んでください。")
            return
        self.efb_var.set(self.efd_var.get())
        self._ef_label()
        self.draw()

    def set_vbm(self):
        d = self.band
        if d is None or d.get("occ") is None:
            self.app.messagebox.showinfo("VBM", "先に BAND の vasprun.xml を読み込んでください。")
            return
        occ = d["occ"][0]
        full = occ > 0.5 * occ.max()
        if not full.any() or full.all():
            return
        vbm = float(d["eig"][0][full].max())
        cbm = float(d["eig"][0][~full].min())
        self.efb_var.set(f"{vbm:.6f}")
        self._ef_label()
        self.app.set_msg(f"BAND の原点を VBM = {vbm:.4f} eV にしました（占有数から判定。CBM − VBM = {cbm - vbm:.4f} eV）。"
                         + ("\n【注意】占有バンドと非占有バンドが重なっています（金属・半金属）。" if cbm < vbm else ""))
        self.draw()

    # ---- 描画 ----
    def _color(self, el):
        return PLOT_COLORS[self.elems.index(el) % len(PLOT_COLORS)]

    def draw(self):
        fig = self.plot.fig
        mode = self.view_var.get()
        have_b, have_d = self.band is not None, self.dos is not None
        if not (have_b or have_d):
            return
        show_b = have_b and mode in ("both", "band")
        show_d = have_d and mode in ("both", "dos")
        if not (show_b or show_d):
            self.plot.message("この表示に必要な vasprun.xml が読み込まれていません")
            return
        # PBAND / PDOS にチェックが入っていれば、そのデータをここで読む（最初の 1 回だけ）
        if show_b and self.pband_var.get() and not self._lazy(self.band, "proj", ("eig", "proj"), "PBAND"):
            self.pband_var.set(False)
        if show_d and self.pdos_var.get() and not self._lazy(self.dos, "dos_partial", ("eig", "dos", "pdos"), "PDOS"):
            self.pdos_var.set(False)
        efb = fnum(self.efb_var, 0.0)
        efd = fnum(self.efd_var, 0.0)
        self._ef_label()
        emin = fnum(self.emin_var, ENERGY_WINDOW[0])
        emax = fnum(self.emax_var, ENERGY_WINDOW[1])
        if emax <= emin:
            emin, emax = ENERGY_WINDOW
        fig.clear()
        if show_b and show_d:
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
            axb = fig.add_subplot(gs[0])
            axd = fig.add_subplot(gs[1], sharey=axb)
            self._draw_band(axb, efb, emin, emax)
            self._draw_dos(axd, efd, emin, emax, horizontal=True)
            axd.tick_params(labelleft=False)
        elif show_b:
            self._draw_band(fig.add_subplot(111), efb, emin, emax)
        else:
            self._draw_dos(fig.add_subplot(111), efd, emin, emax, horizontal=False)
        self.plot.draw()

    def _draw_band(self, ax, ef, emin, emax):
        from matplotlib.collections import LineCollection
        d = self.band
        x, E = d["x"], d["eig"][0] - ef
        # 表示範囲の近くを通るバンドだけを、1 つの線の集まりとして描く（バンド 1 本ずつ描くより速い）
        margin = max(5.0, emax - emin)
        with np.errstate(invalid="ignore"):
            near = ((E > emin - margin) & (E < emax + margin)).any(axis=0)
        bdraw = np.where(near)[0]
        lines = [np.column_stack([x[s:e + 1], E[s:e + 1, b]]) for s, e in d["segs"] for b in bdraw]
        ax.add_collection(LineCollection(lines, colors="#222222", linewidths=1.0, zorder=3))
        for t in d["ticks"]:
            ax.axvline(t, color="#888888", lw=0.6, zorder=1)
        ax.axhline(0.0, color="#888888", lw=0.6, ls="--", zorder=1)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(emin, emax)
        ax.set_xticks(d["ticks"])
        ax.set_xticklabels(d["labels"])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("E − E$_F$ (eV)")
        if self.pband_var.get() and d.get("proj") is not None:
            sc = fnum(self.fat_var, FATBAND_SCALE, 0.1, 1e4)
            with np.errstate(invalid="ignore"):
                inwin = ((E > emin - 0.5) & (E < emax + 0.5)).any(axis=0)
            bsel = np.where(inwin)[0]
            X = np.repeat(x[:, None], len(bsel), axis=1).ravel()
            Y = E[:, bsel].ravel()
            for el in self.elems:
                if not self.elem_vars[el].get() or el not in d["species"]:
                    continue
                ions = np.where(d["groups"] == d["species"].index(el))[0]
                w = np.nan_to_num(d["proj"][:, bsel][:, :, ions, :].sum(axis=(2, 3))).ravel() * sc
                m = w > 0.3                       # ほとんど見えない大きさの点は描かない
                ax.scatter(X[m], Y[m], s=w[m], color=self._color(el), alpha=0.55, linewidths=0, zorder=2,
                           label=el)
            h, l = ax.get_legend_handles_labels()
            if h:
                from matplotlib.lines import Line2D
                hh = [Line2D([], [], marker="o", ls="", color=x_.get_facecolor()[0], markersize=7) for x_ in h]
                ax.legend(hh, l, loc="upper right", fontsize=9, handletextpad=0.2)

    def _draw_dos(self, ax, ef, emin, emax, horizontal):
        d = self.dos
        e = d["dos_e"] - ef
        tot = total_dos(d)
        m = (e >= emin) & (e <= emax)
        ymax = 0.0

        def line(y, **kw):
            nonlocal ymax
            if m.any() and np.isfinite(y[m]).any():
                ymax = max(ymax, float(np.nanmax(y[m])))
            return ax.plot(y, e, **kw) if horizontal else ax.plot(e, y, **kw)

        line(tot, color="#222222", lw=1.0, label="Total")
        if horizontal:
            ax.fill_betweenx(e, 0, tot, color="#bbbbbb", alpha=0.35, lw=0)
        else:
            ax.fill_between(e, 0, tot, color="#bbbbbb", alpha=0.35, lw=0)
        if self.pdos_var.get() and d.get("dos_partial") is not None:
            ls_of = {"s": ":", "p": "--", "d": "-", "f": "-."}
            ol = [orbital_l(o) for o in d["dos_orbitals"]]
            for el in self.elems:
                if not self.elem_vars[el].get() or el not in d["species"]:
                    continue
                ions = np.where(d["groups"] == d["species"].index(el))[0]
                p = d["dos_partial"][ions].sum(axis=0)            # (nedos, norb)
                if self.orb_var.get():
                    for l in "spdf":
                        cols = [i for i, o in enumerate(ol) if o == l]
                        if cols and np.nan_to_num(p[:, cols]).max() > 0:
                            line(p[:, cols].sum(axis=1), color=self._color(el), lw=1.2, ls=ls_of[l],
                                 label=f"{el}-{l}")
                else:
                    line(p.sum(axis=1), color=self._color(el), lw=1.2, label=el)
        dmax = self.dmax_var.get().strip()
        try:
            top = float(dmax) if dmax else ymax * 1.08
        except ValueError:
            self.dmax_var.set("")
            top = ymax * 1.08
        top = top if top > 0 else 1.0
        if horizontal:
            ax.set_xlim(0, top)
            ax.set_ylim(emin, emax)
            ax.axhline(0.0, color="#888888", lw=0.6, ls="--")
            ax.set_xlabel("DOS (states/eV)")
        else:
            ax.set_xlim(emin, emax)
            ax.set_ylim(0, top)
            ax.axvline(0.0, color="#888888", lw=0.6, ls="--")
            ax.set_xlabel("E − E$_F$ (eV)")
            ax.set_ylabel("DOS (states/eV)")
        ax.legend(loc="upper right", fontsize=9)


# ----------------------------------------------------------------------
#  タブ 3: フェルミ面
# ----------------------------------------------------------------------
NO_SPLIT = "なし"


class FermiTab:
    title = "フェルミ面"

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.d = None
        self.cache = {}                    # (バンド, レベル, BZ, 補間) → 三角形
        self.fine, self.fine_key = {}, None     # 補間したバンド
        self.runs, self.wfine, self.vvals, self.wfine_note = [], {}, {}, {}
        self.vsrc, self.vall = {}, None
        self.vbase = {}                    # 三角形ごとの頂点配列（位置と法線。色だけ後で入れる）
        self.vfc = {}                      # 三角形ごとの |v_F|
        self.sel = []
        self.band_vars = {}
        self.elem_vars = {}
        self.bar_photo = None
        self.cur = None                    # いまの色の範囲など（カラーバー・画像保存用）
        side, holder = make_side_layout(app, frame, on_toggle=lambda: self.viewer.fit(self.viewer.view["auto"]))
        self.side = side
        # 描画領域の上: 視点・保存のボタン（電荷密度タブと同じ配置）
        self.toolbar = ttk.Frame(holder, padding=(4, 3, 4, 3))
        self.toolbar.pack(side="top", fill="x")
        self.bar = tk.Label(holder, bd=0, bg="black")
        self.view_holder = ttk.Frame(holder)
        self.view_holder.pack(side="left", fill="both", expand=True)
        self.viewer = Viewer3D(tk, self.view_holder, on_status=self._perf)
        self._bar_h = 0
        self._bar_job = None

        def on_holder_resize(ev):
            # ウィンドウの高さが変わったらカラーバーを作り直す（連続したサイズ変更はまとめて 1 回）
            if abs(ev.height - self._bar_h) > 2 and self.cur is not None:
                if self._bar_job:
                    self.view_holder.after_cancel(self._bar_job)
                self._bar_job = self.view_holder.after(120, self._update_bar)
        self.view_holder.bind("<Configure>", on_holder_resize)

        section(app, side, "① 計算フォルダ / frmsf ファイル")
        fo = ttk.Frame(side)
        fo.pack(anchor="w")
        ttk.Button(fo, text="フォルダを開く…", command=self.choose).pack(side="left")
        ttk.Button(fo, text="frmsf を開く…", command=self.choose_frmsf).pack(side="left", padx=6)
        ttk.Label(side, text="フォルダ: EIGENVAL, KPOINTS, POSCAR, DOSCAR（+ PROCAR）を読みます\n"
                             "frmsf: VASPKIT やこのアプリで作った .frmsf をそのまま表示します",
                  foreground="#666", justify="left").pack(anchor="w")
        self.put_info = info_text(app, side, height=7)

        section(app, side, "フェルミエネルギー")
        fe = ttk.Frame(side)
        fe.pack(anchor="w")
        self.ef_var = tk.StringVar(value="")
        self.shift_var = tk.StringVar(value="0.0")
        for r, (lab, var) in enumerate((("E_F [eV]:", self.ef_var), ("E_F からのシフト [eV]:", self.shift_var))):
            ttk.Label(fe, text=lab).grid(row=r, column=0, sticky="w")
            en = ttk.Entry(fe, textvariable=var, width=10)
            en.grid(row=r, column=1, padx=4, pady=1)
            en.bind("<Return>", lambda ev: self.rebuild(bands_changed=True))
        ttk.Button(fe, text="反映", command=lambda: self.rebuild(bands_changed=True)).grid(row=0, column=2, rowspan=2)
        ttk.Label(side, text="シフト: 半導体でドープ後の等エネルギー面を見るときに使う\n（FermiSurfer の Shift Fermi energy と同じ）",
                  foreground="#666", justify="left").pack(anchor="w")

        section(app, side, "色")
        self.color_var = tk.StringVar(value="vf")
        self.rb_elem = ttk.Radiobutton(side, text="元素・原子の寄与（チェックしたものの重みの和）", value="elem",
                                       variable=self.color_var, command=self._elem_changed)
        self.rb_elem.pack(anchor="w")
        self.elem_frame = ttk.Frame(side)
        self.elem_frame.pack(anchor="w", padx=(18, 0))
        fsp = ttk.Frame(side)
        fsp.pack(anchor="w", padx=(18, 0), pady=(2, 0))
        ttk.Label(fsp, text="原子ごとに分ける元素:").pack(side="left")
        self.split_var = tk.StringVar(value=NO_SPLIT)
        self.split_box = ttk.Combobox(fsp, textvariable=self.split_var, width=6, state="readonly", values=[NO_SPLIT])
        self.split_box.pack(side="left", padx=4)
        self.split_box.bind("<<ComboboxSelected>>", lambda ev: self._split_changed())
        self.atom_frame = ttk.Frame(side)
        self.atom_frame.pack(anchor="w", padx=(18, 0))
        self.atom_vars = {}                # 原子の番号 → チェックの状態（分ける元素の原子だけ）
        self.orb_frame = ttk.Frame(side)
        self.orb_frame.pack(anchor="w", padx=(18, 0), pady=(2, 0))
        self.orb_vars = {}                 # PROCAR の列名（s, py, …）→ チェックの状態
        self.orb_note = ttk.Label(side, text="", foreground="#b05000", wraplength=int(330 * UI_SCALE), justify="left")
        self.orb_note.pack(anchor="w", padx=(18, 0))
        self.pf_lbl = ttk.Label(side, text="", foreground="#555", wraplength=int(330 * UI_SCALE), justify="left")
        self.pf_lbl.pack(anchor="w", padx=(18, 0))
        self.elem_cbs = {}
        self.split_el = None
        ttk.Radiobutton(side, text="フェルミ速度 |v_F|", value="vf", variable=self.color_var,
                        command=self.recolor).pack(anchor="w")
        ttk.Radiobutton(side, text="バンド番号", value="band", variable=self.color_var,
                        command=self.recolor).pack(anchor="w")
        self.rb_file = ttk.Radiobutton(side, text="frmsf に書かれている色のデータ", value="file",
                                       variable=self.color_var, command=self.recolor)
        self.rb_file.pack(anchor="w")
        self.rb_file.state(["disabled"])
        fr = ttk.Frame(side)
        fr.pack(anchor="w", pady=(2, 0))
        self.auto_var = tk.BooleanVar(value=True)
        self.cmin_var = tk.StringVar(value="")
        self.cmax_var = tk.StringVar(value="")
        ttk.Checkbutton(fr, text="色の範囲を自動", variable=self.auto_var, command=self.recolor).grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(fr, text="最小").grid(row=1, column=0)
        e1 = ttk.Entry(fr, textvariable=self.cmin_var, width=9)
        e1.grid(row=1, column=1, padx=2)
        ttk.Label(fr, text="最大").grid(row=1, column=2)
        e2 = ttk.Entry(fr, textvariable=self.cmax_var, width=9)
        e2.grid(row=1, column=3, padx=2)
        for en in (e1, e2):
            en.bind("<Return>", lambda ev: (self.auto_var.set(False), self.recolor()))
        self.thr_on_var = tk.BooleanVar(value=False)
        self.thr_var = tk.StringVar(value="0.02")
        ttk.Checkbutton(fr, text="寄与がしきい値より小さい面を隠す", variable=self.thr_on_var,
                        command=self.recolor).grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        ttk.Label(fr, text="しきい値").grid(row=3, column=0)
        e3 = ttk.Entry(fr, textvariable=self.thr_var, width=9)
        e3.grid(row=3, column=1, padx=2)
        e3.bind("<Return>", lambda ev: (self.thr_on_var.set(True), self.recolor()))
        ttk.Label(side, text="しきい値は「元素・原子の寄与」と「frmsf の色のデータ」の表示で効く",
                  foreground="#666", wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w")

        section(app, side, "バンド（E_F を横切るもの）")
        self.band_frame = ttk.Frame(side)
        self.band_frame.pack(anchor="w")

        section(app, side, "表示")
        self.bz_var = tk.StringVar(value="first")
        fbz = ttk.Frame(side)
        fbz.pack(anchor="w")
        ttk.Radiobutton(fbz, text="第一ブリルアンゾーン", value="first", variable=self.bz_var,
                        command=lambda: self.rebuild(refit=True)).pack(side="left")
        ttk.Radiobutton(fbz, text="逆格子の単位胞", value="prim", variable=self.bz_var,
                        command=lambda: self.rebuild(refit=True)).pack(side="left", padx=6)
        fip = ttk.Frame(side)
        fip.pack(anchor="w", pady=(2, 0))
        ttk.Label(fip, text="補間（k点メッシュを細かくする倍率）:").pack(side="left")
        self.interp_var = tk.StringVar(value="1")
        self.interp_box = ttk.Combobox(fip, textvariable=self.interp_var, width=3, state="readonly",
                                       values=[str(i) for i in range(1, FS_INTERP_MAX + 1)])
        self.interp_box.pack(side="left", padx=4)
        self.interp_box.bind("<<ComboboxSelected>>", lambda ev: self.rebuild())
        ttk.Label(side, text="面のギザギザを減らす。1 = 補間なし。FermiSurfer の Interpolation と同じ 3 次補間を、"
                             "バンドの交差をまたいでつなぎ直してから行う",
                  foreground="#666", wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w")
        self.line_var = tk.BooleanVar(value=True)
        self.bar_var = tk.BooleanVar(value=True)
        self.axes_var = tk.BooleanVar(value=True)
        self.black_var = tk.BooleanVar(value=True)
        for text, var in (("ブリルアンゾーンの線", self.line_var), ("カラーバー", self.bar_var),
                          ("逆格子軸 b1, b2, b3", self.axes_var),
                          ("背景を黒にする（FermiSurfer の既定。オフで白）", self.black_var)):
            ttk.Checkbutton(side, text=text, variable=var, command=self.recolor).pack(anchor="w")

        Bv = lambda i: (lambda: self.d["B"][i])
        view_toolbar(app, self.toolbar, self.viewer, [("b1", Bv(0), Bv(2)), ("b2", Bv(1), Bv(2)), ("b3", Bv(2), Bv(1))],
                     self._home, self.save_image, oblique=FS_OBLIQUE, extra=[(".frmsf を保存…", self.save_frmsf)])

        section(app, side, "frmsf の保存（ボタンは描画領域の上）")
        self.reduce_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="frmsf には E_F を横切るバンドだけ書く", variable=self.reduce_var).pack(anchor="w")
        ttk.Label(side, text="frmsf は VASPKIT 263（元素を選んでいれば 264）と同じ書式。\nFermiSurfer でそのまま開けます",
                  foreground="#666", justify="left").pack(anchor="w")
        self.status = ttk.Label(side, text="", foreground="#333", wraplength=int(330 * UI_SCALE), justify="left")
        self.status.pack(anchor="w", pady=(6, 0))
        self.perf = ttk.Label(side, text="", foreground="#666", wraplength=int(330 * UI_SCALE), justify="left")
        self.perf.pack(anchor="w")

    def _home(self):
        B = self.d["B"]
        return view_matrix(B[0], B[2])

    def _perf(self):
        if not hasattr(self, "perf") or not hasattr(self, "viewer"):
            return
        self.perf.configure(text=f"描画: {self.viewer.engine_text()}\n{self.viewer.fps_text()}")

    # ---- 読み込み ----
    def choose(self):
        # Windows のフォルダ選択画面はファイルを表示しないので、フォルダ内のファイルを選んでもらい、その場所を開く
        p = self.app.filedialog.askopenfilename(
            title="フェルミ面計算フォルダの EIGENVAL（またはフォルダ内のどれか 1 つのファイル）を選択",
            filetypes=[("VASP の出力", ("EIGENVAL", "PROCAR", "KPOINTS", "POSCAR", "CONTCAR", "OUTCAR", "DOSCAR",
                                       "INCAR", "vasprun.xml")),
                       ("All files", "*")])
        if p:
            self.load_dir(os.path.dirname(p))

    def choose_frmsf(self):
        p = self.app.filedialog.askopenfilename(title="frmsf ファイルを選択",
                                                filetypes=[("FermiSurfer", "*.frmsf"), ("All files", "*.*")])
        if p:
            self.load_frmsf(p)

    def _reset_for(self, d):
        """新しいデータ d を表示する準備（キャッシュと、元素・原子・軌道のチェックボックスを消す）"""
        Vb, faces = brillouin_zone(d["B"])
        d["bz"] = (Vb, np.array([Vb[k] for pair in bz_edges(faces) for k in pair], np.float32))
        self.d = d
        self._clear_caches()
        self.ef_var.set(f"{d['efermi']:.6f}")
        self.shift_var.set("0.0")
        # 補間の初期値: メッシュの最も粗い方向が FS_INTERP_TARGET 点以上になる倍率
        f0 = int(np.ceil(FS_INTERP_TARGET / max(1, min(d["mesh"]))))
        self.interp_var.set(str(min(FS_INTERP_MAX, max(1, f0))))
        for fr in (self.elem_frame, self.atom_frame, self.orb_frame):
            for w in fr.winfo_children():
                w.destroy()
        self.elem_vars, self.elem_cbs, self.atom_vars, self.split_el, self.orb_vars = {}, {}, {}, None, {}
        self.orb_note.configure(text="")
        self.pf_lbl.configure(text="")
        self.wfine_note = {}
        self.split_var.set(NO_SPLIT)
        self.color_var.set("vf")

    def _clear_caches(self):
        self.cache, self.vbase, self.vfc, self.sel = {}, {}, {}, []
        self.fine, self.runs, self.fine_key = {}, [], None
        self.wfine, self.vvals = {}, {}          # 選んだ重みの細かい格子での値 / 頂点での値（選び方ごと）
        self.vsrc, self.vall = {}, None

    def load_frmsf(self, path):
        """.frmsf をそのまま表示する。エネルギーは E − E_F で書かれているので E_F = 0 として扱う"""
        app = self.app
        remember_view(app, self)
        ttk = app.ttk
        try:
            with _Busy(app, f"frmsf を読み込み中… {path}"):
                d = read_frmsf(path)
        except Exception as e:
            app.messagebox.showerror("frmsf 読み込みエラー", f"{path}\n\n{e}")
            app.set_msg("")
            return
        self._reset_for(d)
        ttk.Label(self.elem_frame, text="（frmsf から開いたときは使えません）", foreground="#777").grid()
        self.rb_elem.state(["disabled"])
        self.split_box.configure(values=[NO_SPLIT])
        self.split_box.state(["disabled"])
        W = d["frmsf_W"]
        self.rb_file.state(["!disabled"] if W is not None else ["disabled"])
        if W is not None:
            self.color_var.set("file")
        self.side.fit_width()
        m = d["mesh"]
        self.put_info(f"ファイル: {os.path.basename(path)}\n"
                      f"メッシュ: {m[0]} × {m[1]} × {m[2]}（Γ中心）\n"
                      f"バンド数: {d['E'].shape[0]}\n"
                      f"色のデータ: {'なし' if W is None else ('あり' if W.shape[-1] == 1 else f'あり（{W.shape[-1]} 成分。大きさを表示）')}\n"
                      "エネルギーは E − E_F（E_F = 0）")
        self.viewer.view["R"] = FS_OBLIQUE @ self._home()
        self.viewer.scene = None
        app.select_tab(self)
        app.root.update_idletasks()
        app.set_msg(f"frmsf を読み込みました: {path}\n"
                    "frmsf には構造と原子・軌道ごとの重みが入っていないので、元素・原子・軌道の切り替えはできません"
                    "（切り替えるには計算フォルダを開いてください）。")
        self.rebuild(bands_changed=True, refit=True, keep_msg=True)
        app.add_recent("frmsf", path)
        restore_view(app, self, "frmsf:" + os.path.abspath(path))

    STYLE = ("bz_var", "line_var", "bar_var", "axes_var", "black_var", "reduce_var", "thr_var")

    def view_state(self):
        if self.d is None:
            return None
        return {"vars": capture_vars(self), "view": viewer_state(self.viewer)}

    def apply_view_state(self, stt):
        d = self.d
        if d is None or not stt:
            return
        vs = stt.get("vars", {})
        restore_vars(self, vs, skip=("elem_vars", "atom_vars", "orb_vars", "band_vars", "split_var"))
        if d.get("has_procar") and vs.get("split_var") and vs["split_var"] != self.split_var.get():
            self.split_var.set(vs["split_var"])
            self._split_changed(redraw=False)
        restore_vars(self, vs, only=("elem_vars", "atom_vars", "orb_vars", "band_vars"))
        if self.color_var.get() == "elem" and d.get("has_procar"):
            if self._ensure_weights() is None:
                self.color_var.set("vf")
        self.rebuild(bands_changed=True)
        apply_viewer_state(self.viewer, stt.get("view"))

    def load_dir(self, folder):
        """EIGENVAL などを読んで、すぐにフェルミ面を描く。
        PROCAR（数 GB になる）は、元素の寄与を表示するときに初めて読む。"""
        app = self.app
        remember_view(app, self)
        try:
            with _Busy(app, f"フェルミ面のデータを読み込み中… {folder}"):
                d, hit = cached_read(app, "fermi_base", fermi_source_files(folder), None,
                                     lambda: build_fermi_data(folder, use_procar=False))
        except Exception as e:
            app.messagebox.showerror("フェルミ面 読み込みエラー", f"{folder}\n\n{e}")
            app.set_msg("")
            return
        pc = os.path.join(folder, "PROCAR")
        d["folder"] = folder
        d["has_procar"] = os.path.isfile(pc)
        d["orb_names"] = procar_orbital_names(pc) if d["has_procar"] else []
        d["Wir"] = d["Worb"] = None
        self._install_fermi(d, f"フォルダ: {os.path.basename(os.path.normpath(folder))}",
                            f"フェルミ面のデータを読み込みました: {folder}" + ("（キャッシュから）" if hit else ""))
        app.add_recent("fermi", folder)
        restore_view(app, self, "fermi:" + os.path.abspath(folder))

    def _install_fermi(self, d, src_text, first_msg, from_library=False):
        app = self.app
        self._reset_for(d)
        tk, ttk = app.tk, app.ttk
        self.rb_file.state(["disabled"])
        d["labels"] = site_labels(d["st"])
        if d["has_procar"]:
            for k, el in enumerate(d["species"]):
                v = tk.BooleanVar(value=False)
                self.elem_vars[el] = v
                cb = ttk.Checkbutton(self.elem_frame, text=el, variable=v, command=self._elem_changed)
                cb.grid(row=k // 5, column=k % 5, sticky="w", padx=(0, 8))
                self.elem_cbs[el] = cb
            self.rb_elem.state(["!disabled"])
            self.split_box.configure(values=[NO_SPLIT] + list(d["species"]))
            self.split_box.state(["!disabled"])
            self._build_orbitals()
        else:
            ttk.Label(self.elem_frame, text="（PROCAR がありません）", foreground="#777").grid()
            self.rb_elem.state(["disabled"])
            self.split_box.configure(values=[NO_SPLIT])
            self.split_box.state(["disabled"])
        self.side.fit_width()
        m = d["mesh"]
        self.put_info(f"{src_text}\n"
                      f"組成  : {d['st'].formula}（{d['structure_file']}）\n"
                      f"メッシュ: {m[0]} × {m[1]} × {m[2]}（Γ中心）\n"
                      f"既約 k点 {d['nk_ir']} → 全 {m[0] * m[1] * m[2]} 点に展開\n"
                      f"バンド数: {d['E'].shape[0]}\n"
                      f"E_F = {d['efermi']:.4f} eV（{d['efermi_src']}）")
        self.viewer.view["R"] = FS_OBLIQUE @ self._home()
        self.viewer.scene = None
        app.select_tab(self)
        app.root.update_idletasks()
        note = [first_msg]
        complete = d.get("Wir") is not None and d.get("w_loaded") is not None and bool(np.all(d["w_loaded"]))
        if d["has_procar"] and not complete:
            note.append("PROCAR（元素・原子・軌道ごとの寄与）を裏で読み込んでいます。読み込み中も操作できます。")
        if d["note"]:
            note.append("【注意】" + d["note"])
        if d["nspin"] == 2:
            note.append("【注意】ISPIN = 2 の計算です。最初のスピンだけを表示します。")
        app.set_msg("\n".join(note))
        self.rebuild(bands_changed=True, refit=True, keep_msg=True)
        if d.get("st") is not None:
            fd = d.get("folder") or ""
            app.show_structure(d["st"], os.path.join(fd, d.get("structure_file", "")) if fd else "",
                               "フェルミ面タブのフォルダ")
        if d["has_procar"] and not complete:
            self._start_prefetch(d)           # 最初の面を描いてから、PROCAR を裏で読み始める
        elif complete:
            self.pf_lbl.configure(text="元素・原子・軌道ごとの重み: 読み込み済み（ライブラリ）" if from_library else "")

    def level(self):
        ef = fnum(self.ef_var, round(self.d["efermi"], 6))
        return ef, ef + fnum(self.shift_var, 0.0, -50, 50)

    def _split_atoms(self):
        """原子ごとに分ける元素の、原子の番号のリスト"""
        if self.split_el is None:
            return []
        return [i for i, sy in enumerate(self.d["st"].symbols) if sy == self.split_el]

    def _channels(self):
        """面に付けて補間する重みの組（チャンネル）: 元素ごとの和 + 分ける元素の原子 1 個ずつ。
        戻り値: cmat (原子数, チャンネル数)"""
        d = self.d
        st = d["st"]
        atoms = self._split_atoms()
        c = np.zeros((st.natoms, len(d["species"]) + len(atoms)), np.float32)
        for i, sy in enumerate(st.symbols):
            c[i, d["species"].index(sy)] = 1.0
        for j, i in enumerate(atoms):
            c[i, len(d["species"]) + j] = 1.0
        return c

    def _selection(self):
        """チェックされているもの → (チャンネルの番号のリスト, 原子の番号のリスト, 表示名のリスト)"""
        d = self.d
        ne = len(d["species"])
        ch, atoms, names = [], [], []
        for el, v in self.elem_vars.items():
            if v.get() and el != self.split_el:
                ch.append(d["species"].index(el))
                atoms += [i for i, sy in enumerate(d["st"].symbols) if sy == el]
                names.append(el)
        for j, i in enumerate(self._split_atoms()):
            if self.atom_vars[i].get():
                ch.append(ne + j)
                atoms.append(i)
                names.append(d["labels"][i])
        return ch, atoms, names

    def _build_orbitals(self):
        """PROCAR の列（s, py, pz, …）のチェックボックスを作る。最初は全部オン（= 全軌道の和 tot）"""
        tk, ttk = self.app.tk, self.app.ttk
        names = self.d.get("orb_names") or []
        if not names:
            return
        ttk.Label(self.orb_frame, text="軌道（チェックしたものの和）:").grid(row=0, column=0, columnspan=8, sticky="w")
        rows = {}
        for nm in names:
            l = orbital_l_of(nm)
            rows.setdefault(3 if l is None else l, []).append(nm)
        r = 1
        for l in sorted(rows):
            for c, nm in enumerate(rows[l]):
                v = tk.BooleanVar(value=True)
                self.orb_vars[nm] = v
                ttk.Checkbutton(self.orb_frame, text={"x2-y2": "dx2-y2"}.get(nm, nm), variable=v,
                                command=self._orb_changed).grid(row=r + c // 5, column=c % 5, sticky="w", padx=(0, 6))
            r += (len(rows[l]) + 4) // 5
        fb = ttk.Frame(self.orb_frame)
        fb.grid(row=r, column=0, columnspan=8, sticky="w")
        ttk.Button(fb, text="全部", width=5, command=lambda: self._orb_set(None)).pack(side="left")
        for l in sorted(rows):
            if len(rows) > 1:
                ttk.Button(fb, text="spdf"[l] + " だけ", width=6,
                           command=lambda l=l: self._orb_set(l)).pack(side="left", padx=(4, 0))

    def _orb_set(self, l):
        for nm, v in self.orb_vars.items():
            v.set(True if l is None else orbital_l_of(nm) == l)
        self._orb_changed()

    def _orb_mask(self):
        """選んだ軌道（PROCAR の列ごとの True/False）。全部選ばれていれば None（tot を使う）"""
        if not self.orb_vars:
            return None
        m = [v.get() for v in self.orb_vars.values()]
        return None if all(m) else tuple(m)

    def _orb_changed(self):
        d = self.d
        if d is None or not d["has_procar"]:
            return
        self.color_var.set("elem")
        if self._ensure_weights() is not None:
            self.recolor()

    def _orbital_note(self, fw):
        if fw is None or fw.orb is None or fw.exact:
            self.orb_note.configure(text="")
        else:
            self.orb_note.configure(
                text="【注意】選んだ軌道の組は、この結晶の対称操作で他の軌道と混ざります。既約 k点からの展開は"
                     "近似です（軌道どうしの干渉項が PROCAR にないため）。p 全体・d 全体のように回転で閉じた組を"
                     "選ぶか、ISYM = -1 で全 k点を計算すると正確になります。")

    def _selected_fine(self, mode):
        """いま選んでいる重み（元素・原子・軌道の和、または frmsf の色のデータ）を、面を作った細かい格子に
        補間したもの。戻り値 [(キー, {バンド: 1 次元の値})]（frmsf の色のデータが複数成分なら成分ごと）。
        選び方ごとに結果を覚えておくので、前に選んだ組み合わせに戻ったときは計算しない。"""
        d = self.d
        ip = self.fine_key[1] if self.fine_key else 1

        class _Idx:
            def __init__(self, f):
                self.f = f

            def __getitem__(self, b):
                return self.f(b)

        jobs = []
        if mode != "file":
            self._ensure_weights()
        if mode == "file":
            Wfile = d["frmsf_W"]
            for c in range(Wfile.shape[-1]):
                jobs.append((("file", c), _Idx(lambda b, c=c: Wfile[b][..., c])))
        else:
            _, atoms, _ = self._selection()
            mask = self._orb_mask()
            key = ("elem", tuple(sorted(atoms)), mask)
            if key not in self.wfine:
                col = np.zeros((d["st"].natoms, 1), np.float32)
                col[atoms, 0] = 1.0
                fw = FermiWeights(d, col, mask)
                self._orbital_note(fw)
                self.wfine_note[key] = fw.exact
                jobs.append((key, _Idx(lambda b, fw=fw: fw[b][..., 0])))
            else:
                self._orbital_note(None if self.wfine_note.get(key, True) else _NotExact())
                jobs.append((key, None))
        out = []
        for key, getter in jobs:
            if key not in self.wfine:
                if len(self.wfine) >= 8:                     # 覚えておくのは最近の 8 通りまで
                    old = next(iter(self.wfine))
                    self.wfine.pop(old)
                    self.vvals = {k: v for k, v in self.vvals.items() if k[0] != old}
                self.wfine[key] = refine_weights(self.runs, getter, ip)
            out.append((key, self.wfine[key]))
        return out

    def _vertex_vals(self, items, mode):
        """選んだ重みの、各三角形の頂点での値（バンドごとのリスト）"""
        fines = self._selected_fine(mode)
        vals = []
        for b, D in items:
            comps = []
            for key, wf in fines:
                vk = (key, id(D))
                if vk not in self.vvals:
                    if id(D) not in self.vsrc:                  # 頂点の格子番号と重みは 1 度だけ取り出す
                        self.vsrc[id(D)] = vertex_src(D)
                    self.vvals[vk] = vertex_values(D, wf[b], self.vsrc[id(D)])
                comps.append(self.vvals[vk])
            vals.append(comps[0] if len(comps) == 1 else np.sqrt(np.sum([c * c for c in comps], axis=0)))
        return vals

    # ---- PROCAR の先読み（別スレッド）----
    def _start_prefetch(self, d):
        """PROCAR（軌道ごとの重みと tot）を別スレッドで読む。画面は操作できるまま。
        1) 前に読んだことがあればキャッシュから（すぐ終わる）
        2) なければ、まず E_F 付近のバンドだけを読み（ファイルの位置を計算して必要な所だけ読む）、
           読めたらすぐ色を付けられるようにする
        3) 続けて残りのバンドを裏で読み、読み終わったらキャッシュに保存する"""
        import threading
        app = self.app
        folder = d["folder"]
        files = fermi_source_files(folder) + [os.path.join(folder, "PROCAR")]
        store = getattr(app, "store", None)
        if store is not None:
            cb = store.cache_get("fermi_procar", files)
            if cb is not None:
                try:
                    apply_fermi_weights_bundle(d, cb)
                    self.pf_lbl.configure(text="PROCAR: 前回読んだ内容（キャッシュ）を使いました。切り替えはすぐ反映されます。")
                    if self.color_var.get() == "elem":
                        self.recolor()
                    return
                except Exception:
                    pass
        nb = d["E"].shape[0]
        cross = crossing_bands(d["E"], d["efermi"])
        if not cross:                          # E_F を横切るバンドがなければ、E_F に近いバンド
            gap = np.abs(d["E"].reshape(nb, -1) - d["efermi"]).min(axis=1)
            cross = [int(np.argmin(gap))]
        win = list(range(max(0, min(cross) - 4), min(nb, max(cross) + 5)))
        st = {"n": 0, "tot": 0, "err": None, "done": False, "phase": 1, "nwin": len(win), "nb": nb}

        class _Cancel(Exception):
            pass

        def prog(n, tot):
            if self.d is not d:               # 別のフォルダを開いたら読むのをやめる
                raise _Cancel()
            st["n"], st["tot"] = n, tot

        def work():
            try:
                load_fermi_weights(d, progress=prog, bands=win)
                st["phase"] = 2
                load_fermi_weights(d, progress=prog)
                if store is not None:
                    store.cache_put("fermi_procar", files, None, fermi_weights_bundle(d), background=False)
            except _Cancel:
                pass
            except Exception as e:
                st["err"] = e
            finally:
                st["done"] = True

        d["_prefetch"] = st
        threading.Thread(target=work, daemon=True).start()
        app.root.after(300, lambda: self._poll_prefetch(d))

    def _prefetch_text(self, st):
        what = (f"E_F 付近のバンド {st['nwin']} 本" if st["phase"] == 1 else f"残りのバンド（全 {st['nb']} 本）")
        if st["tot"]:
            return f"PROCAR: {what}を読み込み中… k点 {st['n']} / {st['tot']}"
        return f"PROCAR: {what}を読み込み中…"

    def _poll_prefetch(self, d):
        if self.d is not d:
            return
        st = d["_prefetch"]
        if not st["done"]:
            tail = "（裏で読み込み中。操作はできます）"
            if st["phase"] == 2:
                tail = "（E_F 付近は読み込み済みで、色はすぐ付けられます）"
                if not st.get("shown_ready"):
                    st["shown_ready"] = True
                    if self.color_var.get() == "elem":
                        self.recolor()
            self.pf_lbl.configure(text=self._prefetch_text(st) + tail)
            self.app.root.after(300, lambda: self._poll_prefetch(d))
            return
        if st["err"] is not None:
            self.pf_lbl.configure(text="")
            self._procar_failed(st["err"])
            return
        self.pf_lbl.configure(text="PROCAR の読み込み完了（次回からはキャッシュから読みます）: "
                                   "元素・原子・軌道の切り替えはすぐ反映されます。")
        if self.color_var.get() == "elem":
            self.recolor()

    def _procar_failed(self, e):
        d = self.d
        self.app.messagebox.showerror("PROCAR 読み込みエラー", str(e))
        d["has_procar"] = False
        self.color_var.set("vf")
        self.rb_elem.state(["disabled"])
        self.split_box.state(["disabled"])
        self.recolor()

    def _needed_bands(self):
        """いま色を付けるのに必要なバンド（補間に使う前後のバンドを含む）"""
        need = set(self.band_vars.keys())
        for r in getattr(self, "runs", []) or []:
            for mem in r["members"]:
                need.update(mem)
        return sorted(need)

    def _ensure_weights(self, bands=None):
        """必要なバンドの PROCAR の重みが読めていなければ、読めるまで待つ（先読みがなければここで読む）。
        bands: バンド番号のリスト（None ならいま表示に必要なもの、"all" なら全部）。戻り値: "ok" / None（失敗）"""
        import time as _time
        d = self.d
        app = self.app
        nb = d["E"].shape[0]
        need = list(range(nb)) if bands == "all" else (self._needed_bands() if bands is None else list(bands))

        def ready():
            wl = d.get("w_loaded")
            return d.get("Wir") is not None and wl is not None and bool(np.all(wl[need])) if need else d.get("Wir") is not None

        if ready():
            return "ok"
        st = d.get("_prefetch")
        if st is not None and not st["done"]:
            with _Busy(app, self._prefetch_text(st)):
                while not st["done"] and not ready():
                    app.set_msg(self._prefetch_text(st) + "（必要なバンドを読み終わったら表示します）")
                    app.root.update_idletasks()
                    _time.sleep(0.1)
            if st["err"] is not None:
                self._procar_failed(st["err"])
                return None
            if ready():
                return "ok"
        try:
            with _Busy(app, "PROCAR を読み込み中…（大きなファイルは時間がかかります）"):
                load_fermi_weights(d, progress=lambda n, tot: (
                    app.set_msg(f"PROCAR を読み込み中… k点 {n} / {tot}"), app.root.update_idletasks()),
                    bands=need or None)
        except Exception as e:
            self._procar_failed(e)
            return None
        return "ok"

    # ---- ライブラリ ----
    def library_payload(self):
        d = self.d
        if d is None:
            return None
        if d.get("has_procar") and d.get("st") is not None and self._ensure_weights("all") is None:
            return None
        obj = {k: v for k, v in d.items() if not k.startswith("_") and k not in ("bz",) and not k.startswith("nrm_")}
        obj["view_state"] = self.view_state()
        if d.get("st") is not None:
            form = d["st"].formula
            src = d.get("folder", "")
            name = f"{form} フェルミ面 {os.path.basename(os.path.normpath(src))}"
        else:
            form = "frmsf"
            src = d.get("frmsf_path", "")
            name = f"frmsf {os.path.basename(src)}"
        return "fermi", obj, name, {"formula": form, "source": src}

    def library_open(self, obj, meta):
        remember_view(self.app, self)
        self._library_open(obj, meta)
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), obj.get("view_state"))

    def _library_open(self, obj, meta):
        d = dict(obj)
        d.pop("view_state", None)
        if d.get("st") is None:                # frmsf から開いたもの
            self._reset_for(d)
            self.rb_elem.state(["disabled"])
            self.split_box.state(["disabled"])
            W = d.get("frmsf_W")
            self.rb_file.state(["!disabled"] if W is not None else ["disabled"])
            if W is not None:
                self.color_var.set("file")
            self.put_info(f"ライブラリ: {meta.get('name', '')}")
            self.viewer.view["R"] = FS_OBLIQUE @ self._home()
            self.viewer.scene = None
            self.app.select_tab(self)
            self.rebuild(bands_changed=True, refit=True)
            return
        if d.get("Wir") is not None:
            d["w_loaded"] = np.ones(d["E"].shape[0], bool)
        self._install_fermi(d, f"ライブラリ: {meta.get('name', '')}", f"ライブラリから開きました: {meta.get('name', '')}",
                            from_library=True)

    def _elem_changed(self):
        """元素・原子の寄与の表示に切り替える。PROCAR をまだ読んでいなければここで読む"""
        d = self.d
        if d is None:
            return
        if not d["has_procar"]:
            self.color_var.set("vf")
            return
        self.color_var.set("elem")
        if self._ensure_weights() is not None:
            self.recolor()                    # 面の形は変わらないので、色だけ塗り直す

    def _split_changed(self, redraw=True):
        """「原子ごとに分ける元素」が変わった: その元素の原子 1 個ずつのチェックボックスを作る"""
        d = self.d
        if d is None or not d["has_procar"]:
            return
        tk, ttk = self.app.tk, self.app.ttk
        el = self.split_var.get()
        self.split_el = el if el in d["species"] else None
        for w in self.atom_frame.winfo_children():
            w.destroy()
        self.atom_vars = {}
        for name, cb in self.elem_cbs.items():                    # 分ける元素は、元素単位のチェックを使わない
            cb.state(["disabled"] if name == self.split_el else ["!disabled"])
        if self.split_el is not None:
            self.elem_vars[self.split_el].set(False)
            atoms = self._split_atoms()
            for j, i in enumerate(atoms):
                v = tk.BooleanVar(value=False)
                self.atom_vars[i] = v
                ttk.Checkbutton(self.atom_frame, text=d["labels"][i], variable=v, command=self._elem_changed).grid(
                    row=j // 4, column=j % 4, sticky="w", padx=(0, 8))
            r0 = (len(atoms) + 3) // 4
            fb = ttk.Frame(self.atom_frame)
            fb.grid(row=r0, column=0, columnspan=4, sticky="w")
            ttk.Button(fb, text="全部", command=lambda: self._atoms_all(True)).pack(side="left")
            ttk.Button(fb, text="なし", command=lambda: self._atoms_all(False)).pack(side="left", padx=4)
            ttk.Label(self.atom_frame, text="番号は POSCAR の順（電荷密度タブのラベルと同じ）", foreground="#666").grid(
                row=r0 + 1, column=0, columnspan=4, sticky="w")
        self.side.fit_width()
        if not redraw:
            return
        self.color_var.set("elem")
        if self._ensure_weights() is not None:
            self.recolor()

    def _atoms_all(self, on):
        for v in self.atom_vars.values():
            v.set(on)
        self._elem_changed()

    # ---- 三角形の計算と色付け ----
    def rebuild(self, bands_changed=False, refit=False, keep_msg=False):
        d = self.d
        if d is None:
            return
        tk, ttk = self.app.tk, self.app.ttk
        ef, lv = self.level()
        if bands_changed:
            cross = crossing_bands(d["E"], lv)
            old = {b: v.get() for b, v in self.band_vars.items()}
            for w in self.band_frame.winfo_children():
                w.destroy()
            self.band_vars = {}
            for k, b in enumerate(cross):
                v = tk.BooleanVar(value=old.get(b, True))
                self.band_vars[b] = v
                ttk.Checkbutton(self.band_frame, text=f"{b + 1}", variable=v, command=self.rebuild).grid(
                    row=k // 6, column=k % 6, sticky="w", padx=(0, 6))
            if not cross:
                ttk.Label(self.band_frame, text="（なし）", foreground="#777").grid()
        first = self.bz_var.get() == "first"
        try:
            ip = max(1, min(FS_INTERP_MAX, int(self.interp_var.get())))
        except ValueError:
            ip = 1
        while ip > 1 and int(np.prod(d["mesh"])) * ip ** 3 > 30_000_000:      # メモリを使いすぎない範囲に抑える
            ip -= 1
        self.interp_var.set(str(ip))
        sel = [b for b, v in self.band_vars.items() if v.get()]
        try:
            with _Busy(self.app, "フェルミ面を計算中…") if not keep_msg else _NoBusy():
                # 補間: E_F を横切るバンドをまとめて、交差を考慮して細かいメッシュにする（結果は使い回す）
                cross_all = sorted(self.band_vars.keys())
                # 面の形はエネルギーだけで決まる。元素・原子・軌道の重みは、面を作ったあとで頂点ごとに
                # 求める（_vertex_vals）ので、選び方を変えても面は作り直さない
                fkey = (tuple(cross_all), ip)
                if self.fine_key != fkey:
                    self.fine, self.runs = refine_bands_plan(d["E"], cross_all, ip)
                    self.fine_key = fkey
                    self.cache, self.vbase, self.vfc, self.wfine, self.vvals = {}, {}, {}, {}, {}
                    self.vsrc, self.vall = {}, None
                out = []
                for b in sel:
                    Ef = self.fine[b]
                    # どの k点でも縮退しているバンド（SOC + 反転対称の 2 重縮退など）は同じ配列を共有しているので、
                    # 面の計算は 1 回だけ行う
                    key = (id(Ef), round(lv, 9), first)
                    if key not in self.cache:
                        if len(self.cache) > 24:
                            self.cache, self.vbase, self.vfc, self.vvals = {}, {}, {}, {}
                            self.vsrc, self.vall = {}, None
                        self.cache[key] = fermi_band_mesh(Ef, lv, d["B"], None, first, src=True)
                    out.append((b, self.cache[key]))
        except MemoryError as e:
            self.app.messagebox.showwarning("フェルミ面", str(e))
            return
        self.sel = out
        self.recolor(refit=refit)
        uniq = {id(D): D for _, D in self.sel}
        nt = sum(len(D) for D in uniq.values())
        txt = (f"表示レベル: E = {lv:.4f} eV（E_F {ef:.4f} + シフト {lv - ef:+.4f}）\n"
               f"横切るバンド: {', '.join(str(b + 1) for b in self.band_vars) or 'なし'}   三角形 {nt} 枚\n"
               f"補間 ×{ip}（面を作るメッシュ {d['mesh'][0] * ip} × {d['mesh'][1] * ip} × {d['mesh'][2] * ip}）")
        if len(uniq) < len(self.sel):
            txt += f"\n縮退して重なるバンドは 1 枚にまとめて描いています（{len(self.sel)} 本 → {len(uniq)} 枚）。"
        if not self.band_vars:
            txt += "\n【注意】このエネルギーを横切るバンドがありません（半導体・絶縁体では、シフトを入れると等エネルギー面を表示できます）。"
        self.status.configure(text=txt)
        if not keep_msg:
            self.app.set_msg(txt)

    def recolor(self, refit=False):
        d = self.d
        if d is None:
            return
        mode = self.color_var.get()
        if (mode == "elem" and d["Wir"] is None) or (mode == "file" and d.get("frmsf_W") is None):
            mode = "vf"
            self.color_var.set("vf")
        # 重なっているバンドは 1 回だけ描く
        items, seen = [], set()
        for b, D in self.sel:
            if id(D) not in seen and len(D):
                seen.add(id(D))
                items.append((b, D))
        vals, title, nosel = [], "", False
        if mode == "elem":
            _, atoms, chosen = self._selection()
            title = "+".join(chosen) if chosen else "(none)"
            mask = self._orb_mask()
            if mask is not None and chosen:
                olab = orbital_label(d["orb_names"], mask)
                title = f"{title}:{olab or '-'}"
                if len(title) > 17:           # カラーバーの幅に収まる長さにする
                    title = title[:15] + ".."
            nosel = not atoms or (mask is not None and not any(mask))
            if not nosel:
                vals = self._vertex_vals(items, "elem")
            else:
                self.orb_note.configure(text="")
        elif mode == "file":
            vals = self._vertex_vals(items, "file")
            title = "frmsf"
        elif mode == "vf":
            for b, D in items:
                if id(D) not in self.vfc:
                    self.vfc[id(D)] = np.linalg.norm(D[:, :, 3:6], axis=2) * np.float32(EVANG_TO_MS)
                vals.append(self.vfc[id(D)])
            title = "|v_F| (m/s)"
        vmin = min((float(v.min()) for v in vals if v.size), default=0.0)
        vmax = max((float(v.max()) for v in vals if v.size), default=0.0)
        if mode != "band":
            if nosel:
                pass                          # 何も選ばれていないときは、入力欄の範囲を書き換えない
            elif self.auto_var.get():
                self.cmin_var.set(f"{vmin:.4g}")
                self.cmax_var.set(f"{vmax:.4g}")
            else:
                vmin = fnum(self.cmin_var, float(f"{vmin:.4g}"))
                vmax = fnum(self.cmax_var, float(f"{vmax:.4g}"))
        flat = not (vmax - vmin > 1e-5 * max(abs(vmax), abs(vmin), 1e-30))       # 値がどこでも同じ
        span = 1.0 if flat else vmax - vmin
        black = self.black_var.get()
        bg = (0.0, 0.0, 0.0) if black else (1.0, 1.0, 1.0)
        fg = (1.0, 1.0, 1.0) if black else (0.0, 0.0, 0.0)
        sc = empty_scene(bg)
        parts = []
        thr = fnum(self.thr_var, 0.0) if (mode in ("elem", "file") and not nosel and self.thr_on_var.get()) else None
        n_all = n_hid = 0
        nb = len(self.band_vars)
        order = list(self.band_vars.keys())
        # 位置と法線は 1 度だけ計算し、全バンドを 1 つの配列にまとめておく（以後は色だけを書き換える）
        sig = tuple(id(D) for _, D in items)
        if self.vall is None or self.vall[0] != sig:
            for b, D in items:
                if id(D) not in self.vbase:
                    self.vbase[id(D)] = mesh_vertices(D, (1.0, 1.0, 1.0, 1.0))
            arrs = [self.vbase[id(D)] for _, D in items]
            allv = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 10), np.float32)
            offs = np.concatenate([[0], np.cumsum([len(a) for a in arrs])])
            self.vall = (sig, allv, offs)
        _, allv, offs = self.vall
        whole = True                          # 全部の三角形をそのまま描くか（しきい値・非表示がなければ True）
        for k, (b, D) in enumerate(items):
            base = allv[offs[k]:offs[k + 1]]
            v3 = base.reshape(-1, 3, 10)
            if mode == "band":
                v3[:, :, 6:9] = bgr_colors(0.5 if nb <= 1 else order.index(b) / (nb - 1))
            elif nosel:                       # 元素・原子・軌道が何も選ばれていない: 面を描かない
                whole = False
                continue
            elif flat:
                v3[:, :, 6:9] = bgr_colors(0.5)
            else:
                v3[:, :, 6:9] = bgr_colors((vals[k] - vmin) / span)
            if thr is not None:               # 三角形の 3 頂点の平均がしきい値より小さいものは描かない
                keep = vals[k].mean(axis=1) >= thr
                n_all += len(keep)
                n_hid += int(len(keep) - keep.sum())
                if not keep.all():
                    base = np.ascontiguousarray(v3[keep]).reshape(-1, 10)
                    whole = False
            if len(base):
                parts.append(base)
        if whole:
            sc["mesh"] = allv if len(allv) else None
        else:
            sc["mesh"] = (parts[0] if len(parts) == 1 else np.vstack(parts)) if parts else None
        if thr is not None and n_all:
            self.app.set_msg(f"しきい値 {thr:g} より寄与が小さい面を隠しています: 三角形 {n_all} 枚のうち {n_hid} 枚"
                             f"（{100.0 * n_hid / n_all:.1f} %）")
        B = d["B"]
        if self.bz_var.get() == "first":
            Vb, edges = d["bz"]
            sc["center"], sc["fit"] = np.zeros(3), Vb
        else:
            edges = box_edges(B)
            sc["center"] = B.sum(axis=0) / 2
            sc["fit"] = edges
        sc["radius"] = float(np.linalg.norm(np.asarray(sc["fit"]) - sc["center"], axis=1).max())
        if self.line_var.get():
            # 枠は、手前の辺を実線、奥の辺を破線で描く（線画だけだと手前と奥の区別がつかないため）
            key = "nrm_" + self.bz_var.get()
            if key not in d:
                d[key] = edge_face_normals(edges)
            sc["edge_faces"] = (np.asarray(edges, float).reshape(-1, 2, 3), d[key], fg, 1.0)
        if self.axes_var.get():
            sc["triad"] = (B, ("b1", "b2", "b3"))
        self.cur = {"mode": "band" if nosel else mode, "vmin": vmin, "vmax": vmax, "title": title, "bg": bg}
        self.viewer.set_scene(sc, refit=True if refit else None)
        self._update_bar()
        self._perf()

    def _update_bar(self):
        c = self.cur
        show = self.bar_var.get() and c is not None and c["mode"] != "band"
        if not show:
            self.bar.pack_forget()
            return
        from PIL import ImageTk
        h = max(self.view_holder.winfo_height(), 300)
        self._bar_h, self._bar_job = h, None
        img = colorbar_image(int(104 * UI_SCALE), h, c["vmin"], c["vmax"], c["title"], c["bg"], UI_SCALE)
        self.bar_photo = ImageTk.PhotoImage(img, master=self.bar)
        self.bar.configure(image=self.bar_photo, bg="black" if sum(c["bg"]) < 1.5 else "white")
        if not self.bar.winfo_ismapped():
            self.bar.pack(side="right", fill="y", before=self.view_holder)

    # ---- 保存 ----
    def save_frmsf(self):
        d = self.d
        if d is None:
            return
        ef, lv = self.level()
        _, atoms, chosen = self._selection()
        mask = self._orb_mask()
        if mask is not None and (not any(mask) or d.get("Worb") is None):
            mask = None if any(mask) else ()
        with_w = self.color_var.get() == "elem" and d["Wir"] is not None and bool(atoms) and mask != ()
        from_file = self.color_var.get() == "file" and d.get("frmsf_W") is not None
        olab = orbital_label(d["orb_names"], mask) if (with_w and mask) else ""
        name = "FERMISURFACE" + (("_" + "_".join(chosen)) if with_w else "") + (("_" + olab) if olab else "") + ".frmsf"
        if with_w and self._ensure_weights("all") is None:      # frmsf には全部のバンドの重みが要る
            return
        p = self.app.filedialog.asksaveasfilename(title="frmsf を保存", initialdir=d["folder"], initialfile=name,
                                                  defaultextension=".frmsf",
                                                  filetypes=[("FermiSurfer", "*.frmsf"), ("All files", "*.*")])
        if not p:
            return
        bands = crossing_bands(d["E"], lv) if self.reduce_var.get() else None
        if bands is not None and not bands:
            self.app.messagebox.showwarning("frmsf", "E_F を横切るバンドがありません。")
            return
        W = None
        if with_w:
            col = np.zeros((d["st"].natoms, 1), np.float32)
            col[atoms, 0] = 1.0
            W = FermiWeights(d, col, mask if mask else None)   # 選んだ原子（と軌道）の重みの和をメッシュ全体に展開
        elif from_file:
            Wf = d["frmsf_W"]
            W = Wf[..., 0] if Wf.shape[-1] == 1 else np.linalg.norm(Wf, axis=-1)
        try:
            with _Busy(self.app, "frmsf を書き出し中…"):
                write_frmsf(p, d["mesh"], d["B"], d["E"] - ef, W, bands)
        except Exception as e:
            self.app.messagebox.showerror("frmsf", str(e))
            return
        self.app.set_msg(f"frmsf を保存しました: {p}\n"
                         f"エネルギーの原点 E_F = {ef:.6f} eV（シフトは含めていません。FermiSurfer 側の "
                         "Shift Fermi energy で指定してください）"
                         + (f"\n色のデータ: {'+'.join(chosen)}{'（軌道 ' + olab + '）' if olab else ''} の重みの和。"
                            "元素全体・全軌道を選んだ場合は VASPKIT 264 の "
                            "FERMISURFACE_tot.frmsf と同じ値です。原子を 1 個ずつ選んだ場合は、対称操作による原子の"
                            "入れ替えを考慮しているので VASPKIT 264 とは一致しません（VASPKIT は等価な原子があると"
                            "原子 1 個の重みを正しく展開しません）。" if with_w else ""))

    def save_image(self):
        img = self.viewer.grab()
        if img is None:
            return
        p = self.app.filedialog.asksaveasfilename(title="画像を保存", defaultextension=".png",
                                                  filetypes=[("PNG", "*.png")])
        if not p:
            return
        from PIL import Image
        img = img.convert("RGB")
        c = self.cur
        if self.bar_var.get() and c is not None and c["mode"] != "band":
            k = img.height / max(self.viewer.size()[1], 1)
            bar = colorbar_image(int(104 * UI_SCALE * k), img.height, c["vmin"], c["vmax"], c["title"], c["bg"],
                                 UI_SCALE * k)
            out = Image.new("RGB", (img.width + bar.width, img.height))
            out.paste(img, (0, 0))
            out.paste(bar, (img.width, 0))
            img = out
        img.save(p)
        self.app.set_msg(f"画像を保存しました: {p}")


class _NotExact:
    """_orbital_note に「近似」を伝えるための目印"""
    orb = True
    exact = False


class _NoBusy:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------
#  タブ 4: 輸送特性
# ----------------------------------------------------------------------
class TransportTab:
    title = "輸送特性"
    QUANT = [("seebeck", "ゼーベック係数 S", "S (μV/K)"),
             ("sigma", "電気伝導率 σ", "σ (S/m)"),
             ("kappa_e", "電子熱伝導率 κe", "κ$_e$ (W m$^{-1}$ K$^{-1}$)")]

    def __init__(self, app, frame):
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.d = None
        self.inst_vars = []
        self.series = []                   # 直近に描いた系列（CSV 保存用）
        side, holder = make_side_layout(app, frame)
        self.side = side
        self.plot = PlotArea(app, holder)
        self.plot.message("vaspout.h5 を読み込んでください")

        section(app, side, "① vaspout.h5（電子フォノン輸送）")
        ttk.Button(side, text="vaspout.h5 を開く…", command=self.choose).pack(anchor="w")
        self.put_info = info_text(app, side, height=5)

        section(app, side, "方向")
        self.aniso_var = tk.BooleanVar(value=False)
        ttk.Label(side, text="平均（(xx+yy+zz)/3）は常に表示", foreground="#555").pack(anchor="w")
        ttk.Checkbutton(side, text="異方性（a, b, c 軸方向）を表示", variable=self.aniso_var,
                        command=self.draw).pack(anchor="w")
        fa = ttk.Frame(side)
        fa.pack(anchor="w", padx=(18, 0))
        self.axis_vars = {}
        for k, ax in enumerate("abc"):
            v = tk.BooleanVar(value=True)
            self.axis_vars[ax] = v
            tk.Label(fa, text="━", fg=AXIS_COLORS[ax]).grid(row=0, column=2 * k)
            ttk.Checkbutton(fa, text=f"{ax} 軸", variable=v, command=self.draw).grid(row=0, column=2 * k + 1,
                                                                                    padx=(0, 8))

        section(app, side, "表示する量")
        self.q_vars = {}
        for key, lab, _ in self.QUANT:
            v = tk.BooleanVar(value=True)
            self.q_vars[key] = v
            ttk.Checkbutton(side, text=lab, variable=v, command=self.draw).pack(anchor="w")
        self.logs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="σ を対数軸にする", variable=self.logs_var, command=self.draw).pack(anchor="w")

        section(app, side, "横軸")
        self.x_var = tk.StringVar(value="T")
        fx = ttk.Frame(side)
        fx.pack(anchor="w")
        ttk.Radiobutton(fx, text="温度", value="T", variable=self.x_var, command=self.draw).grid(row=0, column=0,
                                                                                                sticky="w")
        self.rb_mu = ttk.Radiobutton(fx, text="キャリア密度 / 化学ポテンシャル", value="mu", variable=self.x_var,
                                     command=self.draw)
        self.rb_mu.grid(row=1, column=0, sticky="w")
        ttk.Label(fx, text="温度 [K]:").grid(row=1, column=1, padx=(8, 2))
        self.t_var = tk.StringVar(value="")
        self.t_box = ttk.Combobox(fx, textvariable=self.t_var, width=8, state="readonly", values=[])
        self.t_box.grid(row=1, column=2)
        self.t_box.bind("<<ComboboxSelected>>", lambda ev: self.draw())

        section(app, side, "データ（散乱近似・キャリア密度の組み合わせ）")
        self.inst_frame = ttk.Frame(side)
        self.inst_frame.pack(anchor="w")
        fb = ttk.Frame(side)
        fb.pack(anchor="w")
        ttk.Button(fb, text="全部", command=lambda: self._all(True)).pack(side="left")
        ttk.Button(fb, text="なし", command=lambda: self._all(False)).pack(side="left", padx=4)
        ttk.Button(side, text="表示中のデータを CSV で保存…", command=self.save_csv).pack(anchor="w", pady=(8, 0))
        ttk.Label(side, text="画像の保存・拡大・移動は、グラフ下のツールバーで行います。",
                  foreground="#666", wraplength=int(330 * UI_SCALE), justify="left").pack(anchor="w", pady=(6, 0))

    def choose(self):
        p = self.app.filedialog.askopenfilename(title="vaspout.h5 を選択",
                                                filetypes=[("HDF5", "*.h5"), ("All files", "*.*")])
        if p:
            self.load(p)

    def load(self, path):
        app = self.app
        remember_view(app, self)
        try:
            with _Busy(app, f"vaspout.h5 を読み込み中… {path}"):
                d = read_transport_h5(path)
        except Exception as e:
            app.messagebox.showerror("vaspout.h5 読み込みエラー", f"{path}\n\n{e}")
            app.set_msg("")
            return
        d["path"] = path
        self._install_tr(d, f"ファイル: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
                         f"輸送特性を読み込みました: {path}")
        app.add_recent("transport", path)
        restore_view(app, self, "tr:" + os.path.abspath(path))

    STYLE = ("aniso_var", "axis_vars", "q_vars", "logs_var", "x_var")

    def view_state(self):
        if self.d is None:
            return None
        return {"vars": capture_vars(self)}

    def apply_view_state(self, stt):
        restore_vars(self, stt.get("vars"))
        self.draw()

    def library_payload(self):
        d = self.d
        if d is None:
            return None
        comp = "".join(f"{s}{n}" for s, n in d["species"]) or "transport"
        d = dict(d, view_state=self.view_state())
        return "transport", d, f"{comp} 輸送特性 {os.path.basename(os.path.dirname(d.get('path', '')))}", \
            {"formula": comp, "source": d.get("path", "")}

    def library_open(self, obj, meta):
        remember_view(self.app, self)
        obj = dict(obj)
        vs = obj.pop("view_state", None)
        self._install_tr(obj, f"ライブラリ: {meta.get('name', '')}", f"ライブラリから開きました: {meta.get('name', '')}")
        restore_view(self.app, self, "lib:" + str(meta.get("_id", meta.get("name", ""))), vs)

    def _install_tr(self, d, src_text, first_msg):
        app = self.app
        tk, ttk = app.tk, app.ttk
        self.d = d
        for w in self.inst_frame.winfo_children():
            w.destroy()
        self.inst_vars = []
        for k, ins in enumerate(d["instances"]):
            v = tk.BooleanVar(value=(k == 0))
            self.inst_vars.append(v)
            ttk.Checkbutton(self.inst_frame, text=transport_label(ins, d["mu_tag"]), variable=v,
                            command=self.draw).pack(anchor="w")
        self.side.fit_width()
        temps = sorted({float(t) for ins in d["instances"] for t in ins["temps"]})
        self.t_box.configure(values=[f"{t:g}" for t in temps])
        if temps:
            self.t_var.set(f"{min(temps, key=lambda t: abs(t - 300)):g}")
        nmu = 0 if d["mu_values"] is None else len(set(np.round(d["mu_values"], 12)))
        self.rb_mu.state(["!disabled"] if nmu >= 2 else ["disabled"])
        if nmu < 2:
            self.x_var.set("T")
        comp = " ".join(f"{s}{n}" for s, n in d["species"])
        lat = d["lattice"]
        if lat is not None:
            la = np.linalg.norm(lat, axis=1)
            cell = f"a,b,c : {la[0]:.4f} {la[1]:.4f} {la[2]:.4f} Å"
        else:
            cell = "格子ベクトルが読めないため、a, b, c の代わりに x, y, z 方向を表示します"
        self.put_info(f"{src_text}\n"
                      f"組成  : {comp or '—'}\n{cell}\n"
                      f"データ数: {len(d['instances'])}   温度 {len(temps)} 点"
                      + (f"（{temps[0]:g}–{temps[-1]:g} K）" if temps else ""))
        app.select_tab(self)
        app.set_msg(f"{first_msg}\n"
                    "a, b, c 軸方向の値は、テンソル T（デカルト座標）と軸方向の単位ベクトル n から n·T·n で求めています。"
                    "平均は (T_xx + T_yy + T_zz) / 3 です。\n"
                    f"ゼーベック係数: ファイル内の値を {d['seebeck_unit_in_file']} として読み、μV/K で表示しています。")
        self.draw()

    def _all(self, on):
        for v in self.inst_vars:
            v.set(on)
        self.draw()

    def directions(self):
        """[(キー, 名前, 単位ベクトル)]"""
        lat = self.d["lattice"]
        if lat is None:
            return [("a", "x", np.array([1.0, 0, 0])), ("b", "y", np.array([0, 1.0, 0])),
                    ("c", "z", np.array([0, 0, 1.0]))]
        return [(k, k, lat[i] / np.linalg.norm(lat[i])) for i, k in enumerate("abc")]

    def draw(self):
        d = self.d
        if d is None:
            return
        fig = self.plot.fig
        fig.clear()
        quants = [(k, lab, yl) for k, lab, yl in self.QUANT if self.q_vars[k].get()]
        sel = [ins for ins, v in zip(d["instances"], self.inst_vars) if v.get()]
        self.series = []
        if not quants or not sel:
            self.plot.message("表示する量とデータを選んでください")
            return
        dirs = [("avg", "avg", None)]
        if self.aniso_var.get():
            dirs += [x for x in self.directions() if self.axis_vars[x[0]].get()]
        ls_of = {"avg": "-", "a": "--", "b": "-.", "c": ":"}
        mode = self.x_var.get()
        axes = fig.subplots(1, len(quants), squeeze=False)[0]

        def comp(T, dk, n):
            return tensor_average(T) if dk == "avg" else tensor_along(T, n)

        if mode == "T":
            multi = len(sel) > 1
            for ax, (qk, qlab, ylab) in zip(axes, quants):
                for i, ins in enumerate(sel):
                    if qk not in ins:
                        continue
                    for dk, dname, n in dirs:
                        y = comp(ins[qk], dk, n)
                        color = PLOT_COLORS[i % len(PLOT_COLORS)] if multi else AXIS_COLORS[dk]
                        lab = dname if not multi else f"#{ins['id']} {dname}"
                        ax.plot(ins["temps"], y, color=color, ls=ls_of[dk] if multi else "-",
                                marker="o", ms=3.5, lw=1.6 if dk == "avg" else 1.2, label=lab)
                        self.series.append((qk, f"{transport_label(ins, d['mu_tag'])} [{dname}]",
                                            "T (K)", ins["temps"], y))
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel(ylab)
        else:
            try:
                T0 = float(self.t_var.get())
            except ValueError:
                T0 = 300.0
            groups = {}
            for ins in sel:
                if ins["mu_value"] is None:
                    continue
                groups.setdefault((ins["approx"], ins["nbands_sum"], ins["delta"]), []).append(ins)
            is_den = d["mu_tag"] == "selfen_carrier_den"
            for ax, (qk, qlab, ylab) in zip(axes, quants):
                for gi, (gkey, lst) in enumerate(groups.items()):
                    signs = [(-1, "hole"), (1, "electron")] if is_den else [(0, "")]
                    for sgn, sname in signs:
                        pts = []
                        for ins in lst:
                            if qk not in ins or (sgn and np.sign(ins["mu_value"]) != sgn):
                                continue
                            it = int(np.argmin(np.abs(ins["temps"] - T0)))
                            if abs(ins["temps"][it] - T0) > 1e-6:
                                continue
                            pts.append((abs(ins["mu_value"]) if is_den else ins["mu_value"], ins, it))
                        if not pts:
                            continue
                        pts.sort(key=lambda t: t[0])
                        xs = np.array([p[0] for p in pts])
                        for dk, dname, n in dirs:
                            y = np.array([comp(p[1][qk][p[2]], dk, n) for p in pts])
                            one = len(groups) == 1 and not is_den
                            color = AXIS_COLORS[dk] if one else PLOT_COLORS[(2 * gi + (sgn > 0)) % len(PLOT_COLORS)]
                            lab = " ".join(x for x in (str(gkey[0] or ""), sname, dname) if x)
                            ax.plot(xs, y, color=color, ls="-" if one else ls_of[dk], marker="o", ms=3.5,
                                    lw=1.6 if dk == "avg" else 1.2, label=lab)
                            self.series.append((qk, f"{lab} T={T0:g}K", d["mu_label"], xs, y))
                if is_den:
                    ax.set_xscale("log")
                    ax.set_xlabel("Carrier density |n| (cm$^{-3}$)")
                else:
                    ax.set_xlabel(d["mu_label"])
                ax.set_ylabel(ylab)
                ax.set_title(f"T = {T0:g} K", fontsize=10)
        for ax, (qk, _, _) in zip(axes, quants):
            if qk == "sigma" and self.logs_var.get():
                ax.set_yscale("log")
            if qk == "seebeck":
                ax.axhline(0.0, color="#888888", lw=0.6, ls="--")
        h, l = axes[0].get_legend_handles_labels()
        if len(h) > 1:
            fig.legend(h, l, loc="outside lower center", ncols=min(4, len(h)), fontsize=9)
        self.plot.draw()

    def save_csv(self):
        if not self.series:
            self.app.messagebox.showinfo("CSV", "保存するデータがありません。")
            return
        p = self.app.filedialog.asksaveasfilename(title="CSV を保存", defaultextension=".csv",
                                                  filetypes=[("CSV", "*.csv")])
        if not p:
            return
        unit = {"seebeck": "S [uV/K]", "sigma": "sigma [S/m]", "kappa_e": "kappa_e [W/(m K)]"}
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            f.write("quantity,series,x_name,x,y\n")
            for qk, name, xn, xs, ys in self.series:
                xn = xn.replace("$", "").replace("^{-3}", "^-3")
                for x, y in zip(xs, ys):
                    f.write(f"{unit.get(qk, qk)},\"{name}\",{xn},{x:.10g},{y:.10g}\n")
        self.app.set_msg(f"CSV を保存しました: {p}")


# ----------------------------------------------------------------------
#  メインウィンドウ
# ----------------------------------------------------------------------
class App:
    pass


def classify_path(path):
    """ドロップされたファイル/フォルダの種類を判定する: "chg" / "xml" / "h5" / "fermi_dir" / None"""
    if os.path.isdir(path):
        return "dir"
    name = os.path.basename(path).upper()
    if name.endswith(".FRMSF"):
        return "frmsf"
    if name.endswith(".H5") or name.endswith(".HDF5"):
        return "h5"
    if name.endswith(".XML"):
        return "xml"
    if name.startswith(("CHGCAR", "CHG", "PARCHG")) or name.endswith((".CHGCAR", ".CHG")):
        return "chg"
    if name.startswith(("POSCAR", "CONTCAR")) or name.endswith((".VASP", ".POSCAR")):
        return "poscar"
    if name.startswith(("EIGENVAL", "PROCAR", "DOSCAR", "KPOINTS", "OUTCAR", "IBZKPT")):
        return "fermi_file"
    return None


# ----------------------------------------------------------------------
#  アプリのアイコン（バンド図と E_F の破線。色は FermiSurfer と同じ青→緑→赤）
#  ウィンドウの左上とタスクバーに使う。PNG を base64 で埋め込んでいるので、この .py だけで表示できる。
#  exe ファイルそのもののアイコン（エクスプローラーでの表示）は vasp_viewer.ico を build_exe.bat で指定する。
# ----------------------------------------------------------------------
APP_ICON_PNG_B64 = {
    64: (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAVvUlEQVR4nNWbeZTV1ZXvP+c33rnmgioGURRkUBEiGjFCTIyJRhPz"
    "UiTEGGOWEUxrq+l+rzuvX7+CjnZ3XqLB1jZoJzG+GElTEeO4WtEXSFrbxAEQZVJQxhqp6d66w284+/3xuwWFUlAItPZ3rbuq1v0N"
    "d+999vnu4ZyjODqopqYmo6NjqspNblVH+ewJRWpLg9TXT5OWliYNSo7ry5ubm42mphXmcX3pCURT0wqzubnZGMm9RxpF1dS0wmhp"
    "mR8C3PW0uH98/uHpxULf1FIhPwEdfjSMYqjQdRK77GRm69gZZ66/4xszBgCamprMlpYWDQzrEcMboLnZYMkSDXD1dx+c1N3TeV3f"
    "vtYv5fr3TSwVcpSKeUTkiBY80RBAKXDcBLF4ilRFzY5M1ainU7UN96644xtvQOTBS8q6vBeHlH/wgebm31mv7Xrl+11tO/+8fc/b"
    "ie72nXilvIiIVuo4z7NjRDQYyrTduKqsHcOoMRO9+jET76uvrfqbn//wumxT0wpz0JOH4n0GGFT+ylvuaxhob2/Z+fa6Obu2v47o"
    "MLAsx1CGMaK59WFBREvg+yEKq/GkaUyYNHNDpq5x/sp7Fm4+lBEONkDZ7a/5ywdG7965/bkt638/ravtHd+JJS0FSuQjNejDQikF"
    "IKViPqioHm1PnjGvbcyY0z+5/J5rN793OgwZTVFNG6ep++4Te/fuHY9uWf/7afva3vXdeMpG5L+M8hBNBxFRbjxpZ3s7gs1rfze6"
    "rf3tJ2746yerynfsH3hr8J+mphajpWV+OJD4wff3bN9wXlfbO34snrK1PiR3/JeAaI3txq1sT7u/Y+vaU103cddPHlryjaamjWZL"
    "CyGUPaC5udloaZkffu2WX53S3bnru7u2rQtdN2mdaOWVij4nEqI1Tixht+7YGHa2vnv11/7iF7NaWlrCwbzGAFi9Ovrb27vrOx17"
    "truitaCOb4RTCkxTMC1BlSdeECiCIPoZZYBpCaYpx2QUVf4MZWoBDMOUtl1b6Wrd+VfRty0H3S8r3hBn6fdufWvjy6vGhWEgSqnj"
    "wvaGAcoQ/JIinzfQocKNadyYkEhHHpbPGpSKilLRwDCFRELjuILWipE4oaHAQAhFEQAhCq0gXuatMnuJ1qGaMvNT2fPmXjHx7r+5"
    "tBNEWc3NopYsUbLyp//3zEK2d7xXyovtxIxjJT3TBBEYyBn4nmJUo88FFw/wsbkDTDyjSEVdQKY6RAHZbpP+TottG2K8vCbJqy/F"
    "ad9rYzlCOqVBQfi+CA6mERkpGxiEKFxTU22EpENNMtRsMV0UYCGglAoDPywM9KV379xxAfDo3LmLTWv16sUGoAu5vunFQg4RCRlC"
    "jkcLw4jcvb/fQATOOz/PFxf08fFLcjSM8dFACQjLHwMYXRviTvL49Jw8NyzqZs8emz88k2LF8gpeejGBUpDJaERAazCN6G9PwSLu"
    "aObWDHCpm+PjhQITuz0SvZqk1iyPVbAw07hfNqWUFHJ94peyZwKP5ia3Kis3uVWxBsKwMN4r5gdj6NErrsAwhWzOJAxg7oUD/Nkt"
    "+5hzSQ4D2Af0BgrLEhzAHnwOMMufIiCBYtwYn5u+1cPCb/Ww6pkUS++qYfWaJKYF6WRIX97EtoRvntHDorE9nFssQBvQC2QBHbn9"
    "VcU+7kzWsMFySYqglEGplFe+VzwJogpy/0iL5gO5vVKRK+aLBsW8yTlnF/if3+vk0iuyBEBboAgMIW1AzBK277XZuC7GjrccWndF"
    "Zmgc53PKaR5nzSgyudEnCeQ0OFpx+SU5Lv9Ujsd+m+b2O+p4eX2cudMHuO1THVwQz8MuCPoUphZUDIgDQVQQP+8k2WY6xEWQ8sCW"
    "c4T9/PbBXV2BYQgFz6CQNzltfIlbru/m24t6sBNCT6DwDaHBEtpLiiefSvPo8grW/ilBd6dJGKj9bC8SRYDqupCZs/N8ZUEfV16W"
    "JeYKOgcyAF+4JMtlZ+d46rE0l43KYmWFcAeYMbBGC1kx+H0hyZ8ycbZZNl1i8Yodw0PhIMOWgyMygAGgQBGFKC2KvKfwPJMJ9R6L"
    "ru1h0Td7qDgtRHvg+4oqW8h6in95sJKf/rSaN1+PoZQQT2rSlZr31lIiikJe8dxTaZ59MsOPzyyy6Lpurv96L1ZMCFoVVih8YU4/"
    "vAuBVph1wuaCy8+2VvL4rjTvFG18FIYdyRwXfVjlR2QABRREEWpFEAJaYRvCWQ1FvjGvl69d1kfN6RFFe+0KZ7RgiPDQQxXccXct"
    "69bFcGOaiqoAiMKaDgfffDBMEzKVUfm+ZbPDjbc0sHRpLff8oJWLL8wR7lCEA2AmwbAEirD+HZdHWjO8k42mk2WAqSTy0BHQ2WEN"
    "IIAnikm2R4WjGZvxOauxyJzT83x8RgFrnIANfpfCrhMcW3juiSS33VPHmheTOK6mujZAa0UYHlkakcFwp0hnNNleEzutGTfJAwVi"
    "gzNeIAeyGXS34isT+/nK1f38+1sJ/n1ngvVdMXblbXKBwVuBsz8MHvUUMIC8GNxR28bC8T3YNQKjgHqgCkIFYS+YdWAjvPiHBHes"
    "qOGJ36fRAtXVAVoOZHpHA9MS+vaZnDUnzyOP7eSkhCbsV1inCWueTiBbFfPGDmBagnSC7oUL4nkuaMyDC/RDIIp/zWVY2B2FweGk"
    "OKQBFOChqDd8bqjpwYwLoQ2GgCoBJTDj4A8oVr2ZZNnqKp55LUXRV1SmNQoIRjDiwymf7TaZdE6Bex/diZvQdHqKuozw8CMZvnXd"
    "GAA+NyPHwpk9fKpyADsm0AdkQWeBIIpMV6X7uLO/hg2+S3KY/s0hDSBEo9qlLR7pSzM/3Y9ZhCCr6PZM3tzt8mxrilXbk6zbE0ML"
    "ZBKaeFyPyNWHg2EJhT6Tk2YU+bvf7MJNRcqPdYTbl1Xzv24dTSajMQx47LU0T6xLM6OhyMUNA3wmk2OaVaI6FmJpAQXP5ZNsCxzi"
    "ShiueXdYDjAVfLu9kZ/3V4EBrdqiyzPpKFoEgcKxNelENOKh5tiUN8HLGVSe5PPd3+4kUx/QWVKMdoV/XFbN/7mlgYrKiGy1hspE"
    "lBluaHd5dW+cH1k11McCas2Q0QRYIrzkxfFE4agPwAHCAfJYNZAs3yzYClKOxnAFLSMrVo4EZUBQVDiVmmse3UnFqIDOoqIxJvxi"
    "eQV33txAZXWISESUEBkcIGELKSfimz7PpEtM3sAFIGnowyp/WAPAgV5yhREVJEIkgJYoFzguUCAadKi45P49jJ5UorOoqI8JKx9P"
    "89NFjaQrD1Z+KIbKYqlogFRZ8sP2w8sYUckbAmFZ8ePdGFOG4PebzF62h5MvydFZUuiY8MJLCX5+7VgsO+ofjCRLFyKlBwutkcj6"
    "oXZ4lS34HRYn/fdOJizoo6ekCFxh01aH5QvGogQMJ/KQE4UPXAscK5QthB0WFV/tY/ySDnK+QmxhoN9g1VfHUeqycDIaHZxYOYZP"
    "hAzjoNJYa83QavFI103z4FWzcGhHwwTpN7E+VqD2nlY8gQGgpGDdtWPJbo4RqxPENxl8zdDnlVIMXZ4QEYb2Lw95fZg5dOhESCkG"
    "BgbwS6WIokUTSyRwXTdagVGKbDZL6PtRPSxCIpXCtu39Rujt7UVCvT8FS6XTmIaJGAIeSCYk9qvdhJUhA55B4Gh23txA9xMZzJoS"
    "3e25svSgTIN0Or1fNt/3yedy+3/btG1SqdR+2UqlEsV8fr/stuuSSCZHZgDDMMjn83z96quYM2cO2WyWVDrNQ798iBdfeJFUKkWh"
    "UOSmm29i6pQp5PN54vEYP7n3PjZt2kQsFkNE+Nvmv6WhoQHf81BK8eM7l9LW1opl2TjaJVi5B+9kj4Jn4DuagWVVZO+vQSoL1NU0"
    "8ve33YyIYDsOra2tLL1zKUopisUiU6ZM4YbvLKRQKJJIJNi4aRPL7r2PeDxGLpfj/Dnn8/Wrv04umyWdTvPCCy/w0EMPE3PsIxtA"
    "KUUQhEyYMIHZ555Ld/c+qqur+ben/40wCDEMA601kyadxuxzZ9Pf308qleRXDy1Ha41SChFh2rSpTJw4kVKphFKKWCxGqDROv0n/"
    "0n34s7tIeop9jmbsW6cii6sh3YcOhJjrMutjsxARXNdl27Zt+2XTWpPJZDhn9jnkcgNkMhmCMEBrjWEYhEFIbU0t5547m+7ubqqr"
    "a9izZw9hEKBc5/3ePuv66+1X77/fv+yaHyzetHZ1c9uuzYHtxKwwDI9qzh/xumtidID3HSjctQ+8LMopYrwTo+KiU9EDGm2E+xdt"
    "jnnOv/e6aRL6XlBZN9aaNvOiB59d3vzNuXObrcOS4OEw4usiiK0wugX/Cy6FpQK+C5aG3gTxq2rwuwJUWqISs4z3kuhQKKWO6fpQ"
    "nPgwaBsY2ZBwikvh5zUgvUASVIz4jRbmKwqpF/A/nJ0GJzYRMhUUQnSNTf7XpyIZF/w42FXEmpPYv5ay8idUisPixBnAUOCFoBT5"
    "5bPQkyqg5IBbg73cwf1hEalXH6rycKIMEHVOwYfCA3MJz2uEogK3FusZSCxqRzIqStw/ZBzZAEpFxbphHlj2OdL9ClRvkcKyz+Ff"
    "MQ2KGmJ1GFstEtdvR0yJPOS4V1YqknFQ3hEs8hyeBJUCHaKKfeUlVhMxbXDc6IcG16oGUVZK9eQp3DUff8FMKO2GWB2qPUvyypeh"
    "14OkFZWXxwODg6I1eCVU6INEIVTcJJjWYUvJ4Q2gFAQ+Es/gz7ocLAOjvxXVtQejdScqn0diLsQTB57RIaq/QGHpdXiLLgRvB7i1"
    "qI4syS+uxNjRj1S4EBwH3x8Ms4U8yishiQS64SSkagw62QCGwtq0BpXtBDvGcO42vAHK9xeuvpvgwvOjbmsCkCJm+1asl9dgP/s0"
    "5sY3QZmIZQCK/C//Dv/L54H3Djg1qFyB5JcfwFzXilQnjl358oir/my0ajxlKv6FlxJMm0uYmQReDDzABvOtTSTv+yoERTAOreow"
    "BlCgfSReQXDqORAAdlnwjEs44UzCz55J6X/cgLV6Ne4//wRj+x4KP/sRwYVngPc2OPWoXI7Elf+E+crOY1deAaaJyuXB9wkuPJ/S"
    "tdcRnP1JCCxoBzolWlQMFXiKcMIUdOVojLat4B6VAQRMGyPXRezppZQ+fy1UxpCKFGSAGNFSbtoiuOLTBJ+8ANVZQE6pgmAvOONR"
    "+/aS/NLtmC9vQaqSx6a8aUAQorp7CGdNo/RXN+N/7jLIA7t1pLRrQG2030N15iDwcFa3YLa/hTjxYXngMFNAI04c9/l7sV9aAYkU"
    "un404bTpBJ/5BOGFZyBeBt7xgRgyJgYlH9xRmOs3kPjyQoy2DqQmDaXgyJtyDwUj2kSkenNIJkHx+zdSuvVGcNKwL4CiAQ0GdIRY"
    "f3wJ63fPYr71BkZXG5TyGLl9iOVGyg8TEY6QCivEjWN4rVDwMfeuxf7jU4SPPELxW0vw530ebBcaQtgn4NpQ3Ye4bYRnNGK0daH6"
    "C0jKHexejlxxQ8FACeX5+F84n+Li76AnfwwkAK8ENS70edgrHsf92QOYm98E3wPbRiwblIE4iSP20w5jAAUEoC0CmYtO1KIbTyIc"
    "P5Nw+mxkXAL6NNQBfrnwqOwGczP69CryK7+H/eQLuEsewXx9N5KwIeEcaCu/1xaKMsEBeQ+V9wjPGEdpyVX4l10MxKDUAW4tODb2"
    "b1bi/nAZ5utvguMgiSQkUhzUPh5BM/EIHhBSKN6Bn7ocGoAxQCNRRAg0iAElsFb9Cfv1h/H+eh7h9ElAJ/gD+J+fQXDRydgP/wfu"
    "PasxNneAoZC4DbZ5YPlWC/ghqlACrdGnj6J448X4X7sISdSDn41I2K3GXP8a7u33Yz+5Ohrtqqr35yNHgQM7RA5KTBUoHwmr8fVl"
    "UfizNIRGRH5FoM/A3L0WZ+0vsdc9jsrnsZ95Bu+mz1P6i88hyST4+5C4xrtuDv6C6ViPb8BeuQFzfRtGx0DEDQCuha5PEpzVgP+l"
    "GQRXzEKS9VFC4/eCXYMayOLe8Qucu3+DGigilRVDl5OPAgqlDuhqpbY0CIDlxHY7sbhETQwBsTGMLhzjQby+q1FOCchjZHdgbX0V"
    "q+v/YXa8igqKSDyDVMchCHBv+zXW4/9BqflK/MunAh4EfUhC4y84G3/BdFR3H8a2How9fYBCj6lET6xFqmuBWKR40A9WBdgG9hPP"
    "4S75V8zX30Wq0khl5qgVV4CIxnXj2I67AyA3uVVZ9fXTBCCWTGyJxdMKRXlCawSXWPw2nHA5qqsE+7KosBcVFsGwETuJOLFox0MQ"
    "Mb3UVWC83UbiK/cQfGYapRs/QfDpU4gcbAACD6l2CavHEp4zngNbFYGgAJaASoElWM+9gnvP81jPbkAcC6mrjBQ/6lGPICIqnszg"
    "xDJvQHmTVEvLfA0wZcrZa/e8/cZex000ig511FJVgIthvR3NdwywYoiVjBQSvT/vjn4BCEKIO0hCYa3aiPX8ZoJPnIJ31VkEF5+M"
    "1CfLCg+dswZggaVQHf1Yq17D+dWrWH94G7QglcnI3YMPpnikPGKalplIVxXrxo57CWDNGrQCmDu32VqzZknw2av/ftmm1363sHPv"
    "tsB2XOtAT8/gAG0fRRFjRgWTypYg0OhxFYQzGgjPHoWeUDGEBMF4tw9zbRvmulaMXb1gmUjajeJ3eIzps1KEgRdkqkab08+5+KlV"
    "yxdfXj5OE1oA8+ah16yBqrqTltY3Tryus3WbKmt6QMIPgrLgknEBheopYD+1BfuxzQd5PmXawVJIwkaqk9EX4eCFY4NCocNANYyb"
    "rKpqG+6Mvm0CWg7kZ4OnKS695h/u3vr6izfu3v6678aTthzPHeNKRc6kDtELiFiqvKR7/BoFShn4pXxQ2zjROn3G3Ceeffh/XzH0"
    "0MSQ/FBUU9N8o25qU3zzxvV/evOV56ZkezsC241bx9UI/4lQhkHglUI3njLPOPezrRNOmTXzwR+92RFdjQwwpCOkZOrUqXLvkvm5"
    "unGTvnj62fNaUxV1llcY8JU6uOf/kYdS5ZEvBG48ZU6ZeVG+vuHk//bgj+a3NTVtVIPKwyFKlMGp0HTTz6bua3935btbXpncumOT"
    "NgxTTMsxVPkcwUftAM0QOpEw8LQOA+oaTzVPmTJ7V3Xt+AUr77vhhSMfmipjvxFu/Zfqge6eO9t3b72mbddW+rpbCQM/VErJcTpO"
    "cNwgohERZZq2ma6qZ/S4yYwae9pv07UNN7b8+Jt7RnxsbhBDieKqWx84r6tz7y09nXs+k8/1VBUG+igV8ojoE3/mZQRQlA9OJjOk"
    "MtW5ytrG5ytqRv9zyz99exXsP0F6yCTiCNJHxDj48J/f/tyo9rbt5+Vz/Wd6pcIEEf2RODoryggdN/5uIpF5o6p29Kv3L7liJxAd"
    "A1y8WN63MfloER2ebvpIKDsSHI28R+e/ImruvMUm84DVRy/YCcU8YDWsmYdmmHPCh8L/B1lGNfNIhZBvAAAAAElFTkSuQmCC"),
    32: (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAH9UlEQVR4nK2Xe4xU1R3HP+fce+e9OzvDDLtAENFZBHazMWJQbGOj"
    "dYsWtEg71hiSGqU2ptqg1Vja6orWpGjgj6qJRW2sNb5WbbXSFLUaHl1BNFZaH1gR6LoLy+4OM8PsY+bee379Y2aXXUVthV9yknty"
    "f+f8vr/H+T1gAnV0dOiJe6XUCVkTKZvNWpNkTPzR2dnpfygSvOa7q5flB3ovLFdGpiPCZ27530kAHDvUm0hP25L9ydrO685TJbJZ"
    "i85OfxxAR0eHXrNmjbl45bqz//1O18OHB3rml0dLGN//inInk7YsgqEo9cmmvbOaz7j+lSc7No4prMY+lly9buH7O1957VDvRxFE"
    "PK2t49F8MomIMT4CdqrpZDKti5b97ak7n89ms5YC1LqnJbTh7kt39ex7N6O1dlHaQeSEyLYUKBEMCgVexXPtppPmHr7wslXN9/+8"
    "PacB+eMTN30nnzuYQcSDEyPc0lX/5l1NTmxKSpNXlm205RUGDyR2bn7uakA0QHGwb0l5uCRKW9Ti5rjItoTiiKZiFBc3HeH+ugP8"
    "9ch+Hi18QlKhRsojMpQf/LbWGlspRaVcbjLGV+o4fa5UdeUKNl+fN8xdZ/Rx7uAwvAMMV3keiiTUViuk/Mpw2rId7NpBc7xaj0Ev"
    "FDU3rRjgrm8eItArlHcp/hyrY3syzAcmyJtOmKjv49dMbY+dPxHCiwWL9fccYNX3c/AxvPB2Hbd3pXk7FwJHoRFiYiZpax/zxv9L"
    "enWV8har1x9g1bU5zCjc8WyaO3+bxrIh6VT1FRSfzizHDUBroTRgs2L9Aa64NsdeT/HLVdN4/OEkDSkPJeDJ5xv4uAAoWxjpt1l0"
    "Sz/nXZdjP7DhZ408vyHJlOkurvvlnp0EQGuNUgpjDCKC1tXHbHwz/h/AGINyBLffZsblBc644xCHgM67G3np3jTJ6RVcV6EtDVLj"
    "VwqtNSKCMebYAIrFIsb3iUSjOI4zvo/V1aG1plAoIEaINUTRhQDBM0eYeV8veQVvPRRn2y0OOjqI59ZjjKFYKKAti7q6OlzXZXho"
    "CG1ZRCKRYwO44IILaEjE2bH9Dfr7+2n/VjuxWIxtW7dRLpdZunQpTsjm9de2k2/IE3q0h1K9YecrYfpvzbD8irNxfY+tm7cSDodZ"
    "fOFiSqUSW7dsJZ1Oc9bFS8gfLtDV1XXU6jWbgoJMc4a2tjYikQjGGJqbM7S2tuI4Dkop5s6fS8u8FgI6wOjDBQqnltm3J8jeK6cT"
    "jgRoaWvhtDlzUUrhOA6tra00N2cwxhCJRGhrayOTyaAUCAIolFKKuWd976Wej//VXiwWfDzPitTX4zgOhUIBfJ9YPI7WmuKRAox4"
    "WA/OxF+Zh2KByEUJnF0B/KhLKVcEpahvaMAYQ6lQAMsiHo9XXVAsgm2bWCymp82c8+6+d19uneSCRCIB2qoFoanuAd8YsBVJU0f5"
    "7maGVtogISLXRXHedJG0Qbs2iVSqyu/7aK2P7kVwgiESjREwPq7rHjsGfN9HDRfB81Daxg+FwQmArVG9RYZvXMjozfOAPkK3HcZ5"
    "soykNbgAgj+xgVG1pOO6qPIIiIexbCQY/ZxnqBT4Hm7L+ci0RnS+F+u9XajBAXCFyg/bGf31YuAwgfv2EFzbg6QC4B2jetoWlF1U"
    "KY+k03gtCzGx6aiBPuw9XeOhdxSA0qjKMN6sBQzf/gicDNSBPtJN4A+Pga0Yve0SIEdgw5uEb96BTAmC+ZTw2rtXg3mkMUX5R9dQ"
    "WbQco2dDCRCIrc2iPt4B2poAQASxA+jB/QT+8ihm3gz8r7ViZs9k9KerIQ5IkeBv/kR41ROYGclqLTPUakE146niCGihctVyyjfd"
    "iJl5MhwE9VYJ/d4H6IP7UYP7EDvIWNNTc4EI2kaP5HAefB6TmossjeOfOwXm2HAqEMtjMmHc8+Zjb99bLS7Bqhaq7AOCtyhDefUP"
    "8NovBaLoj7oJ3Pt7nJc3oQ71otwKEggjygIxRwGI0VqpYTx/AUPJZ2ABkACGwN74OtrponLD6bhLFuJedBLOC2/hPPdP9O4BAMxp"
    "U3GXn4l7yTdAz4bRHoIPbCRw72Ponl4kGgUniDjhas4xRztyW0RwHOegUmHR1j5xys8hexrRg3uwN27G3r8FVSoReLGF0V8sxls8"
    "G3fZmbjL5lNtcyxgSg2xwd70IqFfPYu14z2kPoZMSYHvV00uPoAopUU7gX7fc6sWiE9pfLWvN75i2O0jwir4RMN/XBCNBOuQVArr"
    "H3uJZh/AO38Olcta8M9pwjRGAEH3dWN1vUHg6XewX30fUEg6Cb4Bz5sUp2KMBEJRVZ9IbTbGYANq0SUrnznYvfvOoeLgDFERD1vs"
    "8SZJfPA8JBoCBPvl3dibdiOpCJIMV2MgN4IaGAKtkPpQ1breZ4capZTv+Z6ub0gPn9Jy9kM7N21AZ7NZfc/KeUdOOW3BylTTLDx3"
    "1BbfdRHfR3xTi3VTTY8YiYeNJMKGsmd0d9Ho7qKh7BtJRIzEwwap8Y6dQxnAF2M8t1K2ko2z9Emntv34qfVXdY8NJuNzYfvlt166"
    "78O3Hzjc3zu1Uh7BmC8YzSY8v6p/P59Va40TCFOfnFqc2Xz6DVueXfu78dFsnKs2MF5/15b0zr8/feWR/MDiyuhQI/IF/dSXUK3e"
    "iRUIHqxPTN2WaTnnkcfXrdgPHRrWfLYTnzg6q2r3omwniBP4aqt6NjTeSX1aBsB/AXM/nrwseJv1AAAAAElFTkSuQmCC"),
    16: (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAC3klEQVR4nHWTTWicVRSGn3Pvnb/MfPOTicZk2oZOFoZSF422FZJC"
    "EvxdiIXioiqCBdGCC0EQV1p/Nopumk1EdwqKICJuBGcRq7gRC4q1ylCltsSxLYnTb2Yy38+910UasS2+cJaH9+Wc5xV4WcErbt/i"
    "sanulbUFh8thLeCF6yTeockq0snR+tenTn3YBpQATO1dOtzrbrxn06TO/0gAJZ7YCTaTDasjwQtrv361IvsWj02db58+k0SbRSAV"
    "UIDfXvSAVpBa6CWa8XzCWJLqTj5gdKJ5UP196cI9zqZFgVREDCIKEb092ogeDpXWWvTrc5d1u3pO/7T+e7o47PLboHfYOGcz14zk"
    "xthKw7Av3Loj5dM3LjDbHdJaLvJ5EMh3mYKv29gYEbyIiFIK5z0isn0zbCRkA8fJT/5gdibi+BOTrLRrEAiBR6rivVJKE8cxYRji"
    "vSeKIgabA3CCTeGhDy4iMxELj46y8n6RWiYl193AxTGIQg16IY1Gg7n5eYwxNJtN7tw/S7oO5bc6JAt9Trw4zpUfl7jjwC5QhoPz"
    "h5hsNIg2B6gkSahUKkxP70YpRX3iFnZmJkieV6w/1af17ii/vF1nz/7bqFbHUEoxPb2bSqVCnFqkuffep/9cO7+y2d+0Qa2k484l"
    "okf2wMe70KtnKTw8QOeFsB9itCGXyxGGV22hUNJj45NvGucshYIhbyyu20U/+yRq+UH4uUXx8R8Q4/EKykEZAOcctVoNj8J5h0FF"
    "2LTqo9IR0uP3k56YQ59uMXL0S+gN8SMZsB6H+/e91lpkC2IMuuNtcp8Md7yKlHrknlsm99E7MBzCSA6suxGPbUC9gDfl4K7Vq/lv"
    "bfHsklZnNlI1+Et8sQSF7Ba/N+GF985rY7TkSuUvBGDn7Yee6YeXT1pHBpUF5/5bh5uktSZfrLy01v7mNWGrPG7mwGN398POA97F"
    "+WsZr/MW8KBASHPFauvc95+tAuofXYMqIegvZ5EAAAAASUVORK5CYII="),
}


def set_app_icon(root, tk):
    """ウィンドウとタスクバーのアイコンを設定する。失敗しても何もしない（既定のアイコンのまま）"""
    try:
        if sys.platform == "win32":
            import ctypes
            # python.exe ではなくこのアプリとしてタスクバーにまとめる（.py から起動したときに効く）
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VASPResultViewer")
    except Exception:
        pass
    try:
        imgs = [tk.PhotoImage(master=root, data=APP_ICON_PNG_B64[k]) for k in (64, 32, 16)]
        root.iconphoto(True, *imgs)
        root._app_icons = imgs               # 参照を残しておく（消えるとアイコンも消える）
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
#  ライブラリ（読み込んだデータを名前を付けて保存し、元のファイルがなくても開けるようにする）
# ----------------------------------------------------------------------
LIB_KIND_TEXT = {"chgcar": "電荷密度", "wavecar": "WAVECAR 軌道分解", "dosband": "DOS / BAND",
                 "fermi": "フェルミ面", "transport": "輸送特性", "structure": "結晶構造"}


def _fmt_bytes(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u in ("B", "KB") else f"{n:.1f} {u}"
        n /= 1024.0


def library_save_current(app, tab):
    """開いているタブの内容をライブラリに保存する"""
    store = app.store
    if store is None:
        app.messagebox.showerror("ライブラリ", "データの保存フォルダを作れませんでした。")
        return
    try:
        pl = tab.library_payload()
    except Exception as e:
        app.messagebox.showerror("ライブラリ", f"保存する内容をまとめられませんでした。\n\n{e}")
        return
    if pl is None:
        app.messagebox.showinfo("ライブラリ", "このタブには保存するデータがありません（先に読み込んでください）。")
        return
    kind, obj, name, info = pl
    from tkinter import simpledialog
    name = simpledialog.askstring("ライブラリに保存", "名前:", initialvalue=" ".join(str(name).split()), parent=app.root)
    if not name:
        return
    try:
        with _Busy(app, "ライブラリに保存中…"):
            meta = store.lib_save(name, kind, obj, info)
    except Exception as e:
        app.messagebox.showerror("ライブラリ", f"保存できませんでした。\n\n{e}")
        return
    app.set_msg(f"ライブラリに保存しました: {name}（{LIB_KIND_TEXT.get(kind, kind)}、{_fmt_bytes(meta.get('bytes', 0))}）\n"
                f"保存先: {store.lib_dir}")


def library_open_entry(app, folder, parent=None):
    """ライブラリの 1 件を、種類に合ったタブで開く"""
    store = app.store
    try:
        m = read_bundle_meta(folder)
    except Exception:
        app.messagebox.showerror("ライブラリ", "このデータは見つかりません（削除された可能性があります）。", parent=parent)
        return False
    tab = {"chgcar": app.tabs[0], "wavecar": app.tabs[0], "dosband": app.tabs[1], "fermi": app.tabs[2],
           "transport": app.tabs[3], "structure": app.struct_tab}.get(m.get("kind"))
    if tab is None:
        app.messagebox.showerror("ライブラリ", f"この種類（{m.get('kind')}）は開けません。", parent=parent)
        return False
    try:
        with _Busy(app, f"ライブラリから開いています… {m.get('name', '')}"):
            obj, meta = store.lib_load(folder)
        meta["_id"] = os.path.basename(folder)
        if m.get("kind") == "structure":
            # 構造: 保存したタブ（結晶構造 / BZ）で開き、もう一方のタブにも同じ構造を入れる
            tab = app.bz_tab if obj.get("tab") == "bz" else app.struct_tab
            other = app.struct_tab if tab is app.bz_tab else app.bz_tab
            app.select_tab(tab)
            app.root.update_idletasks()
            other.set_structure(obj["st"], obj.get("path", ""), f"ライブラリ「{meta.get('name', '')}」")
        tab.library_open(obj, meta)
        app.select_tab(tab)
        app.add_recent("library", folder, label=m.get("name", ""))
        return True
    except Exception as e:
        app.messagebox.showerror("ライブラリ", f"開けませんでした。\n\n{e}", parent=parent)
        return False


def library_window(app):
    """ライブラリの一覧（開く・名前の変更・削除）とキャッシュの設定"""
    tk, ttk = app.tk, app.ttk
    store = app.store
    if store is None:
        app.messagebox.showerror("ライブラリ", "データの保存フォルダを作れませんでした。")
        return
    if getattr(app, "_libwin", None) is not None and app._libwin.winfo_exists():
        app._libwin.lift()
        app._libwin.refresh()
        return
    w = tk.Toplevel(app.root)
    app._libwin = w
    w.title("ライブラリ")
    fit_geometry(w, 900, 480)
    top = ttk.Frame(w)
    top.pack(fill="both", expand=True, padx=8, pady=8)
    cols = ("kind", "formula", "created", "size", "source")
    tv = ttk.Treeview(top, columns=cols, show="tree headings", selectmode="browse")
    tv.heading("#0", text="名前")
    for c, t, wd in (("kind", "種類", 130), ("formula", "組成", 110), ("created", "保存日時", 130),
                     ("size", "容量", 80), ("source", "元の場所", 300)):
        tv.heading(c, text=t)
        tv.column(c, width=int(wd * UI_SCALE), anchor="w")
    tv.column("#0", width=int(260 * UI_SCALE))
    sb = ttk.Scrollbar(top, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    tv.pack(side="left", fill="both", expand=True)
    sb.pack(side="left", fill="y")
    rows = {}

    def refresh():
        tv.delete(*tv.get_children())
        rows.clear()
        for folder, m in store.entries("library"):
            iid = tv.insert("", "end", text=m.get("name", ""), values=(
                LIB_KIND_TEXT.get(m.get("kind"), m.get("kind", "")), m.get("formula", ""), m.get("created", ""),
                _fmt_bytes(m.get("bytes", 0)), m.get("source", "")))
            rows[iid] = (folder, m)
        cache_lbl.configure(text=f"キャッシュ: {_fmt_bytes(store.cache_bytes())}（上限 {store.limit_gb:g} GB）  "
                                 f"保存先: {store.root}")
    w.refresh = refresh

    def selected():
        sel = tv.selection()
        return rows.get(sel[0]) if sel else None

    def do_open(ev=None):
        r = selected()
        if r is None:
            return
        library_open_entry(app, r[0], parent=w)

    def do_rename():
        r = selected()
        if r is None:
            return
        from tkinter import simpledialog
        name = simpledialog.askstring("名前の変更", "新しい名前:", initialvalue=r[1].get("name", ""), parent=w)
        if name:
            store.lib_rename(r[0], name)
            refresh()

    def do_delete():
        r = selected()
        if r is None:
            return
        if not app.messagebox.askyesno("削除", f"「{r[1].get('name', '')}」をライブラリから削除しますか？\n"
                                               "（元の計算ファイルは消えません）", parent=w):
            return
        try:
            store.lib_delete(r[0])
        except Exception as e:
            app.messagebox.showerror("削除", f"削除できませんでした（開いている場合は、ほかのデータを開いてから"
                                           f"やり直してください）。\n\n{e}", parent=w)
        refresh()

    def open_folder():
        p = store.root
        try:
            if sys.platform.startswith("win"):
                os.startfile(p)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", p])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            app.messagebox.showerror("フォルダ", str(e), parent=w)

    def clear_cache():
        if app.messagebox.askyesno("キャッシュ", "キャッシュを空にしますか？（ライブラリは消えません。次に開くときは"
                                                 "元のファイルから読み直します）", parent=w):
            store.clear_cache()
            refresh()

    def set_limit():
        from tkinter import simpledialog
        v = simpledialog.askfloat("キャッシュの上限", "上限 [GB]（超えたら最後に使ったのが古いものから消します）:",
                                  initialvalue=store.limit_gb, minvalue=0.5, maxvalue=10000, parent=w)
        if v:
            store.limit_gb = float(v)
            store.save_settings()
            store.prune()
            refresh()

    tv.bind("<Double-1>", do_open)
    fb = ttk.Frame(w)
    fb.pack(fill="x", padx=8)
    for t, cmd in (("開く", do_open), ("名前を変更", do_rename), ("削除", do_delete), ("保存先のフォルダを開く", open_folder)):
        ttk.Button(fb, text=t, command=cmd).pack(side="left", padx=(0, 6))
    fc = ttk.Frame(w)
    fc.pack(fill="x", padx=8, pady=(6, 8))
    cache_lbl = ttk.Label(fc, text="", foreground="#555")
    cache_lbl.pack(side="left")
    ttk.Button(fc, text="キャッシュを空にする", command=clear_cache).pack(side="right")
    ttk.Button(fc, text="上限を変更…", command=set_limit).pack(side="right", padx=6)
    ttk.Label(w, text="ライブラリ: 「表示中の内容をライブラリに保存」で名前を付けて保存したもの。元のファイルがなくても開けます。\n"
                      "キャッシュ: 一度読んだファイルの中身を自動で保存したもの。同じファイル（大きさと更新日時が同じ）を"
                      "次に開くときに使います。", foreground="#666", justify="left").pack(anchor="w", padx=8, pady=(0, 8))
    refresh()


RECENT_KIND_TEXT = {"chgcar": "電荷密度", "wavecar": "WAVECAR", "vasprun_dos": "DOS", "vasprun_band": "BAND",
                    "fermi": "フェルミ面", "frmsf": "frmsf", "transport": "輸送特性", "library": "ライブラリ",
                    "structure": "結晶構造"}


class UISettings:
    """ウィンドウの大きさ・最後に開いたフォルダ・表示の設定・最近開いたもの（ui.json）"""

    def __init__(self, store):
        self.path = os.path.join(store.root, "ui.json") if store is not None else None
        self.data = {"recent": [], "style": {}, "geometry": "", "last_dir": "", "restore_view": True}
        if self.path and os.path.isfile(self.path):
            try:
                import json
                with open(self.path, encoding="utf-8") as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        if not self.path:
            return
        import json
        try:
            with open(self.path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=1, default=float)
            os.replace(self.path + ".tmp", self.path)
        except Exception:
            pass


class _RememberingDialogs:
    """filedialog の代わり: 最後に開いたフォルダを覚えておき、次のダイアログの最初の場所にする"""

    def __init__(self, fd, settings):
        self._fd, self._s = fd, settings

    def _call(self, fn, **kw):
        last = self._s.data.get("last_dir") or ""
        if not kw.get("initialdir") and last and os.path.isdir(last):
            kw["initialdir"] = last
        r = fn(**kw)
        if r:
            p = r[0] if isinstance(r, (tuple, list)) else r
            if p:
                self._s.data["last_dir"] = p if os.path.isdir(p) else os.path.dirname(p)
        return r

    def askopenfilename(self, **kw):
        return self._call(self._fd.askopenfilename, **kw)

    def asksaveasfilename(self, **kw):
        return self._call(self._fd.asksaveasfilename, **kw)

    def askdirectory(self, **kw):
        return self._call(self._fd.askdirectory, **kw)


def run_gui():
    global UI_SCALE
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    dnd_err = ""
    enable_high_dpi()
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        dnd_ok = True
    except Exception as e:
        root = tk.Tk()
        dnd_ok = False
        dnd_err = f"{type(e).__name__}: {e}"

    is_admin = False
    if sys.platform.startswith("win"):
        try:
            import ctypes
            is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            pass

    root.title(f"{APP_NAME}  v{APP_VERSION}  (VASP 6.6)")
    set_app_icon(root, tk)
    try:
        UI_SCALE = max(1.0, root.winfo_fpixels("1i") / 96.0)
    except Exception:
        UI_SCALE = 1.0
    fit_geometry(root, 1320, 860)
    root.minsize(min(int(950 * UI_SCALE), int(root.winfo_screenwidth() * 0.9)),
                 min(int(500 * UI_SCALE), int(root.winfo_screenheight() * 0.6)))
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except Exception:
        pass

    app = App()
    app.tk, app.ttk, app.filedialog, app.messagebox, app.root = tk, ttk, filedialog, messagebox, root
    try:
        app.store = DataStore()
    except Exception:
        app.store = None
    app.ui = UISettings(app.store)
    app.views = ViewMemory(app.store)
    app.filedialog = _RememberingDialogs(filedialog, app.ui)
    app.restore_var = tk.BooleanVar(value=bool(app.ui.data.get("restore_view", True)))
    geo = app.ui.data.get("geometry") or ""
    try:
        import re as _re
        mg = _re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", geo)
        if mg:
            gw, gh, gx, gy = (int(x) for x in mg.groups())
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            if 300 < gw <= sw and 200 < gh <= sh and -50 < gx < sw - 100 and -50 < gy < sh - 100:
                root.geometry(geo)          # 前回のウィンドウの大きさと位置（画面に収まるときだけ）
    except Exception:
        pass

    def add_recent(kind, path, label=None):
        import time as _t
        rec = [r for r in app.ui.data.get("recent", []) if r.get("path") != path]
        rec.insert(0, {"kind": kind, "path": path, "label": label or "", "time": _t.strftime("%Y-%m-%d %H:%M")})
        app.ui.data["recent"] = rec[:20]
        app.ui.save()
    app.add_recent = add_recent

    msg = tk.Text(root, height=4, font=("", 9), wrap="word", bg="#fffbe6", relief="solid", bd=1)
    msg.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

    def set_msg(s):
        msg.configure(state="normal")
        msg.delete("1.0", "end")
        msg.insert("1.0", s)
    app.set_msg = set_msg

    def add_msg(s):
        msg.configure(state="normal")
        msg.insert("end", ("" if msg.get("1.0", "end").strip() == "" else "\n") + s)
    app.add_msg = add_msg

    bar = ttk.Frame(root)
    bar.pack(side="top", fill="x", padx=8, pady=(6, 0))
    nb = ttk.Notebook(root)
    nb.pack(side="top", fill="both", expand=True, padx=8, pady=(4, 8))
    tabs = []
    app.current_tab = lambda: current_tab()
    for cls in (ChargeTab, DosBandTab, FermiTab, TransportTab, StructureTab, BZTab):
        f = ttk.Frame(nb)
        nb.add(f, text=f"  {cls.title}  ")
        t = cls(app, f)
        t.frame = f
        tabs.append(t)
    chg, dosband, fermi, trans, struct, bz = tabs
    app.struct_tab, app.bz_tab = struct, bz
    app.select_tab = lambda t: nb.select(t.frame)
    nb.bind("<<NotebookTabChanged>>", lambda e: getattr(current_tab(), "on_show", lambda: None)(), add="+")

    def show_structure(st, src, origin):
        """読み込んだデータの構造を、結晶構造・BZ タブにも入れる（描くのはタブを開いたとき）"""
        if st is None:
            return
        for t in (struct, bz):
            try:
                t.set_structure(st, src, origin)
            except Exception:
                pass
    app.show_structure = show_structure

    def load_structure(path):
        """POSCAR / CONTCAR（またはそれのあるフォルダ）を開いて、結晶構造タブ（BZ タブを開いているときは BZ タブ）に表示"""
        if os.path.isdir(path):
            q = find_structure_file(path)
            if q is None:
                messagebox.showinfo("結晶構造", f"このフォルダには POSCAR / CONTCAR がありません。\n{path}")
                return
            path = q
        try:
            st = parse_poscar(_read_text(path))
        except Exception as e:
            messagebox.showerror("構造ファイル読み込みエラー", f"{path}\n\n{e}")
            return
        target = bz if current_tab() is bz else struct
        other = struct if target is bz else bz
        other.set_structure(st, path, "開いたファイル")
        target.set_structure(st, path, "開いたファイル", show=True)
        add_recent("structure", path)
        set_msg(f"構造を読み込みました: {path}\n（結晶構造タブとブリルアンゾーンタブの両方に入っています）")
    app.load_structure = load_structure
    ttk.Button(bar, text="ライブラリ…", command=lambda: library_window(app)).pack(side="left")
    ttk.Button(bar, text="表示中の内容をライブラリに保存…",
               command=lambda: library_save_current(app, current_tab())).pack(side="left", padx=6)
    rmb = ttk.Menubutton(bar, text="最近開いたもの ▾")
    rmenu = tk.Menu(rmb, tearoff=False)
    rmb["menu"] = rmenu

    def open_recent(r):
        p, kind = r.get("path", ""), r.get("kind")
        ok = os.path.isdir(p) if kind in ("fermi", "wavecar", "library") else os.path.isfile(p)
        if not ok:
            messagebox.showwarning("最近開いたもの", f"見つかりません（移動・削除された可能性があります）:\n{p}")
            app.ui.data["recent"] = [x for x in app.ui.data.get("recent", []) if x.get("path") != p]
            app.ui.save()
            return
        if kind == "library":
            library_open_entry(app, p)
        elif kind == "wavecar":
            chg.wv_open(p)
            app.select_tab(chg)
        elif kind in ("vasprun_dos", "vasprun_band"):
            dosband.load(p, kind.split("_")[1])
        elif kind == "fermi":
            fermi.load_dir(p)
        elif kind == "frmsf":
            fermi.load_frmsf(p)
        elif kind == "chgcar":
            chg.load(p)
        elif kind == "transport":
            trans.load(p)
        elif kind == "structure":
            load_structure(p)
        else:
            open_path(p)

    def fill_recent():
        rmenu.delete(0, "end")
        rec = app.ui.data.get("recent", [])
        if not rec:
            rmenu.add_command(label="（まだありません）", state="disabled")
            return
        for r in rec:
            p = r.get("path", "")
            name = r.get("label") or (os.path.basename(os.path.normpath(p)) if r.get("kind") in ("fermi", "wavecar")
                                      else f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}")
            rmenu.add_command(label=f"[{RECENT_KIND_TEXT.get(r.get('kind'), '')}] {name}    {r.get('time', '')}",
                              command=lambda r=r: open_recent(r))
        rmenu.add_separator()
        rmenu.add_command(label="一覧を消す", command=lambda: (app.ui.data.update(recent=[]), app.ui.save()))
    rmenu.configure(postcommand=fill_recent)
    rmb.pack(side="left", padx=6)
    app.open_recent = open_recent
    app.fill_recent = fill_recent
    app.recent_menu = rmenu
    ttk.Checkbutton(bar, text="前回の表示の状態を復元する", variable=app.restore_var,
                    command=lambda: (app.ui.data.update(restore_view=app.restore_var.get()), app.ui.save())).pack(
        side="left", padx=8)

    def current_tab():
        try:
            return tabs[nb.index(nb.select())]
        except Exception:
            return tabs[0]

    def open_path(path):
        path = path.strip().strip("{}").strip('"')
        kind = classify_path(path)
        if kind == "dir":
            # 開いているタブで使えるファイルがあればそのタブで、なければ中身から判断する
            has = lambda n: os.path.isfile(os.path.join(path, n))
            join = lambda n: os.path.join(path, n)
            sfile = find_structure_file(path)
            can = {chg: has("CHGCAR") or has("WAVECAR"), dosband: has("vasprun.xml"),
                   fermi: has("EIGENVAL") and has("KPOINTS"), trans: has("vaspout.h5"),
                   struct: sfile is not None, bz: sfile is not None}

            def chg_do():
                if has("CHGCAR"):
                    chg.load(join("CHGCAR"))
                if has("WAVECAR"):
                    chg.wv_open(path)             # WAVECAR があれば軌道分解の準備もする
            do = {chg: chg_do, dosband: lambda: dosband.load(join("vasprun.xml")),
                  fermi: lambda: fermi.load_dir(path), trans: lambda: trans.load(join("vaspout.h5")),
                  struct: lambda: load_structure(sfile), bz: lambda: load_structure(sfile)}
            cur = current_tab()
            for t in [cur, fermi, chg, dosband, trans, struct]:
                if can[t]:
                    do[t]()
                    break
            else:
                messagebox.showinfo("フォルダ", "このフォルダには読み込めるファイルがありません。\n"
                                    "電荷密度: CHGCAR / DOS・BAND: vasprun.xml / フェルミ面: EIGENVAL と KPOINTS / "
                                    "輸送特性: vaspout.h5 / 結晶構造: POSCAR・CONTCAR")
        elif kind == "h5":
            trans.load(path)
        elif kind == "xml":
            dosband.load(path)
        elif kind == "chg":
            chg.load(path)
        elif kind == "frmsf":
            fermi.load_frmsf(path)
        elif kind == "poscar":
            load_structure(path)
        elif kind == "fermi_file":
            fermi.load_dir(os.path.dirname(os.path.abspath(path)))
        else:
            # 名前で判定できないときは、開いているタブに合わせる
            cur = current_tab()
            if cur is chg:
                chg.load(path)
            elif cur is dosband:
                dosband.load(path)
            elif cur is trans:
                trans.load(path)
            elif cur in (struct, bz):
                load_structure(path)
            else:
                fermi.load_dir(os.path.dirname(os.path.abspath(path)))

    def on_drop(e):
        files = root.tk.splitlist(e.data)
        if files:
            root.after(10, lambda: open_path(files[0]))
        return e.action

    if dnd_ok:
        try:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", on_drop)
        except Exception as e:
            dnd_err = f"{type(e).__name__}: {e}"

    start = ["各タブの「開く」ボタン、またはウィンドウへのドラッグ＆ドロップで計算結果を読み込みます。"
             "（CHGCAR → 電荷密度 / vasprun.xml → DOS・BAND / 計算フォルダ → フェルミ面 / vaspout.h5 → 輸送特性 / "
             "POSCAR・CONTCAR → 結晶構造・ブリルアンゾーン）。データを開くと、その構造も結晶構造・ブリルアンゾーンタブに入ります。"]
    if not dnd_ok or dnd_err:
        start.append(f"【注意】ドラッグ＆ドロップ機能を読み込めませんでした（{dnd_err}）。各タブのボタンから選択してください。")
    if is_admin:
        start.append("【注意】管理者として起動しています。Windows の仕様により、エクスプローラーからの"
                     "ドラッグ＆ドロップは届きません。exe を通常の方法（ダブルクリック）で起動し直してください。")
    set_msg("\n".join(start))
    app.tabs = tabs
    app.open_path = open_path
    # 前回の表示の設定（色・背景・線など、データによらないもの）を戻す
    for t in tabs:
        restore_vars(t, app.ui.data.get("style", {}).get(type(t).__name__), only=getattr(t, "STYLE", ()))

    def on_close():
        """終了時: 表示の状態・設定・ウィンドウの大きさを保存する"""
        try:
            for t in tabs:
                remember_view(app, t)
                app.ui.data.setdefault("style", {})[type(t).__name__] = capture_vars(t, names=getattr(t, "STYLE", ()))
            if root.state() == "normal":
                app.ui.data["geometry"] = root.geometry()
            app.ui.save()
            app.views.save()
        except Exception:
            pass
        root.destroy()
    app.on_close = on_close
    root.protocol("WM_DELETE_WINDOW", on_close)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        root.after(200, lambda: open_path(sys.argv[1]))
    return app


if __name__ == "__main__":
    run_gui().root.mainloop()
