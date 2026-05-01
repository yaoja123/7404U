"""
MaxSup Interactive Logit Simulator
COMP7404 Conference Booth Demo

Run:  conda run -n 7606 streamlit run demo/app.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Colours (match poster v4 palette) ────────────────────────────────────────
C_NAVY   = "#1E3A8A"
C_ACCENT = "#C1121F"
C_BAR_BG = "#CCCAC3"
C_INK    = "#0E1116"
C_MUTED  = "#7A7570"
C_GREEN  = "#1B7A4E"
C_PAPER  = "#FFFFFF"
C_GOOD   = "#2D9B6E"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaxSup vs Label Smoothing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Large-font CSS for booth readability
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px !important; }
.stSlider label { font-size: 20px !important; font-weight: 600; }
.stSelectbox label { font-size: 20px !important; font-weight: 600; }
.block-container { padding-top: 1.5rem; }
.metric-box {
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 10px;
    font-size: 20px;
    font-weight: 600;
    line-height: 1.6;
}
.ls-box   { background:#EBF0FC; border-left: 6px solid #1E3A8A; color:#0E1116; }
.ms-box   { background:#FCECEA; border-left: 6px solid #C1121F; color:#0E1116; }
.good-box { background:#E2F4EC; border-left: 6px solid #1B7A4E; color:#0E1116; }
.bad-box  { background:#FFF0CC; border-left: 6px solid #E6A817; color:#0E1116; }
.info-box { background:#F3F4F6; border-left: 6px solid #6B7280; color:#0E1116; }
</style>
""", unsafe_allow_html=True)

# ── Pure-numpy math ───────────────────────────────────────────────────────────
def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()

def ls_gradient(z: np.ndarray, gt_idx: int, alpha: float) -> np.ndarray:
    """∂L_LS/∂z_k = α(1-1/K) for k=gt, −α/K otherwise."""
    K = len(z)
    g = np.full(K, -alpha / K)
    g[gt_idx] += alpha
    return g

def maxsup_gradient(z: np.ndarray, alpha: float) -> np.ndarray:
    """∂L_MaxSup/∂z_k = α(1-1/K) for k=argmax(z), −α/K otherwise."""
    K = len(z)
    g = np.full(K, -alpha / K)
    g[int(np.argmax(z))] += alpha
    return g

# ── Chart helper ──────────────────────────────────────────────────────────────
def draw_logit_chart(
    z: np.ndarray,
    labels: list[str],
    highlight_idx: int,
    method_color: str,
    title: str,
    suppress_label: str,
    gt_idx: int,
    fig_size=(5.5, 5.5),
):
    fig, ax = plt.subplots(figsize=fig_size, facecolor=C_PAPER)
    ax.set_facecolor(C_PAPER)

    K = len(z)
    xs = np.arange(K)
    bar_colors = [C_BAR_BG] * K
    bar_colors[highlight_idx] = method_color
    # mark GT with a slight outline
    bars = ax.bar(xs, z, color=bar_colors, width=0.52, zorder=2,
                  edgecolor=[C_INK if i == gt_idx else "none" for i in range(K)],
                  linewidth=2)

    # Value labels: non-highlighted above bar; highlighted inside bar (white, avoids arrow clash)
    for i, (bar, val) in enumerate(zip(bars, z)):
        bx = bar.get_x() + bar.get_width() / 2
        if i == highlight_idx:
            # Inside the bar at 60% height — completely separated from arrow above
            ax.text(bx, val * 0.55, f"{val:.2f}",
                    ha="center", va="center", fontsize=17,
                    fontweight="bold", color="white", zorder=6)
        else:
            ax.text(bx, val + 0.06, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=17,
                    color=C_MUTED, zorder=6)

    # Big DOWN arrow on the suppressed bar
    arrow_tip_y   = z[highlight_idx] + 0.18
    arrow_start_y = z[highlight_idx] + 1.10
    ax.annotate(
        "",
        xy=(highlight_idx, arrow_tip_y),
        xytext=(highlight_idx, arrow_start_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=method_color,
            lw=4.0,
            mutation_scale=26,
        ),
        zorder=4,
    )
    ax.text(highlight_idx, arrow_start_y + 0.14, suppress_label,
            ha="center", va="bottom", fontsize=15,
            color=method_color, fontweight="bold", zorder=5)

    # Small UP arrows on every OTHER bar — they get nudged up by +α/K
    # This completes the "pressure pattern" and makes LS vs MaxSup visually distinct.
    for i in range(K):
        if i == highlight_idx:
            continue
        up_tip   = z[i] + 0.55
        up_start = z[i] + 0.12
        ax.annotate(
            "",
            xy=(i, up_tip),
            xytext=(i, up_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color=method_color,
                alpha=0.45,
                lw=1.8,
                mutation_scale=14,
            ),
            zorder=3,
        )

    # GT star annotation
    ax.text(gt_idx, -0.28, "GT", ha="center", va="top",
            fontsize=14, color=C_INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8E6DF", edgecolor="none"))

    # Probability strip at bottom of bars
    probs = softmax(z)
    for i, (bar, p) in enumerate(zip(bars, probs)):
        bx = bar.get_x() + bar.get_width() / 2
        ax.text(bx, -0.52, f"p={p:.2f}",
                ha="center", va="top", fontsize=13, color=C_MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=20, fontweight="bold", color=C_INK)
    ax.set_ylim(-0.70, max(z) * 1.85 + 0.5)
    ax.axhline(0, color=C_MUTED, linewidth=0.8, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=13, colors=C_MUTED)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Logit value", fontsize=14, color=C_MUTED)
    ax.set_title(title, fontsize=22, fontweight="bold", color=C_INK, pad=14)
    fig.tight_layout()
    return fig


def draw_step_comparison(
    z_original: np.ndarray,
    z_ls: np.ndarray,
    z_ms: np.ndarray,
    labels: list[str],
    gt_idx: int,
):
    """Before / after bar chart for both methods side by side."""
    K = len(labels)
    xs = np.arange(K)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=C_PAPER, sharey=False)

    for ax, z_new, color, method_name in [
        (axes[0], z_ls, C_NAVY, "After 1 LS step"),
        (axes[1], z_ms, C_ACCENT, "After 1 MaxSup step"),
    ]:
        ax.set_facecolor(C_PAPER)
        b_orig = ax.bar(xs - width / 2, z_original, width, label="Before",
                        color=C_BAR_BG, edgecolor="none")
        b_new  = ax.bar(xs + width / 2, z_new, width, label="After",
                        color=color, alpha=0.85, edgecolor="none")

        for i, (bo, bn) in enumerate(zip(z_original, z_new)):
            delta = bn - bo
            bx = xs[i] + width / 2
            ax.text(bx, bn + 0.05, f"{delta:+.2f}",
                    ha="center", va="bottom", fontsize=12,
                    color=color, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=16, fontweight="bold")
        ax.axhline(0, color=C_MUTED, linewidth=0.7)
        ax.set_title(method_name, fontsize=18, fontweight="bold", color=color, pad=10)
        ax.legend(fontsize=12, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11, colors=C_MUTED)
        ax.set_ylabel("Logit", fontsize=12, color=C_MUTED)
        ax.set_facecolor(C_PAPER)

    fig.tight_layout(pad=2.0)
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")
    st.markdown("---")

    CLASSES = ["cat", "fox", "dog"]

    gt_label = st.selectbox(
        "Ground truth (GT) class",
        options=CLASSES,
        index=0,
        help="The correct label for this sample.",
    )
    gt_idx = CLASSES.index(gt_label)

    st.markdown("### Logit sliders")
    st.caption("Drag to simulate different model outputs.")
    z_vals = []
    defaults = {"cat": 1.0, "fox": 2.8, "dog": 0.4}
    for cls in CLASSES:
        v = st.slider(cls, min_value=0.0, max_value=5.0,
                      value=defaults[cls], step=0.1,
                      format="%.1f")
        z_vals.append(v)
    z = np.array(z_vals)

    st.markdown("---")
    alpha = st.slider("α (smoothing strength)", 0.01, 0.50, 0.10, 0.01,
                      help="Scale factor for both LS and MaxSup regularization.")

    lr = st.slider("Learning rate (for step demo)", 0.1, 2.0, 0.5, 0.1)

    st.markdown("---")
    if st.button("Reset logits", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("**Paper:** MaxSup: Overcoming Representation Collapse in Label Smoothing  \narXiv 2502.15798")


# ── Derived quantities ────────────────────────────────────────────────────────
probs     = softmax(z)
pred_idx  = int(np.argmax(z))
pred_cls  = CLASSES[pred_idx]
correct   = (pred_idx == gt_idx)

g_ls  = ls_gradient(z, gt_idx, alpha)
g_ms  = maxsup_gradient(z, alpha)

ls_suppress_idx  = gt_idx
ms_suppress_idx  = pred_idx

z_after_ls = z - lr * g_ls
z_after_ms = z - lr * g_ms

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# MaxSup vs Label Smoothing")
st.markdown(
    "**One symbol swap** — `z_gt → z_max` — changes which logit gets suppressed. "
    "Drag the sliders to see the effect."
)
st.markdown("---")

# ── Prediction status banner ──────────────────────────────────────────────────
if correct:
    st.markdown(
        f'<div class="metric-box good-box">'
        f'✓ Correct prediction: model picks <b>{pred_cls}</b> = GT <b>{gt_label}</b>'
        f' &nbsp;|&nbsp; Both methods behave similarly here.'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="metric-box bad-box">'
        f'✗ Wrong prediction: model picks <b>{pred_cls}</b> &nbsp;≠&nbsp; GT <b>{gt_label}</b>'
        f' &nbsp;—&nbsp; <b>this is where LS and MaxSup diverge!</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Make the key identity explicit so matching charts do not look like a bug.
if correct:
    st.markdown(
        f'<div class="metric-box info-box">'
        f'In this state, <b><i>z</i><sub>gt</sub> = <i>z</i><sub>max</sub> = {pred_cls}</b>. '
        f'So Label Smoothing and MaxSup should suppress the <b>same</b> logit. '
        f'To see the difference, drag a wrong class above <b>{gt_label}</b>.'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="metric-box info-box">'
        f'Here <b><i>z</i><sub>gt</sub> = {gt_label}</b> but <b><i>z</i><sub>max</sub> = {pred_cls}</b>. '
        f'This is the divergence case: LS suppresses the labeled class, while MaxSup suppresses the current top-1.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Main charts ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        f'<div class="metric-box ls-box">'
        f'<b>Label Smoothing</b><br>'
        f'Suppresses: <b>{CLASSES[ls_suppress_idx]}</b> (<i>z</i><sub>gt</sub>) &nbsp;|&nbsp; '
        f'gradient = <b>{g_ls[ls_suppress_idx]:+.3f}</b>'
        f'{"&nbsp; ← pushing down the <em>right</em> class" if correct else "&nbsp; ← pushing down the <em>labeled</em> class, not the wrong one!"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig_ls = draw_logit_chart(
        z, CLASSES,
        highlight_idx=ls_suppress_idx,
        method_color=C_NAVY,
        title="Label Smoothing",
        suppress_label=f"suppress {CLASSES[ls_suppress_idx]} ↓",
        gt_idx=gt_idx,
    )
    st.pyplot(fig_ls, use_container_width=True)
    plt.close(fig_ls)

with col2:
    st.markdown(
        f'<div class="metric-box ms-box">'
        f'<b>MaxSup</b><br>'
        f'Suppresses: <b>{CLASSES[ms_suppress_idx]}</b> (<i>z</i><sub>max</sub>) &nbsp;|&nbsp; '
        f'gradient = <b>{g_ms[ms_suppress_idx]:+.3f}</b>'
        f'{"&nbsp; ← same as LS because <i>z</i><sub>gt</sub> = <i>z</i><sub>max</sub>" if correct else f"&nbsp; ← pushing down <em>{pred_cls}</em>, the actual wrong prediction!"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig_ms = draw_logit_chart(
        z, CLASSES,
        highlight_idx=ms_suppress_idx,
        method_color=C_ACCENT,
        title="MaxSup",
        suppress_label=f"suppress {CLASSES[ms_suppress_idx]} ↓",
        gt_idx=gt_idx,
    )
    st.pyplot(fig_ms, use_container_width=True)
    plt.close(fig_ms)

# ── Gradient comparison table ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Gradient on each logit (one step pushes these logits by  −lr × gradient)")

cols = st.columns(len(CLASSES) + 1)
cols[0].markdown("**Method**")
for i, cls in enumerate(CLASSES):
    marker = " 🏷️" if i == gt_idx else (" ⚡" if i == pred_idx else "")
    cols[i + 1].markdown(f"**{cls}{marker}**")

cols = st.columns(len(CLASSES) + 1)
cols[0].markdown(f"<span style='color:{C_NAVY};font-weight:700'>Label Smoothing</span>", unsafe_allow_html=True)
for i, g in enumerate(g_ls):
    color = C_NAVY if i == ls_suppress_idx else C_MUTED
    arrow = "⬇" if i == ls_suppress_idx else "⬆"
    cols[i + 1].markdown(f"<span style='color:{color};font-weight:700'>{arrow} {g:+.3f}</span>", unsafe_allow_html=True)

cols = st.columns(len(CLASSES) + 1)
cols[0].markdown(f"<span style='color:{C_ACCENT};font-weight:700'>MaxSup</span>", unsafe_allow_html=True)
for i, g in enumerate(g_ms):
    color = C_ACCENT if i == ms_suppress_idx else C_MUTED
    arrow = "⬇" if i == ms_suppress_idx else "⬆"
    cols[i + 1].markdown(f"<span style='color:{color};font-weight:700'>{arrow} {g:+.3f}</span>", unsafe_allow_html=True)

st.caption("🏷️ = GT class   ⚡ = model's current top-1 prediction")

# ── One-step effect ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### After one gradient step (lr = {:.1f})".format(lr))
st.caption("Lighter bars = before step. Taller/shorter coloured bars = after step. Numbers show the change.")

fig_step = draw_step_comparison(z, z_after_ls, z_after_ms, CLASSES, gt_idx)
st.pyplot(fig_step, use_container_width=True)
plt.close(fig_step)

# Key insight callout (only shown in wrong-prediction case)
if not correct:
    c1, c2 = st.columns(2)
    with c1:
        ls_delta_wrong = z_after_ls[pred_idx] - z[pred_idx]
        ls_delta_gt    = z_after_ls[gt_idx]   - z[gt_idx]
        gap_before = z[pred_idx] - z[gt_idx]
        gap_after_ls = z_after_ls[pred_idx] - z_after_ls[gt_idx]
        direction = "wider" if gap_after_ls > gap_before else "narrower"
        color_class = "bad-box" if direction == "wider" else "good-box"
        st.markdown(
            f'<div class="metric-box {color_class}">'
            f'<b>LS effect on gap:</b> {gap_before:.2f} → {gap_after_ls:.2f} '
            f'(gap becomes <b>{direction}</b>)'
            f'{"<br>⚠ Error amplified!" if direction == "wider" else "<br>✓ Error reduced."}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        gap_after_ms = z_after_ms[pred_idx] - z_after_ms[gt_idx]
        direction_ms = "wider" if gap_after_ms > gap_before else "narrower"
        color_class_ms = "bad-box" if direction_ms == "wider" else "good-box"
        st.markdown(
            f'<div class="metric-box {color_class_ms}">'
            f'<b>MaxSup effect on gap:</b> {gap_before:.2f} → {gap_after_ms:.2f} '
            f'(gap becomes <b>{direction_ms}</b>)'
            f'{"<br>⚠ Error amplified!" if direction_ms == "wider" else "<br>✓ Error reduced!"}'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**The key change — one symbol swap:**")
st.markdown(
    r"""
$$L_{\text{LS}} = \alpha \!\left( z_{\text{gt}} - \frac{1}{K}\sum_k z_k \right)
\quad\xrightarrow{\text{one symbol}}\quad
L_{\text{MaxSup}} = \alpha \!\left( z_{\max} - \frac{1}{K}\sum_k z_k \right)$$
"""
)
st.caption("MaxSup: Overcoming Representation Collapse in Label Smoothing · arXiv 2502.15798")
