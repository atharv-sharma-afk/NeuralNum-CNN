import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import matplotlib.pyplot as plt
from model import CNN

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST Visualiser using CNN", layout="wide")

# ── Anthropic-inspired CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg: #faf9f5; --bg-subtle: #f0efe8; --border: #e8e6dc;
    --text: #141413; --text-muted: #6b6a65;
    --accent: #d97757; --blue: #6a9bcc; --green: #788c5d; --card-bg: #ffffff;
    --conv1: #d97757; --conv2: #6a9bcc;
}

html, body, [class*="st-"] { font-family: 'Lora', Georgia, serif !important; color: var(--text); }

h1,h2,h3,h4,h5,h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    font-family: 'Poppins', Arial, sans-serif !important;
    font-weight: 600 !important; letter-spacing: -0.02em;
}

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1100px !important; padding-top: 2.5rem !important; }

.anthropic-header { text-align: center; padding: 2rem 0 1rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.anthropic-header .logo { font-family: 'Poppins', sans-serif; font-size: 2.4rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em; }
.anthropic-header .logo .star { color: var(--accent); font-size: 1.8rem; margin-right: 0.3rem; }
.anthropic-header .tagline { font-family: 'Lora', serif; font-size: 1.05rem; color: var(--text-muted); margin-top: 0.35rem; font-style: italic; }

.section-label { font-family: 'Poppins', sans-serif; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); margin-bottom: 0.75rem; }

.result-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 2rem; text-align: center; transition: box-shadow 0.3s ease; }
.result-card:hover { box-shadow: 0 8px 30px rgba(20,20,19,0.06); }

.predicted-digit { font-family: 'Poppins', sans-serif; font-size: 5.5rem; font-weight: 800; color: var(--accent); line-height: 1; margin: 0.25rem 0; }
.confidence-badge { display: inline-block; font-family: 'Poppins', sans-serif; font-size: 0.8rem; font-weight: 600; color: var(--card-bg); background: var(--text); border-radius: 100px; padding: 0.35rem 1rem; margin-top: 0.5rem; }
.result-label { font-family: 'Poppins', sans-serif; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-muted); margin-bottom: 0.25rem; }

.prob-row { display: flex; align-items: center; gap: 0.65rem; margin: 6px 0; }
.prob-digit { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 0.85rem; color: var(--text); width: 18px; text-align: right; flex-shrink: 0; }
.prob-track { flex: 1; height: 22px; background: var(--bg-subtle); border-radius: 6px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 6px; transition: width 0.5s cubic-bezier(0.22,1,0.36,1); }
.prob-fill.accent { background: var(--accent); }
.prob-fill.default { background: var(--border); }
.prob-pct { font-family: 'Poppins', sans-serif; font-size: 0.75rem; font-weight: 500; color: var(--text-muted); width: 44px; text-align: right; flex-shrink: 0; }

.empty-state { text-align: center; padding: 3rem 1.5rem; }
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-title { font-family: 'Poppins', sans-serif; font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
.empty-sub { font-family: 'Lora', serif; font-size: 0.9rem; color: var(--text-muted); font-style: italic; }

.footer { text-align: center; font-family: 'Poppins', sans-serif; font-size: 0.7rem; color: var(--text-muted); margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }

canvas { border-radius: 12px !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Activation section ────────────────────────────── */
.act-section {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem; margin-top: 0.5rem;
}
.act-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.act-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
}
.act-title {
    font-family: 'Poppins', sans-serif; font-size: 0.75rem;
    font-weight: 600; color: var(--text);
}
.act-subtitle {
    font-family: 'Poppins', sans-serif; font-size: 0.65rem;
    color: var(--text-muted); margin-left: auto;
}

</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model

model = load_model()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="anthropic-header">
    <div class="logo"><span class="star">✦</span> MNIST Visualiser</div>
    <div class="tagline">Draw a digit. Watch a neural network think.</div>
</div>
""", unsafe_allow_html=True)

# ── SVG Architecture Diagram ─────────────────────────────────────────────────
ARCH_SVG = """
<div style="text-align:center; margin-bottom:1.5rem;">
<svg viewBox="0 0 880 130" xmlns="http://www.w3.org/2000/svg" style="max-width:880px; width:100%; height:auto;">
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="none" stroke="#6b6a65" stroke-width="1.2"/>
    </marker>
    <style>
      .box { stroke-width:1.5; rx:6; ry:6; }
      .lbl { font-family:'Poppins',sans-serif; font-size:10px; font-weight:600; fill:#141413; text-anchor:middle; }
      .dim { font-family:'Poppins',sans-serif; font-size:8px; fill:#6b6a65; text-anchor:middle; }
      .arr { stroke:#6b6a65; stroke-width:1.2; fill:none; marker-end:url(#ah); }
      .group-label { font-family:'Poppins',sans-serif; font-size:7.5px; fill:#6b6a65; text-anchor:middle; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; }
    </style>
  </defs>

  <!-- Group brackets -->
  <rect x="108" y="8" width="270" height="114" rx="10" fill="none" stroke="#d97757" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>
  <text x="243" y="22" class="group-label" fill="#d97757">Block 1</text>

  <rect x="428" y="8" width="270" height="114" rx="10" fill="none" stroke="#6a9bcc" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>
  <text x="563" y="22" class="group-label" fill="#6a9bcc">Block 2</text>

  <!-- Input -->
  <rect x="10" y="38" width="80" height="54" class="box" fill="#f0efe8" stroke="#b0aea5"/>
  <text x="50" y="61" class="lbl">Input</text>
  <text x="50" y="78" class="dim">1 × 28 × 28</text>

  <!-- Arrow -->
  <line x1="92" y1="65" x2="118" y2="65" class="arr"/>

  <!-- Conv1 -->
  <rect x="120" y="38" width="80" height="54" class="box" fill="#fce8df" stroke="#d97757"/>
  <text x="160" y="61" class="lbl">Conv2d</text>
  <text x="160" y="78" class="dim">16 × 3 × 3</text>

  <line x1="202" y1="65" x2="228" y2="65" class="arr"/>

  <!-- ReLU1 -->
  <rect x="230" y="38" width="60" height="54" class="box" fill="#fef4ef" stroke="#d97757"/>
  <text x="260" y="61" class="lbl">ReLU</text>
  <text x="260" y="78" class="dim">16 × 28 × 28</text>

  <line x1="292" y1="65" x2="308" y2="65" class="arr"/>

  <!-- MaxPool1 -->
  <rect x="310" y="38" width="60" height="54" class="box" fill="#fef4ef" stroke="#d97757"/>
  <text x="340" y="61" class="lbl">MaxPool</text>
  <text x="340" y="78" class="dim">16 × 14 × 14</text>

  <line x1="372" y1="65" x2="438" y2="65" class="arr"/>

  <!-- Conv2 -->
  <rect x="440" y="38" width="80" height="54" class="box" fill="#dfe8f0" stroke="#6a9bcc"/>
  <text x="480" y="61" class="lbl">Conv2d</text>
  <text x="480" y="78" class="dim">32 × 3 × 3</text>

  <line x1="522" y1="65" x2="548" y2="65" class="arr"/>

  <!-- ReLU2 -->
  <rect x="550" y="38" width="60" height="54" class="box" fill="#edf2f7" stroke="#6a9bcc"/>
  <text x="580" y="61" class="lbl">ReLU</text>
  <text x="580" y="78" class="dim">32 × 14 × 14</text>

  <line x1="612" y1="65" x2="628" y2="65" class="arr"/>

  <!-- MaxPool2 -->
  <rect x="630" y="38" width="60" height="54" class="box" fill="#edf2f7" stroke="#6a9bcc"/>
  <text x="660" y="61" class="lbl">MaxPool</text>
  <text x="660" y="78" class="dim">32 × 7 × 7</text>

  <line x1="692" y1="65" x2="718" y2="65" class="arr"/>

  <!-- FC -->
  <rect x="720" y="38" width="70" height="54" class="box" fill="#e4ebd9" stroke="#788c5d"/>
  <text x="755" y="58" class="lbl">Linear</text>
  <text x="755" y="70" class="dim">1568→128</text>
  <text x="755" y="82" class="dim">+ ReLU</text>

  <line x1="792" y1="65" x2="808" y2="65" class="arr"/>

  <!-- Output -->
  <rect x="810" y="38" width="60" height="54" class="box" fill="#f5e6d8" stroke="#d97757"/>
  <text x="840" y="61" class="lbl">Linear</text>
  <text x="840" y="78" class="dim">128→10</text>
</svg>
</div>
"""

st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
st.markdown(ARCH_SVG, unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def has_drawing(canvas_data):
    if canvas_data is None:
        return False
    gray = np.mean(canvas_data[:, :, :3], axis=2)
    return np.sum(gray > 30) > 50

def activations_to_fig(act, color_accent):
    """Single activation tensor → matplotlib figure."""
    channels = act.squeeze(0).detach().numpy()
    n = channels.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.0))
    fig.patch.set_facecolor('#ffffff')
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(channels[i], cmap='inferno', interpolation='nearest')
        ax.axis('off')
    plt.subplots_adjust(wspace=0.08, hspace=0.08, left=0.02, right=0.98, top=0.95, bottom=0.02)
    return fig

# ── Layout ───────────────────────────────────────────────────────────────────
col1, spacer, col2 = st.columns([1, 0.08, 1])

with col1:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    canvas = st_canvas(
        fill_color="black", stroke_width=18, stroke_color="#FFFFFF",
        background_color="#141413", width=280, height=280,
        drawing_mode="freedraw", key="canvas",
    )


with col2:
    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)

    if has_drawing(canvas.image_data):
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("L").resize((28, 28))
        tensor = transforms.Normalize((0.1307,), (0.3081,))(
            torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        )

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze().numpy()
            activations = model.get_activations(tensor)

        pred, conf = int(probs.argmax()), probs.max() * 100

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Detected Digit</div>
            <div class="predicted-digit">{pred}</div>
            <div class="confidence-badge">{conf:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.25rem;"><div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)

        bars = ""
        for i in range(10):
            pct = probs[i] * 100
            bars += f'<div class="prob-row"><span class="prob-digit">{i}</span><div class="prob-track"><div class="prob-fill {"accent" if i == pred else "default"}" style="width:{pct}%"></div></div><span class="prob-pct">{pct:.1f}%</span></div>'
        st.markdown(bars + "</div>", unsafe_allow_html=True)
    else:
        activations = None
        st.markdown("""
        <div class="result-card"><div class="empty-state">
            <div class="empty-icon">✎</div>
            <div class="empty-title">No input detected</div>
            <div class="empty-sub">Draw a digit on the canvas to begin</div>
        </div></div>
        """, unsafe_allow_html=True)

# ── Layer Activations (stemmed from diagram) ─────────────────────────────────
if has_drawing(canvas.image_data) and activations is not None:
    st.markdown("---")
    st.markdown('<div class="section-label">Network Activations</div>', unsafe_allow_html=True)


    layer_info = [
        {"name": "Conv Block 1", "detail": "16 filters · 14×14 output", "color": "#d97757", "act": activations[0]},
        {"name": "Conv Block 2", "detail": "32 filters · 7×7 output",  "color": "#6a9bcc", "act": activations[1]},
    ]

    a_col1, a_col2 = st.columns(2)

    for col, info in zip([a_col1, a_col2], layer_info):
        with col:
            st.markdown(f"""
            <div class="act-section" style="border-top: 3px solid {info['color']};">
                <div class="act-header">
                    <div class="act-dot" style="background:{info['color']};"></div>
                    <div class="act-title">{info['name']}</div>
                    <div class="act-subtitle">{info['detail']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig = activations_to_fig(info["act"], info["color"])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">CNN · 2 Conv layers · Trained on MNIST with augmentation · 98%+ accuracy</div>', unsafe_allow_html=True)
