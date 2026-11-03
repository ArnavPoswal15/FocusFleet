"""Visual theme: CSS and static HTML fragments for the FocusFleet UI."""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg-0: #08090b;
    --bg-1: #0f1114;
    --bg-2: #15171b;
    --surface: rgba(255,255,255,0.035);
    --surface-hover: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.14);
    --text-0: #f4f5f7;
    --text-1: #a7adb8;
    --text-2: #6b7280;
    --accent: #5eead4;
    --accent-dim: #14b8a6;
    --accent-glow: rgba(94, 234, 212, 0.35);
    --danger: #f87171;
    --warn: #fbbf24;
    --radius: 16px;
}

* , *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: radial-gradient(circle at 20% 0%, #101418 0%, var(--bg-0) 45%) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-0) !important;
}

.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

/* ===== HEADER ===== */
#ff-header {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3rem 1rem 2rem;
    text-align: center;
}
#ff-header .ff-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-1);
    margin-bottom: 1.4rem;
}
#ff-header .ff-badge .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent-glow);
}
#ff-header h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    letter-spacing: -0.04em;
    line-height: 1.05;
    color: var(--text-0);
}
#ff-header h1 span {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#ff-header .ff-sub {
    color: var(--text-1);
    font-size: 1rem;
    margin: 0.9rem 0 0;
    max-width: 480px;
    font-weight: 400;
}

/* ===== TABS ===== */
.tabs { border: none !important; background: transparent !important; }
.tab-nav {
    display: flex !important;
    justify-content: center !important;
    gap: 0.4rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.35rem !important;
    max-width: fit-content;
    margin: 0 auto 2rem !important;
}
.tab-nav button {
    border: none !important;
    background: transparent !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: var(--surface-hover) !important;
    color: var(--text-0) !important;
    box-shadow: inset 0 0 0 1px var(--border-strong);
}

/* ===== CARDS / PANELS ===== */
.block, .form, fieldset {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
label, .gr-label, span.svelte-1gfkn6j { color: var(--text-1) !important; }

/* ===== INPUTS ===== */
input[type="text"], input[type="password"], textarea {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-0) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="text"]:focus, input[type="password"]:focus, textarea:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.15) !important;
}

/* ===== BUTTONS ===== */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
    border: none !important;
    color: #04231f !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px rgba(20, 184, 166, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(20, 184, 166, 0.4) !important;
}
button.secondary, .gr-button-secondary {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-0) !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
button.secondary:hover { background: var(--surface-hover) !important; }

/* ===== AUTH SECTION LABELS ===== */
.auth-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ===== STATUS BANNER ===== */
@keyframes drowsy-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.0); }
    50%       { box-shadow: 0 0 0 8px rgba(248, 113, 113, 0.12); }
}
.status-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    width: 100%;
    letter-spacing: 0.01em;
    transition: background 0.4s ease, border-color 0.4s ease, color 0.4s ease;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.status-drowsy {
    background: rgba(248, 113, 113, 0.1) !important;
    color: var(--danger) !important;
    border: 1px solid rgba(248, 113, 113, 0.3) !important;
}
.status-drowsy .status-dot { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
.status-drowsy-blink { animation: drowsy-pulse 1.2s ease-in-out infinite !important; }
.status-alert {
    background: rgba(94, 234, 212, 0.08) !important;
    color: var(--accent) !important;
    border: 1px solid rgba(94, 234, 212, 0.25) !important;
}
.status-alert .status-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent-glow); }
.status-warning {
    background: rgba(251, 191, 36, 0.08) !important;
    color: var(--warn) !important;
    border: 1px solid rgba(251, 191, 36, 0.25) !important;
}
.status-warning .status-dot { background: var(--warn); }
.status-neutral {
    background: rgba(255,255,255,0.03) !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border) !important;
}
.status-neutral .status-dot { background: var(--text-2); }

/* ===== SLIDER ===== */
#ear-slider { width: 100% !important; padding: 0.5rem 0 !important; }
input[type="range"] { accent-color: var(--accent) !important; }

/* ===== STATUS HTML WRAPPER ===== */
#status-html, #status-html > div { background: transparent !important; border: none !important; padding: 0 !important; }

/* ===== CAMERA ROW ===== */
#monitor-row { gap: 1.2rem !important; }
#monitor-row .image-container, .image-frame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ===== WELCOME BANNER ===== */
#welcome-msg {
    padding: 0.85rem 1.2rem;
    border-radius: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: var(--text-1);
}

/* ===== UPLOAD RESULT ===== */
#upload-result textarea {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: var(--accent) !important;
    background: var(--bg-2) !important;
}

footer { display: none !important; }
"""

HEADER_HTML = """
<div id="ff-header">
  <div class="ff-badge"><span class="dot"></span>LIVE DETECTION SYSTEM</div>
  <h1>Focus<span>Fleet</span></h1>
  <p class="ff-sub">AI-powered driver drowsiness detection, built to keep you alert and safe behind the wheel.</p>
</div>
"""
