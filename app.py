import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import requests
from langdetect import detect, LangDetectException

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════
MAX_LEN = 100

tfidf           = joblib.load("tfidf_vectorizer.pkl")
model_rlo       = joblib.load("model_rlo.pkl")
model_lstm      = load_model("model_lstm.h5")
tokenizer_lstm  = joblib.load("tokenizer_lstm.pkl")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION ÉMOTIONS
# ═══════════════════════════════════════════════════════════════
EMOTIONS = {
    0: ("Sadness",  "#6B9FD4"),
    1: ("Joy",      "#F5C842"),
    2: ("Love",     "#E87070"),
    3: ("Anger",    "#E07B39"),
    4: ("Fear",     "#9B7FD4"),
    5: ("Surprise", "#5FC4A0"),
    6: ("Neutral",  "#AAAAAA"),
}

# ═══════════════════════════════════════════════════════════════
# TRADUCTION : langdetect (offline) + MyMemory (API gratuite)
# ═══════════════════════════════════════════════════════════════

LANG_NAMES = {
    "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ar": "Arabic",
    "zh-cn": "Chinese", "ja": "Japanese", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "ko": "Korean", "en": "English",
}

def detect_language(text: str) -> str:
    """Détecte la langue du texte via langdetect (100% offline)."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Traduit le texte vers l'anglais via MyMemory API (gratuite).
    Limite : ~1000 mots/jour sans clé, ~10 000 mots/jour avec email.
    """
    if source_lang == "en":
        return text

    try:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={
                "q": text,
                "langpair": f"{source_lang}|en",
                # Décommente et remplis pour augmenter la limite à 10k mots/jour :
                # "de": "ton@email.com",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # MyMemory retourne un code 200 même en cas d'erreur, on vérifie le statut
        if data.get("responseStatus") == 200:
            return data["responseData"]["translatedText"]
        else:
            # En cas d'erreur API, on retourne le texte original
            print(f"[MyMemory] Warning: {data.get('responseDetails', 'Unknown error')}")
            return text

    except requests.exceptions.Timeout:
        print("[MyMemory] Timeout — using original text")
        return text
    except Exception as e:
        print(f"[MyMemory] Error: {e} — using original text")
        return text

# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="Spectral · Emotion Analysis", version="2.0")

# ═══════════════════════════════════════════════════════════════
# HTML FRONTEND
# ═══════════════════════════════════════════════════════════════
HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spectral · Analyse Émotionnelle</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
tailwind.config = {
  theme: {
    extend: {
      fontFamily: {
        'display': ['Space Grotesk', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      }
    }
  }
}
</script>
<style>
  :root {
    --emotion-color: #64748b;
    --emotion-glow: rgba(100, 116, 139, 0.4);
  }
  .spectral-grid {
    background-image:
      linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
  }
  @keyframes scanPass {
    0% { transform: translateY(-100%); opacity: 0; }
    20% { opacity: 1; }
    80% { opacity: 1; }
    100% { transform: translateY(400%); opacity: 0; }
  }
  .scan-line { animation: scanPass 2s ease-in-out forwards; }
  @keyframes glowPulse {
    0%, 100% {
      box-shadow: 0 0 20px var(--emotion-glow), 0 0 40px var(--emotion-glow), inset 0 0 20px var(--emotion-glow);
    }
    50% {
      box-shadow: 0 0 40px var(--emotion-glow), 0 0 80px var(--emotion-glow), inset 0 0 40px var(--emotion-glow);
    }
  }
  .glow-active { animation: glowPulse 3s ease-in-out infinite; }
  @keyframes textReveal {
    from { opacity: 0; transform: translateY(20px); filter: blur(10px); }
    to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
  }
  .reveal-text { animation: textReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
  @keyframes freqEnter {
    from { transform: scaleY(0); opacity: 0; }
    to   { transform: scaleY(1); opacity: 1; }
  }
  .freq-bar { transform-origin: bottom; animation: freqEnter 1s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
  ::-webkit-scrollbar       { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #000; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: #555; }
  .noise::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    opacity: 0.03; pointer-events: none; z-index: 9999;
  }
  @keyframes ringPulse {
    0%   { transform: scale(0.8); opacity: 0.8; }
    100% { transform: scale(1.5); opacity: 0; }
  }
  .spectral-ring { animation: ringPulse 2s ease-out infinite; }
  @keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); max-height: 0; }
    to   { opacity: 1; transform: translateY(0);     max-height: 100px; }
  }
  .translation-banner { animation: slideDown 0.5s ease-out forwards; overflow: hidden; }
  @keyframes langPulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
  }
  .lang-detecting { animation: langPulse 1s ease-in-out infinite; }
</style>
</head>

<body class="bg-black text-white font-display antialiased overflow-x-hidden noise">

  <div id="atmosphere" class="fixed inset-0 pointer-events-none transition-all duration-1000"
       style="background: radial-gradient(ellipse 100% 100% at 50% 0%, var(--emotion-glow), transparent 70%);"></div>
  <div class="fixed inset-0 spectral-grid pointer-events-none opacity-50"></div>

  <div class="relative z-10 min-h-screen flex flex-col">

    <!-- Header -->
    <header class="pt-12 pb-8 px-6">
      <div class="max-w-5xl mx-auto">
        <div class="flex items-center justify-between mb-8">
          <div>
            <p class="font-mono text-[10px] tracking-[0.4em] text-white/40 uppercase">Neural Analysis Engine</p>
            <h1 class="text-3xl md:text-4xl font-bold tracking-tight mt-1">Spectral</h1>
          </div>
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            <span class="font-mono text-xs text-white/40">v2.5.0</span>
          </div>
        </div>
      </div>
    </header>

    <main class="flex-1 px-6 pb-32">
      <div class="max-w-5xl mx-auto">

        <div class="relative">

          <!-- Scan Line -->
          <div id="scanLine" class="absolute inset-0 pointer-events-none overflow-hidden opacity-0 z-20">
            <div class="scan-line absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-white to-transparent"></div>
          </div>

          <!-- Input Card -->
          <div id="inputCard" class="relative bg-white/[0.02] border border-white/10 rounded-2xl overflow-hidden backdrop-blur-xl transition-all duration-500">
            <!-- Top Bar -->
            <div class="flex items-center justify-between px-5 py-3 border-b border-white/5 bg-white/[0.02]">
              <div class="flex items-center gap-3">
                <div class="flex gap-1.5">
                  <div class="w-3 h-3 rounded-full bg-white/10"></div>
                  <div class="w-3 h-3 rounded-full bg-white/10"></div>
                  <div class="w-3 h-3 rounded-full bg-white/10"></div>
                </div>
                <span class="font-mono text-[10px] text-white/30 tracking-widest">INPUT_STREAM</span>
              </div>
              <div class="flex items-center gap-3">
                <!-- Language Badge (live detection) -->
                <div id="langBadge" class="flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/5 border border-white/10 opacity-0 transition-opacity duration-300">
                  <div class="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
                  <span id="langBadgeText" class="font-mono text-[9px] text-white/50 uppercase tracking-widest">--</span>
                </div>
                <span class="font-mono text-[10px] text-white/30" id="charCount">0 chars</span>
              </div>
            </div>

            <!-- Textarea -->
            <div class="p-6">
              <textarea
                id="textInput"
                placeholder="Entrez votre texte ici... / Enter your text here... / Escribe tu texto aquí..."
                class="w-full h-40 bg-transparent border-0 outline-none resize-none text-white/90 placeholder:text-white/20 font-mono text-sm leading-relaxed focus:outline-none"
              ></textarea>
            </div>

            <!-- Bottom Bar -->
            <div class="flex items-center justify-between px-5 py-4 border-t border-white/5 bg-black/20">
              <div class="flex items-center gap-4">
                <span id="statusText" class="font-mono text-[10px] text-white/30">MULTILINGUAL READY</span>
                <span class="font-mono text-[10px] text-white/20">|</span>
                <span class="font-mono text-[10px] text-white/30">2 MODELS READY</span>
              </div>
              <button
                onclick="analyze()"
                id="analyzeBtn"
                class="group relative px-8 py-3 bg-white text-black font-mono text-xs tracking-widest uppercase rounded-lg overflow-hidden transition-all duration-300 hover:bg-white/90"
              >
                <span class="relative z-10 flex items-center gap-2">
                  <svg class="w-4 h-4 transition-transform group-hover:rotate-90 duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                  </svg>
                  Analyze
                </span>
                <div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
              </button>
            </div>
          </div>

          <!-- Error Message -->
          <div id="errorMsg" class="absolute -bottom-10 left-0 right-0 text-center">
            <span class="font-mono text-xs text-red-400 opacity-0 transition-opacity" id="errorText">Please enter text to analyze</span>
          </div>
        </div>

        <!-- ── Translation Banner (shown when non-English input) ── -->
        <div id="translationBanner" class="hidden mt-4">
          <div class="translation-banner flex items-center gap-3 px-5 py-3 rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-xl">
            <svg class="w-4 h-4 text-white/40 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"/>
            </svg>
            <div class="flex-1 min-w-0">
              <span class="font-mono text-[10px] text-white/40 uppercase tracking-wider">Auto-translated · </span>
              <span id="detectedLangLabel" class="font-mono text-[10px] text-white/60 uppercase tracking-wider">--</span>
              <span class="font-mono text-[10px] text-white/40"> → English</span>
            </div>
            <div class="flex-shrink-0 max-w-xs truncate">
              <span class="font-mono text-[10px] text-white/30 italic" id="translatedPreview"></span>
            </div>
          </div>
        </div>

        <!-- ── Results Section ── -->
        <div id="resultsSection" class="mt-12 opacity-0 translate-y-10 transition-all duration-1000 pointer-events-none">

          <div class="flex items-center justify-between mb-8">
            <div>
              <p class="font-mono text-[10px] tracking-[0.3em] text-white/30 uppercase mb-2">Analysis Results</p>
              <h2 class="text-2xl font-bold tracking-tight">Emotional Spectrum</h2>
            </div>
            <div class="flex items-center gap-2">
              <div class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
              <span class="font-mono text-[10px] text-white/40">PROCESSED</span>
            </div>
          </div>

          <!-- Consensus Card -->
          <div id="consensusCard" class="relative mb-10 rounded-2xl border border-white/10 overflow-hidden bg-white/[0.02] backdrop-blur-xl">
            <div class="absolute inset-0 opacity-30" style="background: radial-gradient(ellipse at center, var(--emotion-glow), transparent 70%);"></div>
            <div class="relative p-8 md:p-12">
              <div class="flex flex-col md:flex-row items-center gap-8">
                <div class="relative flex-shrink-0">
                  <div class="absolute inset-0 flex items-center justify-center">
                    <div class="spectral-ring absolute w-32 h-32 rounded-full border border-current opacity-30" style="color: var(--emotion-color);"></div>
                    <div class="spectral-ring absolute w-32 h-32 rounded-full border border-current opacity-20" style="color: var(--emotion-color); animation-delay: 0.5s;"></div>
                  </div>
                  <div id="consensusIcon" class="relative w-24 h-24 flex items-center justify-center rounded-full border-2 transition-all duration-500" style="border-color: var(--emotion-color);"></div>
                </div>
                <div class="flex-1 text-center md:text-left">
                  <p class="font-mono text-[10px] tracking-[0.3em] text-white/40 uppercase mb-2">Primary Emotion</p>
                  <h3 id="consensusLabel" class="text-5xl md:text-6xl font-bold tracking-tight mb-4 transition-colors duration-500" style="color: var(--emotion-color);"></h3>
                  <p id="consensusMeta" class="font-mono text-sm text-white/50"></p>
                </div>
                <div class="flex-shrink-0 w-32">
                  <div class="text-center mb-2">
                    <span id="consensusProb" class="font-mono text-3xl font-bold transition-colors duration-500" style="color: var(--emotion-color);">0%</span>
                  </div>
                  <div class="h-2 bg-white/10 rounded-full overflow-hidden">
                    <div id="consensusBar" class="h-full rounded-full transition-all duration-1000 ease-out" style="width: 0%; background: var(--emotion-color);"></div>
                  </div>
                  <p class="font-mono text-[9px] text-white/30 text-center mt-2 tracking-wider">CONFIDENCE</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Model Cards -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <!-- RLO -->
            <div class="model-card group relative bg-white/[0.02] border border-white/10 rounded-xl overflow-hidden backdrop-blur-xl transition-all duration-300 hover:border-white/20 hover:bg-white/[0.04]">
              <div class="p-5">
                <div class="flex items-center justify-between mb-6">
                  <div>
                    <p class="font-mono text-[9px] tracking-[0.2em] text-white/30 uppercase">Model 01</p>
                    <h4 class="text-lg font-semibold mt-1">Logistic Reg.</h4>
                  </div>
                  <div class="text-right">
                    <p class="font-mono text-[9px] text-white/30">ACCURACY</p>
                    <p class="font-mono text-sm text-white/60">75.23%</p>
                  </div>
                </div>
                <div id="rlo-prediction" class="mb-6"></div>
                <div class="mt-4">
                  <p class="font-mono text-[9px] tracking-wider text-white/30 mb-3">PROBABILITY DISTRIBUTION</p>
                  <div id="rlo-bars" class="flex items-end justify-between h-20 gap-1"></div>
                  <div id="rlo-labels" class="flex justify-between mt-2"></div>
                </div>
              </div>
            </div>
            <!-- LSTM -->
            <div class="model-card group relative bg-white/[0.02] border border-white/10 rounded-xl overflow-hidden backdrop-blur-xl transition-all duration-300 hover:border-white/20 hover:bg-white/[0.04]">
              <div class="p-5">
                <div class="flex items-center justify-between mb-6">
                  <div>
                    <p class="font-mono text-[9px] tracking-[0.2em] text-white/30 uppercase">Model 02</p>
                    <h4 class="text-lg font-semibold mt-1">Bi-LSTM</h4>
                  </div>
                  <div class="text-right">
                    <p class="font-mono text-[9px] text-white/30">ACCURACY</p>
                    <p class="font-mono text-sm text-white/60">N/A</p>
                  </div>
                </div>
                <div id="lstm-prediction" class="mb-6"></div>
                <div class="mt-4">
                  <p class="font-mono text-[9px] tracking-wider text-white/30 mb-3">PROBABILITY DISTRIBUTION</p>
                  <div id="lstm-bars" class="flex items-end justify-between h-20 gap-1"></div>
                  <div id="lstm-labels" class="flex justify-between mt-2"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Detailed Breakdown -->
          <div class="mt-8 bg-white/[0.02] border border-white/10 rounded-xl overflow-hidden backdrop-blur-xl">
            <div class="px-5 py-3 border-b border-white/5 flex items-center justify-between">
              <span class="font-mono text-[10px] tracking-[0.2em] text-white/40">DETAILED BREAKDOWN</span>
              <div class="flex items-center gap-2">
                <div class="w-1.5 h-1.5 rounded-full bg-white/30"></div>
                <span class="font-mono text-[10px] text-white/30">7 EMOTIONS TRACKED</span>
              </div>
            </div>
            <div class="p-6">
              <!-- Column headers -->
              <div class="flex items-center gap-4 mb-3 px-3">
                <div class="w-28 flex-shrink-0"></div>
                <div class="flex-1 grid grid-cols-3 gap-3">
                  <span class="font-mono text-[9px] text-white/25 uppercase">LogReg</span>
                  <span class="font-mono text-[9px] text-white/25 uppercase">LSTM</span>
                  <span class="font-mono text-[9px] text-white/25 uppercase text-right">Avg</span>
                </div>
              </div>
              <div id="detailedBreakdown" class="space-y-3"></div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>

<script>
// ── Emotion Config ──────────────────────────────────────────
const EMOTIONS = {
  0: { name: "Sadness",  color: "#6B9FD4", glow: "rgba(107, 159, 212, 0.3)" },
  1: { name: "Joy",      color: "#F5C842", glow: "rgba(245, 200, 66, 0.3)" },
  2: { name: "Love",     color: "#E87070", glow: "rgba(232, 112, 112, 0.35)" },
  3: { name: "Anger",    color: "#E07B39", glow: "rgba(224, 123, 57, 0.3)" },
  4: { name: "Fear",     color: "#9B7FD4", glow: "rgba(155, 127, 212, 0.3)" },
  5: { name: "Surprise", color: "#5FC4A0", glow: "rgba(95, 196, 160, 0.3)" },
  6: { name: "Neutral",  color: "#94A3B8", glow: "rgba(148, 163, 184, 0.2)" },
};

const LANG_NAMES = {
  fr: "French", es: "Spanish", de: "German", it: "Italian",
  pt: "Portuguese", ar: "Arabic", zh: "Chinese", ja: "Japanese",
  ru: "Russian", nl: "Dutch", pl: "Polish", tr: "Turkish",
  ko: "Korean", en: "English", mg: "Malagasy",
};

const SVG_ICONS = {
  "Sadness": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><path d="M8 15s1.5-2 4-2 4 2 4 2"/><line x1="9" y1="9" x2="9.01" y2="9" stroke-width="2.5" stroke-linecap="round"/><line x1="15" y1="9" x2="15.01" y2="9" stroke-width="2.5" stroke-linecap="round"/></svg>`,
  "Joy":     `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9" stroke-width="2.5" stroke-linecap="round"/><line x1="15" y1="9" x2="15.01" y2="9" stroke-width="2.5" stroke-linecap="round"/></svg>`,
  "Love":    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>`,
  "Anger":   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><path d="M16 16s-1.5-2-4-2-4 2-4 2"/><path d="M7.5 8L10 9"/><path d="M16.5 8L14 9"/></svg>`,
  "Fear":    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><path d="M8 15h8"/><circle cx="9" cy="9" r="1" fill="currentColor"/><circle cx="15" cy="9" r="1" fill="currentColor"/><path d="M10 9V8"/><path d="M14 9V8"/></svg>`,
  "Surprise":`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="15" r="2" fill="currentColor"/><circle cx="9" cy="9" r="1" fill="currentColor"/><circle cx="15" cy="9" r="1" fill="currentColor"/></svg>`,
  "Neutral": `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="w-12 h-12"><circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/><circle cx="9" cy="9" r="1" fill="currentColor"/><circle cx="15" cy="9" r="1" fill="currentColor"/></svg>`,
};

// ── Live language detection in textarea ────────────────────
let detectTimeout = null;
document.getElementById('textInput').addEventListener('input', function() {
  document.getElementById('charCount').textContent = this.value.length + ' chars';

  clearTimeout(detectTimeout);
  const val = this.value.trim();

  if (val.length < 5) {
    document.getElementById('langBadge').style.opacity = '0';
    document.getElementById('statusText').textContent = 'MULTILINGUAL READY';
    return;
  }

  // Debounce 400ms then call /detect
  detectTimeout = setTimeout(async () => {
    try {
      const fd = new FormData();
      fd.append('text', val);
      const res = await fetch('/detect', { method: 'POST', body: fd });
      const d = await res.json();
      const langName = LANG_NAMES[d.lang] || d.lang.toUpperCase();
      const badge = document.getElementById('langBadge');
      document.getElementById('langBadgeText').textContent = langName;
      badge.style.opacity = '1';
      document.getElementById('statusText').textContent =
        d.lang === 'en' ? 'ENGLISH DETECTED · NO TRANSLATION NEEDED' : `${langName.toUpperCase()} DETECTED · WILL AUTO-TRANSLATE`;
    } catch(e) {}
  }, 400);
});

// ── Analyze ────────────────────────────────────────────────
async function analyze() {
  const text = document.getElementById('textInput').value.trim();
  const errorText = document.getElementById('errorText');
  const scanLine = document.getElementById('scanLine');
  const inputCard = document.getElementById('inputCard');
  const resultsSection = document.getElementById('resultsSection');
  const btn = document.getElementById('analyzeBtn');
  const translationBanner = document.getElementById('translationBanner');

  if (!text) {
    errorText.style.opacity = '1';
    setTimeout(() => errorText.style.opacity = '0', 2000);
    return;
  }

  errorText.style.opacity = '0';
  translationBanner.classList.add('hidden');
  resultsSection.classList.add('opacity-0', 'translate-y-10');
  resultsSection.classList.remove('pointer-events-none');

  scanLine.style.opacity = '1';
  inputCard.classList.add('glow-active');
  btn.disabled = true;
  btn.innerHTML = `<span class="flex items-center gap-2"><svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path></svg>Processing...</span>`;

  const fd = new FormData();
  fd.append('text', text);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();

    // Show translation banner if text was translated
    if (data.detected_lang && data.detected_lang !== 'en' && data.text_translated) {
      const langName = LANG_NAMES[data.detected_lang] || data.detected_lang.toUpperCase();
      document.getElementById('detectedLangLabel').textContent = langName;
      const preview = data.text_translated.length > 60
        ? data.text_translated.substring(0, 60) + '...'
        : data.text_translated;
      document.getElementById('translatedPreview').textContent = '"' + preview + '"';
      translationBanner.classList.remove('hidden');
    }

    renderResults(data);
  } catch(e) {
    alert('Analysis error. Please try again.');
  } finally {
    scanLine.style.opacity = '0';
    inputCard.classList.remove('glow-active');
    btn.disabled = false;
    btn.innerHTML = `<span class="relative z-10 flex items-center gap-2"><svg class="w-4 h-4 transition-transform group-hover:rotate-90 duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>Analyze</span><div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>`;
  }
}

function renderResults(data) {
  const resultsSection = document.getElementById('resultsSection');
  const atmosphere = document.getElementById('atmosphere');
  const consEmotion = EMOTIONS[data.consensus.vote];
  document.documentElement.style.setProperty('--emotion-color', consEmotion.color);
  document.documentElement.style.setProperty('--emotion-glow', consEmotion.glow);
  atmosphere.style.background = `radial-gradient(ellipse 100% 100% at 50% 0%, ${consEmotion.glow}, transparent 70%)`;

  document.getElementById('consensusIcon').innerHTML = SVG_ICONS[consEmotion.name];
  document.getElementById('consensusIcon').style.borderColor = consEmotion.color;
  document.getElementById('consensusIcon').style.color = consEmotion.color;
  document.getElementById('consensusLabel').textContent = consEmotion.name;
  document.getElementById('consensusLabel').style.color = consEmotion.color;

  const meta = data.consensus.count >= 2
    ? `Consensus reached (${data.consensus.count}/2 models agree)`
    : 'Models disagree — results uncertain';
  document.getElementById('consensusMeta').textContent = meta;

  const maxProb = Math.max(...data.rlo.proba);
  const probPercent = (maxProb * 100).toFixed(1);
  document.getElementById('consensusProb').textContent = probPercent + '%';
  document.getElementById('consensusProb').style.color = consEmotion.color;
  setTimeout(() => {
    document.getElementById('consensusBar').style.width = probPercent + '%';
    document.getElementById('consensusBar').style.background = consEmotion.color;
  }, 100);

  renderModelCard('rlo', data.rlo);
  renderModelCard('lstm', data.lstm);
  renderDetailedBreakdown(data);

  setTimeout(() => {
    resultsSection.classList.remove('opacity-0', 'translate-y-10');
  }, 300);
}

function renderModelCard(modelKey, modelData) {
  const emotion = EMOTIONS[modelData.pred];
  document.getElementById(modelKey + '-prediction').innerHTML = `
    <div class="flex items-center gap-3 p-3 rounded-lg" style="background: ${emotion.glow};">
      <div style="color: ${emotion.color}">${SVG_ICONS[emotion.name]}</div>
      <div>
        <p class="font-mono text-[9px] text-white/40 uppercase tracking-wider">Predicted</p>
        <p class="text-xl font-semibold" style="color: ${emotion.color}">${emotion.name}</p>
      </div>
    </div>`;

  const barsEl = document.getElementById(modelKey + '-bars');
  const labelsEl = document.getElementById(modelKey + '-labels');
  barsEl.innerHTML = '';
  labelsEl.innerHTML = '';

  modelData.proba.forEach((prob, i) => {
    const e = EMOTIONS[i];
    const height = Math.max(prob * 100, 2);
    barsEl.innerHTML += `
      <div class="flex-1 flex flex-col items-center">
        <div class="w-full h-full flex items-end">
          <div class="freq-bar w-full rounded-t" style="height: ${height}%; background: ${e.color}; animation-delay: ${i * 50}ms;"></div>
        </div>
      </div>`;
    labelsEl.innerHTML += `<span class="font-mono text-[8px] text-white/30">${e.name.substring(0,3)}</span>`;
  });
}

function renderDetailedBreakdown(data) {
  const container = document.getElementById('detailedBreakdown');
  container.innerHTML = '';
  for (let i = 0; i < 7; i++) {
    const e = EMOTIONS[i];
    const rloP  = (data.rlo.proba[i]  * 100).toFixed(1);
    const lstmP = (data.lstm.proba[i] * 100).toFixed(1);
    const avgP  = ((+rloP + +lstmP) / 2).toFixed(1);
    container.innerHTML += `
      <div class="flex items-center gap-4 p-3 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
        <div class="w-28 flex items-center gap-2 flex-shrink-0">
          <span style="color: ${e.color}">${SVG_ICONS[e.name].replace('w-12 h-12','w-5 h-5')}</span>
          <span class="text-sm font-medium">${e.name}</span>
        </div>
        <div class="flex-1 grid grid-cols-3 gap-3">
          ${[rloP, lstmP].map(p => `
          <div>
            <div class="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div class="h-full rounded-full transition-all duration-700" style="width:${p}%;background:${e.color};"></div>
            </div>
            <span class="font-mono text-[10px] text-white/40">${p}%</span>
          </div>`).join('')}
          <div class="text-right">
            <span class="font-mono text-sm font-medium" style="color:${e.color}">${avgP}%</span>
            <p class="font-mono text-[8px] text-white/30">AVG</p>
          </div>
        </div>
      </div>`;
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.ctrlKey) analyze();
});
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/detect")
async def detect_endpoint(text: str = Form(...)):
    """
    Route légère pour la détection de langue en live (pendant la saisie).
    Utilise uniquement langdetect (offline, rapide).
    """
    lang = detect_language(text)
    return {"lang": lang}


@app.post("/analyze")
async def analyze_endpoint(text: str = Form(...)):
    # ── 1. Détection de langue (offline, ~1ms) ──────────────────
    detected_lang = detect_language(text)

    # ── 2. Traduction vers l'anglais si nécessaire ──────────────
    text_en = translate_to_english(text, detected_lang)

    # ── 3. Pipeline ML ──────────────────────────────────────────
    vec       = tfidf.transform([text_en])
    pred_rlo  = int(model_rlo.predict(vec)[0])
    proba_rlo = model_rlo.predict_proba(vec)[0]

    seq        = tokenizer_lstm.texts_to_sequences([text_en])
    pad_seq    = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba_lstm = model_lstm.predict(pad_seq, verbose=0)[0]
    pred_lstm  = int(np.argmax(proba_lstm))

    preds = [pred_rlo, pred_lstm]
    vote, count = Counter(preds).most_common(1)[0]

    return {
        "rlo":  {"pred": pred_rlo,  "proba": [float(p) for p in proba_rlo],  "acc": 0.7523},
        "lstm": {"pred": pred_lstm, "proba": [float(p) for p in proba_lstm], "acc": 0.0},
        "consensus":       {"vote": vote, "count": count},
        # ── Infos traduction ─────────────────────────────────────
        "detected_lang":   detected_lang,
        "text_translated": text_en if detected_lang != "en" else None,
    }


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)