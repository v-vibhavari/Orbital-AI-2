/**
 * SatQuery AI - Frontend Application Logic
 * Handles image upload/preview, preset imagery, quick query chips,
 * API dispatch to POST /analyze, and intelligent mock fallback.
 */

// DOM Elements
const imageInput = document.getElementById('imageInput');
const dropzone = document.getElementById('dropzone');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const resetImageBtn = document.getElementById('resetImageBtn');
const metaFileName = document.getElementById('metaFileName');
const metaDimensions = document.getElementById('metaDimensions');

const questionInput = document.getElementById('questionInput');
const quickChips = document.querySelectorAll('.chip-btn');
const presetBtns = document.querySelectorAll('.preset-btn');

const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const responseContainer = document.getElementById('responseContainer');
const loadingState = document.getElementById('loadingState');
const responseCard = document.getElementById('responseCard');
const responseBody = document.getElementById('responseBody');
const responseSourceBadge = document.getElementById('responseSourceBadge');
const copyResponseBtn = document.getElementById('copyResponseBtn');
const analysisTimestamp = document.getElementById('analysisTimestamp');

// State
let currentFile = null;
let currentImageSrc = null;
let currentPresetKey = null;

// Backend API URL (FastAPI default)
const API_URL = 'http://127.0.0.1:8000/analyze';

/* -------------------------------------------------------------
 * 1. Preset Satellite Imagery Generator (Canvas-based SVGs)
 * ------------------------------------------------------------- */
const SAMPLE_PRESETS = {
  flood: {
    title: 'flood_image.png',
    resolution: '1920 x 1080 (10m GSD - Sentinel-2)',
    category: 'flood',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">
      <defs>
        <linearGradient id="floodWater" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#0284c7"/><stop offset="100%" stop-color="#0369a1"/></linearGradient>
        <linearGradient id="submergedField" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0e7490"/><stop offset="100%" stop-color="#155e75"/></linearGradient>
      </defs>
      <!-- Background rural terrain -->
      <rect width="800" height="500" fill="#4d7c0f"/>
      <!-- Submerged floodplain zones -->
      <path d="M0 100 Q 200 60, 400 130 T 800 110 L 800 390 Q 600 420, 380 340 T 0 380 Z" fill="url(#submergedField)" opacity="0.85"/>
      <!-- Main Swollen River Channel -->
      <path d="M-20 200 Q 220 120, 420 250 T 820 220" stroke="url(#floodWater)" stroke-width="75" fill="none" stroke-linecap="round"/>
      <!-- Inundated agricultural parcels -->
      <rect x="60" y="80" width="110" height="70" rx="4" fill="#0891b2" opacity="0.75" stroke="#164e63" stroke-width="2"/>
      <rect x="220" y="70" width="130" height="60" rx="4" fill="#0e7490" opacity="0.8" stroke="#164e63" stroke-width="2"/>
      <rect x="520" y="270" width="140" height="85" rx="4" fill="#0891b2" opacity="0.75" stroke="#164e63" stroke-width="2"/>
      <rect x="100" y="320" width="120" height="70" rx="4" fill="#0e7490" opacity="0.8" stroke="#164e63" stroke-width="2"/>
      <!-- Severed Highway -->
      <path d="M0 450 L 800 50" stroke="#94a3b8" stroke-width="8" stroke-dasharray="180 80 200 40" fill="none"/>
      <!-- Isolated Settlement Pocket -->
      <circle cx="680" cy="180" r="35" fill="#78350f" stroke="#ca8a04" stroke-width="2"/>
      <rect x="665" y="165" width="12" height="12" fill="#f8fafc"/>
      <rect x="685" y="175" width="14" height="12" fill="#f8fafc"/>
      <!-- Water Sediment Plumes -->
      <ellipse cx="440" cy="260" rx="60" ry="25" fill="#ca8a04" opacity="0.5"/>
    </svg>`
  },
  deforestation: {
    title: 'deforestation_image.png',
    resolution: '1920 x 1080 (5m GSD - PlanetScope)',
    category: 'deforestation',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">
      <!-- Intact Rainforest Canopy (Green) -->
      <rect width="800" height="500" fill="#14532d"/>
      <circle cx="120" cy="100" r="140" fill="#052e16"/>
      <circle cx="700" cy="80" r="160" fill="#052e16"/>
      <circle cx="100" cy="420" r="150" fill="#052e16"/>
      <!-- Clear-Cut / Deforested Exposed Soil (Brown/Tan) -->
      <path d="M300 0 L 520 0 L 580 500 L 260 500 Z" fill="#78350f" opacity="0.95"/>
      <rect x="420" y="80" width="180" height="90" rx="4" fill="#9a3412"/>
      <rect x="220" y="240" width="160" height="110" rx="4" fill="#a16207"/>
      <rect x="480" y="320" width="190" height="100" rx="4" fill="#854d0e"/>
      <!-- Fishbone Logging Roads -->
      <path d="M420 0 L 440 500" stroke="#fef08a" stroke-width="6" fill="none"/>
      <path d="M425 90 L 650 60" stroke="#fde047" stroke-width="4" fill="none"/>
      <path d="M430 180 L 200 160" stroke="#fde047" stroke-width="4" fill="none"/>
      <path d="M435 280 L 680 260" stroke="#fde047" stroke-width="4" fill="none"/>
      <path d="M440 380 L 180 370" stroke="#fde047" stroke-width="4" fill="none"/>
      <!-- Logging Staging Clearing -->
      <rect x="420" y="220" width="40" height="40" rx="3" fill="#ca8a04"/>
    </svg>`
  },
  urban: {
    title: 'urban_development.png',
    resolution: '1920 x 1080 (0.5m GSD - WorldView-3)',
    category: 'urban',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">
      <defs>
        <linearGradient id="bgU" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#1e293b"/><stop offset="100%" stop-color="#0f172a"/></linearGradient>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#475569" stroke-width="1.5"/></pattern>
      </defs>
      <rect width="800" height="500" fill="url(#bgU)"/>
      <rect width="800" height="500" fill="url(#grid)" opacity="0.6"/>
      <!-- Main Arterial Avenues -->
      <path d="M0 120 Q400 140 800 110" stroke="#f1f5f9" stroke-width="10" fill="none"/>
      <path d="M0 380 Q400 350 800 370" stroke="#f1f5f9" stroke-width="10" fill="none"/>
      <path d="M260 0 L280 500" stroke="#cbd5e1" stroke-width="8" fill="none"/>
      <path d="M540 0 L520 500" stroke="#cbd5e1" stroke-width="8" fill="none"/>
      <!-- Commercial & Residential Clusters -->
      <g fill="#94a3b8" opacity="0.85">
        <rect x="60" y="30" width="80" height="60" rx="3"/><rect x="160" y="40" width="70" height="50" rx="3"/>
        <rect x="310" y="30" width="90" height="70" rx="3"/><rect x="420" y="25" width="85" height="75" rx="3"/>
        <rect x="580" y="35" width="100" height="60" rx="3"/><rect x="700" y="40" width="70" height="50" rx="3"/>
        <!-- Central Blocks -->
        <rect x="80" y="160" width="140" height="90" rx="4" fill="#64748b"/>
        <rect x="80" y="270" width="140" height="80" rx="4" fill="#475569"/>
        <rect x="310" y="155" width="180" height="180" rx="6" fill="#334155"/>
        <rect x="330" y="180" width="60" height="60" rx="3" fill="#64748b"/>
        <rect x="410" y="180" width="60" height="60" rx="3" fill="#94a3b8"/>
        <rect x="330" y="260" width="140" height="50" rx="3" fill="#0284c7" opacity="0.7"/>
        <rect x="570" y="160" width="90" height="180" rx="4" fill="#64748b"/>
        <rect x="680" y="160" width="90" height="80" rx="4" fill="#475569"/>
        <rect x="680" y="260" width="90" height="90" rx="4" fill="#334155"/>
      </g>
      <!-- Green Parks in City Grid -->
      <circle cx="150" cy="210" r="30" fill="#15803d" opacity="0.85"/>
      <rect x="600" y="410" width="160" height="60" rx="8" fill="#166534" opacity="0.8"/>
    </svg>`
  },
  agriculture: {
    title: 'agricultural_image.png',
    resolution: '2048 x 1536 (3m GSD - PlanetScope)',
    category: 'agriculture',
    svg: `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500">
      <rect width="800" height="500" fill="#78350f" opacity="0.9"/>
      <!-- Irrigation Circles -->
      <circle cx="150" cy="150" r="110" fill="#15803d"/>
      <circle cx="150" cy="150" r="4" fill="#fef08a"/>
      <path d="M150 150 L250 190" stroke="#fef08a" stroke-width="2"/>
      
      <circle cx="390" cy="140" r="100" fill="#16a34a"/>
      <circle cx="620" cy="150" r="105" fill="#ca8a04"/>
      
      <circle cx="160" cy="370" r="100" fill="#65a30d"/>
      <circle cx="390" cy="360" r="100" fill="#15803d"/>
      <circle cx="620" cy="370" r="100" fill="#166534"/>
      <!-- Rural Dirt Roads & Canals -->
      <path d="M0 255 L800 255" stroke="#d97706" stroke-width="6" fill="none"/>
      <path d="M270 0 L270 500" stroke="#d97706" stroke-width="5" fill="none"/>
      <path d="M505 0 L505 500" stroke="#d97706" stroke-width="5" fill="none"/>
      <!-- Primary Irrigation Canal -->
      <path d="M0 480 Q400 470 800 485" stroke="#0284c7" stroke-width="12" fill="none"/>
    </svg>`
  }
};

// Convert SVG string to data URL
function svgToDataUrl(svgString) {
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgString);
}

// Convert data URL to File object for FormData upload
function dataURLtoFile(dataurl, filename) {
  const [metadata, payload = ''] = dataurl.split(',');
  const mime = metadata.match(/:(.*?)(;|$)/)?.[1] || 'image/png';
  const isBase64 = metadata.includes(';base64');

  if (isBase64) {
    const bstr = atob(payload);
    const u8arr = new Uint8Array(bstr.length);
    for (let i = 0; i < bstr.length; i++) {
      u8arr[i] = bstr.charCodeAt(i);
    }
    return new File([u8arr], filename, { type: mime });
  }

  return new File([decodeURIComponent(payload)], filename, { type: mime });
}

/* -------------------------------------------------------------
 * 2. Image Upload & Selection Handling
 * ------------------------------------------------------------- */
function handleFileSelect(file) {
  if (!file || !file.type.startsWith('image/')) {
    alert('Please select a valid satellite or remote sensing image (JPG, PNG, TIFF, WebP).');
    return;
  }

  currentFile = file;
  currentPresetKey = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    currentImageSrc = e.target.result;
    displayImage(currentImageSrc, file.name);
  };
  reader.readAsDataURL(file);
}

function displayImage(src, filename, resolutionText = null) {
  previewImage.src = src;
  metaFileName.textContent = filename;
  metaFileName.title = filename;

  previewContainer.classList.remove('hidden');
  dropzone.classList.add('hidden');
  resetImageBtn.classList.remove('hidden');

  // Compute resolution once image loads
  previewImage.onload = () => {
    if (resolutionText) {
      metaDimensions.textContent = resolutionText;
    } else {
      metaDimensions.textContent = `${previewImage.naturalWidth} x ${previewImage.naturalHeight} px`;
    }
  };
}

function resetImage() {
  currentFile = null;
  currentImageSrc = null;
  currentPresetKey = null;
  imageInput.value = '';
  previewImage.src = '';
  
  previewContainer.classList.add('hidden');
  dropzone.classList.remove('hidden');
  resetImageBtn.classList.add('hidden');

  // Reset scan state if present
  document.querySelector('.image-wrapper')?.classList.remove('scanning');
}

function setLoadingState(isLoading) {
  responseContainer.classList.remove('hidden');
  loadingState.classList.toggle('hidden', !isLoading);
  responseCard.classList.toggle('hidden', isLoading);
  analyzeBtn.disabled = isLoading;
  analyzeBtnText.textContent = isLoading ? 'Analyzing...' : 'Analyze Image';
  document.querySelector('.image-wrapper')?.classList.toggle('scanning', isLoading);
}

// File Input Event
imageInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

// Drag & Drop Events
['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('drag-over');
  });
});

dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

resetImageBtn.addEventListener('click', resetImage);

/* -------------------------------------------------------------
 * 3. Preset Sample Images Activation
 * ------------------------------------------------------------- */
presetBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const sampleKey = btn.getAttribute('data-sample');
    const preset = SAMPLE_PRESETS[sampleKey];
    if (!preset) return;

    currentPresetKey = sampleKey;
    const dataUrl = svgToDataUrl(preset.svg);
    currentImageSrc = dataUrl;
    currentFile = dataURLtoFile(dataUrl, preset.title);
    
    displayImage(dataUrl, preset.title, preset.resolution);

    // Auto-focus question input
    questionInput.focus();
  });
});

/* -------------------------------------------------------------
 * 4. Quick Question Chips
 * ------------------------------------------------------------- */
quickChips.forEach(chip => {
  chip.addEventListener('click', () => {
    const question = chip.getAttribute('data-question');
    questionInput.value = question;
    questionInput.focus();
    
    // Pulse highlight effect on textarea
    questionInput.style.borderColor = '#38bdf8';
    setTimeout(() => {
      questionInput.style.borderColor = '';
    }, 400);
  });
});

/* -------------------------------------------------------------
 * 5. Analyze Dispatch & AI Response
 * ------------------------------------------------------------- */
analyzeBtn.addEventListener('click', async (event) => {

  event.preventDefault();

  // Validations
  if (!currentFile && !currentImageSrc) {
    alert('Please upload a satellite image or choose one of the preset captures.');
    dropzone.scrollIntoView({ behavior: 'smooth' });
    return;
  }

  const question = questionInput.value.trim();

  if (!question) {
    alert('Please enter a question or click one of the quick prompt buttons.');
    questionInput.focus();
    return;
  }

  // Set UI to Loading State
  setLoadingState(true);

  try {
    if (currentPresetKey) {
      await new Promise(resolve => setTimeout(resolve, 500));
      renderAIResponse(
        generateSmartMockResponse(question, currentPresetKey),
        'SatQuery AI Engine (Demo Mode)'
      );
      return;
    }

    const formData = new FormData();

    // Send uploaded file or preset image
    const fileToSend =
      currentFile ||
      dataURLtoFile(currentImageSrc, 'satellite_capture.png');

    formData.append('image', fileToSend);
    formData.append('question', question);

    // Debugging
    console.log('========== SATQUERY API REQUEST ==========');
    console.log('Sending request to:', API_URL);
    console.log('File:', fileToSend);
    console.log('Question:', question);

    // Give Gemini enough time
    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 30000); // 30 seconds

    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    console.log('Response status:', response.status);

    if (response.ok) {
      const data = await response.json();

      console.log('Gemini/FastAPI response:', data);

      const shouldUsePresetDemo =
        currentPresetKey &&
        !data.image_id &&
        (data.mode || '').toLowerCase().includes('demo');

      renderAIResponse(
        shouldUsePresetDemo
          ? generateSmartMockResponse(question, currentPresetKey)
          : data.answer,
        shouldUsePresetDemo
          ? 'SatQuery AI Engine (Demo Mode)'
          : data.mode || 'FastAPI / Vision AI'
      );

    } else {
      const errorText = await response.text();

      console.error(
        'Server error:',
        response.status,
        errorText
      );

      throw new Error(
        `Server returned HTTP ${response.status}: ${errorText}`
      );
    }

  } catch (err) {

    console.error('========== API ERROR ==========');
    console.error(err);

    console.warn(
      'Backend API not reachable. Using demo engine:',
      err.message
    );

    await new Promise(resolve =>
      setTimeout(resolve, 1400)
    );

    const mockAnswer = generateSmartMockResponse(
      question,
      currentPresetKey
    );

    renderAIResponse(
      mockAnswer,
      'SatQuery AI Engine (Demo Mode)'
    );

  } finally {
    setLoadingState(false);
  }
});

function renderAIResponse(text, sourceLabel) {
  responseSourceBadge.textContent = sourceLabel;
  
  // Format simple markdown into HTML structure
  responseBody.innerHTML = formatMarkdownToHTML(text);
  responseCard.classList.remove('hidden');
  
  // Update timestamp
  const now = new Date();
  analysisTimestamp.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  
  // Smooth scroll to response
  responseCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Convert markdown asterisks and lists into styled HTML
function formatMarkdownToHTML(text) {
  let html = text
    // Escape HTML
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // Headers ###
    .replace(/^### (.*$)/gim, '<h4 style="color:#38bdf8; margin-top:0.8rem; margin-bottom:0.3rem; font-size:0.95rem;">$1</h4>')
    // Bold **text**
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Bullet points * or -
    .replace(/^\s*[\-\*]\s+(.*)$/gim, '<li>$1</li>');

  // Wrap loose <li> tags into <ul>
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  // Clean up adjacent <ul><ul>
  html = html.replace(/<\/ul>\s*<ul>/g, '');

  return html;
}


/* -------------------------------------------------------------
 * 6. Copy Response to Clipboard
 * ------------------------------------------------------------- */
copyResponseBtn.addEventListener('click', () => {
  const text = responseBody.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const originalHTML = copyResponseBtn.innerHTML;
    copyResponseBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5">
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    `;
    setTimeout(() => {
      copyResponseBtn.innerHTML = originalHTML;
    }, 2000);
  });
});

/* -------------------------------------------------------------
 * 7. Intelligent Earth Observation Mock Engine
 * Provides context-aware answers based on queries & presets
 * ------------------------------------------------------------- */
function generateSmartMockResponse(question, presetKey) {
  const q = question.toLowerCase();

  // 1. Flooding & Water Detection
  if (q.includes('water') || q.includes('flood') || q.includes('inundat') || q.includes('submerg') || q.includes('river')) {
    if (presetKey === 'flood') {
      return `### 🌊 Flooding & Inundation Analysis:
* **Water Coverage**: **38.5%** of scene (baseline: 12.0%).
* **Flood Severity**: **Critical (Level 4 of 5)** with inundation depth 0.6m - 2.4m.
* **Impact**: Over 340 hectares of agricultural land submerged; 2 primary transport bridges cut off.
* **Visible Plumes**: High turbidity sediment runoff visible across floodplain.`;
    }
    if (presetKey === 'agriculture') {
      return `### 🌊 Hydrological & Irrigation Network:
* **Water Infrastructure**: Primary engineered concrete irrigation canal along southern boundary.
* **Distribution**: Center-pivot rotational spray systems feeding circular crop parcels.
* **Flooding Status**: No catastrophic flood inundation detected.`;
    }
    return `### 🌊 Water Bodies Assessment:
* **Water Features**: No catastrophic flooding detected.
* **Surface Water**: Localized drainage basins and storm runoff channels consistent with urban/forest terrain.`;
  }

  // 2. Deforestation & Vegetation Cover
  if (q.includes('forest') || q.includes('deforest') || q.includes('tree') || q.includes('vegetation') || q.includes('canopy') || q.includes('crop') || q.includes('green')) {
    if (presetKey === 'deforestation') {
      return `### 🌲 Deforestation & Canopy Loss Assessment:
* **Canopy Loss Area**: ~185 hectares cleared in active logging corridor.
* **Clearance Pattern**: Characteristic **'fishbone'** road-driven clearing pattern.
* **NDVI Metric**: Intact canopy (0.81) vs. Exposed clear-cut patches (0.15).
* **Ecological Risk**: Urgent action alert - severe fragmentation of arboreal wildlife corridors.`;
    }
    if (presetKey === 'agriculture') {
      return `### 🌾 Agricultural Land & Crop Health:
* **Cultivated Area**: **68.4%** active crop parcels with circular center-pivot geometries.
* **Mean NDVI**: **0.74** indicating healthy vegetative biomass.
* **Dominant Crops**: Alfalfa, maize/corn, and cereal grain rotations.
* **Moisture Stress**: Minor stress detected in outer perimeter of Pivot-3.`;
    }
    return `### 🌿 Vegetation & Green Spaces:
* **Green Coverage**: Approximately 14.8% fragmented urban green spaces and roadside trees.
* **Condition**: Managed municipal park vegetation.`;
  }

  // 3. Urban Development & Infrastructure
  if (q.includes('urban') || q.includes('rural') || q.includes('building') || q.includes('road') || q.includes('infrastructure') || q.includes('city')) {
    if (presetKey === 'urban') {
      return `### 🏙️ Urban Development & Infrastructure Grid:
* **Impervious Surface Ratio**: **80.7%** (High-Density Urban).
* **Road Network Density**: **14.2 km/km²** with multi-lane arterial avenues and flyovers.
* **Active Development**: Rapid commercial buildout in southeast quadrant.
* **Microclimate**: Pronounced Urban Heat Island (UHI) temperature elevation (+6-9°C).`;
    }
    return `### 🏗️ Land Use Classification:
* **Classification**: Predominantly **Rural / Natural / Agricultural** landscape.
* **Infrastructure Density**: Low building footprint (< 5%) with unpaved secondary access tracks.`;
  }

  // 4. Environmental Impacts & Recommendations
  if (q.includes('impact') || q.includes('hazard') || q.includes('risk') || q.includes('action') || q.includes('recommend')) {
    if (presetKey === 'flood') {
      return `### ⚠️ Environmental & Disaster Risk:
* **Immediate Hazards**: Crop rotting, severed highway connectivity, potential groundwater contamination.
* **Recommended Action**: Stage emergency water pumping at western intersection; deploy evacuation boats.`;
    }
    if (presetKey === 'deforestation') {
      return `### ⚠️ Environmental Hazards:
* **Immediate Hazards**: Biodiversity corridor collapse, severe monsoon erosion, massive carbon emissions from slash burning.
* **Recommended Action**: Enforce aerial/satellite patrol along access track km-14.`;
    }
    return `### ⚠️ Environmental Observations:
* **Analysis**: Monitored for surface runoff, thermal mass accumulation, and vegetative resilience.`;
  }

  // 5. Default General Overview
  return `### 🛰️ SatQuery AI Scene Overview:
* **Scenario**: ${presetKey ? presetKey.toUpperCase() : 'Calibrated Satellite Observation'}.
* **Visual Summary**: High-resolution multispectral capture analyzed against verified ground-truth knowledge base.
* **Recommendation**: Ask specific questions like *"Are there flooded areas?"*, *"Detect deforestation"*, *"Analyze crop health"*, or *"Assess urban infrastructure"*.`;
}
