const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const fileNameEl = document.getElementById('file-name');
const submitBtn  = document.getElementById('submit-btn');
const loading    = document.getElementById('loading');
const results    = document.getElementById('results');
const errorSec   = document.getElementById('error-section');
const errorMsg   = document.getElementById('error-msg');

let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  fileNameEl.textContent = '📎 ' + file.name;
  fileNameEl.classList.remove('hidden');
  submitBtn.disabled = false;
}

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  submitBtn.disabled = true;
  loading.classList.remove('hidden');
  results.classList.add('hidden');
  errorSec.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('titulo', document.getElementById('titulo').value);
  formData.append('resumo', document.getElementById('resumo').value);

  try {
    const res = await fetch('/api/classify', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Erro desconhecido' }));
      throw new Error(err.detail || 'Erro no servidor');
    }
    displayResults(await res.json());
  } catch (e) {
    errorMsg.textContent = '⚠ ' + e.message;
    errorSec.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
    submitBtn.disabled = false;
  }
});

function displayResults(data) {
  const convSection = document.getElementById('convergence-section');
  if (data.convergencia != null) {
    const c = data.convergencia;
    const [cls, label] = c > 0.85
      ? ['badge-high', 'Alta']
      : c > 0.70
      ? ['badge-medium', 'Média']
      : ['badge-low', 'Baixa'];
    const warning = c <= 0.70
      ? '<br><small style="color:#991b1b;display:block;margin-top:.25rem">⚠ Convergência baixa — revisão humana recomendada</small>'
      : '';
    convSection.innerHTML = `<span class="convergence-badge ${cls}">Convergência: ${c.toFixed(3)} — ${label}</span>${warning}`;
  } else {
    convSection.innerHTML = '';
  }

  document.getElementById('meta-section').innerHTML =
    `<div class="meta-info">${data.num_chunks} chunks analisados · ${data.habilidades.length} habilidades identificadas</div>`;

  document.getElementById('competencies').innerHTML = data.habilidades.map(h => {
    const pct = Math.round(h.confianca * 100);
    return `
      <div class="result-card">
        <div class="result-header">
          <span class="code-badge">${h.codigo}</span>
          <span class="area-label">${h.area}</span>
        </div>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar">
            <div class="confidence-fill" style="width:${pct}%"></div>
          </div>
          <span class="confidence-value">${pct}%</span>
        </div>
        <div class="desc">${h.descricao}</div>
      </div>`;
  }).join('');

  results.classList.remove('hidden');
  results.scrollIntoView({ behavior: 'smooth' });
}
