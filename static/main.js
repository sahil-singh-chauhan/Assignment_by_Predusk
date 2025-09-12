const uploadForm = document.getElementById('upload-form');
const pdfInput = document.getElementById('pdf-file');
const popup = document.getElementById('upload-popup');
const uploading = document.getElementById('uploading-popup');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const chatBox = document.getElementById('chat-box');

function showPopup() {
  popup.classList.remove('hidden');
  setTimeout(() => popup.classList.add('hidden'), 2000);
}

function appendMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = `msg ${role}`;
  msg.textContent = content;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendMetrics(metrics) {
  if (!metrics) return;
  const m = document.createElement('div');
  m.className = 'metrics';
  m.textContent = `time: ${metrics.elapsed_ms} ms`;
  chatBox.appendChild(m);
  chatBox.scrollTop = chatBox.scrollHeight;
}

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!pdfInput.files.length) return;
  const formData = new FormData();
  formData.append('file', pdfInput.files[0]);
  // show uploading popup
  uploading.classList.remove('hidden');
  const res = await fetch('/upload', { method: 'POST', body: formData });
  const data = await res.json();
  // hide uploading popup
  uploading.classList.add('hidden');
  if (data.success) {
    showPopup();
    // Clear chat history when new PDF is uploaded
    chatBox.innerHTML = '';
  } else {
    // hide uploading popup in case of error
    uploading.classList.add('hidden');
    if (data.error && data.error.includes('refresh the page')) {
      alert('Please refresh the page to upload a new PDF');
    } else {
      alert(data.error || 'Upload failed');
    }
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  appendMessage('user', question);
  questionInput.value = '';
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  const data = await res.json();
  if (data.success) {
    appendMessage('assistant', data.answer);
    appendMetrics(data.metrics);
  } else {
    appendMessage('assistant', `Error: ${data.error || 'Something went wrong'}`);
  }
});


