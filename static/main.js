const uploadForm = document.getElementById('upload-form');
const pdfInput = document.getElementById('pdf-file');
const popup = document.getElementById('upload-popup');
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

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!pdfInput.files.length) return;
  const formData = new FormData();
  formData.append('file', pdfInput.files[0]);
  const res = await fetch('/upload', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.success) {
    showPopup();
  } else {
    alert(data.error || 'Upload failed');
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
  } else {
    appendMessage('assistant', `Error: ${data.error || 'Something went wrong'}`);
  }
});


