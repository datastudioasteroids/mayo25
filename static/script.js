document.addEventListener('DOMContentLoaded', () => {
  const AVATAR_IMG = {
    martin_gueemes: 'guemes.png',
    juana_azurduy: 'juana.png',
    manuel_belgrano: 'manuel_belgrano.png',
    milicias_urbanas: 'milicias_urbanas.png'
  };
  const PUEBLO_IMG = 'pueblo.jpeg';

  const form = document.querySelector('.chat-input');
  const input = form.querySelector('input[name="message"]');
  const avatarInput = form.querySelector('input[name="avatar"]');
  const chatWindow = document.querySelector('.chat-window');
  const avatarCards = document.querySelectorAll('.avatar-card');

  function appendMessage(text, sender) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', sender);

    const img = document.createElement('img');
    img.classList.add('msg-avatar');

    if (sender === 'bot') {
      // usa la key para escoger el png correcto
      img.src = `/static/${AVATAR_IMG[avatarInput.value]}`;
    } else {
      img.src = `/static/${PUEBLO_IMG}`;
    }

    const bubble = document.createElement('div');
    bubble.classList.add('msg-bubble');
    bubble.textContent = text;

    if (sender === 'bot') {
      wrapper.append(img, bubble);
    } else {
      wrapper.append(bubble, img);
    }

    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  // Saludo inicial
  appendMessage(
    `👋 ¡Hola! Soy ${document.querySelector('.avatar-card.active h2').textContent}. ¿En qué puedo ayudarte hoy?`,
    'bot'
  );

  // Cambio de avatar
  avatarCards.forEach(card => {
    card.addEventListener('click', () => {
      avatarCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      avatarInput.value = card.dataset.avatar;

      // reiniciar chat con nuevo saludo
      chatWindow.innerHTML = '';
      appendMessage(
        `👋 ¡Hola! Soy ${card.querySelector('h2').textContent}. ¿En qué puedo ayudarte hoy?`,
        'bot'
      );
    });
  });

  // Envío de mensajes
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    appendMessage(message, 'user');
    input.value = '';
    input.focus();

    try {
      const res = await fetch('/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          avatar: avatarInput.value
        })
      });
      const { reply } = await res.json();
      appendMessage(reply, 'bot');
    } catch (err) {
      console.error(err);
      appendMessage('Error al contactar al servidor. Intenta de nuevo.', 'bot');
    }
  });
});
