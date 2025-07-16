class PicoQwenChat {
    constructor() {
        this.settings = this.loadSettings();
        this.messages = this.loadMessages();
        this.isConnected = false;
        this.ws = null;
        this.currentModel = null;
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.renderMessages();
        this.updateStatus('Offline');
        this.connectWebSocket();
        this.loadModels();
    }

    bindEvents() {
        // Chat form
        const form = document.getElementById('chat-form');
        const input = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-chat');

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        input.addEventListener('input', () => {
            this.updateCharCount();
            this.autoResize(input);
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        sendButton.addEventListener('click', () => this.sendMessage());
        clearButton.addEventListener('click', () => this.clearChat());

        // Settings
        const settingsToggle = document.getElementById('settings-toggle');
        const settingsPanel = document.getElementById('settings-panel');
        const saveSettings = document.getElementById('save-settings');

        settingsToggle.addEventListener('click', () => {
            settingsPanel.classList.toggle('open');
        });

        saveSettings.addEventListener('click', () => this.saveSettings());

        // Settings controls
        const temperature = document.getElementById('temperature');
        const maxTokens = document.getElementById('max-tokens');
        const apiUrl = document.getElementById('api-url');

        temperature.addEventListener('input', (e) => {
            document.getElementById('temperature-value').textContent = e.target.value;
        });

        maxTokens.addEventListener('input', (e) => {
            document.getElementById('max-tokens-value').textContent = e.target.value;
        });

        // Close settings on outside click
        document.addEventListener('click', (e) => {
            if (!settingsPanel.contains(e.target) && !settingsToggle.contains(e.target)) {
                settingsPanel.classList.remove('open');
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '/':
                        e.preventDefault();
                        document.getElementById('message-input').focus();
                        break;
                    case ',':
                        e.preventDefault();
                        settingsPanel.classList.toggle('open');
                        break;
                }
            }
        });

        // Auto-focus input
        document.getElementById('message-input').focus();
    }

    loadSettings() {
        const defaultSettings = {
            apiUrl: 'http://localhost:8080',
            temperature: 0.7,
            maxTokens: 512,
            model: null
        };

        try {
            const saved = localStorage.getItem('pico-qwen-settings');
            return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
        } catch {
            return defaultSettings;
        }
    }

    saveSettings() {
        const settings = {
            apiUrl: document.getElementById('api-url').value,
            temperature: parseFloat(document.getElementById('temperature').value),
            maxTokens: parseInt(document.getElementById('max-tokens').value),
            model: document.getElementById('model-select').value
        };

        this.settings = settings;
        localStorage.setItem('pico-qwen-settings', JSON.stringify(settings));
        
        // Update UI
        document.getElementById('settings-panel').classList.remove('open');
        this.showMessage('Settings saved', 'system');
        
        // Reconnect with new settings
        this.connectWebSocket();
    }

    loadMessages() {
        try {
            const saved = localStorage.getItem('pico-qwen-messages');
            return saved ? JSON.parse(saved) : [];
        } catch {
            return [];
        }
    }

    saveMessages() {
        localStorage.setItem('pico-qwen-messages', JSON.stringify(this.messages));
    }

    renderMessages() {
        const container = document.getElementById('messages');
        container.innerHTML = '';

        this.messages.forEach(message => {
            this.addMessageToDOM(message);
        });

        this.scrollToBottom();
    }

    addMessageToDOM(message) {
        const container = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = message.content;

        messageDiv.appendChild(contentDiv);
        container.appendChild(messageDiv);
    }

    showMessage(content, role = 'assistant') {
        const message = { role, content, timestamp: Date.now() };
        this.messages.push(message);
        this.addMessageToDOM(message);
        this.scrollToBottom();
        this.saveMessages();
    }

    async sendMessage() {
        const input = document.getElementById('message-input');
        const content = input.value.trim();

        if (!content || !this.isConnected) return;

        // Add user message
        this.showMessage(content, 'user');
        input.value = '';
        this.updateCharCount();
        this.autoResize(input);

        // Show loading
        this.showLoading(true);

        try {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    action: 'chat',
                    message: { role: 'user', content },
                    model: this.settings.model,
                    temperature: this.settings.temperature,
                    max_tokens: this.settings.maxTokens
                }));
            } else {
                // Fallback to HTTP
                await this.sendMessageHTTP(content);
            }
        } catch (error) {
            this.showMessage(`Error: ${error.message}`, 'system');
            this.showLoading(false);
        }
    }

    async sendMessageHTTP(content) {
        const response = await fetch(`${this.settings.apiUrl}/api/v1/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: this.settings.model || 'default',
                messages: [{ role: 'user', content }],
                max_tokens: this.settings.maxTokens,
                temperature: this.settings.temperature
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const assistantMessage = data.choices?.[0]?.message?.content || 'No response';
        
        this.showMessage(assistantMessage, 'assistant');
        this.showLoading(false);
    }

    async loadModels() {
        try {
            const response = await fetch(`${this.settings.apiUrl}/api/v1/models`);
            if (response.ok) {
                const data = await response.json();
                this.populateModels(data.models || []);
            }
        } catch (error) {
            console.warn('Could not load models:', error);
        }
    }

    populateModels(models) {
        const select = document.getElementById('model-select');
        select.innerHTML = '';

        if (models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No models available';
            select.appendChild(option);
            return;
        }

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.id} (${Math.round(model.size / 1024 / 1024)}MB)`;
            select.appendChild(option);
        });

        if (this.settings.model) {
            select.value = this.settings.model;
        }
    }

    connectWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        const wsUrl = this.settings.apiUrl.replace('http', 'ws');
        this.ws = new WebSocket(`${wsUrl}/ws`);

        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateStatus('Online');
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.action === 'chat_response') {
                    const content = data.data?.message?.content || 'No response';
                    this.showMessage(content, 'assistant');
                    this.showLoading(false);
                }
            } catch (error) {
                console.error('WebSocket message error:', error);
            }
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateStatus('Offline');
            console.log('WebSocket disconnected');
            
            // Reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Error');
        };
    }

    updateStatus(status) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        statusText.textContent = status;
        statusDot.className = `status-dot ${status.toLowerCase()}`;
    }

    updateCharCount() {
        const input = document.getElementById('message-input');
        const count = document.getElementById('char-count');
        const length = input.value.length;
        const max = 4000;
        
        count.textContent = `${length}/${max}`;
        count.style.color = length > max * 0.9 ? 'var(--error)' : 'var(--text-muted)';
    }

    autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        const container = document.getElementById('messages');
        container.scrollTop = container.scrollHeight;
    }

    clearChat() {
        if (confirm('Clear all messages?')) {
            this.messages = [];
            this.saveMessages();
            this.renderMessages();
            this.showMessage('Chat cleared', 'system');
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    }

    applySettings() {
        document.getElementById('api-url').value = this.settings.apiUrl;
        document.getElementById('temperature').value = this.settings.temperature;
        document.getElementById('temperature-value').textContent = this.settings.temperature;
        document.getElementById('max-tokens').value = this.settings.maxTokens;
        document.getElementById('max-tokens-value').textContent = this.settings.maxTokens;
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    new PicoQwenChat();
});

// Service Worker registration
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js')
        .then(registration => console.log('SW registered'))
        .catch(error => console.log('SW registration failed'));
}

// Keyboard shortcuts help
document.addEventListener('keydown', (e) => {
    if (e.key === '?' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        alert(`Keyboard shortcuts:
Ctrl/Cmd + / : Focus input
Ctrl/Cmd + , : Toggle settings
Ctrl/Cmd + ? : Show this help`);
    }
});