let currentChatId = null;

function addMessage(role, text, isError = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} ${isError ? 'error' : ''}`;

    // Remove bold labels for a cleaner chat look
    // const roleLabel = role === 'user' ? 'You' : role === 'model' ? 'AI' : 'System';
    // messageDiv.innerHTML = `<strong>${roleLabel}:</strong> ${text}`;
    
    // Just the text, maybe with an icon if model
    if (role === 'model') {
        messageDiv.innerHTML = `<i class="fas fa-robot" style="margin-right:8px; color:#007bff;"></i> ${text}`;
    } else {
        messageDiv.textContent = text;
    }

    if (role === 'system') {
         messageDiv.innerHTML = `<i class="fas fa-info-circle"></i> ${text}`;
    }

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function updateChatId(chatId) {
    if (chatId) {
        currentChatId = chatId;
        document.getElementById('chatId').value = chatId;
        localStorage.setItem('chatId', chatId);
    }
}

function loadChatId() {
    const savedChatId = localStorage.getItem('chatId');
    if (savedChatId) {
        currentChatId = savedChatId;
        document.getElementById('chatId').value = savedChatId;
    }
}

function getApiUrls() {
    let rawInput = document.getElementById('apiUrl').value.trim();
    rawInput = rawInput.replace(/\/+$/, '');

    let baseUrl;
    if (rawInput.endsWith('/api/chat')) {
        baseUrl = rawInput.substring(0, rawInput.length - '/api/chat'.length);
    } else if (rawInput.endsWith('/api')) {
        baseUrl = rawInput.substring(0, rawInput.length - '/api'.length);
    } else {
        baseUrl = rawInput;
    }
    baseUrl = baseUrl.replace(/\/+$/, '');

    return {
        chat: `${baseUrl}/api/chat`,
        voice: `${baseUrl}/api/voice/chat`,
        image: `${baseUrl}/api/ai/analyze-image`
    };
}

let mediaRecorder;
let audioChunks = [];

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            // Convert WebM Blob to WAV Blob manually before sending
            // This bypasses the backend need for ffmpeg
            const webmBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            try {
                // Convert WebM -> AudioBuffer -> WAV (PCM)
                const arrayBuffer = await webmBlob.arrayBuffer();
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                const wavBlob = await audioBufferToWav(audioBuffer);
                await sendVoiceToBackend(wavBlob);
            } catch (e) {
                console.error("Audio conversion failed in browser:", e);
                addMessage('system', 'Error processing audio in browser: ' + e.message, true);
                document.getElementById('voiceStatus').textContent = 'Error';
            }
        };

        mediaRecorder.start();
        document.getElementById('voiceBtn').classList.add('recording');
        // document.getElementById('voiceBtn').innerHTML = '🔴 Recording...'; // Removed text change
        document.getElementById('voiceStatus').textContent = 'Release to send...';
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone. Please allow permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        document.getElementById('voiceBtn').classList.remove('recording');
        // document.getElementById('voiceBtn').innerHTML = 'Voice'; // No longer text
        document.getElementById('voiceStatus').textContent = 'Processing...';
        
        // Add a placeholder message for the user's voice input
        addMessage('user', '🎤 Voice Message sent...');
    }
}

// --- BROWSER-SIDE WAV CONVERSION HELPERS ---
function audioBufferToWav(buffer, opt) {
    opt = opt || {};
    var numChannels = buffer.numberOfChannels;
    var sampleRate = buffer.sampleRate;
    var format = opt.float32 ? 3 : 1;
    var bitDepth = format === 3 ? 32 : 16;

    var result;
    if (numChannels === 2) {
        result = interleave(buffer.getChannelData(0), buffer.getChannelData(1));
    } else {
        result = buffer.getChannelData(0);
    }

    return encodeWAV(result, format, sampleRate, numChannels, bitDepth);
}

function interleave(inputL, inputR) {
    var length = inputL.length + inputR.length;
    var result = new Float32Array(length);

    var index = 0;
    var inputIndex = 0;

    while (index < length) {
        result[index++] = inputL[inputIndex];
        result[index++] = inputR[inputIndex];
        inputIndex++;
    }
    return result;
}

function encodeWAV(samples, format, sampleRate, numChannels, bitDepth) {
    var bytesPerSample = bitDepth / 8;
    var blockAlign = numChannels * bytesPerSample;

    var buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
    var view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* RIFF chunk length */
    view.setUint32(4, 36 + samples.length * bytesPerSample, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, format, true);
    /* channel count */
    view.setUint16(22, numChannels, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * blockAlign, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, blockAlign, true);
    /* bits per sample */
    view.setUint16(34, bitDepth, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * bytesPerSample, true);

    if (format === 1) { // Raw PCM
        floatTo16BitPCM(view, 44, samples);
    } else {
        writeFloat32(view, 44, samples);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (var i = 0; i < input.length; i++, offset += 2) {
        var s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeFloat32(output, offset, input) {
    for (var i = 0; i < input.length; i++, offset += 4) {
        output.setFloat32(offset, input[i], true);
    }
}

function writeString(view, offset, string) {
    for (var i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}
// --------------------------------------------


async function sendVoiceToBackend(audioBlob) {
    const urls = getApiUrls();

    try {
        // Since we are now sending a proper WAV (which is basically PCM with a header),
        // we can send it directly. The backend's native wave module will handle it.
        // Or if we want to be super safe, we could strip the header and send raw PCM,
        // but standard WAV is safer because it contains sample rate info.
        
        // However, the backend 'voice_service.py' logic for 'convert_to_pcm16_mono_16k'
        // now expects a valid WAV file (which we just created) OR raw PCM.
        // Sending the WAV blob directly as 'file' is the robust path.

        const formData = new FormData();
        // Name it .wav so backend treats it as wav
        formData.append('file', audioBlob, 'voice_query.wav'); 

        const response = await fetch(urls.voice, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server Error: ${errorText}`);
        }

        const responseBlob = await response.blob();
        const audioUrl = URL.createObjectURL(responseBlob);

        const audio = new Audio(audioUrl);
        
        // Show indicator that AI is speaking
        addMessage('model', '🔊 (Playing Audio Response...)');
        
        audio.play();
        
        audio.onended = () => {
             // Optional: Update the last message to indicate finished or remove the indicator
             // For now, we leave it as a record of the interaction
        };

        addMessage('system', 'Voice response received.');
        document.getElementById('voiceStatus').textContent = '';
    } catch (error) {
        console.error('Voice upload failed', error);
        addMessage('system', `Voice Error: ${error.message}`, true);
        document.getElementById('voiceStatus').textContent = 'Error';
    }
}

async function decodeToPcm16k(arrayBuffer) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.numberOfChannels > 1
        ? mixDownToMono(audioBuffer)
        : audioBuffer.getChannelData(0);

    const resampled = resampleTo16k(channelData, audioBuffer.sampleRate);
    return floatTo16BitPcm(resampled);
}

function mixDownToMono(audioBuffer) {
    const length = audioBuffer.length;
    const channelCount = audioBuffer.numberOfChannels;
    const mixed = new Float32Array(length);

    for (let channel = 0; channel < channelCount; channel += 1) {
        const data = audioBuffer.getChannelData(channel);
        for (let i = 0; i < length; i += 1) {
            mixed[i] += data[i];
        }
    }

    for (let i = 0; i < length; i += 1) {
        mixed[i] /= channelCount;
    }

    return mixed;
}

function resampleTo16k(data, sampleRate) {
    if (sampleRate === 16000) {
        return data;
    }

    const ratio = sampleRate / 16000;
    const newLength = Math.round(data.length / ratio);
    const resampled = new Float32Array(newLength);

    for (let i = 0; i < newLength; i += 1) {
        const index = i * ratio;
        const i0 = Math.floor(index);
        const i1 = Math.min(i0 + 1, data.length - 1);
        const frac = index - i0;
        resampled[i] = data[i0] * (1 - frac) + data[i1] * frac;
    }

    return resampled;
}

function floatTo16BitPcm(float32Array) {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);

    for (let i = 0; i < float32Array.length; i += 1) {
        let sample = Math.max(-1, Math.min(1, float32Array[i]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        view.setInt16(i * 2, sample, true);
    }

    return buffer;
}

async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    const urls = getApiUrls();
    const userId = document.getElementById('userId').value;
    const chatIdInput = document.getElementById('chatId').value.trim();
    const includeHistory = document.getElementById('includeHistory').checked;
    const streamMode = document.getElementById('streamMode').checked;

    addMessage('user', message);
    messageInput.value = '';

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    sendBtn.textContent = 'Sending...';

    const requestBody = {
        message: message,
        chat_id: chatIdInput || currentChatId || null,
        user_id: userId
    };

    try {
        if (streamMode) {
            await handleStreaming(urls.chat, requestBody, includeHistory);
        } else {
            await handleNormal(urls.chat, requestBody, includeHistory);
        }
    } catch (error) {
        let errorMsg = error.message;
        if (errorMsg.includes('Failed to fetch')) {
            errorMsg = 'Cannot connect to server. Ensure backend is running.';
        }
        addMessage('system', `Error: ${errorMsg}`, true);
        console.error('Chat Error:', error);
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    }
}

async function handleNormal(apiUrl, requestBody, includeHistory) {
    const url = new URL(apiUrl);
    if (includeHistory) {
        url.searchParams.append('include_history', 'true');
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        let errorMessage = 'Request failed';
        try {
            const error = await response.json();
            const detail = error.detail;
            if (typeof detail === 'string') {
                errorMessage = detail;
            } else if (detail && detail.message) {
                errorMessage = detail.message;
            } else if (detail && detail.error) {
                errorMessage = detail.error;
            } else {
                errorMessage = error.message || errorMessage;
            }
            if (response.status === 503 || errorMessage.includes('overloaded')) {
                errorMessage += '\n\nTip: Try using Streaming Mode for better availability.';
            }
        } catch (e) {
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
    }

    const data = await response.json();
    if (data.chat_id) {
        updateChatId(data.chat_id);
    }
    addMessage('model', data.response);

    if (data.history && data.history.length > 0) {
        addMessage('system', `History loaded: ${data.history.length} messages`);
    }
}

async function handleStreaming(apiUrl, requestBody, includeHistory) {
    const url = new URL(apiUrl);
    url.searchParams.append('stream', 'true');
    if (includeHistory) {
        url.searchParams.append('include_history', 'true');
    }

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail?.error || error.detail || 'Request failed');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let messageDiv = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
            if (!line.startsWith('data: ')) {
                continue;
            }
            try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'chunk') {
                    fullResponse += data.text;
                    if (!messageDiv) {
                        messageDiv = document.createElement('div');
                        messageDiv.className = 'message model streaming';
                        messageDiv.innerHTML = '<strong>AI:</strong> ';
                        document.getElementById('chatMessages').appendChild(messageDiv);
                    }
                    messageDiv.innerHTML = `<strong>AI:</strong> ${fullResponse}`;
                    document.getElementById('chatMessages').scrollTop =
                        document.getElementById('chatMessages').scrollHeight;
                } else if (data.type === 'complete') {
                    if (messageDiv) {
                        messageDiv.classList.remove('streaming');
                    }
                    if (data.chat_id) {
                        updateChatId(data.chat_id);
                    }
                    if (data.history && data.history.length > 0) {
                        addMessage('system', `History loaded: ${data.history.length} messages`);
                    }
                } else if (data.type === 'error') {
                    throw new Error(data.error || 'Streaming error');
                }
            } catch (e) {
                console.error('Error parsing SSE data:', e);
            }
        }
    }
}

async function handleSmartImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const resultDiv = document.getElementById('smartResult');
    const healthContext = document.getElementById('smartHealthContextInput').value.trim();
    const userId = document.getElementById('userId').value;
    const urls = getApiUrls();

    resultDiv.className = 'smart-result loading show';
    resultDiv.innerHTML = '<p>Analyzing image...</p>';

    try {
        const smartUrl = new URL(urls.image);
        if (healthContext) smartUrl.searchParams.append('health_context', healthContext);
        if (userId) smartUrl.searchParams.append('user_id', userId);
        if (currentChatId) smartUrl.searchParams.append('chat_id', currentChatId);

        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(smartUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to analyze image');
        }

        const data = await response.json();
        resultDiv.className = 'smart-result show';

        if (data.chat_id) updateChatId(data.chat_id);

        if (data.type === 'glucose') {
            resultDiv.innerHTML = `
                <h4>Glucose Meter</h4>
                <strong>Reading:</strong> ${data.reading.value} ${data.reading.unit}<br>
                <em>${data.analysis || ''}</em>
            `;
        } else if (data.type === 'food') {
            resultDiv.innerHTML = `
                <h4>Food Analysis</h4>
                <strong>Meal:</strong> ${data.meal.meal_name}<br>
                <strong>Calories:</strong> ${data.meal.calories || 'N/A'}<br>
                <em>${data.recommendation || ''}</em>
            `;
        } else {
            resultDiv.innerHTML = '<p>Could not identify image.</p>';
        }
    } catch (error) {
        resultDiv.className = 'smart-result error show';
        resultDiv.innerHTML = `<h4>Error</h4><p>${error.message}</p>`;
        console.error('Image upload error:', error);
    }
}

function clearChat() {
    document.getElementById('chatMessages').innerHTML =
        '<div class="message system"><strong>System:</strong> Chat cleared.</div>';
    currentChatId = null;
    document.getElementById('chatId').value = '';
    localStorage.removeItem('chatId');
}

async function testConnection() {
    const apiUrl = document.getElementById('apiUrl').value;
    const testBtn = document.getElementById('testBtn');

    testBtn.disabled = true;
    testBtn.textContent = 'Testing...';

    try {
        const baseUrl = apiUrl.replace('/api/chat', '');
        const healthUrl = `${baseUrl}/health`;

        const response = await fetch(healthUrl);
        if (response.ok) {
            addMessage('system', 'Connection successful! Backend server is running.');
        } else {
            throw new Error('Server responded but health check failed');
        }
    } catch (error) {
        addMessage(
            'system',
            `Connection failed: ${error.message}\n\nMake sure:\n1. Backend server is running\n2. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000\n3. API URL is correct`,
            true
        );
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = 'Test Connection';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    loadChatId();
    const messageInput = document.getElementById('messageInput');
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
