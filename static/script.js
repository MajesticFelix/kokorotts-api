let currentAudioUrl = null;
let voiceRowCounter = 2;
let currentAbortController = null;
let isGenerating = false;
let currentAudioFormat = 'mp3';
let currentFileName = null;

// Initialize page
window.onload = function() {
    toggleVoiceMode();
    loadVoices();
    updateRemoveButtons();
};

// Voice mode toggle
function toggleVoiceMode() {
    const mode = document.getElementById('voiceMode').value;
    const singleSection = document.getElementById('singleVoiceSection');
    const blendSection = document.getElementById('voiceBlendSection');

    if (mode === 'single') {
        singleSection.style.display = 'block';
        blendSection.style.display = 'none';
    } else {
        singleSection.style.display = 'none';
        blendSection.style.display = 'block';
    }
}

// Resource loading functions
async function loadVoices() {
    const status = document.getElementById('resourceStatus');
    showStatus('Loading available voices...', 'loading', 'resourceStatus');
    
    try {
        const response = await fetch('/v1/audio/voices');
        const data = await response.json();
        const voices = data.voices;

        // Update all voice dropdowns
        updateVoiceDropdowns(voices);
        
        showStatus(`✅ Loaded ${voices.length} voices successfully`, 'success', 'resourceStatus');
    } catch (error) {
        showStatus(`❌ Error loading voices: ${error.message}`, 'error', 'resourceStatus');
    }
}

async function loadLanguages() {
    showStatus('Loading supported languages...', 'loading', 'resourceStatus');
    
    try {
        const response = await fetch('/v1/audio/languages');
        const data = await response.json();
        const languages = data.languages;
        
        // Update language dropdown
        const languageSelect = document.getElementById('languageSelect');
        languageSelect.innerHTML = '';
        
        languages.forEach(lang => {
            const option = document.createElement('option');
            option.value = lang.code;
            option.textContent = lang.name;
            if (lang.code === 'a') option.selected = true; // Default to American English
            languageSelect.appendChild(option);
        });
        
        const langInfo = languages.map(lang => `${lang.name} (${lang.code})`).join(', ');
        showStatus(`🌐 Supported languages: ${langInfo}`, 'success', 'resourceStatus');
    } catch (error) {
        showStatus(`❌ Error loading languages: ${error.message}`, 'error', 'resourceStatus');
    }
}

async function testConnection() {
    showStatus('Testing API connection...', 'loading', 'resourceStatus');
    
    try {
        const response = await fetch('/');
        const data = await response.json();
        showStatus(`✅ API connection successful: ${data.message}`, 'success', 'resourceStatus');
    } catch (error) {
        showStatus(`❌ Connection failed: ${error.message}`, 'error', 'resourceStatus');
    }
}

function updateVoiceDropdowns(voices) {
    const singleVoice = document.getElementById('singleVoice');
    const voiceSelects = document.querySelectorAll('.voice-select');

    // Update single voice dropdown
    const currentSingle = singleVoice.value;
    singleVoice.innerHTML = '';
    voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice;
        option.textContent = voice;
        singleVoice.appendChild(option);
    });
    if (voices.includes(currentSingle)) {
        singleVoice.value = currentSingle;
    }

    // Update voice blend dropdowns
    voiceSelects.forEach((select, index) => {
        const currentValue = select.value;
        select.innerHTML = '';
        voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice;
            select.appendChild(option);
        });
        
        if (voices.includes(currentValue)) {
            select.value = currentValue;
        } else if (voices.length > index) {
            select.value = voices[index];
        }
    });
}

// Voice blending functions
function addVoiceRow() {
    voiceRowCounter++;
    const container = document.getElementById('voiceBlendContainer');
    
    const row = document.createElement('div');
    row.className = 'voice-row';
    row.innerHTML = `
        <select class="voice-select">
            <option value="af_heart">af_heart</option>
            <option value="af_sarah">af_sarah</option>
            <option value="am_adam">am_adam</option>
        </select>
        <input type="number" class="voice-weight" placeholder="Weight" value="1" step="0.1" min="0">
        <button type="button" class="btn btn-secondary remove-voice" onclick="removeVoiceRow(this)">Remove</button>
    `;
    
    container.appendChild(row);
    
    // Copy current voice options if already loaded
    const firstSelect = container.querySelector('.voice-select');
    const newSelect = row.querySelector('.voice-select');
    if (firstSelect.options.length > 3) {
        newSelect.innerHTML = firstSelect.innerHTML;
    }
    
    updateRemoveButtons();
}

function removeVoiceRow(button) {
    button.parentElement.remove();
    updateRemoveButtons();
}

function updateRemoveButtons() {
    const rows = document.querySelectorAll('#voiceBlendContainer .voice-row');
    const removeButtons = document.querySelectorAll('.remove-voice');
    
    removeButtons.forEach((button, index) => {
        if (rows.length <= 2) {
            button.style.display = 'none';
        } else {
            button.style.display = index === 0 ? 'none' : 'inline-block';
        }
    });
}

// Voice specification builder
function buildVoiceSpec() {
    const mode = document.getElementById('voiceMode').value;
    
    if (mode === 'single') {
        return document.getElementById('singleVoice').value;
    } else {
        const selects = document.querySelectorAll('.voice-select');
        const weights = document.querySelectorAll('.voice-weight');
        
        const parts = [];
        selects.forEach((select, index) => {
            const voice = select.value;
            const weight = parseFloat(weights[index].value) || 1;
            if (voice && weight > 0) {
                parts.push(`${voice}(${weight})`);
            }
        });
        
        return parts.length > 0 ? parts.join('+') : 'af_heart';
    }
}

// Download audio
function downloadAudio() {
    if (!currentAudioUrl) {
        showStatus('❌ No audio available for download', 'error', 'mainStatus');
        return;
    }
    
    const link = document.createElement('a');
    link.href = currentAudioUrl;
    link.download = currentFileName || `kokoro-tts-audio.${currentAudioFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showStatus('📥 Audio download started', 'success', 'mainStatus');
}

// Cancel generation
function cancelGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    
    isGenerating = false;
    const generateBtn = document.getElementById('generateBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    generateBtn.disabled = false;
    generateBtn.textContent = '🎤 Generate Speech';
    cancelBtn.style.display = 'none';
    downloadBtn.style.display = 'none';
    
    showStatus('❌ Generation cancelled by user', 'error', 'mainStatus');
    
    // Stop any audio playback
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
}

// Main speech generation
async function generateSpeech() {
    const text = document.getElementById('textInput').value.trim();
    const voiceSpec = buildVoiceSpec();
    const format = document.getElementById('audioFormat').value;
    const speed = parseFloat(document.getElementById('speechSpeed').value);
    const streaming = document.getElementById('streamingMode').value === 'true';
    const language = document.getElementById('languageSelect').value;
    
    if (!text) {
        showStatus('❌ Please enter some text to convert', 'error', 'mainStatus');
        return;
    }

    if (isGenerating) {
        showStatus('❌ Generation already in progress', 'error', 'mainStatus');
        return;
    }

    isGenerating = true;
    currentAbortController = new AbortController();
    
    const generateBtn = document.getElementById('generateBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    
    generateBtn.disabled = true;
    generateBtn.textContent = streaming ? '🔄 Streaming...' : '⏳ Generating...';
    cancelBtn.style.display = 'inline-block';
    downloadBtn.style.display = 'none';

    // Store current format and generate filename
    currentAudioFormat = format;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    currentFileName = `kokoro-${voiceSpec}-${timestamp}.${format}`;

    // Reset audio player
    resetAudioPlayer();

    const payload = {
        model: "kokoro",
        input: text,
        voice: voiceSpec,
        response_format: format,
        speed: speed,
        stream: streaming,
        include_captions: false, // temp
        language: language
    };

    showStatus(`${streaming ? '🔄 Starting streaming' : '⏳ Generating'} with voice: ${voiceSpec}...`, 'loading', 'mainStatus');

    try {
        if (streaming) {
            await handleStreamingGeneration(payload);
        } else {
            await handleStandardGeneration(payload);
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            showStatus('❌ Generation cancelled', 'error', 'mainStatus');
        } else {
            showStatus(`❌ Generation failed: ${error.message}`, 'error', 'mainStatus');
        }
    } finally {
        isGenerating = false;
        currentAbortController = null;
        generateBtn.disabled = false;
        generateBtn.textContent = '🎤 Generate Speech';
        cancelBtn.style.display = 'none';
    }
}

async function handleStandardGeneration(payload) {
    const startTime = Date.now();
    
    // Show debug log for standard generation
    const debugContainer = document.getElementById('streamingDebug');
    const debugTitle = document.getElementById('debugTitle');
    const debugLog = document.getElementById('debugLog');
    debugContainer.style.display = 'block';
    debugTitle.textContent = '🎯 Standard Generation Debug Log:';
    debugLog.textContent = '';

    function addDebugLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        debugLog.textContent += `[${timestamp}] ${message}\n`;
        debugLog.scrollTop = debugLog.scrollHeight;
    }

    addDebugLog('🎯 Starting standard generation request...');
    addDebugLog(`📝 Request: ${JSON.stringify(payload, null, 2)}`);
    addDebugLog(`🔧 Language: ${payload.language}, Voice: ${payload.voice}, Format: ${payload.response_format}`);
    addDebugLog(`⚡ Speed: ${payload.speed}x, Text length: ${payload.input.length} characters`);
    
    // Estimate text chunks (backend uses 800-char chunks)
    const estimatedChunks = Math.ceil(payload.input.length / 800);
    addDebugLog(`📊 Estimated text chunks: ${estimatedChunks} (backend uses ~800 char chunks)`);
    addDebugLog(`🌐 Sending request to /v1/audio/speech...`);
    
    const response = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: currentAbortController.signal
    });

    if (!response.ok) {
        const error = await response.json();
        addDebugLog(`❌ Request failed: HTTP ${response.status}`);
        addDebugLog(`❌ Error: ${JSON.stringify(error, null, 2)}`);
        throw new Error(error.error?.message || `HTTP ${response.status}`);
    }

    const responseTime = Date.now();
    const requestTime = (responseTime - startTime) / 1000;
    addDebugLog(`✅ Response received (HTTP ${response.status}) in ${requestTime.toFixed(3)}s`);
    addDebugLog(`📊 Content-Type: ${response.headers.get('content-type')}`);
    
    const blob = await response.blob();
    const endTime = Date.now();
    const totalTime = (endTime - startTime) / 1000;
    const downloadTime = (endTime - responseTime) / 1000;

    addDebugLog(`📦 Audio blob received: ${formatBytes(blob.size)}`);
    addDebugLog(`⏱️ Download time: ${downloadTime.toFixed(3)}s, Total time: ${totalTime.toFixed(3)}s`);

    // Update performance stats
    updatePerformanceStats({
        firstAudio: totalTime,
        totalTime: totalTime,
        chunks: 1,
        size: blob.size
    });

    addDebugLog(`🎵 Creating audio URL and setting up player...`);

    // Setup audio player
    const audioUrl = URL.createObjectURL(blob);
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.src = audioUrl;
    audioPlayer.style.display = 'block';
    
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        addDebugLog(`🗑️ Cleaned up previous audio URL`);
    }
    currentAudioUrl = audioUrl;

    addDebugLog(`✅ Standard generation completed successfully!`);
    
    // Calculate efficiency metrics
    const throughputKBps = (blob.size / totalTime / 1024).toFixed(1);
    const charsPerSecond = (payload.input.length / totalTime).toFixed(0);
    addDebugLog(`📊 Throughput: ${throughputKBps} KB/s, ${charsPerSecond} chars/s`);
    addDebugLog(`📊 Final stats: ${totalTime.toFixed(3)}s total, ${formatBytes(blob.size)} audio`);

    showStatus(`✅ Audio generated in ${totalTime.toFixed(2)}s (${formatBytes(blob.size)})`, 'success', 'mainStatus');

    // Show download button
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = 'inline-block';
    addDebugLog(`📥 Download button enabled`);

    // Auto-play
    try {
        await audioPlayer.play();
        addDebugLog(`▶️ Audio playback started automatically`);
    } catch (e) {
        addDebugLog(`⚠️ Auto-play prevented by browser: ${e.message}`);
        console.log('Auto-play prevented by browser');
    }
}

async function handleStreamingGeneration(payload) {
    const startTime = Date.now();
    let firstChunkTime = null;
    let chunkCount = 0;
    let totalSize = 0;
    const audioChunks = [];

    // Show streaming debug
    const debugContainer = document.getElementById('streamingDebug');
    const debugTitle = document.getElementById('debugTitle');
    const debugLog = document.getElementById('debugLog');
    debugContainer.style.display = 'block';
    debugTitle.textContent = '🔄 Streaming Generation Debug Log:';
    debugLog.textContent = '';

    function addDebugLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        debugLog.textContent += `[${timestamp}] ${message}\n`;
        debugLog.scrollTop = debugLog.scrollHeight;
    }

    addDebugLog('🚀 Starting streaming request...');
    addDebugLog(`📝 Request: ${JSON.stringify(payload, null, 2)}`);

    const response = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: currentAbortController.signal
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.style.display = 'block';

    addDebugLog('📡 Response received, starting chunk processing...');

    let hasStartedPlaying = false;

    while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
            const endTime = Date.now();
            const totalTime = (endTime - startTime) / 1000;
            const firstChunkDelay = firstChunkTime ? (firstChunkTime - startTime) / 1000 : totalTime;

            // Final audio update with complete stream
            try {
                const currentTime = audioPlayer.currentTime || 0;
                const wasPlaying = !audioPlayer.paused;
                
                const finalBlob = new Blob(audioChunks, { 
                    type: `audio/${payload.response_format}` 
                });
                
                if (currentAudioUrl) {
                    URL.revokeObjectURL(currentAudioUrl);
                }
                
                currentAudioUrl = URL.createObjectURL(finalBlob);
                audioPlayer.src = currentAudioUrl;
                audioPlayer.currentTime = currentTime;
                
                if (wasPlaying) {
                    audioPlayer.play().catch(e => {
                        addDebugLog(`⚠️ Final playback resume failed: ${e.message}`);
                    });
                }
                
                addDebugLog(`🎯 Final update: Complete audio source set (${formatBytes(totalSize)})`);
            } catch (e) {
                addDebugLog(`❌ Final audio update failed: ${e.message}`);
            }

            addDebugLog(`✅ Streaming completed! Total: ${totalTime.toFixed(2)}s, Chunks: ${chunkCount}, Size: ${formatBytes(totalSize)}`);
            
            updatePerformanceStats({
                firstAudio: firstChunkDelay,
                totalTime: totalTime,
                chunks: chunkCount,
                size: totalSize
            });

            showStatus(`✅ Streaming completed in ${totalTime.toFixed(2)}s (${chunkCount} chunks, ${formatBytes(totalSize)})`, 'success', 'mainStatus');
            
            // Show download button
            const downloadBtn = document.getElementById('downloadBtn');
            downloadBtn.style.display = 'inline-block';
            addDebugLog(`📥 Download button enabled`);
            
            break;
        }

        const chunkTime = Date.now();
        chunkCount++;
        totalSize += value.length;
        audioChunks.push(value);

        if (!firstChunkTime) {
            firstChunkTime = chunkTime;
            const delay = (firstChunkTime - startTime) / 1000;
            addDebugLog(`🎯 FIRST CHUNK! Received in ${delay.toFixed(3)}s - Audio can start playing now!`);
        }

        const elapsed = (chunkTime - startTime) / 1000;
        addDebugLog(`📦 Chunk ${chunkCount}: ${formatBytes(value.length)} (${elapsed.toFixed(3)}s, total: ${formatBytes(totalSize)})`);

        // Update status periodically
        if (chunkCount % 10 === 0 || chunkCount <= 5) {
            showStatus(`🔄 Streaming... chunk ${chunkCount} (${formatBytes(totalSize)} total)`, 'loading', 'mainStatus');
        }

        // Smart audio updates - start playback immediately, then update periodically
        const shouldUpdate = chunkCount === 1 || chunkCount % 20 === 0;
        
        if (shouldUpdate) {
            try {
                const currentTime = audioPlayer.currentTime || 0;
                const wasPlaying = !audioPlayer.paused;
                
                const progressiveBlob = new Blob(audioChunks, { 
                    type: `audio/${payload.response_format}` 
                });
                
                if (currentAudioUrl) {
                    URL.revokeObjectURL(currentAudioUrl);
                }
                
                currentAudioUrl = URL.createObjectURL(progressiveBlob);
                audioPlayer.src = currentAudioUrl;
                
                if (chunkCount === 1) {
                    // First chunk - start playback
                    addDebugLog('🎵 Starting audio playback...');
                    try {
                        await audioPlayer.play();
                        hasStartedPlaying = true;
                        addDebugLog('✅ Audio playback started successfully!');
                    } catch (e) {
                        addDebugLog(`⚠️ Auto-play prevented: ${e.message}`);
                    }
                } else {
                    // Subsequent chunks - preserve playback state
                    audioPlayer.currentTime = currentTime;
                    if (wasPlaying) {
                        audioPlayer.play().catch(e => {
                            addDebugLog(`⚠️ Playback resume failed: ${e.message}`);
                        });
                    }
                }
                
                addDebugLog(`🔄 Updated audio source (chunk ${chunkCount})`);
            } catch (e) {
                addDebugLog(`❌ Audio update failed: ${e.message}`);
            }
        }
    }
}

function resetAudioPlayer() {
    const audioPlayer = document.getElementById('audioPlayer');
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    }
    
    audioPlayer.src = '';
    audioPlayer.style.display = 'none';
    
    // Hide debug, stats, and download button
    document.getElementById('streamingDebug').style.display = 'none';
    document.getElementById('performanceStats').style.display = 'none';
    document.getElementById('downloadBtn').style.display = 'none';
}

function updatePerformanceStats(stats) {
    document.getElementById('statFirstAudio').textContent = `${stats.firstAudio.toFixed(2)}s`;
    document.getElementById('statTotalTime').textContent = `${stats.totalTime.toFixed(2)}s`;
    document.getElementById('statChunks').textContent = stats.chunks;
    document.getElementById('statSize').textContent = formatBytes(stats.size);
    document.getElementById('performanceStats').style.display = 'block';
}

// Quick test presets
function quickTest(type) {
    const textInput = document.getElementById('textInput');
    const voiceMode = document.getElementById('voiceMode');
    const streamingMode = document.getElementById('streamingMode');
    const audioFormat = document.getElementById('audioFormat');

    switch (type) {
        case 'short':
            textInput.value = "The old lighthouse keeper squinted through the morning fog as a mysterious ship approached the rocky shore. Captain Elara had been sailing for three weeks, searching for the legendary island of Whispers. Her compass spun wildly, but she trusted the ancient map tucked safely in her coat pocket. As the fog cleared, she gasped—before her stood towering crystal spires that seemed to hum with magical energy. The keeper smiled knowingly and waved her toward the hidden harbor, where adventure awaited among the singing stones and glowing tide pools.";
            streamingMode.value = 'false';
            break;
        
        case 'medium':
            textInput.value = "Deep in the Enchanted Valley, young Lyra discovered she could speak to the wind itself. It started on her sixteenth birthday when she heard whispers carrying secrets from distant lands. The wind told her of dragons sleeping beneath mountain peaks, of mermaids singing in forgotten caves, and of a darkness growing in the Shadow Realm that threatened all magical creatures. Lyra's grandmother had been the last Wind Whisperer, and now the ancient power had awakened within her granddaughter. With her loyal companion, a silver fox named Zephyr, Lyra embarked on a quest to unite the four elemental guardians. She would need to master her gift quickly, for the Shadow King's army was already marching toward the Valley, and only the combined power of earth, water, fire, and air could stop the approaching doom. As storm clouds gathered overhead, Lyra took her first step into a world of magic, mystery, and incredible danger.";
            streamingMode.value = 'false';
            break;
        
        case 'long':
            textInput.value = "In the sprawling metropolis of New Arcanum, where steam-powered airships soared between gleaming skyscrapers and magical energy flowed through crystal conduits, Detective Thorne Blackwood investigated the most peculiar case of his career. Someone was stealing dreams from the city's sleeping citizens, leaving them to wake each morning feeling empty and gray. The victims remembered nothing of their nocturnal visions, but witnesses reported seeing shimmering, ethereal wisps floating through bedroom windows in the dead of night. Thorne's investigation led him through the gaslit streets of the Merchant Quarter, past the towering Academy of Mystical Arts, and deep into the Underground Markets where forbidden knowledge was traded like currency. His partner, the brilliant artificer Maya Cogsworth, had designed a special device capable of tracking dream essence—a delicate contraption of brass gears and glowing gems that hummed with otherworldly energy. Together, they discovered that the dream thief was actually a heartbroken wizard named Morpheus Nightshade, who had lost his own ability to dream after a magical accident years ago. Desperate to experience the wonder of dreams once more, he had been collecting them in crystal vials, creating a vast library of stolen sleep visions in his hidden laboratory beneath the city. But when Thorne and Maya confronted him, they learned of an even greater threat: Morpheus had inadvertently awakened something ancient and hungry in the Dream Realm—a nightmare entity that fed on human imagination and was now seeping into the waking world. The three unlikely allies would need to venture into the ever-shifting landscape of dreams themselves, where logic held no sway and willpower was the only weapon against creatures of pure terror and wonder.";
            streamingMode.value = 'false';
            break;
    }

    // Scroll to the generate button
    document.getElementById('generateBtn').scrollIntoView({ behavior: 'smooth' });
}

// Utility functions
function showStatus(message, type, containerId) {
    const container = document.getElementById(containerId);
    container.textContent = message;
    container.className = `status ${type}`;
    container.style.display = 'block';
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Debug and Monitoring Functions
async function checkHealth() {
    showStatus('Checking API health...', 'loading', 'debugStatus');
    
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const uptime = (data.uptime_seconds / 3600).toFixed(2);
        showStatus(`✅ API Healthy - Uptime: ${uptime}h`, 'success', 'debugStatus');
        
        document.getElementById('debugDetails').style.display = 'block';
        document.getElementById('debugDetails').innerHTML = `
            <h4>🏥 Health Status</h4>
            <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
    } catch (error) {
        showStatus(`❌ Health check failed: ${error.message}`, 'error', 'debugStatus');
    }
}

async function loadMetrics() {
    showStatus('Loading system metrics...', 'loading', 'debugStatus');
    
    try {
        const response = await fetch('/metrics');
        const data = await response.json();
        
        const cpu = data.cpu_percent.toFixed(1);
        const memory = data.memory_percent.toFixed(1);
        const gpuStatus = data.gpu_available ? `GPU: ${data.gpu_memory_used_mb.toFixed(0)}MB/${data.gpu_memory_total_mb.toFixed(0)}MB` : 'No GPU';
        
        showStatus(`📊 CPU: ${cpu}%, Memory: ${memory}%, ${gpuStatus}`, 'success', 'debugStatus');
        
        document.getElementById('debugDetails').style.display = 'block';
        document.getElementById('debugDetails').innerHTML = `
            <h4>📊 System Metrics</h4>
            <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
    } catch (error) {
        showStatus(`❌ Failed to load metrics: ${error.message}`, 'error', 'debugStatus');
    }
}

async function loadPipelineStatus() {
    showStatus('Loading pipeline status...', 'loading', 'debugStatus');
    
    try {
        const response = await fetch('/pipeline/status');
        const data = await response.json();
        
        const languages = data.loaded_languages.join(', ');
        showStatus(`🔧 ${data.pipeline_count} pipelines loaded (${languages}) on ${data.device}`, 'success', 'debugStatus');
        
        document.getElementById('debugDetails').style.display = 'block';
        document.getElementById('debugDetails').innerHTML = `
            <h4>🔧 Pipeline Status</h4>
            <pre>${JSON.stringify(data, null, 2)}</pre>
        `;
    } catch (error) {
        showStatus(`❌ Failed to load pipeline status: ${error.message}`, 'error', 'debugStatus');
    }
}

async function loadDebugInfo() {
    showStatus('Loading full debug information...', 'loading', 'debugStatus');
    
    try {
        const response = await fetch('/debug');
        const data = await response.json();
        
        const pythonVersion = data.system_info.python_version;
        const torchVersion = data.system_info.torch_version;
        const device = data.system_info.device;
        
        showStatus(`🔍 Python ${pythonVersion}, PyTorch ${torchVersion}, Device: ${device}`, 'success', 'debugStatus');
        
        document.getElementById('debugDetails').style.display = 'block';
        document.getElementById('debugDetails').innerHTML = `
            <h4>🔍 Complete Debug Information</h4>
            
            <h4>💻 System Information</h4>
            <pre>${JSON.stringify(data.system_info, null, 2)}</pre>
            
            <h4>🔧 Pipeline Status</h4>
            <pre>${JSON.stringify(data.pipeline_status, null, 2)}</pre>
            
            <h4>⚡ Performance Data</h4>
            <pre>${JSON.stringify(data.recent_performance, null, 2)}</pre>
        `;
    } catch (error) {
        showStatus(`❌ Failed to load debug info: ${error.message}`, 'error', 'debugStatus');
    }
}