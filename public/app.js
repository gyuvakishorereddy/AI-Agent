// Main Application Logic
console.log('app.js loading...');
(function() {
    'use strict';
    
    console.log('IIFE started');
    
    // DOM Elements
    const elements = {
        sidebar: document.getElementById('sidebar'),
        menuBtn: document.getElementById('menuBtn'),
        closeSidebar: document.getElementById('closeSidebar'),
        newChatBtn: document.getElementById('newChatBtn'),
        chatHistoryList: document.getElementById('chatHistoryList'),
        mainContainer: document.querySelector('.main-container'),
        welcomeScreen: document.getElementById('welcomeScreen'),
        messagesContainer: document.getElementById('messagesContainer'),
        chatContainer: document.getElementById('chatContainer'),
        userInput: document.getElementById('userInput'),
        sendBtn: document.getElementById('sendBtn'),
        themeToggle: document.getElementById('themeToggle'),
        settingsBtn: document.getElementById('settingsBtn'),
        voiceBtn: document.getElementById('voiceBtn'),
        replayBtn: document.getElementById('replayBtn'),
        speakerBtn: document.getElementById('speakerBtn'),
        stopAudioBtn: document.getElementById('stopAudioBtn'),
        settingsModal: document.getElementById('settingsModal'),
        closeSettings: document.getElementById('closeSettings'),
        wordCount: document.getElementById('wordCount'),
        quickActions: document.querySelectorAll('.quick-action'),
        promptCards: document.querySelectorAll('.prompt-card'),
        themeSelect: document.getElementById('themeSelect'),
        fontSizeSelect: document.getElementById('fontSizeSelect'),
        autoScroll: document.getElementById('autoScroll'),
        soundEffects: document.getElementById('soundEffects'),
        voiceOutput: document.getElementById('voiceOutput'),
        showTimestamps: document.getElementById('showTimestamps'),
        clearHistory: document.getElementById('clearHistory'),
        exportChat: document.getElementById('exportChat'),
        languageIndicator: document.getElementById('languageIndicator'),
        mapBtn: document.getElementById('mapBtn'),
        mapModal: document.getElementById('mapModal'),
        closeMap: document.getElementById('closeMap')
    };
    
    // State
    let isProcessing = false;
    let currentLanguage = 'en'; // Track current detected language for voice output
    let lastBotMessage = null; // Store last bot message for replay

    // Settings manager: handles app preferences stored in localStorage
    const settingsManager = (function() {
        const defaults = {
            theme: 'dark',
            fontSize: 'medium',
            autoScroll: true,
            soundEffects: false,
            voiceOutput: false,
            showTimestamps: true
        };

        function get(key) {
            const stored = localStorage.getItem('kare_settings_' + key);
            return stored !== null ? JSON.parse(stored) : defaults[key];
        }

        function set(key, value) {
            localStorage.setItem('kare_settings_' + key, JSON.stringify(value));
            applySettings();
        }

        function getAll() {
            const result = {};
            for (const key in defaults) {
                result[key] = get(key);
            }
            return result;
        }

        return { get, set, getAll };
    })();

    // Chat manager: handles chat history and current chat state
    const chatManager = (function() {
        let chats = [];
        let currentChatId = null;

        function loadChats() {
            const stored = localStorage.getItem('kare_chats');
            chats = stored ? JSON.parse(stored) : [];
            if (chats.length === 0) {
                createNewChat();
            } else {
                currentChatId = chats[0].id;
            }
        }

        function saveChats() {
            localStorage.setItem('kare_chats', JSON.stringify(chats));
        }

        function createNewChat() {
            const chat = {
                id: 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
                title: 'New Chat',
                messages: [],
                lastUpdated: new Date().toISOString()
            };
            chats.unshift(chat);
            currentChatId = chat.id;
            saveChats();
        }

        function switchChat(chatId) {
            currentChatId = chatId;
        }

        function addMessage(msg) {
            const chat = chats.find(c => c.id === currentChatId);
            if (chat) {
                chat.messages.push(msg);
                chat.lastUpdated = new Date().toISOString();
                // Update title if it's still "New Chat"
                if (chat.title === 'New Chat' && msg.role === 'user') {
                    chat.title = msg.content.substring(0, 30) + (msg.content.length > 30 ? '...' : '');
                }
                saveChats();
            }
        }

        function getMessages() {
            const chat = chats.find(c => c.id === currentChatId);
            return chat ? chat.messages : [];
        }

        function getAllChats() {
            return chats;
        }

        function clearAllChats() {
            chats = [];
            localStorage.removeItem('kare_chats');
            createNewChat();
        }

        function exportChat() {
            const chat = chats.find(c => c.id === currentChatId);
            return chat ? JSON.stringify(chat, null, 2) : null;
        }

        function getCurrentChatId() {
            return currentChatId;
        }

        loadChats();

        return {
            currentChatId: currentChatId,
            getCurrentChatId,
            createNewChat,
            switchChat,
            addMessage,
            getMessages,
            getAllChats,
            clearAllChats,
            exportChat
        };
    })();

    // Voice manager: handles speech recognition and text-to-speech
    const voiceManager = (function() {
        let recognition = null;
        let listening = false;
        let onErrorCb = null;

        function supportsRecognition() {
            return ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window);
        }

        function createRecognition() {
            const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!Rec) return null;
            const r = new Rec();
            // Support multiple languages for recognition
            r.lang = 'en-US';
            r.interimResults = true;
            r.continuous = false;
            r.maxAlternatives = 1;
            return r;
        }

        function startListening(onFinal, onError) {
            console.log('voiceManager.startListening called');
            console.log('Browser supports Speech Recognition:', supportsRecognition());
            
            if (listening) {
                console.log('Already listening, returning');
                return;
            }
            
            onErrorCb = onError;
            if (!supportsRecognition()) {
                console.error('Speech Recognition not supported');
                console.error('SpeechRecognition:', window.SpeechRecognition);
                console.error('webkitSpeechRecognition:', window.webkitSpeechRecognition);
                if (onError) onError('Speech Recognition not supported in this browser');
                return;
            }

            console.log('Creating recognition instance...');
            recognition = createRecognition();
            if (!recognition) {
                console.error('Failed to create recognition instance');
                if (onError) onError('Failed to create speech recognition instance');
                return;
            }

            console.log('Recognition instance created, starting...');

            let finalTranscript = '';

            recognition.onresult = (event) => {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const res = event.results[i];
                    if (res.isFinal) {
                        finalTranscript += res[0].transcript;
                    } else {
                        interim += res[0].transcript;
                    }
                }

                // Live interim feedback in input box
                if (interim) {
                    elements.userInput.value = interim;
                    handleInputChange();
                }

                // Only call final callback once a final transcript is available
                if (finalTranscript && typeof onFinal === 'function') {
                    const text = finalTranscript.trim();
                    // Clear interim UI
                    elements.userInput.value = '';
                    handleInputChange();
                    onFinal(text);
                }
            };

            recognition.onerror = (e) => {
                listening = false;
                voiceManager.isListening = false;
                elements.voiceBtn.textContent = '🎤';
                if (onError) onError(e.error || e.message || String(e));
            };

            recognition.onend = () => {
                listening = false;
                voiceManager.isListening = false;
                elements.voiceBtn.textContent = '🎤';
            };

            try {
                console.log('Calling recognition.start()...');
                recognition.start();
                console.log('recognition.start() succeeded');
                listening = true;
                voiceManager.isListening = true;
                elements.voiceBtn.textContent = '⏹';
                elements.voiceBtn.title = 'Stop Listening (Click to stop)';
                
                // Auto-stop after 30 seconds to prevent hanging
                setTimeout(() => {
                    if (listening) {
                        console.log('Auto-stopping recognition after 30s');
                        stopListening();
                    }
                }, 30000);
            } catch (err) {
                console.error('Error calling recognition.start():', err.message || String(err));
                listening = false;
                voiceManager.isListening = false;
                elements.voiceBtn.textContent = '🎤';
                elements.voiceBtn.title = 'Voice Input';
                if (onError) onError('Failed to start voice recognition: ' + (err.message || String(err)));
            }
        }

        function stopListening() {
            if (recognition) {
                try { 
                    recognition.stop(); 
                    console.log('Recognition stopped');
                } catch (e) {
                    console.error('Error stopping recognition:', e);
                }
                recognition = null;
            }
            listening = false;
            voiceManager.isListening = false;
            elements.voiceBtn.textContent = '🎤';
            elements.voiceBtn.title = 'Voice Input (Click to start)';
        }

        function speak(text, lang = 'en') {
            if (!('speechSynthesis' in window)) return;
            try {
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance(text);
                utter.lang = lang || 'en-US';
                window.speechSynthesis.speak(utter);
                elements.stopAudioBtn.classList.remove('hidden');
            } catch (e) {
                console.error('TTS error:', e);
            }
        }

        function stopSpeaking() {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                elements.stopAudioBtn.classList.add('hidden');
            }
        }

        return {
            isListening: false,
            startListening,
            stopListening,
            speak,
            stopSpeaking
        };
    })();
    
    // Initialize app
    function init() {
        console.log('Initializing KARE AI...');
        
        // Verify all elements are loaded
        if (!elements.voiceBtn) {
            console.error('voiceBtn element not found!');
            return;
        }
        
        setupEventListeners();
        renderChatHistory();
        loadCurrentChat();
        applySettings();
        adjustTextareaHeight();
        
        console.log('KARE AI initialized successfully');
    }
    
    // Setup event listeners
    function setupEventListeners() {
        // Sidebar
        elements.menuBtn.addEventListener('click', toggleSidebar);
        elements.closeSidebar.addEventListener('click', toggleSidebar);
        elements.newChatBtn.addEventListener('click', createNewChat);
        
        // Input
        elements.userInput.addEventListener('input', handleInputChange);
        elements.userInput.addEventListener('keydown', handleKeyDown);
        elements.sendBtn.addEventListener('click', handleSendMessage);
        
        // Theme
        elements.themeToggle.addEventListener('click', toggleTheme);
        
        // Settings
        elements.settingsBtn.addEventListener('click', () => elements.settingsModal.classList.add('active'));
        elements.closeSettings.addEventListener('click', () => elements.settingsModal.classList.remove('active'));
        
        // Settings controls
        elements.themeSelect.addEventListener('change', (e) => settingsManager.set('theme', e.target.value));
        elements.fontSizeSelect.addEventListener('change', (e) => settingsManager.set('fontSize', e.target.value));
        elements.autoScroll.addEventListener('change', (e) => settingsManager.set('autoScroll', e.target.checked));
        elements.soundEffects.addEventListener('change', (e) => settingsManager.set('soundEffects', e.target.checked));
        elements.voiceOutput.addEventListener('change', (e) => {
            const isEnabled = e.target.checked;
            settingsManager.set('voiceOutput', isEnabled);
            updateSpeakerButton();
        });
        elements.showTimestamps.addEventListener('change', (e) => settingsManager.set('showTimestamps', e.target.checked));
        
        // Data management
        elements.clearHistory.addEventListener('click', clearChatHistory);
        elements.exportChat.addEventListener('click', exportCurrentChat);
        
        // Voice
        elements.voiceBtn.addEventListener('click', handleVoiceInput);
        elements.replayBtn.addEventListener('click', replayLastMessage);
        elements.speakerBtn.addEventListener('click', toggleVoiceOutput);
        elements.stopAudioBtn.addEventListener('click', stopCurrentAudio);
        
        // Map button - Open in new tab
        if (elements.mapBtn) {
            console.log('Map button found, adding event listener');
            elements.mapBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Map button clicked! Opening in new tab...');
                
                // Kalasalingam University coordinates
                const kareLatLng = { lat: 9.2108, lng: 77.3636 };
                
                // Open Google Maps in new tab with directions to KARE
                const mapsUrl = `https://www.google.com/maps/dir/?api=1&destination=${kareLatLng.lat},${kareLatLng.lng}&destination_place_id=ChIJYTN9FHRPqDsRwfu0hYbvCgQ`;
                window.open(mapsUrl, '_blank');
            });
        } else {
            console.error('Map button not found!');
        }
        
        if (elements.closeMap) {
            elements.closeMap.addEventListener('click', () => {
                elements.mapModal.classList.remove('active');
            });
        }
        
        // Close map when clicking outside
        if (elements.mapModal) {
            elements.mapModal.addEventListener('click', (e) => {
                if (e.target === elements.mapModal) {
                    elements.mapModal.classList.remove('active');
                }
            });
        }
        
        // Transport mode buttons
        document.querySelectorAll('.transport-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.transport-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentTransportMode = btn.dataset.mode;
            });
        });
        
        // Use current location button
        const useLocationBtn = document.getElementById('useCurrentLocation');
        if (useLocationBtn) {
            useLocationBtn.addEventListener('click', async () => {
                try {
                    const coords = await getCurrentLocation();
                    document.getElementById('startLocation').value = `${coords.lat}, ${coords.lng}`;
                    alert('Current location detected! Click "Get Directions" to see the route.');
                } catch (error) {
                    alert('Could not get your location. Please enter your address manually.');
                }
            });
        }
        
        // Get directions button
        const getDirectionsBtn = document.getElementById('getDirections');
        if (getDirectionsBtn) {
            getDirectionsBtn.addEventListener('click', async () => {
                const startLocation = document.getElementById('startLocation').value.trim();
                
                if (!startLocation) {
                    alert('Please enter a starting location or use your current location.');
                    return;
                }
                
                try {
                    let startCoords;
                    
                    // Check if input is coordinates
                    if (startLocation.match(/^-?\d+\.?\d*,\s*-?\d+\.?\d*$/)) {
                        const [lat, lng] = startLocation.split(',').map(s => parseFloat(s.trim()));
                        startCoords = { lat, lng };
                    } else {
                        // Geocode address
                        startCoords = await geocodeAddress(startLocation);
                    }
                    
                    showRoute(startCoords);
                } catch (error) {
                    alert('Could not find the location. Please try a different address.');
                }
            });
        }
        
        // Quick actions
        elements.quickActions.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.action;
                handleQuickAction(action);
            });
        });
        
        // Prompt cards
        elements.promptCards.forEach(card => {
            card.addEventListener('click', (e) => {
                const prompt = e.currentTarget.dataset.prompt;
                elements.userInput.value = prompt;
                handleSendMessage();
            });
        });
        
        // Close modal on outside click
        elements.settingsModal.addEventListener('click', (e) => {
            if (e.target === elements.settingsModal) {
                elements.settingsModal.classList.remove('active');
            }
        });
    }
    
    // Toggle sidebar
    function toggleSidebar() {
        elements.sidebar.classList.toggle('active');
        if (window.innerWidth > 768) {
            elements.mainContainer.classList.toggle('full-width');
            elements.sidebar.classList.toggle('hidden');
        }
    }
    
    // Create new chat
    function createNewChat() {
        chatManager.createNewChat();
        renderChatHistory();
        loadCurrentChat();
    }
    
    // Render chat history
    function renderChatHistory() {
        const chats = chatManager.getAllChats();
        elements.chatHistoryList.innerHTML = '';
        
        chats.forEach(chat => {
            const div = document.createElement('div');
            div.className = 'chat-history-item';
            if (chat.id === chatManager.currentChatId) {
                div.classList.add('active');
            }
            
            const date = new Date(chat.lastUpdated);
            const timeAgo = getTimeAgo(date);
            
            div.innerHTML = `
                <div class="chat-title">${chat.title}</div>
                <div class="chat-time">${timeAgo}</div>
            `;
            
            div.addEventListener('click', () => {
                chatManager.switchChat(chat.id);
                renderChatHistory();
                loadCurrentChat();
                if (window.innerWidth <= 768) {
                    toggleSidebar();
                }
            });
            
            elements.chatHistoryList.appendChild(div);
        });
    }
    
    // Load current chat
    function loadCurrentChat() {
        const messages = chatManager.getMessages();
        elements.messagesContainer.innerHTML = '';
        
        if (messages.length === 0) {
            elements.welcomeScreen.style.display = 'block';
            elements.messagesContainer.style.display = 'none';
        } else {
            elements.welcomeScreen.style.display = 'none';
            elements.messagesContainer.style.display = 'block';
            
            messages.forEach(msg => {
                if (msg.type === 'news') {
                    addNewsMessage(msg.content, msg.news);
                } else {
                    addMessage(msg.content, msg.role, false);
                }
            });
            
            scrollToBottom();
        }
    }
    
    // Handle input change
    function handleInputChange() {
        const text = elements.userInput.value;
        const wordCount = text.length;
        elements.wordCount.textContent = `${wordCount} / 2000`;
        adjustTextareaHeight();
        
        // Enable/disable send button
        elements.sendBtn.disabled = text.trim().length === 0;
    }
    
    // Adjust textarea height
    function adjustTextareaHeight() {
        elements.userInput.style.height = 'auto';
        elements.userInput.style.height = elements.userInput.scrollHeight + 'px';
    }
    
    // Handle keyboard shortcuts
    function handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    }
    
    // Handle send message
    async function handleSendMessage() {
        const message = elements.userInput.value.trim();
        if (!message || isProcessing) return;
        
        // Add user message
        addMessage(message, 'user', true);
        chatManager.addMessage({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });
        
        // Clear input
        elements.userInput.value = '';
        elements.wordCount.textContent = '0 / 2000';
        elements.sendBtn.disabled = true;
        adjustTextareaHeight();
        
        // Hide welcome screen
        elements.welcomeScreen.style.display = 'none';
        elements.messagesContainer.style.display = 'block';
        
        // Process query
        isProcessing = true;
        const loadingId = showLoading();
        
        try {
            await processQuery(message);
        } catch (error) {
            console.error('Error processing query:', error);
            addMessage('Sorry, I encountered an error. Please try again.', 'bot', true);
        } finally {
            removeLoading(loadingId);
            isProcessing = false;
            renderChatHistory();
        }
    }
    
    // Update language indicator
    function updateLanguageIndicator(langCode) {
        if (!elements.languageIndicator) return;
        
        const langNames = {
            'en': 'EN', 'hi': 'HI', 'ta': 'TA', 'te': 'TE', 'bn': 'BN',
            'mr': 'MR', 'gu': 'GU', 'kn': 'KN', 'ml': 'ML', 'pa': 'PA',
            'or': 'OR', 'as': 'AS', 'ur': 'UR'
        };
        
        const fullLangNames = {
            'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'bn': 'Bengali',
            'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi',
            'or': 'Odia', 'as': 'Assamese', 'ur': 'Urdu'
        };
        
        const code = langNames[langCode] || 'EN';
        const fullName = fullLangNames[langCode] || 'English';
        
        // Update display with language code and add emoji indicator
        elements.languageIndicator.textContent = ` 🌐 ${code}`;
        elements.languageIndicator.title = `🔄 Auto-Detected Language: ${fullName}\n📍 Input Language: ${fullName}\n💬 Response will be in ${fullName}`;
        
        // Add visual indicator when non-English
        if (langCode !== 'en') {
            elements.languageIndicator.style.backgroundColor = '#667eea';
            elements.languageIndicator.style.color = 'white';
        } else {
            elements.languageIndicator.style.backgroundColor = '';
            elements.languageIndicator.style.color = '';
        }
    }
    
    // Process user query
    async function processQuery(query) {
        const lowerQuery = query.toLowerCase();
        
        // Detect language of input - simple detection based on Unicode ranges
        let detectedLang = 'en';
        if (/[\u0B80-\u0BFF]/.test(query)) {
            detectedLang = 'ta'; // Tamil
        } else if (/[\u0C00-\u0C7F]/.test(query)) {
            detectedLang = 'te'; // Telugu
        } else if (/[\u0C80-\u0CFF]/.test(query)) {
            detectedLang = 'kn'; // Kannada
        } else if (/[\u0D00-\u0D7F]/.test(query)) {
            detectedLang = 'ml'; // Malayalam
        } else if (/[\u0900-\u097F]/.test(query)) {
            detectedLang = 'hi'; // Hindi
        } else if (/[\u0A80-\u0AFF]/.test(query)) {
            detectedLang = 'gu'; // Gujarati
        } else if (/[\u0A00-\u0A7F]/.test(query)) {
            detectedLang = 'pa'; // Punjabi
        } else if (/[\u0980-\u09FF]/.test(query)) {
            detectedLang = 'bn'; // Bengali
        }
        
        console.log('Detected language:', detectedLang, 'for query:', query);
        currentLanguage = detectedLang; // Store for voice output
        updateLanguageIndicator(detectedLang);
        
        // Check conversational responses with language support
        let conversational;
        if (typeof getTranslatedConversationalResponse !== 'undefined') {
            const translatedResponse = getTranslatedConversationalResponse(query, detectedLang);
            if (translatedResponse) {
                conversational = { content: translatedResponse, type: 'conversation' };
            }
        }
        
        if (!conversational) {
            conversational = getConversationalResponse(lowerQuery);
        }
        
        if (conversational) {
            await delay(500);
            addMessage(conversational.content, 'bot', true);
            chatManager.addMessage({
                role: 'bot',
                content: conversational.content,
                timestamp: new Date().toISOString()
            });
            return;
        }
        
        // Check if news query
        if (isNewsQuery(lowerQuery)) {
            const category = determineNewsCategory(lowerQuery);
            const searchTerm = extractSearchTerm(query);
            const news = await fetchNews(category, searchTerm, 5);
            
            if (news.length > 0) {
                const categoryName = category.charAt(0).toUpperCase() + category.slice(1);
                const text = searchTerm 
                    ? `Here are the latest articles about "${searchTerm}":`
                    : `Here are the latest ${categoryName} news:`;
                
                addNewsMessage(text, news);
                chatManager.addMessage({
                    role: 'bot',
                    type: 'news',
                    content: text,
                    news: news,
                    timestamp: new Date().toISOString()
                });
            } else {
                addMessage(`I couldn't find recent news for "${query}". Try a different topic or ask me another question!`, 'bot', true);
                chatManager.addMessage({
                    role: 'bot',
                    content: `No news found for "${query}"`,
                    timestamp: new Date().toISOString()
                });
            }
            return;
        }
        
        // Call backend API for knowledge base queries
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    language: detectedLang,  // Send detected language
                    session_id: chatManager.getCurrentChatId()
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                const backendResponse = data.response || data;
                const detectedLanguage = data.detected_language || detectedLang;
                
                // Update language indicator with backend detection
                updateLanguageIndicator(detectedLanguage);
                currentLanguage = detectedLanguage;
                
                // Only use backend response if it's not the default fallback message
                if (backendResponse && backendResponse.trim() !== '') {
                    await delay(500);
                    
                    // Add language notification if not English
                    let responseTitle = 'KARE AI Response';
                    if (detectedLanguage !== 'en') {
                        const langMap = {'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu', 'bn': 'Bengali', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam', 'pa': 'Punjabi', 'or': 'Odia'};
                        responseTitle += ` (in ${langMap[detectedLanguage] || 'Selected Language'})`;
                    }
                    
                    addFormattedMessage(responseTitle, backendResponse, 'bot');
                    chatManager.addMessage({
                        role: 'bot',
                        content: backendResponse,
                        detected_language: detectedLanguage,
                        timestamp: new Date().toISOString()
                    });
                    return;
                }
            }
        } catch (error) {
            console.error('Backend API error:', error);
            // Continue to fallback if API fails
        }
        
        // Fallback response if API doesn't return meaningful data
        await delay(500);
        const fallbackResponse = generateFallbackResponse(query);
        addMessage(fallbackResponse, 'bot', true);
        chatManager.addMessage({
            role: 'bot',
            content: fallbackResponse,
            timestamp: new Date().toISOString()
        });
    }
    
    // Generate fallback response
    function generateFallbackResponse(query) {
        const lowerQuery = query.toLowerCase();
        
        if (lowerQuery.includes('fee') || lowerQuery.includes('cost') || lowerQuery.includes('tuition')) {
            return ` **Fee Information**\n\nI can help you with fee structure! Try asking:\n\n• "What is the fee structure?"\n• "B.Tech fees"\n• "MBA fees"\n• "Hostel fees"\n• "Scholarships available"`;
        }
        
        if (lowerQuery.includes('admission') || lowerQuery.includes('apply') || lowerQuery.includes('entrance')) {
            return ` **Admission Information**\n\nI can provide admission details! Ask me:\n\n• "How to apply for B.Tech?"\n• "What is the admission process?"\n• "Entrance exams accepted"\n• "Eligibility criteria"`;
        }
        
        if (lowerQuery.includes('placement') || lowerQuery.includes('job') || lowerQuery.includes('company')) {
            return ` **Placement Information**\n\nI can tell you about placements! Try:\n\n• "What are the placements like?"\n• "Which companies recruit?"\n• "Average package"\n• "Placement statistics"`;
        }
        
        return ` **KARE Information Assistant**\n\nI can help you with information about Kalasalingam University:\n\n **Academics:** Programs, courses, departments\n **Fees:** Complete fee structure and scholarships\n **Admissions:** Process, eligibility, entrance exams\n **Faculty:** Department information\n **Campus:** Facilities, hostels, labs\n **Placements:** Companies, packages, statistics\n **Research:** Innovation, patents, Ph.D programs\n **Student Life:** Clubs, events, activities\n\n**Try asking:**\n• "What is the fee structure?"\n• "Tell me about CSE department"\n• "How to apply for MBA?"\n• "Which companies come for placements?"\n• "What facilities are available?"`;
    }
    
    // Add message to UI
    function addMessage(content, role, save = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '' : '';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Convert markdown-style formatting
        const formattedContent = formatMessage(content);
        contentDiv.innerHTML = formattedContent;
        
        if (settingsManager.get('showTimestamps')) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            contentDiv.appendChild(timeDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        elements.messagesContainer.appendChild(messageDiv);
        
        // Store bot messages for replay (but don't auto-speak - only speak when user clicks Read Aloud)
        if (role === 'bot') {
            // Extract plain text from HTML content for speech
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = formattedContent;
            const plainText = tempDiv.textContent || tempDiv.innerText || '';
            
            // Clean up text for better speech (remove special characters, extra spaces)
            const cleanText = plainText
                .replace(/\*\*/g, '') // Remove markdown bold markers
                .replace(/\*/g, '')   // Remove markdown italic markers
                .replace(/`/g, '')    // Remove code markers
                .replace(/\[|\]/g, '') // Remove brackets
                .replace(/\n+/g, '. ') // Replace newlines with periods
                .replace(/\s+/g, ' ')  // Normalize spaces
                .replace(/\.{2,}/g, '.') // Replace multiple periods
                .trim();
            
            // Store for replay when user clicks "Read Aloud" button
            lastBotMessage = {
                text: cleanText,
                language: currentLanguage
            };
            
            // Show replay button
            elements.replayBtn.classList.remove('hidden');
        }
        
        if (settingsManager.get('autoScroll')) {
            scrollToBottom();
        }
    }
    
    // Add formatted message
    function addFormattedMessage(title, content, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<h3>${title}</h3>${formatMessage(content)}`;
        
        if (settingsManager.get('showTimestamps')) {
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            contentDiv.appendChild(timeDiv);
        }
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        elements.messagesContainer.appendChild(messageDiv);
        
        // Store formatted messages for replay (but don't auto-speak - only speak when user clicks Read Aloud)
        const cleanTitle = title.replace(/[]/g, '').trim();
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = formatMessage(content);
        const cleanContent = (tempDiv.textContent || tempDiv.innerText || '')
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/`/g, '')
            .replace(/\[|\]/g, '')
            .replace(/\n+/g, '. ')
            .replace(/\s+/g, ' ')
            .replace(/\.{2,}/g, '.')
            .trim();
        
        const plainText = cleanTitle ? `${cleanTitle}. ${cleanContent}` : cleanContent;
        
        // Store for replay when user clicks "Read Aloud" button
        lastBotMessage = {
            text: plainText,
            language: currentLanguage
        };
        
        // Show replay button
        elements.replayBtn.classList.remove('hidden');
        
        if (settingsManager.get('autoScroll')) {
            scrollToBottom();
        }
    }
    
    // Add news message
    function addNewsMessage(text, newsItems) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `<strong>${text}</strong>`;
        
        newsItems.forEach(article => {
            const newsItem = document.createElement('div');
            newsItem.className = 'news-item';
            newsItem.innerHTML = `
                <span class="source-badge">${article.source}</span>
                <a href="${article.link}" target="_blank" class="news-title">${article.title}</a>
                <div class="news-meta">
                    <span> ${getTimeAgo(new Date(article.pubDate))}</span>
                </div>
                <div class="news-description">${article.description}</div>
            `;
            contentDiv.appendChild(newsItem);
        });
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        elements.messagesContainer.appendChild(messageDiv);
        
        // Store news headline for replay (but don't auto-speak - only speak when user clicks Read Aloud)
        const newsCount = newsItems.length;
        const cleanText = text.replace(/[]/g, '').trim();
        const headline = `${cleanText}. Found ${newsCount} news ${newsCount === 1 ? 'article' : 'articles'}.`;
        
        // Store for replay when user clicks "Read Aloud" button
        lastBotMessage = {
            text: headline,
            language: currentLanguage
        };
        
        // Show replay button
        elements.replayBtn.classList.remove('hidden');
        
        if (settingsManager.get('autoScroll')) {
            scrollToBottom();
        }
    }
    
    // Format message content
    function formatMessage(content) {
        return content
            // Convert URLs to clickable links FIRST (before other replacements)
            .replace(/(https?:\/\/[^\s<)]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/```(\w+)?\n([\s\S]+?)```/g, '<div class="code-block">$2</div>')
            .replace(/`(.+?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    // Show loading indicator
    function showLoading() {
        const loadingId = 'loading-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.id = loadingId;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        elements.messagesContainer.appendChild(messageDiv);
        scrollToBottom();
        
        return loadingId;
    }
    
    // Remove loading indicator
    function removeLoading(loadingId) {
        const loading = document.getElementById(loadingId);
        if (loading) loading.remove();
    }
    
    // Scroll to bottom
    function scrollToBottom() {
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }
    
    // Toggle theme
    function toggleTheme() {
        const currentTheme = settingsManager.get('theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        settingsManager.set('theme', newTheme);
        elements.themeToggle.textContent = newTheme === 'dark' ? '' : '';
    }
    
    // Apply settings
    function applySettings() {
        const settings = settingsManager.getAll();
        elements.themeSelect.value = settings.theme;
        elements.fontSizeSelect.value = settings.fontSize;
        elements.autoScroll.checked = settings.autoScroll;
        elements.soundEffects.checked = settings.soundEffects;
        elements.voiceOutput.checked = settings.voiceOutput;
        elements.showTimestamps.checked = settings.showTimestamps;
        elements.themeToggle.textContent = settings.theme === 'dark' ? '' : '';
        updateSpeakerButton();
    }
    
    // Handle quick actions
    function handleQuickAction(action) {
        const queries = {
            'admissions': 'How to apply for admissions?',
            'fees': 'What is the fee structure?',
            'placements': 'Tell me about placements',
            'facilities': 'What facilities are available?',
            'programs': 'What programs are offered?',
            'contact': 'Contact information'
        };
        
        elements.userInput.value = queries[action] || '';
        handleSendMessage();
    }
    
    // Handle voice input
    function handleVoiceInput() {
        console.log('Voice button clicked, isListening:', voiceManager.isListening);
        
        if (voiceManager.isListening) {
            console.log('Stopping voice listening...');
            voiceManager.stopListening();
            elements.voiceBtn.textContent = '🎤';
        } else {
            console.log('Starting voice listening...');
            elements.voiceBtn.textContent = '⏹';
            voiceManager.startListening(
                (transcript) => {
                    console.log('Transcript received:', transcript);
                    elements.userInput.value = transcript;
                    elements.voiceBtn.textContent = '🎤';
                    handleInputChange();
                },
                (error) => {
                    console.error('Voice error:', error);
                    elements.voiceBtn.textContent = '🎤';
                    alert('Voice recognition error: ' + error);
                }
            );
        }
    }
    
    // Toggle voice output
    function toggleVoiceOutput() {
        const currentSetting = settingsManager.get('voiceOutput');
        const newSetting = !currentSetting;
        
        settingsManager.set('voiceOutput', newSetting);
        elements.voiceOutput.checked = newSetting;
        updateSpeakerButton();
    }
    
    // Stop current audio playback
    function stopCurrentAudio() {
        voiceManager.stopSpeaking();
        elements.stopAudioBtn.classList.add('hidden');
    }
    
    // Replay last bot message
    function replayLastMessage() {
        if (lastBotMessage) {
            voiceManager.speak(lastBotMessage.text, lastBotMessage.language);
        }
    }
    
    // Update speaker button icon
    function updateSpeakerButton() {
        const isEnabled = settingsManager.get('voiceOutput');
        elements.speakerBtn.textContent = isEnabled ? '' : '';
        elements.speakerBtn.title = isEnabled ? 'Voice Output: ON (Click to disable)' : 'Voice Output: OFF (Click to enable)';
    }
    
    // Clear chat history
    function clearChatHistory() {
        if (confirm('Are you sure you want to clear all chat history?')) {
            chatManager.clearAllChats();
            renderChatHistory();
            loadCurrentChat();
            elements.settingsModal.classList.remove('active');
        }
    }
    
    // Export current chat
    function exportCurrentChat() {
        const exportData = chatManager.exportChat();
        if (exportData) {
            const blob = new Blob([exportData], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `kare-ai-chat-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    }
    
    // Helper: delay
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Helper: get time ago
    function getTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);
        const intervals = {
            year: 31536000,
            month: 2592000,
            week: 604800,
            day: 86400,
            hour: 3600,
            minute: 60
        };
        
        for (const [unit, secondsInUnit] of Object.entries(intervals)) {
            const interval = Math.floor(seconds / secondsInUnit);
            if (interval >= 1) {
                return `${interval} ${unit}${interval > 1 ? 's' : ''} ago`;
            }
        }
        return 'Just now';
    }

    // Helper: get conversational response
    function getConversationalResponse(query) {
        const greetings = {
            'hi': 'Hello! How can I help you today?',
            'hello': 'Hi there! What can I assist you with?',
            'hey': 'Hey! What would you like to know?',
            'howdy': 'Howdy! How can I assist?',
            'good morning': 'Good morning! How can I help?',
            'good afternoon': 'Good afternoon! What do you need?',
            'good evening': 'Good evening! How can I help?',
            'thanks': 'You\'re welcome!',
            'thank you': 'Happy to help!',
            'bye': 'Goodbye! Feel free to ask anytime.',
            'goodbye': 'See you later!',
            'ok': 'Sure, anything else?',
            'okay': 'Alright!'
        };

        for (const [key, response] of Object.entries(greetings)) {
            if (query.includes(key)) {
                return { content: response, type: 'greeting' };
            }
        }
        return null;
    }

    // Helper: find AI knowledge (placeholder - extend as needed)
    function findAIKnowledge(query) {
        // This would integrate with your knowledge base
        return null;
    }

    // Helper: check if news query
    function isNewsQuery(query) {
        const newsKeywords = ['news', 'latest', 'recent', 'trending', 'update', 'breaking'];
        return newsKeywords.some(kw => query.includes(kw));
    }

    // Helper: determine news category
    function determineNewsCategory(query) {
        if (query.includes('tech')) return 'technology';
        if (query.includes('business') || query.includes('finance')) return 'business';
        if (query.includes('sport')) return 'sports';
        if (query.includes('health') || query.includes('science')) return 'science';
        return 'general';
    }

    // Helper: extract search term
    function extractSearchTerm(query) {
        return query.replace(/news|latest|recent|trending|update|breaking/gi, '').trim();
    }

    // Helper: fetch news (placeholder)
    async function fetchNews(category, searchTerm, limit) {
        return [];
    }

    // Helper: translate response (placeholder)
    async function translateResponse(text, language) {
        return text;
    }

    // Helper: detect language
    function detectLanguage(text) {
        if (/[\u0900-\u097F]/.test(text)) return 'hi';
        if (/[\u0C00-\u0C7F]/.test(text)) return 'te';
        if (/[\u0B80-\u0BFF]/.test(text)) return 'ta';
        return 'en';
    }
    // Map functionality
    let map = null;
    let routingControl = null;
    let currentTransportMode = 'car';
    
    // Kalasalingam University coordinates
    const kareLocation = {
        lat: 9.2100,
        lng: 77.5656,
        name: 'Kalasalingam Academy of Research and Education'
    };
    
    // Initialize map
    function initMap() {
        if (!map) {
            map = L.map('map').setView([kareLocation.lat, kareLocation.lng], 13);
            
            // Add OpenStreetMap tiles
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 19
            }).addTo(map);
            
            // Add marker for Kalasalingam University
            L.marker([kareLocation.lat, kareLocation.lng])
                .addTo(map)
                .bindPopup(kareLocation.name)
                .openPopup();
        }
    }
    
    // Get user's current location
    function getCurrentLocation() {
        return new Promise((resolve, reject) => {
            if (!navigator.geolocation) {
                reject(new Error('Geolocation not supported'));
                return;
            }
            
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    resolve({
                        lat: position.coords.latitude,
                        lng: position.coords.longitude
                    });
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }
    
    // Geocode address to coordinates
    async function geocodeAddress(address) {
        try {
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`
            );
            const data = await response.json();
            
            if (data && data.length > 0) {
                return {
                    lat: parseFloat(data[0].lat),
                    lng: parseFloat(data[0].lon)
                };
            }
            throw new Error('Location not found');
        } catch (error) {
            console.error('Geocoding error:', error);
            throw error;
        }
    }
    
    // Get routing profile based on transport mode
    function getRoutingProfile(mode) {
        switch(mode) {
            case 'car': return 'mapbox/driving';
            case 'bike': return 'mapbox/cycling';
            case 'foot': return 'mapbox/walking';
            default: return 'mapbox/driving';
        }
    }
    
    // Show route on map
    function showRoute(startCoords) {
        if (routingControl) {
            map.removeControl(routingControl);
        }
        
        routingControl = L.Routing.control({
            waypoints: [
                L.latLng(startCoords.lat, startCoords.lng),
                L.latLng(kareLocation.lat, kareLocation.lng)
            ],
            router: L.Routing.osrmv1({
                serviceUrl: 'https://router.project-osrm.org/route/v1',
                profile: currentTransportMode === 'foot' ? 'foot' : currentTransportMode === 'bike' ? 'bike' : 'car'
            }),
            createMarker: function(i, waypoint, n) {
                const marker = L.marker(waypoint.latLng, {
                    draggable: false
                });
                
                if (i === 0) {
                    marker.bindPopup('Starting Point');
                } else {
                    marker.bindPopup(kareLocation.name);
                }
                
                return marker;
            },
            lineOptions: {
                styles: [{color: currentTransportMode === 'car' ? 'blue' : currentTransportMode === 'bike' ? 'green' : 'orange', weight: 5}]
            },
            show: true,
            addWaypoints: false,
            routeWhileDragging: false,
            fitSelectedRoutes: true,
            showAlternatives: true
        }).addTo(map);
        
        routingControl.on('routesfound', function(e) {
            const routes = e.routes;
            const summary = routes[0].summary;
            
            const distance = (summary.totalDistance / 1000).toFixed(2);
            const duration = Math.round(summary.totalTime / 60);
            
            const routeInfo = document.getElementById('routeInfo');
            const routeDetails = document.getElementById('routeDetails');
            
            let transportIcon = '';
            let transportName = '';
            switch(currentTransportMode) {
                case 'car':
                    transportIcon = 'Car';
                    transportName = 'Driving';
                    break;
                case 'bike':
                    transportIcon = 'Bike';
                    transportName = 'Cycling';
                    break;
                case 'foot':
                    transportIcon = 'Walk';
                    transportName = 'Walking';
                    break;
            }
            
            routeDetails.innerHTML = `
                <p><strong>Transport Mode:</strong> ${transportName}</p>
                <p><strong>Total Distance:</strong> ${distance} km</p>
                <p><strong>Estimated Time:</strong> ${duration} minutes</p>
                <p><strong>Alternative Routes:</strong> ${routes.length > 1 ? routes.length + ' routes available' : 'See map for route'}</p>
            `;
            
            routeInfo.style.display = 'block';
        });
    }
    
    // Initialize on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
