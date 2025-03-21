// Global variables
let sessionId = null;
let websocket = null;
let currentQuery = null;
let currentAnswer = "";
let isStreaming = false;
let uploadedDocuments = [];
let documentStats = null;
let processingCompleted = false;
const API_BASE_URL = window.location.protocol + '//' + window.location.host;
const WS_BASE_URL = API_BASE_URL.replace('http', 'ws');

document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM Elements
    const elements = {
        uploadArea: document.getElementById('upload-area'),
        fileInput: document.getElementById('file-input'),
        processingStatus: document.getElementById('processing-status'),
        statusMessage: document.getElementById('status-message'),
        progressBarFill: document.getElementById('progress-bar-fill'),
        uploadSuccess: document.getElementById('upload-success'),
        uploadError: document.getElementById('upload-error'),
        errorMessage: document.getElementById('error-message'),
        queryInput: document.getElementById('query-input'),
        sendButton: document.getElementById('send-button'),
        messagesContainer: document.getElementById('messages'),
        summaryToggle: document.getElementById('summary-toggle'),
        summaryContainer: document.getElementById('summary-container'),
        documentSummary: document.getElementById('document-summary'),
        summaryChevron: document.getElementById('summary-chevron'),
        comparisonSection: document.getElementById('comparison-section'),
        documentSelector: document.getElementById('document-selector'),
        compareButton: document.getElementById('compare-button'),
        comparisonResults: document.getElementById('comparison-results'),
        similarityScore: document.getElementById('similarity-score'),
        commonTopics: document.getElementById('common-topics'),
        uniqueTopics1: document.getElementById('unique-topics1'),
        uniqueTopics2: document.getElementById('unique-topics2'),
        comparisonTypeOptions: document.querySelectorAll('.comparison-type-option'),
        insightsSection: document.getElementById('insights-section'),
        generateInsightsButton: document.getElementById('generate-insights-button'),
        insightsContainer: document.getElementById('insights-container'),
        tabButtons: document.querySelectorAll('.tab-button'),
        tabContents: document.querySelectorAll('.tab-content'),
        advancedNlpToggle: document.getElementById('advanced-nlp-toggle'),
        uploadAnotherButton: document.getElementById('upload-another-button'),
        tryAgainButton: document.getElementById('try-again-button'),
        scrollControls: document.querySelector('.summary-controls'),
        scrollIndicator: document.getElementById('scroll-indicator')
    };

    // Setup event listeners
    setupEventListeners(elements);

    // Load previously uploaded documents
    loadUploadedDocuments();

    // UI Utility Functions
    function toggleSummary() {
        const isHidden = elements.summaryContainer.classList.contains('hidden');
        elements.summaryContainer.classList.toggle('hidden', !isHidden);
        elements.summaryToggle.textContent = isHidden ? "Hide Summary" : "Show Summary";
        elements.scrollControls.classList.toggle('hidden', !isHidden);
    }

    function scrollSummary(amount) {
        elements.summaryContainer.scrollBy({
            top: amount,
            behavior: 'smooth'
        });
        updateScrollIndicator();
    }

    function updateScrollIndicator() {
        const maxScroll = elements.summaryContainer.scrollHeight - elements.summaryContainer.clientHeight;
        const currentScroll = elements.summaryContainer.scrollTop;
        if (maxScroll > 0) {
            const scrollPercentage = Math.round((currentScroll / maxScroll) * 100);
            elements.scrollIndicator.textContent = `Scroll position: ${scrollPercentage}%`;
            elements.scrollIndicator.classList.remove('hidden');
        } else {
            elements.scrollIndicator.classList.add('hidden');
        }
    }

    function resetUI() {
        if (websocket) {
            websocket.close();
            websocket = null;
        }
        sessionId = null;
        elements.fileInput.value = '';
        elements.progressBarFill.style.width = '0%';
        elements.queryInput.disabled = true;
        elements.sendButton.disabled = true;
    }

    function showError(message) {
        elements.processingStatus.classList.add('hidden');
        elements.uploadError.classList.remove('hidden');
        elements.uploadArea.classList.remove('hidden');
        elements.errorMessage.textContent = message;
    }

    // Message Handling
    function addMessage(content, type) {
        const messageElement = document.createElement('div');
        messageElement.className = `${type}-message p-3 text-gray-700 self-${type === 'user' ? 'end' : 'start'} max-w-3xl`;
        const formattedContent = type === 'user' ? escapeHtml(content) : formatMessage(content);
        messageElement.innerHTML = `<p>${formattedContent}</p>`;
        elements.messagesContainer.querySelector('.flex').appendChild(messageElement);
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    }

    function formatMessage(message) {
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // File Upload Handling
    function handleFileUpload(file) {
        processingCompleted = false;
        const useAdvancedNlp = elements.advancedNlpToggle.checked;

        elements.uploadArea.classList.add('hidden');
        elements.processingStatus.classList.remove('hidden');
        elements.uploadSuccess.classList.add('hidden');
        elements.uploadError.classList.add('hidden');
        elements.statusMessage.textContent = "Uploading document...";
        elements.progressBarFill.style.width = "10%";

        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_advanced_nlp', useAdvancedNlp);

        fetch(`${API_BASE_URL}/upload/`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error('Failed to upload file');
            return response.json();
        })
        .then(data => {
            sessionId = data.session_id;
            const documentInfo = {
                id: sessionId,
                name: file.name,
                timestamp: new Date().toISOString()
            };
            uploadedDocuments.push(documentInfo);
            localStorage.setItem('uploadedDocuments', JSON.stringify(uploadedDocuments));
            updateDocumentSelector();
            elements.comparisonSection.classList.remove('hidden');
            elements.progressBarFill.style.width = "20%";
            connectWebSocket();
            elements.insightsSection.classList.remove('hidden');
        })
        .catch(error => showError(error.message));
    }

    // Document Processing
    function startProcessing() {
        if (!sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        processingCompleted = false;
        websocket.send(JSON.stringify({
            type: "start_processing",
            use_advanced_nlp: elements.advancedNlpToggle.checked
        }));
    }

    // WebSocket Communication
    function connectWebSocket() {
        if (websocket) websocket.close();
        websocket = new WebSocket(`${WS_BASE_URL}/ws/${sessionId}`);

        websocket.onopen = function() {
            console.log("WebSocket connection established");
            startProcessing();
        };

        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const messageHandlers = {
                status: handleStatusMessage,
                summary: handleSummaryMessage,
                start_stream: handleStartStreamMessage,
                token: handleTokenMessage,
                end_stream: handleEndStreamMessage,
                error: handleErrorMessage,
                feedback_received: handleFeedbackMessage,
                comparison_result: handleComparisonResultMessage,
                visualization_result: handleVisualizationResultMessage
            };
            if (messageHandlers[data.type]) messageHandlers[data.type](data);
        };

        websocket.onclose = function(event) {
            if (!event.wasClean) console.error(`WebSocket closed unexpectedly: ${event.code}`);
        };

        websocket.onerror = function(error) {
            console.error(`WebSocket error: ${error.message}`);
            showError("WebSocket connection error. Please try refreshing the page.");
        };
    }

    function handleStatusMessage(data) {
        elements.statusMessage.textContent = data.content;
        if (data.content.includes("Starting to process")) {
            elements.progressBarFill.style.width = "30%";
        } else if (data.content.includes("Processing page")) {
            const match = data.content.match(/Processing page (\d+) of (\d+)/);
            if (match) {
                const current = parseInt(match[1]);
                const total = parseInt(match[2]);
                const progress = 30 + (current / total) * 30;
                elements.progressBarFill.style.width = `${progress}%`;
            }
        } else if (data.content.includes("Text extraction complete")) {
            elements.progressBarFill.style.width = "60%";
        } else if (data.content.includes("Chunking complete")) {
            elements.progressBarFill.style.width = "70%";
        } else if (data.content.includes("Vector indexing complete")) {
            elements.progressBarFill.style.width = "80%";
        } else if (data.content.includes("Generating document summary")) {
            elements.progressBarFill.style.width = "90%";
        } else if (data.content.includes("Document processing complete")) {
            elements.progressBarFill.style.width = "100%";
            elements.processingStatus.classList.add('hidden');
            elements.uploadSuccess.classList.remove('hidden');
            elements.queryInput.disabled = false;
            elements.sendButton.disabled = false;
            if (!processingCompleted) {
                addMessage('Document processed successfully! You can now ask questions about it.', 'bot');
                processingCompleted = true;
            }
        } else if (data.content.includes("Thinking")) {
            const typingIndicator = document.createElement('div');
            typingIndicator.className = 'bot-message p-3 text-gray-700 self-start max-w-3xl typing-indicator';
            typingIndicator.innerHTML = `
                <div class="flex items-center">
                    <span class="mr-2">Thinking</span>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            elements.messagesContainer.querySelector('.flex').appendChild(typingIndicator);
            elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
        }
    }

    function handleSummaryMessage(data) {
        elements.documentSummary.innerHTML = formatMessage(data.content);
        elements.summaryContainer.classList.add('open');
        elements.scrollControls.classList.remove('hidden');
        elements.summaryChevron.innerHTML = `
            <path fill-rule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 111.414 1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clip-rule="evenodd" />
        `;
        updateScrollIndicator();
    }

    function handleStartStreamMessage() {
        const typingIndicator = elements.messagesContainer.querySelector('.typing-indicator');
        if (typingIndicator) typingIndicator.remove();
        isStreaming = true;
        currentAnswer = "";
        const messageElement = document.createElement('div');
        messageElement.className = 'bot-message p-3 text-gray-700 self-start max-w-3xl streaming-message';
        messageElement.innerHTML = `<p></p>`;
        elements.messagesContainer.querySelector('.flex').appendChild(messageElement);
    }

    // ... existing code ...

function handleTokenMessage(data) {
    currentAnswer += data.content;
    const streamingMessage = elements.messagesContainer.querySelector('.streaming-message p');
    if (streamingMessage) {
        // Create typing effect by adding one character at a time
        if (!streamingMessage.dataset.fullText) {
            streamingMessage.dataset.fullText = currentAnswer;
            streamingMessage.dataset.displayedText = '';
            setTimeout(simulateTyping, 30, streamingMessage);
        } else {
            streamingMessage.dataset.fullText = currentAnswer;
        }
    }
}

function simulateTyping(element) {
    const fullText = element.dataset.fullText;
    const displayedText = element.dataset.displayedText;
    
    if (displayedText.length < fullText.length) {
        // Add the next character
        const nextChar = fullText.charAt(displayedText.length);
        element.dataset.displayedText += nextChar;
        element.innerHTML = formatMessage(element.dataset.displayedText);
        
        // Scroll to bottom
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
        
        // Random delay between 15-70ms for natural typing feel
        const typingSpeed = Math.floor(Math.random() * 56) + 15;
        setTimeout(simulateTyping, typingSpeed, element);
    }
}


    function handleEndStreamMessage() {
        isStreaming = false;
        const streamingMessage = elements.messagesContainer.querySelector('.streaming-message');
        if (streamingMessage) {
            streamingMessage.classList.remove('streaming-message');
            const feedbackDiv = document.createElement('div');
            feedbackDiv.className = 'feedback-buttons';
            feedbackDiv.innerHTML = `
                <button class="feedback-button thumbs-up" title="This was helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                    </svg>
                </button>
                <button class="feedback-button thumbs-down" title="This was not helpful">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.095c-.5 0-.905-.405-.905-.905 0-.714.211-1.412.608-2.006L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                    </svg>
                </button>
            `;
            streamingMessage.appendChild(feedbackDiv);
            const thumbsUp = feedbackDiv.querySelector('.thumbs-up');
            const thumbsDown = feedbackDiv.querySelector('.thumbs-down');
            thumbsUp.addEventListener('click', () => {
                sendFeedback('thumbs_up');
                thumbsUp.classList.add('active');
                thumbsDown.classList.remove('active');
            });
            thumbsDown.addEventListener('click', () => {
                sendFeedback('thumbs_down');
                thumbsDown.classList.add('active');
                thumbsUp.classList.remove('active');
            });
        }
    }

    function handleErrorMessage(data) {
        showError(data.content);
    }

    function handleFeedbackMessage(data) {
        console.log("Feedback received:", data.content);
    }

    function handleComparisonResultMessage(data) {
        displayComparisonResults(data.content);
    }

    function handleVisualizationResultMessage(data) {
        displayVisualizationResults(data.content);
    }

    function sendQuery() {
        const query = elements.queryInput.value.trim();
        if (!query || !sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        currentQuery = query;
        addMessage(query, 'user');
        elements.queryInput.value = '';
        websocket.send(JSON.stringify({ type: "query", content: query }));
    }

    function sendFeedback(rating) {
        if (!sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        websocket.send(JSON.stringify({
            type: "feedback",
            rating: rating,
            query: currentQuery,
            answer: currentAnswer
        }));
    }

    // Document Comparison
    function compareDocuments() {
        const otherDocumentId = elements.documentSelector.value;
        if (!otherDocumentId || !sessionId) return;
        const comparisonType = document.querySelector('.comparison-type-option.active')?.dataset.type;
        elements.compareButton.disabled = true;
        elements.compareButton.textContent = "Comparing...";
        websocket.send(JSON.stringify({
            type: "compare_documents",
            session_id1: sessionId,
            session_id2: otherDocumentId,
            comparison_type: comparisonType
        }));
    }

    function displayComparisonResults(result) {
        const score = Math.round(result.similarity_score * 100);
        elements.similarityScore.textContent = `${score}%`;
        elements.similarityScore.style.backgroundColor = score >= 80 ? "#10b981" : score >= 50 ? "#f59e0b" : "#ef4444";
        displayTopics(elements.commonTopics, result.common_topics);
        displayTopics(elements.uniqueTopics1, result.unique_topics1);
        displayTopics(elements.uniqueTopics2, result.unique_topics2);
        elements.comparisonResults.classList.remove('hidden');
        elements.compareButton.disabled = false;
        elements.compareButton.textContent = "Compare Documents";
    }

    function displayTopics(container, topics) {
        container.innerHTML = '';
        topics.forEach(topic => {
            const topicElement = document.createElement('span');
            topicElement.className = 'topic-item';
            topicElement.textContent = topic;
            container.appendChild(topicElement);
        });
    }

    function updateDocumentSelector() {
        while (elements.documentSelector.options.length > 1) elements.documentSelector.remove(1);
        uploadedDocuments.forEach(doc => {
            if (doc.id !== sessionId) {
                const option = document.createElement('option');
                option.value = doc.id;
                option.textContent = doc.name;
                elements.documentSelector.appendChild(option);
            }
        });
        elements.compareButton.disabled = elements.documentSelector.options.length <= 1;
    }

    // Document Insights
    function generateDocumentInsights() {
        if (!sessionId) return;
        elements.generateInsightsButton.disabled = true;
        elements.generateInsightsButton.textContent = "Generating insights...";
        fetch(`${API_BASE_URL}/document_stats/${sessionId}`)
            .then(response => {
                if (!response.ok) throw new Error('Failed to generate document insights');
                return response.json();
            })
            .then(data => {
                documentStats = data;
                displayDocumentInsights(data);
                elements.insightsContainer.classList.remove('hidden');
                elements.generateInsightsButton.disabled = false;
                elements.generateInsightsButton.textContent = "Refresh Document Insights";
            })
            .catch(error => {
                console.error('Error generating insights:', error);
                elements.generateInsightsButton.disabled = false;
                elements.generateInsightsButton.textContent = "Generate Document Insights";
                alert('Error generating document insights. Please try again.');
            });
    }

    function displayDocumentInsights(data) {
        document.getElementById('word-count').textContent = data.basic_stats.word_count.toLocaleString();
        document.getElementById('sentence-count').textContent = data.basic_stats.sentence_count.toLocaleString();
        document.getElementById('paragraph-count').textContent = data.basic_stats.paragraph_count.toLocaleString();
        document.getElementById('readability-score').textContent = data.basic_stats.readability_score;
        document.getElementById('readability-level').textContent = data.basic_stats.readability_level;
        const readabilityFill = document.getElementById('readability-fill');
        const readabilityScore = data.basic_stats.readability_score;
        readabilityFill.style.width = `${readabilityScore}%`;
        readabilityFill.style.backgroundColor = readabilityScore >= 70 ? '#10b981' : readabilityScore >= 50 ? '#f59e0b' : '#ef4444';
        renderEntities(data.top_entities);
        document.getElementById('people-count').textContent = data.top_entities.people.length;
        document.getElementById('org-count').textContent = data.top_entities.organizations.length;
        document.getElementById('location-count').textContent = data.top_entities.locations.length;
        document.getElementById('date-count').textContent = data.top_entities.dates.length;
        renderKeyPhrases(data.top_phrases);
        renderWordCloud(data.top_words);
    }

    function renderEntities(entities) {
        renderEntityType('people-entities', entities.people, 'person');
        renderEntityType('organization-entities', entities.organizations, 'organization');
        renderEntityType('location-entities', entities.locations, 'location');
        renderEntityType('date-entities', entities.dates, 'date');
    }

    function renderEntityType(containerId, entities, className) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        if (entities.length === 0) {
            container.innerHTML = `<p class="text-sm text-gray-500">No ${className}s detected in this document.</p>`;
        } else {
            entities.forEach(entity => {
                const entityElement = document.createElement('span');
                entityElement.className = `entity-tag ${className}`;
                entityElement.textContent = entity;
                container.appendChild(entityElement);
            });
        }
    }

    function renderKeyPhrases(phrases) {
        const phrasesContainer = document.getElementById('key-phrases');
        phrasesContainer.innerHTML = '';
        if (phrases.length === 0) {
            phrasesContainer.innerHTML = '<p class="text-sm text-gray-500">No key phrases detected in this document.</p>';
            return;
        }
        phrases.forEach(phrase => {
            const phraseElement = document.createElement('div');
            phraseElement.className = 'phrase-card';
            phraseElement.textContent = phrase;
            phrasesContainer.appendChild(phraseElement);
        });
    }

    function renderWordCloud(wordFrequencies) {
        const container = document.getElementById('word-cloud');
        container.innerHTML = '';
        const width = container.clientWidth;
        const height = container.clientHeight;
        const svg = d3.select('#word-cloud')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        const words = Object.entries(wordFrequencies).map(([text, value]) => ({
            text,
            size: Math.log(value) * 10 + 10
        }));
        const colorScale = d3.scaleOrdinal()
            .domain(words.map(d => d.text))
            .range(d3.schemeCategory10);
        const layout = d3.layout.cloud()
            .size([width, height])
            .words(words)
            .padding(5)
            .rotate(() => 0)
            .fontSize(d => d.size)
            .on('end', draw);
        layout.start();

        function draw(words) {
            svg.append('g')
                .attr('transform', `translate(${width / 2},${height / 2})`)
                .selectAll('text')
                .data(words)
                .enter()
                .append('text')
                .style('font-size', d => `${d.size}px`)
                .style('fill', d => colorScale(d.text))
                .attr('text-anchor', 'middle')
                .attr('transform', d => `translate(${d.x},${d.y})`)
                .text(d => d.text);
        }
    }

    // Visualization (Placeholder - Customize as needed)
    function displayVisualizationResults(data) {
        // Assuming data contains visualization info (e.g., chart data)
        console.log("Visualization data received:", data);
        // Example: Display a simple message or integrate with a charting library like Chart.js
        const vizContainer = document.getElementById('visualization-container'); // Ensure this exists in HTML
        if (vizContainer) {
            vizContainer.innerHTML = `<p>Visualization data: ${JSON.stringify(data)}</p>`;
        }
    }

    // Data Management
    function loadUploadedDocuments() {
        const storedDocuments = localStorage.getItem('uploadedDocuments');
        if (storedDocuments) {
            try {
                uploadedDocuments = JSON.parse(storedDocuments);
            } catch (e) {
                console.error('Error parsing stored documents:', e);
                uploadedDocuments = [];
            }
        }
    }

    // Setup Event Listeners
    function setupEventListeners(elements) {
        // File Upload Listeners
        elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
        elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.uploadArea.classList.add('border-blue-500');
        });
        elements.uploadArea.addEventListener('dragleave', () => {
            elements.uploadArea.classList.remove('border-blue-500');
        });
        elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.uploadArea.classList.remove('border-blue-500');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                handleFileUpload(files[0]);
            } else {
                showError('Please upload a PDF file');
            }
        });
        elements.fileInput.addEventListener('change', () => {
            if (elements.fileInput.files.length > 0) handleFileUpload(elements.fileInput.files[0]);
        });

        // Query Input Listeners
        elements.sendButton.addEventListener('click', sendQuery);
        elements.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuery();
        });

        // Summary Listeners
        elements.summaryToggle.addEventListener('click', toggleSummary);
        document.getElementById('scroll-up').addEventListener('click', () => scrollSummary(-100));
        document.getElementById('scroll-down').addEventListener('click', () => scrollSummary(100));
        elements.summaryContainer.addEventListener('scroll', updateScrollIndicator);

        // Keyboard Navigation for Summary
        document.addEventListener('keydown', (e) => {
            if (elements.summaryContainer.classList.contains('open') && document.activeElement !== elements.queryInput) {
                switch (e.key) {
                    case 'ArrowUp':
                        e.preventDefault();
                        scrollSummary(-100);
                        break;
                    case 'ArrowDown':
                        e.preventDefault();
                        scrollSummary(100);
                        break;
                    case 'Home':
                        e.preventDefault();
                        elements.summaryContainer.scrollTop = 0;
                        updateScrollIndicator();
                        break;
                    case 'End':
                        e.preventDefault();
                        elements.summaryContainer.scrollTop = elements.summaryContainer.scrollHeight;
                        updateScrollIndicator();
                        break;
                }
            }
        });

        // Document Comparison Listeners
        elements.compareButton.addEventListener('click', compareDocuments);
        elements.documentSelector.addEventListener('change', () => {
            elements.compareButton.disabled = !elements.documentSelector.value;
        });
        elements.comparisonTypeOptions.forEach(option => {
            option.addEventListener('click', function() {
                elements.comparisonTypeOptions.forEach(opt => opt.classList.remove('active'));
                this.classList.add('active');
            });
        });

        // Insights Listeners
        elements.generateInsightsButton.addEventListener('click', generateDocumentInsights);

        // Tab Switching
        elements.tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const tabId = this.dataset.tab;
                elements.tabButtons.forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                elements.tabContents.forEach(content => content.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
            });
        });

        // Reset UI Buttons
        elements.uploadAnotherButton.addEventListener('click', () => {
            resetUI();
            elements.uploadArea.classList.remove('hidden');
            elements.uploadSuccess.classList.add('hidden');
            elements.processingStatus.classList.add('hidden');
            processingCompleted = false;
        });
        elements.tryAgainButton.addEventListener('click', () => {
            resetUI();
            elements.uploadArea.classList.remove('hidden');
            elements.uploadError.classList.add('hidden');
            processingCompleted = false;
        });
    }
});