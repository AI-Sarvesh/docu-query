// Global variables
let sessionId = null;
let websocket = null;
let currentQuery = null;
let currentAnswer = "";
let isStreaming = false;
let uploadedDocuments = [];
let visualizationData = null;
let documentStats = null;
const API_BASE_URL = window.location.protocol + '//' + window.location.host;
// Flag to track if processing has been completed to prevent duplicate messages
let processingCompleted = false;

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const processingStatus = document.getElementById('processing-status');
    const statusMessage = document.getElementById('status-message');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const uploadSuccess = document.getElementById('upload-success');
    const uploadError = document.getElementById('upload-error');
    const errorMessage = document.getElementById('error-message');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.getElementById('messages');
    const summaryToggle = document.getElementById('summary-toggle');
    const summaryContainer = document.getElementById('summary-container');
    const documentSummary = document.getElementById('document-summary');
    const summaryChevron = document.getElementById('summary-chevron');
    const comparisonSection = document.getElementById('comparison-section');
    const documentSelector = document.getElementById('document-selector');
    const compareButton = document.getElementById('compare-button');
    const comparisonResults = document.getElementById('comparison-results');
    const similarityScore = document.getElementById('similarity-score');
    const commonTopics = document.getElementById('common-topics');
    const uniqueTopics1 = document.getElementById('unique-topics1');
    const uniqueTopics2 = document.getElementById('unique-topics2');
    const comparisonTypeOptions = document.querySelectorAll('.comparison-type-option');
    const insightsSection = document.getElementById('insights-section');
    const generateInsightsButton = document.getElementById('generate-insights-button');
    const insightsContainer = document.getElementById('insights-container');
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    const advancedNlpToggle = document.getElementById('advanced-nlp-toggle');
    const uploadAnotherButton = document.getElementById('upload-another-button');
    const tryAgainButton = document.getElementById('try-again-button');
    
    // Handle file upload via click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Handle file upload via drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-blue-500');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-blue-500');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-blue-500');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            handleFileUpload(files[0]);
        } else {
            showError('Please upload a PDF file');
        }
    });
    
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
    
    sendButton.addEventListener('click', sendQuery);
    
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });
    
    // Toggle summary visibility
    summaryToggle.addEventListener('click', () => {
        toggleSummary();
    });
    
    // Add scroll controls for summary
    document.getElementById('scroll-up').addEventListener('click', () => {
        scrollSummary(-100);
    });
    
    document.getElementById('scroll-down').addEventListener('click', () => {
        scrollSummary(100);
    });
    
    // Function to scroll summary with animation
    function scrollSummary(amount) {
        summaryContainer.scrollBy({
            top: amount,
            behavior: 'smooth'
        });
        
        // Show scroll position indicator
        updateScrollIndicator();
    }
    
    // Update scroll indicator
    function updateScrollIndicator() {
        const scrollIndicator = document.getElementById('scroll-indicator');
        const maxScroll = summaryContainer.scrollHeight - summaryContainer.clientHeight;
        const currentScroll = summaryContainer.scrollTop;
        
        if (maxScroll > 0) {
            const scrollPercentage = Math.round((currentScroll / maxScroll) * 100);
            scrollIndicator.textContent = `Scroll position: ${scrollPercentage}%`;
            scrollIndicator.classList.remove('hidden');
        } else {
            scrollIndicator.classList.add('hidden');
        }
    }
    
    // Add keyboard navigation for summary
    document.addEventListener('keydown', (e) => {
        if (summaryContainer.classList.contains('open') && document.activeElement !== queryInput) {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                scrollSummary(-100);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                scrollSummary(100);
            } else if (e.key === 'Home') {
                e.preventDefault();
                summaryContainer.scrollTop = 0;
                updateScrollIndicator();
            } else if (e.key === 'End') {
                e.preventDefault();
                summaryContainer.scrollTop = summaryContainer.scrollHeight;
                updateScrollIndicator();
            }
        }
    });
    
    // Add scroll event listener to update indicator
    summaryContainer.addEventListener('scroll', updateScrollIndicator);
    
    // Document comparison event listeners
    compareButton.addEventListener('click', compareDocuments);
    
    documentSelector.addEventListener('change', function() {
        compareButton.disabled = !documentSelector.value;
    });
    
    comparisonTypeOptions.forEach(option => {
        option.addEventListener('click', function() {
            comparisonTypeOptions.forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Insights event listeners
    generateInsightsButton.addEventListener('click', generateDocumentInsights);
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Add event listeners for the new buttons
    uploadAnotherButton.addEventListener('click', function() {
        resetUI();
        uploadArea.classList.remove('hidden');
        uploadSuccess.classList.add('hidden');
        processingStatus.classList.add('hidden');
        
        processingCompleted = false;
    });
    
    tryAgainButton.addEventListener('click', function() {
        resetUI();
        uploadArea.classList.remove('hidden');
        uploadError.classList.add('hidden');
        
        processingCompleted = false;
    });
    
    // Helper function to reset UI for new uploads
    function resetUI() {
        if (websocket) {
            websocket.close();
            websocket = null;
        }
        
        sessionId = null;
        fileInput.value = '';
        progressBarFill.style.width = '0%';
        queryInput.disabled = true;
        sendButton.disabled = true;
    }
    
    // Function to handle file upload
    function handleFileUpload(file) {
        processingCompleted = false;
        
        const useAdvancedNlp = advancedNlpToggle.checked;
        
        uploadArea.classList.add('hidden');
        processingStatus.classList.remove('hidden');
        uploadSuccess.classList.add('hidden');
        uploadError.classList.add('hidden');
        statusMessage.textContent = "Uploading document...";
        progressBarFill.style.width = "10%";
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_advanced_nlp', useAdvancedNlp);
        
        fetch(`${API_BASE_URL}/upload/`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to upload file');
            }
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
            comparisonSection.classList.remove('hidden');
            progressBarFill.style.width = "20%";
            
            connectWebSocket();
            pollProcessingStatus();
            insightsSection.classList.remove('hidden');
        })
        .catch(error => {
            showError(error.message);
        });
    }
    
    // Function to update document selector
    function updateDocumentSelector() {
        while (documentSelector.options.length > 1) {
            documentSelector.remove(1);
        }
        
        uploadedDocuments.forEach(doc => {
            if (doc.id !== sessionId) {
                const option = document.createElement('option');
                option.value = doc.id;
                option.textContent = doc.name;
                documentSelector.appendChild(option);
            }
        });
        
        compareButton.disabled = documentSelector.options.length <= 1;
    }
    
    // Function to compare documents
    function compareDocuments() {
        const otherDocumentId = documentSelector.value;
        if (!otherDocumentId || !sessionId) return;
        
        const comparisonType = document.querySelector('.comparison-type-option.active').dataset.type;
        
        compareButton.disabled = true;
        compareButton.textContent = "Comparing...";
        
        websocket.send(JSON.stringify({
            type: "compare_documents",
            session_id1: sessionId,
            session_id2: otherDocumentId,
            comparison_type: comparisonType
        }));
    }
    
    // Function to display comparison results
    function displayComparisonResults(result) {
        const score = Math.round(result.similarity_score * 100);
        similarityScore.textContent = `${score}%`;
        
        if (score >= 80) {
            similarityScore.style.backgroundColor = "#10b981";
        } else if (score >= 50) {
            similarityScore.style.backgroundColor = "#f59e0b";
        } else {
            similarityScore.style.backgroundColor = "#ef4444";
        }
        
        commonTopics.innerHTML = '';
        result.common_topics.forEach(topic => {
            const topicElement = document.createElement('span');
            topicElement.className = 'topic-item';
            topicElement.textContent = topic;
            commonTopics.appendChild(topicElement);
        });
        
        uniqueTopics1.innerHTML = '';
        result.unique_topics1.forEach(topic => {
            const topicElement = document.createElement('span');
            topicElement.className = 'topic-item';
            topicElement.textContent = topic;
            uniqueTopics1.appendChild(topicElement);
        });
        
        uniqueTopics2.innerHTML = '';
        result.unique_topics2.forEach(topic => {
            const topicElement = document.createElement('span');
            topicElement.className = 'topic-item';
            topicElement.textContent = topic;
            uniqueTopics2.appendChild(topicElement);
        });
        
        comparisonResults.classList.remove('hidden');
        compareButton.disabled = false;
        compareButton.textContent = "Compare Documents";
    }
    
    // Load previously uploaded documents from localStorage
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
    
    loadUploadedDocuments();
    
    // Function to connect to WebSocket
    function connectWebSocket() {
        if (websocket) {
            websocket.close();
        }
        
        websocket = new WebSocket(`${API_BASE_URL.replace('http', 'ws')}/ws/${sessionId}`);
        
        websocket.onopen = function(e) {
            console.log("WebSocket connection established");
            startProcessing();
        };
        
        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === "status") {
                statusMessage.textContent = data.content;
                
                if (data.content.includes("Starting to process")) {
                    progressBarFill.style.width = "30%";
                } else if (data.content.includes("Processing page")) {
                    const match = data.content.match(/Processing page (\d+) of (\d+)/);
                    if (match) {
                        const current = parseInt(match[1]);
                        const total = parseInt(match[2]);
                        const progress = 30 + (current / total) * 30;
                        progressBarFill.style.width = `${progress}%`;
                    }
                } else if (data.content.includes("Text extraction complete")) {
                    progressBarFill.style.width = "60%";
                } else if (data.content.includes("Chunking complete")) {
                    progressBarFill.style.width = "70%";
                } else if (data.content.includes("Vector indexing complete")) {
                    progressBarFill.style.width = "80%";
                } else if (data.content.includes("Generating document summary")) {
                    progressBarFill.style.width = "90%";
                } else if (data.content.includes("Document processing complete")) {
                    progressBarFill.style.width = "100%";
                    processingStatus.classList.add('hidden');
                    uploadSuccess.classList.remove('hidden');
                    queryInput.disabled = false;
                    sendButton.disabled = false;
                    
                    if (!processingCompleted) {
                        addBotMessage('Document processed successfully! You can now ask questions about it.');
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
                    messagesContainer.querySelector('.flex').appendChild(typingIndicator);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
            } else if (data.type === "summary") {
                documentSummary.innerHTML = formatMessage(data.content);
                
                summaryContainer.classList.add('open');
                document.querySelector('.summary-controls').classList.remove('hidden');
                summaryChevron.innerHTML = `
                    <path fill-rule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 111.414 1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clip-rule="evenodd" />
                `;
                
                updateScrollIndicator();
            } else if (data.type === "start_stream") {
                const typingIndicator = messagesContainer.querySelector('.typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
                
                isStreaming = true;
                currentAnswer = "";
                
                const messageElement = document.createElement('div');
                messageElement.className = 'bot-message p-3 text-gray-700 self-start max-w-3xl streaming-message';
                messageElement.innerHTML = `<p></p>`;
                messagesContainer.querySelector('.flex').appendChild(messageElement);
            } else if (data.type === "token") {
                currentAnswer += data.content;
                
                const streamingMessage = messagesContainer.querySelector('.streaming-message p');
                if (streamingMessage) {
                    streamingMessage.innerHTML = formatMessage(currentAnswer);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
            } else if (data.type === "end_stream") {
                isStreaming = false;
                
                const streamingMessage = messagesContainer.querySelector('.streaming-message');
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
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                            </svg>
                        </button>
                    `;
                    streamingMessage.appendChild(feedbackDiv);
                    
                    const thumbsUp = feedbackDiv.querySelector('.thumbs-up');
                    const thumbsDown = feedbackDiv.querySelector('.thumbs-down');
                    
                    thumbsUp.addEventListener('click', function() {
                        sendFeedback('thumbs_up', currentQuery, currentAnswer);
                        thumbsUp.classList.add('active');
                        thumbsDown.classList.remove('active');
                    });
                    
                    thumbsDown.addEventListener('click', function() {
                        sendFeedback('thumbs_down', currentQuery, currentAnswer);
                        thumbsDown.classList.add('active');
                        thumbsUp.classList.remove('active');
                    });
                }
            } else if (data.type === "error") {
                showError(data.content);
            } else if (data.type === "feedback_received") {
                console.log("Feedback received:", data.content);
            } else if (data.type === "comparison_result") {
                displayComparisonResults(data.content);
            } else if (data.type === "visualization_result") {
                displayVisualizationResults(data.content);
            }
        };
        
        websocket.onclose = function(event) {
            if (!event.wasClean) {
                console.error(`WebSocket connection closed unexpectedly: ${event.code}`);
            }
        };
        
        websocket.onerror = function(error) {
            console.error(`WebSocket error: ${error.message}`);
            showError("WebSocket connection error. Please try refreshing the page.");
        };
    }
    
    // Function to poll for processing status
    function pollProcessingStatus() {
        if (processingCompleted) {
            return;
        }
        
        fetch(`${API_BASE_URL}/status/${sessionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to check status');
            }
            return response.json();
        })
        .then(data => {
            if (data.status === 'ready') {
                progressBarFill.style.width = "100%";
                processingStatus.classList.add('hidden');
                uploadSuccess.classList.remove('hidden');
                queryInput.disabled = false;
                sendButton.disabled = false;
                
                fetch(`${API_BASE_URL}/summary/${sessionId}`)
                .then(response => response.json())
                .then(summaryData => {
                    documentSummary.innerHTML = formatMessage(summaryData.summary);
                    
                    summaryContainer.classList.add('open');
                    document.querySelector('.summary-controls').classList.remove('hidden');
                    summaryChevron.innerHTML = `
                        <path fill-rule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 111.414 1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clip-rule="evenodd" />
                    `;
                    
                    updateScrollIndicator();
                })
                .catch(error => console.error('Error fetching summary:', error));
                
                if (!processingCompleted) {
                    addBotMessage('Document processed successfully! You can now ask questions about it.');
                    processingCompleted = true;
                }
            } else {
                statusMessage.textContent = data.message;
                setTimeout(pollProcessingStatus, 2000);
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            setTimeout(pollProcessingStatus, 3000);
        });
    }
    
    // Function to send a query via WebSocket
    function sendQuery() {
        const query = queryInput.value.trim();
        if (!query || !sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        
        currentQuery = query;
        addUserMessage(query);
        queryInput.value = '';
        
        websocket.send(JSON.stringify({
            type: "query",
            content: query
        }));
    }
    
    // Function to send feedback via WebSocket
    function sendFeedback(rating, query, answer) {
        if (!sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        
        websocket.send(JSON.stringify({
            type: "feedback",
            rating: rating,
            query: query,
            answer: answer
        }));
    }
    
    // Function to add a user message to the chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'user-message p-3 text-gray-700 self-end max-w-3xl';
        messageElement.innerHTML = `<p>${escapeHtml(message)}</p>`;
        
        messagesContainer.querySelector('.flex').appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Function to add a bot message to the chat
    function addBotMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'bot-message p-3 text-gray-700 self-start max-w-3xl';
        messageElement.innerHTML = `<p>${formatMessage(message)}</p>`;
        
        messagesContainer.querySelector('.flex').appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Function to format message with markdown-like syntax
    function formatMessage(message) {
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }
    
    // Function to show error
    function showError(message) {
        processingStatus.classList.add('hidden');
        uploadError.classList.remove('hidden');
        uploadArea.classList.remove('hidden');
        errorMessage.textContent = message;
    }
    
    // Function to escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Function to generate document insights
    function generateDocumentInsights() {
        if (!sessionId) return;
        
        generateInsightsButton.disabled = true;
        generateInsightsButton.textContent = "Generating insights...";
        
        fetch(`${API_BASE_URL}/document_stats/${sessionId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to generate document insights');
                }
                return response.json();
            })
            .then(data => {
                documentStats = data;
                displayDocumentInsights(data);
                insightsContainer.classList.remove('hidden');
                generateInsightsButton.disabled = false;
                generateInsightsButton.textContent = "Refresh Document Insights";
            })
            .catch(error => {
                console.error('Error generating insights:', error);
                generateInsightsButton.disabled = false;
                generateInsightsButton.textContent = "Generate Document Insights";
                alert('Error generating document insights. Please try again.');
            });
    }
    
    // Function to display document insights
    function displayDocumentInsights(data) {
        document.getElementById('word-count').textContent = data.basic_stats.word_count.toLocaleString();
        document.getElementById('sentence-count').textContent = data.basic_stats.sentence_count.toLocaleString();
        document.getElementById('paragraph-count').textContent = data.basic_stats.paragraph_count.toLocaleString();
        document.getElementById('readability-score').textContent = data.basic_stats.readability_score;
        const readabilityLevel = document.getElementById('readability-level');
        readabilityLevel.textContent = data.basic_stats.readability_level;
        
        const readabilityFill = document.getElementById('readability-fill');
        const readabilityScore = data.basic_stats.readability_score;
        readabilityFill.style.width = `${readabilityScore}%`;
        
        if (readabilityScore >= 70) {
            readabilityFill.style.backgroundColor = '#10b981';
        } else if (readabilityScore >= 50) {
            readabilityFill.style.backgroundColor = '#f59e0b';
        } else {
            readabilityFill.style.backgroundColor = '#ef4444';
        }
        
        renderEntities(data.top_entities);
        
        document.getElementById('people-count').textContent = data.top_entities.people.length;
        document.getElementById('org-count').textContent = data.top_entities.organizations.length;
        document.getElementById('location-count').textContent = data.top_entities.locations.length;
        document.getElementById('date-count').textContent = data.top_entities.dates.length;
        
        renderKeyPhrases(data.top_phrases);
        renderWordCloud(data.top_words);
    }
    
    // Function to render entities
    function renderEntities(entities) {
        const peopleContainer = document.getElementById('people-entities');
        peopleContainer.innerHTML = '';
        
        if (entities.people.length === 0) {
            peopleContainer.innerHTML = '<p class="text-sm text-gray-500">No people detected in this document.</p>';
        } else {
            entities.people.forEach(person => {
                const entityElement = document.createElement('span');
                entityElement.className = 'entity-tag person';
                entityElement.textContent = person;
                peopleContainer.appendChild(entityElement);
            });
        }
        
        const organizationContainer = document.getElementById('organization-entities');
        organizationContainer.innerHTML = '';
        
        if (entities.organizations.length === 0) {
            organizationContainer.innerHTML = '<p class="text-sm text-gray-500">No organizations detected in this document.</p>';
        } else {
            entities.organizations.forEach(org => {
                const entityElement = document.createElement('span');
                entityElement.className = 'entity-tag organization';
                entityElement.textContent = org;
                organizationContainer.appendChild(entityElement);
            });
        }
        
        const locationContainer = document.getElementById('location-entities');
        locationContainer.innerHTML = '';
        
        if (entities.locations.length === 0) {
            locationContainer.innerHTML = '<p class="text-sm text-gray-500">No locations detected in this document.</p>';
        } else {
            entities.locations.forEach(location => {
                const entityElement = document.createElement('span');
                entityElement.className = 'entity-tag location';
                entityElement.textContent = location;
                locationContainer.appendChild(entityElement);
            });
        }
        
        const dateContainer = document.getElementById('date-entities');
        dateContainer.innerHTML = '';
        
        if (entities.dates.length === 0) {
            dateContainer.innerHTML = '<p class="text-sm text-gray-500">No dates detected in this document.</p>';
        } else {
            entities.dates.forEach(date => {
                const entityElement = document.createElement('span');
                entityElement.className = 'entity-tag date';
                entityElement.textContent = date;
                dateContainer.appendChild(entityElement);
            });
        }
    }
    
    // Function to render key phrases
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
    
    // Function to render word cloud
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
    
    // Function to toggle summary visibility
    function toggleSummary() {
        const isHidden = summaryContainer.classList.contains('hidden');
        
        if (isHidden) {
            summaryContainer.classList.remove('hidden');
            summaryToggle.textContent = "Hide Summary";
            document.getElementById('summary-controls').classList.remove('hidden');
        } else {
            summaryContainer.classList.add('hidden');
            summaryToggle.textContent = "Show Summary";
            document.getElementById('summary-controls').classList.add('hidden');
        }
    }
    
    // Function to start processing
    function startProcessing() {
        if (!sessionId || !websocket || websocket.readyState !== WebSocket.OPEN) return;
        
        processingCompleted = false;
        
        const useAdvancedNlp = advancedNlpToggle.checked;
        
        websocket.send(JSON.stringify({
            type: "start_processing",
            use_advanced_nlp: useAdvancedNlp
        }));
    }
});