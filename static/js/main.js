document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const sidebar = document.getElementById('sidebar');
    const toggleSidebar = document.getElementById('toggleSidebar');
    const closeSidebar = document.getElementById('closeSidebar');
    const sessionHistory = document.getElementById('sessionHistory');
    const fileUpload = document.getElementById('fileUpload');
    const uploadStatus = document.getElementById('uploadStatus');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatContainer = document.getElementById('chatContainer');
    const fileList = document.getElementById('fileList');
    const refreshFiles = document.getElementById('refreshFiles');
    const sessionTitleHeader = document.getElementById('session-title');
    const toggleTheme = document.getElementById('toggleTheme');

    let isUploading = false;
    let currentActiveFile = null;

    // session_id for Firebase tracking
    let sessionId = localStorage.getItem('rag_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('rag_session_id', sessionId);
    }

    // --- Initialization ---
    fetchFiles();
    fetchSessions();

    const newChatBtn = document.getElementById('newChatBtn');

    // --- Sidebar Toggle ---
    toggleSidebar.onclick = () => {
        sidebar.classList.remove('hidden');
    };

    closeSidebar.onclick = () => {
        sidebar.classList.add('hidden');
    };

    newChatBtn.onclick = () => {
        const newId = 'session_' + Math.random().toString(36).substr(2, 9);
        sessionId = newId;
        localStorage.setItem('rag_session_id', newId);
        chatContainer.innerHTML = '';
        sessionTitleHeader.innerText = 'New Intelligence Session';
        currentActiveFile = null;
        document.querySelectorAll('.file-item').forEach(i => i.classList.remove('active'));

        // Add welcome hero back
        const welcomeHero = `
            <div class="welcome-hero">
                <div class="hero-content">
                    <div class="hero-icon">
                        <i class="fas fa-wand-magic-sparkles"></i>
                    </div>
                    <h2>What can I help you extract today?</h2>
                    <p>Upload your research papers, financial statements, or datasets to unlock deep insights with AI-powered retrieval.</p>

                    <div class="feature-grid">
                        <div class="feature-card">
                            <i class="fas fa-chart-pie"></i>
                            <span>Dynamic Charts</span>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-table"></i>
                            <span>Table Analysis</span>
                        </div>
                        <div class="feature-card">
                            <i class="fas fa-quote-left"></i>
                            <span>Smart Citations</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        chatContainer.innerHTML = welcomeHero;

        // Refresh session list to show active state
        fetchSessions();
    };

    // --- Session History ---
    async function fetchSessions() {
        try {
            const response = await fetch('/sessions');
            const data = await response.json();
            sessionHistory.innerHTML = '';

            if (data.sessions && data.sessions.length > 0) {
                // Show latest first
                data.sessions.reverse().forEach(id => {
                    const item = document.createElement('div');
                    item.className = 'session-item' + (id === sessionId ? ' active' : '');
                    item.innerHTML = `
                        <div class="session-main" onclick="loadSession('${id}')">
                            <i class="fas fa-history"></i>
                            <span>${id}</span>
                        </div>
                        <button class="delete-session-btn" onclick="event.stopPropagation(); deleteSession('${id}')">
                            <i class="fas fa-trash-can"></i>
                        </button>
                    `;
                    sessionHistory.appendChild(item);
                });
            } else {
                sessionHistory.innerHTML = '<div class="empty-state">No history</div>';
            }
        } catch (error) {
            console.error('Error fetching sessions:', error);
        }
    }

    window.deleteSession = async (id) => {
        if (!confirm('Are you sure you want to delete this session?')) return;

        try {
            const response = await fetch(`/delete_session/${id}`, { method: 'DELETE' });
            if (response.ok) {
                if (id === sessionId) {
                    newChatBtn.click(); // Reset if active deleted
                }
                fetchSessions();
            }
        } catch (error) {
            console.error('Delete error:', error);
        }
    };

    function checkFileLoaded() {
        if (!currentActiveFile) {
            document.getElementById('selectionModal').classList.remove('hidden');
            return false;
        }
        return true;
    }

    async function loadSession(id) {
        if (id === sessionId && chatContainer.querySelectorAll('.message').length > 0) return;

        sessionId = id;
        localStorage.setItem('rag_session_id', id);

        // Update active state in UI
        document.querySelectorAll('.session-item').forEach(i => {
            i.classList.toggle('active', i.querySelector('span').innerText === id);
        });

        // Clear and show loading
        chatContainer.innerHTML = '<div class="thinking-indicator"><i class="fas fa-spinner fa-spin"></i> <span>Loading history...</span></div>';

        try {
            const response = await fetch(`/session/${id}`);
            const data = await response.json();
            chatContainer.innerHTML = '';

            if (data.history) {
                data.history.forEach(msg => {
                    if (msg.role === 'user') {
                        addUserMessage(msg.content);
                    } else {
                        const botMsgDiv = createBotMessageShell();
                        const responseTextContainer = document.createElement('div');
                        responseTextContainer.className = 'response-text';
                        responseTextContainer.innerHTML = formatMarkdown(msg.content);
                        botMsgDiv.querySelector('.message-content').innerHTML = '';
                        botMsgDiv.querySelector('.message-content').appendChild(responseTextContainer);
                    }
                });
                scrollToBottom();
            }
        } catch (error) {
            console.error('Error loading session:', error);
            chatContainer.innerHTML = '<div class="error-state">Failed to load session history</div>';
        }
    }

    // --- File Management ---
    refreshFiles.onclick = (e) => {
        e.stopPropagation();
        fetchFiles();
    };

    async function fetchFiles() {
        try {
            // Show loading shimmer
            fileList.innerHTML = `
                <div class="shimmer"></div>
                <div class="shimmer"></div>
                <div class="shimmer"></div>
            `;

            const response = await fetch('/files');
            const data = await response.json();
            fileList.innerHTML = '';

            if (data.files && data.files.length > 0) {
                data.files.forEach(file => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.innerHTML = `
                        <i class="fas fa-file-lines"></i>
                        <div class="file-info">
                            <span class="file-name" title="${file}">${file}</span>
                        </div>
                        <div class="file-actions">
                            <a href="/uploads/${file}" target="_blank" class="action-btn-small" title="Open File" onclick="event.stopPropagation();">
                                <i class="fas fa-external-link-alt"></i>
                            </a>
                            <button class="delete-file-btn" onclick="event.stopPropagation(); deleteFile('${file}')" title="Delete File">
                                <i class="fas fa-trash-can"></i>
                            </button>
                        </div>
                    `;
                    fileItem.onclick = (e) => {
                        // Mark as active
                        document.querySelectorAll('.file-item').forEach(i => i.classList.remove('active'));
                        fileItem.classList.add('active');
                        selectExistingFile(file);
                    };
                    fileList.appendChild(fileItem);
                });
            } else {
                fileList.innerHTML = '<div class="empty-state">No documents found</div>';
            }
        } catch (error) {
            console.error('Error fetching files:', error);
            fileList.innerHTML = '<div class="error-state">Failed to load files</div>';
        }
    }

    async function selectExistingFile(filename) {
        if (isUploading) return;
        isUploading = true;

        showUploadStatus(`Analyzing ${filename}...`);

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000); // 1 minute timeout

            const response = await fetch('/process_existing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename }),
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            const data = await response.json();

            if (data.message) {
                hideUploadStatus();
                currentActiveFile = filename;
                sessionTitleHeader.innerText = filename;
                addBotMessage(`### Context Updated\nI have successfully loaded and analyzed **${filename}**. What would you like to know about it?`);
            } else {
                hideUploadStatus();
                addBotMessage('### Note\nI was unable to process that document. It might be corrupt or in an unsupported format.');
            }
        } catch (error) {
            console.error('Analysis Error:', error);
            hideUploadStatus(); // Keep UI clean as requested
            addBotMessage(`### Note\nI encountered an issue connecting to the engine. Please try again in a moment.`);
        } finally {
            isUploading = false;
        }
    }

    // --- File Upload ---
    const attachBtn = document.getElementById('attachBtn');
    if (attachBtn) {
        attachBtn.addEventListener('click', () => fileUpload.click());
    }

    fileUpload.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files.length) return;

        isUploading = true;
        const fileCount = files.length;
        showUploadStatus(`⚡ Starting upload of ${fileCount} file${fileCount > 1 ? 's' : ''}...`);

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('file', files[i]);
        }

        fileUpload.value = ''; // Reset

        // Show initial message in chat
        const uploadMsgId = 'upload-' + Date.now();
        addBotMessage(`📂 **Processing ${fileCount} document${fileCount > 1 ? 's' : ''}...**\nPreparing to extract and index...`);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') break;

                        try {
                            const data = JSON.parse(dataStr);

                            if (data.status === 'error') {
                                addBotMessage(`### ⚠️ Upload Issue\n${data.msg}`);
                                hideUploadStatus();
                            } else if (data.status === 'complete') {
                                hideUploadStatus();
                                currentActiveFile = "Uploaded Files";
                                fetchFiles(); // Refresh list
                                addBotMessage(`### ✅ Indexing Complete!\n${data.message}\n\nYou can now query across all uploaded documents.`);
                            } else if (data.msg) {
                                // Update status pill
                                showUploadStatus(data.msg);
                            }
                        } catch (e) { console.error('Parse Error', e); }
                    }
                }
            }

        } catch (error) {
            console.error('Upload Error:', error);
            hideUploadStatus();
            addBotMessage(`### ⚠️ Upload Issue\nConnection error: ${error.message}\n\nPlease check your connection and try again.`);
        } finally {
            isUploading = false;
            setTimeout(hideUploadStatus, 2000);
        }
    });

    window.deleteFile = async (filename) => {
        if (!confirm(`Are you sure you want to remove ${filename} from the knowledge base?`)) return;

        showUploadStatus(`Removing ${filename}...`);
        try {
            const response = await fetch('/delete_file', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename })
            });
            const data = await response.json();
            if (data.message) {
                hideUploadStatus();
                if (currentActiveFile === filename) currentActiveFile = null;
                fetchFiles();
                addBotMessage(`🗑️ **Document Removed**\nI have removed **${filename}** and updated the knowledge base.`);
            }
        } catch (error) {
            hideUploadStatus();
        }
    };

    // --- Chat Logic ---
    async function sendMessage() {
        if (!checkFileLoaded()) return;
        const query = userInput.value.trim();
        if (!query) return;

        // Reset UI
        userInput.value = '';
        userInput.style.height = 'auto';

        // Update Viewport
        addUserMessage(query);
        const botMsgDiv = createBotMessageShell();
        const responseTextContainer = document.createElement('div');
        responseTextContainer.className = 'response-text';
        botMsgDiv.querySelector('.message-content').appendChild(responseTextContainer);

        scrollToBottom();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId })
            });

            if (!response.ok) throw new Error('AI engine unreachable');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let isThinking = true;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        if (dataStr === '[DONE]') break;

                        try {
                            const data = JSON.parse(dataStr);

                            // Handle Progress Steps
                            if (data.step) {
                                let stepContainer = botMsgDiv.querySelector('.processing-steps');
                                if (!stepContainer) {
                                    stepContainer = document.createElement('div');
                                    stepContainer.className = 'processing-steps';
                                    botMsgDiv.querySelector('.message-content').prepend(stepContainer);
                                }

                                const stepId = `step-${data.step}`;
                                let existingStep = stepContainer.querySelector(`#${stepId}`);

                                if (data.step === 'done') {
                                    stepContainer.querySelectorAll('.step-item').forEach(s => {
                                        s.classList.remove('active');
                                        s.classList.add('done');
                                        s.querySelector('i').className = 'fas fa-check-circle';
                                    });
                                } else {
                                    stepContainer.querySelectorAll('.step-item').forEach(s => {
                                        s.classList.remove('active');
                                        s.classList.add('done');
                                        s.querySelector('i').className = 'fas fa-check-circle';
                                    });

                                    if (!existingStep) {
                                        const newStep = document.createElement('div');
                                        newStep.className = 'step-item active';
                                        newStep.id = stepId;
                                        newStep.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <span>${data.msg}</span>`;
                                        stepContainer.appendChild(newStep);

                                        // Chart Placeholder
                                        if (data.step === 'chart' && !botMsgDiv.querySelector('.chart-placeholder')) {
                                            const ph = document.createElement('div');
                                            ph.className = 'chart-placeholder';
                                            ph.innerHTML = `<i class="fas fa-chart-line fa-pulse"></i><span>Generating Visualization...</span>`;
                                            botMsgDiv.querySelector('.message-content').appendChild(ph);
                                        }
                                    }
                                }
                            }

                            // Handle Image arrival
                            if (data.image) {
                                const img = document.createElement('img');
                                img.src = 'data:image/png;base64,' + data.image;
                                img.className = 'generated-chart';
                                img.onload = scrollToBottom;

                                const ph = botMsgDiv.querySelector('.chart-placeholder');
                                if (ph) ph.replaceWith(img);
                                else botMsgDiv.querySelector('.message-content').appendChild(img);
                            }

                            // Handle Text arrival
                            if (data.text) {
                                if (isThinking) {
                                    isThinking = false;
                                    botMsgDiv.querySelector('.thinking-indicator').remove();
                                }
                                fullText += data.text;
                                responseTextContainer.innerHTML = formatMarkdown(fullText);
                                scrollToBottom();
                            }

                        } catch (e) { console.error('Parse Error', e); }
                    }
                }
            }
            // Update session list to reflect new history
            fetchSessions();
        } catch (error) {
            console.error('Chat Error:', error);
            botMsgDiv.innerHTML = `<div class="message-content error">### Connection Lost\n${error.message}</div>`;
        }
    }

    // --- Tools & Analysis (Right Panel) ---
    const intelPanel = document.getElementById('intelPanel');
    const intelTitle = document.getElementById('intelTitle');
    const intelBody = document.getElementById('intelBody');
    const intelFooter = document.getElementById('intelFooter');
    const closeIntel = document.getElementById('closeIntel');

    let currentQuizData = null;
    let quizState = {
        currentIndex: 0,
        answers: [], // {selected, isCorrect}
        completed: false
    };

    closeIntel.onclick = () => intelPanel.classList.add('hidden');

    function showIntel(title, contentHTML, footerHTML = '') {
        intelTitle.innerText = title;
        intelBody.innerHTML = contentHTML;
        intelFooter.innerHTML = footerHTML;
        intelPanel.classList.remove('hidden');
        intelBody.scrollTop = 0;
    }

    let currentQuizTitle = "Quiz";

    function renderQuizQuestion() {
        if (!currentQuizData) return;
        const q = currentQuizData[quizState.currentIndex];
        const state = quizState.answers[quizState.currentIndex];

        let navHTML = '<div class="quiz-nav">';
        currentQuizData.forEach((_, i) => {
            const statusClass = quizState.answers[i] ? 'answered' : '';
            const activeClass = i === quizState.currentIndex ? 'active' : '';
            navHTML += `<div class="nav-dot ${statusClass} ${activeClass}" onclick="jumpToQuestion(${i})">${i + 1}</div>`;
        });
        navHTML += '</div>';

        let optionsHTML = '<div class="quiz-options">';
        q.options.forEach(opt => {
            let style = '';
            if (state) {
                if (opt === q.answer) style = 'background: rgba(16, 185, 129, 0.2); border-color: #10b981;';
                else if (opt === state.selected && !state.isCorrect) style = 'background: rgba(244, 63, 94, 0.2); border-color: #f43f5e;';
            }
            optionsHTML += `
                <div class="quiz-option" style="${style}" onclick="handleQuizSelection('${escapeHTML(opt)}')">
                    ${escapeHTML(opt)}
                </div>`;
        });
        optionsHTML += '</div>';

        let feedbackHTML = '';
        if (state) {
            feedbackHTML = state.isCorrect
                ? '<div class="quiz-feedback correct"><i class="fas fa-check-circle"></i> Correct! Well done.</div>'
                : `<div class="quiz-feedback incorrect"><i class="fas fa-times-circle"></i> Incorrect. The correct answer is <strong>${escapeHTML(q.answer)}</strong></div>`;
        }

        const bodyHTML = `
            <div class="quiz-ui-container">
                <div class="quiz-header-info" style="text-align:center; margin-bottom:15px; border-bottom:1px solid var(--border); padding-bottom:10px;">
                    <h4 style="color:var(--primary-light);">${escapeHTML(currentQuizTitle)}</h4>
                </div>
                ${navHTML}
                <div class="quiz-question-container">
                    <div class="quiz-question">${quizState.currentIndex + 1}. ${escapeHTML(q.question)}</div>
                    ${optionsHTML}
                    ${feedbackHTML}
                </div>
            </div>
        `;

        let footerHTML = '';
        if (quizState.currentIndex < currentQuizData.length - 1) {
            footerHTML = `<button class="btn-primary" onclick="jumpToQuestion(${quizState.currentIndex + 1})">Next Question <i class="fas fa-arrow-right"></i></button>`;
        } else if (quizState.answers.filter(a => a).length === currentQuizData.length) {
            footerHTML = `<button class="btn-primary" onclick="finishQuiz()">Finish & View Score</button>`;
        }

        intelBody.innerHTML = bodyHTML;
        intelTitle.innerText = "Active Quiz";
        intelFooter.innerHTML = footerHTML;
    }

    window.handleQuizSelection = (selected) => {
        if (quizState.answers[quizState.currentIndex]) return; // Already answered

        const correct = currentQuizData[quizState.currentIndex].answer;
        quizState.answers[quizState.currentIndex] = {
            selected,
            isCorrect: selected === correct
        };
        renderQuizQuestion();

        // Auto Next Logic
        setTimeout(() => {
            if (quizState.currentIndex < currentQuizData.length - 1) {
                jumpToQuestion(quizState.currentIndex + 1);
            } else {
                // If last question, ensure the footer reflects the finish button
                renderQuizQuestion();
            }
        }, 1500);
    };

    window.jumpToQuestion = (index) => {
        quizState.currentIndex = index;
        renderQuizQuestion();
    };

    window.finishQuiz = () => {
        const score = quizState.answers.filter(a => a.isCorrect).length;
        const total = currentQuizData.length;
        const percent = Math.round((score / total) * 100);

        const html = `
            <div class="quiz-score-container">
                <div class="score-circle">${percent}%</div>
                <h3>Quiz Completed!</h3>
                <p>You scored **${score} out of ${total}**</p>
                <div class="score-summary" style="margin-top: 20px; text-align: left;">
                    ${quizState.answers.map((a, i) => `
                        <div style="margin-bottom: 8px; font-size: 0.9rem; color: ${a.isCorrect ? 'var(--secondary)' : '#f43f5e'}">
                            <i class="fas ${a.isCorrect ? 'fa-check' : 'fa-times'}"></i> Question ${i + 1}: ${a.isCorrect ? 'Correct' : 'Incorrect'}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        const footer = `
            <button class="btn-secondary" onclick="retakeQuiz()">Retake Quiz</button>
            <button class="btn-primary" onclick="generateNewQuiz()">Generate New Quiz</button>
        `;

        showIntel('Quiz Results', html, footer);
    };

    window.retakeQuiz = () => {
        quizState = { currentIndex: 0, answers: [], completed: false };
        renderQuizQuestion();
    };

    window.fetchAndShowQuizzes = async () => {
        showIntel('Available Quizzes', '<div class="shimmer"></div>');
        try {
            const response = await fetch('/get_all_quizzes');
            const data = await response.json();

            let html = `
                 <div class="quiz-list-container">
                    <button class="btn-primary full-width" onclick="generateNewQuiz()">
                        <i class="fas fa-plus"></i> Create New Quiz
                    </button>
                    <div class="quiz-list-items">
             `;

            const icons = ['fa-flask', 'fa-atom', 'fa-dna', 'fa-globe-americas', 'fa-calculator', 'fa-book-open', 'fa-code', 'fa-chart-pie', 'fa-brain', 'fa-microscope'];

            if (data.quizzes && data.quizzes.length > 0) {
                data.quizzes.forEach((quiz, index) => {
                    const id = quiz.id;
                    const title = quiz.title;
                    const hash = id.split('').reduce((a, b) => a + b.charCodeAt(0), 0);
                    const icon = icons[hash % icons.length];

                    html += `
                        <div class="result-card" onclick="loadQuizById('${id}')">
                            <i class="fas ${icon}"></i>
                            <div class="card-content">
                                <strong>${escapeHTML(title)}</strong>
                                <span class="card-subtitle">ID: ${id.substring(0, 8)}...</span>
                            </div>
                            <i class="fas fa-chevron-right action-icon"></i>
                        </div>
                     `;
                });
            } else {
                html += `<div class="empty-state">No saved quizzes found.</div>`;
            }

            html += `</div></div>`;
            showIntel('Quiz Library', html);
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    window.generateNewQuiz = async () => {
        if (!checkFileLoaded()) return;
        showIntel('Generating Quiz...', '<div class="shimmer"></div><div class="shimmer"></div>');
        try {
            const response = await fetch('/generate_quiz', { method: 'POST' });
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Backend returns { quiz_id, quiz: { title, questions } }
            currentQuizData = data.quiz.questions;
            currentQuizTitle = data.quiz.title;

            quizState = { currentIndex: 0, answers: [], completed: false };
            renderQuizQuestion();
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    window.loadQuizById = async (id) => {
        showIntel('Loading Quiz...', '<div class="shimmer"></div>');
        try {
            const response = await fetch(`/get_quiz/${id}`);
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Backend returns { quiz: { title, questions } }
            currentQuizData = data.quiz.questions;
            currentQuizTitle = data.quiz.title;

            quizState = { currentIndex: 0, answers: [], completed: false };
            renderQuizQuestion();
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    document.getElementById('genQuiz').onclick = window.fetchAndShowQuizzes;

    document.getElementById('genFlashcards').onclick = async () => {
        if (!checkFileLoaded()) return;
        showIntel('Study Flashcards', '<div class="shimmer"></div><div class="shimmer"></div>');
        try {
            const response = await fetch('/generate_flashcards', { method: 'POST' });
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            renderFlashcardStack(data);
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    window.renderFlashcardStack = (cards) => {
        const stackHTML = `
            <div class="flashcard-ui-container">
                <div class="flashcards-stack" id="flashStack">
                    <div class="swipe-indicator like">KEEP</div>
                    <div class="swipe-indicator nope">SKIP</div>
                    ${[...cards].reverse().map((card, i) => {
            const randomRot = (Math.random() - 0.5) * 10; // Random rotation between -5 and 5 deg
            const depth = cards.length - 1 - i;
            const scale = 1 - depth * 0.05;
            const yOffset = depth * -12;
            return `
                        <div class="flashcard-stack-item flashcard" 
                             data-term="${escapeHTML(card.term)}" 
                             data-definition="${escapeHTML(card.definition)}"
                             data-base-rot="${randomRot}"
                             style="z-index: ${i}; transform: scale(${scale}) translateY(${yOffset}px) rotate(${randomRot}deg); pointer-events: auto; opacity: 1;">
                            <div class="flashcard-inner" onclick="this.parentElement.classList.toggle('flipped')">
                                <div class="flashcard-front"><span>${escapeHTML(card.term)}</span></div>
                                <div class="flashcard-back"><span>${escapeHTML(card.definition)}</span></div>
                            </div>
                        </div>
                    `;
        }).join('')}
                    <div class="flashcard-empty-state hidden">
                        <i class="fas fa-check-double" style="font-size: 3rem; margin-bottom: 15px; color: var(--secondary);"></i>
                        <h3>All caught up!</h3>
                        <p>You've reviewed all flashcards in this set.</p>
                    </div>
                </div>
            </div>
        `;

        const footerHTML = `
            <button class="btn-secondary" onclick="viewSavedFlashcards()"><i class="fas fa-bookmark"></i> View Saved</button>
            <button class="btn-primary" onclick="document.getElementById('genFlashcards').click()"><i class="fas fa-sync"></i> Refresh</button>
        `;

        showIntel('Interactive Flashcards', stackHTML, footerHTML);
        initStackDragging();
    };

    function initStackDragging() {
        const stack = document.getElementById('flashStack');
        const cards = stack.querySelectorAll('.flashcard-stack-item');
        if (cards.length === 0) return;

        // Start with the topmost card
        const topCard = cards[cards.length - 1];
        if (!topCard) return;

        // Enable interaction for the top card
        topCard.style.pointerEvents = 'auto';
        topCard.style.opacity = '1';

        let startX, startY, moveX, moveY;
        let isDragging = false;
        const baseRot = parseFloat(topCard.getAttribute('data-base-rot') || 0);

        topCard.onmousedown = (e) => {
            if (topCard.classList.contains('flipped')) return;
            e.preventDefault();
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            topCard.style.transition = 'none';
        };

        window.onmousemove = (e) => {
            if (!isDragging) return;
            moveX = e.clientX - startX;
            moveY = e.clientY - startY;
            const rotate = baseRot + (moveX / 10);
            topCard.style.transform = `translate(${moveX}px, ${moveY}px) rotate(${rotate}deg)`;

            const like = stack.querySelector('.swipe-indicator.like');
            const nope = stack.querySelector('.swipe-indicator.nope');
            if (moveX > 50) {
                like.style.opacity = Math.min(moveX / 150, 1);
                nope.style.opacity = 0;
            } else if (moveX < -50) {
                nope.style.opacity = Math.min(Math.abs(moveX) / 150, 1);
                like.style.opacity = 0;
            } else {
                like.style.opacity = 0;
                nope.style.opacity = 0;
            }
        };

        window.onmouseup = () => {
            if (!isDragging) return;
            isDragging = false;
            topCard.style.transition = '';
            stack.querySelector('.swipe-indicator.like').style.opacity = 0;
            stack.querySelector('.swipe-indicator.nope').style.opacity = 0;

            if (Math.abs(moveX) > 150) {
                if (moveX > 0) {
                    topCard.classList.add('swiped-right');
                    saveFlashcard({
                        term: topCard.getAttribute('data-term'),
                        definition: topCard.getAttribute('data-definition')
                    });
                } else {
                    topCard.classList.add('swiped-left');
                }
                setTimeout(() => {
                    topCard.remove();
                    // Make next card interactive
                    initStackDragging();
                    if (stack.querySelectorAll('.flashcard-stack-item').length === 0) {
                        stack.querySelector('.flashcard-empty-state').classList.remove('hidden');
                    }
                }, 300);
            } else {
                topCard.style.transform = `scale(1) translateY(0) rotate(${baseRot}deg)`;
            }
            moveX = 0; moveY = 0;
        };
    }

    async function saveFlashcard(card) {
        try {
            await fetch('/save_flashcard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ card, session_id: sessionId })
            });
        } catch (e) {
            console.error('Save error:', e);
        }
    }

    window.viewSavedFlashcards = async () => {
        showIntel('Saved Flashcards', '<div class="shimmer"></div>');
        try {
            const response = await fetch(`/get_saved_flashcards?session_id=${sessionId}`);
            const data = await response.json();

            if (!data.flashcards || data.flashcards.length === 0) {
                showIntel('Saved Flashcards', '<div class="empty-state">No saved cards found</div>', '<button class="btn-primary" onclick="document.getElementById(\'genFlashcards\').click()">Generate Some</button>');
                return;
            }

            let html = '<div class="flashcards-list" style="display: flex; flex-direction: column; gap: 12px;">';
            data.flashcards.forEach(card => {
                html += `
                    <div class="flashcard saved-item" style="height: auto; min-height: 100px;" onclick="this.classList.toggle('flipped')">
                        <div class="flashcard-inner">
                            <div class="flashcard-front"><span>${escapeHTML(card.term)}</span></div>
                            <div class="flashcard-back"><span>${escapeHTML(card.definition)}</span></div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            showIntel('Saved Flashcards', html, '<button class="btn-primary" onclick="document.getElementById(\'genFlashcards\').click()">Back to Stack</button>');
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    document.getElementById('deepAnalysis').onclick = async () => {
        if (!checkFileLoaded()) return;
        showIntel('Performing Deep Analysis...', '<div class="shimmer"></div><div class="shimmer"></div>');
        try {
            const response = await fetch('/deep_analysis', { method: 'POST' });
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            showIntel('Deep Intelligence Analysis', `<div class="analysis-content">${formatMarkdown(data.analysis)}</div>`);
        } catch (e) {
            showIntel('Error', `<p class="error-text">${e.message}</p>`);
        }
    };

    // --- Helpers ---

    function showUploadStatus(msg, isError = false) {
        uploadStatus.classList.remove('hidden');
        uploadStatus.querySelector('span').innerText = msg;
        if (isError) {
            uploadStatus.style.background = 'rgba(244, 63, 94, 0.1)';
            uploadStatus.style.color = '#f43f5e';
            uploadStatus.querySelector('i').className = 'fas fa-exclamation-circle';
        } else {
            uploadStatus.style.background = 'rgba(16, 185, 129, 0.1)';
            uploadStatus.style.color = '#10b981';
            uploadStatus.querySelector('i').className = 'fas fa-circle-notch fa-spin';
        }
    }

    function hideUploadStatus() {
        uploadStatus.classList.add('hidden');
    }

    function addUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message user-message';
        msgDiv.innerHTML = `<div class="message-content">${escapeHTML(text)}</div>`;
        chatContainer.appendChild(msgDiv);

        const hero = document.querySelector('.welcome-hero');
        if (hero) hero.remove();
    }

    function createBotMessageShell() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        msgDiv.innerHTML = `
            <div class="message-content">
                <div class="thinking-indicator">
                    <i class="fas fa-brain fa-pulse"></i>
                    <span>Connecting to Intelligence Engine...</span>
                </div>
            </div>
        `;
        chatContainer.appendChild(msgDiv);
        return msgDiv;
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, m => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
        }[m]));
    }

    function formatMarkdown(text) {
        let formatted = escapeHTML(text);

        // Split lines for processing
        const lines = formatted.split('\n');
        let processed = [];
        let inTable = false;
        let tableHTML = '';

        lines.forEach(line => {
            const trimmed = line.trim();

            // Table Logic
            if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
                if (!inTable) {
                    inTable = true;
                    tableHTML = '<div class="table-container"><table>';
                }
                if (trimmed.includes('---')) return;

                const cells = trimmed.split('|').map(c => c.trim()).filter((c, i, a) => i > 0 && i < a.length - 1);
                tableHTML += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
            } else {
                if (inTable) {
                    tableHTML += '</table></div>';
                    processed.push(tableHTML);
                    inTable = false;
                }

                // Bold
                let l = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong class="highlight">$1</strong>');

                // Headers
                if (l.startsWith('### ')) processed.push(`<h3>${l.substring(4)}</h3>`);
                else if (l.startsWith('## ')) processed.push(`<h2>${l.substring(3)}</h2>`);
                else if (l.startsWith('# ')) processed.push(`<h1>${l.substring(2)}</h1>`);
                // Lists
                else if (l.startsWith('* ') || l.startsWith('- ')) processed.push(`<li>${l.substring(2)}</li>`);
                else processed.push(l ? `<p>${l}</p>` : '');
            }
        });

        if (inTable) processed.push(tableHTML + '</table></div>');

        let finalHTML = processed.join('');

        // Citations
        finalHTML = finalHTML.replace(
            /\[Source:\s*(.*?)\s*\((.*?)\)\]/g,
            (m, file, page) => `<a href="/uploads/${file}" target="_blank" class="citation-pill"><i class="fas fa-file-pdf"></i> ${file.split('.').shift()} <span>${page}</span></a>`
        );

        return finalHTML;
    }

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Enter to send
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // --- Theme Toggle ---
    toggleTheme.onclick = () => {
        const icon = toggleTheme.querySelector('i');
        if (document.body.classList.contains('light-theme')) {
            document.body.classList.remove('light-theme');
            icon.className = 'fas fa-moon';
        } else {
            document.body.classList.add('light-theme');
            icon.className = 'fas fa-sun';
        }
    };
});
