import API from './api.js';

class ConversationManager {
    constructor(app) {
        this.app = app;
        this.lastRenderTime = 0; // 用于流式渲染性能优化
    }

    async startNewConversation() {
        try {
            const data = await API.startConversation();
            
            this.app.currentSessionId = data.session_id;
            const conversation = {
                id: this.app.currentSessionId,
                title: '新对话',
                messages: [],
                createdAt: new Date()
            };
            
            this.app.conversations.set(this.app.currentSessionId, conversation);
            this.saveConversationsToStorage();
            
            this.updateConversationsList();
            this.updateChatTitle('新对话');
            this.clearMessages();
            
            let statusMessage = '您好！新的对话已开始，';
            if (this.app.modelLoaded) {
                statusMessage += `我是盘古AI助手${this.app.currentModelType === 'ascend' ? ' (昇腾NPU)' : ' (VLLM)'}`;
                if (this.app.currentQuantization !== 'none') {
                    statusMessage += ` (${this.app.currentQuantization.toUpperCase()}量化)`;
                }
                statusMessage += '，有什么可以帮您的吗？';
            } else {
                statusMessage += '请先在侧边栏选择并加载模型后再开始对话。';
            }
            
            this.addMessage('bot', statusMessage);
            
        } catch (error) {
            console.error('开始新对话失败:', error);
            alert('❌ 开始新对话失败，请检查服务器连接');
        }
    }

    switchConversation(sessionId) {
        const conversation = this.app.conversations.get(sessionId);
        if (conversation) {
            this.app.currentSessionId = sessionId;
            this.updateChatTitle(conversation.title);
            this.clearMessages();
            
            conversation.messages.forEach(msg => {
                this.addMessage(msg.role, msg.content, false);
            });
            
            // 重新渲染所有AI消息的Markdown
            setTimeout(() => {
                this.rerenderAllBotMessages();
            }, 100);
            
            this.updateConversationsList();
        }
    }

    async clearCurrentConversation() {
        if (!this.app.currentSessionId) return;
        
        if (confirm('确定要清空当前对话的历史吗？')) {
            try {
                await API.clearConversation(this.app.currentSessionId);
                
                const conversation = this.app.conversations.get(this.app.currentSessionId);
                if (conversation) {
                    conversation.messages = [];
                    this.saveConversationsToStorage();
                }
                
                this.clearMessages();
                this.addMessage('bot', '对话历史已清空，有什么可以帮您的吗？');
                
            } catch (error) {
                console.error('清空历史失败:', error);
            }
        }
    }

    deleteConversation(sessionId, event) {
        event.stopPropagation();
        
        if (confirm('确定要删除这个对话吗？此操作不可撤销。')) {
            this.app.conversations.delete(sessionId);
            this.saveConversationsToStorage();
            
            if (sessionId === this.app.currentSessionId) {
                this.startNewConversation();
            } else {
                this.updateConversationsList();
            }
        }
    }

    updateConversationTitle(firstMessage) {
        const conversation = this.app.conversations.get(this.app.currentSessionId);
        if (conversation && conversation.title === '新对话') {
            const title = firstMessage.length > 20 ? 
                firstMessage.substring(0, 20) + '...' : firstMessage;
            conversation.title = title;
            this.updateChatTitle(title);
            this.updateConversationsList();
            this.saveConversationsToStorage();
        }
    }

    saveMessageToConversation(role, content) {
        const conversation = this.app.conversations.get(this.app.currentSessionId);
        if (conversation) {
            conversation.messages.push({ role, content });
            this.saveConversationsToStorage();
        }
    }

    updateConversationsList() {
        const conversationsList = document.getElementById('conversationsList');
        conversationsList.innerHTML = '';
        
        const sortedConversations = Array.from(this.app.conversations.values())
            .sort((a, b) => b.createdAt - a.createdAt);
        
        sortedConversations.forEach(conversation => {
            const item = document.createElement('div');
            item.className = `conversation-item ${conversation.id === this.app.currentSessionId ? 'active' : ''}`;
            item.onclick = () => this.switchConversation(conversation.id);
            
            item.innerHTML = `
                <span>💬</span>
                <div class="conversation-preview" title="${conversation.title}">
                    ${conversation.title}
                </div>
                <button class="delete-conversation" onclick="deleteConversation('${conversation.id}', event)">
                    ×
                </button>
            `;
            
            conversationsList.appendChild(item);
        });
    }

    saveConversationsToStorage() {
        const data = Array.from(this.app.conversations.entries());
        localStorage.setItem('pangu_conversations', JSON.stringify(data));
    }

    loadConversationsFromStorage() {
        const stored = localStorage.getItem('pangu_conversations');
        if (stored) {
            try {
                const data = JSON.parse(stored);
                this.app.conversations = new Map(data);
                this.updateConversationsList();
            } catch (e) {
                console.error('加载对话历史失败:', e);
            }
        }
    }

    updateChatTitle(title) {
        document.getElementById('chatTitle').textContent = title;
    }

    addMessage(role, content, scroll = true) {
        const messagesContainer = document.getElementById('messagesContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`;
        
        const avatar = role === 'user' ? 
            '<div class="message-avatar user-avatar">您</div>' :
            '<div class="message-avatar bot-avatar">AI</div>';

        let contentHtml = '';
        
        if (role === 'user') {
            // 用户消息：保持原有格式，只处理换行
            const formattedContent = content.replace(/\n/g, '<br>');
            contentHtml = formattedContent;
        } else {
            // AI消息：添加Markdown容器，但不立即渲染（流式渲染时再渲染）
            contentHtml = content;
        }
        
        messageDiv.innerHTML = `
            ${avatar}
            <div class="message-content ${role === 'bot' ? 'markdown-content' : ''}" 
                 ${role === 'bot' ? `data-raw-text="${this.escapeHtml(content)}"` : ''}>
                ${role === 'user' ? contentHtml : ''}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        
        if (role === 'bot' && content) {
            // 如果有初始内容，立即渲染
            this.renderBotMessageContent(messageDiv, content);
        }
        
        if (scroll) {
            this.scrollToBottom();
        }
        
        return messageDiv;
    }

    /**
     * 渲染AI消息的Markdown内容
     */
    renderBotMessageContent(messageDiv, content) {
        const contentElement = messageDiv.querySelector('.message-content');
        if (!contentElement) return;
        
        try {
            // 保存原始文本
            contentElement.dataset.rawText = content;
            
            // 使用marked渲染Markdown
            const renderedHtml = marked.parse(content);
            contentElement.innerHTML = renderedHtml;
            
            // 应用代码高亮
            if (typeof hljs !== 'undefined') {
                contentElement.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
            
            // 如果是最终内容，添加复制按钮
            if (!content.includes('...') && content.length > 10) {
                this.addCopyCodeButtons(contentElement);
            }
            
        } catch (error) {
            console.error('渲染Markdown失败:', error);
            // 如果渲染失败，显示原始文本
            contentElement.textContent = content;
        }
    }

    /**
     * 更新AI消息内容（用于流式渲染）
     */
    updateBotMessageContent(messageDiv, newContent) {
        const contentElement = messageDiv.querySelector('.message-content');
        if (!contentElement) return;
        
        // 更新原始文本
        contentElement.dataset.rawText = newContent;
        
        // 性能优化：限制渲染频率（每100ms最多渲染一次）
        const now = Date.now();
        if (this.lastRenderTime && now - this.lastRenderTime < 100) {
            return;
        }
        
        try {
            // 使用marked渲染Markdown
            const renderedHtml = marked.parse(newContent);
            contentElement.innerHTML = renderedHtml;
            
            // 应用代码高亮
            if (typeof hljs !== 'undefined') {
                contentElement.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
            
            this.lastRenderTime = now;
            this.scrollToBottom();
            
        } catch (error) {
            console.error('更新Markdown失败:', error);
            contentElement.textContent = newContent;
        }
    }

    /**
     * 重新渲染所有AI消息的Markdown
     */
    rerenderAllBotMessages() {
        const botMessages = document.querySelectorAll('.bot-message .message-content.markdown-content');
        botMessages.forEach(contentElement => {
            const rawText = contentElement.dataset.rawText;
            if (rawText) {
                try {
                    const renderedHtml = marked.parse(rawText);
                    contentElement.innerHTML = renderedHtml;
                    
                    if (typeof hljs !== 'undefined') {
                        contentElement.querySelectorAll('pre code').forEach((block) => {
                            hljs.highlightElement(block);
                        });
                    }
                    
                    this.addCopyCodeButtons(contentElement);
                } catch (error) {
                    console.error('重新渲染Markdown失败:', error);
                    contentElement.textContent = rawText;
                }
            }
        });
    }

    /**
     * 添加复制代码按钮
     */
    addCopyCodeButtons(container) {
        if (!container) return;
        
        container.querySelectorAll('pre').forEach((preElement) => {
            // 检查是否已经有复制按钮
            if (!preElement.querySelector('.copy-code-btn')) {
                const copyButton = document.createElement('button');
                copyButton.className = 'copy-code-btn';
                copyButton.innerHTML = '📋';
                copyButton.title = '复制代码';
                copyButton.style.cssText = `
                    position: absolute;
                    top: 8px;
                    right: 8px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    color: white;
                    padding: 4px 8px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                    opacity: 0.7;
                    transition: opacity 0.2s;
                    z-index: 10;
                `;
                
                copyButton.addEventListener('mouseenter', () => {
                    copyButton.style.opacity = '1';
                });
                
                copyButton.addEventListener('mouseleave', () => {
                    copyButton.style.opacity = '0.7';
                });
                
                copyButton.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const codeElement = preElement.querySelector('code');
                    const codeText = codeElement ? codeElement.textContent : preElement.textContent;
                    
                    try {
                        await navigator.clipboard.writeText(codeText);
                        copyButton.innerHTML = '✅';
                        copyButton.title = '已复制';
                        
                        setTimeout(() => {
                            copyButton.innerHTML = '📋';
                            copyButton.title = '复制代码';
                        }, 2000);
                    } catch (error) {
                        console.error('复制失败:', error);
                        copyButton.innerHTML = '❌';
                        copyButton.title = '复制失败';
                        
                        setTimeout(() => {
                            copyButton.innerHTML = '📋';
                            copyButton.title = '复制代码';
                        }, 2000);
                    }
                });
                
                preElement.style.position = 'relative';
                preElement.appendChild(copyButton);
            }
        });
    }

    /**
     * HTML转义
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    clearMessages() {
        const messagesContainer = document.getElementById('messagesContainer');
        const warningMessage = document.getElementById('warningMessage');
        const metricsCard = document.getElementById('metricsCard');
        
        // 清空消息容器但保留警告消息和指标卡片
        messagesContainer.innerHTML = '';
        
        if (warningMessage) {
            messagesContainer.appendChild(warningMessage);
        }
        if (metricsCard) {
            messagesContainer.appendChild(metricsCard);
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
}

export default ConversationManager;