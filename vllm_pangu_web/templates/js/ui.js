class UIManager {
    constructor(app) {
        this.app = app;
        this.currentConversation = null;
        this.typingAnimation = null;
    }

    /**
     * 更新模型状态显示
     */
    updateModelStatus(status, message, details = '') {
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const statusDetails = document.getElementById('statusDetails');
        
        // 更新类名
        statusIndicator.className = 'status-indicator';
        switch (status) {
            case 'unloaded':
                statusIndicator.classList.add('status-unloaded');
                break;
            case 'loading':
                statusIndicator.classList.add('status-loading');
                statusText.textContent = '模型加载中...';
                statusDetails.textContent = '请稍候...';
                return; // 立即返回，不更新下面的文本
            case 'loaded':
                statusIndicator.classList.add('status-loaded');
                break;
            case 'error':
                statusIndicator.classList.add('status-error');
                break;
        }
        
        // 更新文本
        statusText.textContent = message;
        statusDetails.textContent = details;
    }
    /**
     * 更新模型徽章
     */
    updateModelBadge(modelName, modelType = 'full') {
        const badge = document.getElementById('modelBadge');
        if (badge) {
            const typeText = modelType === 'quantized' ? '(量化)' : '';
            badge.textContent = `${modelName} ${typeText}`;
            badge.className = `model-badge model-badge-${modelType}`;
        }
    }

    /**
     * 启用聊天输入
     */
    enableChatInput() {
        document.getElementById('chatInput').disabled = false;
        document.getElementById('chatInput').placeholder = '请输入您的问题...';
        document.getElementById('sendButton').disabled = false;
        document.getElementById('warningMessage').style.display = 'none';
    }

    /**
     * 禁用聊天输入
     */
    disableChatInput() {
        document.getElementById('chatInput').disabled = true;
        document.getElementById('chatInput').placeholder = '请先加载模型...';
        document.getElementById('sendButton').disabled = true;
        document.getElementById('warningMessage').style.display = 'block';
    }

    /**
     * 显示通知
     * @param {string} message - 通知内容
     * @param {string} type - 通知类型：'info'|'success'|'error'|'warning'
     * @param {number} duration - 显示时长（毫秒），0表示不自动隐藏
     */
    showNotification(message, type = 'info', duration = 5000) {
        // 先移除现有通知
        this.hideNotification();
        
        // 创建通知元素
        this.currentNotification = document.createElement('div');
        this.currentNotification.className = `notification notification-${type}`;
        
        // 添加图标
        let icon = 'ℹ️';
        if (type === 'success') icon = '✅';
        if (type === 'error') icon = '❌';
        if (type === 'warning') icon = '⚠️';
        
        this.currentNotification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icon}</span>
                <span class="notification-message">${message}</span>
            </div>
            <button class="notification-close" onclick="window.app.uiManager.hideNotification()">×</button>
        `;
        
        // 添加到页面
        document.body.appendChild(this.currentNotification);
        
        // 添加显示动画
        setTimeout(() => {
            if (this.currentNotification) {
                this.currentNotification.classList.add('show');
            }
        }, 10);
        
        // 设置自动隐藏（如果duration > 0）
        if (duration > 0) {
            this.notificationTimeout = setTimeout(() => {
                this.hideNotification();
            }, duration);
        }
        
        console.log(`显示通知: ${type} - ${message}`);
    }

    /**
     * 隐藏当前通知
     */
    hideNotification() {
        // 清除超时计时器
        if (this.notificationTimeout) {
            clearTimeout(this.notificationTimeout);
            this.notificationTimeout = null;
        }
        
        if (this.currentNotification) {
            // 添加淡出动画
            this.currentNotification.classList.remove('show');
            
            // 延迟移除元素
            setTimeout(() => {
                if (this.currentNotification && this.currentNotification.parentElement) {
                    this.currentNotification.parentElement.removeChild(this.currentNotification);
                }
                this.currentNotification = null;
            }, 300); // 匹配CSS过渡时间
        }
        
        console.log('隐藏通知');
    }

    updateModelStatusUI(status) {
        console.log('更新模型状态UI:', status);
        
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const statusDetails = document.getElementById('statusDetails');
        const quantizationStatus = document.getElementById('quantizationStatus');
        const warningMessage = document.getElementById('warningMessage');
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        const modelBadge = document.getElementById('modelBadge');
        const unloadBtn = document.getElementById('unloadBtn');
        
        if (!statusIndicator || !statusText) {
            console.error('找不到模型状态元素');
            return;
        }
        
        // 重置状态
        statusIndicator.className = 'status-indicator';
        
        if (status.loaded) {
            // 模型已加载
            statusIndicator.classList.add('status-loaded');
            statusText.textContent = '模型已加载';
            
            let details = `${status.model_name || '模型'}`;
            if (status.quantization && status.quantization !== 'none') {
                details += ` | ${status.quantization.toUpperCase()}`;
            }
            if (status.tensor_parallel_size && status.tensor_parallel_size > 1) {
                details += ` | TP=${status.tensor_parallel_size}`;
            }
            statusDetails.textContent = details;
            
            // 更新量化状态显示
            if (status.quantization && status.quantization !== 'none') {
                quantizationStatus.textContent = `量化方法: ${status.quantization.toUpperCase()}`;
                quantizationStatus.style.display = 'block';
            } else {
                quantizationStatus.style.display = 'none';
            }
            
            // 更新模型徽章
            let badgeText = status.model_name || '已加载模型';
            if (status.quantization && status.quantization !== 'none') {
                badgeText += ` (${status.quantization})`;
            }
            modelBadge.textContent = badgeText;
            
            // 启用聊天
            warningMessage.style.display = 'none';
            chatInput.placeholder = '输入消息...';
            chatInput.disabled = false;
            sendButton.disabled = false;
            
            // 启用卸载按钮，禁用加载按钮
            if (unloadBtn) unloadBtn.disabled = false;
            document.querySelectorAll('.control-btn:not(#unloadBtn)').forEach(btn => {
                btn.disabled = true;
            });
            
            console.log('UI更新为: 模型已加载');
            
        } else {
            // 模型未加载
            statusIndicator.classList.add('status-unloaded');
            statusText.textContent = '模型未加载';
            statusDetails.textContent = '请选择模型加载';
            quantizationStatus.style.display = 'none';
            modelBadge.textContent = '未加载';
            
            // 禁用聊天
            warningMessage.style.display = 'block';
            chatInput.placeholder = '请先加载模型...';
            chatInput.disabled = true;
            sendButton.disabled = true;
            
            // 启用加载按钮，禁用卸载按钮
            if (unloadBtn) unloadBtn.disabled = true;
            document.querySelectorAll('.control-btn:not(#unloadBtn)').forEach(btn => {
                btn.disabled = false;
            });
            
            console.log('UI更新为: 模型未加载');
        }
        
        // 处理加载中状态
        if (status.loading) {
            statusIndicator.classList.add('status-loading');
            statusText.textContent = '模型加载中...';
            statusDetails.textContent = '请稍候...';
            quantizationStatus.style.display = 'none';
            
            // 禁用所有按钮
            document.querySelectorAll('.control-btn').forEach(btn => {
                btn.disabled = true;
            });
        }
    }

    async sendMessage() {
        console.log('尝试发送消息...');

        const chatInput = document.getElementById('chatInput');
        const userInput = chatInput.value.trim();
        
        if (!userInput) return;

        // 触发消息发送开始事件
        const startEvent = new CustomEvent('messageSendStart', {
            detail: {
                message: userInput,
                timestamp: performance.now()
            }
        });
        document.dispatchEvent(startEvent);
        
        // 立即显示实时指标区域
        if (this.app.metricsMonitor) {
            this.app.metricsMonitor.showRealtimeMetrics();
        }
        
        const sendButton = document.getElementById('sendButton');
        sendButton.disabled = true;
        chatInput.disabled = true;
        
        this.app.conversationManager.addMessage('user', userInput);
        
        chatInput.value = '';
        this.adjustTextareaHeight(chatInput);
        
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.style.display = 'flex';

        // 记录开始时间用于TTFT计算
        const startTime = performance.now();
        let firstTokenReceived = false;
        let tokenCount = 0;
        
        try {
            console.log('调用API生成响应...');

            const response = await this.app.api.generate(userInput, this.app.currentSessionId);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let botResponse = '';
            
            // 创建AI消息元素
            const messageElement = this.app.conversationManager.addMessage('bot', '');
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const data = JSON.parse(line);
                            if (data.text) {
                                botResponse += data.text;
                                
                                // 更新AI消息内容（使用ConversationManager的方法）
                                this.app.conversationManager.updateBotMessageContent(messageElement, botResponse);
                                
                                // 更新Token计数
                                const newTokens = data.text.length > 0 ? 1 : 0;
                                tokenCount += newTokens;
                                
                                // 触发Token生成事件
                                document.dispatchEvent(new CustomEvent('tokenGenerated', {
                                    detail: {
                                        count: newTokens,
                                        total: tokenCount,
                                        timestamp: Date.now()
                                    }
                                }));

                                // 如果这是第一个Token，触发事件
                                if (!firstTokenReceived) {
                                    const ttft = performance.now() - startTime;
                                    console.log(`首Token到达时间(TTFT): ${ttft.toFixed(2)}ms`);
                                    
                                    document.dispatchEvent(new CustomEvent('firstTokenReceived', {
                                        detail: {
                                            ttft: ttft,
                                            timestamp: Date.now()
                                        }
                                    }));
                                    firstTokenReceived = true;
                                }
                            }
                            
                            if (data.finished) {
                                console.log('响应完成');
                                
                                // 最终渲染，添加复制按钮
                                const contentElement = messageElement.querySelector('.message-content');
                                if (contentElement) {
                                    this.app.conversationManager.addCopyCodeButtons(contentElement);
                                }
                                
                                if (data.session_id && data.session_id !== this.app.currentSessionId) {
                                    this.app.currentSessionId = data.session_id;
                                }
                                
                                // 触发对话结束事件
                                document.dispatchEvent(new CustomEvent('conversationEnd', {
                                    detail: {
                                        tokens: tokenCount,
                                        duration: performance.now() - startTime,
                                        timestamp: Date.now()
                                    }
                                }));

                                break;
                            }
                        } catch (e) {
                            // 忽略解析错误
                        }
                    }
                }
            }
            
            // 更新对话标题和历史
            this.app.conversationManager.updateConversationTitle(userInput);
            this.app.conversationManager.saveMessageToConversation('user', userInput);
            this.app.conversationManager.saveMessageToConversation('assistant', botResponse);
            
        } catch (error) {
            console.error('发送消息失败:', error);
            this.app.conversationManager.addMessage('bot', '抱歉，发生错误：' + error.message);

            // 触发结束事件
            const errorEndEvent = new CustomEvent('conversationEnd', {
                detail: {
                    tokens: tokenCount,
                    duration: performance.now() - startTime,
                    error: error.message,
                    timestamp: Date.now()
                }
            });
            document.dispatchEvent(errorEndEvent);
            
        } finally {
            // 清理流式渲染状态
            this.app.conversationManager.lastRenderTime = 0;
            
            typingIndicator.style.display = 'none';
            sendButton.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
            
            // 延迟隐藏实时指标区域
            setTimeout(() => {
                if (this.app.metricsMonitor) {
                    this.app.metricsMonitor.hideRealtimeMetrics();
                }
            }, 3000);
        }
    }


    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
        
        if (event.key === 'Enter' && event.shiftKey) {
            return;
        }
        
        setTimeout(() => {
            this.adjustTextareaHeight(event.target);
        }, 0);
    }

    adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
    }

}

export default UIManager;