// 主应用入口
import API from './api.js';
import ModelManager from './models.js';
import QuantizationManager from './quantization.js';
import ConversationManager from './conversations.js';
import UIManager from './ui.js';
import MetricsMonitor from './metrics.js';
import ConfigManager from './config.js';

class App {
    constructor() {
        this.currentSessionId = null;
        this.conversations = new Map();
        this.modelLoaded = false;
        this.currentModelType = '';
        this.currentQuantization = 'none';
        this.quantizationInProgress = false;
        this.modelConfig = null;
        this.metricsEnabled = true;
        this.realtimeUpdateInterval = null;

        this.currentNotification = null; //
        this.notificationTimeout = null;  //

        this.uiManager = new UIManager(this);
        this.modelManager = new ModelManager(this);
        this.quantizationManager = new QuantizationManager(this);
        this.conversationManager = new ConversationManager(this);
        this.metricsMonitor = new MetricsMonitor();
        this.ConfigManager = ConfigManager;
        
        this.init();
    }

    async init() {
        console.log('应用初始化开始...');

        // 检查模型加载状态
        await this.checkInitialModelStatus();
        
        // 清理旧的对话历史
        localStorage.removeItem('pangu_conversations');
        this.conversations.clear();
        
        // 开始新对话
        // await this.conversationManager.startNewConversation();
        
        // 检查量化状态
        await this.quantizationManager.checkQuantizationStatus();
        
        // 加载保存的配置
        this.loadSavedConfig();
        
        // 初始化指标监控
        this.initMetrics();

        // 启动实时指标更新
        this.startRealtimeUpdates();
        
        console.log('应用初始化完成');
    }

    // 获取API实例
    get api() {
        return API;
    }

    // // 更新应用状态
    // updateState(newState) {
    //     Object.assign(this, newState);
    //     this.uiManager.updateUI();
    // }

    /**
     * 初始检查模型状态
     */
    async checkInitialModelStatus() {
        console.log('初始检查模型状态...');
        
        try {
            // 显示加载状态
            this.uiManager.updateModelStatus('loading', '检查模型状态...');
            
            // 检查模型状态
            const isLoaded = await this.modelManager.checkModelStatus();
            
            if (isLoaded) {
                console.log('初始检查: 模型已加载');
            } else {
                console.log('初始检查: 模型未加载');
            }
            
        } catch (error) {
            console.error('初始模型状态检查失败:', error);
            // 即使检查失败，也显示未加载状态
            this.uiManager.updateModelStatus('error', '检查失败', error.message);
        }
    }

    //加载保存的配置
    loadSavedConfig() {
        if (ConfigManager.loadConfigFromStorage()) {
            console.log('✅ 已加载保存的模型配置');
        }
    }

    //保存模型配置
    async saveModelConfig() {
        const config = ConfigManager.getCurrentConfig();
        
        // 验证配置
        if (!config.model_path.trim()) {
            alert('请输入模型加载路径');
            return;
        }
        
        if (!config.quantized_path.trim()) {
            alert('请输入量化模型保存路径');
            return;
        }
        
        try {
            // 保存到本地存储
            ConfigManager.saveConfigToStorage(config);
            
            // 发送到后端
            const result = await API.saveModelConfig(config);
            
            if (result.status === 'success') {
                alert('✅ 模型配置已保存');
                console.log('配置已保存:', config);
            } else {
                alert(`❌ 保存配置失败: ${result.message}`);
            }
        } catch (error) {
            console.error('保存配置失败:', error);
            alert('❌ 保存配置失败，请检查网络连接');
        }
    }
    
    /**
     * 加载自定义配置的模型
     */
    async loadCustomModel() {
        const config = ConfigManager.getCurrentConfig();
        
        if (!config.model_path.trim()) {
            alert('请输入模型加载路径');
            return;
        }
        
        try {
            // 显示加载状态
            this.uiManager.showNotification('正在加载自定义模型...', 'info');
            alert('正在加载自定义模型，请耐心等待...');
            this.modelManager.setModelStatus('loading', '正在加载自定义模型...');
            
            // 构建vLLM配置
            const vllmConfig = {
                model_type: 'vllm',
                model_path: config.model_path,
                tensor_parallel_size: config.tensor_parallel,
                max_num_batched_tokens: config.max_batch_tokens,
                max_num_seqs: config.max_seqs,
                gpu_memory_utilization: config.gpu_memory_utilization,
            };
            
            // 调用API加载模型
            const result = await API.loadCustomModel(vllmConfig);
            
            this.modelManager.handleLoadResult(result, 'full_precision', config, vllmConfig);

            // 再次检查模型状态以确保UI同步
            await this.checkInitialModelStatus();
            
        } catch (error) {
            console.error('加载自定义模型失败:', error);
            alert('❌ 加载自定义模型失败: ' + error.message);
        } finally {
            this.uiManager.hideNotification();
        }
    }

    /**
     * 加载自定义配置的模型
     */
    async loadCustomQuantizedModel() {
        const config = ConfigManager.getCurrentConfig();
        
        if (!config.model_path.trim()) {
            alert('请输入模型加载路径');
            return;
        }
        
        try {
            // 显示加载状态
            this.uiManager.showNotification('正在加载自定义量化模型...', 'info');
            alert('正在加载自定义量化模型，请耐心等待...');
            this.modelManager.setModelStatus('loading', '正在加载自定义量化模型...');
            
            // 构建vLLM配置
            const vllmConfig = {
                model_type: 'vllm',
                model_path: config.quantized_path,
                tensor_parallel_size: config.tensor_parallel,
                max_num_batched_tokens: config.max_batch_tokens,
                max_num_seqs: config.max_seqs,
                gpu_memory_utilization: config.gpu_memory_utilization,
                // quantization: config.quantization_bits === 4 ? 'bitsandbytes' : null
            };
            
            // 调用API加载模型
            const result = await API.loadCustomModel(vllmConfig);
            
            this.modelManager.handleLoadResult(result, 'quantized', config, vllmConfig);

            // 再次检查模型状态以确保UI同步
            await this.checkInitialModelStatus();
            
        } catch (error) {
            console.error('加载自定义量化模型失败:', error);
            alert('❌ 加载自定义量化模型失败: ' + error.message);
        } finally {
            this.uiManager.hideNotification();
        }
    }

    /**
     * 检查自定义的量化模型路径上是否已有量化模型
     */
    async checkQuantizationModelStatus() {
        const config = ConfigManager.getCurrentConfig();
        if (!config.quantized_path || !config.quantized_path.trim()) {
            alert('⚠️ 未设置量化模型路径，请先设置后再检查量化状态');
        }
            
        try {
            // 显示检查状态
            this.uiManager.showNotification('正在检查量化模型...', 'info');
            
            // 调用API检查量化模型
            const result = await API.checkQuantizationModelPath(config);
            
            if (result.bnb_model_exists !== undefined) {
                const message = result.bnb_model_exists 
                ? `✅ 量化模型已存在\n\n路径: ${result.quantized_path}\n量化功能: ${result.quantization_available ? '可用' : '不可用'}`
                : `⚠️ 量化模型不存在\n\n路径: ${result.quantized_path}\n量化功能: ${result.quantization_available ? '可用' : '不可用'}\n\n建议：请先进行模型量化`;
                
                alert(message);

            } else {
                // 如果响应格式不符合预期
                console.warn('API返回格式不符合预期:', result);
                alert('❌ 检查量化模型失败：响应格式错误');
            }
        } catch (error) {
            console.error('检查量化模型状态失败:', error);
            this.uiManager.showNotification('检查量化模型状态失败', 'error');
        } finally {
            this.uiManager.hideNotification();
        }
    }

    /**
     * 根据自定义配置进行模型量化
     */
    async quantizeFromConfig() {
        const config = ConfigManager.getCurrentConfig();
        // 验证配置
        if (!config.model_path || !config.model_path.trim()) {
            alert('请输入原始模型路径');
            return;
        }
        if (!config.quantized_path || !config.quantized_path.trim()) {
            alert('请输入量化模型保存路径');
            return;
        }
        try {
            // 确认对话框
            const confirmMessage = `
                                    开始量化模型：

                                    原始模型路径: ${config.model_path}
                                    量化保存路径: ${config.quantized_path}
                                    量化位数: ${config.quantization_bits}bit
                                    张量并行度: ${config.tensor_parallel}

                                    注意：量化过程可能需要较长时间，请确保：
                                    1. 有足够的磁盘空间
                                    2. 模型文件完整
                                    3. 网络连接稳定

                                    确定要开始量化吗？
                                    `.trim();
            
            if (!confirm(confirmMessage)) {
                return;
            }

            // 调用API开始量化
            const result = await API.quantizeModelFromConfig(config);
            
            console.log('量化结果:', result);
            
            if (result.status === 'success') {
                // 量化成功
                alert(result.message);
            } else {
                // 量化失败
                alert('模型量化失败');
            }
            
        } catch (error) {
            console.error('量化过程出错:', error);
            alert(error.message);
        }
    }

    /**
     * 启动实时指标更新
     */
    startRealtimeUpdates() {
        // 每100ms更新一次实时指标
        this.realtimeUpdateInterval = setInterval(() => {
            if (this.metricsMonitor && this.metricsEnabled) {
                // 强制更新实时显示
                this.metricsMonitor.updateRealtimeMetrics();
            }
        }, 100);
    }
    /**
     * 停止实时指标更新
     */
    stopRealtimeUpdates() {
        if (this.realtimeUpdateInterval) {
            clearInterval(this.realtimeUpdateInterval);
            this.realtimeUpdateInterval = null;
        }
    }

    /**
     * 初始化指标监控
     */
    initMetrics() {
        console.log('初始化指标监控...');
        
        // 监听消息发送开始 - 发送消息时触发
        document.addEventListener('messageSendStart', (event) => {
            console.log('收到消息发送开始事件');
            if (this.metricsEnabled) {
                this.metricsMonitor.startNewConversation();
            }
        });
        
        // 监听第一个Token到达
        document.addEventListener('firstTokenReceived', (event) => {
            console.log('收到第一个Token事件');
            if (this.metricsEnabled) {
                this.metricsMonitor.recordFirstToken();
            }
        });
        
        // 监听Token生成
        document.addEventListener('tokenGenerated', (event) => {
            console.log('收到Token生成事件:', event.detail);
            if (this.metricsEnabled && this.metricsMonitor) {
                const tokenCount = event.detail?.count || 1;
                this.metricsMonitor.recordTokenGeneration(tokenCount);
            }
        });
        
        // 监听对话结束
        document.addEventListener('conversationEnd', (event) => {
            console.log('收到对话结束事件');
            if (this.metricsEnabled && this.metricsMonitor) {
                const totalTokens = event.detail?.tokens || 0;
                this.metricsMonitor.endConversation(totalTokens);
            }
        });
        
        console.log('指标监控初始化完成');
    }
    
    /**
     * 切换指标监控状态
     */
    toggleMetrics() {
        this.metricsEnabled = !this.metricsEnabled;
        console.log(`指标监控: ${this.metricsEnabled ? '启用' : '禁用'}`);
    }
    
    /**
     * 刷新指标显示
     */
    refreshMetrics() {
        // 可以从后端获取更详细的指标
        API.getPerformanceMetrics().then(data => {
            if (data && data.metrics) {
                this.updateMetricsDisplay(data.metrics);
            }
        }).catch(error => {
            console.error('获取性能指标失败:', error);
        });
    }
    
    /**
     * 更新指标显示
     */
    updateMetricsDisplay(metrics) {
        // 更新各个指标显示
        // 这里可以根据后端返回的详细指标进行更新
    }

    /**
     * 获取当前指标
     */
    getCurrentMetrics() {
        return this.metricsMonitor.getCurrentMetrics();
    }
    
    /**
     * 清空指标
     */
    clearMetrics() {
        this.metricsMonitor.clearMetrics();
    }
    
    /**
     * 隐藏指标卡片
     */
    hideMetricsCard() {
        this.metricsMonitor.hideMetricsCard();
    }
}

// 启动应用
const app = new App();
window.app = app;

// 导出全局函数供HTML使用

window.showQuantizationDialog = () => app.quantizationManager.showQuantizationDialog();
window.hideQuantizationDialog = () => app.quantizationManager.hideQuantizationDialog();
window.toggleSidebar = () => app.uiManager.toggleSidebar();
window.handleKeyDown = (event) => app.uiManager.handleKeyDown(event);



//已调试
window.startNewConversation = () => app.conversationManager.startNewConversation();
window.clearCurrentConversation = () => app.conversationManager.clearCurrentConversation();
window.switchConversation = (sessionId) => app.conversationManager.switchConversation(sessionId);
window.deleteConversation = (sessionId, event) => app.conversationManager.deleteConversation(sessionId, event);

window.checkQuantizationModelStatus = () => app.checkQuantizationModelStatus();
window.quantizeFromConfig = () => app.quantizeFromConfig();

window.saveModelConfig = () => app.saveModelConfig();
window.loadCustomModel = () => app.loadCustomModel();
window.loadCustomQuantizedModel = () => app.loadCustomQuantizedModel();

window.sendMessage = () => app.uiManager.sendMessage();
window.unloadModel = () => app.modelManager.unloadModel();
window.clearCurrentConversation = () => app.conversationManager.clearCurrentConversation();

window.toggleAdvancedConfig = () => {
    const options = document.getElementById('advancedOptions');
    const arrow = document.getElementById('advancedArrow');
    
    if (options.style.display === 'none') {
        options.style.display = 'block';
        arrow.textContent = '▲';
        document.getElementById('advancedToggle').classList.add('active');
    } else {
        options.style.display = 'none';
        arrow.textContent = '▼';
        document.getElementById('advancedToggle').classList.remove('active');
    }
};
window.refreshMetrics = () => app.refreshMetrics();
window.clearMetrics = () => app.clearMetrics();
window.hideMetricsCard = () => app.hideMetricsCard();

// GPU内存利用率滑块事件
document.getElementById('gpuMemoryUtil').addEventListener('input', function() {
    document.getElementById('gpuMemoryValue').textContent = `${this.value}%`;
});


// 调整侧边栏高度
function adjustSidebarHeight() {
    const sidebar = document.getElementById('sidebar');
    const windowHeight = window.innerHeight;
    sidebar.style.height = `${windowHeight}px`;
    
    // 计算可用高度
    const configHeight = document.querySelector('.model-config').offsetHeight;
    const statusHeight = document.querySelector('.model-status').offsetHeight;
    const controlsHeight = document.querySelector('.model-controls').offsetHeight;
    const metricsHeight = document.querySelector('.metrics-panel').offsetHeight;
    const newChatBtnHeight = document.querySelector('.new-chat-btn').offsetHeight;
    
    const fixedHeight = configHeight + statusHeight + controlsHeight + metricsHeight + newChatBtnHeight + 40;
    const conversationsList = document.getElementById('conversationsList');
    
    if (conversationsList) {
        const availableHeight = windowHeight - fixedHeight - 20; // 减去边距
        conversationsList.style.maxHeight = `${Math.max(200, availableHeight)}px`;
        conversationsList.style.overflowY = 'auto';
    }
}

// 在窗口大小改变时重新调整
window.addEventListener('resize', adjustSidebarHeight);

document.addEventListener('DOMContentLoaded', () => {
    
    adjustSidebarHeight();
    
    // 如果内容仍然被截断，尝试滚动到底部
    setTimeout(() => {
        const sidebar = document.getElementById('sidebar');
        sidebar.scrollTop = sidebar.scrollHeight;
    }, 100);
});

// 页面加载完成后立即检查模型状态
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM加载完成...');
    
    // 延迟一点确保所有元素都加载完成
    setTimeout(() => {
        if (window.app) {
            console.log('应用已存在，重新检查模型状态...');
            // 如果应用已经创建，再次检查状态确保UI同步
            window.app.checkInitialModelStatus();
        } else {
            console.log('等待应用初始化...');
        }
    }, 500);
});

// 监听模型状态变化事件
document.addEventListener('modelLoaded', () => {
    console.log('收到modelLoaded事件，更新UI');
});

document.addEventListener('modelUnloaded', () => {
    console.log('收到modelUnloaded事件，更新UI');
});

