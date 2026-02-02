class ModelManager {
    constructor(app) {
        this.app = app;
        this.modelStatus = 'unloaded';
        this.currentModel = null;
        this.quantizationSupported = false;
        this.modelInfo = null;
    }

    /**
     * 检查模型状态（页面刷新后调用）
     */
    async checkModelStatus() {
        try {
            console.log('检查后端模型状态...');
            
            // 调用API获取模型状态
            const response = await fetch('/admin/model_status');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${await response.text()}`);
            }
            
            const result = await response.json();
            console.log('模型状态响应:', result);
            
            if (result.loaded) {
                // 模型已加载
                this.handleExistingModel(result);
                return true;
            } else {
                // 模型未加载
                this.handleNoModel();
                return false;
            }
            
        } catch (error) {
            console.error('检查模型状态失败:', error);
            this.handleCheckError(error);
            return false;
        }
    }

    /**
     * 处理已加载的模型
     */
    handleExistingModel(result) {
        console.log('检测到模型已加载:', result);
        
        this.modelStatus = 'loaded';
        this.currentModel = {
            type: result.model_type || 'vllm',
            config: result.model_info || {},
            name: result.model_name || '已加载模型',
            loaded_at: new Date().toISOString(),
            tensor_parallel: result.tensor_parallel_size || 1
        };
        
        this.modelInfo = {
            name: result.model_name || '已加载模型',
            path: result.model_path || '未知路径',
            type: result.model_type || 'vllm',
            tensor_parallel: result.tensor_parallel_size || 1,
            quantization: result.quantization || 'none'
        };

        // 更新UI状态
        this.setModelStatus('loaded', 
            '模型已加载',
            `模型: ${this.modelInfo.name}\nTP: ${this.modelInfo.tensor_parallel}`
        );
        
        // 更新UI元素
        this.app.uiManager.updateModelBadge(this.modelInfo.name, this.modelInfo.type);
        this.app.uiManager.enableChatInput();
        
        // 更新模型状态UI
        this.app.uiManager.updateModelStatusUI({
            loaded: true,
            model_type: this.modelInfo.type,
            quantization: this.modelInfo.quantization,
            model_name: this.modelInfo.name,
            tensor_parallel_size: this.modelInfo.tensor_parallel
        });
        
        console.log('模型状态已更新:', this.modelInfo);
    }

    /**
     * 处理未加载模型的情况
     */
    handleNoModel() {
        console.log('模型未加载');
        
        this.modelStatus = 'unloaded';
        this.currentModel = null;
        this.modelInfo = null;
        
        // 更新UI状态
        this.setModelStatus('unloaded', '模型未加载', '请选择模型加载');
        this.app.uiManager.disableChatInput();
        this.app.uiManager.updateModelBadge('未加载', 'none');
        
        // 更新模型状态UI
        this.app.uiManager.updateModelStatusUI({
            loaded: false,
            model_type: '',
            quantization: 'none',
            model_name: '模型未加载'
        });
    }

    /**
     * 处理检查错误
     */
    handleCheckError(error) {
        console.error('模型状态检查失败:', error);
        
        this.modelStatus = 'error';
        
        // 更新UI状态
        this.setModelStatus('error', '检查失败', error.message);
        this.app.uiManager.disableChatInput();
        this.app.uiManager.updateModelBadge('检查失败', 'error');
        
        // 显示错误通知
        this.app.uiManager.showNotification(`检查模型状态失败: ${error.message}`, 'error', 3000);
    }




   
    /**
     * 处理加载成功
     */
    handleLoadSuccess(result, modelType, config, loadConfig) {
        console.log('处理加载成功:', result);
        
        this.modelStatus = 'loaded';
        this.currentModel = {
            type: modelType,
            config: loadConfig,
            name: config.quantized_path,
            loaded_at: new Date().toISOString(),
            tensor_parallel: loadConfig.tensor_parallel_size
        };
        
        this.modelInfo = {
            name: result.model_name || this.currentModel.name,
            path: config.quantized_path,
            type: modelType,
            tensor_parallel: loadConfig.tensor_parallel_size,
            quantization: modelType === 'quantized' ? 'W8A8' : 'none'
        };

        // 显示成功消息
        const successMsg = `✅ ${modelType === 'quantized' ? '量化' : '全精度'}模型加载成功\n\n` +
                        `模型: ${this.modelInfo.name}\n` +
                        `TP: ${loadConfig.tensor_parallel_size}\n` +
                        `量化: ${modelType === 'quantized' ? 'W8A8' : '无'}\n` +
                        `路径: ${config.quantized_path}`;
        
        this.app.uiManager.showNotification(successMsg, 'success');
        alert(successMsg);

        // 更新UI
        this.setModelStatus('loaded', 
            `${modelType === 'quantized' ? '量化' : '全精度'}模型已加载`,
            `模型: ${this.modelInfo.name}\n TP: ${loadConfig.tensor_parallel_size}`
        );
        
        this.app.uiManager.updateModelBadge(this.modelInfo.name, modelType);
        this.app.uiManager.enableChatInput();
        this.app.uiManager.updateModelControls();
        
        // 保存加载历史
        this.saveLoadHistory(config.model_path, loadConfig.tensor_parallel_size, modelType === 'quantized');
        
        console.log('模型加载成功:', this.currentModel);
        
        // 触发模型加载事件
        window.dispatchEvent(new CustomEvent('modelLoaded', { 
            detail: { 
                model: this.currentModel,
                result: result
            }
        }));

    }

    /**
     * 处理加载错误
     */
    handleLoadError(error) {
        console.log('处理加载失败');

        this.setModelStatus('error', '模型加载失败');
        const errorMsg = `❌ 模型加载失败\n\n${error.message}`;
        this.app.uiManager.showNotification(errorMsg, 'error');
        
        window.dispatchEvent(new CustomEvent('modelLoadFailed', { 
            detail: { error: error.message }
        }));
    }

    /**
     * 处理加载结果
     */
    async handleLoadResult(result, modelType, config, loadConfig) {
        console.log('处理加载结果:', result);
        
        if (result.status === 'success' || result.success === true) {
            this.handleLoadSuccess(result, modelType, config, loadConfig);
        } else {
            this.handleLoadError(result);
        }
    }

    /**
     * 更新模型状态显示
     */
    setModelStatus(status, message, details = '') {
        this.modelStatus = status;
        this.app.uiManager.updateModelStatus(status, message, details);
        // this.app.uiManager.updateModelStatusUI(status);
    }
   
    /**
     * 卸载模型
     */
    async unloadModel() {
        try {
            if (this.modelStatus !== 'loaded') {
                this.app.uiManager.showNotification('没有已加载的模型', 'warning');
                return;
            }

            if (!confirm('确定要卸载当前模型吗？')) return;

            this.setModelStatus('unloading', '正在卸载模型...');
            
            const response = await fetch('/admin/unload_model', {
                method: 'POST'
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.handleUnloadSuccess();
            } else {
                throw new Error(result.message || '卸载失败');
            }
            
        } catch (error) {
            console.error('卸载模型失败:', error);
            this.handleUnloadError(error);
        } finally {
            // 检查模型加载状态
            await this.app.checkInitialModelStatus();
        }
    }

    /**
     * 处理卸载成功
     */
    handleUnloadSuccess() {
        this.modelStatus = 'unloaded';
        this.currentModel = null;
        this.modelInfo = null;
        
        this.setModelStatus('unloaded', '模型已卸载');
        this.app.uiManager.updateModelBadge('未加载', 'none');
        this.app.uiManager.disableChatInput();
        this.app.uiManager.showNotification('✅ 模型已卸载', 'success');
        
        window.dispatchEvent(new CustomEvent('modelUnloaded'));
    }

    /**
     * 处理卸载错误
     */
    handleUnloadError(error) {
        this.setModelStatus('error', '卸载失败');
        this.app.uiManager.showNotification(`❌ 卸载失败: ${error.message}`, 'error');
    }

    /**
     * 保存加载历史
     */
    saveLoadHistory(modelPath, tensorParallel, isQuantized) {
        try {
            const history = JSON.parse(localStorage.getItem('modelLoadHistory') || '[]');
            
            const historyItem = {
                modelPath,
                tensorParallel,
                isQuantized,
                timestamp: new Date().toISOString(),
                modelName: this.getModelNameFromPath(modelPath)
            };
            
            // 添加到历史记录开头
            history.unshift(historyItem);
            
            // 只保留最近的10条记录
            if (history.length > 10) {
                history.pop();
            }
            
            localStorage.setItem('modelLoadHistory', JSON.stringify(history));
            
        } catch (error) {
            console.warn('保存加载历史失败:', error);
        }
    }
}

export default ModelManager;