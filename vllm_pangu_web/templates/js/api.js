// API接口封装
const API_BASE = 'http://localhost:1040';

class API {
    static async loadCustomModel(config) {
        const response = await fetch(`${API_BASE}/admin/load_model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        return response.json();
    }

    static async checkQuantizationModelPath(config) {
        const response = await fetch(`${API_BASE}/admin/check_quantization_path`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`检查量化模型路径失败: ${response.status} ${errorText}`);
        }
        return response.json();
    }

    static async quantizeModelFromConfig(config) {
        const response = await fetch(`${API_BASE}/admin/quantize_model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        return response.json();
    }

    static async startConversation() {
        const response = await fetch(`${API_BASE}/conversation/start`, {
            method: 'POST'
        });
        return response.json();
    }

    static async clearConversation(sessionId) {
        const response = await fetch(`${API_BASE}/conversation/${sessionId}/clear`, {
            method: 'POST'
        });
        return response.json();
    }

    static async generate(prompt, sessionId, options = {}) {
        const response = await fetch(`${API_BASE}/conversation/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt,
                session_id: sessionId,
                max_tokens: 2048,
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.1,
                stream: true,
                ...options
            })
        });
        return response;
    }

    static async getModelConfig() {
        const response = await fetch(`${API_BASE}/admin/model_config`);
        return response.json();
    }

    static async saveModelConfig(config) {
        const response = await fetch(`${API_BASE}/admin/save_config`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        });
        return response.json();
    }

    static async getPerformanceMetrics() {
        const response = await fetch(`${API_BASE}/admin/performance_metrics`);
        return response.json();
    }

    async loadVllmModel(config) {
        try {
            const response = await fetch('/admin/load_vllm_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('加载VLLM模型失败:', error);
            return {
                success: false,
                message: error.message
            };
        }
    }

    /**
     * 获取模型状态
     */
    async getModelStatus() {
        try {
            const response = await fetch('/admin/model_status');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('获取模型状态失败:', error);
            return {
                success: false,
                loaded: false
            };
        }
    }

    /**
     * 卸载模型
     */
    async unloadModel() {
        try {
            const response = await fetch('/admin/unload_model', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('卸载模型失败:', error);
            return {
                success: false,
                message: error.message
            };
        }
    }
}

export default API;