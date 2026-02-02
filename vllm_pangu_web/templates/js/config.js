// 添加配置管理对象
class ConfigManager {
    static getCurrentConfig() {
        try {
            const gpuMemElement = document.getElementById('gpuMemoryUtil');
            const gpuMemValue = gpuMemElement ? parseInt(gpuMemElement.value) : 90;
            
            const config = {
                model_path: document.getElementById('modelPath')?.value || '',
                quantized_path: document.getElementById('quantizedPath')?.value || '',
                tensor_parallel: parseInt(document.getElementById('tensorParallel')?.value) || 4,
                max_batch_tokens: parseInt(document.getElementById('maxBatchTokens')?.value) || 4096,
                max_seqs: parseInt(document.getElementById('maxNumSeqs')?.value) || 32,
                gpu_memory_utilization: gpuMemValue / 100,
                quantization_bits: 8,// 默认8bit量化
                quantization: 'W8A8',
                max_sequence_length: parseInt(document.getElementById('maxSeqLength')?.value) || 4096,
                num_calibration_samples: parseInt(document.getElementById('numCalibrationSamples')?.value) || 512
            };
            
            console.log('获取当前配置:', config);
            return config;
            
        } catch (error) {
            console.error('获取配置失败:', error);
            return this.getDefaultConfig();
        }
    }
    
    static getDefaultConfig() {
        return {
            model_path: '/opt/pangu/openPangu-Embedded-7B-V1.1',
            quantized_path: '/opt/pangu/openPangu-Embedded-7B-V1.1-W8A8-Dynamic-Per-Token',
            tensor_parallel: 1,
            max_batch_tokens: 4096,
            max_seqs: 32,
            gpu_memory_utilization: 0.9,
            quantization_bits: 8,
            quantization: 'W8A8',
            max_sequence_length: 4096,
            num_calibration_samples: 512
        };
    }
    
    static loadConfigFromStorage() {
        const savedConfig = localStorage.getItem('pangu_model_config');
        if (savedConfig) {
            try {
                const config = JSON.parse(savedConfig);
                
                // 安全地设置UI元素值
                this.setElementValue('modelPath', config.model_path);
                this.setElementValue('quantizedPath', config.quantized_path);
                this.setElementValue('tensorParallel', config.tensor_parallel);
                this.setElementValue('maxBatchTokens', config.max_batch_tokens);
                this.setElementValue('maxNumSeqs', config.max_seqs);
                this.setElementValue('maxSeqLength', config.max_sequence_length);
                this.setElementValue('numCalibrationSamples', config.num_calibration_samples);
                
                const gpuMemValue = Math.round((config.gpu_memory_utilization || 0.9) * 100);
                this.setElementValue('gpuMemoryUtil', gpuMemValue);
                
                const gpuMemoryValueElement = document.getElementById('gpuMemoryValue');
                if (gpuMemoryValueElement) {
                    gpuMemoryValueElement.textContent = `${gpuMemValue}%`;
                }
                
                console.log('从存储加载配置:', config);
                return true;
                
            } catch (e) {
                console.error('加载配置失败:', e);
                return false;
            }
        }
        return false;
    }
    
    static setElementValue(elementId, value) {
        const element = document.getElementById(elementId);
        if (element && value !== undefined && value !== null) {
            element.value = value;
        }
    }
    
    static saveConfigToStorage(config) {
        try {
            localStorage.setItem('pangu_model_config', JSON.stringify(config));
            console.log('配置保存到存储:', config);
            return true;
        } catch (e) {
            console.error('保存配置失败:', e);
            return false;
        }
    }
}

export default ConfigManager;