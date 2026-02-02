import API from './api.js';

class QuantizationManager {
    constructor(app) {
        this.app = app;
    }

    async checkQuantizationStatus() {
        try {
            const data = await API.getQuantizationStatus();
            this.updateQuantizationUI(data);
            return data;
        } catch (error) {
            console.error('检查量化状态失败:', error);
            return null;
        }
    }

    updateQuantizationUI(data) {
        const quantizationBtns = document.querySelectorAll('.quantization-btn');
        quantizationBtns.forEach(btn => {
            if (data && data.bnb_model_exists) {
                btn.innerHTML = '✅ 量化模型已存在';
                btn.disabled = true;
            } else {
                btn.disabled = false;
            }
        });
    }

    showQuantizationDialog(defaultBits = 4) {
        const modal = document.getElementById('quantizationModal');
        const radioButtons = document.querySelectorAll('input[name="quantizationBits"]');
        
        // 设置默认选择
        radioButtons.forEach(radio => {
            radio.checked = radio.value === defaultBits.toString();
        });
        
        modal.style.display = 'flex';
    }

    hideQuantizationDialog() {
        const modal = document.getElementById('quantizationModal');
        modal.style.display = 'none';
    }

    updateQuantizationProgress(percent, statusText) {
        const progressElement = document.getElementById('quantizationProgress');
        const progressFill = document.getElementById('quantizationProgressFill');
        const statusElement = document.getElementById('quantizationStatusText');
        
        progressElement.style.display = 'block';
        progressFill.style.width = `${percent}%`;
        statusElement.textContent = statusText;
    }
}

export default QuantizationManager;