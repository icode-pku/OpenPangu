/**
 * 推理指标监控模块
 */

class MetricsMonitor {
    constructor() {
        this.metrics = {
            // 当前对话指标
            current: {
                startTime: 0,
                requestSentTime: 0, 
                endTime: 0,
                firstTokenTime: 0,
                tokensGenerated: 0,
                totalTokens: 0,
                interTokenTimes: []
            },
            
            // 历史统计
            history: {
                ttftHistory: [],      // TTFT历史
                tpsHistory: [],       // TPS历史
                tokenCounts: [],      // Token数量历史
                e2eHistory: []        // 端到端延迟历史
            },
            
            // 实时数据
            realtime: {
                lastTokenTime: 0,
                tokenCount: 0,
                tpsWindow: []
            }
        };
        
        // Chart.js 实例
        this.chart = null;

        // 确保DOM加载完成后再初始化图表
        this.initChartWhenReady();
    }

    /**
     * 在DOM就绪时初始化图表
     */
    initChartWhenReady() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.initializeChart();
            });
        } else {
            // DOM已就绪，直接初始化
            setTimeout(() => this.initializeChart(), 1000); //1000ms延时确保Chart.js加载完成
        }
    }
    
    /**
     * 初始化性能图表
     */
    initializeChart() {
        try {
            // 检查 Chart.js 是否已加载
            if (typeof Chart === 'undefined') {
                console.error('Chart.js 库未加载');
                return;
            }
            
            const ctx = document.getElementById('metricsChart');
            if (!ctx) {
                console.warn('未找到 metricsChart 元素');
                return;
            }
            
            // 检查是否已经初始化过
            if (Chart.getChart(ctx)) {
                Chart.getChart(ctx).destroy();
            }
            
            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 10}, (_, i) => ''),
                    datasets: [
                        {
                            label: 'TPS',
                            data: Array(10).fill(0),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 2
                        },
                        {
                            label: 'ITL(ms)',
                            data: Array(10).fill(0),
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            tension: 0.4,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {
                        duration: 0
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#9ca3af',
                                font: {
                                    size: 10
                                },
                                padding: 10
                            },
                            position: 'top',
                        },
                        tooltip: {
                            enabled: true,
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: {
                                display: false
                            },
                            ticks: {
                                color: '#9ca3af',
                                font: {
                                    size: 9
                                }
                            },
                            title: {
                                display: true,
                                text: '时间点',
                                color: '#9ca3af'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#9ca3af',
                                font: {
                                    size: 9
                                }
                            },
                            title: {
                                display: true,
                                text: '数值',
                                color: '#9ca3af'
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'nearest'
                    }
                }
            });
            
            console.log('指标图表初始化成功');
        } catch (error) {
            console.error('初始化图表失败:', error);
        }
    }
    
    /**
     * 开始新的对话指标记录
     */
    startNewConversation() {
        console.log('开始新的对话指标记录');
        
        this.metrics.current = {
            startTime: performance.now(),
            requestSentTime: performance.now(), 
            endTime: 0,
            firstTokenTime: 0,
            tokensGenerated: 0,
            totalTokens: 0,
            interTokenTimes: []
        };
        
        this.metrics.realtime = {
            lastTokenTime: performance.now(),
            tokenCount: 0,
            tpsWindow: []
        };
        
        // 显示实时指标
        this.showRealtimeMetrics();
        
        // 重置图表
        this.resetChart();
    }

    /**
     * 重置图表数据
     */
    resetChart() {
        if (this.chart) {
            this.chart.data.datasets[0].data = Array(10).fill(0);
            this.chart.data.datasets[1].data = Array(10).fill(0);
            this.chart.data.labels = Array.from({length: 10}, (_, i) => `${i + 1}`);
            this.chart.update();
        }
    }

    // /**
    //  * 用户发送消息时调用
    //  */
    // recordMessageSent() {
    //     const now = performance.now();
    //     // 设置对话开始时间（用户发送消息的时间）
    //     this.metrics.current.startTime = now;
    //     this.metrics.current.requestSentTime = now;
        
    //     console.log('用户消息已发送，开始 E2E 计时');
    // }

    
    /**
     * 记录第一个Token的时间
     */
    recordFirstToken() {
        if (this.metrics.current.firstTokenTime === 0) {
            this.metrics.current.firstTokenTime = performance.now() - this.metrics.current.requestSentTime;
            
            console.log(`首Token时间(TTFT): ${this.metrics.current.firstTokenTime.toFixed(1)}ms`);
            
            // 更新TTFT显示
            this.updateMetricDisplay('metricTTFT', `${this.metrics.current.firstTokenTime.toFixed(1)} ms`);
            
            // 添加到历史
            this.metrics.history.ttftHistory.push(this.metrics.current.firstTokenTime);
            if (this.metrics.history.ttftHistory.length > 10) {
                this.metrics.history.ttftHistory.shift();
            }
        }
    }
    
    /**
     * 记录新生成的Token
     * @param {number} tokenCount - 新生成的Token数量
     */
    recordTokenGeneration(tokenCount) {
        console.log(`记录Token生成: ${tokenCount} tokens`);
        
        const currentTime = performance.now();
        
        // 更新当前对话指标
        this.metrics.current.tokensGenerated += tokenCount;
        this.metrics.current.totalTokens = Math.max(this.metrics.current.totalTokens, this.metrics.current.tokensGenerated);
        
        // 计算Token间延迟
        if (this.metrics.realtime.lastTokenTime > 0) {
            const interTokenTime = currentTime - this.metrics.realtime.lastTokenTime;
            this.metrics.current.interTokenTimes.push(interTokenTime);
        }
        
        // 更新实时窗口数据
        this.metrics.realtime.tokenCount += tokenCount;
        this.metrics.realtime.lastTokenTime = currentTime;
        this.metrics.realtime.tpsWindow.push(currentTime);
        
        // 清理过期的数据（1秒前的）
        const windowStart = currentTime - 1000;
        this.metrics.realtime.tpsWindow = this.metrics.realtime.tpsWindow.filter(
            time => time > windowStart
        );
        
        console.log(`累计Token数: ${this.metrics.current.tokensGenerated}, 实时窗口大小: ${this.metrics.realtime.tpsWindow.length}`);
        
        // 更新所有实时显示
        this.updateRealtimeMetrics();
        
        // 更新图表
        const realtimeTPS = this.metrics.realtime.tpsWindow.length;
        const avgTokenTime = this.metrics.current.interTokenTimes.length > 0 
            ? this.metrics.current.interTokenTimes.slice(-5).reduce((a, b) => a + b, 0) / 
              Math.min(this.metrics.current.interTokenTimes.length, 5)
            : 0;
        
        this.updateChart(realtimeTPS, avgTokenTime);
        
        // 根据性能设置颜色
        this.setPerformanceColors(realtimeTPS, avgTokenTime);
    }
    
    /**
     * 结束当前对话指标记录
     * @param {number} totalTokens - 总Token数
     */
    endConversation(totalTokens) {
        this.metrics.current.endTime = performance.now();
        // 只有真正开始了对话才计算 E2E
        let totalTime = 0;
        if (this.metrics.current.startTime > 0) {
            totalTime = this.metrics.current.endTime - this.metrics.current.startTime;
        }
        
        // 更新最终显示
        this.updateMetricDisplay('metricE2E', `${totalTime.toFixed(1)} ms`);

        this.metrics.current.totalTokens = totalTokens || this.metrics.current.tokensGenerated;
        
        // 计算最终指标
        // const totalTime = this.metrics.current.endTime - this.metrics.current.startTime;
        // const avgTPS = totalTime > 0 
        //     ? (this.metrics.current.tokensGenerated / totalTime) * 1000 
        //     : 0;
        
        // 更新最终显示
        // this.updateMetricDisplay('metricE2E', `${totalTime.toFixed(1)} ms`);
        this.updateMetricDisplay('metricTokens', `${this.metrics.current.tokensGenerated}`);
        
        // 计算RPS（基于历史）
        const recentConversations = this.metrics.history.e2eHistory.slice(-5);
        const avgE2E = recentConversations.length > 0 
            ? recentConversations.reduce((a, b) => a + b, 0) / recentConversations.length 
            : totalTime;
        
        const rps = avgE2E > 0 ? 1000 / avgE2E : 0;
        this.updateMetricDisplay('metricRPS', `${rps.toFixed(2)}`);
        
        // 保存到历史
        this.metrics.history.tpsHistory.push(avgTPS);
        this.metrics.history.tokenCounts.push(this.metrics.current.tokensGenerated);
        this.metrics.history.e2eHistory.push(totalTime);
        
        // 限制历史记录数量
        const maxHistory = 10;
        if (this.metrics.history.tpsHistory.length > maxHistory) {
            this.metrics.history.tpsHistory.shift();
            this.metrics.history.tokenCounts.shift();
            this.metrics.history.e2eHistory.shift();
        }
    }
    
    /**
     * 显示实时指标区域
     */
    showRealtimeMetrics() {
        const realtimeMetrics = document.getElementById('realtimeMetrics');
        if (realtimeMetrics) {
            console.log('显示实时指标区域');
            realtimeMetrics.style.display = 'flex';
            realtimeMetrics.style.opacity = '1';
            realtimeMetrics.style.transition = 'opacity 0.3s ease';
        }
    }
    
    /**
     * 隐藏实时指标区域
     */
    hideRealtimeMetrics() {
        const realtimeMetrics = document.getElementById('realtimeMetrics');
        if (realtimeMetrics) {
            console.log('隐藏实时指标区域');
            realtimeMetrics.style.display = 'none';
        }
    }

    /**
     * 更新实时指标显示
     */
    updateRealtimeMetrics() {
        const currentTime = performance.now();
        // const elapsedTime = currentTime - this.metrics.current.startTime;

        // 计算 E2E 延迟 - 只在发送消息后才开始计时
        let e2eDisplay;
        let elapsedTime;

        if (this.metrics.current.endTime > 0) {
            // 对话已结束，显示最终 E2E
            const totalTime = this.metrics.current.endTime - this.metrics.current.startTime;
            e2eDisplay = `${totalTime.toFixed(1)} ms`;
            elapsedTime = totalTime;
        } else if (this.metrics.current.startTime > 0) {
            // 消息已发送，显示当前 E2E
            elapsedTime = currentTime - this.metrics.current.startTime;
            e2eDisplay = `${elapsedTime.toFixed(1)} ms`;
        } else {
            // 消息还没发送
            e2eDisplay = '- ms';
            elapsedTime = 0;
        }
        
        // 更新显示
        this.updateMetricDisplay('metricE2E', e2eDisplay);

        // 计算实时TPS
        const windowStart = currentTime - 1000;
        this.metrics.realtime.tpsWindow = this.metrics.realtime.tpsWindow.filter(
            time => time > windowStart
        );
        const realtimeTPS = this.metrics.realtime.tpsWindow.length;
        
        // 更新显示
        this.updateMetricDisplay('realtimeTPS', `${realtimeTPS.toFixed(1)}`);
        this.updateMetricDisplay('realtimeTokens', `${this.metrics.current.tokensGenerated} tokens`);
        this.updateMetricDisplay('realtimeTime', `${elapsedTime.toFixed(0)} ms`);
        
        // 更新主指标面板
        this.updateMetricDisplay('metricTTFT', this.metrics.current.firstTokenTime > 0 
            ? `${this.metrics.current.firstTokenTime.toFixed(1)} ms` 
            : '- ms');
        
        // 计算并显示平均ITL
        if (this.metrics.current.interTokenTimes.length > 0) {
            const avgITL = this.metrics.current.interTokenTimes.reduce((a, b) => a + b, 0) / 
                          this.metrics.current.interTokenTimes.length;
            this.updateMetricDisplay('metricITL', `${avgITL.toFixed(1)} ms`);
        }
        
        // 显示TPS
        this.updateMetricDisplay('metricTPS', `${realtimeTPS.toFixed(1)}`);
        
        // 计算RPS（简化的）
        let rps = 0;
        if (elapsedTime > 0) {
            rps = 1000 / elapsedTime;
        } else if (this.metrics.history.e2eHistory.length > 0) {
            rps = 1000 / this.metrics.history.e2eHistory.slice(-1)[0];
        }
        this.updateMetricDisplay('metricRPS', `${rps.toFixed(2)}`);
        
        // 显示Token数
        this.updateMetricDisplay('metricTokens', `${this.metrics.current.tokensGenerated}`);
    }
    
    /**
     * 更新指标显示
     * @param {string} elementId - 元素ID
     * @param {string} value - 显示值
     */
    updateMetricDisplay(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            // 检查是否需要添加动画效果
            if (elementId.includes('realtime')) {
                element.classList.add('updating');
                setTimeout(() => {
                    element.classList.remove('updating');
                }, 300);
            }
            element.textContent = value;
            // console.log(`更新 ${elementId}: ${value}`);
        } else {
            console.warn(`找不到元素: ${elementId}`);
        }
    }
    
    /**
     * 更新图表数据
     * @param {number} tps - 当前TPS
     * @param {number} latency - 当前延迟
     */
    updateChart(tps, latency) {
        if (!this.chart) return;
        
        // 更新数据
        this.chart.data.datasets[0].data.push(tps);
        this.chart.data.datasets[1].data.push(latency);
        
        // 保持最多10个数据点
        const maxPoints = 10;
        if (this.chart.data.datasets[0].data.length > maxPoints) {
            this.chart.data.datasets[0].data.shift();
            this.chart.data.datasets[1].data.shift();
        }
        
        // 更新时间标签
        const labels = [];
        for (let i = 0; i < this.chart.data.datasets[0].data.length; i++) {
            labels.push(`${i + 1}s前`);
        }
        this.chart.data.labels = labels;
        
        // 更新图表
        this.chart.update('none');
    }
    
    /**
     * 根据性能设置颜色
     * @param {number} tps - 当前TPS
     * @param {number} latency - 当前延迟(ms)
     */
    setPerformanceColors(tps, latency) {
        // TPS颜色
        const tpsElement = document.getElementById('metricTPS');
        if (tpsElement) {
            if (tps > 50) tpsElement.className = 'metric-value high';
            else if (tps > 20) tpsElement.className = 'metric-value medium';
            else tpsElement.className = 'metric-value low';
        }
        
        // ITL颜色
        const itlElement = document.getElementById('metricITL');
        if (itlElement && latency > 0) {
            if (latency < 50) itlElement.className = 'metric-value high';
            else if (latency < 200) itlElement.className = 'metric-value medium';
            else itlElement.className = 'metric-value low';
        }
    }
    
    /**
     * 获取当前指标数据
     */
    getCurrentMetrics() {
        const totalTime = this.metrics.current.endTime > 0 
            ? this.metrics.current.endTime - this.metrics.current.startTime
            : performance.now() - this.metrics.current.startTime;
        
        return {
            ttft: this.metrics.current.firstTokenTime,
            itl: this.metrics.current.interTokenTimes.length > 0 
                ? this.metrics.current.interTokenTimes.reduce((a, b) => a + b, 0) / this.metrics.current.interTokenTimes.length
                : 0,
            tps: totalTime > 0 ? (this.metrics.current.tokensGenerated / totalTime) * 1000 : 0,
            e2e_latency: totalTime,
            rps: 0,
            tokens_generated: this.metrics.current.tokensGenerated,
            total_tokens: this.metrics.current.totalTokens
        };
    }
    
    /**
     * 清空所有指标
     */
    clearMetrics() {
        this.metrics = {
            current: {
                startTime: 0,
                endTime: 0,
                firstTokenTime: 0,
                tokensGenerated: 0,
                totalTokens: 0,
                interTokenTimes: []
            },
            history: {
                ttftHistory: [],
                tpsHistory: [],
                tokenCounts: [],
                e2eHistory: []
            },
            realtime: {
                lastTokenTime: 0,
                tokenCount: 0,
                tpsWindow: []
            }
        };
        
        // 重置显示
        this.resetDisplays();
        
        // 重置图表
        if (this.chart) {
            this.chart.data.datasets[0].data = [0, 0, 0, 0, 0];
            this.chart.data.datasets[1].data = [0, 0, 0, 0, 0];
            this.chart.update();
        }
        
        console.log('📊 已清空所有推理指标');
    }
    
    /**
     * 重置显示
     */
    resetDisplays() {
        const metrics = [
            'metricTTFT', 'metricITL', 'metricTPS', 'metricE2E', 'metricRPS', 'metricTokens',
            'realtimeTPS', 'realtimeTokens', 'realtimeTime'
        ];
        
        metrics.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                if (id.includes('TTFT') || id.includes('ITL') || id.includes('E2E') || id.includes('Time')) {
                    element.textContent = '- ms';
                } else if (id.includes('Tokens')) {
                    element.textContent = '0 tokens';
                } else {
                    element.textContent = '-';
                }
                
                // 移除颜色类
                element.className = element.className.replace(/\b(high|medium|low)\b/g, '');
            }
        });
        
        // 隐藏卡片
        this.hideRealtimeMetrics();
    }
}

export default MetricsMonitor;