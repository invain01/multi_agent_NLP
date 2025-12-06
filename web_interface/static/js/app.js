/**
 * 多智能体学术写作优化系统 - 前端应用
 */

class MultiAgentApp {
    constructor() {
        this.socket = null;
        this.currentTaskId = null;
        this.currentTab = 'text-optimization';
        this.roundMetrics = [];
        this.roundChart = null;
        
        this.init();
    }

    init() {
        this.checkBootstrap();
        this.setupSocketConnection();
        this.setupEventListeners();
        this.loadConfiguration();
        this.setupTabNavigation();
    }

    checkBootstrap() {
        // 检查Bootstrap是否正确加载
        if (typeof bootstrap === 'undefined') {
            console.error('Bootstrap JavaScript未加载！');
            return;
        }
        console.log('Bootstrap版本:', bootstrap.Tooltip.VERSION);
        
        // 确保所有Bootstrap组件初始化
        this.initializeBootstrapComponents();
        console.log('Bootstrap已成功加载，所有交互功能应该可以正常使用');
    }

    initializeBootstrapComponents() {
        // 专门为高级设置accordion添加点击处理
        this.initAdvancedSettingsAccordion();

        // 初始化所有tooltip组件
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(tooltipEl => {
            new bootstrap.Tooltip(tooltipEl);
        });

        // 初始化所有popover组件
        document.querySelectorAll('[data-bs-toggle="popover"]').forEach(popoverEl => {
            new bootstrap.Popover(popoverEl);
        });
    }

    initAdvancedSettingsAccordion() {
        // 获取高级设置的按钮和内容区域
        const accordionButton = document.querySelector('#advancedSettings .accordion-button');
        const collapseElement = document.getElementById('collapseAdvanced');
        
        if (!accordionButton || !collapseElement) {
            console.warn('高级设置accordion元素未找到');
            return;
        }

        console.log('初始化高级设置accordion');

        // 移除Bootstrap的默认data属性，使用纯JavaScript控制
        accordionButton.removeAttribute('data-bs-toggle');
        accordionButton.removeAttribute('data-bs-target');

        // 添加点击事件处理器
        accordionButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            console.log('高级设置按钮被点击');
            
            const isExpanded = collapseElement.classList.contains('show');
            
            if (isExpanded) {
                // 折叠
                collapseElement.classList.remove('show');
                accordionButton.classList.add('collapsed');
                accordionButton.setAttribute('aria-expanded', 'false');
                console.log('高级设置已折叠');
            } else {
                // 展开
                collapseElement.classList.add('show');
                accordionButton.classList.remove('collapsed');
                accordionButton.setAttribute('aria-expanded', 'true');
                console.log('高级设置已展开');
            }
        });

        console.log('高级设置accordion初始化完成');
    }

    setupSocketConnection() {
        // 初始化Socket.IO连接
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to server');
                this.updateStatus('已连接到服务器', 'success');
            });

            this.socket.on('disconnect', () => {
                console.log('Disconnected from server');
                this.updateStatus('与服务器断开连接', 'warning');
            });

            this.socket.on('task_update', (taskData) => {
                console.log('收到任务更新:', taskData);
                this.handleTaskUpdate(taskData);
            });

            this.socket.on('round_update', (data) => {
                console.log('收到轮次更新:', data);
                this.handleRoundUpdate(data);
            });

            this.socket.on('connect_error', (error) => {
                console.error('Socket连接错误:', error);
                this.updateStatus('连接服务器失败', 'danger');
            });
        } catch (error) {
            console.error('Socket.IO初始化失败:', error);
            this.socket = null;
        }
    }

    setupEventListeners() {
        // 导航切换
        document.querySelectorAll('[data-tab]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const tab = e.target.getAttribute('data-tab');
                this.switchTab(tab);
            });
        });

        // 输入模式切换
        document.querySelectorAll('input[name="inputMode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.toggleInputMode(e.target.id);
            });
        });

        // 视图模式切换
        document.querySelectorAll('input[name="viewMode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.toggleViewMode(e.target.id);
            });
        });

        // 按钮事件
        this.setupButtonEvents();
        
        // 文件上传事件
        this.setupFileUploadEvents();

        // 配置管理
        this.setupConfigEvents();
    }

    setupButtonEvents() {
        // 文本优化
        const startOptimizationBtn = document.getElementById('startOptimization');
        if (startOptimizationBtn) {
            startOptimizationBtn.addEventListener('click', () => {
                console.log('开始优化按钮被点击');
                this.startTextOptimization();
            });
        } else {
            console.error('找不到开始优化按钮');
        }

        const clearFormBtn = document.getElementById('clearForm');
        if (clearFormBtn) {
            clearFormBtn.addEventListener('click', () => {
                console.log('清空表单按钮被点击');
                this.clearForm();
            });
        }

        // 数据合成功能已移除（前端）

        // 评估分析
        document.getElementById('startEvaluation')?.addEventListener('click', () => {
            this.startEvaluation();
        });

        // 数据蒸馏
        document.getElementById('startDistillation')?.addEventListener('click', () => {
            this.startDistillation();
        });

        // 下载按钮
        document.getElementById('downloadText')?.addEventListener('click', () => {
            this.downloadResult('text');
        });

        document.getElementById('downloadHTML')?.addEventListener('click', () => {
            this.downloadResult('html');
        });

        document.getElementById('downloadJSON')?.addEventListener('click', () => {
            this.downloadResult('json');
        });
    }

    setupFileUploadEvents() {
        const fileInput = document.getElementById('fileUpload');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                this.handleFileUpload(e.target.files[0]);
            });
        }

        // 拖拽上传支持
        const fileInputArea = document.getElementById('fileInputArea');
        if (fileInputArea) {
            fileInputArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                fileInputArea.classList.add('drag-over');
            });

            fileInputArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                fileInputArea.classList.remove('drag-over');
            });

            fileInputArea.addEventListener('drop', (e) => {
                e.preventDefault();
                fileInputArea.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileUpload(files[0]);
                }
            });
        }
    }

    setupConfigEvents() {
        document.getElementById('saveConfig')?.addEventListener('click', () => {
            this.saveConfiguration();
        });
    }

    setupTabNavigation() {
        // 设置默认标签
        this.switchTab('text-optimization');
    }

    switchTab(tabName) {
        // 隐藏所有tab内容
        document.querySelectorAll('.tab-content').forEach(content => {
            content.style.display = 'none';
        });

        // 显示选中的tab
        const targetTab = document.getElementById(tabName);
        if (targetTab) {
            targetTab.style.display = 'block';
        }

        // 更新导航样式
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        const activeLink = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        this.currentTab = tabName;
    }

    toggleInputMode(mode) {
        const textArea = document.getElementById('textInputArea');
        const fileArea = document.getElementById('fileInputArea');

        if (mode === 'textInput') {
            textArea.style.display = 'block';
            fileArea.style.display = 'none';
        } else {
            textArea.style.display = 'none';
            fileArea.style.display = 'block';
        }
    }

    toggleViewMode(mode) {
        const comparisonView = document.getElementById('comparisonView');
        const finalResultView = document.getElementById('finalResultView');
        const roundDetailsView = document.getElementById('roundDetailsView');

        // 隐藏所有视图
        comparisonView.style.display = 'none';
        finalResultView.style.display = 'none';
        roundDetailsView.style.display = 'none';

        // 显示选中的视图
        switch (mode) {
            case 'sideBySide':
                comparisonView.style.display = 'block';
                break;
            case 'finalOnly':
                finalResultView.style.display = 'block';
                break;
            case 'roundByRound':
                roundDetailsView.style.display = 'block';
                break;
        }
    }

    async handleFileUpload(file) {
        if (!file) return;

        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showAlert('文件太大，请选择小于10MB的文件', 'error');
            return;
        }

        try {
            const text = await this.readFileAsText(file);
            document.getElementById('fileContent').textContent = text.substring(0, 2000) + (text.length > 2000 ? '\n...(预览已截断)' : '');
            document.getElementById('filePreview').style.display = 'block';
            
            // 保存完整内容到隐藏字段
            document.getElementById('fileUpload').dataset.content = text;
        } catch (error) {
            this.showAlert('文件读取失败: ' + error.message, 'error');
        }
    }

    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('文件读取失败'));
            reader.readAsText(file, 'UTF-8');
        });
    }

    async startTextOptimization() {
        console.log('startTextOptimization 函数开始执行');
        
        const inputMode = document.querySelector('input[name="inputMode"]:checked').id;
        console.log('输入模式:', inputMode);
        
        let text = '';
        let isFileMode = false;

        if (inputMode === 'textInput') {
            text = document.getElementById('originalText').value.trim();
            console.log('文本输入模式，文本长度:', text.length);
        } else {
            text = document.getElementById('fileUpload').dataset.content;
            isFileMode = true;
            console.log('文件上传模式，文本长度:', text ? text.length : 0);
        }

        if (!text) {
            console.log('文本为空，显示警告');
            this.showAlert('请输入要优化的文本或上传文件', 'warning');
            return;
        }

        const requirements = document.getElementById('requirements').value.trim();
        const rounds = parseInt(document.getElementById('rounds').value);
        const language = document.getElementById('language').value;
        const enableTools = document.getElementById('enableTools').checked;
        const enableMemory = document.getElementById('enableMemory').checked;

        try {
            this.showProgress(true);
            this.updateStatus('正在启动优化任务...', 'info');

            let response;
            if (isFileMode) {
                // 文件模式
                const formData = new FormData();
                const blob = new Blob([text], { type: 'text/plain' });
                formData.append('file', blob, 'uploaded_text.txt');
                formData.append('requirements', requirements);
                formData.append('rounds', rounds);
                formData.append('chunk_size', document.getElementById('chunkSize').value);
                formData.append('overlap', document.getElementById('chunkOverlap').value);
                formData.append('max_chunks', document.getElementById('maxChunks').value);
                formData.append('enable_tools', enableTools);
                formData.append('enable_memory', enableMemory);

                response = await fetch('/api/optimize/file', {
                    method: 'POST',
                    body: formData
                });
            } else {
                // 文本模式
                response = await fetch('/api/optimize/text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text,
                        requirements,
                        rounds,
                        language,
                        enable_tools: enableTools,
                        enable_memory: enableMemory
                    })
                });
            }

            const result = await response.json();
            
            if (result.status === 'success') {
                this.currentTaskId = result.task_id;
                this.updateStatus('优化任务已启动，正在处理...', 'info');
                this.updateProgress(10);
                
                // 加入任务房间接收更新
                this.socket.emit('join_task', { task_id: this.currentTaskId });
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            this.showAlert('启动优化失败: ' + error.message, 'error');
            this.showProgress(false);
        }
    }

    // 数据合成功能已移除（前端）

    async startEvaluation() {
        const cases = document.getElementById('evalCases').value.trim();
        if (!cases) {
            this.showAlert('请输入评估用例', 'warning');
            return;
        }

        const rounds = parseInt(document.getElementById('evalRounds').value);

        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    cases,
                    rounds
                })
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.currentTaskId = result.task_id;
                this.socket.emit('join_task', { task_id: this.currentTaskId });
                this.showAlert('评估任务已启动', 'success');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            this.showAlert('启动评估失败: ' + error.message, 'error');
        }
    }

    async startDistillation() {
        const fileInput = document.getElementById('distillSource');
        if (!fileInput.files[0]) {
            this.showAlert('请选择源JSONL文件', 'warning');
            return;
        }

        const outputFilename = document.getElementById('distillOutput').value.trim();
        
        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('output_filename', outputFilename);

            const response = await fetch('/api/distill', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.showDistillationResult(result.message, result.output_path);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            this.showAlert('蒸馏失败: ' + error.message, 'error');
        }
    }

    async saveConfiguration() {
        const config = {
            openai_api_key: document.getElementById('openaiApiKey').value,
            openai_base_url: document.getElementById('openaiBaseUrl').value,
            llm_model: document.getElementById('llmModel').value,
            serpapi_api_key: document.getElementById('serpApiKey').value
        };

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.showAlert('配置保存成功', 'success');
                this.saveConfigToLocalStorage(config);
                
                // 关闭模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('configModal'));
                if (modal) modal.hide();
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            this.showAlert('保存配置失败: ' + error.message, 'error');
        }
    }

    handleRoundUpdate(data) {
        if (data.task_id !== this.currentTaskId) return;
        
        const roundData = data.round_data;
        if (!roundData) return;
        
        // 添加轮次日志
        this.addLogMessage(`第${roundData.round}轮完成`, 'success');
        
        if (roundData.scores) {
            const scoreStr = Object.entries(roundData.scores)
                .map(([k, v]) => `${k}:${typeof v === 'number' ? v.toFixed(1) : v}`)
                .join(', ');
            this.addLogMessage(`评分: {${scoreStr}}`, 'info');
        }
        
        // 如果已经显示了结果区域，更新轮次详情
        if (document.getElementById('resultsArea').style.display !== 'none') {
            this.appendRoundDetail(roundData);
        }

        // 更新每一轮的评价指标折线图
        this.updateRoundMetrics(roundData);
    }

    handleTaskUpdate(taskData) {
        if (taskData.id !== this.currentTaskId) return;

        console.log('处理任务更新:', taskData);

        // 更新进度和状态
        this.updateProgress(taskData.progress || 0);
        this.updateStatus(this.getStatusMessage(taskData), this.getStatusClass(taskData.status));

        // 添加日志消息
        if (taskData.message) {
            this.addLogMessage(taskData.message, 'info');
        }

        // 处理任务完成
        if (taskData.status === 'completed') {
            this.handleTaskCompletion(taskData);
        } else if (taskData.status === 'failed') {
            this.handleTaskFailure(taskData);
        }
    }

    appendRoundDetail(roundData) {
        const roundsContainer = document.getElementById('roundsContainer');
        if (!roundsContainer) return;
        
        const roundDiv = document.createElement('div');
        roundDiv.className = 'round-item';
        roundDiv.id = `round-${roundData.round}`;
        roundDiv.innerHTML = `
            <div class="round-header">
                <h6>第 ${roundData.round} 轮优化</h6>
                <small>${roundData.timestamp ? new Date(roundData.timestamp).toLocaleString() : ''}</small>
            </div>
            <div class="round-content">
                <h6>优化文本：</h6>
                <pre class="mb-3">${roundData.optimized_text || ''}</pre>
                
                <h6>反馈评价：</h6>
                <div class="round-feedback">
                    <pre>${roundData.agent_b_feedback || ''}</pre>
                </div>
                
                ${roundData.tool_observations && roundData.tool_observations !== '(无)' ? `
                    <details>
                        <summary>工具观察</summary>
                        <pre>${roundData.tool_observations}</pre>
                    </details>
                ` : ''}
                
                ${roundData.scores ? `
                    <div class="mt-2">
                        <small>评分: ${JSON.stringify(roundData.scores, null, 2)}</small>
                    </div>
                ` : ''}
            </div>
        `;
        
        // 如果轮次已存在，替换；否则添加
        const existingRound = document.getElementById(`round-${roundData.round}`);
        if (existingRound) {
            existingRound.replaceWith(roundDiv);
        } else {
            roundsContainer.appendChild(roundDiv);
        }
        
        // 滚动到最新轮次
        roundDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    handleTaskCompletion(taskData) {
        this.showProgress(false);
        this.displayResults(taskData.result);
        this.showAlert('任务完成！', 'success');
        
        // 根据任务类型更新相应的UI
        if (taskData.type === 'evaluation') {
            this.displayEvaluationResults(taskData.result);
        }
    }

    handleTaskFailure(taskData) {
        this.showProgress(false);
        this.showAlert('任务失败: ' + (taskData.error || '未知错误'), 'error');
        this.updateStatus('任务失败', 'error');
    }

    displayResults(result) {
        if (!result) return;

        // 重置轮次指标缓存和图表
        this.roundMetrics = [];
        if (this.roundChart) {
            this.roundChart.destroy();
            this.roundChart = null;
        }
        const roundMetricsDisplay = document.getElementById('roundMetricsDisplay');
        if (roundMetricsDisplay) {
            roundMetricsDisplay.style.display = 'none';
        }

        // 显示结果区域
        document.getElementById('resultsArea').style.display = 'block';

        // 填充文本对比
        if (result.original_text && result.final_text) {
            document.getElementById('originalTextDisplay').textContent = result.original_text;
            document.getElementById('optimizedTextDisplay').textContent = result.final_text;
            document.getElementById('finalTextDisplay').textContent = result.final_text;
        }

        // 显示评分
        this.displayScores(result);

        // 显示轮次详情
        this.displayRoundDetails(result);

        // 基于完整日志构建每轮评价指标曲线
        this.buildRoundMetricsFromLog(result);

        // 滚动到结果区域
        document.getElementById('resultsArea').scrollIntoView({ behavior: 'smooth' });
    }

    displayScores(result) {
        const scoresContainer = document.getElementById('scoresContainer');
        if (!scoresContainer) return;

        scoresContainer.innerHTML = '';

        let scores = {};
        if (result.log && result.log.length > 0) {
            const lastRound = result.log[result.log.length - 1];
            scores = lastRound.scores || {};
        }

        const scoreLabels = {
            quality: '质量',
            rigor: '严谨性',
            logic: '逻辑',
            novelty: '新颖性'
        };

        // 定义评分对应的图标
        const scoreIcons = {
            quality: 'bi-star-fill',
            rigor: 'bi-shield-check',
            logic: 'bi-diagram-3',
            novelty: 'bi-lightbulb'
        };

        Object.entries(scores).forEach(([key, value]) => {
            if (typeof value === 'number') {
                const scoreCard = document.createElement('div');
                scoreCard.className = 'score-card';
                const icon = scoreIcons[key] || 'bi-circle-fill';
                scoreCard.innerHTML = `
                    <span class="score-value">${value.toFixed(1)}</span>
                    <span class="score-label">
                        <i class="bi ${icon} me-1"></i>${scoreLabels[key] || key}
                    </span>
                `;
                scoresContainer.appendChild(scoreCard);
            }
        });
        
        // 显示高级学术指标
        this.displayAdvancedMetrics(result);
    }

    displayAdvancedMetrics(result) {
        const advancedDisplay = document.getElementById('advancedMetricsDisplay');
        const advancedContainer = document.getElementById('advancedMetricsContainer');
        
        if (!advancedContainer) {
            console.warn('Advanced metrics container not found');
            return;
        }

        advancedContainer.innerHTML = '';
        
        // 首先尝试从result.advanced_metrics获取（来自后端）
        let advancedMetrics = result.advanced_metrics || {};
        
        // 如果没有，则从log中获取
        if (Object.keys(advancedMetrics).length === 0 && result.log && result.log.length > 0) {
            const lastRound = result.log[result.log.length - 1];
            advancedMetrics = lastRound.advanced_metrics || {};
        }

        // 如果没有高级指标，隐藏容器并返回
        if (Object.keys(advancedMetrics).length === 0) {
            if (advancedDisplay) {
                advancedDisplay.style.display = 'none';
            }
            return;
        }

        // 显示容器
        if (advancedDisplay) {
            advancedDisplay.style.display = 'block';
        }

        const metricLabels = {
            'academic_formality': '学术规范性',
            'academic_formality_score': '学术规范性',
            'citation_completeness': '引用完整性',
            'citation_completeness_score': '引用完整性',
            'novelty': '创新度',
            'novelty_score': '创新度',
            'language_fluency': '语言流畅度',
            'language_fluency_score': '语言流畅度',
            'sentence_balance': '句子平衡',
            'sentence_balance_score': '句子平衡',
            'argumentation': '论证强度',
            'argumentation_strength': '论证强度',
            'expression_diversity': '表达多样性',
            'expression_diversity_score': '表达多样性',
            'structure_completeness': '结构完整性',
            'structure_completeness_score': '结构完整性',
            'tense_consistency': '时态一致',
            'tense_consistency_score': '时态一致',
            'overall_quality': '总体质量'
        };

        let metricsCount = 0;

        // 尝试从首轮日志中获取初始分，用于当后端未提供改进值时计算差值
        let initialMetrics = {};
        if (result.log && result.log.length > 0) {
            const firstRound = result.log[0].advanced_metrics || {};
            initialMetrics = firstRound;
        }

        // 定义我们关心的九个学术指标顺序（显示顺序固定）
        const metricKeys = [
            'academic_formality',
            'citation_completeness',
            'novelty',
            'language_fluency',
            'sentence_balance',
            'argumentation',
            'expression_diversity',
            'structure_completeness',
            'tense_consistency'
        ];

        metricKeys.forEach((metricKey) => {
            // 忽略整体质量等聚合字段
            if (metricKey === 'overall_quality') return;

            // 兼容后端可能使用的字段名变体
            const finalCandidates = [metricKey, `${metricKey}_score`, `${metricKey}_value`];
            const improvementCandidates = [
                `${metricKey}_improvement`,
                `${metricKey}_delta`,
                `${metricKey}_gain`,
                `${metricKey}_diff`
            ];

            let finalVal = null;
            for (const c of finalCandidates) {
                if (Object.prototype.hasOwnProperty.call(advancedMetrics, c) && typeof advancedMetrics[c] === 'number') {
                    finalVal = advancedMetrics[c];
                    break;
                }
            }

            // 获取显式的改进字段（如果存在）
            let improvementVal = null;
            for (const c of improvementCandidates) {
                if (Object.prototype.hasOwnProperty.call(advancedMetrics, c) && typeof advancedMetrics[c] === 'number') {
                    improvementVal = advancedMetrics[c];
                    break;
                }
            }

            // 如果后端没有提供 improvement，而我们有首轮初始值，则用 final - initial
            if (improvementVal === null && finalVal !== null) {
                for (const c of finalCandidates) {
                    // 尝试从初始轮找对应字段
                    if (Object.prototype.hasOwnProperty.call(initialMetrics, c) && typeof initialMetrics[c] === 'number') {
                        improvementVal = finalVal - initialMetrics[c];
                        break;
                    }
                }
            }

            // 如果既没有 final 也没有 improvement，则跳过
            if (finalVal === null && improvementVal === null) return;

            const label = metricLabels[`${metricKey}_score`] || metricLabels[metricKey] || metricKey;

            const displayFinal = (typeof finalVal === 'number') ? finalVal.toFixed(3) : 'N/A';

            // 计算用于显示改进的样式与符号
            let improvementClass = 'neutral';
            let symbol = '→';
            let displayImprovement = improvementVal !== null ? improvementVal : 0;

            if (typeof displayImprovement === 'number') {
                if (displayImprovement > 0.02) {
                    improvementClass = 'positive';
                    symbol = '↑';
                } else if (displayImprovement < -0.02) {
                    improvementClass = 'negative';
                    symbol = '↓';
                }
            }

            // 进度条基于最终分数（若无final则用改进的相对映射）
            let barPercent = 0;
            if (typeof finalVal === 'number') {
                barPercent = Math.max(0, Math.min(100, finalVal * 100));
            } else if (typeof displayImprovement === 'number') {
                barPercent = Math.max(0, Math.min(100, (displayImprovement + 1) * 50));
            }

            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card';
            metricCard.innerHTML = `
                <div class="metric-name">${label}</div>
                <div class="metric-value">${displayFinal}</div>
                <div class="metric-bar">
                    <div class="metric-bar-fill" style="width: ${barPercent}%"></div>
                </div>
            `;
            advancedContainer.appendChild(metricCard);
            metricsCount++;
        });

        if (metricsCount === 0) {
            if (advancedDisplay) {
                advancedDisplay.style.display = 'none';
            }
        }
    }

    displayRoundDetails(result) {
        const roundsContainer = document.getElementById('roundsContainer');
        if (!roundsContainer) return;

        roundsContainer.innerHTML = '';

        const log = result.log || [];
        log.forEach((round, index) => {
            if (index === 0) return; // 跳过初始轮次

            const roundDiv = document.createElement('div');
            roundDiv.className = 'round-item';
            roundDiv.innerHTML = `
                <div class="round-header">
                    <h6>第 ${round.round} 轮优化</h6>
                    <small>${round.timestamp ? new Date(round.timestamp).toLocaleString() : ''}</small>
                </div>
                <div class="round-content">
                    <h6>优化文本：</h6>
                    <pre class="mb-3">${round.optimized_text || ''}</pre>
                    
                    <h6>反馈评价：</h6>
                    <div class="round-feedback">
                        <pre>${round.agent_b_feedback || ''}</pre>
                    </div>
                    
                    ${round.tool_observations ? `
                        <details>
                            <summary>工具观察</summary>
                            <pre>${round.tool_observations}</pre>
                        </details>
                    ` : ''}
                </div>
            `;
            roundsContainer.appendChild(roundDiv);
        });
    }

    buildRoundMetricsFromLog(result) {
        this.roundMetrics = [];
        const log = result.log || [];

        log.forEach((round, index) => {
            if (index === 0) return; // 跳过初始轮次
            const metrics = round.advanced_metrics || {};
            if (!metrics || Object.keys(metrics).length === 0) return;
            this.roundMetrics.push({
                round: round.round,
                metrics: metrics
            });
        });

        this.renderRoundMetricsChart();
    }

    updateRoundMetrics(roundData) {
        const metrics = roundData.advanced_metrics || {};
        if (!metrics || Object.keys(metrics).length === 0) {
            return;
        }

        if (!this.roundMetrics) {
            this.roundMetrics = [];
        }

        const index = typeof roundData.round === 'number' ? roundData.round - 1 : this.roundMetrics.length;
        this.roundMetrics[index] = {
            round: roundData.round,
            metrics: metrics
        };

        this.renderRoundMetricsChart();
    }

    renderRoundMetricsChart() {
        const container = document.getElementById('roundMetricsDisplay');
        const canvas = document.getElementById('roundMetricsChart');

        if (!canvas || typeof Chart === 'undefined') {
            return;
        }

        const validRounds = (this.roundMetrics || [])
            .filter(Boolean)
            .sort((a, b) => (a.round || 0) - (b.round || 0));

        if (validRounds.length === 0) {
            if (container) {
                container.style.display = 'none';
            }
            return;
        }

        const labels = validRounds.map(r => `第${r.round}轮`);

        const metricLabelMap = {
            overall_quality: '总体质量',
            academic_formality: '学术规范性',
            citation_completeness: '引用完整性',
            novelty: '创新度',
            language_fluency: '语言流畅度',
            sentence_balance: '句子平衡',
            argumentation: '论证强度'
        };

        const palette = [
            '#0d6efd', // 蓝
            '#198754', // 绿
            '#ffc107', // 黄
            '#dc3545', // 红
            '#6f42c1', // 紫
            '#20c997', // 青
            '#fd7e14'  // 橙
        ];

        const firstMetrics = validRounds[0].metrics || {};
        const candidateKeys = [
            'overall_quality',
            'academic_formality',
            'citation_completeness',
            'novelty',
            'language_fluency'
        ];
        let metricKeys = candidateKeys.filter(k => Object.prototype.hasOwnProperty.call(firstMetrics, k));

        if (metricKeys.length === 0) {
            metricKeys = Object.keys(firstMetrics).filter(k => typeof firstMetrics[k] === 'number').slice(0, 4);
        }

        const datasets = metricKeys.map((key, idx) => {
            return {
                label: metricLabelMap[key] || key,
                data: validRounds.map(r => {
                    const m = r.metrics || {};
                    const v = m[key];
                    return typeof v === 'number' ? v : null;
                }),
                borderColor: palette[idx % palette.length],
                backgroundColor: palette[idx % palette.length] + '33',
                tension: 0.25,
                spanGaps: true
            };
        });

        if (this.roundChart) {
            this.roundChart.destroy();
        }

        const ctx = canvas.getContext('2d');
        this.roundChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const label = context.dataset.label || '';
                                const value = typeof context.parsed.y === 'number' ? context.parsed.y.toFixed(3) : 'N/A';
                                return `${label}: ${value}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1,
                        ticks: {
                            stepSize: 0.1
                        }
                    }
                }
            }
        });

        if (container) {
            container.style.display = 'block';
        }
    }

    displayEvaluationResults(result) {
        const resultsContainer = document.getElementById('evaluationResults');
        if (!resultsContainer) return;

        resultsContainer.innerHTML = '';

        if (result.summary) {
            const summary = result.summary;
            resultsContainer.innerHTML = `
                <h6>评估指标汇总：</h6>
                <div class="row">
                    <div class="col-md-6">
                        <div class="card mb-2">
                            <div class="card-body">
                                <h6>长度增益: ${summary.len_gain_avg}</h6>
                                <h6>TTR增益: ${summary.ttr_gain_avg}</h6>
                                <h6>重复度降低: ${summary.repetition_delta_avg}</h6>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card mb-2">
                            <div class="card-body">
                                <h6>可读性提升: ${summary.readability_gain_avg}</h6>
                                <h6>连贯性提升: ${summary.coherence_gain_avg}</h6>
                                <h6>样本数: ${summary.n}</h6>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    // 合成进度相关函数已移除（前端）

    showDistillationResult(message, outputPath) {
        const resultsContainer = document.getElementById('distillationResults');
        if (resultsContainer) {
            document.getElementById('distillationMessage').textContent = message;
            resultsContainer.style.display = 'block';
        }
        this.showAlert(message, 'success');
    }

    async downloadResult(type) {
        if (!this.currentTaskId) {
            this.showAlert('没有可下载的结果', 'warning');
            return;
        }

        try {
            const url = `/api/download/${this.currentTaskId}/${type}`;
            const link = document.createElement('a');
            link.href = url;
            link.download = '';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } catch (error) {
            this.showAlert('下载失败: ' + error.message, 'error');
        }
    }

    showProgress(show) {
        const progressContainer = document.getElementById('progressContainer');
        const idleMessage = document.getElementById('idleMessage');
        
        if (show) {
            progressContainer.style.display = 'block';
            idleMessage.style.display = 'none';
        } else {
            progressContainer.style.display = 'none';
            idleMessage.style.display = 'block';
        }
    }

    updateProgress(percentage) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
        }
    }

    updateStatus(message, type) {
        const statusElement = document.getElementById('currentStatus');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `badge bg-${type === 'info' ? 'primary' : type === 'success' ? 'success' : type === 'warning' ? 'warning' : 'danger'}`;
        }
    }

    addLogMessage(message, type = 'info') {
        const liveLog = document.getElementById('liveLog');
        if (liveLog) {
            const logEntry = document.createElement('div');
            logEntry.className = `log-${type}`;
            logEntry.innerHTML = `<span class="log-timestamp">[${new Date().toLocaleTimeString()}]</span> ${message}`;
            liveLog.appendChild(logEntry);
            
            // 限制日志数量，防止内存溢出
            const logEntries = liveLog.querySelectorAll('div');
            if (logEntries.length > 100) {
                logEntries[0].remove();
            }
            
            // 自动滚动到底部
            liveLog.scrollTop = liveLog.scrollHeight;
        }
    }

    clearForm() {
        // 清空表单
        document.getElementById('originalText').value = '';
        document.getElementById('requirements').value = '学术表达提升,逻辑结构优化';
        document.getElementById('rounds').value = '3';
        document.getElementById('fileUpload').value = '';
        document.getElementById('filePreview').style.display = 'none';
        
        // 隐藏结果
        document.getElementById('resultsArea').style.display = 'none';
        
        // 重置进度
        this.showProgress(false);
        this.updateProgress(0);
    }

    showAlert(message, type) {
        // 创建Bootstrap警告框
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // 3秒后自动消失
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 3000);
    }

    getStatusMessage(taskData) {
        switch (taskData.status) {
            case 'created': return '任务已创建';
            case 'running': return '正在处理中...';
            case 'completed': return '任务完成';
            case 'failed': return '任务失败';
            default: return '状态未知';
        }
    }

    getStatusClass(status) {
        switch (status) {
            case 'created': return 'info';
            case 'running': return 'primary';
            case 'completed': return 'success';
            case 'failed': return 'danger';
            default: return 'secondary';
        }
    }

    saveConfigToLocalStorage(config) {
        // 敏感信息不保存到localStorage
        const safeConfig = {
            openai_base_url: config.openai_base_url,
            llm_model: config.llm_model
        };
        localStorage.setItem('multiAgent_config', JSON.stringify(safeConfig));
    }

    loadConfiguration() {
        try {
            const saved = localStorage.getItem('multiAgent_config');
            if (saved) {
                const config = JSON.parse(saved);
                if (config.openai_base_url) {
                    document.getElementById('openaiBaseUrl').value = config.openai_base_url;
                }
                if (config.llm_model) {
                    document.getElementById('llmModel').value = config.llm_model;
                }
            }
        } catch (error) {
            console.warn('Failed to load saved configuration:', error);
        }
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM加载完成，开始初始化应用');
    
    try {
        const app = new MultiAgentApp();
        
        // 将应用实例暴露给全局作用域，方便调试
        window.multiAgentApp = app;
        console.log('应用初始化完成');
        
        // 确保Bootstrap组件在DOM完全渲染后再次初始化
        setTimeout(() => {
            console.log('延迟初始化Bootstrap组件...');
            app.initializeBootstrapComponents();
            
            // 测试基本功能
            const startBtn = document.getElementById('startOptimization');
            if (startBtn) {
                console.log('找到开始优化按钮');
            } else {
                console.error('未找到开始优化按钮');
            }
            
            const textArea = document.getElementById('originalText');
            if (textArea) {
                console.log('找到文本输入框');
            } else {
                console.error('未找到文本输入框');
            }
        }, 300);
        
    } catch (error) {
        console.error('应用初始化失败:', error);
    }
});

// 页面卸载时清理WebSocket连接
window.addEventListener('beforeunload', () => {
    if (window.multiAgentApp && window.multiAgentApp.socket) {
        window.multiAgentApp.socket.disconnect();
    }
});