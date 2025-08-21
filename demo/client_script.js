// 全局变量
let currentQueryId = null;
let pollingInterval = null;
let serverUrl = 'http://localhost:8080';
let currentTrajectory = [];
let serviceStatus = {};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    checkServerHealth();
    loadSavedTrajectories();
});

// 检查服务器健康状态
async function checkServerHealth() {
    try {
        const response = await fetch(`${serverUrl}/health`);
        if (response.ok) {
            serviceStatus = await response.json();
            updateStatusBar(true);
            updateAvailableOptions();
        } else {
            updateStatusBar(false);
        }
    } catch (error) {
        console.error('健康检查失败:', error);
        updateStatusBar(false);
    }
}

// 更新状态栏
function updateStatusBar(isHealthy) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const serviceInfo = document.getElementById('serviceInfo');

    if (isHealthy) {
        statusDot.style.background = '#28a745';
        statusText.textContent = '服务正常';
        serviceInfo.innerHTML = `
            <span>LLM: ${serviceStatus.llm_status}</span> |
            <span>模型: ${serviceStatus.model_name || 'N/A'}</span> 
        `;
    } else {
        statusDot.style.background = '#dc3545';
        statusText.textContent = '服务不可用';
        serviceInfo.textContent = '请检查服务器连接';
    }
}

// 更新可用选项
function updateAvailableOptions() {
    if (!serviceStatus.available_agent_types) return;

    const agentTypeSelect = document.getElementById('agentType');
    const promptTypeSelect = document.getElementById('promptType');

    // 更新Agent类型
    agentTypeSelect.innerHTML = '';
    serviceStatus.available_agent_types.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        // 设置asearcher为默认选中
        if (type === 'asearcher') {
            option.selected = true;
        }
        agentTypeSelect.appendChild(option);
    });

    // 更新Prompt类型
    promptTypeSelect.innerHTML = '';
    serviceStatus.available_prompt_types.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.textContent = type;
        // 设置ASearcher为默认选中
        if (type === 'asearcher') {
            option.selected = true;
        }
        promptTypeSelect.appendChild(option);
    });
}

// 键盘事件处理
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        startQuery();
    }
}

// 开始查询
async function startQuery() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) {
        alert('请输入查询内容');
        return;
    }

    // 重置状态
    resetQueryState();

    const requestData = {
        query: query,
        agent_type: document.getElementById('agentType').value,
        prompt_type: document.getElementById('promptType').value,
        max_turns: parseInt(document.getElementById('maxTurns').value),
        use_jina: document.getElementById('useJina').value === 'true',
        temperature: parseFloat(document.getElementById('temperature').value),
        search_client_type: "async-web-search-access"
    };

    try {
        const response = await fetch(`${serverUrl}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        if (response.ok) {
            const result = await response.json();
            currentQueryId = result.query_id;
            
            // 确保轨迹已清空并显示加载状态
            clearTrajectory();
            const trajectory = document.getElementById('trajectory');
            trajectory.innerHTML = '<div class="loading" id="loadingIndicator"><div class="spinner"></div>查询进行中...</div>';
            
            // 重置网络错误计数器
            window.networkErrorCount = 0;
            
            // 更新UI状态
            document.getElementById('startBtn').disabled = true;
            document.getElementById('cancelBtn').disabled = false;
            document.getElementById('loadingIndicator').style.display = 'flex';
            
            // 显示运行指示器
            showAgentRunningIndicator();
            
            // 开始轮询
            startPolling();
        } else {
            const error = await response.json();
            alert(`查询启动失败: ${error.detail || '未知错误'}`);
            resetQueryState(); // 失败时也要重置状态
        }
    } catch (error) {
        console.error('启动查询失败:', error);
        alert('网络错误，请检查服务器连接');
        resetQueryState(); // 失败时也要重置状态
    }
}

// 取消查询
async function cancelQuery() {
    if (!currentQueryId) return;

    try {
        const response = await fetch(`${serverUrl}/query/${currentQueryId}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            hideAgentRunningIndicator();
            stopPolling();
            addStep('cancelled', '查询已取消', '用户主动取消了查询');
            // 重置查询状态
            currentQueryId = null;
        }
    } catch (error) {
        console.error('取消查询失败:', error);
        // 即使取消失败也重置状态
        hideAgentRunningIndicator();
        stopPolling();
        currentQueryId = null;
    }
}

// 开始轮询
function startPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(async () => {
        if (!currentQueryId) return;
        
        try {
            const response = await fetch(`${serverUrl}/query/${currentQueryId}`);
            if (response.ok) {
                const result = await response.json();
                // 成功获取响应，重置网络错误计数
                window.networkErrorCount = 0;
                updateTrajectory(result);
                
                if (result.status === 'completed' || result.status === 'error' || result.status === 'cancelled') {
                    // 隐藏运行指示器
                    hideAgentRunningIndicator();
                    // 查询完成后先重置currentQueryId，再停止轮询
                    currentQueryId = null;
                    stopPolling();
                }
            } else {
                if (response.status === 404) {
                    console.error(`查询ID ${currentQueryId} 在服务器上未找到 (可能服务已重启). 停止轮询.`);
                    showError(`查询任务 (ID: ${currentQueryId}) 已失效或不存在，可能服务已重启。请刷新页面或开始新的查询。`);
                    hideAgentRunningIndicator();
                    stopPolling();
                } else {
                    const errorText = await response.text();
                    console.error(`服务器错误: ${response.status} - ${errorText}. 停止轮询.`);
                    showError(`服务器返回错误: ${response.status}. 轮询已停止。`);
                    hideAgentRunningIndicator();
                    stopPolling();
                }
            }
        } catch (error) {
            // 如果currentQueryId已被清空，说明查询已完成，不需要报错
            if (!currentQueryId) {
                console.log('查询已完成，停止轮询');
                return;
            }
            
            console.error('轮询网络错误:', error);
            // 网络错误可能是临时的，不要立即显示错误，而是记录并等待下次重试
            console.log('网络错误，将在下次轮询时重试...');
            // 如果连续多次网络错误，才显示错误消息
            if (!window.networkErrorCount) {
                window.networkErrorCount = 1;
            } else {
                window.networkErrorCount++;
                if (window.networkErrorCount >= 5) {
                    showError('连续网络错误，轮询已停止。请检查服务器连接。');
                    hideAgentRunningIndicator();
                    stopPolling();
                }
            }
        }
    }, 1000);
}

// 停止轮询
function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    
    // 恢复UI状态
    document.getElementById('startBtn').disabled = false;
    document.getElementById('cancelBtn').disabled = true;
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

// 重置查询状态
function resetQueryState() {
    // 停止任何正在进行的轮询
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    
    // 隐藏运行指示器
    hideAgentRunningIndicator();
    
    // 重置全局状态
    currentQueryId = null;
    window.networkErrorCount = 0;
    
    // 清空轨迹显示（使用专门的清空函数）
    clearTrajectory();
    
    // 重置UI状态
    document.getElementById('startBtn').disabled = false;
    document.getElementById('cancelBtn').disabled = true;
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
    
    console.log('查询状态已重置');
}

// 更新轨迹显示
function updateTrajectory(result) {
    const trajectory = document.getElementById('trajectory');
    
    // 如果有新步骤，添加它们
    if (result.steps && result.steps.length > currentTrajectory.length) {
        const newSteps = result.steps.slice(currentTrajectory.length);
        newSteps.forEach((step, index) => {
            addStep(step);
        });
        currentTrajectory = result.steps;
        // 新步骤添加完成后，确保滚动到底部
        scrollToBottom(trajectory);
    }
    
    // 如果查询完成，显示最终答案
    if (result.status === 'completed' && result.pred_answer) {
        showFinalAnswer(result.pred_answer);
    } else if (result.status === 'error') {
        showError(result.error_message || '处理过程中出现未知错误');
    }
}

// 平滑并稳健地滚动到容器底部
function scrollToBottom(container) {
    if (!container) return;
    // 使用 requestAnimationFrame 确保在浏览器完成一次布局后再滚动
    window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
            // 优先使用最后一个子元素的 scrollIntoView，避免某些情况下 scrollTop 不生效
            const lastChild = container.lastElementChild;
            if (lastChild && typeof lastChild.scrollIntoView === 'function') {
                lastChild.scrollIntoView({ behavior: 'smooth', block: 'end' });
            } else {
                container.scrollTop = container.scrollHeight;
            }
        });
    });
}

// 添加步骤
function addStep(step) {
    const trajectoryContainer = document.getElementById('trajectory');
    // Remove any existing loading indicator before adding a new step
    const existingLoading = document.getElementById('loadingIndicator');
    if (existingLoading) {
        existingLoading.remove();
    }

    const stepElement = document.createElement('div');
    stepElement.className = `step step-${step.type}`;
    
    const contentId = `content-${step.step_id}`;
    let contentHTML = step.content;

    if (step.content && step.content.length > 300) {
        contentHTML = `<div class="scrollable-content">${step.content.replace(/\n/g, '<br>')}</div>`;
    } else {
        contentHTML = `<div class="step-content">${step.content.replace(/\n/g, '<br>')}</div>`;
    }

    stepElement.innerHTML = `
        <div class="step-header">
            <div class="step-title">
                <span class="step-number">${step.step_id}</span>
                <span>${step.title}</span>
            </div>
            <div class="step-meta">
                <span class="step-timestamp" style="font-size: 16px;">${step.timestamp}</span>
                <span class="step-type">${step.type}</span>
            </div>
        </div>
        ${contentHTML}
    `;
    trajectoryContainer.appendChild(stepElement);
    
    // 如果存在运行指示器，确保它始终在最下方
    const runningIndicator = document.getElementById('agent-running-indicator');
    if (runningIndicator) {
        // 将指示器移动到最后
        trajectoryContainer.appendChild(runningIndicator);
    }
    // 确保滚动到底部
    scrollToBottom(trajectoryContainer);
}

// 显示Agent运行指示器
function showAgentRunningIndicator() {
    console.log('显示Agent运行指示器');
    const trajectoryContainer = document.getElementById('trajectory');
    
    if (!trajectoryContainer) {
        console.error('找不到trajectory容器');
        return;
    }
    
    // 移除已存在的指示器（如果有）
    const existingIndicator = document.getElementById('agent-running-indicator');
    if (existingIndicator) {
        existingIndicator.remove();
    }
    
    // 创建新的运行指示器
    const indicator = document.createElement('div');
    indicator.id = 'agent-running-indicator';
    indicator.className = 'agent-running-indicator';
    indicator.innerHTML = `
        <span class="agent-running-text" style="font-size: 16px;"><strong>Agent正在思考</strong></span>
        <div class="bouncing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    trajectoryContainer.appendChild(indicator);
    console.log('运行指示器已添加到底部');
    // 确保滚动到底部显示指示器
    scrollToBottom(trajectoryContainer);
}

// 隐藏Agent运行指示器
function hideAgentRunningIndicator() {
    console.log('隐藏Agent运行指示器');
    const indicator = document.getElementById('agent-running-indicator');
    if (indicator) {
        indicator.remove();
        console.log('运行指示器已移除');
    } else {
        console.log('没有找到运行指示器');
    }
}




// 格式化内容
function formatContent(content) {
    if (typeof content !== 'string') {
        content = JSON.stringify(content, null, 2);
    }
    
    // 只转义 & 符号以避免XSS，但保留HTML标签
    content = content.replace(/&/g, '&amp;');
    
    // 处理URL链接
    content = content.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" style="color: #4facfe; text-decoration: underline;">$1</a>');
    
    return content;
}

// 显示最终答案
function showFinalAnswer(answer) {
    const trajectory = document.getElementById('trajectory');
    const answerDiv = document.createElement('div');
    answerDiv.className = 'final-answer';
    answerDiv.innerHTML = `
        <h3>🎯 最终答案</h3>
        <div>${formatContent(answer)}</div>
    `;
    trajectory.appendChild(answerDiv);
    answerDiv.scrollIntoView({ behavior: 'smooth' });
    
    // 查询成功完成，确保状态重置（UI状态已由stopPolling处理）
    console.log('查询成功完成，最终答案已显示');
}

// 显示错误
function showError(errorMessage) {
    const trajectory = document.getElementById('trajectory');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'step step-error';
    errorDiv.innerHTML = `
        <div class="step-header">
            <div class="step-title">
                <div class="step-number">❌</div>
                <span>处理出错</span>
            </div>
            <div class="step-type">error</div>
        </div>
        <div class="step-content">
            ${formatContent(errorMessage)}
            <div style="margin-top: 10px;">
                <button onclick="resetQueryState(); this.parentElement.parentElement.parentElement.remove();" 
                        style="background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">
                    关闭并重置
                </button>
            </div>
        </div>
    `;
    trajectory.appendChild(errorDiv);
    errorDiv.scrollIntoView({ behavior: 'smooth' });
    
    // 自动重置状态（但不移除错误信息）
    resetQueryState();
}

// 清空轨迹
function clearTrajectory() {
    document.getElementById('trajectory').innerHTML = `
        <div class="loading" style="display: none;" id="loadingIndicator">
            <div class="spinner"></div>
            等待开始...
        </div>
    `;
    currentTrajectory = [];
}

// 导出轨迹
function exportTrajectory() {
    if (currentTrajectory.length === 0) {
        alert('没有可导出的轨迹');
        return;
    }
    
    const exportData = {
        query: document.getElementById('queryInput').value,
        timestamp: new Date().toISOString(),
        steps: currentTrajectory,
        config: {
            agent_type: document.getElementById('agentType').value,
            prompt_type: document.getElementById('promptType').value,
            max_turns: document.getElementById('maxTurns').value,
            use_jina: document.getElementById('useJina').value,
            temperature: document.getElementById('temperature').value
        }
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trajectory_${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// 保存功能
function openSaveModal() {
    if (currentTrajectory.length === 0) {
        alert('没有可保存的轨迹');
        return;
    }
    
    document.getElementById('saveModal').style.display = 'block';
    document.getElementById('trajectoryName').value = `查询_${new Date().toLocaleString()}`;
    loadSavedTrajectories();
}

function closeSaveModal() {
    document.getElementById('saveModal').style.display = 'none';
}

function saveCurrentTrajectory() {
    const name = document.getElementById('trajectoryName').value.trim();
    const description = document.getElementById('trajectoryDescription').value.trim();
    
    if (!name) {
        alert('请输入轨迹名称');
        return;
    }
    
    const trajectoryData = {
        id: Date.now().toString(),
        name: name,
        description: description,
        query: document.getElementById('queryInput').value,
        timestamp: new Date().toISOString(),
        steps: currentTrajectory,
        config: {
            agent_type: document.getElementById('agentType').value,
            prompt_type: document.getElementById('promptType').value,
            max_turns: document.getElementById('maxTurns').value,
            use_jina: document.getElementById('useJina').value,
            temperature: document.getElementById('temperature').value
        }
    };
    
    // 保存到localStorage
    const savedTrajectories = JSON.parse(localStorage.getItem('savedTrajectories') || '[]');
    savedTrajectories.push(trajectoryData);
    localStorage.setItem('savedTrajectories', JSON.stringify(savedTrajectories));
    
    alert('轨迹保存成功！');
    closeSaveModal();
}

function loadSavedTrajectories() {
    const savedTrajectories = JSON.parse(localStorage.getItem('savedTrajectories') || '[]');
    const container = document.getElementById('savedTrajectories');
    
    if (savedTrajectories.length === 0) {
        container.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">暂无保存的轨迹</p>';
        return;
    }
    
    container.innerHTML = savedTrajectories.map(trajectory => `
        <div class="trajectory-item">
            <div class="trajectory-info">
                <div class="trajectory-name">${trajectory.name}</div>
                <div class="trajectory-meta">
                    查询: ${trajectory.query.substring(0, 50)}${trajectory.query.length > 50 ? '...' : ''}<br>
                    时间: ${new Date(trajectory.timestamp).toLocaleString()}<br>
                    步骤: ${trajectory.steps.length} 个
                </div>
            </div>
            <div class="trajectory-actions">
                <button class="btn btn-secondary btn-small" onclick="loadTrajectory('${trajectory.id}')">
                    📂 加载
                </button>
                <button class="btn btn-secondary btn-small" onclick="downloadTrajectory('${trajectory.id}')">
                    📤 下载
                </button>
                <button class="btn btn-danger btn-small" onclick="deleteTrajectory('${trajectory.id}')">
                    🗑️ 删除
                </button>
            </div>
        </div>
    `).join('');
}

function loadTrajectory(id) {
    const savedTrajectories = JSON.parse(localStorage.getItem('savedTrajectories') || '[]');
    const trajectory = savedTrajectories.find(t => t.id === id);
    
    if (!trajectory) {
        alert('轨迹不存在');
        return;
    }
    
    // 恢复配置
    document.getElementById('queryInput').value = trajectory.query;
    document.getElementById('agentType').value = trajectory.config.agent_type;
    document.getElementById('promptType').value = trajectory.config.prompt_type;
    document.getElementById('maxTurns').value = trajectory.config.max_turns;
    document.getElementById('useJina').value = trajectory.config.use_jina;
    document.getElementById('temperature').value = trajectory.config.temperature;
    
    // 重放轨迹
    clearTrajectory();
    currentTrajectory = trajectory.steps;
    
    trajectory.steps.forEach((step, index) => {
        setTimeout(() => {
            addStep(step);
        }, index * 100);
    });
    
    closeSaveModal();
    alert('轨迹加载成功！');
}

function downloadTrajectory(id) {
    const savedTrajectories = JSON.parse(localStorage.getItem('savedTrajectories') || '[]');
    const trajectory = savedTrajectories.find(t => t.id === id);
    
    if (!trajectory) {
        alert('轨迹不存在');
        return;
    }
    
    const blob = new Blob([JSON.stringify(trajectory, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${trajectory.name}_${trajectory.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function deleteTrajectory(id) {
    if (!confirm('确定要删除这个轨迹吗？')) {
        return;
    }
    
    const savedTrajectories = JSON.parse(localStorage.getItem('savedTrajectories') || '[]');
    const filteredTrajectories = savedTrajectories.filter(t => t.id !== id);
    localStorage.setItem('savedTrajectories', JSON.stringify(filteredTrajectories));
    
    loadSavedTrajectories();
    alert('轨迹删除成功！');
}

// 点击模态框外部关闭
window.onclick = function(event) {
    const saveModal = document.getElementById('saveModal');
    const importModal = document.getElementById('importModal');
    
    if (event.target === saveModal) {
        closeSaveModal();
    }
    
    if (event.target === importModal) {
        closeImportModal();
    }
}

// =================== 导入功能 ===================

// 全局变量存储选择的文件内容
let selectedTrajectoryData = null;

// 打开导入模态框
function openImportModal() {
    document.getElementById('importModal').style.display = 'block';
    clearFileSelection();
}

// 关闭导入模态框
function closeImportModal() {
    document.getElementById('importModal').style.display = 'none';
    clearFileSelection();
}

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) {
        clearFileSelection();
        return;
    }

    // 检查文件类型
    if (!file.name.toLowerCase().endsWith('.json')) {
        alert('请选择JSON格式的文件');
        clearFileSelection();
        return;
    }

    // 读取文件
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const trajectoryData = JSON.parse(e.target.result);
            
            // 验证文件格式
            if (!validateTrajectoryData(trajectoryData)) {
                alert('文件格式不正确，请确保是有效的轨迹JSON文件');
                clearFileSelection();
                return;
            }

            // 存储文件数据
            selectedTrajectoryData = trajectoryData;
            
            // 显示文件预览
            showFilePreview(file, trajectoryData);
            
        } catch (error) {
            console.error('解析JSON文件失败:', error);
            alert('文件格式错误，请确保是有效的JSON文件');
            clearFileSelection();
        }
    };

    reader.onerror = function() {
        alert('读取文件失败，请重试');
        clearFileSelection();
    };

    reader.readAsText(file);
}

// 验证轨迹数据格式
function validateTrajectoryData(data) {
    // 检查必需的字段
    if (!data || typeof data !== 'object') {
        return false;
    }

    // 检查必需的属性
    const requiredFields = ['query', 'steps'];
    for (const field of requiredFields) {
        if (!(field in data)) {
            return false;
        }
    }

    // 检查steps是否是数组
    if (!Array.isArray(data.steps)) {
        return false;
    }

    // 检查每个step的基本格式
    for (const step of data.steps) {
        if (!step || typeof step !== 'object') {
            return false;
        }
        if (!('type' in step) || !('title' in step) || !('content' in step)) {
            return false;
        }
    }

    return true;
}

// 显示文件预览
function showFilePreview(file, trajectoryData) {
    const preview = document.getElementById('filePreview');
    const info = document.getElementById('fileInfo');
    
    // 格式化文件信息
    const fileSize = (file.size / 1024).toFixed(2) + ' KB';
    const stepCount = trajectoryData.steps ? trajectoryData.steps.length : 0;
    const queryText = trajectoryData.query || '未知查询';
    const timestamp = trajectoryData.timestamp ? 
        new Date(trajectoryData.timestamp).toLocaleString() : 
        '未知时间';
    
    info.innerHTML = `
        <div><strong>文件名:</strong> ${file.name}</div>
        <div><strong>文件大小:</strong> ${fileSize}</div>
        <div><strong>查询内容:</strong> ${queryText}</div>
        <div><strong>执行步骤:</strong> ${stepCount} 个</div>
        <div><strong>执行时间:</strong> ${timestamp}</div>
        ${trajectoryData.name ? `<div><strong>轨迹名称:</strong> ${trajectoryData.name}</div>` : ''}
        ${trajectoryData.description ? `<div><strong>描述:</strong> ${trajectoryData.description}</div>` : ''}
    `;
    
    preview.style.display = 'block';
}

// 清除文件选择
function clearFileSelection() {
    document.getElementById('trajectoryFile').value = '';
    document.getElementById('filePreview').style.display = 'none';
    selectedTrajectoryData = null;
}

// 导入并展示轨迹
function importTrajectory() {
    if (!selectedTrajectoryData) {
        alert('请先选择要导入的文件');
        return;
    }

    try {
        // 使用现有的轨迹加载逻辑
        loadTrajectoryFromData(selectedTrajectoryData);
        
        // 关闭模态框
        closeImportModal();
        
        alert('轨迹导入成功！');
        
    } catch (error) {
        console.error('导入轨迹失败:', error);
        alert('导入轨迹失败，请检查文件格式');
    }
}

// 从数据加载轨迹（修改现有的loadTrajectory函数逻辑）
function loadTrajectoryFromData(trajectoryData) {
    // 恢复配置
    if (trajectoryData.query) {
        document.getElementById('queryInput').value = trajectoryData.query;
    }
    
    if (trajectoryData.config) {
        const config = trajectoryData.config;
        if (config.agent_type) document.getElementById('agentType').value = config.agent_type;
        if (config.prompt_type) document.getElementById('promptType').value = config.prompt_type;
        if (config.max_turns) document.getElementById('maxTurns').value = config.max_turns;
        if (config.use_jina !== undefined) document.getElementById('useJina').value = config.use_jina;
        if (config.temperature !== undefined) document.getElementById('temperature').value = config.temperature;
    }
    
    // 重放轨迹
    clearTrajectory();
    currentTrajectory = trajectoryData.steps || [];
    
    // 逐步显示轨迹
    trajectoryData.steps.forEach((step, index) => {
        setTimeout(() => {
            // 兼容旧格式和新格式
            const stepType = step.step_type || step.type;
            addStep(step);
            
            // 如果是最后一步且有最终答案，显示它
            if (index === trajectoryData.steps.length - 1 && trajectoryData.pred_answer) {
                setTimeout(() => {
                    showFinalAnswer(trajectoryData.pred_answer);
                }, 200);
            }
        }, index * 100);
    });
}
