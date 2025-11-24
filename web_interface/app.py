#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flask Web服务器，为多智能体学术写作优化系统提供RESTful API接口
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue
import traceback
import tempfile

# 加载环境变量
from dotenv import load_dotenv

# 尝试加载.env文件（优先级：当前目录 -> 上级目录）
env_paths = [
    Path(__file__).parent / '.env',  # web_interface/.env
    Path(__file__).parent.parent / '.env',  # project_root/.env
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# 导入主要的多智能体系统
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from multi_agent_nlp_project import (
    DualAgentAcademicSystem,
    init_llm, 
    TOOLS, 
    vectorstore,
    optimize_text_file,
    parse_requirements,
    load_seeds_from_file,
    generate_html_report
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB文件上传限制

# 启用CORS和SocketIO
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局存储处理任务和结果
active_tasks = {}
task_results = {}


class TaskManager:
    """任务管理器，用于跟踪长时间运行的任务"""
    
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()
    
    def create_task(self, task_type: str, params: Dict) -> str:
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = {
                'id': task_id,
                'type': task_type,
                'params': params,
                'status': 'created',
                'progress': 0,
                'result': None,
                'error': None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
        return task_id
    
    def update_task(self, task_id: str, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
                self.tasks[task_id]['updated_at'] = datetime.now().isoformat()
                
                # 发送WebSocket更新到所有连接的客户端
                update_data = self.tasks[task_id].copy()
                
                # 如果有轮次结果，单独发送
                if 'round_result' in kwargs:
                    socketio.emit('round_update', {
                        'task_id': task_id,
                        'round_data': kwargs['round_result']
                    })
                
                # 发送任务状态更新
                socketio.emit('task_update', update_data)
                
                # 输出到控制台以便调试
                if 'message' in kwargs:
                    logger.info(f'Task {task_id[:8]}: {kwargs["message"]}')
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.lock:
            return self.tasks.get(task_id)
    
    def delete_task(self, task_id: str):
        with self.lock:
            self.tasks.pop(task_id, None)


task_manager = TaskManager()


def run_text_optimization_task(task_id: str, text: str, requirements: List[str], 
                             rounds: int = 3, enable_tools: bool = True, 
                             enable_memory: bool = True, language: str = 'zh'):
    """运行文本优化任务"""
    try:
        task_manager.update_task(task_id, status='running', progress=10, message='初始化智能体系统...')
        
        # 自定义的智能体系统，支持实时更新
        class RealTimeAgentSystem(DualAgentAcademicSystem):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.task_id = task_id
            
            def collaborate(self, user_text: str, user_requirements: List[str], language: str = "中文", rounds: int = 3):
                task_manager.update_task(self.task_id, progress=20, message=f'开始{rounds}轮协作优化...')
                
                self.collaboration_log = [{"round": 0, "user_input": user_text, "requirements": user_requirements, "timestamp": datetime.now().isoformat()}]
                current_text = user_text
                previous_feedback = ""
                last_scores = {}
                
                if self.memory_enabled:
                    self.memory.add_memory(user_text, {"type": "user_input"})
                
                for r in range(1, rounds + 1):
                    # 更新进度
                    progress = 20 + (r / rounds) * 70  # 20-90%的进度
                    task_manager.update_task(self.task_id, progress=progress, message=f'正在进行第{r}轮优化...')
                    
                    mem_snippets = []
                    if self.memory_enabled:
                        mem_snippets = self.memory.recall(current_text, k=3)
                    
                    tool_obs = self._plan_and_act(current_text, user_requirements)
                    
                    # Agent A 优化
                    task_manager.update_task(self.task_id, message=f'第{r}轮 - Agent A 正在优化文本...')
                    a_input = {
                        "round_num": r,
                        "text_to_optimize": current_text,
                        "user_requirements": ', '.join(user_requirements),
                        "previous_feedback": previous_feedback,
                        "memory_snippets": '\n'.join(mem_snippets) if mem_snippets else "(无)",
                        "tool_observations": tool_obs,
                        "last_scores": last_scores if last_scores else "(无)"
                    }
                    a_resp = self.agent_a_chain.invoke(a_input)
                    optimized_text = self._extract_section(a_resp, "**优化版本：**", "**修改说明：**") or current_text
                    
                    # Agent B 评审
                    task_manager.update_task(self.task_id, message=f'第{r}轮 - Agent B 正在评审...')
                    b_input = {
                        "round_num": r,
                        "optimized_text": optimized_text,
                        "user_requirements": ', '.join(user_requirements)
                    }
                    b_resp = self.agent_b_chain.invoke(b_input)
                    last_scores = self._parse_scores(b_resp)
                    
                    diff_str = self._compute_diff(current_text, optimized_text)
                    
                    if self.memory_enabled:
                        self.memory.add_memory(optimized_text, {"type": "optimized_text", "round": r})
                        self.memory.add_memory(b_resp, {"type": "feedback", "round": r})
                    
                    round_log = {
                        "round": r,
                        "agent_a_response": a_resp,
                        "optimized_text": optimized_text,
                        "agent_b_feedback": b_resp,
                        "scores": last_scores,
                        "tool_observations": tool_obs,
                        "diff": diff_str,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.collaboration_log.append(round_log)
                    
                    # 发送轮次完成更新
                    task_manager.update_task(
                        self.task_id, 
                        progress=progress,
                        message=f'第{r}轮完成 | 评分: {last_scores}',
                        round_result=round_log
                    )
                    
                    previous_feedback = b_resp
                    current_text = optimized_text
                    print(f"✅ Round {r} 完成 | 评分: {last_scores if last_scores else '{}'}")
                    time.sleep(0.15)
                
                task_manager.update_task(self.task_id, progress=95, message='准备最终结果...')
                return current_text, self.collaboration_log
        
        # 初始化实时智能体系统
        system = RealTimeAgentSystem(
            init_llm(), TOOLS, vectorstore,
            enable_tools=enable_tools,
            enable_memory=enable_memory
        )
        
        task_manager.update_task(task_id, progress=15, message='系统初始化完成，开始优化...')
        
        # 执行优化
        final_text, log = system.collaborate(text, requirements, language, rounds)
        
        # 完成任务
        task_manager.update_task(
            task_id, 
            status='completed', 
            progress=100,
            message='优化完成！',
            result={
                'final_text': final_text,
                'log': log,
                'original_text': text,
                'requirements': requirements
            }
        )
        
    except Exception as e:
        logger.error(f"Text optimization task {task_id} failed: {e}")
        logger.error(traceback.format_exc())
        task_manager.update_task(
            task_id,
            status='failed',
            error=str(e)
        )


def run_file_optimization_task(task_id: str, file_content: str, requirements: List[str],
                             rounds: int = 3, chunk_size: int = 5000, 
                             overlap: int = 200, max_chunks: int = 0,
                             enable_tools: bool = True, enable_memory: bool = True):
    """运行文件优化任务"""
    try:
        task_manager.update_task(task_id, status='running', progress=10)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            system = DualAgentAcademicSystem(
                init_llm(), TOOLS, vectorstore,
                enable_tools=enable_tools,
                enable_memory=enable_memory
            )
            
            task_manager.update_task(task_id, progress=20)
            
            # 执行文件优化
            final_text, aggregated = optimize_text_file(
                system, temp_path, requirements, rounds, 
                chunk_size, overlap, max_chunks
            )
            
            task_manager.update_task(
                task_id,
                status='completed',
                progress=100,
                result={
                    'final_text': final_text,
                    'aggregated': aggregated,
                    'original_text': file_content,
                    'requirements': requirements
                }
            )
            
        finally:
            # 清理临时文件
            os.unlink(temp_path)
            
    except Exception as e:
        logger.error(f"File optimization task {task_id} failed: {e}")
        logger.error(traceback.format_exc())
        task_manager.update_task(
            task_id,
            status='failed',
            error=str(e)
        )


def run_synthesis_task(task_id: str, seeds: List[str], requirements: List[str], rounds: int = 3):
    """运行数据合成任务"""
    try:
        task_manager.update_task(task_id, status='running', progress=10)
        
        system = DualAgentAcademicSystem(init_llm(), TOOLS, vectorstore)
        
        task_manager.update_task(task_id, progress=20)
        
        # 执行数据合成
        output_path = system.synthesize_dataset(seeds, requirements, rounds)
        
        task_manager.update_task(
            task_id,
            status='completed',
            progress=100,
            result={
                'output_path': str(output_path),
                'seeds_count': len(seeds),
                'requirements': requirements
            }
        )
        
    except Exception as e:
        logger.error(f"Synthesis task {task_id} failed: {e}")
        logger.error(traceback.format_exc())
        task_manager.update_task(
            task_id,
            status='failed',
            error=str(e)
        )


def run_evaluation_task(task_id: str, test_cases: List[tuple], rounds: int = 2):
    """运行评估任务"""
    try:
        task_manager.update_task(task_id, status='running', progress=10)
        
        system = DualAgentAcademicSystem(init_llm(), TOOLS, vectorstore)
        
        task_manager.update_task(task_id, progress=20)
        
        # 执行评估
        report = system.evaluate(test_cases, rounds)
        
        task_manager.update_task(
            task_id,
            status='completed',
            progress=100,
            result=report
        )
        
    except Exception as e:
        logger.error(f"Evaluation task {task_id} failed: {e}")
        logger.error(traceback.format_exc())
        task_manager.update_task(
            task_id,
            status='failed',
            error=str(e)
        )


@app.route('/')
def index():
    """提供主页面"""
    return send_from_directory('.', 'index.html')


@app.route('/api/config', methods=['POST'])
def update_config():
    """更新系统配置"""
    try:
        config = request.get_json()
        
        # 更新环境变量（仅在当前会话中有效）
        if 'openai_api_key' in config:
            os.environ['OPENAI_API_KEY'] = config['openai_api_key']
        if 'openai_base_url' in config:
            os.environ['OPENAI_BASE_URL'] = config['openai_base_url']
        if 'llm_model' in config:
            os.environ['LLM_MODEL'] = config['llm_model']
        if 'serpapi_api_key' in config:
            os.environ['SERPAPI_API_KEY'] = config['serpapi_api_key']
        
        return jsonify({'status': 'success', 'message': '配置已更新'})
    
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/optimize/text', methods=['POST'])
def optimize_text():
    """文本优化API"""
    try:
        data = request.get_json()
        
        text = data.get('text', '').strip()
        requirements_str = data.get('requirements', '学术表达提升,逻辑结构优化')
        rounds = int(data.get('rounds', 3))
        enable_tools = data.get('enable_tools', True)
        enable_memory = data.get('enable_memory', True)
        language = data.get('language', 'zh')
        
        if not text:
            return jsonify({'status': 'error', 'message': '文本不能为空'}), 400
        
        requirements = parse_requirements(requirements_str, ['学术表达提升'])
        
        # 创建任务
        task_id = task_manager.create_task('text_optimization', {
            'text': text,
            'requirements': requirements,
            'rounds': rounds,
            'enable_tools': enable_tools,
            'enable_memory': enable_memory,
            'language': language
        })
        
        # 启动后台任务
        thread = threading.Thread(
            target=run_text_optimization_task,
            args=(task_id, text, requirements, rounds, enable_tools, enable_memory, language)
        )
        thread.start()
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': '文本优化任务已启动'
        })
        
    except Exception as e:
        logger.error(f"Text optimization failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/optimize/file', methods=['POST'])
def optimize_file():
    """文件优化API"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '没有选择文件'}), 400
        
        # 读取文件内容
        file_content = file.read().decode('utf-8')
        
        requirements_str = request.form.get('requirements', '学术表达提升,逻辑结构优化')
        rounds = int(request.form.get('rounds', 3))
        chunk_size = int(request.form.get('chunk_size', 5000))
        overlap = int(request.form.get('overlap', 200))
        max_chunks = int(request.form.get('max_chunks', 0))
        enable_tools = request.form.get('enable_tools', 'true').lower() == 'true'
        enable_memory = request.form.get('enable_memory', 'true').lower() == 'true'
        
        requirements = parse_requirements(requirements_str, ['学术表达提升'])
        
        # 创建任务
        task_id = task_manager.create_task('file_optimization', {
            'file_content': file_content,
            'requirements': requirements,
            'rounds': rounds,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'max_chunks': max_chunks,
            'enable_tools': enable_tools,
            'enable_memory': enable_memory
        })
        
        # 启动后台任务
        thread = threading.Thread(
            target=run_file_optimization_task,
            args=(task_id, file_content, requirements, rounds, chunk_size, overlap, max_chunks, enable_tools, enable_memory)
        )
        thread.start()
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': '文件优化任务已启动'
        })
        
    except Exception as e:
        logger.error(f"File optimization failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
def synthesize_data():
    """数据合成API"""
    try:
        data = request.get_json()
        
        seeds_text = data.get('seeds', '').strip()
        seeds = [line.strip() for line in seeds_text.split('\n') if line.strip()]
        
        if not seeds:
            return jsonify({'status': 'error', 'message': '种子文本不能为空'}), 400
        
        requirements_str = data.get('requirements', '学术表达提升,结构清晰,可读性增强')
        rounds = int(data.get('rounds', 3))
        
        requirements = parse_requirements(requirements_str, ['学术表达提升'])
        
        # 创建任务
        task_id = task_manager.create_task('synthesis', {
            'seeds': seeds,
            'requirements': requirements,
            'rounds': rounds
        })
        
        # 启动后台任务
        thread = threading.Thread(
            target=run_synthesis_task,
            args=(task_id, seeds, requirements, rounds)
        )
        thread.start()
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': '数据合成任务已启动'
        })
        
    except Exception as e:
        logger.error(f"Data synthesis failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_system():
    """评估API"""
    try:
        data = request.get_json()
        
        cases_text = data.get('cases', '').strip()
        if not cases_text:
            return jsonify({'status': 'error', 'message': '测试用例不能为空'}), 400
        
        # 解析测试用例
        test_cases = []
        for line in cases_text.split('\n'):
            if '|||' in line:
                text, reqs_str = line.split('|||', 1)
                reqs = parse_requirements(reqs_str.strip(), ['严谨性'])
                test_cases.append((text.strip(), reqs))
        
        if not test_cases:
            return jsonify({'status': 'error', 'message': '没有有效的测试用例'}), 400
        
        rounds = int(data.get('rounds', 2))
        
        # 创建任务
        task_id = task_manager.create_task('evaluation', {
            'test_cases': test_cases,
            'rounds': rounds
        })
        
        # 启动后台任务
        thread = threading.Thread(
            target=run_evaluation_task,
            args=(task_id, test_cases, rounds)
        )
        thread.start()
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': '评估任务已启动'
        })
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/distill', methods=['POST'])
def distill_data():
    """数据蒸馏API"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传JSONL文件'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.jsonl'):
            return jsonify({'status': 'error', 'message': '请上传.jsonl格式文件'}), 400
        
        # 读取JSONL文件
        content = file.read().decode('utf-8')
        
        # 创建临时文件进行处理
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(content)
            input_path = temp_file.name
        
        try:
            output_filename = request.form.get('output_filename', 'distill_pairs.jsonl')
            output_path = Path('data') / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            system = DualAgentAcademicSystem(init_llm(), TOOLS, vectorstore)
            result_path = system.prepare_distillation_pairs(Path(input_path), output_path)
            
            return jsonify({
                'status': 'success',
                'message': f'蒸馏数据已生成: {result_path}',
                'output_path': str(result_path)
            })
            
        finally:
            os.unlink(input_path)
        
    except Exception as e:
        logger.error(f"Data distillation failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if not task:
        return jsonify({'status': 'error', 'message': '任务不存在'}), 404
    
    return jsonify(task)


@app.route('/api/download/<task_id>/text')
def download_optimized_text(task_id: str):
    """下载优化后的文本"""
    task = task_manager.get_task(task_id)
    if not task or not task.get('result'):
        return jsonify({'status': 'error', 'message': '结果不存在'}), 404
    
    final_text = task['result']['final_text']
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(final_text)
        temp_path = temp_file.name
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f'optimized_text_{task_id[:8]}.txt',
        mimetype='text/plain'
    )


@app.route('/api/download/<task_id>/html')
def download_html_report(task_id: str):
    """下载HTML报告"""
    task = task_manager.get_task(task_id)
    if not task or not task.get('result'):
        return jsonify({'status': 'error', 'message': '结果不存在'}), 404
    
    result = task['result']
    final_text = result['final_text']
    
    if 'log' in result:
        log = result['log']
        title = '文本优化报告'
    else:
        log = result.get('aggregated', {}).get('segments', [])
        title = '文件优化报告'
    
    html = generate_html_report(title, final_text, log)
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(html)
        temp_path = temp_file.name
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f'report_{task_id[:8]}.html',
        mimetype='text/html'
    )


@app.route('/api/download/<task_id>/json')
def download_json_data(task_id: str):
    """下载JSON数据"""
    task = task_manager.get_task(task_id)
    if not task or not task.get('result'):
        return jsonify({'status': 'error', 'message': '结果不存在'}), 404
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
        json.dump(task['result'], temp_file, ensure_ascii=False, indent=2)
        temp_path = temp_file.name
    
    return send_file(
        temp_path,
        as_attachment=True,
        download_name=f'data_{task_id[:8]}.json',
        mimetype='application/json'
    )


@socketio.on('connect')
def handle_connect():
    """WebSocket连接处理"""
    logger.info(f'Client connected: {request.sid}')
    emit('connected', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """WebSocket断开连接处理"""
    logger.info(f'Client disconnected: {request.sid}')


@socketio.on('join_task')
def handle_join_task(data):
    """加入任务房间以接收更新"""
    task_id = data.get('task_id')
    if task_id:
        logger.info(f'Client {request.sid} joined task {task_id}')
        # 可以使用room功能实现更精确的消息推送
        # join_room(task_id)


if __name__ == '__main__':
    # 确保必要的目录存在
    Path('data').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    
    print("🚀 多智能体学术优化系统Web服务器启动中...")
    print("📝 访问地址: http://localhost:5000")
    
    # 启动SocketIO服务器
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        use_reloader=False  # 避免在多线程环境下重载问题
    )