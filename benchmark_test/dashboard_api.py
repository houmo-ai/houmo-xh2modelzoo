import json
import os
import re
from flask import Blueprint, jsonify, request, render_template_string
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

ABS_PATH = os.path.dirname(__file__)
DASHBOARD_DATA_FILE = os.path.join(ABS_PATH, 'dashboard_data.json')
EMAIL_REPORT_FILE = os.path.join(ABS_PATH, 'email_report.html')

def load_dashboard_data():
    """加载看板数据"""
    try:
        with open(DASHBOARD_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"dashboards": [], "tags": [], "models": []}
    except json.JSONDecodeError:
        return {"dashboards": [], "tags": [], "models": []}

def save_dashboard_data(data):
    """保存看板数据"""
    try:
        with open(DASHBOARD_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存看板数据失败: {str(e)}")
        return False

def parse_email_report():
    """解析email_report.html文件，提取测试结果"""
    test_results = {}
    
    try:
        with open(EMAIL_REPORT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        h4_pattern = r'<h4[^>]*>(.*?)</h4>'
        h4_matches = list(re.finditer(h4_pattern, content, re.DOTALL))
        
        for h4_match in h4_matches:
            model_name = h4_match.group(1).strip()
            table_content = content[h4_match.end():h4_match.end()+2000]
            
            status = None
            if 'passed' in table_content.lower():
                status = 'passed'
            elif 'failed' in table_content.lower() or 'broken' in table_content.lower():
                status = 'failed'
            
            if status:
                if model_name not in test_results:
                    test_results[model_name] = {'passed': 0, 'failed': 0}
                
                if status == 'passed':
                    test_results[model_name]['passed'] = 1
                elif status == 'failed':
                    test_results[model_name]['failed'] = 1
            
    except FileNotFoundError:
        print(f"警告: {EMAIL_REPORT_FILE} 文件不存在")
    except Exception as e:
        print(f"解析email_report失败: {str(e)}")
    
    return test_results

@dashboard_bp.route('/dashboard')
def dashboard_page():
    """看板页面"""
    return render_template_string(open(os.path.join(os.path.dirname(__file__), 'dashboard.html'), 'r', encoding='utf-8').read())

@dashboard_bp.route('/api/dashboard/data', methods=['GET'])
def get_dashboard_data():
    """获取看板数据"""
    data = load_dashboard_data()
    
    test_results = parse_email_report()
    
    models = data.get('models', [])
    
    for model in models:
        model['tests_passed'] = 0
        model['tests_failed'] = 0
    
    for model in models:
        model_name = model.get('name', '')
        
        for test_name, test_result in test_results.items():
            test_name_clean = test_name.replace('测试', '').strip()
            test_name_normalized = test_name_clean.replace('-', '').replace('_', '').replace(' ', '').lower()
            model_name_normalized = model_name.replace('-', '').replace('_', '').replace(' ', '').lower()
            
            if model_name == test_name or model_name == test_name_clean or \
               model_name_normalized == test_name_normalized or \
               model_name_normalized in test_name_normalized or \
               test_name_normalized in model_name_normalized:
                model['tests_passed'] = test_result['passed']
                model['tests_failed'] = test_result['failed']
                
                if test_result['failed'] > 0:
                    model['status'] = '异常'
                
                break
    
    return jsonify(data)

@dashboard_bp.route('/api/dashboard/dashboards', methods=['GET'])
def get_dashboards():
    """获取所有看板列表"""
    data = load_dashboard_data()
    return jsonify(data.get('dashboards', []))

@dashboard_bp.route('/api/dashboard/dashboards/<dashboard_id>', methods=['GET'])
def get_dashboard(dashboard_id):
    """获取指定看板"""
    data = load_dashboard_data()
    dashboard = next((d for d in data.get('dashboards', []) if d['id'] == dashboard_id), None)
    if dashboard:
        return jsonify(dashboard)
    return jsonify({'error': 'Dashboard not found'}), 404

@dashboard_bp.route('/api/dashboard/dashboards', methods=['POST'])
def create_dashboard():
    """创建新看板"""
    data = load_dashboard_data()
    new_dashboard = request.json
    
    if not new_dashboard.get('id'):
        return jsonify({'error': 'Dashboard ID is required'}), 400
    
    if any(d['id'] == new_dashboard['id'] for d in data.get('dashboards', [])):
        return jsonify({'error': 'Dashboard ID already exists'}), 400
    
    new_dashboard.setdefault('name', '新看板')
    new_dashboard.setdefault('description', '')
    new_dashboard.setdefault('tags', [])
    new_dashboard.setdefault('is_default', False)
    new_dashboard.setdefault('charts', [])
    
    data.setdefault('dashboards', []).append(new_dashboard)
    save_dashboard_data(data)
    
    return jsonify(new_dashboard), 201

@dashboard_bp.route('/api/dashboard/dashboards/<dashboard_id>', methods=['PUT'])
def update_dashboard(dashboard_id):
    """更新看板"""
    data = load_dashboard_data()
    dashboards = data.get('dashboards', [])
    
    for i, dashboard in enumerate(dashboards):
        if dashboard['id'] == dashboard_id:
            updated_data = request.json
            dashboards[i].update(updated_data)
            save_dashboard_data(data)
            return jsonify(dashboards[i])
    
    return jsonify({'error': 'Dashboard not found'}), 404

@dashboard_bp.route('/api/dashboard/dashboards/<dashboard_id>', methods=['DELETE'])
def delete_dashboard(dashboard_id):
    """删除看板"""
    data = load_dashboard_data()
    dashboards = data.get('dashboards', [])
    
    original_length = len(dashboards)
    data['dashboards'] = [d for d in dashboards if d['id'] != dashboard_id]
    
    if len(data['dashboards']) < original_length:
        save_dashboard_data(data)
        return jsonify({'message': 'Dashboard deleted successfully'})
    
    return jsonify({'error': 'Dashboard not found'}), 404

@dashboard_bp.route('/api/dashboard/tags', methods=['GET'])
def get_tags():
    """获取所有标签"""
    data = load_dashboard_data()
    return jsonify(data.get('tags', []))

@dashboard_bp.route('/api/dashboard/tags', methods=['POST'])
def create_tag():
    """创建新标签"""
    data = load_dashboard_data()
    new_tag = request.json
    
    if not new_tag.get('id') or not new_tag.get('name'):
        return jsonify({'error': 'Tag ID and name are required'}), 400
    
    if any(t['id'] == new_tag['id'] for t in data.get('tags', [])):
        return jsonify({'error': 'Tag ID already exists'}), 400
    
    new_tag.setdefault('color', '#007bff')
    data.setdefault('tags', []).append(new_tag)
    save_dashboard_data(data)
    
    return jsonify(new_tag), 201

@dashboard_bp.route('/api/dashboard/models', methods=['GET'])
def get_models():
    """获取所有模型"""
    data = load_dashboard_data()
    models = data.get('models', [])
    
    tag_filter = request.args.get('tags')
    if tag_filter:
        filter_tags = tag_filter.split(',')
        models = [m for m in models if any(tag in m.get('tags', []) for tag in filter_tags)]
    
    return jsonify(models)

@dashboard_bp.route('/api/dashboard/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """获取指定模型"""
    data = load_dashboard_data()
    model = next((m for m in data.get('models', []) if m['id'] == model_id), None)
    if model:
        return jsonify(model)
    return jsonify({'error': 'Model not found'}), 404

@dashboard_bp.route('/api/dashboard/models', methods=['POST'])
def create_model():
    """创建新模型"""
    data = load_dashboard_data()
    new_model = request.json
    
    if not new_model.get('id'):
        return jsonify({'error': 'Model ID is required'}), 400
    
    if any(m['id'] == new_model['id'] for m in data.get('models', [])):
        return jsonify({'error': 'Model ID already exists'}), 400
    
    new_model.setdefault('name', '新模型')
    new_model.setdefault('category', '未分类')
    new_model.setdefault('tags', [])
    new_model.setdefault('metrics', {})
    new_model.setdefault('last_test_date', datetime.now().strftime('%Y-%m-%d'))
    new_model.setdefault('status', 'active')
    
    data.setdefault('models', []).append(new_model)
    save_dashboard_data(data)
    
    return jsonify(new_model), 201

@dashboard_bp.route('/api/dashboard/models/<model_id>', methods=['PUT'])
def update_model(model_id):
    """更新模型"""
    data = load_dashboard_data()
    models = data.get('models', [])
    
    for i, model in enumerate(models):
        if model['id'] == model_id:
            updated_data = request.json
            models[i].update(updated_data)
            save_dashboard_data(data)
            return jsonify(models[i])
    
    return jsonify({'error': 'Model not found'}), 404

@dashboard_bp.route('/api/dashboard/models', methods=['PUT'])
def update_models():
    """批量更新模型属性"""
    data = load_dashboard_data()
    models = data.get('models', [])
    model_updates = request.json
    
    print(f"收到模型更新请求: {model_updates}")
    
    if not isinstance(model_updates, dict):
        print(f"错误: 无效的模型更新格式，期望dict，收到 {type(model_updates)}")
        return jsonify({'error': 'Invalid model updates format'}), 400
    
    updated_models = []
    for model_id, updates in model_updates.items():
        print(f"处理模型 {model_id} 的更新: {updates}")
        for i, model in enumerate(models):
            if model['id'] == model_id:
                print(f"找到模型 {model_id}，当前属性: {model.get('attributes', {})}")
                if 'attributes' in updates:
                    if not models[i].get('attributes'):
                        models[i]['attributes'] = {}
                    models[i]['attributes'].update(updates['attributes'])
                    print(f"更新后的属性: {models[i]['attributes']}")
                else:
                    models[i].update(updates)
                updated_models.append(models[i])
                break
    
    print(f"成功更新 {len(updated_models)} 个模型")
    
    if save_dashboard_data(data):
        print("数据保存成功")
        return jsonify({'message': 'Models updated successfully', 'updated_count': len(updated_models)})
    else:
        print("数据保存失败")
        return jsonify({'error': 'Failed to save data'}), 500

@dashboard_bp.route('/api/dashboard/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    data = load_dashboard_data()
    models = data.get('models', [])
    
    original_length = len(models)
    data['models'] = [m for m in models if m['id'] != model_id]
    
    if len(data['models']) < original_length:
        save_dashboard_data(data)
        return jsonify({'message': 'Model deleted successfully'})
    
    return jsonify({'error': 'Model not found'}), 404

@dashboard_bp.route('/api/dashboard/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    data = load_dashboard_data()
    models = data.get('models', [])
    
    stats = {
        'total_models': len(models),
        'active_models': len([m for m in models if m.get('status') == 'active']),
        'total_tests_passed': sum(m.get('metrics', {}).get('test_passed', 0) for m in models),
        'total_tests_failed': sum(m.get('metrics', {}).get('test_failed', 0) for m in models),
        'average_accuracy': 0,
        'categories': {}
    }
    
    if models:
        total_accuracy = sum(m.get('metrics', {}).get('accuracy', 0) for m in models)
        stats['average_accuracy'] = round(total_accuracy / len(models), 2)
    
    for model in models:
        category = model.get('category', '未分类')
        stats['categories'].setdefault(category, 0)
        stats['categories'][category] += 1
    
    return jsonify(stats)

@dashboard_bp.route('/api/dashboard/charts/<dashboard_id>/<chart_id>', methods=['GET'])
def get_chart_data(dashboard_id, chart_id):
    """获取指定看板的图表数据"""
    data = load_dashboard_data()
    dashboard = next((d for d in data.get('dashboards', []) if d['id'] == dashboard_id), None)
    
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    chart = next((c for c in dashboard.get('charts', []) if c['id'] == chart_id), None)
    
    if chart:
        return jsonify(chart)
    
    return jsonify({'error': 'Chart not found'}), 404

@dashboard_bp.route('/api/dashboard/charts/<dashboard_id>', methods=['POST'])
def create_chart(dashboard_id):
    """为指定看板创建新图表"""
    data = load_dashboard_data()
    dashboard = next((d for d in data.get('dashboards', []) if d['id'] == dashboard_id), None)
    
    if not dashboard:
        return jsonify({'error': 'Dashboard not found'}), 404
    
    new_chart = request.json
    
    if not new_chart.get('id'):
        return jsonify({'error': 'Chart ID is required'}), 400
    
    if any(c['id'] == new_chart['id'] for c in dashboard.get('charts', [])):
        return jsonify({'error': 'Chart ID already exists'}), 400
    
    new_chart.setdefault('title', '新图表')
    new_chart.setdefault('type', 'bar')
    new_chart.setdefault('data', {'labels': [], 'datasets': []})
    
    dashboard.setdefault('charts', []).append(new_chart)
    save_dashboard_data(data)
    
    return jsonify(new_chart), 201

@dashboard_bp.route('/api/dashboard/reset', methods=['POST'])
def reset_dashboard_data():
    """重置看板数据到初始状态"""
    from dashboard_data_generator import generate_initial_data
    
    initial_data = generate_initial_data()
    save_dashboard_data(initial_data)
    
    return jsonify({'message': 'Dashboard data reset successfully'})

@dashboard_bp.route('/api/dashboard/tag-values', methods=['GET'])
def get_tag_values():
    """获取所有标签值"""
    data = load_dashboard_data()
    return jsonify(data.get('tag_values', {}))

@dashboard_bp.route('/api/dashboard/tag-values', methods=['PUT'])
def update_tag_values():
    """更新标签值"""
    data = load_dashboard_data()
    new_tag_values = request.json
    
    if not isinstance(new_tag_values, dict):
        return jsonify({'error': 'Invalid tag values format'}), 400
    
    data['tag_values'] = new_tag_values
    save_dashboard_data(data)
    
    return jsonify({'message': 'Tag values updated successfully', 'tag_values': new_tag_values})

@dashboard_bp.route('/api/dashboard/tag-values/<tag_name>', methods=['GET'])
def get_tag_value(tag_name):
    """获取指定标签的值"""
    data = load_dashboard_data()
    tag_values = data.get('tag_values', {})
    return jsonify(tag_values.get(tag_name, []))

@dashboard_bp.route('/api/dashboard/tag-values/<tag_name>', methods=['PUT'])
def update_tag_value(tag_name):
    """更新指定标签的值"""
    data = load_dashboard_data()
    tag_values = data.get('tag_values', {})
    
    new_values = request.json
    if not isinstance(new_values, list):
        return jsonify({'error': 'Invalid tag value format'}), 400
    
    tag_values[tag_name] = new_values
    data['tag_values'] = tag_values
    save_dashboard_data(data)
    
    return jsonify({'message': 'Tag value updated successfully', 'tag_name': tag_name, 'values': new_values})

@dashboard_bp.route('/api/dashboard/tag-values/<tag_name>', methods=['DELETE'])
def delete_tag_value(tag_name):
    """删除指定标签的值"""
    data = load_dashboard_data()
    tag_values = data.get('tag_values', {})
    
    if tag_name in tag_values:
        del tag_values[tag_name]
        data['tag_values'] = tag_values
        save_dashboard_data(data)
        return jsonify({'message': 'Tag value deleted successfully'})
    
    return jsonify({'error': 'Tag value not found'}), 404