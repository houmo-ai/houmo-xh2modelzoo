import openpyxl
import json
import uuid

wb = openpyxl.load_workbook('Model Zoo.xlsx')
ws = wb['支持模型列表']

models = []

with open('dashboard_data.json', 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

for row in ws.iter_rows(min_row=2, values_only=True):
    model_name = row[0]
    if not model_name or model_name == 'AAA-模型名称':
        continue
    
    model_type = row[1] or '其他'
    quantization_status = row[2] or '未量化'
    algorithm_doc = row[3] or ''
    jira_link = row[4] or ''
    owner = row[5] or '未分配'
    chip_type = row[6] or '其他'
    
    model = {
        'id': str(uuid.uuid4()),
        'name': model_name,
        'type': model_type,
        'accuracy': 90.0,
        'inference_time': 50,
        'memory_usage': 8,
        'tests_passed': 10,
        'tests_failed': 0,
        'status': '正常',
        'attributes': {
            '模型类型': model_type,
            '量化状态': quantization_status,
            '算法文档': algorithm_doc,
            'jira链接': jira_link,
            '负责人': owner,
            '芯片类型': chip_type,
            '精度': 'FP16'
        }
    }
    models.append(model)

existing_data['models'] = models

with open('dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=2)

print(f'成功导入 {len(models)} 个模型数据到 dashboard_data.json')
print(f'\n模型类型统计:')
model_types = {}
for model in models:
    model_type = model['attributes']['模型类型']
    model_types[model_type] = model_types.get(model_type, 0) + 1
for model_type, count in sorted(model_types.items()):
    print(f'  {model_type}: {count}个')

print(f'\n芯片类型统计:')
chip_types = {}
for model in models:
    chip_type = model['attributes']['芯片类型']
    chip_types[chip_type] = chip_types.get(chip_type, 0) + 1
for chip_type, count in sorted(chip_types.items()):
    print(f'  {chip_type}: {count}个')

print(f'\n量化状态统计:')
quantization_statuses = {}
for model in models:
    status = model['attributes']['量化状态']
    quantization_statuses[status] = quantization_statuses.get(status, 0) + 1
for status, count in sorted(quantization_statuses.items()):
    print(f'  {status}: {count}个')