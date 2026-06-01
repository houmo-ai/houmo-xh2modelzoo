import json
import random

with open('dashboard_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

models = data['models']

for model in models:
    model_type = model.get('attributes', {}).get('模型类型', '其他')
    
    if model_type in ['文本生成', '多模态', 'VLA']:
        model['accuracy'] = round(random.uniform(85, 95), 1)
        model['inference_time'] = random.randint(80, 200)
        model['memory_usage'] = random.randint(16, 64)
    elif model_type in ['CV', '检测', '分割', '开放世界目标检测']:
        model['accuracy'] = round(random.uniform(88, 98), 1)
        model['inference_time'] = random.randint(20, 60)
        model['memory_usage'] = random.randint(4, 16)
    elif model_type in ['OCR', '图像特征提取', '图像生成', '超分']:
        model['accuracy'] = round(random.uniform(90, 97), 1)
        model['inference_time'] = random.randint(30, 80)
        model['memory_usage'] = random.randint(2, 12)
    elif model_type in ['embedding', '光流模型', '深度估计']:
        model['accuracy'] = round(random.uniform(85, 95), 1)
        model['inference_time'] = random.randint(40, 100)
        model['memory_usage'] = random.randint(4, 20)
    else:
        model['accuracy'] = round(random.uniform(85, 95), 1)
        model['inference_time'] = random.randint(30, 90)
        model['memory_usage'] = random.randint(4, 24)
    
    model['tests_passed'] = random.randint(5, 20)
    model['tests_failed'] = random.randint(0, 3)

top_models = sorted(models, key=lambda x: x.get('accuracy', 0), reverse=True)[:10]

model_names = [model['name'] for model in top_models]
accuracies = [model.get('accuracy', 0) for model in top_models]
inference_times = [model.get('inference_time', 0) for model in top_models]

colors = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
    "#FF9F40", "#FF6384", "#C9CBCF", "#4BC0C0", "#FF6384"
]

for dashboard in data['dashboards']:
    if dashboard['id'] == 'model_performance':
        for chart in dashboard['charts']:
            if chart['id'] == 'accuracy_comparison':
                chart['data']['labels'] = model_names
                chart['data']['datasets'][0]['data'] = accuracies
                chart['data']['datasets'][0]['backgroundColor'] = colors
                print(f'更新准确率对比图表')
                for i, name in enumerate(model_names):
                    print(f'  {name}: {accuracies[i]}%')
            elif chart['id'] == 'inference_time':
                chart['data']['labels'] = model_names
                chart['data']['datasets'][0]['data'] = inference_times
                print(f'\n更新推理时间对比图表')
                for i, name in enumerate(model_names):
                    print(f'  {name}: {inference_times[i]}ms')

with open('dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('\n成功更新模型性能数据和图表')