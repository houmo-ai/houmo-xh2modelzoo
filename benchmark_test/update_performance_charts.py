import json

with open('dashboard_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

models = data['models']

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
                print(f'模型: {model_names}')
                print(f'准确率: {accuracies}')
            elif chart['id'] == 'inference_time':
                chart['data']['labels'] = model_names
                chart['data']['datasets'][0]['data'] = inference_times
                print(f'\n更新推理时间对比图表')
                print(f'模型: {model_names}')
                print(f'推理时间: {inference_times}')

with open('dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('\n成功更新模型性能图表')