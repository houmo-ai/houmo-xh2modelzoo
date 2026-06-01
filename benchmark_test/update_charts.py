import json

with open('dashboard_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

model_type_counts = {}
for model in data['models']:
    model_type = model.get('attributes', {}).get('模型类型', '其他')
    model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1

sorted_types = sorted(model_type_counts.items(), key=lambda x: x[1], reverse=True)

labels = [item[0] for item in sorted_types]
values = [item[1] for item in sorted_types]

colors = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
    "#FF9F40", "#FF6384", "#C9CBCF", "#4BC0C0", "#FF6384",
    "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40", "#C9CBCF"
]

for dashboard in data['dashboards']:
    if dashboard['id'] == 'model_category':
        for chart in dashboard['charts']:
            if chart['id'] == 'model_category_distribution':
                chart['data']['labels'] = labels
                chart['data']['datasets'][0]['data'] = values
                chart['data']['datasets'][0]['backgroundColor'] = colors[:len(labels)]
                print(f'更新模型类别分布图表:')
                print(f'标签: {labels}')
                print(f'数据: {values}')

with open('dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('\n成功更新模型类别分布图表')