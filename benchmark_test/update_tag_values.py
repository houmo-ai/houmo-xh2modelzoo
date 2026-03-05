import json

with open('dashboard_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

model_types = set()
chip_types = set()
owners = set()
quantization_statuses = set()

for model in data['models']:
    attributes = model.get('attributes', {})
    if '模型类型' in attributes:
        model_types.add(attributes['模型类型'])
    if '芯片类型' in attributes:
        chip_types.add(attributes['芯片类型'])
    if '负责人' in attributes:
        owners.add(attributes['负责人'])
    if '量化状态' in attributes:
        quantization_statuses.add(attributes['量化状态'])

data['tag_values']['模型类型'] = sorted(list(model_types))
data['tag_values']['芯片类型'] = sorted(list(chip_types))
data['tag_values']['负责人'] = sorted(list(owners))
data['tag_values']['量化状态'] = sorted(list(quantization_statuses))

with open('dashboard_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('成功更新 tag_values')
print(f'\n模型类型 ({len(model_types)}个):')
for mt in sorted(model_types):
    print(f'  - {mt}')

print(f'\n芯片类型 ({len(chip_types)}个):')
for ct in sorted(chip_types):
    print(f'  - {ct}')

print(f'\n负责人 ({len(owners)}个):')
for owner in sorted(owners):
    print(f'  - {owner}')

print(f'\n量化状态 ({len(quantization_statuses)}个):')
for qs in sorted(quantization_statuses):
    print(f'  - {qs}')