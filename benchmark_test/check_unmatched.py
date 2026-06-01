import json
import re

EMAIL_REPORT_FILE = '/data01/home/xuchen/xh2/xh2_model_zoo/benchmark_test/email_report.html'
DASHBOARD_DATA_FILE = '/data01/home/xuchen/xh2/xh2_model_zoo/benchmark_test/dashboard_data.json'

def parse_email_report():
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
        print(f'警告: {EMAIL_REPORT_FILE} 文件不存在')
    except Exception as e:
        print(f'解析email_report失败: {str(e)}')
    
    return test_results

test_results = parse_email_report()

with open(DASHBOARD_DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

models = data.get('models', [])

matched_tests = set()

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
            matched_tests.add(test_name)
            break

print(f'email_report中的测试总数: {len(test_results)}')
print(f'匹配到的测试数量: {len(matched_tests)}')
print(f'未匹配的测试数量: {len(test_results) - len(matched_tests)}')

print('\n未匹配的测试及其结果:')
for test_name in test_results.keys():
    if test_name not in matched_tests:
        result = test_results[test_name]
        status = '通过' if result['passed'] > 0 else '失败'
        print(f'{test_name}: {status}')


# BGE-Reranker测试: 通过
# Camerc3测试: 通过
# EDSR测试: 通过
# Efficient 模型测试: 通过
# Internvl测试: 通过
# Minicpm 模型测试: 通过
# Qwen25vl_0.5b测试: 通过
# Qwen25vl_14b测试: 通过
# Qwen25vl_32b测试: 通过
# Qwen25vl_3b测试: 通过
# Qwen25vl_72b测试: 通过
# Qwen25vl测试: 通过
# Qwen25vl_coder_7B测试: 通过
# Qwen3-8B-moe-2507测试: 通过
# Qwen3-8B-moe-coder测试: 通过
# Qwen3-4B-2507测试: 通过
# Deepseek-0528-Qwen3-8B测试: 失败
# RTDetr测试: 通过
# SAM 模型测试: 通过