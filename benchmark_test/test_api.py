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

with open(DASHBOARD_DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

test_results = parse_email_report()
models = data.get('models', [])

for model in models:
    model['tests_passed'] = 0
    model['tests_failed'] = 0

print('匹配结果:')
for i, model in enumerate(models):
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
    
    status = '✓' if model['tests_passed'] > 0 or model['tests_failed'] > 0 else '✗'
    if status == '✓':
        model_status = model.get('status', 'N/A')
        print(f'{i+1}. {model_name} {status}: tests_passed={model.get("tests_passed", 0)}, tests_failed={model.get("tests_failed", 0)}, status={model_status}')

total_passed = sum(m.get('tests_passed', 0) for m in models)
total_failed = sum(m.get('tests_failed', 0) for m in models)
print(f'\n总测试通过: {total_passed}')
print(f'总测试失败: {total_failed}')
