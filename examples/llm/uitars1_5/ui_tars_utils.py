import base64
import re
import ast
from io import BytesIO
from typing import List, Dict, Any, Tuple

from PIL import Image

MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>')\n\n## User Instruction
{instruction}"""


def convert_pil_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _normalize_language(language: str) -> str:
    if not language:
        return "English"
    lang = language.strip().lower()
    if lang in {"en", "english"}:
        return "English"
    if lang in {"zh", "zh-cn", "zh_cn", "chinese", "cn"}:
        return "Chinese"
    return language


def construct_prompt(image: Image.Image, instruction: str, language: str = "English", mode: str = "agent") -> List[Dict[str, Any]]:
    if mode == "grounding":
        formatted_system_prompt = GROUNDING_DOUBAO.format(
            instruction=instruction
        )
    elif mode == "computer_use":
        formatted_system_prompt = COMPUTER_USE_DOUBAO.format(
            instruction=instruction,
            language=_normalize_language(language),
        )
    else:
        formatted_system_prompt = MOBILE_USE_DOUBAO.format(
            instruction=instruction,
            language=_normalize_language(language),
        )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + convert_pil_image_to_base64(image)
                    }
                },
                {
                    "type": "text",
                    "text": formatted_system_prompt
                }
            ]
        }
    ]


def convert_point_to_coordinates(text: str) -> str:
    pattern = r"<point>(\d+)\s+(\d+)</point>"

    def replace_match(match):
        x1, y1 = map(int, match.groups())
        return f"({x1},{y1})"

    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def parse_action_ast(action_str: str) -> Dict[str, Any]:
    try:
        node = ast.parse(action_str, mode="eval")
        if not isinstance(node, ast.Expression):
            return None

        call = node.body
        if not isinstance(call, ast.Call):
            return None

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}
    except Exception:
        return None


def escape_single_quotes(text: str) -> str:
    return re.sub(r"(?<!\\)'", r"\\'", text)


def parse_output(output_text: str) -> Dict[str, Any]:
    result = {
        "thought": "",
        "action_type": None,
        "action_inputs": {},
        "raw_output": output_text
    }

    thought_match = re.search(r"Thought:\s*(.+?)(?=\s*Action:|$)", output_text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+)$", output_text, re.DOTALL)
    if not action_match:
        return result

    action_str = action_match.group(1).strip()

    if "<point>" in action_str:
        action_str = convert_point_to_coordinates(action_str)

    if "start_point=" in action_str:
        action_str = action_str.replace("start_point=", "start_box=")
    if "end_point=" in action_str:
        action_str = action_str.replace("end_point=", "end_box=")
    if "point=" in action_str:
        action_str = action_str.replace("point=", "start_box=")

    if "type(content=" in action_str:
        try:
            ast.parse(action_str, mode="eval")
        except SyntaxError:
            match = re.search(r"type\(content='(.*)'\)", action_str)
            if match:
                action_str = f"type(content='{escape_single_quotes(match.group(1))}')"

    parsed_ast = parse_action_ast(action_str)
    if parsed_ast:
        result["action_type"] = parsed_ast["function"]
        result["action_inputs"] = parsed_ast["args"]
        return result

    func_match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
    if func_match:
        result["action_type"] = func_match.group(1)
        args_str = func_match.group(2)

        start_box_match = re.search(r"start_box='(.*?)'", args_str)
        if start_box_match:
            result["action_inputs"]["start_box"] = start_box_match.group(1)

        end_box_match = re.search(r"end_box='(.*?)'", args_str)
        if end_box_match:
            result["action_inputs"]["end_box"] = end_box_match.group(1)

        content_match = re.search(r"content='(.*?)'", args_str)
        if content_match:
            result["action_inputs"]["content"] = content_match.group(1)

        key_match = re.search(r"key='(.*?)'", args_str)
        if key_match:
            result["action_inputs"]["key"] = key_match.group(1)

        direction_match = re.search(r"direction='(.*?)'", args_str)
        if direction_match:
            result["action_inputs"]["direction"] = direction_match.group(1)

        app_name_match = re.search(r"app_name='(.*?)'", args_str)
        if app_name_match:
            result["action_inputs"]["app_name"] = app_name_match.group(1)

    return result



def parse_coordinates(box_str: str) -> Tuple[int, int]:
    if not box_str:
        return (0, 0)

    clean_str = box_str.replace("<|box_start|>", "").replace("<|box_end|>", "")
    clean_str = clean_str.replace("(", "").replace(")", "")

    parts = clean_str.split(",")
    if len(parts) >= 2:
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            pass

    return (0, 0)
