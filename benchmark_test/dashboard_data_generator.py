import json
from datetime import datetime

def generate_initial_data():
    """生成初始看板数据"""
    
    tags = [
        {"id": "tag_performance", "name": "性能", "color": "#FF6384"},
        {"id": "tag_model", "name": "模型", "color": "#36A2EB"},
        {"id": "tag_ai", "name": "AI", "color": "#FFCE56"},
        {"id": "tag_test", "name": "测试", "color": "#4BC0C0"},
        {"id": "tag_coverage", "name": "覆盖率", "color": "#9966FF"},
        {"id": "tag_quality", "name": "质量", "color": "#FF9F40"},
        {"id": "tag_resource", "name": "资源", "color": "#C9CBCF"},
        {"id": "tag_monitor", "name": "监控", "color": "#7BC225"},
        {"id": "tag_system", "name": "系统", "color": "#4BC0C0"}
    ]
    
    tag_values = {
        "模型类型": ["目标检测", "图像分类", "大语言模型", "视觉语言模型", "图像分割", "其他"],
        "量化状态": ["FP32", "FP16", "INT8", "未量化"],
        "算法文档": ["已完善", "待完善", "缺失"],
        "负责人": ["张三", "李四", "王五", "赵六", "待分配"],
        "精度": ["高精度", "中精度", "低精度"],
        "测试通过": ["全部通过", "部分通过", "未测试"],
        "推理时间": ["快速", "中等", "慢速"],
        "内存使用": ["低内存", "中内存", "高内存"]
    }
    
    models = [
        {
            "id": "yolov5",
            "name": "YOLOv5",
            "category": "目标检测",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "目标检测",
                "量化状态": "FP32",
                "算法文档": "已完善",
                "负责人": "张三",
                "精度": "中精度",
                "测试通过": "全部通过",
                "推理时间": "快速",
                "内存使用": "低内存"
            },
            "metrics": {
                "accuracy": 89.5,
                "inference_time": 45,
                "memory_usage": 2.3,
                "cpu_usage": 45,
                "test_passed": 45,
                "test_failed": 3,
                "test_skipped": 2
            },
            "last_test_date": "2026-02-10",
            "status": "active"
        },
        {
            "id": "yolov8",
            "name": "YOLOv8",
            "category": "目标检测",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "目标检测",
                "量化状态": "FP16",
                "算法文档": "已完善",
                "负责人": "李四",
                "精度": "高精度",
                "测试通过": "全部通过",
                "推理时间": "中等",
                "内存使用": "中内存"
            },
            "metrics": {
                "accuracy": 92.3,
                "inference_time": 52,
                "memory_usage": 2.8,
                "cpu_usage": 52,
                "test_passed": 38,
                "test_failed": 5,
                "test_skipped": 1
            },
            "last_test_date": "2026-02-10",
            "status": "active"
        },
        {
            "id": "yolov10",
            "name": "YOLOv10",
            "category": "目标检测",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "目标检测",
                "量化状态": "FP16",
                "算法文档": "待完善",
                "负责人": "王五",
                "精度": "高精度",
                "测试通过": "未测试",
                "推理时间": "快速",
                "内存使用": "低内存"
            },
            "metrics": {
                "accuracy": 94.1,
                "inference_time": 38,
                "memory_usage": 2.5,
                "cpu_usage": 38,
                "test_passed": 0,
                "test_failed": 0,
                "test_skipped": 0
            },
            "last_test_date": "2026-02-09",
            "status": "active"
        },
        {
            "id": "yolov11",
            "name": "YOLOv11",
            "category": "目标检测",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "目标检测",
                "量化状态": "FP16",
                "算法文档": "待完善",
                "负责人": "赵六",
                "精度": "高精度",
                "测试通过": "未测试",
                "推理时间": "快速",
                "内存使用": "低内存"
            },
            "metrics": {
                "accuracy": 95.2,
                "inference_time": 42,
                "memory_usage": 2.6,
                "cpu_usage": 42,
                "test_passed": 0,
                "test_failed": 0,
                "test_skipped": 0
            },
            "last_test_date": "2026-02-09",
            "status": "active"
        },
        {
            "id": "resnet50",
            "name": "ResNet50",
            "category": "图像分类",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "图像分类",
                "量化状态": "FP32",
                "算法文档": "已完善",
                "负责人": "张三",
                "精度": "中精度",
                "测试通过": "全部通过",
                "推理时间": "快速",
                "内存使用": "低内存"
            },
            "metrics": {
                "accuracy": 88.7,
                "inference_time": 35,
                "memory_usage": 1.8,
                "cpu_usage": 35,
                "test_passed": 22,
                "test_failed": 2,
                "test_skipped": 1
            },
            "last_test_date": "2026-02-10",
            "status": "active"
        },
        {
            "id": "vit",
            "name": "ViT",
            "category": "图像分类",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "图像分类",
                "量化状态": "FP32",
                "算法文档": "待完善",
                "负责人": "李四",
                "精度": "高精度",
                "测试通过": "未测试",
                "推理时间": "中等",
                "内存使用": "低内存"
            },
            "metrics": {
                "accuracy": 91.8,
                "inference_time": 48,
                "memory_usage": 2.1,
                "cpu_usage": 48,
                "test_passed": 0,
                "test_failed": 0,
                "test_skipped": 0
            },
            "last_test_date": "2026-02-08",
            "status": "active"
        },
        {
            "id": "qwen2_7b",
            "name": "Qwen2-7B",
            "category": "大语言模型",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "大语言模型",
                "量化状态": "FP16",
                "算法文档": "已完善",
                "负责人": "王五",
                "精度": "中精度",
                "测试通过": "全部通过",
                "推理时间": "慢速",
                "内存使用": "高内存"
            },
            "metrics": {
                "accuracy": 87.5,
                "inference_time": 125,
                "memory_usage": 14.2,
                "cpu_usage": 78,
                "test_passed": 15,
                "test_failed": 2,
                "test_skipped": 1
            },
            "last_test_date": "2026-02-10",
            "status": "active"
        },
        {
            "id": "qwen3_vl_8b",
            "name": "Qwen3-VL-8B",
            "category": "视觉语言模型",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "视觉语言模型",
                "量化状态": "FP16",
                "算法文档": "已完善",
                "负责人": "赵六",
                "精度": "高精度",
                "测试通过": "全部通过",
                "推理时间": "慢速",
                "内存使用": "高内存"
            },
            "metrics": {
                "accuracy": 90.2,
                "inference_time": 145,
                "memory_usage": 16.5,
                "cpu_usage": 85,
                "test_passed": 12,
                "test_failed": 1,
                "test_skipped": 0
            },
            "last_test_date": "2026-02-09",
            "status": "active"
        },
        {
            "id": "unet",
            "name": "UNet",
            "category": "图像分割",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "图像分割",
                "量化状态": "FP32",
                "算法文档": "待完善",
                "负责人": "张三",
                "精度": "中精度",
                "测试通过": "全部通过",
                "推理时间": "中等",
                "内存使用": "中内存"
            },
            "metrics": {
                "accuracy": 86.3,
                "inference_time": 55,
                "memory_usage": 2.9,
                "cpu_usage": 55,
                "test_passed": 8,
                "test_failed": 1,
                "test_skipped": 0
            },
            "last_test_date": "2026-02-08",
            "status": "active"
        },
        {
            "id": "sam",
            "name": "SAM",
            "category": "图像分割",
            "tags": ["性能", "模型", "AI"],
            "attributes": {
                "模型类型": "图像分割",
                "量化状态": "FP16",
                "算法文档": "已完善",
                "负责人": "李四",
                "精度": "高精度",
                "测试通过": "全部通过",
                "推理时间": "中等",
                "内存使用": "中内存"
            },
            "metrics": {
                "accuracy": 92.8,
                "inference_time": 68,
                "memory_usage": 3.2,
                "cpu_usage": 62,
                "test_passed": 10,
                "test_failed": 0,
                "test_skipped": 1
            },
            "last_test_date": "2026-02-10",
            "status": "active"
        }
    ]
    
    dashboards = [
        {
            "id": "model_performance",
            "name": "模型性能看板",
            "description": "展示各种模型的性能指标",
            "tags": ["性能", "模型", "AI"],
            "is_default": True,
            "charts": [
                {
                    "id": "accuracy_comparison",
                    "title": "模型准确率对比",
                    "type": "bar",
                    "data": {
                        "labels": ["YOLOv5", "YOLOv8", "YOLOv10", "YOLOv11", "ResNet50", "ViT"],
                        "datasets": [
                            {
                                "label": "准确率 (%)",
                                "data": [89.5, 92.3, 94.1, 95.2, 88.7, 91.8],
                                "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF", "#FF9F40"]
                            }
                        ]
                    }
                },
                {
                    "id": "inference_time",
                    "title": "推理时间对比 (ms)",
                    "type": "line",
                    "data": {
                        "labels": ["YOLOv5", "YOLOv8", "YOLOv10", "YOLOv11", "ResNet50", "ViT"],
                        "datasets": [
                            {
                                "label": "推理时间",
                                "data": [45, 52, 38, 42, 35, 48],
                                "borderColor": "#36A2EB",
                                "backgroundColor": "rgba(54, 162, 235, 0.1)"
                            }
                        ]
                    }
                }
            ]
        },
        {
            "id": "test_coverage",
            "name": "测试覆盖率看板",
            "description": "展示测试覆盖率和测试结果",
            "tags": ["测试", "覆盖率", "质量"],
            "is_default": True,
            "charts": [
                {
                    "id": "coverage_pie",
                    "title": "测试覆盖率分布",
                    "type": "pie",
                    "data": {
                        "labels": ["已覆盖", "未覆盖", "部分覆盖"],
                        "datasets": [
                            {
                                "data": [65, 20, 15],
                                "backgroundColor": ["#4BC0C0", "#FF6384", "#FFCE56"]
                            }
                        ]
                    }
                },
                {
                    "id": "test_results",
                    "title": "测试结果统计",
                    "type": "bar",
                    "data": {
                        "labels": ["YOLO系列", "Qwen系列", "ResNet系列", "其他模型"],
                        "datasets": [
                            {
                                "label": "通过",
                                "data": [45, 38, 22, 31],
                                "backgroundColor": "#4BC0C0"
                            },
                            {
                                "label": "失败",
                                "data": [3, 5, 2, 4],
                                "backgroundColor": "#FF6384"
                            },
                            {
                                "label": "跳过",
                                "data": [2, 1, 1, 2],
                                "backgroundColor": "#FFCE56"
                            }
                        ]
                    }
                }
            ]
        },
        {
            "id": "resource_usage",
            "name": "资源使用看板",
            "description": "展示系统资源使用情况",
            "tags": ["资源", "监控", "系统"],
            "is_default": False,
            "charts": [
                {
                    "id": "memory_usage",
                    "title": "内存使用趋势",
                    "type": "line",
                    "data": {
                        "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"],
                        "datasets": [
                            {
                                "label": "内存使用 (GB)",
                                "data": [12.5, 11.2, 15.8, 18.3, 16.7, 14.2],
                                "borderColor": "#9966FF",
                                "backgroundColor": "rgba(153, 102, 255, 0.1)"
                            }
                        ]
                    }
                },
                {
                    "id": "cpu_usage",
                    "title": "CPU使用率 (%)",
                    "type": "bar",
                    "data": {
                        "labels": ["YOLOv5", "YOLOv8", "YOLOv10", "YOLOv11", "ResNet50", "ViT"],
                        "datasets": [
                            {
                                "label": "CPU使用率",
                                "data": [45, 52, 38, 42, 35, 48],
                                "backgroundColor": "#FF9F40"
                            }
                        ]
                    }
                }
            ]
        }
    ]
    
    return {
        "dashboards": dashboards,
        "tags": tags,
        "models": models,
        "tag_values": tag_values
    }

if __name__ == "__main__":
    data = generate_initial_data()
    print(json.dumps(data, ensure_ascii=False, indent=2))