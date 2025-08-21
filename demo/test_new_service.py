#!/usr/bin/env python3
"""
测试新的ASearcher服务
验证LLM集成和完整workflow
"""

import asyncio
import aiohttp
import json
import time

async def test_service():
    """测试服务功能"""
    server_url = "http://0.0.0.0:8080"
    
    print("🧪 测试ASearcher Agent服务 v2")
    print("=" * 40)
    
    async with aiohttp.ClientSession() as session:
        # 1. 健康检查
        print("1. 健康检查...")
        try:
            async with session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ 服务状态: {data['status']}")
                    print(f"   🤖 LLM状态: {data['llm_status']}")
                    print(f"   🔧 LLM类型: {data.get('llm_type', 'N/A')}")
                    if data.get('model_name'):
                        print(f"   📋 模型名称: {data['model_name']}")
                    if data.get('model_path'):
                        print(f"   📁 模型路径: {data['model_path']}")
                    if data.get('openai_base_url'):
                        print(f"   🌐 Base URL: {data['openai_base_url']}")
                    print(f"   🔑 API Key状态: {data.get('api_key_status', 'N/A')}")
                else:
                    print(f"   ❌ 健康检查失败: {response.status}")
                    return
        except Exception as e:
            print(f"   ❌ 无法连接服务: {e}")
            return
        
        # 2. 启动查询
        print("\n2. 启动测试查询...")
        test_query = {
            "query": "B站Up主HOPICO对方大同的专访视频获得了第多少期的每周必看？",
            "max_turns": 32,
            "search_client_type": "async-web-search-access",
            "use_jina": True,
            "temperature": 0.6,  # 更新默认温度
            "max_tokens_per_call": 4096,
            "agent_type": "asearcher",
            "prompt_type": "ASearcher"  # 更新默认prompt类型
        }
        
        try:
            async with session.post(f"{server_url}/query", json=test_query) as response:
                if response.status == 200:
                    data = await response.json()
                    query_id = data['query_id']
                    print(f"   ✅ 查询已启动: {query_id[:8]}...")
                else:
                    print(f"   ❌ 启动查询失败: {response.status}")
                    return
        except Exception as e:
            print(f"   ❌ 启动查询异常: {e}")
            return
        
        # 3. 轮询查询状态
        print("\n3. 监控查询进度...")
        last_step_count = 0
        start_time = time.time()
        timeout = 1800  # 3分钟超时
        
        while time.time() - start_time < timeout:
            try:
                async with session.get(f"{server_url}/query/{query_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data['status']
                        steps = data['steps']
                        
                        # 显示新步骤
                        if len(steps) > last_step_count:
                            for i in range(last_step_count, len(steps)):
                                step = steps[i]
                                step_type = step['step_type']
                                title = step['title']
                                content = step['content']
                                
                                print(f"   📝 步骤 {i+1}: {step_type} - {title}")
                                
                                # 根据步骤类型显示不同详细程度的内容
                                if step_type == "info" and "搜索结果" in title:
                                    # 搜索结果步骤 - 显示更多信息
                                    if "搜索结果:" in content and len(content) > 200:
                                        # 尝试解析搜索结果格式
                                        lines = content.split('\n')
                                        print(f"      内容: {lines[0] if lines else '搜索结果:'}")
                                        
                                        # 显示前3个搜索结果的详细信息
                                        result_count = 0
                                        for line in lines[1:]:
                                            line = line.strip()
                                            if line and result_count < 3:
                                                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                                                    # 提取URL和标题
                                                    if '(' in line and ')' in line:
                                                        title_part = line.split('(')[0].strip()
                                                        url_part = line.split('(')[1].split(')')[0] if ')' in line else ""
                                                        print(f"        🔗 {title_part}")
                                                        if url_part:
                                                            print(f"           URL: {url_part}")
                                                    else:
                                                        preview = line[:]
                                                        print(f"        {preview}")
                                                    result_count += 1
                                    else:
                                        content_preview = content[:]
                                        print(f"      内容: {content_preview}")
                                elif step_type == "final-result":
                                    # 最终答案 - 显示完整内容
                                    print(f"      内容: {content}")
                                elif step_type == "search":
                                    # 搜索查询 - 显示查询内容
                                    print(f"      内容: {content}")
                                elif step_type == "access":
                                    # 网页访问 - 显示URL
                                    print(f"      内容: {content}")
                                else:
                                    # 其他步骤 - 适度截断
                                    content_preview = content[:]
                                    print(f"      内容: {content_preview}")
                            
                            last_step_count = len(steps)
                        
                        # 检查是否完成
                        if status in ['completed', 'error', 'cancelled']:
                            print(f"\n🏁 查询完成，状态: {status}")
                            
                            if data.get('pred_answer'):
                                print(f"\n📋 最终答案:")
                                print(f"{data['pred_answer']}")
                            
                            if data.get('error_message'):
                                print(f"\n❌ 错误信息: {data['error_message']}")
                            
                            print(f"\n📊 统计信息:")
                            print(f"   总步骤数: {len(steps)}")
                            print(f"   执行时间: {time.time() - start_time:.2f}秒")
                            
                            # 分析步骤类型统计
                            step_stats = {}
                            for step in steps:
                                step_type = step.get('step_type', 'unknown')
                                step_stats[step_type] = step_stats.get(step_type, 0) + 1
                            
                            print(f"   步骤类型统计:")
                            for step_type, count in step_stats.items():
                                type_name = {
                                    'question': '用户问题',
                                    'thinking': 'Agent思考',
                                    'search': '搜索查询',
                                    'access': '网页访问',
                                    'info': '信息获取',
                                    'response': '响应生成',
                                    'final_result': '最终结果',
                                    'final-result': '最终结果',
                                    'error': '错误处理',
                                    'completed': '处理完成',
                                    'cancelled': '已取消'
                                }.get(step_type, step_type)
                                print(f"     - {type_name}: {count}次")
                            
                            return
                    else:
                        print(f"   ❌ 获取状态失败: {response.status}")
                        break
            except Exception as e:
                print(f"   ❌ 轮询异常: {e}")
                break
            
            await asyncio.sleep(2)
        
        print(f"\n⏰ 查询超时 ({timeout}秒)")

if __name__ == "__main__":
    print("请确保ASearcher服务正在运行:")
    print("python demo/asearcher_service_old.py \\")
    print("  --host 0.0.0.0 \\")
    print("  --port 8080 \\")
    print("  --model-name 'ASearcher-Web-7B' \\")
    print("  --model-path '/Users/hechuyi/ASearcher-Web-7B' \\")
    print("  --openai-api-key 'empty' \\")
    print("  --openai-base-url 'http://localhost:50000/v1'")
    print()
    
    try:
        asyncio.run(test_service())
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
