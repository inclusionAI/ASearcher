#!/usr/bin/env python3
"""ASearcher Visual Demo Service - Async processing with vLLM OpenAI API"""

import asyncio
import uuid
import os
import sys
import re
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import AsyncOpenAI
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import make_agent
from tools.search_utils import make_search_client
from evaluation.config_loader import load_config_and_set_env
from evaluation.utils import make_prompt, PROMPT_TYPES


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    max_turns: Optional[int] = 32
    search_client_type: Optional[str] = "async-web-search-access"
    use_jina: Optional[bool] = True
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    max_tokens_per_call: Optional[int] = 4096
    agent_type: Optional[str] = "asearcher"
    prompt_type: Optional[str] = "asearcher"

class AgentStep(BaseModel):
    step_id: int
    type: str
    title: str
    content: str
    timestamp: str

class QueryResponse(BaseModel):
    query_id: str
    status: str
    steps: List[AgentStep]
    pred_answer: Optional[str] = None
    error_message: Optional[str] = None

class ServiceHealth(BaseModel):
    status: str
    llm_status: str
    llm_type: str
    model_name: Optional[str] = None
    version: str
    available_agent_types: List[str]
    available_prompt_types: List[str]

class CompatibleLLMResponse:
    def __init__(self, text: str, input_len: Optional[int] = None, 
                 input_tokens: Optional[List[int]] = None,
                 output_len: Optional[int] = None,
                 output_tokens: Optional[List[int]] = None,
                 output_logprobs: Optional[List[float]] = None,
                 output_versions: Optional[List[int]] = None,
                 finish_reason: Optional[str] = None):
        self.text = text
        self.input_len = input_len
        self.input_tokens = input_tokens or []
        self.output_len = output_len
        self.output_tokens = output_tokens or []
        self.output_logprobs = output_logprobs or []
        self.output_versions = output_versions or []
        self.finish_reason = finish_reason

class AsyncVLLMClient:
    """Async vLLM client using OpenAI compatible API"""
    def __init__(self, llm_url: str, model_name: str = "default", api_key: str = "EMPTY"):
        self.llm_url = llm_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        
        self.client = AsyncOpenAI(
            base_url=f"{self.llm_url}/v1",
            api_key=self.api_key,
        )
        logger.info(f"Initialized AsyncOpenAI client: {self.llm_url}")
    
    async def async_generate(self, prompt: str, sampling_kwargs: Dict) -> Dict:
        """Generate text asynchronously"""
        completion_kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": sampling_kwargs.get("max_new_tokens", 4096),
            "temperature": sampling_kwargs.get("temperature", 0.0),
            "top_p": sampling_kwargs.get("top_p", 1.0),
            "stream": False,
        }
        
        stop_sequences = sampling_kwargs.get("stop", [])
        if stop_sequences:
            completion_kwargs["stop"] = stop_sequences
        
        extra_body = {}
        if "top_k" in sampling_kwargs and sampling_kwargs["top_k"] > 0:
            extra_body["top_k"] = sampling_kwargs["top_k"]
        
        if "stop_token_ids" in sampling_kwargs:
            extra_body["stop_token_ids"] = sampling_kwargs["stop_token_ids"]
        
        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        
        logger.info(f"Calling vLLM API: {completion_kwargs}")
        
        response = await self.client.completions.create(**completion_kwargs)
        
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            return {"text": choice.text, "finish_reason": choice.finish_reason}
        else:
            return {"text": "", "finish_reason": "unknown"}
    
    async def close(self):
        """Close client connection"""
        if self.client:
            await self.client.close()


class AsearcherVisualDemo:
    """ASearcher Visual Demo Service"""
    
    def __init__(self, llm_url: str = None, model_name: str = "default", api_key: str = "EMPTY"):
        self.app = FastAPI(title="ASearcher Visual Demo API", version="1.0.0")
        self.setup_cors()
        self.setup_routes()
        
        self.active_queries: Dict[str, Dict] = {}
        self.search_clients = {}
        
        self.llm_url = llm_url or "http://localhost:8000"
        self.model_name = model_name
        self.api_key = api_key
        self.llm = None
        self.tokenizer = None
        
        self._load_config()
        self._initialize_vllm_client()

    def _load_config(self):
        """Load configuration"""
        config_path = os.path.join(project_root, "evaluation", "eval_config.yaml")
        if os.path.exists(config_path):
            load_config_and_set_env(config_path)
            logger.info("Configuration loaded successfully")

    def _initialize_vllm_client(self):
        """Initialize vLLM client"""
        self.llm = AsyncVLLMClient(self.llm_url, self.model_name, self.api_key)
        
        model_path = "/Users/hechuyi/ASearcher-Web-QwQ"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Tokenizer loaded: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {model_path}: {e}, using default")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        
        logger.info("vLLM client initialized successfully")

    def setup_cors(self):
        """Setup CORS"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def get_search_client(self, search_client_type: str, use_jina: bool = False):
        """Get search client"""
        client_key = f"{search_client_type}_{use_jina}"
        if client_key not in self.search_clients:
            jina_api_key = os.environ.get('JINA_API_KEY', '') if use_jina else None
            self.search_clients[client_key] = make_search_client(
                search_client_type, use_jina=use_jina, jina_api_key=jina_api_key
            )
        return self.search_clients[client_key]

    def setup_routes(self):
        """Setup routes"""
        
        @self.app.get("/health", response_model=ServiceHealth)
        async def health_check():
            """Health check endpoint"""
            available_agents = ["asearcher", "asearcher-reasoning", "search-r1"]
            available_prompts = list(PROMPT_TYPES.keys())
            
            return ServiceHealth(
                status="healthy",
                llm_status="available" if self.llm else "unavailable",
                llm_type="vLLM",
                model_name=self.model_name,
                version="1.0.0",
                available_agent_types=available_agents,
                available_prompt_types=available_prompts
            )

        @self.app.post("/query")
        async def start_query(request: QueryRequest, background_tasks: BackgroundTasks):
            """Start query processing"""
            if not self.llm:
                raise HTTPException(status_code=503, detail="LLM service unavailable")
            
            query_id = str(uuid.uuid4())
            query_data = {
                "query_id": query_id,
                "status": "running",
                "request": request,
                "steps": [],
                "start_time": time.time(),
                "pred_answer": None,
                "error_message": None
            }
            
            self.active_queries[query_id] = query_data
            background_tasks.add_task(self.process_query, query_id)
            
            return {"query_id": query_id, "status": "started"}

        @self.app.get("/query/{query_id}", response_model=QueryResponse)
        async def get_query_status(query_id: str):
            """Get query status"""
            if query_id not in self.active_queries:
                raise HTTPException(status_code=404, detail="Query not found")
            
            query_data = self.active_queries[query_id]
            return QueryResponse(
                query_id=query_id,
                status=query_data["status"],
                steps=query_data["steps"],
                pred_answer=query_data.get("pred_answer"),
                error_message=query_data.get("error_message")
            )

        @self.app.delete("/query/{query_id}")
        async def cancel_query(query_id: str):
            """Cancel query"""
            if query_id not in self.active_queries:
                raise HTTPException(status_code=404, detail="Query not found")
            
            query_data = self.active_queries[query_id]
            query_data["status"] = "cancelled"
            query_data["error_message"] = "Query cancelled by user"
            
            return {"message": "Query cancelled"}

    async def process_query(self, query_id: str):
        """Process single query (background task)"""
        query_data = self.active_queries[query_id]
        request = query_data["request"]
        
        await self.add_step(query_id, "start", "Start processing", f"Query: <strong>{request.query}</strong>")
        
        agent = make_agent(request.agent_type)
        if hasattr(agent, 'set_tokenizer'):
            agent.set_tokenizer(self.tokenizer)
        
        try:
            prompt = make_prompt(request.query, request.prompt_type)
        except:
            prompt = request.query
        
        agent.initialize_with_prompt(prompt)
        agent.max_turns = request.max_turns
        
        search_client = self.get_search_client(request.search_client_type, request.use_jina)
        
        process = {
            "id": query_id,
            "prompt": prompt,
            "history": [],
            "running": True,
            "pred_answer": None
        }
        
        await self.process_agent_turns(query_id, agent, search_client, request, process)
        
        # If the process finished without a clear answer, mark as completed.
        if query_data["status"] == "running":
            query_data["status"] = "completed"
            await self.add_step(query_id, "completed", "Query completed", "No clear answer found")

    async def process_agent_turns(self, query_id: str, agent, search_client, request, process):
        """Process agent multi-turn conversation"""
        turn_counter = 1
        page_counter = 1
        consecutive_thoughts = [] # Buffer for 'thinking' phase thoughts

        while (process["running"] and
               agent.num_turns < agent.max_turns and
               self.active_queries[query_id]["status"] == "running"):

            if agent.is_finished:
                logger.info(f"Agent marked as finished (query {query_id})")
                process["running"] = False
                break

            # Determine agent phase based on its internal state
            is_browsing = False
            page_content_to_display = None
            agent_type = "reasoning" if hasattr(agent, "current_process") else "default"

            if agent_type == "default" and not agent.job_queue.empty():
                is_browsing = True
                next_job = agent.job_queue.queue[0]
                if next_job.get("type") == "webpage":
                    # Clean up the page content for better readability
                    raw_content = next_job.get("text", "")
                    # Remove information tags and page markers
                    cleaned_content = re.sub(r'</?information>', '', raw_content)
                    cleaned_content = re.sub(r'>>>> Page \d+ >>>>', '', cleaned_content).strip()
                    page_content_to_display = cleaned_content
            elif agent_type == "reasoning" and agent.current_process:
                # For reasoning agent, browsing can be detected either by remaining page_cache
                # or by the latest history entry being a just-accessed page (single-page cases)
                history = agent.current_process.get("history") or []
                page_cache = agent.current_process.get("page_cache") or []
                has_pending_pages = len(page_cache) > 0
                last_is_page = (len(history) > 0 and isinstance(history[-1], dict) and history[-1].get("type") == "page")
                
                # Only set is_browsing=True when we have actual page content to display
                if has_pending_pages:
                    is_browsing = True
                    raw_content = page_cache[0]
                    cleaned_content = re.sub(r'>>>> Page \d+ >>>>', '', raw_content).strip()
                    page_content_to_display = cleaned_content
                elif last_is_page:
                    # Just finished processing a page, but no new content to display
                    # Still consider this browsing phase for thought categorization, but no content display
                    is_browsing = True

            # --- Display Logic for Browsing Content (AsearcherAgent only) ---
            displayed_page_content = False
            if page_content_to_display:
                await self.add_step(query_id, "browsing", f"Browsing (Page {page_counter})", page_content_to_display)
                displayed_page_content = True

            prompt, sampling_params = agent.prepare_llm_query()

            llm_response = await self.process_single_llm_query(prompt, sampling_params, request, query_id)
            completion_text = llm_response.text

            # --- Thought and Action Separation ---
            # Prefer extracting the latest <thought>...</thought> (or <think>...</think>) block
            thought = ""
            thought_match = re.findall(r'<thought>(.*?)</thought>', completion_text, re.DOTALL)
            if thought_match:
                thought = thought_match[-1].strip()
            else:
                think_match = re.findall(r'<think>(.*?)</think>', completion_text, re.DOTALL)
                if think_match:
                    thought = think_match[-1].strip()
                else:
                    # Fallback: take text before the last action tag
                    last_search_end = completion_text.rfind('</search>')
                    last_access_end = completion_text.rfind('</access>')
                    last_answer_end = completion_text.rfind('</answer>')
                    last_action_pos = max(
                        completion_text.rfind('<search>', 0, last_search_end) if last_search_end != -1 else -1,
                        completion_text.rfind('<access>', 0, last_access_end) if last_access_end != -1 else -1,
                        completion_text.rfind('<answer>', 0, last_answer_end) if last_answer_end != -1 else -1
                    )
                    if last_action_pos != -1:
                        thought = completion_text[:last_action_pos].strip()
                    else:
                        thought = completion_text
            # Remove any remaining tool tags from the extracted thought
            thought = re.sub(r'</?(search|access|answer|think|thought)>', '', thought).strip()

            tool_calls_raw = agent.consume_llm_response(llm_response, completion_text)
            tool_calls = self.parse_tool_calls(tool_calls_raw)

            # --- Display Logic ---
            if is_browsing:
                # 仅当本轮确实显示了页面内容时，才生成对应页面的总结，避免重复总结
                if thought and displayed_page_content:
                    title = f"Summarize (Page {page_counter})"
                    await self.add_step(query_id, "summarize", title, thought)
                    page_counter += 1
                elif thought and not displayed_page_content:
                    # 如果有思考但没有显示页面内容，可能是在处理已缓存的页面，跳过总结
                    pass
                consecutive_thoughts = []
            else:  # In thinking phase
                if thought:
                    consecutive_thoughts.append(thought)
                
                if tool_calls and consecutive_thoughts:
                    full_thought = "\n\n".join(consecutive_thoughts)
                    await self.add_step(query_id, "thinking", f"Thinking", full_thought)
                    turn_counter += 1
                    consecutive_thoughts = []

            # --- Action Processing ---
            if tool_calls:
                access_tool_used = await self.process_tool_calls(query_id, tool_calls, agent, search_client, process)
                if access_tool_used:
                    # Reset page counter for the new browsing session
                    page_counter = 1
        
        if consecutive_thoughts:
            full_thought = "\n\n".join(consecutive_thoughts)
            await self.add_step(query_id, "thinking", f"Thinking (Turn {turn_counter})", full_thought)
            
            if self.active_queries.get(query_id, {}).get("status") == "running":
                 await self.add_step(query_id, "error", "Error", "Agent stopped due to maximum turn limit without providing a final answer.")

    async def process_tool_calls(self, query_id: str, tool_calls, agent, search_client, process) -> bool:
        """Process a list of tool calls, like search or access"""
        access_tool_used = False
        for tool_call in tool_calls:
            if self.active_queries[query_id]["status"] != "running":
                break
                
            if tool_call["type"] == "search":
                query = tool_call["query"]
                await self.add_step(query_id, "search", "Search", f"<strong>Search query:</strong> {query}")
                
                search_result = await self.process_single_search_query(search_client, query)
                
                # We still need to call consume_tool_response to add jobs to the queue
                tool_response_data = {"type": "search"}
                if search_result:
                    documents, urls = self._extract_search_results(search_result)
                    tool_response_data["documents"] = documents
                    tool_response_data["urls"] = urls
                    
                    results_html = "<ol>"
                    for i, doc in enumerate(documents[:3]):
                        if isinstance(doc, dict):
                            title = doc.get('title', 'No Title')
                            url = doc.get('url', urls[i] if i < len(urls) else '#')
                            results_html += f"<li><a href='{url}' target='_blank'>{title}</a></li><br>"
                        elif isinstance(doc, str):
                            title = doc
                            url = urls[i] if i < len(urls) else '#'
                            results_html += f"<li><a href='{url}' target='_blank'>{title}</a></li><br>"
                    results_html += "</ol>"

                    if len(documents) > 0:
                        await self.add_step(query_id, "search", "Search results", 
                                     f"<strong>Top {min(len(documents), 3)} results:<br></strong>{results_html}")
                else:
                    await self.add_step(query_id, "search", "Search results", "No results found")
                    tool_response_data["documents"] = []
                    tool_response_data["urls"] = []

                # Pass None for reasoning agent as it handles this internally
                if hasattr(agent, "consume_tool_response"):
                    agent.consume_tool_response(tool_response_data)
            
            elif tool_call["type"] == "access":
                access_tool_used = True
                url = tool_call["url"]
                await self.add_step(query_id, "access", "Access page", f"<strong>Accessing URL:</strong> {url}")
                
                access_result = await self.process_single_access_query(search_client, url)
                if access_result:
                    page = self._extract_page_content(access_result)
                        
                    await self.add_step(query_id, "access", "Page content", page if page else f"No information found in {url}")

                    tool_response = {"type": "access", "page": page}
                    if hasattr(agent, "consume_tool_response"):
                        agent.consume_tool_response(tool_response)
                else:
                    await self.add_step(query_id, "access", "Page access failed", f"Cannot access: {url}")
            
            elif tool_call["type"] == "answer":
                answer = tool_call["content"]
                # Clean up answer by removing query ID if present
                answer = re.sub(r'\s*\(query\s+[a-f0-9-]+\)\s*$', '', answer).strip()
                await self.add_step(query_id, "answer", "Provide answer", answer)
                self.active_queries[query_id]["status"] = "completed"
                self.active_queries[query_id]["pred_answer"] = answer
                logger.info(f"Agent provided answer: {answer} (query {query_id})")
                process["pred_answer"] = answer
                process["running"] = False

        return access_tool_used

    async def add_step(self, query_id: str, step_type: str, title: str, content: str):
        """Add a new step to the query trajectory"""
        if query_id in self.active_queries:
            query_data = self.active_queries[query_id]
            step_id = len(query_data["steps"]) + 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            step = AgentStep(step_id=step_id, type=step_type, title=title, content=content, timestamp=timestamp)
            query_data["steps"].append(step)

    def _extract_search_results(self, search_result):
        """Extract documents and URLs from search result"""
        documents, urls = [], []
        
        if isinstance(search_result, dict):
            documents = search_result.get("documents", []) or []
            urls = search_result.get("urls", []) or []
        elif isinstance(search_result, list):
            for result in search_result:
                if isinstance(result, dict):
                    documents.extend(result.get("documents", []) or [])
                    urls.extend(result.get("urls", []) or [])
        
        return documents, urls

    def _extract_page_content(self, access_result):
        """Extract page content from access result"""
        # access_result is typically a list of dictionaries from access_async
        if isinstance(access_result, list) and len(access_result) > 0:
            # Get the first result (usually only one URL is accessed)
            first_result = access_result[0]
            if isinstance(first_result, dict):
                return first_result.get("page", "") or ""
        elif isinstance(access_result, dict):
            return access_result.get("page", "") or ""
        elif isinstance(access_result, str):
            return access_result
        else:
            return str(access_result) if access_result else ""

    async def process_single_llm_query(self, prompt: str, sampling_params: Dict, request, qid=None) -> CompatibleLLMResponse:
        """Process single LLM query"""
        sampling_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_new_tokens": request.max_tokens_per_call,
        }
        
        if sampling_params.get("stop") and isinstance(sampling_params["stop"], list):
            sampling_kwargs["stop"] = sampling_params["stop"]
        
        output = await self.llm.async_generate(prompt, sampling_kwargs)
        text = output.get('text', '')
        
        # Complete incomplete tool calls (vLLM doesn't return stop sequences)
        text = self.complete_incomplete_tool_calls(text, sampling_params.get("stop", []))
        
        # Post-process: truncate at first complete tool call
        if sampling_params.get("stop") and sampling_params["stop"] != ["</think>"]:
            text = self.truncate_at_first_complete_tool_call(text)
        
        input_tokens = output_tokens = None
        if self.tokenizer:
            try:
                input_tokens = self.tokenizer.encode(prompt)
                if text:
                    output_tokens = self.tokenizer.encode(text)
            except Exception as e:
                logger.warning(f"Tokenizer encoding failed: {e}")
        
        return CompatibleLLMResponse(
            text=text,
            input_len=len(input_tokens) if input_tokens else None,
            input_tokens=input_tokens,
            output_len=len(output_tokens) if output_tokens else None,
            output_tokens=output_tokens,
            finish_reason=output.get('finish_reason')
        )

    async def process_single_search_query(self, search_client, query: str, topk: int = 3):
        """Process single search query"""
        req_meta = {"queries": [query], "topk": topk, "return_scores": False}
        results = await search_client.query_async(req_meta)
        return results if results else None

    async def process_single_access_query(self, search_client, url: str):
        """Process single page access query"""
        results = await search_client.access_async([url])
        return results if results else None

    def truncate_at_first_complete_tool_call(self, text: str) -> str:
        """Truncate text at first complete tool call"""
        patterns = [r'(<search>.*?</search>)', r'(<access>.*?</access>)', r'(<answer>.*?</answer>)']
        
        earliest_end = len(text)
        found_tool_call = False
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                tool_call_end = match.end()
                if tool_call_end < earliest_end:
                    earliest_end = tool_call_end
                    found_tool_call = True
        
        return text[:earliest_end] if found_tool_call else text

    def convert_agent_tool_calls_to_dict(self, agent_tool_calls):
        """Convert agent tool calls to dict format"""
        dict_tool_calls = []
        
        for tool_call_str in agent_tool_calls:
            for tag, call_type, key in [
                ('search', 'search', 'query'), 
                ('access', 'access', 'url'), 
                ('answer', 'answer', 'content')
            ]:
                match = re.search(f'<{tag}>(.*?)</{tag}>', tool_call_str, re.DOTALL)
                if match:
                    dict_tool_calls.append({"type": call_type, key: match.group(1).strip()})
                    break
        
        return dict_tool_calls

    def complete_incomplete_tool_calls(self, text: str, stop_sequences: List[str]) -> str:
        """Complete incomplete tool calls by adding missing end tags"""
        if not text.strip():
            return text
        
        completed_text = text
        tool_patterns = [('<search>', '</search>'), ('<access>', '</access>'), ('<answer>', '</answer>'), ('<think>', '</think>')]
        
        for start_tag, end_tag in tool_patterns:
            if end_tag in stop_sequences and start_tag in completed_text:
                start_positions = [pos for pos in range(len(completed_text)) if completed_text.startswith(start_tag, pos)]
                
                for start_pos in reversed(start_positions):
                    text_after_start = completed_text[start_pos + len(start_tag):]
                    
                    if end_tag not in text_after_start:
                        next_tool_call_pos = len(text_after_start)
                        for other_start, _ in tool_patterns:
                            if other_start != start_tag and other_start in text_after_start:
                                pos = text_after_start.find(other_start)
                                if pos < next_tool_call_pos:
                                    next_tool_call_pos = pos
                        
                        text_to_check = text_after_start[:next_tool_call_pos]
                        if end_tag not in text_to_check:
                            logger.info(f"Completing incomplete tool call: {start_tag}... with {end_tag}")
                            if next_tool_call_pos < len(text_after_start):
                                insert_pos = start_pos + len(start_tag) + next_tool_call_pos
                                completed_text = completed_text[:insert_pos] + end_tag + completed_text[insert_pos:]
                            else:
                                completed_text = completed_text.rstrip() + end_tag
                            break
        
        return completed_text

    def parse_tool_calls(self, tool_calls_raw):
        """Parse raw tool calls from agent response."""
        tool_calls = []
        for tool_call_str in tool_calls_raw:
            for tag, call_type, key in [
                ('search', 'search', 'query'), 
                ('access', 'access', 'url'), 
                ('answer', 'answer', 'content')
            ]:
                match = re.search(f'<{tag}>(.*?)</{tag}>', tool_call_str, re.DOTALL)
                if match:
                    tool_calls.append({"type": call_type, key: match.group(1).strip()})
                    break
        return tool_calls

def create_app(llm_url: str = None, model_name: str = "default", api_key: str = "EMPTY"):
    """Create FastAPI application"""
    service = AsearcherVisualDemo(llm_url, model_name, api_key)
    return service.app

def get_app():
    """Get app instance for uvicorn reload mode"""
    llm_url = os.environ.get('ASEARCHER_LLM_URL', 'http://0.0.0.0:50000')
    model_name = os.environ.get('ASEARCHER_MODEL_NAME', 'default')
    api_key = os.environ.get('ASEARCHER_API_KEY', 'EMPTY')
    return create_app(llm_url, model_name, api_key)

app = None

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ASearcher Visual Demo Service")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--llm-url", default="http://0.0.0.0:50000", help="vLLM server URL")
    parser.add_argument("--model-name", default="ASearcher-Web-7B", help="Model name")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting ASearcher Visual Demo Service")
    print(f"   Service address: http://{args.host}:{args.port}")
    print(f"   vLLM server: {args.llm_url}")
    print(f"   Model name: {args.model_name}")
    print(f"   API key: {args.api_key}")
    
    if args.reload:
        os.environ.update({
            'ASEARCHER_LLM_URL': args.llm_url,
            'ASEARCHER_MODEL_NAME': args.model_name,
            'ASEARCHER_API_KEY': args.api_key
        })
        if args.model_name != "default":
            os.environ['MODEL_PATH'] = args.model_name
        
        print("🔄 Auto reload mode enabled")
        uvicorn.run("asearcher_visual_demo:get_app", host=args.host, port=args.port, reload=True)
    else:
        app = create_app(args.llm_url, args.model_name, args.api_key)
        uvicorn.run(app, host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
