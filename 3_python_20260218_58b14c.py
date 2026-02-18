"""
Task Agent - Agente autônomo para execução de tarefas complexas
Inspirado em AutoGPT com planning e execução multi-step
"""

import asyncio
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json
import re

# LLM imports
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain.agents import (
    Tool, 
    AgentExecutor, 
    LLMSingleActionAgent,
    AgentOutputParser
)
from langchain.schema import AgentAction, AgentFinish

# Custom tools
from integrations.calendar_sync import CalendarTool
from integrations.email_agent import EmailTool
from integrations.whatsapp_bot import WhatsAppTool
from integrations.home_assistant import HomeAssistantTool

class TaskStatus(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskAgent:
    """
    Agente autônomo para execução de tarefas complexas
    Features:
    - Planejamento multi-step
    - Execução de tarefas em background
    - Gestão de dependências
    - Re-tentativas inteligentes
    - Notificações de progresso
    """
    
    def __init__(self, memory_graph):
        self.memory_graph = memory_graph
        self.llm = OpenAI(temperature=0.2, model_name="gpt-4")
        
        # Registro de tarefas
        self.tasks = {}
        self.active_tasks = {}
        self.task_queue = asyncio.Queue()
        
        # Tools disponíveis
        self.tools = self._setup_tools()
        
        # Agent executor
        self.agent = self._setup_agent()
        
        # Worker tasks
        self.workers = []
        self._start_workers()
        
    def _setup_tools(self) -> List[Tool]:
        """Configura ferramentas disponíveis para o agente"""
        return [
            Tool(
                name="Calendar",
                func=CalendarTool().schedule_event,
                description="Agenda eventos no Google Calendar"
            ),
            Tool(
                name="Email",
                func=EmailTool().send_email,
                description="Envia emails"
            ),
            Tool(
                name="WhatsApp",
                func=WhatsAppTool().send_message,
                description="Envia mensagens WhatsApp"
            ),
            Tool(
                name="SmartHome",
                func=HomeAssistantTool().control_device,
                description="Controla dispositivos da casa"
            ),
            Tool(
                name="WebSearch",
                func=self._web_search,
                description="Busca informações na web"
            ),
            Tool(
                name="Reminder",
                func=self._set_reminder,
                description="Configura lembretes"
            ),
            StructuredTool.from_function(
                name="TaskPlanner",
                func=self._plan_task,
                description="Planeja execução de tarefas complexas"
            )
        ]
    
    def _setup_agent(self):
        """Configura o agente LLM para planejamento"""
        prompt = PromptTemplate(
            template="""Você é um agente autônomo de execução de tarefas.
            
Tarefa: {task_description}

Contexto:
- Usuário: {user_id}
- Prioridade: {priority}
- Dependências: {dependencies}
- Recursos disponíveis: {available_tools}

Crie um plano detalhado para executar esta tarefa, considerando:
1. Passos necessários em ordem
2. Ferramentas necessárias para cada passo
3. Possíveis pontos de falha e contingências
4. Estimativa de tempo para cada passo

Formato de resposta (JSON):
{{
    "steps": [
        {{
            "step_id": 1,
            "description": "descrição do passo",
            "tool": "nome_da_ferramenta",
            "input": "parâmetros necessários",
            "estimated_time": "tempo estimado",
            "fallback": "plano alternativo se falhar"
        }}
    ],
    "total_estimated_time": "tempo total",
    "critical_path": ["step_ids"],
    "success_criteria": "critérios de sucesso"
}}
""",
            input_variables=[
                "task_description", "user_id", "priority", 
                "dependencies", "available_tools"
            ]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain
    
    async def process(self, context) -> Dict:
        """
        Processa solicitação de tarefa
        """
        task_description = context.input_text
        
        # 1. Extrair intenção e parâmetros
        task_info = await self._extract_task_info(task_description, context)
        
        # 2. Verificar se é uma tarefa executável
        if task_info["type"] == "simple_query":
            return {
                "agent": "task",
                "type": "query",
                "text": "Isso parece uma pergunta, não uma tarefa.",
                "confidence": 0.8
            }
        
        # 3. Criar tarefa
        task_id = self._create_task(task_info, context)
        
        # 4. Planejar execução
        plan = await self._plan_task(task_id)
        
        # 5. Adicionar à fila
        await self.task_queue.put(task_id)
        
        return {
            "agent": "task",
            "type": "task_created",
            "text": f"✅ Tarefa criada: {task_info['description']}",
            "task_id": task_id,
            "plan_summary": self._summarize_plan(plan),
            "estimated_completion": plan.get("total_estimated_time", "desconhecido"),
            "confidence": 0.95
        }
    
    async def _extract_task_info(self, description: str, context) -> Dict:
        """
        Extrai informações estruturadas da descrição da tarefa
        """
        prompt = f"""
Analise a seguinte solicitação e extraia informações estruturadas sobre a tarefa:

Solicitação: {description}

Contexto do usuário: {context.user_preferences}

Extraia:
1. Tipo de tarefa (reminder/email/calendar/smart_home/complex)
2. Descrição clara
3. Parâmetros específicos (data/hora/destinatário/etc)
4. Prioridade (low/medium/high/critical)
5. Dependências (se houver)

Responda em JSON.
"""
        
        response = await self.llm.agenerate([prompt])
        
        try:
            return json.loads(response.generations[0][0].text)
        except:
            # Fallback
            return {
                "type": "complex",
                "description": description,
                "priority": "medium",
                "parameters": {},
                "dependencies": []
            }
    
    def _create_task(self, task_info: Dict, context) -> str:
        """Cria registro da tarefa"""
        import uuid
        
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "id": task_id,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "description": task_info["description"],
            "type": task_info["type"],
            "priority": TaskPriority[task_info.get("priority", "MEDIUM").upper()],
            "parameters": task_info.get("parameters", {}),
            "dependencies": task_info.get("dependencies", []),
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "history": []
        }
        
        return task_id
    
    async def _plan_task(self, task_id: str) -> Dict:
        """Planeja execução da tarefa"""
        task = self.tasks[task_id]
        
        # Atualizar status
        task["status"] = TaskStatus.PLANNING
        task["updated_at"] = datetime.now().isoformat()
        
        # Gerar plano
        plan = await self.agent.apredict(
            task_description=task["description"],
            user_id=task["user_id"],
            priority=task["priority"].name,
            dependencies=json.dumps(task["dependencies"]),
            available_tools=[t.name for t in self.tools]
        )
        
        try:
            plan_dict = json.loads(plan)
        except:
            # Fallback para plano simples
            plan_dict = {
                "steps": [
                    {
                        "step_id": 1,
                        "description": "Executar tarefa",
                        "tool": "unknown",
                        "input": task["description"],
                        "estimated_time": "5 minutos"
                    }
                ],
                "total_estimated_time": "5 minutos"
            }
        
        # Armazenar plano
        task["plan"] = plan_dict
        task["status"] = TaskStatus.PENDING
        
        return plan_dict
    
    async def _execute_task(self, task_id: str):
        """Executa uma tarefa (worker)"""
        task = self.tasks[task_id]
        
        task["status"] = TaskStatus.EXECUTING
        task["started_at"] = datetime.now().isoformat()
        
        # Executar cada passo do plano
        for step in task.get("plan", {}).get("steps", []):
            step_result = await self._execute_step(step, task)
            
            task["history"].append({
                "step": step["step_id"],
                "result": step_result,
                "timestamp": datetime.now().isoformat()
            })
            
            if step_result.get("status") == "failed":
                # Tentar fallback
                if "fallback" in step:
                    fallback_result = await self._execute_fallback(step, task)
                    if fallback_result.get("status") == "failed":
                        task["status"] = TaskStatus.FAILED
                        await self._notify_failure(task)
                        return
        
        # Tarefa concluída
        task["status"] = TaskStatus.COMPLETED
        task["completed_at"] = datetime.now().isoformat()
        
        # Notificar sucesso
        await self._notify_success(task)
    
    async def _execute_step(self, step: Dict, task: Dict) -> Dict:
        """Executa um passo específico"""
        tool_name = step.get("tool")
        
        # Encontrar tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        
        if not tool:
            return {
                "status": "failed",
                "error": f"Ferramenta {tool_name} não encontrada"
            }
        
        try:
            # Executar tool
            if isinstance(tool, StructuredTool):
                result = await tool.acall(**step.get("input", {}))
            else:
                result = await tool.acall(step.get("input", ""))
            
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _execute_fallback(self, step: Dict, task: Dict) -> Dict:
        """Executa plano de fallback"""
        fallback_desc = step.get("fallback", "")
        
        # Criar passo alternativo
        fallback_step = {
            "step_id": step["step_id"],
            "description": fallback_desc,
            "tool": "unknown",
            "input": {"description": fallback_desc}
        }
        
        return await self._execute_step(fallback_step, task)
    
    def _start_workers(self, num_workers: int = 3):
        """Inicia workers para processar fila de tarefas"""
        async def worker():
            while True:
                task_id = await self.task_queue.get()
                try:
                    await self._execute_task(task_id)
                except Exception as e:
                    print(f"Erro no worker: {e}")
                finally:
                    self.task_queue.task_done()
        
        # Iniciar workers
        for _ in range(num_workers):
            worker_task = asyncio.create_task(worker())
            self.workers.append(worker_task)
    
    async def _web_search(self, query: str) -> str:
        """Busca na web"""
        # Implementar integração com API de busca
        return f"Resultados para: {query}"
    
    async def _set_reminder(self, reminder_data: str) -> str:
        """Configura lembrete"""
        # Implementar lógica de lembretes
        return f"Lembrete configurado: {reminder_data}"
    
    async def _notify_success(self, task: Dict):
        """Notifica usuário sobre sucesso"""
        notification = {
            "type": "task_completed",
            "task_id": task["id"],
            "message": f"✅ Tarefa concluída: {task['description']}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Enviar notificação (WhatsApp/Email/Push)
        await self._send_notification(task["user_id"], notification)
    
    async def _notify_failure(self, task: Dict):
        """Notifica usuário sobre falha"""
        notification = {
            "type": "task_failed",
            "task_id": task["id"],
            "message": f"❌ Falha na tarefa: {task['description']}",
            "error": task["history"][-1]["result"].get("error", "Erro desconhecido"),
            "timestamp": datetime.now().isoformat()
        }
        
        await self._send_notification(task["user_id"], notification)
    
    async def _send_notification(self, user_id: str, notification: Dict):
        """Envia notificação para o usuário"""
        # Implementar canais de notificação
        print(f"Notificação para {user_id}: {notification}")
    
    def _summarize_plan(self, plan: Dict) -> str:
        """Cria resumo do plano para o usuário"""
        steps = len(plan.get("steps", []))
        time_est = plan.get("total_estimated_time", "desconhecido")
        
        return f"Plano criado com {steps} passos. Tempo estimado: {time_est}"
    
    async def get_task_status(self, task_id: str) -> Dict:
        """Retorna status de uma tarefa"""
        if task_id not in self.tasks:
            return {"error": "Tarefa não encontrada"}
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task_id,
            "status": task["status"].value,
            "progress": len(task["history"]) / len(task.get("plan", {}).get("steps", [])),
            "created_at": task["created_at"],
            "completed_at": task.get("completed_at"),
            "history": task["history"]
        }