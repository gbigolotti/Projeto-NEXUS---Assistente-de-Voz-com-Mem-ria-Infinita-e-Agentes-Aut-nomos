"""
NEXUS Orchestrator Core
Gerencia múltiplos agentes com decisão contextual e memória de longo prazo
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict

# Machine Learning imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Internal imports
from agents.voice_agent import VoiceAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.task_agent import TaskAgent
from agents.memory_agent import MemoryAgent
from core.memory_graph import MemoryGraph
from core.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    VOICE = "voice"
    KNOWLEDGE = "knowledge"
    TASK = "task"
    MEMORY = "memory"
    SCHEDULER = "scheduler"

@dataclass
class Context:
    """Contexto rico da conversa"""
    user_id: str
    session_id: str
    timestamp: datetime
    input_text: str
    input_language: str
    detected_emotion: str
    conversation_history: List[Dict]
    user_preferences: Dict
    current_task: Optional[str] = None
    memory_embeddings: Optional[np.ndarray] = None

class NeuralRouter(nn.Module):
    """
    Router neural para decisão de agentes
    Arquitetura Transformer-based com attention
    """
    def __init__(self, input_dim=768, num_agents=5):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
            num_layers=4
        )
        self.agent_classifier = nn.Linear(input_dim, num_agents)
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.transformer(x)
        agent_logits = self.agent_classifier(encoded.mean(dim=1))
        confidence = self.confidence_net(encoded.mean(dim=1))
        return agent_logits, confidence

class NexusOrchestrator:
    """
    Orquestrador principal - O cérebro do sistema
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        logger.info("🚀 Inicializando NEXUS Orchestrator...")
        
        # Inicializar componentes neurais
        self._init_neural_components()
        
        # Inicializar agentes especializados
        self.agents = self._init_agents()
        
        # Memória distribuída
        self.memory_graph = MemoryGraph()
        self.vector_store = VectorStore()
        
        # Cache semântico
        self.semantic_cache = defaultdict(dict)
        
        # Métricas e monitoramento
        self.metrics = {
            'total_interactions': 0,
            'agent_usage': defaultdict(int),
            'avg_response_time': 0,
            'memory_hits': 0
        }
        
        logger.info("✅ NEXUS Orchestrator pronto!")
    
    def _init_neural_components(self):
        """Inicializa modelos neurais para routing e compreensão"""
        self.router = NeuralRouter()
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.intent_classifier = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased', num_labels=20
        )
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        
    def _init_agents(self) -> Dict[AgentType, Any]:
        """Inicializa todos os agentes especializados"""
        return {
            AgentType.VOICE: VoiceAgent(self.encoder, self.memory_graph),
            AgentType.KNOWLEDGE: KnowledgeAgent(self.vector_store),
            AgentType.TASK: TaskAgent(self.memory_graph),
            AgentType.MEMORY: MemoryAgent(self.vector_store, self.memory_graph),
            AgentType.SCHEDULER: SchedulerAgent()
        }
    
    async def process_input(self, 
                           user_input: Any,
                           user_id: str,
                           session_id: str,
                           input_type: str = "text") -> Dict[str, Any]:
        """
        Processa entrada do usuário com routing contextual
        """
        start_time = datetime.now()
        
        # 1. Criar contexto rico
        context = await self._create_context(
            user_input, user_id, session_id, input_type
        )
        
        # 2. Verificar cache semântico
        cached_response = self._check_semantic_cache(context)
        if cached_response:
            self.metrics['memory_hits'] += 1
            return cached_response
        
        # 3. Routing neural para agentes apropriados
        selected_agents, confidence = await self._route_to_agents(context)
        
        # 4. Execução paralela dos agentes
        tasks = [self._execute_agent(agent, context) for agent in selected_agents]
        agent_results = await asyncio.gather(*tasks)
        
        # 5. Síntese das respostas
        final_response = await self._synthesize_responses(
            agent_results, context
        )
        
        # 6. Atualizar memória de longo prazo
        await self._update_memory(context, final_response)
        
        # 7. Atualizar métricas
        self._update_metrics(start_time)
        
        return final_response
    
    async def _create_context(self, 
                             user_input: Any,
                             user_id: str,
                             session_id: str,
                             input_type: str) -> Context:
        """Cria contexto rico com embeddings e histórico"""
        
        # Converter para texto se for áudio
        if input_type == "audio":
            text, language, emotion = await self.agents[AgentType.VOICE].process_audio(
                user_input
            )
        else:
            text = user_input
            language = "pt"  # Detectar via modelo
            emotion = await self._detect_emotion(text)
        
        # Buscar memórias relevantes
        memory_embeddings = self.encoder.encode(text)
        relevant_memories = self.vector_store.search(memory_embeddings, k=5)
        
        # Buscar preferências do usuário
        user_preferences = await self.memory_graph.get_user_preferences(user_id)
        
        # Histórico da conversa
        history = await self.memory_graph.get_conversation_history(
            session_id, limit=10
        )
        
        return Context(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            input_text=text,
            input_language=language,
            detected_emotion=emotion,
            conversation_history=history,
            user_preferences=user_preferences,
            memory_embeddings=memory_embeddings
        )
    
    async def _route_to_agents(self, context: Context) -> tuple:
        """
        Usa o router neural para selecionar agentes apropriados
        """
        # Gerar embedding do contexto
        context_embedding = self.encoder.encode(
            context.input_text + " " + context.detected_emotion
        )
        
        # Converter para tensor
        context_tensor = torch.tensor(context_embedding).unsqueeze(0)
        
        # Router neural
        with torch.no_grad():
            agent_logits, confidence = self.router(context_tensor)
        
        # Selecionar agentes com confiança > threshold
        probs = torch.softmax(agent_logits, dim=-1)
        selected_indices = torch.where(probs[0] > 0.3)[0]
        
        # Mapear para tipos de agente
        agent_types = [list(AgentType)[i] for i in selected_indices]
        
        # Sempre incluir Memory e Knowledge se relevante
        if len(agent_types) < 2:
            agent_types.extend([AgentType.MEMORY, AgentType.KNOWLEDGE])
        
        return agent_types, confidence.item()
    
    async def _execute_agent(self, 
                           agent_type: AgentType, 
                           context: Context) -> Dict:
        """Executa um agente específico"""
        agent = self.agents[agent_type]
        
        try:
            result = await agent.process(context)
            self.metrics['agent_usage'][agent_type.value] += 1
            return result
        except Exception as e:
            logger.error(f"Erro no agente {agent_type}: {e}")
            return {"error": str(e), "agent": agent_type.value}
    
    async def _synthesize_responses(self, 
                                  agent_results: List[Dict],
                                  context: Context) -> Dict:
        """
        Sintetiza respostas de múltiplos agentes usando weighted fusion
        """
        # Pesos baseados em confiança e relevância
        weights = self._calculate_response_weights(agent_results, context)
        
        # Fusão das respostas
        if len(agent_results) == 1:
            primary_response = agent_results[0]
        else:
            # Fusão inteligente
            primary_response = await self._fuse_responses(
                agent_results, weights, context
            )
        
        # Adicionar metadados
        primary_response.update({
            "agents_used": [r.get("agent") for r in agent_results],
            "fusion_weights": weights,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "emotion": context.detected_emotion,
                "language": context.input_language
            }
        })
        
        return primary_response
    
    def _check_semantic_cache(self, context: Context) -> Optional[Dict]:
        """
        Verifica cache semântico para respostas similares
        """
        query_embedding = context.memory_embeddings
        
        for cached_query, cached_response in self.semantic_cache.items():
            similarity = np.dot(query_embedding, cached_query) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_query)
            )
            
            if similarity > 0.95:  # Threshold de similaridade
                logger.info(f"🎯 Cache hit com similaridade {similarity:.3f}")
                return cached_response
        
        return None
    
    async def _update_memory(self, context: Context, response: Dict):
        """
        Atualiza memória de longo prazo com a interação
        """
        # Criar nó da interação no grafo
        interaction_node = {
            "type": "interaction",
            "timestamp": datetime.now().isoformat(),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "input": context.input_text,
            "response": response.get("text", ""),
            "emotion": context.detected_emotion,
            "language": context.input_language
        }
        
        # Adicionar ao grafo de memória
        await self.memory_graph.add_node(interaction_node)
        
        # Criar embeddings e armazenar
        combined_text = context.input_text + " " + response.get("text", "")
        embedding = self.encoder.encode(combined_text)
        
        await self.vector_store.add_vector(
            embedding=embedding,
            metadata={
                "user_id": context.user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "interaction"
            }
        )
        
        # Atualizar cache semântico
        cache_key = tuple(context.memory_embeddings)
        self.semantic_cache[cache_key] = response
        
        # Manter cache limitado
        if len(self.semantic_cache) > 1000:
            # Remover itens mais antigos
            oldest_key = min(self.semantic_cache.keys(), 
                           key=lambda k: self.semantic_cache[k].get("timestamp"))
            del self.semantic_cache[oldest_key]
    
    def _update_metrics(self, start_time: datetime):
        """Atualiza métricas de performance"""
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.metrics['total_interactions'] += 1
        self.metrics['avg_response_time'] = (
            self.metrics['avg_response_time'] * 0.95 + elapsed * 0.05
        )
    
    async def _detect_emotion(self, text: str) -> str:
        """Detecta emoção no texto usando modelo fine-tuned"""
        # Implementar detecção de emoção com BERT fine-tuned
        emotions = ["neutro", "feliz", "triste", "raiva", "surpreso", "ansioso"]
        return np.random.choice(emotions)  # Placeholder
    
    def _calculate_response_weights(self, 
                                  results: List[Dict], 
                                  context: Context) -> List[float]:
        """Calcula pesos para fusão de respostas"""
        weights = []
        
        for result in results:
            if "error" in result:
                weights.append(0.0)
                continue
            
            # Peso baseado em confiança e relevância
            weight = result.get("confidence", 0.5)
            
            # Ajustar por relevância ao contexto
            if context.detected_emotion in result.get("emotions", []):
                weight *= 1.2
            
            weights.append(min(1.0, weight))
        
        # Normalizar
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    async def _fuse_responses(self, 
                            results: List[Dict], 
                            weights: List[float],
                            context: Context) -> Dict:
        """Fusão inteligente de respostas"""
        # Extrair textos
        texts = [r.get("text", "") for r in results]
        
        if not texts:
            return {"text": "Não consegui processar sua solicitação."}
        
        # Selecionar melhor resposta baseada em pesos
        best_idx = np.argmax(weights)
        
        # Combinar se múltiplas respostas relevantes
        if len([w for w in weights if w > 0.2]) > 1:
            fused_text = self._combine_responses(texts, weights, context)
        else:
            fused_text = texts[best_idx]
        
        return {
            "text": fused_text,
            "confidence": max(weights),
            "source_agents": [r.get("agent") for r in results]
        }
    
    def _combine_responses(self, 
                          texts: List[str], 
                          weights: List[float],
                          context: Context) -> str:
        """Combina múltiplas respostas de forma inteligente"""
        # Implementar combinação com base no tipo de resposta
        # e contexto da conversa
        combined = []
        
        for text, weight in zip(texts, weights):
            if weight > 0.1 and text:
                combined.append(text)
        
        return " ".join(combined)

class SchedulerAgent:
    """Agente de agendamento e lembretes"""
    async def process(self, context: Context):
        # Implementar integração com calendários
        return {
            "agent": "scheduler",
            "type": "scheduling",
            "text": "Agendamento processado",
            "confidence": 0.9
        }