"""
Knowledge Agent with Advanced RAG (Retrieval-Augmented Generation)
Multi-source knowledge retrieval with real-time web search
"""

import asyncio
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
import aiohttp
import json

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    WebBaseLoader, 
    TextLoader, 
    PDFLoader,
    UnstructuredURLLoader
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationSummaryMemory

# Custom imports
from core.vector_store import VectorStore
from knowledge.web_crawler import SmartCrawler
from knowledge.document_processor import DocumentProcessor

class KnowledgeAgent:
    """
    Agente especializado em conhecimento com capacidades:
    - RAG (Retrieval-Augmented Generation)
    - Web search em tempo real
    - Processamento de documentos
    - Memória de conhecimento de longo prazo
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.crawler = SmartCrawler()
        self.doc_processor = DocumentProcessor()
        
        # LLM config
        self.llm = OpenAI(temperature=0.3, model_name="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        
        # Retrieval setup
        self.retriever = self._setup_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        
        # Cache com TTL
        self.knowledge_cache = {}
        self.cache_ttl = timedelta(hours=24)
        
        # Knowledge graph
        self.knowledge_graph = {}
        
    def _setup_retriever(self):
        """Configura retrievers com compressão contextual"""
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        # Compressor para melhor relevância
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    async def process(self, context) -> Dict:
        """
        Processa consulta de conhecimento
        """
        query = context.input_text
        
        # 1. Verificar cache
        cached = self._check_cache(query)
        if cached:
            return cached
        
        # 2. Buscar em múltiplas fontes
        sources = await asyncio.gather(
            self._search_vector_store(query),
            self._search_web(query),
            self._search_documents(query),
            self._search_knowledge_graph(query)
        )
        
        # 3. Combinar e ranquear resultados
        all_docs = self._merge_and_rank_sources(sources)
        
        # 4. Gerar resposta com RAG
        response = await self._generate_response(query, all_docs, context)
        
        # 5. Atualizar cache e knowledge graph
        self._update_cache(query, response)
        await self._update_knowledge_graph(query, response)
        
        return {
            "agent": "knowledge",
            "text": response["answer"],
            "sources": response.get("sources", []),
            "confidence": response.get("confidence", 0.9),
            "type": "knowledge_retrieval"
        }
    
    async def _search_vector_store(self, query: str) -> List[Dict]:
        """Busca no vector store local"""
        results = self.vector_store.search(
            query=query,
            k=5,
            include_metadata=True
        )
        return results
    
    async def _search_web(self, query: str) -> List[Dict]:
        """Busca na web em tempo real"""
        # Determinar se precisa de web search
        if self._needs_web_search(query):
            web_results = await self.crawler.search(query, num_results=3)
            return web_results
        return []
    
    async def _search_documents(self, query: str) -> List[Dict]:
        """Busca em documentos processados"""
        return self.doc_processor.search(query, k=3)
    
    async def _search_knowledge_graph(self, query: str) -> List[Dict]:
        """Busca no grafo de conhecimento"""
        # Implementar busca em grafo
        related = []
        for node, connections in self.knowledge_graph.items():
            if query.lower() in node.lower():
                related.append({
                    "content": node,
                    "connections": connections,
                    "source": "knowledge_graph"
                })
        return related
    
    def _needs_web_search(self, query: str) -> bool:
        """Determina se precisa de dados atualizados da web"""
        time_sensitive_keywords = [
            "notícia", "hoje", "agora", "último", "atual", 
            "previsão", "tempo", "cotação", "preço"
        ]
        return any(kw in query.lower() for kw in time_sensitive_keywords)
    
    def _merge_and_rank_sources(self, sources: List[List[Dict]]) -> List[Dict]:
        """Combina e ranqueia fontes por relevância"""
        all_docs = []
        
        for source_list in sources:
            all_docs.extend(source_list)
        
        # Ranquear por score e freshness
        ranked = sorted(
            all_docs,
            key=lambda x: (
                x.get("score", 0) * 0.7 + 
                self._freshness_score(x.get("timestamp")) * 0.3
            ),
            reverse=True
        )
        
        return ranked[:10]  # Top 10
    
    def _freshness_score(self, timestamp: Optional[str]) -> float:
        """Calcula score baseado na atualidade"""
        if not timestamp:
            return 0.5
        
        try:
            doc_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - doc_time
            
            if age < timedelta(days=1):
                return 1.0
            elif age < timedelta(days=7):
                return 0.8
            elif age < timedelta(days=30):
                return 0.5
            else:
                return 0.2
        except:
            return 0.5
    
    async def _generate_response(self, 
                               query: str, 
                               docs: List[Dict],
                               context) -> Dict:
        """
        Gera resposta usando RAG com os documentos recuperados
        """
        # Preparar contexto
        context_text = "\n\n".join([
            f"[Fonte {i+1}]: {doc.get('content', '')}" 
            for i, doc in enumerate(docs[:5])
        ])
        
        # Prompt otimizado
        prompt = f"""Com base nas seguintes fontes de informação, responda à pergunta do usuário.

Fontes disponíveis:
{context_text}

Pergunta: {query}

Contexto adicional:
- Idioma: {context.input_language}
- Emoção detectada: {context.detected_emotion}
- Histórico: {len(context.conversation_history)} mensagens anteriores

Forneça uma resposta precisa, bem fundamentada e no mesmo idioma da pergunta.
"""
        
        # Gerar resposta
        response = await self.llm.agenerate([prompt])
        
        # Extrair fontes usadas
        used_sources = []
        for doc in docs[:3]:
            if doc.get("source"):
                used_sources.append({
                    "title": doc.get("title", "Fonte"),
                    "url": doc.get("url"),
                    "relevance": doc.get("score", 0)
                })
        
        return {
            "answer": response.generations[0][0].text,
            "sources": used_sources,
            "confidence": 0.95 if docs else 0.7
        }
    
    def _check_cache(self, query: str) -> Optional[Dict]:
        """Verifica cache com fuzzy matching"""
        query_embedding = self._get_embedding(query)
        
        for cached_query, cached_response in self.knowledge_cache.items():
            cached_embedding = cached_response.get("embedding")
            if cached_embedding:
                similarity = np.dot(query_embedding, cached_embedding)
                if similarity > 0.92:  # Muito similar
                    # Verificar TTL
                    timestamp = cached_response.get("timestamp")
                    if datetime.fromisoformat(timestamp) + self.cache_ttl > datetime.now():
                        return cached_response["data"]
        
        return None
    
    def _update_cache(self, query: str, response: Dict):
        """Atualiza cache com embedding"""
        self.knowledge_cache[query] = {
            "data": response,
            "embedding": self._get_embedding(query),
            "timestamp": datetime.now().isoformat()
        }
        
        # Limpar cache antigo
        self._clean_cache()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Gera embedding para texto"""
        # Usar modelo de embeddings
        return np.random.randn(768)  # Placeholder
    
    def _clean_cache(self):
        """Remove entradas expiradas do cache"""
        now = datetime.now()
        expired = []
        
        for query, data in self.knowledge_cache.items():
            timestamp = datetime.fromisoformat(data["timestamp"])
            if now - timestamp > self.cache_ttl:
                expired.append(query)
        
        for query in expired:
            del self.knowledge_cache[query]
    
    async def _update_knowledge_graph(self, query: str, response: Dict):
        """Atualiza grafo de conhecimento com nova informação"""
        # Extrair entidades
        entities = self._extract_entities(query + " " + response["answer"])
        
        for entity in entities:
            if entity not in self.knowledge_graph:
                self.knowledge_graph[entity] = []
            
            # Adicionar conexões
            for other in entities:
                if other != entity:
                    self.knowledge_graph[entity].append({
                        "entity": other,
                        "strength": 1.0,
                        "timestamp": datetime.now().isoformat()
                    })
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extrai entidades do texto usando NER"""
        # Implementar NER com spaCy ou similar
        # Placeholder
        return text.split()[:5]