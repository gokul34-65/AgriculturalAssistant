"""
Fixed Web Search Module with Better Error Handling
This fixes the 'NoneType' has no attribute 'get' error
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import time
from urllib.parse import quote_plus
import re


class WebSearchService:
    """Web search service using DuckDuckGo"""
    
    def __init__(self, cache_duration_minutes: int = 60):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.base_url = "https://api.duckduckgo.com/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search using DuckDuckGo"""
        try:
            cache_key = f"ddg_{query}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_data
            
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Get abstract/instant answer if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Answer'),
                    'snippet': data.get('Abstract', ''),
                    'url': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Instant Answer'
                })
            
            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo Related'
                    })
            
            self.cache[cache_key] = (results, datetime.now())
            return results[:max_results]
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def search(self, query: str, max_results: int = 5, api_key: Optional[str] = None) -> List[Dict]:
        """Main search function"""
        return self.search_duckduckgo(query, max_results)
    
    def search_agricultural_news(self, topic: str = "Tamil Nadu agriculture", days: int = 7) -> List[Dict]:
        """Search for agricultural news"""
        query = f"{topic} news last {days} days"
        return self.search(query, max_results=10)
    
    def search_government_schemes(self, scheme_type: str = "agriculture subsidy", state: str = "Tamil Nadu") -> List[Dict]:
        """Search for government schemes"""
        queries = [
            f"{state} {scheme_type} 2024",
            f"{scheme_type} {state} farmers"
        ]
        
        all_results = []
        for query in queries:
            results = self.search(query, max_results=3)
            all_results.extend(results)
            time.sleep(1)
        
        # Remove duplicates
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        return unique_results[:10]
    
    def search_crop_advisory(self, crop: str, issue: Optional[str] = None, district: Optional[str] = None) -> List[Dict]:
        """Search for crop advisories"""
        query_parts = [crop]
        
        if issue:
            query_parts.append(issue)
        
        if district:
            query_parts.append(district)
        else:
            query_parts.append("Tamil Nadu")
        
        query_parts.append("TNAU advisory")
        query = " ".join(query_parts)
        
        return self.search(query, max_results=5)


class AgriculturalSearchAssistant:
    """Agricultural search assistant"""
    
    def __init__(self, web_search: WebSearchService):
        self.web_search = web_search
        self.search_patterns = {
            'schemes': ['scheme', 'subsidy', 'yojana', 'loan', 'insurance', 'grant'],
            'prices': ['price', 'rate', 'msp', 'market', 'agmarknet'],
            'news': ['news', 'update', 'alert', 'announcement', 'latest'],
            'advisory': ['advisory', 'recommendation', 'tnau', 'guideline'],
            'research': ['research', 'study', 'paper', 'innovation', 'technology']
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        for category, keywords in self.search_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def search_with_context(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Search with user context"""
        category = self.classify_query(query)
        
        enhanced_query = query
        if user_context:
            if user_context.get('district'):
                enhanced_query += f" {user_context['district']}"
            if user_context.get('crop'):
                enhanced_query += f" {user_context['crop']}"
        
        if 'tamil nadu' not in enhanced_query.lower():
            enhanced_query += " Tamil Nadu"
        
        # Search based on category
        if category == 'schemes':
            results = self.web_search.search_government_schemes(
                scheme_type=query,
                state=user_context.get('state', 'Tamil Nadu') if user_context else 'Tamil Nadu'
            )
        elif category == 'news':
            results = self.web_search.search_agricultural_news(topic=enhanced_query)
        elif category == 'advisory':
            crop = user_context.get('crop', 'agriculture') if user_context else 'agriculture'
            district = user_context.get('district') if user_context else None
            results = self.web_search.search_crop_advisory(
                crop=crop,
                district=district
            )
        else:
            results = self.web_search.search(enhanced_query, max_results=5)
        
        return {
            'query': query,
            'enhanced_query': enhanced_query,
            'category': category,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def format_results_for_llm(self, search_results: List[Dict]) -> str:
        """Format search results"""
        if not search_results:
            return "No relevant web results found."
        
        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(search_results, 1):
            formatted += f"{i}. {result.get('title', 'No title')}\n"
            formatted += f"   {result.get('snippet', 'No description')}\n"
            formatted += f"   Source: {result.get('url', 'No URL')}\n\n"
        
        return formatted


class EnhancedRAGWithWebSearch:
    """Enhanced RAG with web search - FIXED VERSION"""
    
    def __init__(self, rag_system, web_search_service: Optional[WebSearchService] = None, serpapi_key: Optional[str] = None):
        self.rag = rag_system
        self.web_search = web_search_service or WebSearchService()
        self.serpapi_key = serpapi_key
        self.search_assistant = AgriculturalSearchAssistant(self.web_search)
    
    def should_use_web_search(self, query: str) -> bool:
        """Determine if query needs web search"""
        web_search_keywords = [
            'latest', 'current', 'recent', 'today', '2024', '2025', '2026',
            'news', 'update', 'new scheme', 'latest price', 'current rate',
            'announcement', 'alert'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_search_keywords)
    
    def query(
        self,
        question: str,
        user_context: Optional[Dict] = None,
        k: int = 3,
        use_web_search: Optional[bool] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        FIXED: Enhanced query with proper error handling
        """
        try:
            # Determine if we should use web search
            if use_web_search is None:
                use_web_search = self.should_use_web_search(question)
            
            # Get RAG results
            print(f"📚 Searching knowledge base...")
            rag_results = self.rag.query(
                question=question,
                k=k,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # CRITICAL FIX: Check if rag_results is None
            if rag_results is None:
                rag_results = {
                    'question': question,
                    'answer': 'No results found in knowledge base.',
                    'sources': [],
                    'retrieved_chunks': []
                }
            
            # Get web search results if needed
            web_results = None
            if use_web_search:
                print(f"🌐 Searching web for latest information...")
                try:
                    web_results = self.search_assistant.search_with_context(
                        query=question,
                        user_context=user_context
                    )
                except Exception as e:
                    print(f"Web search failed: {e}")
                    web_results = None
            
            # Combine results
            if web_results and web_results.get('results'):
                # Format web results
                web_context = self.search_assistant.format_results_for_llm(
                    web_results['results']
                )
                
                # Create enhanced prompt
                enhanced_prompt = f"""Based on the following information, please answer the question.

Knowledge Base Information:
{rag_results.get('answer', 'No knowledge base results')}

Latest Web Information:
{web_context}

Question: {question}

Please provide a comprehensive answer combining both sources.

Answer:"""
                
                # Query LLM with enhanced context
                print(f"🤖 Generating enhanced response...")
                
                try:
                    response = requests.post(
                        f"{self.rag.ollama_url}/api/generate",
                        json={
                            "model": self.rag.model_name,
                            "prompt": enhanced_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        final_answer = response.json().get('response', '').strip()
                    else:
                        final_answer = rag_results.get('answer', 'Error generating response')
                        
                except Exception as e:
                    print(f"Error generating enhanced response: {e}")
                    final_answer = rag_results.get('answer', 'Error generating response')
                
                # Combine sources
                rag_sources = rag_results.get('sources', [])
                web_sources = [r['url'] for r in web_results['results'] if r.get('url')]
                all_sources = list(set(rag_sources + web_sources))
                
                return {
                    'question': question,
                    'answer': final_answer,
                    'sources': all_sources,
                    'rag_sources': rag_sources,
                    'web_sources': web_sources,
                    'used_web_search': True,
                    'web_results_count': len(web_results['results']),
                    'retrieved_chunks': rag_results.get('retrieved_chunks', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Return RAG results only
                return {
                    'question': rag_results.get('question', question),
                    'answer': rag_results.get('answer', 'No answer available'),
                    'sources': rag_results.get('sources', []),
                    'rag_sources': rag_results.get('sources', []),
                    'web_sources': [],
                    'used_web_search': False,
                    'web_results_count': 0,
                    'retrieved_chunks': rag_results.get('retrieved_chunks', []),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error in enhanced query: {e}")
            # Return safe fallback
            return {
                'question': question,
                'answer': f'Error processing query: {str(e)}',
                'sources': [],
                'rag_sources': [],
                'web_sources': [],
                'used_web_search': False,
                'web_results_count': 0,
                'retrieved_chunks': [],
                'timestamp': datetime.now().isoformat()
            }


if __name__ == "__main__":
    print("Fixed Web Search Module")
    print("=" * 60)
    print("This version has better error handling")
    print("Testing basic search...")
    
    web_search = WebSearchService()
    results = web_search.search("Tamil Nadu agriculture schemes 2024", max_results=3)
    
    print(f"\nFound {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.get('title', 'No title')}")