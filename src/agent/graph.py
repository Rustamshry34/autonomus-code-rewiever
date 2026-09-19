"""
LangGraph agent workflow.
Tüm node'ları birleştirir ve pipeline akışını yönetir.
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    fetch_pr_node,
    analyze_code_node,
    run_tests_node,
    generate_review_node,
    generate_fix_node,
    post_results_node
)
import structlog

logger = structlog.get_logger()


# ============================================================================
# Graph Factory
# ============================================================================

def create_review_graph() -> StateGraph:
    """
    Code review agent graph'ını oluşturur ve compile eder.
    
    Pipeline akışı:
    
    START
      ↓
    fetch_pr_node
      ↓
    analyze_code_node
      ↓
      ├─ Kritik bulgu var → run_tests_node
      │                       ↓
      │                       ├─ Başarısız test → generate_fix_node
      │                       │                       ↓
      │                       │                       generate_review_node
      │                       │
      │                       └─ Tüm testler geçti → generate_review_node
      │
      └─ Kritik bulgu yok → generate_review_node
                              ↓
                         post_results_node
                              ↓
                            END
    
    Returns:
        Compiled LangGraph instance
    """
    logger.info("Creating review graph")
    
    # 1. Graph oluştur
    workflow = StateGraph(AgentState)
    
    # 2. Node'ları ekle
    workflow.add_node("fetch_pr", fetch_pr_node)
    workflow.add_node("analyze_code", analyze_code_node)
    workflow.add_node("run_tests", run_tests_node)
    workflow.add_node("generate_review", generate_review_node)
    workflow.add_node("generate_fix", generate_fix_node)
    workflow.add_node("post_results", post_results_node)
    
    logger.info(
        "Nodes added to graph",
        nodes=["fetch_pr", "analyze_code", "run_tests", 
               "generate_review", "generate_fix", "post_results"]
    )
    
    # 3. Entry point belirle
    workflow.set_entry_point("fetch_pr")
    
    # 4. Edge'leri tanımla
    
    # fetch_pr → analyze_code (her zaman)
    workflow.add_edge("fetch_pr", "analyze_code")
    
    # analyze_code → run_tests veya generate_review (conditional)
    workflow.add_conditional_edges(
        "analyze_code",
        lambda state: state["current_step"],
        {
            "run_tests": "run_tests",
            "generate_review": "generate_review",
            "error": END
        }
    )
    
    # run_tests → generate_fix veya generate_review (conditional)
    workflow.add_conditional_edges(
        "run_tests",
        lambda state: state["current_step"],
        {
            "generate_fix": "generate_fix",
            "generate_review": "generate_review",
            "error": END
        }
    )
    
    # generate_fix → generate_review (her zaman)
    workflow.add_edge("generate_fix", "generate_review")
    
    # generate_review → post_results (her zaman)
    workflow.add_edge("generate_review", "post_results")
    
    # post_results → END (her zaman)
    workflow.add_edge("post_results", END)
    
    logger.info("Graph edges defined")
    
    # 5. Graph'ı compile et
    compiled_graph = workflow.compile()
    
    logger.info("Graph compiled successfully")
    
    return compiled_graph


# ============================================================================
# Singleton Instance
# ============================================================================

review_graph = create_review_graph()


# ============================================================================
# Graph Execution Helper
# ============================================================================

async def run_review_pipeline(pr_number: int) -> AgentState:
    """
    Code review pipeline'ını çalıştırır.
    
    Bu fonksiyon, PR numarasını alır, initial state'i oluşturur
    ve LangGraph pipeline'ını çalıştırır.
    
    Args:
        pr_number: GitHub PR numarası
        
    Returns:
        Final AgentState
    """
    from src.agent.state import create_initial_state, PRInfo
    
    logger.info(
        "Starting review pipeline",
        pr_number=pr_number
    )
    
    # 1. Initial state oluştur
    initial_state = create_initial_state(
        pr_info=PRInfo(
            number=pr_number,
            title="",
            body="",
            author="",
            base_branch="",
            head_branch=""
        ),
        max_iterations=3
    )
    
    # 2. Graph'ı çalıştır
    try:
        final_state = await review_graph.ainvoke(initial_state)
        
        logger.info(
            "Review pipeline completed",
            pr_number=pr_number,
            final_step=final_state.get("current_step"),
            error=final_state.get("error")
        )
        
        return final_state
        
    except Exception as e:
        logger.error(
            "Review pipeline failed",
            pr_number=pr_number,
            error=str(e),
            exc_info=True
        )
        raise
