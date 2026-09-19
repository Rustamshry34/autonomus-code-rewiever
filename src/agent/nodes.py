"""
LangGraph agent node'ları.
Her node, AgentState'i okur ve günceller.
"""

from typing import Any
from src.agent.state import AgentState, PRInfo, CodeFinding
from src.tools.github_tools import github_client
from src.tools.code_analysis import code_analyzer
import structlog

logger = structlog.get_logger()


# ============================================================================
# Node 1: Fetch PR Information
# ============================================================================

async def fetch_pr_node(state: AgentState) -> dict[str, Any]:
    """
    GitHub'dan Pull Request bilgilerini çeker.
    
    Bu node pipeline'ın ilk adımıdır. State'teki pr_info.number'ı kullanarak
    GitHub API'den tam PR bilgilerini (diff, files, metadata) çeker.
    
    Input State:
        - pr_info: PRInfo (number alanı dolu olmalı)
        
    Output State Updates:
        - pr_info: PRInfo (tam bilgilerle güncellenmiş)
        - current_step: "analyze_code"
        - error: None (veya hata mesajı)
        
    Returns:
        State güncellemeleri dictionary'si
    """
    logger.info("Starting fetch_pr_node")
    
    # PR numarasını doğrula
    pr_info = state.get("pr_info")
    
    if not pr_info or not pr_info.number:
        error_msg = "PR number not found in state. Cannot fetch PR info."
        logger.error(error_msg)
        return {
            "current_step": "error",
            "error": error_msg
        }
    
    pr_number = pr_info.number
    
    try:
        # GitHub API'den PR bilgilerini çek
        logger.info(
            "Fetching PR info from GitHub",
            pr_number=pr_number
        )
        
        fetched_pr_info = await github_client.get_pr_info(pr_number)
        
        logger.info(
            "PR info fetched successfully",
            pr_number=pr_number,
            files_count=len(fetched_pr_info.files_changed),
            diff_length=len(fetched_pr_info.diff_content)
        )
        
        # State'i güncelle
        return {
            "pr_info": fetched_pr_info,
            "current_step": "analyze_code",
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Failed to fetch PR #{pr_number}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "current_step": "error",
            "error": error_msg
        }


# ============================================================================
# Node 2: Analyze Code
# ============================================================================

async def analyze_code_node(state: AgentState) -> dict[str, Any]:
    """
    PR'daki kodu analiz eder (güvenlik, kalite, karmaşıklık).
    
    Bu node, PR diff'ini parse eder, sadece değişen Python dosyalarını
    analiz eder ve bulguları severity'ye göre sıralar.
    
    Input State:
        - pr_info: PRInfo (diff_content ve files_changed dolu)
        
    Output State Updates:
        - findings: list[CodeFinding] (analiz bulguları)
        - current_step: "run_tests" veya "generate_review"
        - error: None (veya hata mesajı)
        
    Returns:
        State güncellemeleri dictionary'si
    """
    logger.info("Starting analyze_code_node")
    
    pr_info = state.get("pr_info")
    
    if not pr_info or not pr_info.diff_content:
        error_msg = "PR info or diff content not found. Cannot analyze code."
        logger.error(error_msg)
        return {
            "current_step": "error",
            "error": error_msg
        }
    
    try:
        # Sadece Python dosyalarını analiz et
        python_files = [
            f for f in pr_info.files_changed 
            if f.endswith('.py')
        ]
        
        if not python_files:
            logger.info("No Python files to analyze, skipping to review")
            return {
                "findings": [],
                "current_step": "generate_review",
                "error": None
            }
        
        logger.info(
            "Analyzing Python files",
            files_count=len(python_files),
            files=python_files
        )
        
        # Her Python dosyası için analiz yap
        all_findings: list[CodeFinding] = []
        
        for file_path in python_files:
            # Diff'ten bu dosyanın kodunu çıkar
            file_code = _extract_file_code_from_diff(
                pr_info.diff_content, 
                file_path
            )
            
            if file_code:
                logger.info(
                    "Analyzing file",
                    file_path=file_path,
                    code_length=len(file_code)
                )
                
                file_findings = code_analyzer.analyze_python_code(
                    file_code, 
                    file_path
                )
                all_findings.extend(file_findings)
        
        # Bulguları severity'ye göre sırala
        severity_order = {
            "critical": 0, 
            "high": 1, 
            "medium": 2, 
            "low": 3, 
            "info": 4
        }
        all_findings.sort(
            key=lambda f: severity_order.get(f.severity, 5)
        )
        
        # İstatistikler
        critical_count = len([f for f in all_findings if f.severity == "critical"])
        high_count = len([f for f in all_findings if f.severity == "high"])
        
        logger.info(
            "Code analysis completed",
            total_findings=len(all_findings),
            critical_count=critical_count,
            high_count=high_count
        )
        
        # Bir sonraki adıma karar ver
        # Kritik veya yüksek bulgular varsa test çalıştır, yoksa direkt review
        has_critical = critical_count > 0 or high_count > 0
        next_step = "run_tests" if has_critical else "generate_review"
        
        return {
            "findings": all_findings,
            "current_step": next_step,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Code analysis failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "current_step": "error",
            "error": error_msg
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_file_code_from_diff(diff_content: str, file_path: str) -> str | None:
    """
    Diff içeriğinden belirli bir dosyanın eklenen kodunu çıkarır.
    
    Git diff formatı:
    +++ b/path/to/file.py
    +added line
    +another added line
    
    Args:
        diff_content: Git diff içeriği
        file_path: Dosya yolu
        
    Returns:
        Dosya kodu (veya None)
    """
    lines = diff_content.split('\n')
    
    in_target_file = False
    code_lines = []
    
    for line in lines:
        # Dosya başlığını kontrol et
        if line.startswith('+++ b/'):
            current_file = line[6:]  # '+++ b/' prefix'ini kaldır
            in_target_file = (current_file == file_path)
            continue
        
        # Hedef dosyadaysak, eklenen satırları topla
        if in_target_file:
            if line.startswith('+') and not line.startswith('+++'):
                # '+' karakterini kaldır
                code_lines.append(line[1:])
            elif line.startswith('diff --git') or line.startswith('--- '):
                # Yeni dosya başladı, dur
                break
    
    return '\n'.join(code_lines) if code_lines else None


