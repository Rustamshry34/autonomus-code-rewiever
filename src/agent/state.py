from typing import Annotated, Any, Literal
from dataclasses import dataclass, field
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ============================================================================
# Veri Modelleri
# ============================================================================

@dataclass
class PRInfo:
    """Pull Request temel bilgileri"""
    number: int
    title: str
    body: str
    author: str
    base_branch: str
    head_branch: str
    files_changed: list[str] = field(default_factory=list)
    diff_content: str = ""


@dataclass
class CodeFinding:
    """Kod analizi bulgusu (güvenlik, performans, stil vb.)"""
    file_path: str
    line_number: int
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: Literal["security", "performance", "style", "bug", "maintainability"]
    description: str
    suggestion: str
    code_snippet: str = ""


@dataclass
class TestResult:
    """Sandbox'ta çalıştırılan test sonucu"""
    test_name: str
    passed: bool
    output: str
    error_message: str = ""
    execution_time: float = 0.0


@dataclass
class FixProposal:
    """Otomatik düzeltme önerisi"""
    file_path: str
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float  # 0.0 - 1.0


# ============================================================================
# LangGraph State Tanımı
# ============================================================================

class AgentState(TypedDict):
    """
    Ana agent state yapısı.
    LangGraph node'ları bu state'i okur ve günceller.
    
    Annotated ile reducer fonksiyonları tanımlanabilir:
    - add_messages: Mesajları birleştirir (append)
    - add_findings: Bulguları birleştirir (append)
    """
    
    # ------------------------------------------------------------------------
    # Pipeline Kontrolü
    # ------------------------------------------------------------------------
    current_step: str  # "fetch_pr", "analyze", "test", "review", "fix", "complete"
    error: str | None  # Hata mesajı (varsa)
    
    # ------------------------------------------------------------------------
    # PR Bilgileri (GitHub'dan gelir)
    # ------------------------------------------------------------------------
    pr_info: PRInfo | None
    
    # ------------------------------------------------------------------------
    # Analiz Sonuçları
    # ------------------------------------------------------------------------
    # Annotated ile reducer: Her node yeni bulgular ekleyebilir
    findings: Annotated[list[CodeFinding], lambda x, y: x + y]
    
    # ------------------------------------------------------------------------
    # Test Sonuçları (Sandbox'tan gelir)
    # ------------------------------------------------------------------------
    test_results: Annotated[list[TestResult], lambda x, y: x + y]
    
    # ------------------------------------------------------------------------
    # Düzeltme Önerileri
    # ------------------------------------------------------------------------
    fix_proposals: Annotated[list[FixProposal], lambda x, y: x + y]
    
    # ------------------------------------------------------------------------
    # LLM Conversation (Reasoning tracking için)
    # ------------------------------------------------------------------------
    # Annotated ile reducer: Mesajları birleştirir
    messages: Annotated[list[dict[str, Any]], add_messages]
    
    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------
    iteration_count: int  # Kaç döngü yapıldı (sonsuz döngüyü önlemek için)
    max_iterations: int  # Maksimum döngü sayısı


# ============================================================================
# State Factory Fonksiyonu
# ============================================================================

def create_initial_state(
    pr_info: PRInfo | None = None,
    max_iterations: int = 3
) -> AgentState:
    """
    Yeni bir agent state'i oluşturur.
    
    Args:
        pr_info: Başlangıç PR bilgileri
        max_iterations: Maksimum analiz düzeltme döngüsü sayısı
        
    Returns:
        Temiz, başlangıç state'i
    """
    return AgentState(
        current_step="fetch_pr",
        error=None,
        pr_info=pr_info,
        findings=[],
        test_results=[],
        fix_proposals=[],
        messages=[],
        iteration_count=0,
        max_iterations=max_iterations
    )


# ============================================================================
# Helper Fonksiyonlar
# ============================================================================

def get_critical_findings(state: AgentState) -> list[CodeFinding]:
    """Critical ve high severity bulguları döndürür"""
    return [
        f for f in state["findings"]
        if f.severity in ["critical", "high"]
    ]


def has_failed_tests(state: AgentState) -> bool:
    """Başarısız test var mı kontrol eder"""
    return any(not t.passed for t in state["test_results"])


def should_continue_iteration(state: AgentState) -> bool:
    """Döngüye devam edilip edilmeyeceğini kontrol eder"""
    return state["iteration_count"] < state["max_iterations"]
