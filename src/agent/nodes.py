"""
LangGraph agent node'ları.
Her node, AgentState'i okur ve günceller.
"""

from typing import Any
from src.agent.state import AgentState, PRInfo, CodeFinding, TestResult, FixProposal
from src.tools.github_tools import github_client
from src.tools.code_analysis import code_analyzer
from src.tools.sandbox_executor import sandbox_executor
from src.llm.client import llm_client
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
# Node 3: Run Tests
# ============================================================================

async def run_tests_node(state: AgentState) -> dict[str, Any]:
    """
    Testleri izole Docker sandbox'ında çalıştırır.
    
    Bu node, PR'daki test dosyalarını veya kritik bulgular için
    otomatik üretilen testleri sandbox'ta çalıştırır.
    
    Input State:
        - pr_info: PRInfo (files_changed ve diff_content dolu)
        - findings: list[CodeFinding]
        
    Output State Updates:
        - test_results: list[TestResult]
        - current_step: "generate_fix" veya "generate_review"
        - error: None (veya hata mesajı)
        
    Returns:
        State güncellemeleri dictionary'si
    """
    logger.info("Starting run_tests_node")
    
    pr_info = state.get("pr_info")
    findings = state.get("findings", [])
    
    if not pr_info:
        error_msg = "PR info not found. Cannot run tests."
        logger.error(error_msg)
        return {
            "current_step": "error",
            "error": error_msg
        }
    
    try:
        test_results: list[TestResult] = []
        
        # 1. PR'da test dosyaları var mı kontrol et
        test_files = [
            f for f in pr_info.files_changed 
            if f.startswith('test_') or f.endswith('_test.py') or 'tests/' in f
        ]
        
        # 2. Test dosyaları varsa, onları çalıştır
        if test_files:
            logger.info(
                "Running existing test files",
                test_files_count=len(test_files),
                test_files=test_files
            )
            
            for test_file in test_files:
                # Diff'ten test kodunu çıkar
                test_code = _extract_file_code_from_diff(
                    pr_info.diff_content,
                    test_file
                )
                
                if test_code:
                    logger.info(
                        "Running test file",
                        test_file=test_file,
                        code_length=len(test_code)
                    )
                    
                    result = sandbox_executor.run_pytest(
                        test_code=test_code,
                        file_name=test_file
                    )
                    test_results.append(result)
        
        # 3. Test dosyası yoksa ve kritik bulgular varsa, otomatik test üret
        elif any(f.severity in ["critical", "high"] for f in findings):
            logger.info(
                "No test files found, generating tests for critical findings",
                critical_findings_count=len([
                    f for f in findings 
                    if f.severity in ["critical", "high"]
                ])
            )
            
            # Kritik bulgular için test üret
            generated_tests = _generate_tests_for_findings(findings)
            
            if generated_tests:
                logger.info(
                    "Running auto-generated tests",
                    code_length=len(generated_tests)
                )
                
                result = sandbox_executor.run_pytest(
                    test_code=generated_tests,
                    file_name="auto_generated_tests.py"
                )
                test_results.append(result)
        
        # 4. Test sonuçlarını değerlendir
        failed_tests = [t for t in test_results if not t.passed]
        
        logger.info(
            "Test execution completed",
            total_tests=len(test_results),
            failed_tests=len(failed_tests),
            passed_tests=len(test_results) - len(failed_tests)
        )
        
        # 5. Bir sonraki adıma karar ver
        # Başarısız testler varsa düzeltme öner, yoksa review oluştur
        next_step = "generate_fix" if failed_tests else "generate_review"
        
        return {
            "test_results": test_results,
            "current_step": next_step,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Test execution failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "current_step": "error",
            "error": error_msg
        }


# ============================================================================
# Node 4: Generate Review
# ============================================================================

async def generate_review_node(state: AgentState) -> dict[str, Any]:
    """
    Analiz sonuçlarını LLM ile yorumlar ve review comment oluşturur.
    
    Bu node, tüm analiz bulgularını, test sonuçlarını ve PR bilgilerini
    LLM'e göndererek kapsamlı, yapılandırılmış bir review comment üretir.
    
    Input State:
        - pr_info: PRInfo
        - findings: list[CodeFinding]
        - test_results: list[TestResult]
        
    Output State Updates:
        - messages: list[dict] (LLM conversation history)
        - current_step: "post_results"
        - error: None (veya hata mesajı)
        
    Returns:
        State güncellemeleri dictionary'si
    """
    logger.info("Starting generate_review_node")
    
    pr_info = state.get("pr_info")
    findings = state.get("findings", [])
    test_results = state.get("test_results", [])
    
    if not pr_info:
        error_msg = "PR info not found. Cannot generate review."
        logger.error(error_msg)
        return {
            "current_step": "error",
            "error": error_msg
        }
    
    try:
        # 1. Prompt oluştur
        prompt = _build_review_prompt(pr_info, findings, test_results)
        
        logger.info(
            "Sending review request to LLM",
            prompt_length=len(prompt),
            findings_count=len(findings),
            test_results_count=len(test_results)
        )
        
        # 2. LLM'e gönder (reasoning-enabled)
        result = llm_client.chat(
            user_message=prompt,
            preserve_reasoning=True
        )
        
        review_content = result["content"]
        
        logger.info(
            "Review generated successfully",
            review_length=len(review_content)
        )
        
        # 3. State'i güncelle
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": review_content}
            ],
            "current_step": "post_results",
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Review generation failed: {str(e)}"
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


def _generate_tests_for_findings(findings: list[CodeFinding]) -> str | None:
    """
    Kritik bulgular için otomatik test kodu üretir.
    
    Bu fonksiyon, kritik ve yüksek severity bulgular için
    basit pytest testleri üretir.
    
    Args:
        findings: Bulgu listesi
        
    Returns:
        Pytest test kodu (veya None)
    """
    critical_findings = [
        f for f in findings 
        if f.severity in ["critical", "high"]
    ]
    
    if not critical_findings:
        return None
    
    # Test şablonu
    test_code = '''"""
Auto-generated tests for critical security and quality findings.
These tests verify that critical issues are addressed.
"""

import pytest


'''
    
    # Her kritik bulgu için bir test ekle
    for i, finding in enumerate(critical_findings, 1):
        test_code += f'''def test_finding_{i}_{finding.category}():
    """
    Test for {finding.category} finding:
    {finding.description}
    
    File: {finding.file_path}:{finding.line_number}
    Severity: {finding.severity}
    Suggestion: {finding.suggestion}
    """
    # TODO: Implement actual test based on the finding
    # This is a placeholder that should be replaced with
    # actual test logic that verifies the fix
    
    # Örnek: SQL injection için parameterized query kontrolü
    # Örnek: Hardcoded secret için environment variable kontrolü
    
    # Şimdilik placeholder - gerçek implementasyonda
    # düzeltilmiş kodu test etmeli
    assert True, "Placeholder test for {finding.category} finding"


'''
    
    return test_code


def _build_review_prompt(
    pr_info: PRInfo,
    findings: list[CodeFinding],
    test_results: list[TestResult]
) -> str:
    """
    LLM için kapsamlı review prompt'u oluşturur.
    
    Args:
        pr_info: PR bilgileri
        findings: Analiz bulguları
        test_results: Test sonuçları
        
    Returns:
        Prompt metni
    """
    prompt = f"""You are a Senior Code Reviewer with expertise in security, performance, and code quality. Analyze the following code review findings and generate a comprehensive, actionable review comment.

## Pull Request Information
- **PR #{pr_info.number}**: {pr_info.title}
- **Author**: {pr_info.author}
- **Files Changed**: {len(pr_info.files_changed)}
- **Description**: {pr_info.body or "No description provided"}

## Code Analysis Findings
"""
    
    if not findings:
        prompt += "\n✅ **No issues found!** The code looks clean and follows best practices.\n"
    else:
        # Severity'ye göre grupla
        critical = [f for f in findings if f.severity == "critical"]
        high = [f for f in findings if f.severity == "high"]
        medium = [f for f in findings if f.severity == "medium"]
        low = [f for f in findings if f.severity in ["low", "info"]]
        
        if critical:
            prompt += "\n### 🚨 Critical Issues (Must Fix)\n"
            for i, f in enumerate(critical, 1):
                prompt += f"""
**{i}. {f.category.upper()}** - {f.file_path}:{f.line_number}
- Issue: {f.description}
- Suggestion: {f.suggestion}
"""
        
        if high:
            prompt += "\n### ⚠️ High Priority Issues\n"
            for i, f in enumerate(high, 1):
                prompt += f"""
**{i}. {f.category.upper()}** - {f.file_path}:{f.line_number}
- Issue: {f.description}
- Suggestion: {f.suggestion}
"""
        
        if medium:
            prompt += f"\n### 📝 Medium Priority Issues ({len(medium)} found)\n"
            for i, f in enumerate(medium, 1):
                prompt += f"- {f.file_path}:{f.line_number}: {f.description}\n"
        
        if low:
            prompt += f"\n### ℹ️ Low Priority / Info ({len(low)} found)\n"
            for i, f in enumerate(low, 1):
                prompt += f"- {f.file_path}:{f.line_number}: {f.description}\n"
    
    prompt += "\n## Test Results\n"
    
    if not test_results:
        prompt += "\nℹ️ **No tests were run** for this PR.\n"
    else:
        passed = len([t for t in test_results if t.passed])
        failed = len([t for t in test_results if not t.passed])
        
        prompt += f"\n- **Total Tests**: {len(test_results)}\n"
        prompt += f"- **Passed**: {passed} ✅\n"
        prompt += f"- **Failed**: {failed} {'❌' if failed > 0 else '✅'}\n"
        
        if failed > 0:
            prompt += "\n### Failed Tests:\n"
            for test in test_results:
                if not test.passed:
                    prompt += f"- **{test.test_name}**: {test.error_message}\n"
    
    prompt += """
## Your Task

Generate a comprehensive code review comment in **Markdown format** suitable for GitHub. The comment should include:

1. **📋 Summary**: Brief overview of the review (2-3 sentences)
2. **🚨 Critical Issues**: Must-fix issues with clear explanations (if any)
3. **⚠️ Recommendations**: Suggestions for improvement
4. **🧪 Test Coverage**: Assessment of test results and coverage gaps
5. **✅ Action Items**: Clear, actionable next steps for the developer

**Guidelines:**
- Be constructive and specific
- Provide code examples where helpful
- Prioritize security and correctness
- Acknowledge good practices when you see them
- Format for GitHub (use headers, bullet points, code blocks)

**Output only the Markdown comment, no additional explanations.**
"""
    
    return prompt

# ============================================================================
# Node 5: Generate Fix
# ============================================================================

async def generate_fix_node(state: AgentState) -> dict[str, Any]:
    """
    Başarısız testler ve kritik bulgular için düzeltme önerileri üretir.
    
    Bu node, LLM'i kullanarak düzeltme kodu üretir ve FixProposal'lar oluşturur.
    
    Input State:
        - pr_info: PRInfo
        - findings: list[CodeFinding]
        - test_results: list[TestResult]
        
    Output State Updates:
        - fix_proposals: list[FixProposal]
        - current_step: "generate_review"
        - error: None (veya hata mesajı)
        
    Returns:
        State güncellemeleri dictionary'si
    """
    logger.info("Starting generate_fix_node")
    
    pr_info = state.get("pr_info")
    findings = state.get("findings", [])
    test_results = state.get("test_results", [])
    
    if not pr_info:
        error_msg = "PR info not found. Cannot generate fixes."
        logger.error(error_msg)
        return {
            "current_step": "error",
            "error": error_msg
        }
    
    try:
        # 1. Düzeltme gerektiren sorunları belirle
        critical_findings = [
            f for f in findings 
            if f.severity in ["critical", "high"]
        ]
        failed_tests = [
            t for t in test_results 
            if not t.passed
        ]
        
        if not critical_findings and not failed_tests:
            logger.info("No critical issues or failed tests, skipping fix generation")
            return {
                "fix_proposals": [],
                "current_step": "generate_review",
                "error": None
            }
        
        logger.info(
            "Generating fixes",
            critical_findings_count=len(critical_findings),
            failed_tests_count=len(failed_tests)
        )
        
        # 2. Prompt oluştur
        prompt = _build_fix_prompt(pr_info, critical_findings, failed_tests)
        
        # 3. LLM'e gönder
        result = llm_client.chat(
            user_message=prompt,
            preserve_reasoning=True
        )
        
        fix_response = result["content"]
        
        # 4. LLM cevabını parse et ve FixProposal'lara dönüştür
        fix_proposals = _parse_fix_response(fix_response, critical_findings)
        
        logger.info(
            "Fix proposals generated",
            proposals_count=len(fix_proposals)
        )
        
        # 5. State'i güncelle
        return {
            "fix_proposals": fix_proposals,
            "current_step": "generate_review",
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Fix generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        return {
            "current_step": "error",
            "error": error_msg
        }


def _build_fix_prompt(
    pr_info: PRInfo,
    critical_findings: list[CodeFinding],
    failed_tests: list[TestResult]
) -> str:
    """
    LLM için düzeltme prompt'u oluşturur.
    
    Args:
        pr_info: PR bilgileri
        critical_findings: Kritik bulgular
        failed_tests: Başarısız testler
        
    Returns:
        Prompt metni
    """
    prompt = f"""You are a Senior Software Engineer. Generate code fixes for the following critical issues and failed tests.

## Context
- PR #{pr_info.number}: {pr_info.title}
- Files Changed: {', '.join(pr_info.files_changed)}

## Critical Issues to Fix
"""
    
    for i, finding in enumerate(critical_findings, 1):
        prompt += f"""
### Issue {i}: {finding.category.upper()}
- File: {finding.file_path}:{finding.line_number}
- Problem: {finding.description}
- Suggestion: {finding.suggestion}
"""
    
    if failed_tests:
        prompt += "\n## Failed Tests\n"
        for test in failed_tests:
            prompt += f"""
- Test: {test.test_name}
- Error: {test.error_message}
- Output: {test.output[:500] if test.output else "No output"}
"""
    
    prompt += """
## Your Task

For each critical issue, provide a fix in the following format:

FILE: <file_path>

ORIGINAL:
<original_code>

FIXED:
<fixed_code>

EXPLANATION:
<explanation_of_the_fix>

CONFIDENCE: <0.0-1.0>


**Guidelines:**
- Provide complete, working code fixes
- Follow Python best practices
- Ensure fixes address the root cause
- Include error handling where appropriate
- Be specific and actionable

**Output only the fixes in the specified format, no additional explanations.**
"""
    
    return prompt


def _parse_fix_response(
    fix_response: str,
    critical_findings: list[CodeFinding]
) -> list[FixProposal]:
    """
    LLM cevabını parse eder ve FixProposal'lara dönüştürür.
    
    Args:
        fix_response: LLM'in ürettiği düzeltme metni
        critical_findings: Kritik bulgular (referans için)
        
    Returns:
        FixProposal listesi
    """
    from src.agent.state import FixProposal
    
    proposals = []
    
    # Basit parse: "FILE:" bloklarını bul
    blocks = fix_response.split("FILE:")[1:]  # İlk boş bloğu atla
    
    for block in blocks:
        try:
            # File path'i çıkar
            lines = block.strip().split('\n')
            file_path = lines[0].strip()
            
            # Bölümleri bul
            original_start = block.find("ORIGINAL:")
            fixed_start = block.find("FIXED:")
            explanation_start = block.find("EXPLANATION:")
            confidence_start = block.find("CONFIDENCE:")
            
            # Kod bloklarını çıkar
            original_code = ""
            if original_start != -1 and fixed_start != -1:
                original_code = block[original_start + 9:fixed_start].strip()
            
            fixed_code = ""
            if fixed_start != -1 and explanation_start != -1:
                fixed_code = block[fixed_start + 6:explanation_start].strip()
            
            explanation = ""
            if explanation_start != -1 and confidence_start != -1:
                explanation = block[explanation_start + 12:confidence_start].strip()
            
            confidence = 0.5
            if confidence_start != -1:
                confidence_str = block[confidence_start + 11:].strip()
                try:
                    confidence = float(confidence_str)
                except ValueError:
                    confidence = 0.5
            
            proposals.append(FixProposal(
                file_path=file_path,
                original_code=original_code,
                fixed_code=fixed_code,
                explanation=explanation,
                confidence=confidence
            ))
            
        except Exception as e:
            logger.warning(f"Failed to parse fix block: {e}")
            continue
    
    return proposals


