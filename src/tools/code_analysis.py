import ast
import re
from typing import Any
from langchain_core.tools import tool
from src.agent.state import CodeFinding
from src.llm.client import llm_client
import structlog

logger = structlog.get_logger()


# ============================================================================
# Diff Parser
# ============================================================================

def parse_diff(diff_content: str) -> dict[str, dict[str, Any]]:
    """
    Git diff içeriğini parse eder ve her dosya için değişiklikleri çıkarır.
    
    Args:
        diff_content: Git diff formatında string
        
    Returns:
        {
            "file_path": {
                "added_lines": [(line_number, content), ...],
                "removed_lines": [(line_number, content), ...],
                "context": str  # Tüm diff içeriği
            }
        }
    """
    files = {}
    current_file = None
    current_line_num = 0
    
    lines = diff_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Yeni dosya başlangıcı
        if line.startswith('+++ b/'):
            current_file = line[6:]
            files[current_file] = {
                "added_lines": [],
                "removed_lines": [],
                "context": ""
            }
            i += 1
            continue
        
        # Hunk başlangıcı (@@ -old_start,old_count +new_start,new_count @@)
        if line.startswith('@@') and current_file:
            match = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if match:
                current_line_num = int(match.group(1))
            files[current_file]["context"] += line + '\n'
            i += 1
            continue
        
        # Eklenen satır
        if line.startswith('+') and not line.startswith('+++') and current_file:
            files[current_file]["added_lines"].append((current_line_num, line[1:]))
            files[current_file]["context"] += line + '\n'
            current_line_num += 1
            i += 1
            continue
        
        # Silinen satır
        if line.startswith('-') and not line.startswith('---') and current_file:
            files[current_file]["removed_lines"].append((current_line_num, line[1:]))
            files[current_file]["context"] += line + '\n'
            i += 1
            continue
        
        # Context satırı
        if current_file:
            files[current_file]["context"] += line + '\n'
            if not line.startswith('\\'):
                current_line_num += 1
        
        i += 1
    
    return files


# ============================================================================
# AST Analyzer
# ============================================================================

class SecurityASTVisitor(ast.NodeVisitor):
    """
    AST üzerinden güvenlik açıklarını tespit eden visitor.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.findings: list[CodeFinding] = []
        
    def visit_Call(self, node: ast.Call):
        """Fonksiyon çağrılarını analiz et"""
        
        # eval() kullanımı
        if isinstance(node.func, ast.Name) and node.func.id == 'eval':
            self.findings.append(CodeFinding(
                file_path=self.file_path,
                line_number=node.lineno,
                severity="critical",
                category="security",
                description="Use of eval() is dangerous and can lead to code injection",
                suggestion="Avoid eval(). Use ast.literal_eval() for safe evaluation or refactor the logic.",
                code_snippet=ast.unparse(node)
            ))
        
        # exec() kullanımı
        if isinstance(node.func, ast.Name) and node.func.id == 'exec':
            self.findings.append(CodeFinding(
                file_path=self.file_path,
                line_number=node.lineno,
                severity="critical",
                category="security",
                description="Use of exec() is dangerous and can lead to arbitrary code execution",
                suggestion="Avoid exec(). Refactor to use safer alternatives.",
                code_snippet=ast.unparse(node)
            ))
        
        # pickle.loads() kullanımı
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'loads' and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'pickle':
                    self.findings.append(CodeFinding(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        severity="high",
                        category="security",
                        description="pickle.loads() can execute arbitrary code during deserialization",
                        suggestion="Use json or a safer serialization format. If pickle is necessary, validate the source.",
                        code_snippet=ast.unparse(node)
                    ))
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import):
        """Import ifadelerini analiz et"""
        for alias in node.names:
            if alias.name in ['telnetlib', 'ftplib', 'poplib']:
                self.findings.append(CodeFinding(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="medium",
                    category="security",
                    description=f"Importing insecure protocol library: {alias.name}",
                    suggestion=f"Use secure alternatives (e.g., SSH instead of telnet, SFTP instead of FTP).",
                    code_snippet=ast.unparse(node)
                ))
        self.generic_visit(node)


def analyze_ast(code: str, file_path: str) -> list[CodeFinding]:
    """
    Python kodunu AST ile analiz eder.
    
    Args:
        code: Python kodu
        file_path: Dosya yolu
        
    Returns:
        CodeFinding listesi
    """
    try:
        tree = ast.parse(code)
        visitor = SecurityASTVisitor(file_path)
        visitor.visit(tree)
        return visitor.findings
    except SyntaxError as e:
        logger.warning("AST parse failed", file_path=file_path, error=str(e))
        return []


# ============================================================================
# LangChain Tools
# ============================================================================

@tool
async def analyze_security(code: str, file_path: str) -> dict[str, Any]:
    """
    Analyzes Python code for security vulnerabilities using AST and pattern matching.
    
    Args:
        code: Python source code to analyze
        file_path: Path of the file being analyzed
        
    Returns:
        Dictionary with list of security findings
    """
    try:
        findings = analyze_ast(code, file_path)
        
        # Ek pattern-based kontroller
        patterns = [
            (r'password\s*=\s*["\'].*["\']', "hardcoded_password", "high", 
             "Hardcoded password detected", "Use environment variables or a secrets manager."),
            (r'api_key\s*=\s*["\'].*["\']', "hardcoded_api_key", "high",
             "Hardcoded API key detected", "Use environment variables or a secrets manager."),
            (r'SELECT\s+\*.*FROM.*%s', "sql_injection", "critical",
             "Potential SQL injection vulnerability", "Use parameterized queries."),
        ]
        
        for pattern, category, severity, desc, suggestion in patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                findings.append(CodeFinding(
                    file_path=file_path,
                    line_number=line_num,
                    severity=severity,
                    category="security",
                    description=desc,
                    suggestion=suggestion,
                    code_snippet=match.group(0)
                ))
        
        logger.info(
            "Security analysis completed",
            file_path=file_path,
            findings_count=len(findings)
        )
        
        return {
            "success": True,
            "findings": [
                {
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "severity": f.severity,
                    "category": f.category,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "code_snippet": f.code_snippet
                }
                for f in findings
            ]
        }
        
    except Exception as e:
        logger.error("Security analysis failed", error=str(e))
        return {"success": False, "error": str(e)}


@tool
async def analyze_complexity(code: str, file_path: str) -> dict[str, Any]:
    """
    Analyzes code complexity using AST metrics.
    
    Args:
        code: Python source code to analyze
        file_path: Path of the file being analyzed
        
    Returns:
        Dictionary with complexity metrics and findings
    """
    try:
        findings = []
        
        # AST ile fonksiyon karmaşıklığını hesapla
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Basit siklometrik karmaşıklık (if, for, while, try sayacı)
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                        complexity += 1
                
                # Fonksiyon uzunluğu
                func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                
                # Argüman sayısı
                arg_count = len(node.args.args)
                
                # Yüksek karmaşıklık
                if complexity > 10:
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="medium",
                        category="maintainability",
                        description=f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                        suggestion="Consider breaking down the function into smaller, more focused functions.",
                        code_snippet=f"def {node.name}(...)"
                    ))
                
                # Çok uzun fonksiyon
                if func_lines > 50:
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="low",
                        category="maintainability",
                        description=f"Function '{node.name}' is too long ({func_lines} lines)",
                        suggestion="Consider refactoring into smaller functions.",
                        code_snippet=f"def {node.name}(...)"
                    ))
                
                # Çok fazla argüman
                if arg_count > 5:
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="low",
                        category="maintainability",
                        description=f"Function '{node.name}' has too many arguments ({arg_count})",
                        suggestion="Consider using a dataclass or dictionary to group related parameters.",
                        code_snippet=f"def {node.name}(...)"
                    ))
        
        logger.info(
            "Complexity analysis completed",
            file_path=file_path,
            findings_count=len(findings)
        )
        
        return {
            "success": True,
            "findings": [
                {
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                    "severity": f.severity,
                    "category": f.category,
                    "description": f.description,
                    "suggestion": f.suggestion,
                    "code_snippet": f.code_snippet
                }
                for f in findings
            ]
        }
        
    except Exception as e:
        logger.error("Complexity analysis failed", error=str(e))
        return {"success": False, "error": str(e)}


@tool
async def analyze_diff(diff_content: str) -> dict[str, Any]:
    """
    Parses and analyzes a Git diff to identify changed files and lines.
    
    Args:
        diff_content: Git diff content
        
    Returns:
        Dictionary with parsed diff information
    """
    try:
        files = parse_diff(diff_content)
        
        logger.info(
            "Diff parsed",
            files_count=len(files),
            total_additions=sum(len(f["added_lines"]) for f in files.values()),
            total_removals=sum(len(f["removed_lines"]) for f in files.values())
        )
        
        return {
            "success": True,
            "files": {
                file_path: {
                    "added_count": len(data["added_lines"]),
                    "removed_count": len(data["removed_lines"]),
                    "added_lines": data["added_lines"],
                    "removed_lines": data["removed_lines"]
                }
                for file_path, data in files.items()
            }
        }
        
    except Exception as e:
        logger.error("Diff analysis failed", error=str(e))
        return {"success": False, "error": str(e)}


@tool
async def analyze_code_with_llm(code: str, file_path: str, context: str = "") -> dict[str, Any]:
    """
    Uses LLM to perform contextual code analysis for bugs, logic errors, and best practices.
    
    Args:
        code: Python source code to analyze
        file_path: Path of the file being analyzed
        context: Additional context about the code's purpose
        
    Returns:
        Dictionary with LLM-generated findings
    """
    try:
        prompt = f"""You are an expert Python code reviewer. Analyze the following code for:
1. Logic errors and bugs
2. Performance issues
3. Best practice violations
4. Potential edge cases

File: {file_path}
Context: {context or "No additional context"}

Code:
```python
{code}
