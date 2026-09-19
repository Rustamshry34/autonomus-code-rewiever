import ast
import re
from typing import Any
from langchain_core.tools import tool
from src.agent.state import CodeFinding
import structlog

logger = structlog.get_logger()


# ============================================================================
# Code Analyzer Engine
# ============================================================================

class CodeAnalyzer:
    """
    Statik kod analizi motoru.
    AST analizi, güvenlik taraması ve karmaşıklık analizi yapar.
    """
    
    def analyze_python_code(self, code: str, file_path: str) -> list[CodeFinding]:
        """
        Python kodunu kapsamlı olarak analiz eder.
        
        Args:
            code: Analiz edilecek kaynak kod
            file_path: Dosya yolu (raporlama için)
            
        Returns:
            Tüm bulguların listesi
        """
        findings = []
        
        # 1. AST analizi
        findings.extend(self._analyze_ast(code, file_path))
        
        # 2. Güvenlik pattern'leri
        findings.extend(self._analyze_security_patterns(code, file_path))
        
        # 3. Karmaşıklık analizi
        findings.extend(self._analyze_complexity(code, file_path))
        
        # 4. Best practices
        findings.extend(self._analyze_best_practices(code, file_path))
        
        logger.info(
            "Code analysis completed",
            file_path=file_path,
            findings_count=len(findings)
        )
        
        return findings
    
    def _analyze_ast(self, code: str, file_path: str) -> list[CodeFinding]:
        """
        AST tabanlı analiz.
        Tehlikeli fonksiyonlar, hardcoded secrets, syntax hataları.
        """
        findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            findings.append(CodeFinding(
                file_path=file_path,
                line_number=e.lineno or 1,
                severity="critical",
                category="bug",
                description=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before proceeding"
            ))
            return findings
        
        for node in ast.walk(tree):
            # Tehlikeli fonksiyonlar: eval, exec, __import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        findings.append(CodeFinding(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="critical",
                            category="security",
                            description=f"Use of dangerous function '{node.func.id}'",
                            suggestion="Avoid using eval/exec/__import__, use safer alternatives"
                        ))
                
                # subprocess with shell=True
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen']:
                        findings.append(CodeFinding(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="high",
                            category="security",
                            description=f"Use of potentially dangerous '{node.func.attr}'",
                            suggestion="Use subprocess.run() with list arguments instead"
                        ))
            
            # Hardcoded secrets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        secret_keywords = ['password', 'secret', 'api_key', 'token', 'credential']
                        if any(keyword in target.id.lower() for keyword in secret_keywords):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                findings.append(CodeFinding(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    severity="high",
                                    category="security",
                                    description="Hardcoded secret detected",
                                    suggestion="Use environment variables or secret management"
                                ))
        
        return findings
    
    def _analyze_security_patterns(self, code: str, file_path: str) -> list[CodeFinding]:
        """
        Regex tabanlı güvenlik pattern analizi.
        SQL injection, command injection, path traversal.
        """
        findings = []
        
        # SQL injection riski
        sql_patterns = [
            (r'execute\s*\(\s*f["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*["\']\s*\)', 
             "SQL injection risk: f-string in SQL query"),
            (r'execute\s*\(\s*["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*["\']\s*%\s*', 
             "SQL injection risk: string formatting in SQL query"),
            (r'execute\s*\(\s*["\'].*(?:SELECT|INSERT|UPDATE|DELETE).*["\']\.format\s*\(', 
             "SQL injection risk: .format() in SQL query"),
        ]
        
        for i, line in enumerate(code.split('\n'), 1):
            for pattern, description in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=i,
                        severity="critical",
                        category="security",
                        description=description,
                        suggestion="Use parameterized queries with placeholders (%s or ?)"
                    ))
                    break
        
        # Command injection riski
        cmd_patterns = [
            (r'os\.system\s*\(', 
             "Command injection risk: os.system() usage"),
            (r'subprocess\.(?:call|run|Popen)\s*\(\s*f["\']', 
             "Command injection risk: f-string in subprocess command"),
            (r'subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True', 
             "Command injection risk: shell=True in subprocess"),
        ]
        
        for i, line in enumerate(code.split('\n'), 1):
            for pattern, description in cmd_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=i,
                        severity="high",
                        category="security",
                        description=description,
                        suggestion="Use subprocess.run() with list arguments, avoid shell=True"
                    ))
                    break
        
        # Path traversal riski
        path_patterns = [
            (r'open\s*\(\s*f["\'].*{.*}.*["\']', 
             "Path traversal risk: user input in file path"),
            (r'os\.path\.join\s*\([^)]*\+', 
             "Path traversal risk: string concatenation in path"),
        ]
        
        for i, line in enumerate(code.split('\n'), 1):
            for pattern, description in path_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(CodeFinding(
                        file_path=file_path,
                        line_number=i,
                        severity="medium",
                        category="security",
                        description=description,
                        suggestion="Validate and sanitize file paths, use os.path.basename()"
                    ))
                    break
        
        return findings
    
    def _analyze_complexity(self, code: str, file_path: str) -> list[CodeFinding]:
        """
        Kod karmaşıklığı analizi.
        Fonksiyon uzunluğu, nested loop derinliği.
        """
        findings = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Fonksiyon uzunluğu
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_lines = node.end_lineno - node.lineno + 1
                    if func_lines > 50:
                        findings.append(CodeFinding(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            category="maintainability",
                            description=f"Function '{node.name}' is too long ({func_lines} lines)",
                            suggestion="Consider breaking down into smaller functions"
                        ))
                    
                    # Nested loop derinliği
                    max_depth = self._get_max_loop_depth(node)
                    if max_depth > 3:
                        findings.append(CodeFinding(
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            category="maintainability",
                            description=f"Function '{node.name}' has deep nesting (depth={max_depth})",
                            suggestion="Consider extracting nested loops into helper functions"
                        ))
        
        except SyntaxError:
            pass
        
        return findings
    
    def _analyze_best_practices(self, code: str, file_path: str) -> list[CodeFinding]:
        """
        Best practices ve stil kontrolleri.
        """
        findings = []
        
        # TODO/FIXME yorumları
        for i, line in enumerate(code.split('\n'), 1):
            if re.search(r'#\s*(TODO|FIXME|HACK|XXX)', line, re.IGNORECASE):
                findings.append(CodeFinding(
                    file_path=file_path,
                    line_number=i,
                    severity="info",
                    category="style",
                    description="Unresolved TODO/FIXME comment",
                    suggestion="Consider resolving or tracking in issue tracker"
                ))
        
        # Print statements (production kodunda olmamalı)
        for i, line in enumerate(code.split('\n'), 1):
            if re.search(r'^\s*print\s*\(', line) and 'logging' not in line:
                findings.append(CodeFinding(
                    file_path=file_path,
                    line_number=i,
                    severity="low",
                    category="style",
                    description="Print statement found (consider using logging)",
                    suggestion="Use logging module instead of print for production code"
                ))
        
        return findings
    
    def _get_max_loop_depth(self, node: ast.AST) -> int:
        """Bir node'daki maksimum loop derinliğini hesaplar"""
        max_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                depth = 1 + self._get_loop_depth(child)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _get_loop_depth(self, loop_node: ast.AST) -> int:
        """Loop içindeki nested loop derinliğini hesaplar"""
        max_depth = 0
        
        for child in ast.walk(loop_node):
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)) and child != loop_node:
                depth = 1 + self._get_loop_depth(child)
                max_depth = max(max_depth, depth)
        
        return max_depth


# ============================================================================
# Singleton Instance
# ============================================================================

code_analyzer = CodeAnalyzer()


# ============================================================================
# LangChain Tool Wrapper'ları
# ============================================================================

@tool
def analyze_code_security(code: str, file_path: str) -> list[dict[str, Any]]:
    """
    Analyzes Python code for security vulnerabilities.
    
    Args:
        code: Source code to analyze
        file_path: File path for reporting
        
    Returns:
        List of security findings
    """
    findings = code_analyzer.analyze_python_code(code, file_path)
    security_findings = [f for f in findings if f.category == "security"]
    
    return [
        {
            "file_path": f.file_path,
            "line_number": f.line_number,
            "severity": f.severity,
            "description": f.description,
            "suggestion": f.suggestion
        }
        for f in security_findings
    ]


@tool
def analyze_code_quality(code: str, file_path: str) -> list[dict[str, Any]]:
    """
    Analyzes Python code for quality and maintainability issues.
    
    Args:
        code: Source code to analyze
        file_path: File path for reporting
        
    Returns:
        List of quality findings
    """
    findings = code_analyzer.analyze_python_code(code, file_path)
    quality_findings = [f for f in findings if f.category in ["maintainability", "style", "bug"]]
    
    return [
        {
            "file_path": f.file_path,
            "line_number": f.line_number,
            "severity": f.severity,
            "description": f.description,
            "suggestion": f.suggestion
        }
        for f in quality_findings
    ]


@tool
def analyze_code_full(code: str, file_path: str) -> list[dict[str, Any]]:
    """
    Performs comprehensive code analysis (security + quality + complexity).
    
    Args:
        code: Source code to analyze
        file_path: File path for reporting
        
    Returns:
        List of all findings
    """
    findings = code_analyzer.analyze_python_code(code, file_path)
    
    return [
        {
            "file_path": f.file_path,
            "line_number": f.line_number,
            "severity": f.severity,
            "category": f.category,
            "description": f.description,
            "suggestion": f.suggestion
        }
        for f in findings
    ]


@tool
def parse_diff_and_analyze(diff_content: str) -> list[dict[str, Any]]:
    """
    Parses a git diff and analyzes changed code sections.
    
    Args:
        diff_content: Git diff content
        
    Returns:
        List of findings for changed code
    """
    findings = []
    
    # Diff'i parse et (basit yaklaşım)
    current_file = None
    code_lines = []
    
    for line in diff_content.split('\n'):
        if line.startswith('+++ b/'):
            # Önceki dosyayı analiz et
            if current_file and code_lines:
                code = '\n'.join(code_lines)
                file_findings = code_analyzer.analyze_python_code(code, current_file)
                findings.extend(file_findings)
            
            # Yeni dosyaya geç
            current_file = line[6:]
            code_lines = []
        elif line.startswith('+') and not line.startswith('+++'):
            # Eklenen satır
            code_lines.append(line[1:])
    
    # Son dosyayı da analiz et
    if current_file and code_lines:
        code = '\n'.join(code_lines)
        file_findings = code_analyzer.analyze_python_code(code, current_file)
        findings.extend(file_findings)
    
    return [
        {
            "file_path": f.file_path,
            "line_number": f.line_number,
            "severity": f.severity,
            "category": f.category,
            "description": f.description,
            "suggestion": f.suggestion
        }
        for f in findings
    ]
