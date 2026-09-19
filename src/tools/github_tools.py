import httpx
from typing import Any
from langchain_core.tools import tool
from config.settings import settings
from src.agent.state import PRInfo, CodeFinding, FixProposal
import structlog

logger = structlog.get_logger()


# ============================================================================
# GitHub API Client
# ============================================================================

class GitHubClient:
    """
    GitHub REST API client.
    PR okuma, yorum ekleme, branch/commit işlemleri için.
    """
    
    def __init__(self):
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {settings.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.owner = settings.github_repo_owner
        self.repo = settings.github_repo_name
        
    async def get_pr_info(self, pr_number: int) -> PRInfo:
        """
        Pull Request bilgilerini çeker.
        
        Args:
            pr_number: PR numarası
            
        Returns:
            PRInfo dataclass instance
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                # Diff içeriğini çek
                diff_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
                diff_headers = {**self.headers, "Accept": "application/vnd.github.v3.diff"}
                diff_response = await client.get(diff_url, headers=diff_headers, timeout=30.0)
                diff_content = diff_response.text if diff_response.status_code == 200 else ""
                
                # Değişen dosyaları çek
                files_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/files"
                files_response = await client.get(files_url, headers=self.headers, timeout=30.0)
                files_data = files_response.json() if files_response.status_code == 200 else []
                files_changed = [f["filename"] for f in files_data]
                
                pr_info = PRInfo(
                    number=pr_number,
                    title=data["title"],
                    body=data.get("body", ""),
                    author=data["user"]["login"],
                    base_branch=data["base"]["ref"],
                    head_branch=data["head"]["ref"],
                    files_changed=files_changed,
                    diff_content=diff_content
                )
                
                logger.info(
                    "PR info fetched",
                    pr_number=pr_number,
                    files_count=len(files_changed),
                    diff_length=len(diff_content)
                )
                
                return pr_info
                
            except httpx.HTTPError as e:
                logger.error("Failed to fetch PR info", error=str(e), pr_number=pr_number)
                raise
    
    async def add_review_comment(
        self,
        pr_number: int,
        body: str,
        file_path: str | None = None,
        line: int | None = None,
        commit_id: str | None = None
    ) -> dict[str, Any]:
        """
        PR'a review comment ekler.
        
        Args:
            pr_number: PR numarası
            body: Yorum içeriği (Markdown destekli)
            file_path: Dosya yolu (inline comment için)
            line: Satır numarası (inline comment için)
            commit_id: Commit SHA (inline comment için)
            
        Returns:
            API response
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
        
        payload = {
            "body": body,
        }
        
        # Inline comment ise
        if file_path and line and commit_id:
            payload.update({
                "path": file_path,
                "line": line,
                "commit_id": commit_id,
                "side": "RIGHT"
            })
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info(
                    "Review comment added",
                    pr_number=pr_number,
                    file_path=file_path,
                    line=line
                )
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error("Failed to add review comment", error=str(e))
                raise
    
    async def add_pr_comment(self, pr_number: int, body: str) -> dict[str, Any]:
        """
        PR'a genel yorum ekler (inline değil).
        
        Args:
            pr_number: PR numarası
            body: Yorum içeriği
            
        Returns:
            API response
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/issues/{pr_number}/comments"
        
        payload = {"body": body}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("PR comment added", pr_number=pr_number)
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error("Failed to add PR comment", error=str(e))
                raise
    
    async def create_branch(self, branch_name: str, base_sha: str) -> dict[str, Any]:
        """
        Yeni branch oluşturur.
        
        Args:
            branch_name: Branch adı
            base_sha: Base commit SHA
            
        Returns:
            API response
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/refs"
        
        payload = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info("Branch created", branch_name=branch_name)
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error("Failed to create branch", error=str(e))
                raise
    
    async def create_commit(
        self,
        branch_name: str,
        message: str,
        files: dict[str, str]
    ) -> dict[str, Any]:
        """
        Commit oluşturur ve branch'e push eder.
        
        Args:
            branch_name: Branch adı
            message: Commit mesajı
            files: {file_path: content} dictionary
            
        Returns:
            API response
        """
        # 1. Branch'in son commit'ini al
        ref_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/refs/heads/{branch_name}"
        
        async with httpx.AsyncClient() as client:
            try:
                ref_response = await client.get(ref_url, headers=self.headers, timeout=30.0)
                ref_response.raise_for_status()
                current_sha = ref_response.json()["object"]["sha"]
                
                # 2. Commit object oluştur
                commit_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/commits"
                
                # Tree oluştur (dosyaları ekle)
                tree_items = []
                for file_path, content in files.items():
                    # Blob oluştur
                    blob_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/blobs"
                    blob_payload = {
                        "content": content,
                        "encoding": "utf-8"
                    }
                    blob_response = await client.post(
                        blob_url,
                        headers=self.headers,
                        json=blob_payload,
                        timeout=30.0
                    )
                    blob_response.raise_for_status()
                    blob_sha = blob_response.json()["sha"]
                    
                    tree_items.append({
                        "path": file_path,
                        "mode": "100644",
                        "type": "blob",
                        "sha": blob_sha
                    })
                
                # Tree oluştur
                tree_url = f"{self.base_url}/repos/{self.owner}/{self.repo}/git/trees"
                tree_payload = {
                    "base_tree": current_sha,
                    "tree": tree_items
                }
                tree_response = await client.post(
                    tree_url,
                    headers=self.headers,
                    json=tree_payload,
                    timeout=30.0
                )
                tree_response.raise_for_status()
                tree_sha = tree_response.json()["sha"]
                
                # Commit oluştur
                commit_payload = {
                    "message": message,
                    "tree": tree_sha,
                    "parents": [current_sha]
                }
                commit_response = await client.post(
                    commit_url,
                    headers=self.headers,
                    json=commit_payload,
                    timeout=30.0
                )
                commit_response.raise_for_status()
                new_commit_sha = commit_response.json()["sha"]
                
                # 3. Branch'i güncelle
                update_payload = {"sha": new_commit_sha}
                update_response = await client.patch(
                    ref_url,
                    headers=self.headers,
                    json=update_payload,
                    timeout=30.0
                )
                update_response.raise_for_status()
                
                logger.info(
                    "Commit created",
                    branch_name=branch_name,
                    commit_sha=new_commit_sha,
                    files_count=len(files)
                )
                
                return commit_response.json()
                
            except httpx.HTTPError as e:
                logger.error("Failed to create commit", error=str(e))
                raise
    
    async def create_pr(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str
    ) -> dict[str, Any]:
        """
        Yeni Pull Request oluşturur.
        
        Args:
            title: PR başlığı
            body: PR açıklaması
            head_branch: Kaynak branch
            base_branch: Hedef branch
            
        Returns:
            API response
        """
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls"
        
        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info(
                    "PR created",
                    title=title,
                    head_branch=head_branch,
                    base_branch=base_branch
                )
                
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error("Failed to create PR", error=str(e))
                raise


# ============================================================================
# LangChain Tool Wrapper'ları
# ============================================================================

github_client = GitHubClient()


@tool
async def fetch_pr_info(pr_number: int) -> dict[str, Any]:
    """
    Fetches Pull Request information including diff and changed files.
    
    Args:
        pr_number: The PR number to fetch
        
    Returns:
        Dictionary with PR metadata, diff content, and changed files
    """
    try:
        pr_info = await github_client.get_pr_info(pr_number)
        return {
            "success": True,
            "pr_info": {
                "number": pr_info.number,
                "title": pr_info.title,
                "body": pr_info.body,
                "author": pr_info.author,
                "base_branch": pr_info.base_branch,
                "head_branch": pr_info.head_branch,
                "files_changed": pr_info.files_changed,
                "diff_content": pr_info.diff_content
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
async def add_review_comment(
    pr_number: int,
    body: str,
    file_path: str | None = None,
    line: int | None = None
) -> dict[str, Any]:
    """
    Adds a review comment to a Pull Request.
    Can be inline (specific file/line) or general.
    
    Args:
        pr_number: The PR number
        body: Comment content (Markdown supported)
        file_path: File path for inline comment (optional)
        line: Line number for inline comment (optional)
        
    Returns:
        API response
    """
    try:
        # Inline comment için commit_id gerekli, şimdilik genel yorum olarak ekle
        result = await github_client.add_pr_comment(pr_number, body)
        return {
            "success": True,
            "comment_id": result.get("id")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
async def create_fix_pr(
    original_pr_number: int,
    title: str,
    body: str,
    files_to_fix: dict[str, str]
) -> dict[str, Any]:
    """
    Creates a new PR with automated fixes.
    
    Args:
        original_pr_number: Original PR number (for reference)
        title: New PR title
        body: New PR description
        files_to_fix: Dictionary of {file_path: fixed_content}
        
    Returns:
        New PR URL and metadata
    """
    try:
        # 1. Orijinal PR'ın base commit'ini al
        original_pr = await github_client.get_pr_info(original_pr_number)
        base_sha = original_pr.diff_content  # Simplified - should get actual SHA
        
        # 2. Yeni branch oluştur
        branch_name = f"ai-fix/pr-{original_pr_number}"
        await github_client.create_branch(branch_name, base_sha)
        
        # 3. Commit oluştur
        commit_message = f"🤖 AI: Automated fixes for PR #{original_pr_number}"
        await github_client.create_commit(
            branch_name=branch_name,
            message=commit_message,
            files=files_to_fix
        )
        
        # 4. Yeni PR oluştur
        pr_body = f"""
## 🤖 Automated Fix Proposal

This PR contains automated fixes suggested by the AI Code Reviewer.

**Original PR:** #{original_pr_number}

### Changes:
{body}

---
*Generated by AI Code Reviewer Agent*
"""
        
        new_pr = await github_client.create_pr(
            title=title,
            body=pr_body,
            head_branch=branch_name,
            base_branch=original_pr.base_branch
        )
        
        return {
            "success": True,
            "pr_url": new_pr.get("html_url"),
            "pr_number": new_pr.get("number")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# Helper Fonksiyonlar
# ============================================================================

def format_findings_as_comment(findings: list[CodeFinding]) -> str:
    """
    CodeFinding listesini Markdown formatında yorum olarak formatlar.
    
    Args:
        findings: Bulgu listesi
        
    Returns:
        Markdown formatlı yorum
    """
    if not findings:
        return "✅ No issues found!"
    
    # Severity'ye göre grupla
    critical = [f for f in findings if f.severity == "critical"]
    high = [f for f in findings if f.severity == "high"]
    medium = [f for f in findings if f.severity == "medium"]
    low = [f for f in findings if f.severity in ["low", "info"]]
    
    comment = "## 🔍 AI Code Review Results\n\n"
    
    if critical:
        comment += "### 🚨 Critical Issues\n"
        for f in critical:
            comment += f"- **{f.file_path}:{f.line_number}** - {f.description}\n"
            comment += f"  - Suggestion: {f.suggestion}\n\n"
    
    if high:
        comment += "### ⚠️ High Priority Issues\n"
        for f in high:
            comment += f"- **{f.file_path}:{f.line_number}** - {f.description}\n"
            comment += f"  - Suggestion: {f.suggestion}\n\n"
    
    if medium:
        comment += "### 📝 Medium Priority Issues\n"
        for f in medium:
            comment += f"- **{f.file_path}:{f.line_number}** - {f.description}\n"
    
    if low:
        comment += f"\n### ℹ️ Low Priority / Info\n"
        comment += f"- {len(low)} minor issues found (see details in commit messages)\n"
    
    comment += "\n---\n*Generated by AI Code Reviewer Agent*"
    
    return comment
