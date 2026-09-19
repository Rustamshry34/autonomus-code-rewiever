"""
FastAPI webhook listener.
GitHub webhook'larını dinler ve code review pipeline'ını başlatır.
"""

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import hmac
import hashlib
import json
import asyncio
from config.settings import settings
from src.agent.graph import run_review_pipeline
import structlog

logger = structlog.get_logger()

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="AI Code Reviewer",
    description="Autonomous code review agent powered by LangGraph and Docker",
    version="1.0.0"
)


# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Docker healthcheck ve monitoring için kullanılır.
    """
    return {
        "status": "healthy",
        "service": "ai-code-reviewer",
        "version": "1.0.0"
    }


# ============================================================================
# GitHub Webhook Endpoint
# ============================================================================

@app.post("/webhook")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None)
):
    """
    GitHub webhook listener.
    
    PR event'lerini yakalar ve code review pipeline'ını başlatır.
    
    Headers:
        - X-Hub-Signature-256: HMAC-SHA256 signature (doğrulama için)
        - X-GitHub-Event: Event tipi (pull_request, push, vb.)
    
    Body:
        - JSON payload (PR bilgileri)
    """
    # 1. Payload'u al
    payload = await request.body()
    
    # 2. Signature'ı doğrula
    if not _verify_signature(payload, x_hub_signature_256):
        logger.warning("Invalid webhook signature")
        raise HTTPException(
            status_code=403,
            detail="Invalid signature"
        )
    
    # 3. Event tipini kontrol et
    if x_github_event != "pull_request":
        logger.info(
            "Ignoring non-PR event",
            event_type=x_github_event
        )
        return JSONResponse(
            content={"message": "Event ignored"}
        )
    
    # 4. Payload'u parse et
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.error("Invalid JSON payload")
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON"
        )
    
    # 5. PR action'ını kontrol et
    action = data.get("action")
    
    if action not in ["opened", "synchronize", "reopened"]:
        logger.info(
            "Ignoring PR action",
            action=action
        )
        return JSONResponse(
            content={"message": "Action ignored"}
        )
    
    # 6. PR numarasını al
    pr_number = data.get("pull_request", {}).get("number")
    
    if not pr_number:
        logger.error("PR number not found in payload")
        raise HTTPException(
            status_code=400,
            detail="PR number not found"
        )
    
    logger.info(
        "PR event received",
        pr_number=pr_number,
        action=action
    )
    
    # 7. Pipeline'ı async olarak başlat
    # (webhook response'unu hızlı dönmek için background task)
    asyncio.create_task(_run_pipeline_async(pr_number))
    
    return JSONResponse(
        content={
            "message": "Pipeline started",
            "pr_number": pr_number
        },
        status_code=202  # Accepted
    )


# ============================================================================
# Helper Functions
# ============================================================================

def _verify_signature(payload: bytes, signature: str) -> bool:
    """
    GitHub webhook signature'ını doğrular.
    
    GitHub, webhook payload'unu HMAC-SHA256 ile imzalar ve
    X-Hub-Signature-256 header'ında gönderir.
    
    Args:
        payload: Raw request body
        signature: X-Hub-Signature-256 header değeri
        
    Returns:
        True ise signature geçerli
    """
    # Secret yoksa doğrulama yapma (development mode)
    if not settings.github_webhook_secret:
        logger.warning(
            "No webhook secret configured, skipping verification"
        )
        return True
    
    if not signature:
        logger.warning("No signature header found")
        return False
    
    # Signature formatı: sha256=<hash>
    if not signature.startswith("sha256="):
        logger.warning("Invalid signature format")
        return False
    
    expected_signature = signature[7:]  # "sha256=" prefix'ini kaldır
    
    # HMAC-SHA256 hesapla
    mac = hmac.new(
        settings.github_webhook_secret.encode(),
        msg=payload,
        digestmod=hashlib.sha256
    )
    
    actual_signature = mac.hexdigest()
    
    # Güvenli karşılaştırma (timing attack önleme)
    return hmac.compare_digest(expected_signature, actual_signature)


async def _run_pipeline_async(pr_number: int):
    """
    Pipeline'ı background task olarak çalıştırır.
    
    Webhook response'undan bağımsız çalışır, böylece
    GitHub'a hızlı response dönebiliriz.
    
    Args:
        pr_number: GitHub PR numarası
    """
    try:
        logger.info(
            "Starting pipeline in background",
            pr_number=pr_number
        )
        
        final_state = await run_review_pipeline(pr_number)
        
        logger.info(
            "Pipeline completed in background",
            pr_number=pr_number,
            final_step=final_state.get("current_step"),
            error=final_state.get("error")
        )
        
    except Exception as e:
        logger.error(
            "Pipeline failed in background",
            pr_number=pr_number,
            error=str(e),
            exc_info=True
        )


# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "Starting development server",
        host=settings.app_host,
        port=settings.app_port
    )
    
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level="info"
    )
