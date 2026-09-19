import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LLM / Infron API Ayarları
    # ------------------------------------------------------------------
    llm_base_url: str = Field(
        default="https://llm.onerouter.pro/v1",
        description="LLM API base URL (Infron/OneRouter)"
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.environ.get("INFRON_API_KEY", ""),
        description="Infron API Key"
    )
    llm_model: str = Field(
        default="deepseek/deepseek-v4-flash:free",
        description="Kullanılacak LLM model adı"
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM API çağrıları için timeout süresi (saniye)"
    )
    llm_max_retries: int = Field(
        default=0,
        description="LLM API çağrıları için maksimum yeniden deneme sayısı"
    )

    # ------------------------------------------------------------------
    # GitHub Ayarları
    # ------------------------------------------------------------------
    github_token: str = Field(
        default="",
        description="GitHub Personal Access Token (PR okuma/yazma yetkisi ile)"
    )
    github_webhook_secret: str = Field(
        default="",
        description="GitHub webhook payload'larını doğrulamak için secret token"
    )
    github_repo_owner: str = Field(
        default="",
        description="Hedef GitHub repository sahibi (örn: 'rustamshiriyev')"
    )
    github_repo_name: str = Field(
        default="",
        description="Hedef GitHub repository adı"
    )

    # ------------------------------------------------------------------
    # Docker / Sandbox Ayarları
    # ------------------------------------------------------------------
    sandbox_image: str = Field(
        default="code-reviewer-sandbox:latest",
        description="Kod çalıştırmak için kullanılacak Docker imajı"
    )
    sandbox_timeout: int = Field(
        default=30,
        description="Sandbox içindeki kod çalıştırma işlemi için timeout (saniye)"
    )

    # ------------------------------------------------------------------
    # Uygulama Ayarları
    # ------------------------------------------------------------------
    app_host: str = Field(default="0.0.0.0", description="FastAPI sunucu host adresi")
    app_port: int = Field(default=8000, description="FastAPI sunucu portu")
    log_level: str = Field(default="INFO", description="Uygulama log seviyesi (DEBUG, INFO, WARNING, ERROR)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

# Singleton instance oluştur
settings = Settings()
