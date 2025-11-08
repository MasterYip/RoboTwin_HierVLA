#!/usr/bin/env python3
import os
import sys
import time
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError

# ===== 配置 =====
REPO_ID = "TianxingChen/RoboTwin2.0"
REPO_TYPE = "dataset"
TARGET_FILES = [
    "background_texture.zip",
    "embodiments.zip",
    "objects.zip",
]
LOCAL_DIR = "."
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
ENDPOINT = "https://hf-mirror.com"

# ===== 日志设置 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RoboTwinDownloader")

def check_existing_files():
    """检查已有文件是否完整（暂按存在性+非空判断；如需严格校验可扩展）"""
    existing = []
    for f in TARGET_FILES:
        p = Path(LOCAL_DIR) / f
        if p.exists() and p.stat().st_size > 0:
            existing.append(f)
    return existing

def download_with_retry():
    api = HfApi(endpoint=ENDPOINT)

    # Step 1: 提前检查 repo 是否可访问（快速 fail-fast）
    try:
        repo_info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        logger.info(f"✅ Repo '{REPO_ID}' found. Last updated: {repo_info.lastModified}")
    except (HFValidationError, RepositoryNotFoundError) as e:
        logger.error(f"❌ Repo not found or invalid: {e}")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"⚠️  Cannot fetch repo info (may still work): {e}")

    # Step 2: 检查本地已有文件
    existing = check_existing_files()
    missing = [f for f in TARGET_FILES if f not in existing]
    if existing:
        logger.info(f"📁 Found {len(existing)} existing files: {existing}")
    if not missing:
        logger.info("🎉 All target files already present. Skipping download.")
        return True

    logger.info(f"📥 Need to download: {missing}")

    # Step 3: 尝试下载（带重试）
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"🔄 Attempt {attempt}/{MAX_RETRIES} starting...")
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                allow_patterns=missing,  # 只下缺失的
                local_dir=LOCAL_DIR,
                endpoint=ENDPOINT,
                max_workers=4,
                # resume_download 已默认启用，无需指定
            )
            # 验证是否真下载成功
            newly_existing = [f for f in missing if (Path(LOCAL_DIR) / f).exists()]
            if len(newly_existing) == len(missing):
                logger.info("✅ Download completed successfully.")
                return True
            else:
                failed = [f for f in missing if f not in newly_existing]
                logger.warning(f"⚠️  Partial success. Missing after download: {failed}")
        except KeyboardInterrupt:
            logger.error("🛑 Download interrupted by user.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Attempt {attempt} failed: {type(e).__name__}: {e}")

        if attempt < MAX_RETRIES:
            logger.info(f"⏳ Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)

    logger.error(f"💥 All {MAX_RETRIES} attempts failed. Please check network or use manual download.")
    return False

if __name__ == "__main__":
    logger.info("🚀 Starting RoboTwin assets download (enhanced version)...")
    logger.info(f"🔗 Using mirror: {ENDPOINT}")
    success = download_with_retry()
    if not success:
        sys.exit(1)