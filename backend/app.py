"""FastAPI application factory, middleware, and lifecycle events."""

import asyncio
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import ClassVar

from .utils.rocm_env import should_probe_rocminfo


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors matching uvicorn's style."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Configure logging to match uvicorn's format with colors
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(ColoredFormatter("%(levelname)s:     %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)

logger = logging.getLogger(__name__)

# An empty HSA_OVERRIDE_GFX_VERSION poisons the ROCm HSA runtime. It is
# treated as "force-empty" and no GPU is detected, even natively supported
# ones (e.g. gfx1201 / RX 9070 on ROCm 7.2). docker-compose can't
# conditionally omit an env var, so we clean it up here before torch loads.
if not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)

# AMD GPU environment variables must be set before torch import
# Only set HSA_OVERRIDE_GFX_VERSION for older GPUs that need it.
# RDNA 3+ (gfx1100+) and RDNA 4 (gfx1200+) are natively supported by ROCm
# and the override can cause suboptimal performance or errors.
# rocminfo is ROCm tooling. macOS uses Metal/MLX and must not probe or warn about a Linux AMD
# utility that is intentionally absent; the warning looked like the reason startup had failed.
if should_probe_rocminfo(sys.platform) and not os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Collect all GPUs found in rocminfo output
            gfx_versions = []
            for line in result.stdout.splitlines():
                line_lower = line.lower()
                if "gfx" in line_lower:
                    match = re.search(r"(gfx\d+)", line_lower)
                    if match:
                        gfx_versions.append(match.group(1))

            if gfx_versions:
                # Check if any GPU needs the override (RDNA 2 and older)
                # Use the oldest GPU (lowest gfx number) for the decision
                try:
                    gfx_nums = []
                    for v in gfx_versions:
                        m = re.search(r"\d+", v)
                        if m:
                            gfx_nums.append(int(m.group()))
                    if gfx_nums:
                        oldest_num = min(gfx_nums)
                        oldest_gfx = gfx_versions[gfx_nums.index(oldest_num)]
                        if oldest_num < 1100:
                            os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
                            logger.info(
                                "AMD GPU detected (%s), setting HSA_OVERRIDE_GFX_VERSION=10.3.0 for compatibility. All GPUs: %s",
                                oldest_gfx,
                                ", ".join(gfx_versions),
                            )
                        else:
                            logger.info(
                                "AMD GPU detected (%s), native ROCm support available, skipping HSA_OVERRIDE_GFX_VERSION. All GPUs: %s",
                                oldest_gfx,
                                ", ".join(gfx_versions),
                            )
                except (ValueError, AttributeError) as e:
                    logger.info("Could not parse GPU version from rocminfo output: %s", e)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.info(
            "Could not detect AMD GPU via rocminfo, skipping automatic HSA_OVERRIDE_GFX_VERSION configuration: %s",
            e,
        )
if not os.environ.get("MIOPEN_LOG_LEVEL"):
    os.environ["MIOPEN_LOG_LEVEL"] = "4"

logger.info("Loading Voicebox dependencies; first startup can take up to a minute...")
from urllib.parse import quote

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, config, database
from .api_security import (
    LocalAPISecurityMiddleware,
    configured_browser_origins,
)
from .database import get_db
from .request_limits import RequestBodyLimitMiddleware
from .routes import register_routers
from .services import llm, transcribe, tts
from .services.task_queue import (
    create_background_task,
    init_queue,
    shutdown_background_tasks,
)
from .utils.platform_detect import get_backend_type
from .utils.progress import get_progress_manager


def safe_content_disposition(disposition_type: str, filename: str) -> str:
    """Build a Content-Disposition header safe for non-ASCII filenames.

    Uses RFC 5987 ``filename*`` parameter so browsers can decode UTF-8
    filenames while the ``filename`` fallback stays ASCII-only.
    """
    ascii_name = "".join(c for c in filename if c.isascii() and (c.isalnum() or c in " -_.")).strip() or "download"
    utf8_name = quote(filename, safe="")
    return f"{disposition_type}; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from .mcp_server.context import ClientIdMiddleware
    from .mcp_server.server import build_mcp_server, compose_lifespan

    # Build the MCP app up-front so we can wire its lifespan into FastAPI's —
    # FastMCP's Streamable HTTP transport only works if its session manager
    # runs inside the parent ASGI lifespan.
    mcp = build_mcp_server()
    mcp_app = mcp.http_app(path="/", transport="http")

    @asynccontextmanager
    async def voicebox_lifespan(app: FastAPI):
        try:
            await _run_startup(app)
            yield
        finally:
            # Paired with _run_startup via try/finally: runs whether or
            # not the nested MCP lifespan entered cleanly, so a partial
            # startup still unloads whatever models were loaded.
            await _run_shutdown(app)

    # compose_lifespan enters factories in order (voicebox startup →
    # MCP startup) and exits in LIFO (MCP teardown first → models
    # unload last). That ordering matters on shutdown: FastMCP's
    # __aexit__ cancels in-flight session tasks, and we want that to
    # happen *before* _run_shutdown yanks the TTS / Whisper / LLM
    # models out from under any MCP request that was still generating.
    lifespan = compose_lifespan(voicebox_lifespan, mcp_app.router.lifespan_context)

    application = FastAPI(
        title="voicebox API",
        description="Production-quality Qwen3-TTS voice cloning API",
        version=__version__,
        lifespan=lifespan,
    )

    application.add_middleware(ClientIdMiddleware)
    # This must sit outside Starlette's multipart parser. FastAPI applies user
    # middleware in reverse add order: the security boundary stays outermost,
    # then CORS can decorate size/admission failures for trusted browser origins.
    application.add_middleware(RequestBodyLimitMiddleware)
    _configure_cors(application)
    application.add_middleware(LocalAPISecurityMiddleware)
    register_routers(application)
    application.mount("/mcp", mcp_app)
    logger.info("MCP: mounted at /mcp")
    _mount_frontend(application)

    return application


def _configure_cors(application: FastAPI) -> None:
    """Set up CORS middleware with local-first defaults."""
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured_browser_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _mount_frontend(application: FastAPI) -> None:
    """Serve the built web frontend when present (Docker / web deployment).

    The Dockerfile copies the Vite build output to ``/app/frontend/``.  When
    that directory exists we mount static assets and add a catch-all route so
    the React SPA handles client-side routing.  In dev or API-only mode the
    directory is absent and this function is a no-op.
    """
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if not frontend_dir.is_dir():
        return

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Mount hashed assets (JS, CSS, images) that Vite places under /assets
    assets_dir = frontend_dir / "assets"
    if assets_dir.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="frontend-assets",
        )

    # SPA catch-all: serve files if they exist, otherwise index.html for
    # client-side routes like /voices, /stories, /models, etc.
    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = (frontend_dir / full_path).resolve()
        # Guard against path traversal — only serve files inside frontend_dir
        if full_path and file_path.is_file() and file_path.is_relative_to(frontend_dir):
            return FileResponse(file_path)
        return FileResponse(frontend_dir / "index.html", media_type="text/html")

    logger.info("Frontend: serving SPA from %s", frontend_dir)


def _get_gpu_status() -> str:
    """Return a human-readable string describing GPU availability."""
    backend_type = get_backend_type()
    if torch.cuda.is_available():
        from .backends.base import check_cuda_compatibility

        device_name = torch.cuda.get_device_name(0)
        compatible, _warning = check_cuda_compatibility()
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if is_rocm:
            label = f"ROCm ({device_name})"
        else:
            label = f"CUDA ({device_name})"
        if not compatible:
            label += " [UNSUPPORTED - see logs]"
        return label
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    if backend_type == "mlx":
        return "Metal (Apple Silicon via MLX)"

    # Intel XPU (Arc / Data Center) via IPEX
    try:
        import intel_extension_for_pytorch  # noqa: F401

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                xpu_name = torch.xpu.get_device_name(0)
            except Exception:
                xpu_name = "Intel GPU"
            return f"XPU ({xpu_name})"
    except ImportError:
        pass

    return "None (CPU only)"


def _recover_managed_storage(db) -> None:
    """Reconcile valid publication intents before the queue accepts work."""
    from .services.deletion_journal import recover_interrupted_deletions

    try:
        recovery = recover_interrupted_deletions(db)
    except Exception as exc:
        raise RuntimeError("Managed-storage crash recovery failed; refusing startup") from exc

    if recovery.restored or recovery.discarded or recovery.cleared or recovery.unresolved or recovery.malformed:
        logger.info(
            "Deletion recovery: %d restored, %d discarded, %d cleared, %d unresolved, %d malformed",
            recovery.restored,
            recovery.discarded,
            recovery.cleared,
            recovery.unresolved,
            recovery.malformed,
        )
    if recovery.malformed:
        # Malformed entries carry no trustworthy path or inode identity, so
        # they cannot be acted upon. Keep them for manual inspection without
        # letting an invalid file alone become a startup-denial primitive.
        logger.error(
            "Retained %d malformed deletion journal entr%s for manual inspection",
            recovery.malformed,
            "y" if recovery.malformed == 1 else "ies",
        )
    if recovery.unresolved:
        raise RuntimeError("Managed-storage crash recovery is unresolved; refusing startup")


def _reconcile_stale_generations(db) -> None:
    """Make killed in-flight rows terminal before accepting new work."""
    from sqlalchemy import text as sa_text

    try:
        result = db.execute(
            sa_text(
                "UPDATE generations SET status = 'failed', "
                "error = 'Server was shut down during generation' "
                "WHERE status IN ('generating', 'loading_model')"
            )
        )
        if result.rowcount > 0:
            logger.info("Marked %d stale generation(s) as failed", result.rowcount)

        from .database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile

        profile_count = db.query(DBVoiceProfile).count()
        generation_count = db.query(DBGeneration).count()
        logger.info("Profiles: %d, Generations: %d", profile_count, generation_count)
        db.commit()
    except Exception as exc:
        try:
            db.rollback()
        except Exception as rollback_exc:
            raise RuntimeError("Stale-generation reconciliation rollback failed; refusing startup") from rollback_exc
        raise RuntimeError("Stale-generation reconciliation failed; refusing startup") from exc


def _prune_abandoned_exact_checkpoints(db) -> None:
    """Boundedly reclaim safe checkpoint garbage before workers start."""
    try:
        from .services.exact_chunk_checkpoints import (
            ExactChunkCheckpointStore,
            garbage_collect_exact_chunk_checkpoints,
        )

        store = ExactChunkCheckpointStore()
    except Exception:
        logger.warning("Could not initialize exact checkpoint cleanup", exc_info=True)
        return

    try:
        removed = store.prune_abandoned_temporary_files()
    except Exception:
        # Final checkpoints remain independently validated and quota-bound.
        # Optional cache cleanup must not prevent unrelated engines starting.
        logger.warning("Could not prune abandoned exact checkpoint temp files", exc_info=True)
    else:
        if removed:
            logger.info("Removed %d abandoned exact checkpoint temp file(s)", removed)

    try:
        report = garbage_collect_exact_chunk_checkpoints(db, store=store)
    except Exception:
        # Ownership uncertainty always retains final checkpoints. The hard
        # byte quota still prevents this optional cleanup failure from
        # exhausting the rest of the data volume.
        logger.warning("Could not garbage-collect exact chunk checkpoints", exc_info=True)
        return
    if report.removed:
        logger.info(
            "Reclaimed %d completed or orphaned exact checkpoint request(s)",
            report.removed,
        )
    if report.refused:
        logger.warning(
            "Refused %d unsafe exact checkpoint request %s",
            report.refused,
            "directory" if report.refused == 1 else "directories",
        )


def _prune_abandoned_exact_voice_snapshots(db) -> None:
    """Reclaim only ownership-proven snapshot garbage before routes start."""
    try:
        from .services.profiles import garbage_collect_exact_voice_snapshots

        report = garbage_collect_exact_voice_snapshots(db)
    except Exception:
        # Ownership uncertainty retains every finalized directory. Hard write
        # quotas remain active even if optional startup reclamation cannot run.
        logger.warning("Could not garbage-collect exact voice snapshots", exc_info=True)
        return
    if report.pending_removed:
        logger.info(
            "Removed %d abandoned exact voice snapshot director%s",
            report.pending_removed,
            "y" if report.pending_removed == 1 else "ies",
        )
    if report.finalized_removed:
        logger.info(
            "Reclaimed %d completed or orphaned exact voice snapshot%s",
            report.finalized_removed,
            "" if report.finalized_removed == 1 else "s",
        )
    if report.refused:
        logger.warning(
            "Refused %d unsafe exact voice snapshot entr%s",
            report.refused,
            "y" if report.refused == 1 else "ies",
        )


def _prune_abandoned_story_audio_exports() -> None:
    """Remove response scratch left by a killed Story export process."""
    try:
        from .services.stories import cleanup_abandoned_story_audio_exports

        removed, refused, truncated = cleanup_abandoned_story_audio_exports()
    except Exception:
        # New exports still preflight free space and repeat this cleanup before
        # allocating. Optional cache reclamation must not block all engines.
        logger.warning("Could not clean abandoned Story audio exports", exc_info=True)
        return
    if removed:
        logger.info("Removed %d abandoned Story audio export(s)", removed)
    if refused or truncated:
        logger.warning(
            "Retained unsafe or excess Story export scratch (refused=%d, truncated=%s)",
            refused,
            truncated,
        )


def _prune_abandoned_effects_processing() -> None:
    """Remove preview/decoder scratch left by a killed effects request."""
    try:
        from .services.effects_processing import cleanup_abandoned_effects_processing

        removed, refused, truncated = cleanup_abandoned_effects_processing()
    except Exception:
        # New requests repeat bounded cleanup and capacity admission. Optional
        # cache reclamation must not prevent unrelated engines from starting.
        logger.warning("Could not clean abandoned effects processing scratch", exc_info=True)
        return
    if removed:
        logger.info("Removed %d abandoned effects processing director%s", removed, "y" if removed == 1 else "ies")
    if refused or truncated:
        logger.warning(
            "Retained unsafe or excess effects scratch (refused=%d, truncated=%s)",
            refused,
            truncated,
        )


def _prune_voice_prompt_cache() -> None:
    """Bound retained voice-conditioning prompts before requests start."""
    try:
        from .utils.cache import prune_voice_prompt_cache

        removed = prune_voice_prompt_cache()
    except Exception:
        # New writes remain hard bounded even if optional startup reclamation
        # cannot inspect an unsafe entry or a temporarily unavailable volume.
        logger.warning("Could not prune the bounded voice-prompt cache", exc_info=True)
        return
    if removed:
        logger.info("Reclaimed %d stale voice-prompt cache entr%s", removed, "y" if removed == 1 else "ies")


async def _run_startup(application: FastAPI) -> None:
    """Database init, warnings, model-cache prep. Runs on lifespan entry."""
    import platform
    import sys

    logger.info("Voicebox v%s starting up", __version__)
    logger.info(
        "Python %s on %s %s (%s)",
        sys.version.split()[0],
        platform.system(),
        platform.release(),
        platform.machine(),
    )

    # Permission initialization creates/validates the lexical root without
    # touching SQLite. The lifetime lock then serializes migrations, recovery,
    # deletion staging, and all subsequent writes across backend processes.
    config.initialize_data_permissions()
    from .data_root_lock import acquire_data_root_lock

    data_root_lock = acquire_data_root_lock()
    application.state.data_root_lock = data_root_lock
    try:
        database.init_db()
    except BaseException:
        data_root_lock.release()
        del application.state.data_root_lock
        raise

    from .database.session import _db_path

    logger.info("Database: %s", _db_path)
    logger.info("Data directory: %s", config.get_data_dir())

    db = next(get_db())
    try:
        _recover_managed_storage(db)
        _reconcile_stale_generations(db)
        _prune_abandoned_exact_checkpoints(db)
        _prune_abandoned_exact_voice_snapshots(db)
        _prune_abandoned_story_audio_exports()
        _prune_abandoned_effects_processing()
        _prune_voice_prompt_cache()
    finally:
        db.close()

    # The queue must not accept new work until filesystem/DB deletion recovery
    # and stale-generation reconciliation have completed.
    init_queue()

    backend_type = get_backend_type()
    logger.info("Backend: %s", backend_type.upper())
    logger.info("GPU: %s", _get_gpu_status())

    from .backends.base import check_cuda_compatibility

    _compatible, _cuda_warning = check_cuda_compatibility()
    if not _compatible:
        logger.warning("GPU COMPATIBILITY: %s", _cuda_warning)

    from .services.cuda import check_and_update_cuda_binary, recover_cuda_backend_install
    from .services.rocm import check_and_update_rocm_binary, recover_rocm_backend_install

    await recover_cuda_backend_install()
    await recover_rocm_backend_install()

    create_background_task(check_and_update_cuda_binary())
    create_background_task(check_and_update_rocm_binary())

    try:
        progress_manager = get_progress_manager()
        progress_manager._set_main_loop(asyncio.get_running_loop())
    except Exception as e:
        logger.warning("Could not initialize progress manager event loop: %s", e)

    try:
        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model cache: %s", cache_dir)
    except Exception as e:
        logger.warning("Could not create HuggingFace cache directory: %s", e)

    logger.info("Ready")


async def _run_shutdown(application: FastAPI | None = None) -> None:
    """Unload models on lifespan exit."""
    logger.info("Voicebox server shutting down...")
    # If draining ever fails unexpectedly, retain the process lock. Releasing
    # it while a writer may still be alive is less safe than letting process
    # teardown close the descriptor.
    await shutdown_background_tasks()
    try:
        try:
            tts.unload_tts_model()
        except Exception:
            logger.exception("Failed to unload TTS model")
        try:
            transcribe.unload_whisper_model()
        except Exception:
            logger.exception("Failed to unload Whisper model")
        try:
            llm.unload_llm_model()
        except Exception:
            logger.exception("Failed to unload LLM model")
    finally:
        data_root_lock = getattr(application.state, "data_root_lock", None) if application is not None else None
        if data_root_lock is not None:
            data_root_lock.release()
            del application.state.data_root_lock


app = create_app()
