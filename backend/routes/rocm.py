"""ROCm backend management endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..utils.progress import get_progress_manager

router = APIRouter()


@router.get("/backend/rocm-status")
async def get_rocm_status():
    """Get ROCm backend download/availability status."""
    from ..services import rocm

    return rocm.get_rocm_status()


@router.post("/backend/download-rocm")
async def download_rocm_backend():
    """Download the ROCm backend binary."""
    from ..services import rocm

    try:
        rocm.schedule_rocm_binary_download()
    except rocm.BackendOperationBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"message": "ROCm backend download started", "progress_key": rocm.PROGRESS_KEY}


@router.delete("/backend/rocm")
async def delete_rocm_backend():
    """Delete the downloaded ROCm backend binary."""
    from ..services import rocm

    if rocm.is_rocm_active():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete ROCm backend while it is active. Switch to CPU first.",
        )

    try:
        deleted = await rocm.delete_rocm_binary()
    except rocm.BackendOperationBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="No ROCm backend found to delete")

    return {"message": "ROCm backend deleted"}


@router.get("/backend/rocm-progress")
async def get_rocm_download_progress():
    """Get ROCm backend download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("rocm-backend"):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
