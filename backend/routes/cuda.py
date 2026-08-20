"""CUDA backend management endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..utils.progress import get_progress_manager

router = APIRouter()


@router.get("/backend/cuda-status")
async def get_cuda_status():
    """Get CUDA backend download/availability status."""
    from ..services import cuda

    return cuda.get_cuda_status()


@router.post("/backend/download-cuda")
async def download_cuda_backend():
    """Download the CUDA backend binary."""
    from ..services import cuda

    unsupported_reason = cuda.get_cuda_download_unsupported_reason()
    if unsupported_reason:
        raise HTTPException(status_code=409, detail=unsupported_reason)

    try:
        cuda.schedule_cuda_binary_download()
    except (cuda.BackendOperationBusyError, cuda.BackendAlreadyInstalledError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"message": "CUDA backend download started", "progress_key": "cuda-backend"}


@router.delete("/backend/cuda")
async def delete_cuda_backend():
    """Delete the downloaded CUDA backend binary."""
    from ..services import cuda

    if cuda.is_cuda_active():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete CUDA backend while it is active. Switch to CPU first.",
        )

    try:
        deleted = await cuda.delete_cuda_binary()
    except cuda.BackendOperationBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="No CUDA backend found to delete")

    return {"message": "CUDA backend deleted"}


@router.get("/backend/cuda-progress")
async def get_cuda_download_progress():
    """Get CUDA backend download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("cuda-backend"):
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
