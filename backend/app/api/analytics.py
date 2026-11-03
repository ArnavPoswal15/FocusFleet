from fastapi import APIRouter, Depends

from app.core.deps import require_driver
from app.services.session_logs import aggregate_driver_logs

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
def analytics_summary(user: dict = Depends(require_driver)) -> dict:
    return aggregate_driver_logs(user["name"])
