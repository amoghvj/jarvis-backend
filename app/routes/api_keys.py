import psycopg
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.database import get_db
from app.dependencies.authentication import get_current_user, require_trusted_origin
from app.schemas import ApiKeyStatusOut, MessageOut, RotatedKeyOut
from app.services import api_key_service

router = APIRouter(prefix="/api-keys", tags=["api-keys"])

NO_STORE = {"Cache-Control": "no-store"}


@router.post(
    "/rotate",
    response_model=RotatedKeyOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Create or rotate the account API key (raw key shown once)",
)
def rotate(user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)):
    result = api_key_service.rotate_api_key(conn, user["id"])
    return JSONResponse(content=RotatedKeyOut(**result).model_dump(mode="json"), headers=NO_STORE)


@router.get("/status", response_model=ApiKeyStatusOut, summary="API key status (never the key)")
def key_status(user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)):
    result = api_key_service.get_status(conn, user["id"])
    return JSONResponse(content=ApiKeyStatusOut(**result).model_dump(mode="json"), headers=NO_STORE)


@router.delete(
    "",
    response_model=MessageOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Revoke the account API key",
)
def revoke(user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)):
    revoked = api_key_service.revoke_api_key(conn, user["id"])
    message = "API key revoked" if revoked else "No active API key"
    return JSONResponse(content={"status": "success", "message": message}, headers=NO_STORE)

