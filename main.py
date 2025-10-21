# app.py
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from PIL import Image
from matcher import SuperPointMatcher, MODEL_NAME
import requests
from pydantic import BaseModel, HttpUrl
import os  # <-- NEW

app = FastAPI(title="SuperPoint Match API", version="1.0")

# Where to fetch the gallery list (a JSON array of image URLs).
# e.g. ["https://example1.png","https://example2.png", ...]
CANDIDATE_LIST_ENDPOINT = os.getenv("CANDIDATE_LIST_ENDPOINT")  # <-- NEW


# Load the model once at startup
@app.on_event("startup")
def _load_model():
    app.state.matcher = SuperPointMatcher()  # uses env SUPERPOINT_LOCAL_DIR if set

@app.get("/health")
def health():
    return {"status": "ok", "model_source": getattr(app.state.matcher, "source", "unknown")}

@app.post("/match")
async def match_images(
    image1: UploadFile = File(..., description="First image file"),
    image2: UploadFile = File(..., description="Second image file"),

    # Optional tuning parameters (all have sane defaults)
    upscale: float = Query(1.5, ge=1.0, le=3.0),
    ratio: float = Query(0.9, ge=0.5, le=0.99),
    mutual_check: bool = Query(False),
    ransac_reproj: float = Query(3.0, ge=0.1, le=10.0),
    ransac_iters: int = Query(5000, ge=100, le=100000),
    ransac_conf: float = Query(0.999, ge=0.5, le=0.9999),
    geom_model: str = Query("homography", pattern="^(homography|affine)$"),

    score_min: float = Query(0.30, ge=0.0, le=1.0),
    min_inliers: int = Query(12, ge=0),
    min_inlier_ratio: float = Query(0.15, ge=0.0, le=1.0),
    max_mean_inlier_desc_dist: float = Query(1.20, ge=0.0, le=2.0),
    min_mean_score_product: float = Query(0.12, ge=0.0, le=1.0),
    adaptive_score_factor: float = Query(0.50, ge=0.0, le=2.0),
    fraction_high_score: float = Query(0.50, ge=0.0, le=1.0),
    high_score_gate: float = Query(0.15, ge=0.0, le=1.0),
):
    def _valid_img(content_type: Optional[str]) -> bool:
        return content_type and any(
            content_type.lower().startswith(ct)
            for ct in ("image/", "application/octet-stream")
        )

    if not _valid_img(image1.content_type) or not _valid_img(image2.content_type):
        raise HTTPException(status_code=415, detail="Both files must be images")

    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1 = Image.open(BytesIO(img1_bytes)).convert("RGB")
        img2 = Image.open(BytesIO(img2_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode images: {e}")

    matcher = app.state.matcher
    result = matcher.evaluate(
        img1, img2,
        upscale=upscale,
        ratio=ratio,
        mutual_check=mutual_check,
        ransac_reproj=ransac_reproj,
        ransac_iters=ransac_iters,
        ransac_conf=ransac_conf,
        geom_model=geom_model,
        score_min=score_min,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_mean_inlier_desc_dist=max_mean_inlier_desc_dist,
        min_mean_score_product=min_mean_score_product,
        adaptive_score_factor=adaptive_score_factor,
        fraction_high_score=fraction_high_score,
        high_score_gate=high_score_gate,
    )
    return JSONResponse(result)


# ✅ UPDATED schema for /match-urls — now accepts ONE test image URL
class SingleUrl(BaseModel):
    url: HttpUrl


def _fetch_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Fetch an image from a URL and return a PIL Image (RGB)."""
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def _load_gallery_list(endpoint: str, timeout: int = 10) -> List[str]:
    """Fetch a JSON array (or {'images': [...]}) of image URLs from an endpoint."""
    resp = requests.get(endpoint, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        urls = data
    elif isinstance(data, dict) and "images" in data and isinstance(data["images"], list):
        urls = data["images"]
    else:
        raise HTTPException(
            status_code=502,
            detail="Gallery endpoint did not return a JSON list of URLs or an 'images' array."
        )
    # basic validation
    urls = [str(u) for u in urls if isinstance(u, (str,))]
    if not urls:
        raise HTTPException(status_code=404, detail="Gallery list is empty.")
    return urls


@app.post("/match-urls")
def match_from_urls(
    payload: SingleUrl,
    candidates_endpoint: Optional[HttpUrl] = Query(
        default=None,
        description="Optional override for the gallery list endpoint that returns a JSON array of image URLs."
    ),
    upscale: float = Query(1.5, ge=1.0, le=3.0),
    ratio: float = Query(0.9, ge=0.5, le=0.99),
    mutual_check: bool = Query(False),
    ransac_reproj: float = Query(3.0, ge=0.1, le=10.0),
    ransac_iters: int = Query(5000, ge=100, le=100000),
    ransac_conf: float = Query(0.999, ge=0.5, le=0.9999),
    geom_model: str = Query("homography", pattern="^(homography|affine)$"),

    score_min: float = Query(0.30, ge=0.0, le=1.0),
    min_inliers: int = Query(12, ge=0),
    min_inlier_ratio: float = Query(0.15, ge=0.0, le=1.0),
    max_mean_inlier_desc_dist: float = Query(1.20, ge=0.0, le=2.0),
    min_mean_score_product: float = Query(0.12, ge=0.0, le=1.0),
    adaptive_score_factor: float = Query(0.50, ge=0.0, le=2.0),
    fraction_high_score: float = Query(0.50, ge=0.0, le=1.0),
    high_score_gate: float = Query(0.15, ge=0.0, le=1.0),
):
    """
    Compare a single test image URL against a gallery of image URLs fetched from an external endpoint.
    Returns immediately on the first positive match; otherwise 'no match' after testing all candidates.
    """
    gallery_endpoint = str(candidates_endpoint or CANDIDATE_LIST_ENDPOINT or "").strip()
    if not gallery_endpoint:
        raise HTTPException(
            status_code=500,
            detail="No gallery endpoint configured. Set CANDIDATE_LIST_ENDPOINT env var or pass 'candidates_endpoint' query param."
        )

    # Fetch test image
    try:
        test_img = _fetch_image_from_url(str(payload.url))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch test image: {e}")

    # Load gallery list
    try:
        gallery_urls = _load_gallery_list(gallery_endpoint)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch gallery list: {e}")

    matcher = app.state.matcher

    tested = 0
    last_error_count = 0

    for candidate_url in gallery_urls:
        tested += 1
        try:
            cand_img = _fetch_image_from_url(candidate_url)
        except Exception:
            # Skip bad/404 images but track issues
            last_error_count += 1
            continue

        result = matcher.evaluate(
            test_img, cand_img,
            upscale=upscale,
            ratio=ratio,
            mutual_check=mutual_check,
            ransac_reproj=ransac_reproj,
            ransac_iters=ransac_iters,
            ransac_conf=ransac_conf,
            geom_model=geom_model,
            score_min=score_min,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            max_mean_inlier_desc_dist=max_mean_inlier_desc_dist,
            min_mean_score_product=min_mean_score_product,
            adaptive_score_factor=adaptive_score_factor,
            fraction_high_score=fraction_high_score,
            high_score_gate=high_score_gate,
        )

        if result.get("match") == "TRUE":
            # Return immediately on first match
            return JSONResponse({
                "match": "TRUE",
                "matched_url": candidate_url,
                "tested_candidates": tested,
                "total_candidates": len(gallery_urls),
                "diagnostics": result.get("diagnostics", {}),
				"metadata": result,
            })

    # If we get here, no match was found
    return JSONResponse({
        "match": "FALSE",
        "tested_candidates": tested,
        "total_candidates": len(gallery_urls),
        "skipped_or_failed_downloads": last_error_count,
        "reason": "Exhausted gallery without a positive match.",
		"metadata": result,
    })


@app.get("/")
def root():
    return {
        "message": "SuperPoint Match API",
        "post_to": "/match",
        "docs": "/docs",
        "model": MODEL_NAME,
        "match_urls_note": "POST /match-urls with {'url': 'https://...'}; compares against gallery from CANDIDATE_LIST_ENDPOINT."
    }
