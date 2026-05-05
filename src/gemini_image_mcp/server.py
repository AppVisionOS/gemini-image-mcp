"""Gemini Image (Nano Banana) MCP Server.

Wraps Google AI Studio's Gemini image generation API directly.
Tools: generate_image, edit_image, compose_images, get_status.
"""
from __future__ import annotations

import base64
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from google import genai
from google.genai import types
import PIL.Image

mcp = FastMCP("gemini-image")


# =========================================================================
# Configuration
# =========================================================================

DEFAULT_MODEL = os.environ.get(
    "GEMINI_IMAGE_MODEL",
    # Nano Banana 2 model id; update when Google ships a new version.
    "gemini-2.5-flash-image-preview",
)
OUTPUT_DIR = Path(
    os.environ.get("GEMINI_IMAGE_OUTPUT_DIR", "/tmp/gemini-images")
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Get one at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


# =========================================================================
# Helpers
# =========================================================================

def _resolve_image_input(path_or_url: str) -> PIL.Image.Image:
    """Accept local path or http(s) URL, return a PIL.Image."""
    if path_or_url.startswith(("http://", "https://")):
        with httpx.Client(timeout=30) as client:
            resp = client.get(path_or_url)
            resp.raise_for_status()
            return PIL.Image.open(io.BytesIO(resp.content))
    p = Path(path_or_url).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    return PIL.Image.open(p)


def _safe_filename(suggested: str) -> str:
    """Sanitize user-suggested filename, fall back to UUID."""
    if not suggested:
        return f"{uuid.uuid4().hex[:12]}.png"
    name = "".join(c for c in suggested if c.isalnum() or c in ("-", "_", "."))
    if not name:
        return f"{uuid.uuid4().hex[:12]}.png"
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name += ".png"
    return name


def _parse_response(response: Any, output_path: Path) -> dict[str, Any]:
    """Pull image bytes + any text out of a Gemini response. Save image to path."""
    if not response.candidates:
        return {"error": "no candidates returned", "raw": str(response)}

    parts = response.candidates[0].content.parts or []
    image_bytes: bytes | None = None
    text_parts: list[str] = []

    for part in parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            image_bytes = part.inline_data.data
            if isinstance(image_bytes, str):
                image_bytes = base64.b64decode(image_bytes)
        elif getattr(part, "text", None):
            text_parts.append(part.text)

    if not image_bytes:
        return {
            "error": "no image in response",
            "text_response": " ".join(text_parts).strip() or None,
        }

    output_path.write_bytes(image_bytes)
    img = PIL.Image.open(output_path)
    return {
        "path": str(output_path),
        "size_bytes": len(image_bytes),
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "model": DEFAULT_MODEL,
        "text_response": " ".join(text_parts).strip() or None,
    }


def _fmt(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)


# =========================================================================
# Tools
# =========================================================================

@mcp.tool()
def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    output_filename: str = "",
) -> str:
    """Generate an image from a text prompt using Gemini (Nano Banana 2).

    Args:
        prompt: Text description of the image. Be cinematically specific —
            subject + action + camera + lighting + style. Aspect ratio is
            included as a hint in the prompt; specify it explicitly here too.
        aspect_ratio: One of "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "4:5", "5:4".
            Defaults to "1:1". Note: Gemini approximates aspect ratios via prompt
            interpretation; precise pixel dimensions are not guaranteed.
        output_filename: Optional filename (e.g. "minime_hero_v1.png").
            Defaults to a UUID. Saved under GEMINI_IMAGE_OUTPUT_DIR.
    """
    try:
        client = _get_client()
        full_prompt = f"{prompt}\n\nAspect ratio: {aspect_ratio}."
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        out = OUTPUT_DIR / _safe_filename(output_filename)
        result = _parse_response(response, out)
        result["prompt"] = prompt
        result["aspect_ratio"] = aspect_ratio
        return _fmt(result)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


@mcp.tool()
def edit_image(
    input_path_or_url: str,
    prompt: str,
    output_filename: str = "",
) -> str:
    """Edit an existing image with a text instruction (image + text → image).

    Use cases: change lighting, swap background, restyle, repaint regions
    described in prompt, character pose adjustment, color grading.

    Args:
        input_path_or_url: Local path or http(s) URL of the source image.
        prompt: What to change. Be specific about what to keep ("preserve
            the subject's face") and what to change ("replace background
            with golden hour beach").
        output_filename: Optional filename. Defaults to a UUID.
    """
    try:
        client = _get_client()
        image = _resolve_image_input(input_path_or_url)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        out = OUTPUT_DIR / _safe_filename(output_filename)
        result = _parse_response(response, out)
        result["source"] = input_path_or_url
        result["prompt"] = prompt
        return _fmt(result)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


@mcp.tool()
def compose_images(
    input_paths_or_urls: list[str],
    prompt: str,
    output_filename: str = "",
) -> str:
    """Combine multiple input images guided by a text prompt.

    Use cases: parent-photos → baby prediction, character + background,
    product + scene, brand asset + creative composition.

    Args:
        input_paths_or_urls: List of 2-9 local paths or URLs.
        prompt: How to compose. Reference inputs naturally ("the woman from
            the first image, the man from the second, predicted child").
        output_filename: Optional filename. Defaults to a UUID.
    """
    try:
        if not (1 <= len(input_paths_or_urls) <= 9):
            return "Error: provide 1-9 input images"
        client = _get_client()
        images = [_resolve_image_input(p) for p in input_paths_or_urls]
        contents = images + [prompt]
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        out = OUTPUT_DIR / _safe_filename(output_filename)
        result = _parse_response(response, out)
        result["sources"] = input_paths_or_urls
        result["prompt"] = prompt
        return _fmt(result)
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


@mcp.tool()
def get_status() -> str:
    """Return server config — model id, output dir, key presence (masked)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return _fmt({
        "model": DEFAULT_MODEL,
        "output_dir": str(OUTPUT_DIR),
        "output_dir_writable": os.access(OUTPUT_DIR, os.W_OK),
        "api_key_present": bool(api_key),
        "api_key_prefix": api_key[:6] + "..." if api_key else None,
        "pricing_note": "Free tier on AI Studio; paid ~$0.039/image. Confirm at ai.google.dev/pricing.",
    })


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
