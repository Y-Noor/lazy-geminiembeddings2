"""Trimodal Gemini Embedding 2 pipeline with a pure JAX analytics engine.

Indexes text, video, and audio files dropped in ``data/trimodal``, embeds
each file with ``gemini-embedding-2`` at 768 dimensions (Matryoshka / MRL),
and runs
cosine similarity, cross-modal top-k retrieval, near-duplicate detection,
k-means, and PCA entirely with JIT-compiled JAX.

Core matrix work uses ``jax.numpy`` only. NumPy and scikit-learn are not used.

Usage
-----
Put your key in the ``.env`` file next to this script (``GEMINI_API_KEY=...``),
then run::

    python jax_gemini2_trimodal.py

Check the JAX kernels without calling the API::

    python jax_gemini2_trimodal.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import plotly.express as px
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "trimodal"
DEFAULT_CACHE = ROOT / "cache" / "embeddings.json"
DEFAULT_HTML = ROOT / "artifacts" / "trimodal_clusters.html"
DEFAULT_LOG = ROOT / "artifacts" / "jax_benchmark.log"

EMBED_MODEL = "gemini-embedding-2"
OUTPUT_DIM = 768
USER_AGENT = "geminiembeddings2-trimodal/1.0 (educational; Wikimedia Commons open media)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

MODALITY_CODE = {"Text": 0, "Image": 1, "Audio": 2, "Video": 3, "PDF": 4}
SYMBOL_FOR_MODALITY = {
    "Text": "circle",
    "Image": "square",
    "Audio": "diamond",
    "Video": "triangle-up",
    "PDF": "star",
}
TEXT_EXTENSIONS = {".txt", ".md", ".text"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mpeg", ".mpg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".m4a", ".flac"}
MEDIA_MIME = {
    ".txt": "text/plain",
    ".md": "text/plain",
    ".text": "text/plain",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".pdf": "application/pdf",
}
# gemini-embedding-2 accepts one PDF of at most 6 pages. One page per vector
# is the quality recommendation, so each page is embedded on its own.
MAX_PDF_BYTES = 80 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_MEDIA_BYTES = 20 * 1024 * 1024
# Inline embed requests stay small. Larger videos are cut into clips first.
INLINE_MEDIA_BYTES = MAX_MEDIA_BYTES
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
# gemini-embedding-2 accepts at most 120 seconds and keeps 32 frames.
VIDEO_LIMIT_SECONDS = 120.0
VIDEO_WINDOW_SECONDS = 110.0
VIDEO_WINDOW_STEP = 95.0

# Open-access Wikimedia Commons files. Each row is one synced triplet:
# a transcript, a still image, and a short audio clip for the same concept.
# Full CMU-MOSEI / UR-FUNNY releases are access-restricted and very large;
# this is a small programmatic trimodal subset in the same spirit.
TRIPLETS: tuple[dict[str, str], ...] = (
    {
        "domain": "Sports",
        "concept": "soccer_stadium",
        "text": (
            "Players compete in a soccer match inside a large open stadium "
            "while the crowd applauds."
        ),
        "image_title": "File:Kashima Soccer Stadium 1.jpg",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "c/c3/Kashima_Soccer_Stadium_1.jpg/500px-Kashima_Soccer_Stadium_1.jpg"
        ),
        "audio_title": "File:Applause.ogg",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Applause.ogg",
    },
    {
        "domain": "Sports",
        "concept": "basketball_game",
        "text": (
            "A basketball player goes up for a hook shot as a referee whistle "
            "cuts across the arena."
        ),
        "image_title": "File:Kent Benson attempts a hook shot over Ken Ferdinand.jpg",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/"
            "Kent_Benson_attempts_a_hook_shot_over_Ken_Ferdinand.jpg/"
            "500px-Kent_Benson_attempts_a_hook_shot_over_Ken_Ferdinand.jpg"
        ),
        "audio_title": "File:Whistle.ogg",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/Whistle.ogg",
    },
    {
        "domain": "Tech",
        "concept": "laptop_keyboard",
        "text": "Fingers type quickly on a laptop computer keyboard in an office.",
        "image_title": "File:HP Pavilion Computer laptop keyboard closeup.jpg",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/"
            "HP_Pavilion_Computer_laptop_keyboard_closeup.jpg/"
            "500px-HP_Pavilion_Computer_laptop_keyboard_closeup.jpg"
        ),
        "audio_title": "File:Computer keyboard.ogg",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Computer_keyboard.ogg",
    },
    {
        "domain": "Tech",
        "concept": "dialup_modem",
        "text": (
            "A dial-up modem screeches through its handshake while connecting "
            "a computer to the network."
        ),
        "image_title": "File:Modem-usr-courier-v34-front-online.jpg",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/"
            "Modem-usr-courier-v34-front-online.jpg/"
            "500px-Modem-usr-courier-v34-front-online.jpg"
        ),
        "audio_title": "File:Dial up modem noises.ogg",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/3/33/Dial_up_modem_noises.ogg",
    },
    {
        "domain": "Nature",
        "concept": "redwood_forest",
        "text": "Birds sing along a trail through a quiet redwood forest.",
        "image_title": "File:California redwood trees nature trail path.JPG",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/"
            "California_redwood_trees_nature_trail_path.JPG/"
            "500px-California_redwood_trees_nature_trail_path.JPG"
        ),
        "audio_title": "File:Birdsong mild sunny day.ogg",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/7/75/Birdsong_mild_sunny_day.ogg",
    },
    {
        "domain": "Nature",
        "concept": "ocean_waves",
        "text": "Ocean waves break and wash onto the shore.",
        "image_title": "File:Ocean Waves.jpg",
        "image_url": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/"
            "5/55/Ocean_Waves.jpg/500px-Ocean_Waves.jpg"
        ),
        "audio_title": "File:Sea waves.wav",
        "audio_url": "https://upload.wikimedia.org/wikipedia/commons/8/84/Sea_waves.wav",
    },
)


@dataclass(frozen=True)
class Sample:
    """One embedded item. Image and audio rows point at a local file."""

    domain: str
    concept: str
    modality: str
    text: str
    source_title: str
    source_url: str
    local_path: Path | None
    mime: str | None
    clip_start: float | None = None
    clip_end: float | None = None

    @property
    def modality_id(self) -> int:
        return MODALITY_CODE[self.modality]


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        self.lines.append(text)
        print(text)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs into the environment without overriding existing keys."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _urlopen(request: urllib.request.Request, timeout: int):
    """Open a URL, backing off when Wikimedia answers 429."""
    delay = 2.0
    last_error: BaseException | None = None
    for _attempt in range(6):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
            time.sleep(min(wait, 45.0))
            delay = min(delay * 2, 45.0)
        except urllib.error.URLError as exc:
            last_error = exc
            time.sleep(delay)
            delay = min(delay * 2, 45.0)
    if last_error is None:
        raise RuntimeError(f"Failed to open {request.full_url}")
    raise last_error


def _request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _urlopen(request, 60) as response:
        return json.loads(response.read().decode("utf-8"))


def _mime_for(url: str, kind: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    if kind == "image":
        if path.endswith(".png"):
            return "image/png"
        if path.endswith(".webp"):
            return "image/webp"
        return "image/jpeg"
    if path.endswith(".wav"):
        return "audio/wav"
    if path.endswith(".mp3"):
        return "audio/mpeg"
    return "audio/ogg"


def _payload_is_valid(path: Path, mime: str) -> bool:
    head = path.read_bytes()[:16]
    if mime == "image/jpeg":
        return head.startswith(b"\xff\xd8\xff")
    if mime == "image/png":
        return head.startswith(b"\x89PNG")
    if mime == "image/webp":
        return head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if mime == "audio/wav":
        return head.startswith(b"RIFF")
    if mime == "audio/ogg":
        return head.startswith(b"OggS")
    if mime == "audio/mpeg":
        return head.startswith(b"ID3") or head[:2] == b"\xff\xfb"
    return len(head) > 8


def resolve_commons(title: str, fallback_url: str, kind: str) -> tuple[str, str]:
    """Resolve a Commons file to a download URL. Fall back to a pinned URL."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }
    if kind == "image":
        params["iiurlwidth"] = "500"
    api_url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    try:
        payload = _request_json(api_url)
        page = next(iter(payload["query"]["pages"].values()))
        if "missing" not in page and page.get("imageinfo"):
            info = page["imageinfo"][0]
            if kind == "image" and info.get("thumburl"):
                return info["thumburl"], _mime_for(info["thumburl"], "image")
            if info.get("url"):
                remote_mime = info.get("mime") or ""
                if remote_mime in {"audio/wav", "audio/x-wav"}:
                    mime = "audio/wav"
                elif remote_mime in {"audio/mpeg", "audio/mp3"}:
                    mime = "audio/mpeg"
                elif kind == "audio":
                    mime = _mime_for(info["url"], "audio")
                else:
                    mime = _mime_for(info["url"], "image")
                return info["url"], mime
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, StopIteration):
        pass
    return fallback_url, _mime_for(fallback_url, kind)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _urlopen(request, 180) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    temporary.replace(dest)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _ffmpeg_exe() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _media_duration(path: Path) -> float | None:
    exe = _ffmpeg_exe()
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _video_windows(duration: float) -> list[tuple[float, float]]:
    """Overlapping clips that stay inside the 120 second embedding limit."""
    duration = round(duration, 2)
    if duration <= VIDEO_LIMIT_SECONDS:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 0.5:
        end = min(start + VIDEO_WINDOW_SECONDS, duration)
        if end - start < 1.0:
            break
        windows.append((round(start, 2), round(end, 2)))
        if end >= duration - 0.25:
            break
        start += VIDEO_WINDOW_STEP
    return windows


def _video_needs_proxy(path: Path, size: int, duration: float | None) -> bool:
    native = path.suffix.lower() in {".mp4", ".mov"}
    if not native or size > INLINE_MEDIA_BYTES:
        return True
    return duration is not None and duration > VIDEO_LIMIT_SECONDS


def _transcode_clip(path: Path, start: float, end: float) -> bytes:
    """Shrink one clip to 1 fps at 480p. The model keeps at most 32 frames."""
    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg is required for videos over 20 MB or longer than 2 minutes. "
            "Install it with: pip install imageio-ffmpeg"
        )
    duration = max(end - start, 0.1)
    scales = ("scale=480:-2", "scale=320:-2")
    crfs = ("32", "38")
    last_error = "ffmpeg produced no clip"
    for scale, crf in zip(scales, crfs):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "clip.mp4"
            proc = subprocess.run(
                [
                    exe,
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    str(path),
                    "-t",
                    f"{duration:.3f}",
                    "-an",
                    "-vf",
                    f"fps=1,{scale}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    crf,
                    "-movflags",
                    "+faststart",
                    str(out),
                ],
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
                last_error = proc.stderr.decode("utf-8", errors="replace")[-400:]
                continue
            data = out.read_bytes()
        if len(data) <= INLINE_MEDIA_BYTES:
            return data
        last_error = f"clip was still {len(data)} bytes"
    raise RuntimeError(f"Could not prepare a clip of {path.name}: {last_error}")


def _video_samples(path: Path, size: int) -> tuple[list[Sample], str]:
    """One row per clip. The original file stays the playback path."""
    concept = path.stem
    rel = path.name
    if size > MAX_VIDEO_BYTES:
        return [], f"  skip {rel}: larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB"
    duration = _media_duration(path)
    if not _video_needs_proxy(path, size, duration):
        sample = Sample(
            domain=concept,
            concept=concept,
            modality="Video",
            text="",
            source_title=path.name,
            source_url="",
            local_path=path,
            mime=MEDIA_MIME[path.suffix.lower()],
        )
        return [sample], f"  Video {concept} <- {rel} ({size} bytes)"
    if _ffmpeg_exe() is None:
        return [], (
            f"  skip {rel}: videos over 20 MB or longer than 2 minutes need ffmpeg. "
            "Install it with: pip install imageio-ffmpeg"
        )
    if duration is None:
        return [], f"  skip {rel}: could not read the video duration"
    windows = _video_windows(duration)
    samples = [
        Sample(
            domain=concept,
            concept=concept,
            modality="Video",
            text=f"{_clock(start)}-{_clock(end)}",
            source_title=path.name,
            source_url="",
            local_path=path,
            mime="video/mp4",
            clip_start=start,
            clip_end=end,
        )
        for start, end in windows
    ]
    return samples, (
        f"  Video {concept} <- {rel} ({size} bytes, {_clock(duration)}) "
        f"split into {len(samples)} clips"
    )


def _pdf_page_bytes(path: Path, page_number: int) -> bytes:
    """Write one page to a new PDF. Page numbers are 1-based."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    data = buffer.getvalue()
    if len(data) > INLINE_MEDIA_BYTES:
        raise RuntimeError(
            f"Page {page_number} of {path.name} is {len(data)} bytes, above the inline limit."
        )
    return data


def pdf_page_preview(path: Path, page_number: int, limit: int = 1500) -> str:
    """Plain text of one PDF page, for the search card."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = reader.pages[page_number - 1].extract_text() or ""
    text = " ".join(raw.split())
    return text[:limit]


def _pdf_samples(path: Path, size: int) -> tuple[list[Sample], str]:
    """One embedding row per page. The original file stays the file shown in search."""
    from pypdf import PdfReader

    concept = path.stem
    rel = path.name
    if size > MAX_PDF_BYTES:
        return [], f"  skip {rel}: larger than {MAX_PDF_BYTES // (1024 * 1024)} MB"
    try:
        page_count = len(PdfReader(str(path)).pages)
    except Exception as exc:
        return [], f"  skip {rel}: could not read the PDF ({exc})"
    if page_count < 1:
        return [], f"  skip {rel}: PDF has no pages"
    indexed = min(page_count, MAX_PDF_PAGES)
    samples = []
    for page in range(1, indexed + 1):
        preview = ""
        try:
            preview = pdf_page_preview(path, page, limit=500)
        except Exception:
            preview = ""
        samples.append(
            Sample(
                domain=concept,
                concept=concept,
                modality="PDF",
                text=preview or f"page {page}",
                source_title=path.name,
                source_url="",
                local_path=path,
                mime="application/pdf",
                clip_start=float(page),
                clip_end=float(page),
            )
        )
    note = f"  PDF {concept} <- {rel} ({size} bytes) split into {indexed} pages"
    if page_count > indexed:
        note += f" (stopped at {MAX_PDF_PAGES} of {page_count})"
    return samples, note


def _modality_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "Text"
    if suffix in IMAGE_EXTENSIONS:
        return "Image"
    if suffix in VIDEO_EXTENSIONS:
        return "Video"
    if suffix in AUDIO_EXTENSIONS:
        return "Audio"
    if suffix in PDF_EXTENSIONS:
        return "PDF"
    return None


def scan_media_dir(data_dir: Path, report: Report) -> list[Sample]:
    """Index text, image, video, audio, and PDF files placed directly in the data folder.

    Matching filenames are one datapoint: forest.txt, forest.jpg, forest.pdf,
    forest.mp4, and forest.wav belong together. Subfolders are ignored.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and not path.name.startswith(".") and path.suffix != ".part"
    ]
    samples: list[Sample] = []
    report.add(f"Data folder: {data_dir}")
    report.add("Looking for .txt/.md, .pdf, .jpg/.png, .mp4/.mov/.webm, and .wav/.mp3/.ogg")
    for path in files:
        modality = _modality_for(path)
        if modality is None:
            continue
        size = path.stat().st_size
        if modality == "Video":
            video_samples, note = _video_samples(path, size)
            if note:
                report.add(note)
            samples.extend(video_samples)
            continue
        if modality == "PDF":
            pdf_samples, note = _pdf_samples(path, size)
            if note:
                report.add(note)
            samples.extend(pdf_samples)
            continue
        if size > MAX_MEDIA_BYTES:
            report.add(f"  skip {path.name}: larger than {MAX_MEDIA_BYTES // (1024 * 1024)} MB")
            continue
        concept = path.stem
        text = ""
        if modality == "Text":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                report.add(f"  skip empty {path.name}")
                continue
        samples.append(
            Sample(
                domain=concept,
                concept=concept,
                modality=modality,
                text=text,
                source_title=path.name,
                source_url="",
                local_path=path,
                mime=None if modality == "Text" else MEDIA_MIME[path.suffix.lower()],
            )
        )
        report.add(f"  {modality} {concept} <- {path.relative_to(data_dir).as_posix()} ({size} bytes)")
    concepts = sorted({sample.concept for sample in samples})
    report.add(f"Found {len(samples)} files across {len(concepts)} datapoints.")
    if not samples:
        report.add("Drop forest.txt, forest.mp4, and forest.wav directly into this folder.")
    return samples


def download_dataset(data_dir: Path, report: Report) -> list[Sample]:
    """Download synced text, image, and audio samples for each concept."""
    report.add("Dataset: Wikimedia Commons open-access trimodal subset")
    report.add("Domains: Sports, Tech, Nature")
    samples: list[Sample] = []
    for spec in TRIPLETS:
        concept = spec["concept"]
        report.add(f"  concept {spec['domain']}/{concept}")
        image_url, image_mime = resolve_commons(spec["image_title"], spec["image_url"], "image")
        audio_url, audio_mime = resolve_commons(spec["audio_title"], spec["audio_url"], "audio")
        image_ext = ".png" if image_mime == "image/png" else ".jpg"
        audio_ext = Path(urllib.parse.urlparse(audio_url).path).suffix.lower() or ".ogg"
        image_path = data_dir / f"{_safe_name(concept)}_image{image_ext}"
        audio_path = data_dir / f"{_safe_name(concept)}_audio{audio_ext}"
        for path, url, mime in (
            (image_path, image_url, image_mime),
            (audio_path, audio_url, audio_mime),
        ):
            if not (path.is_file() and path.stat().st_size > 4096 and _payload_is_valid(path, mime)):
                report.add(f"    download {path.name}")
                _download(url, path)
                time.sleep(0.5)
            if not _payload_is_valid(path, mime):
                raise RuntimeError(f"Downloaded file is not valid {mime}: {path}")
            report.add(f"    ready {path.name} ({path.stat().st_size} bytes, {mime})")
        samples.append(
            Sample(
                spec["domain"],
                concept,
                "Text",
                spec["text"],
                spec["concept"],
                "",
                None,
                None,
            )
        )
        samples.append(
            Sample(
                spec["domain"],
                concept,
                "Image",
                spec["text"],
                spec["image_title"],
                image_url,
                image_path,
                image_mime,
            )
        )
        samples.append(
            Sample(
                spec["domain"],
                concept,
                "Audio",
                spec["text"],
                spec["audio_title"],
                audio_url,
                audio_path,
                audio_mime,
            )
        )
    report.add(f"Ingested {len(samples)} synced items across {len(TRIPLETS)} concepts.")
    return samples


def _stored_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_stamp(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _cache_signature(samples: list[Sample], model: str) -> str:
    payload = [
        {
            "model": model,
            "dim": OUTPUT_DIM,
            "domain": sample.domain,
            "concept": sample.concept,
            "modality": sample.modality,
            "text": sample.text,
            "source": sample.source_title,
            "path": "" if sample.local_path is None else sample.local_path.as_posix(),
            "stamp": _file_stamp(sample.local_path),
            **(
                {}
                if sample.clip_start is None
                else {"clip_start": sample.clip_start, "clip_end": sample.clip_end}
            ),
        }
        for sample in samples
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path, signature: str) -> list[list[float]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("signature") != signature or payload.get("dim") != OUTPUT_DIM:
        return None
    vectors = payload.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        return None
    if any(len(row) != OUTPUT_DIM for row in vectors):
        return None
    return vectors


def _write_cache(path: Path, signature: str, model: str, samples: list[Sample], vectors: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "model": model,
        "dim": OUTPUT_DIM,
        "items": [
            {
                "domain": sample.domain,
                "concept": sample.concept,
                "modality": sample.modality,
                "text": sample.text if sample.modality in {"Text", "Video", "PDF"} else "",
                "path": _stored_path(sample.local_path),
                "stamp": _file_stamp(sample.local_path),
                **(
                    {}
                    if sample.clip_start is None
                    else {"start": sample.clip_start, "end": sample.clip_end}
                ),
            }
            for sample in samples
        ],
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


ENV_PATH = ROOT / ".env"


def _api_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            f"Add your key to {ENV_PATH} as GEMINI_API_KEY=... "
            "Run with --self-test to exercise the JAX engine without the API."
        )
    return key


def _vector_from_response(response: types.EmbedContentResponse) -> list[float]:
    embeddings = list(response.embeddings or [])
    if not embeddings or embeddings[0].values is None:
        raise RuntimeError("Gemini returned an empty embedding.")
    vector = [float(value) for value in embeddings[0].values]
    if len(vector) != OUTPUT_DIM:
        raise RuntimeError(
            f"Expected output_dimensionality={OUTPUT_DIM}, received {len(vector)}."
        )
    return vector


def _contents_for(sample: Sample, payload: bytes | None, mime: str | None) -> types.Content:
    if sample.modality == "Text":
        # gemini-embedding-2 uses a prompt prefix instead of task_type.
        text = f"task: clustering | query: {sample.text[:8000]}"
        return types.Content(parts=[types.Part.from_text(text=text)])
    if payload is None or mime is None:
        raise RuntimeError(f"Missing media bytes for {sample.concept} {sample.modality}")
    return types.Content(
        parts=[types.Part.from_bytes(data=payload, mime_type=mime)]
    )


def _payload_for(sample: Sample) -> tuple[bytes | None, str | None]:
    if sample.local_path is None or sample.modality == "Text":
        return None, None
    if sample.modality == "Video" and sample.clip_start is not None and sample.clip_end is not None:
        return _transcode_clip(sample.local_path, sample.clip_start, sample.clip_end), "video/mp4"
    if sample.modality == "PDF" and sample.clip_start is not None:
        return _pdf_page_bytes(sample.local_path, int(sample.clip_start)), "application/pdf"
    return sample.local_path.read_bytes(), sample.mime


async def _embed_one(
    client: genai.Client,
    model: str,
    sample: Sample,
    payload: bytes | None,
    mime: str | None,
) -> list[float]:
    contents = _contents_for(sample, payload, mime)
    config = types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM)
    delay = 1.0
    last_error: BaseException | None = None
    for _attempt in range(5):
        try:
            response = await client.aio.models.embed_content(
                model=model,
                contents=contents,
                config=config,
            )
            return _vector_from_response(response)
        except Exception as exc:
            last_error = exc
            if isinstance(exc, ValueError):
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Embedding failed for {sample.concept}/{sample.modality}") from last_error


async def _embed_batch(
    client: genai.Client,
    model: str,
    samples: list[Sample],
    concurrency: int,
    report: Report,
) -> list[list[float]]:
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def run(sample: Sample) -> list[float]:
        nonlocal completed
        async with semaphore:
            payload, mime = await asyncio.to_thread(_payload_for, sample)
            vector = await _embed_one(client, model, sample, payload, mime)
            completed += 1
            report.add(
                f"  embedded {completed}/{len(samples)} "
                f"{sample.modality} {sample.domain}/{sample.concept} "
                f"dim={len(vector)}"
            )
            return vector

    return list(await asyncio.gather(*(run(sample) for sample in samples)))


async def embed_samples(
    samples: list[Sample],
    concurrency: int,
    report: Report,
) -> tuple[list[list[float]], str]:
    client = genai.Client(
        api_key=_api_key(),
        http_options=types.HttpOptions(timeout=180_000),
    )
    report.add(f"Embedding model: {EMBED_MODEL}")
    report.add(f"output_dimensionality: {OUTPUT_DIM} (Matryoshka / MRL)")
    try:
        vectors = await _embed_batch(client, EMBED_MODEL, samples, concurrency, report)
        return vectors, EMBED_MODEL
    finally:
        closer = getattr(client.aio, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:
                pass


def block_until_ready(tree: object) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.3f} ms"
    return f"{seconds:.3f} s"


@jax.jit
def cosine_similarity_matrix(a: jax.Array, b: jax.Array) -> jax.Array:
    """Row-wise cosine similarity, Sim(A, B) = (A @ B.T) / (|A|_2 |B|_2)."""
    numer = a @ b.T
    a_norm = jnp.linalg.norm(a, axis=1)
    b_norm = jnp.linalg.norm(b, axis=1)
    denom = jnp.maximum(a_norm[:, None] * b_norm[None, :], jnp.float32(1e-12))
    return numer / denom


@partial(jax.jit, static_argnames=("k",))
def cross_modal_topk(
    query: jax.Array,
    image_bank: jax.Array,
    audio_bank: jax.Array,
    k: int,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Top-k image and audio hits for one text query, via jax.lax.top_k."""

    def search(bank: jax.Array) -> tuple[jax.Array, jax.Array]:
        sims = cosine_similarity_matrix(query.reshape(1, -1), bank)[0]
        return jax.lax.top_k(sims, k)

    return search(image_bank), search(audio_bank)


@partial(jax.jit, static_argnames=("k",))
def topk_against_index(
    query: jax.Array,
    matrix: jax.Array,
    k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score one query against every row and return top-k values, indices, and all scores."""
    sims = cosine_similarity_matrix(query.reshape(1, -1), matrix)[0]
    values, indices = jax.lax.top_k(sims, k)
    return values, indices, sims


@jax.jit
def cross_modal_duplicates(
    sim: jax.Array,
    modality_ids: jax.Array,
    tau: jax.Array,
) -> jax.Array:
    """Boolean adjacency for cross-modal pairs with similarity above tau."""
    different = modality_ids[:, None] != modality_ids[None, :]
    off_diagonal = ~jnp.eye(sim.shape[0], dtype=bool)
    return (sim > tau) & different & off_diagonal


@partial(jax.jit, static_argnames=("k", "n_iters"))
def kmeans(
    x: jax.Array,
    k: int,
    n_iters: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Lloyd k-means. Iterations use jax.lax.scan; centroid updates use vmap.

    Returns final centroids (k, d), labels (n,), and per-iteration inertia (n_iters,).
    """
    perm = jax.random.permutation(key, x.shape[0])
    init = x[perm[:k]]

    def pairwise_sqdist(centroids: jax.Array) -> jax.Array:
        x2 = jnp.sum(x * x, axis=1, keepdims=True)
        c2 = jnp.sum(centroids * centroids, axis=1)
        return x2 - jnp.float32(2.0) * (x @ centroids.T) + c2

    def update(centroids: jax.Array, _step: jax.Array) -> tuple[jax.Array, jax.Array]:
        dist = pairwise_sqdist(centroids)
        labels = jnp.argmin(dist, axis=1)

        def new_centroid(j: jax.Array) -> jax.Array:
            mask = (labels == j).astype(x.dtype)
            count = jnp.sum(mask)
            summed = jnp.sum(x * mask[:, None], axis=0)
            updated = summed / jnp.maximum(count, jnp.float32(1.0))
            return jnp.where(count > 0.0, updated, centroids[j])

        new_centroids = jax.vmap(new_centroid)(jnp.arange(k))
        inertia = jnp.sum(dist[jnp.arange(x.shape[0]), labels])
        return new_centroids, inertia

    final_centroids, inertia_trace = jax.lax.scan(update, init, jnp.arange(n_iters))
    labels = jnp.argmin(pairwise_sqdist(final_centroids), axis=1)
    return final_centroids, labels, inertia_trace


@jax.jit
def kmeans_inertia(x: jax.Array, centroids: jax.Array, labels: jax.Array) -> jax.Array:
    assigned = centroids[labels]
    return jnp.sum((x - assigned) ** 2)


@jax.jit
def pca_project(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Project rows onto the top two right singular vectors. Returns coords and explained ratios."""
    centered = x - jnp.mean(x, axis=0, keepdims=True)
    _u, singular, vt = jnp.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    variance = singular**2 / jnp.maximum(x.shape[0] - 1, 1)
    explained = variance[:2] / jnp.maximum(jnp.sum(variance), jnp.float32(1e-12))
    return coords, explained


@jax.jit
def grouped_mean_similarity(
    sim: jax.Array,
    concept_ids: jax.Array,
    modality_ids: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Mean upper-triangle cross-modal similarity for same-concept and different-concept pairs."""
    n = sim.shape[0]
    different_mod = modality_ids[:, None] != modality_ids[None, :]
    upper = jnp.arange(n)[None, :] > jnp.arange(n)[:, None]
    cross = different_mod & upper
    same = cross & (concept_ids[:, None] == concept_ids[None, :])
    different = cross & (concept_ids[:, None] != concept_ids[None, :])

    def masked_mean(mask: jax.Array) -> jax.Array:
        weight = mask.astype(sim.dtype)
        return jnp.sum(sim * weight) / jnp.maximum(jnp.sum(weight), jnp.float32(1.0))

    return masked_mean(same), masked_mean(different)


@partial(jax.jit, static_argnames=("k",))
def top_cross_modal_pairs(
    sim: jax.Array,
    modality_ids: jax.Array,
    k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = sim.shape[0]
    different = modality_ids[:, None] != modality_ids[None, :]
    upper = jnp.arange(n)[None, :] > jnp.arange(n)[:, None]
    scored = jnp.where(different & upper, sim, -jnp.inf)
    values, flat = jax.lax.top_k(scored.ravel(), k)
    return values, flat // n, flat % n


def benchmark(name: str, fn, args: tuple, report: Report) -> object:
    """Time a JIT function. The first call includes XLA compilation."""
    started = time.perf_counter()
    result = fn(*args)
    block_until_ready(result)
    cold = time.perf_counter() - started
    steady_samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        result = fn(*args)
        block_until_ready(result)
        steady_samples.append(time.perf_counter() - started)
    steady = sorted(steady_samples)[1]
    report.add(f"JAX benchmark: {name}")
    report.add(f"  compile + first execution: {_fmt_seconds(cold)}")
    report.add(f"  steady-state execution (median of 3): {_fmt_seconds(steady)}")
    report.add(f"  compile overhead: {_fmt_seconds(max(cold - steady, 0.0))}")
    return result


def _ids(samples: list[Sample]) -> tuple[jax.Array, jax.Array]:
    concept_names = list(dict.fromkeys(sample.concept for sample in samples))
    concept_index = {name: idx for idx, name in enumerate(concept_names)}
    concept_ids = jnp.array([concept_index[sample.concept] for sample in samples], dtype=jnp.int32)
    modality_ids = jnp.array([sample.modality_id for sample in samples], dtype=jnp.int32)
    return concept_ids, modality_ids


def build_figure(
    coords: jax.Array,
    explained: jax.Array,
    labels: jax.Array,
    samples: list[Sample],
    title: str = "Gemini Embedding 2 trimodal space (JAX PCA + JAX K-Means)",
):
    points = coords.tolist()
    cluster_labels = [f"cluster {int(label)}" for label in labels.tolist()]
    hover = [
        f"{sample.domain} | {sample.concept} | {sample.modality}<br>{sample.text}"
        for sample in samples
    ]
    figure = px.scatter(
        x=[row[0] for row in points],
        y=[row[1] for row in points],
        color=cluster_labels,
        symbol=[sample.modality for sample in samples],
        hover_name=hover,
        title=title,
        labels={"x": "PC1", "y": "PC2", "color": "K-Means cluster", "symbol": "Modality"},
        color_discrete_sequence=px.colors.qualitative.Dark2,
    )
    for trace in figure.data:
        name = trace.name or ""
        for modality, symbol in SYMBOL_FOR_MODALITY.items():
            if modality in name:
                trace.marker.symbol = symbol
                break
        trace.marker.size = 14
        trace.marker.line = dict(width=1, color="#1f1f1f")
    pc1 = 100.0 * float(explained[0])
    pc2 = 100.0 * float(explained[1])
    figure.update_layout(
        template="plotly_white",
        width=1100,
        height=740,
        margin=dict(l=60, r=20, t=70, b=60),
        xaxis_title=f"PC1 ({pc1:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2:.1f}% variance)",
        legend_title="Cluster and modality",
    )
    return figure


def write_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False},
    )


def _majority(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def run_analytics(
    embeddings: jax.Array,
    samples: list[Sample],
    clusters: int,
    tau: float,
    topk: int,
    kmeans_iters: int,
    report: Report,
    html_path: Path,
    show_browser: bool,
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[1] != OUTPUT_DIM:
        raise RuntimeError(f"Expected embeddings shape (N, {OUTPUT_DIM}), got {embeddings.shape}.")
    report.add("")
    report.add(f"Embeddings stacked with jnp.array: shape {tuple(int(v) for v in embeddings.shape)}")
    report.add(f"JAX backend: {jax.default_backend()} devices={jax.devices()}")

    concept_ids, modality_ids = _ids(samples)
    image_rows = [i for i, sample in enumerate(samples) if sample.modality == "Image"]
    audio_rows = [i for i, sample in enumerate(samples) if sample.modality == "Audio"]
    text_rows = [i for i, sample in enumerate(samples) if sample.modality == "Text"]
    image_bank = embeddings[jnp.array(image_rows)]
    audio_bank = embeddings[jnp.array(audio_rows)]
    k_retrieve = min(topk, len(image_rows), len(audio_rows))
    if k_retrieve < 1:
        raise RuntimeError("Need at least one image and one audio embedding.")

    report.add("")
    sim = benchmark(
        f"cosine similarity matrix {tuple(int(v) for v in embeddings.shape)}",
        cosine_similarity_matrix,
        (embeddings, embeddings),
        report,
    )
    repeats = max(1, (512 + embeddings.shape[0] - 1) // embeddings.shape[0])
    scaled = jnp.repeat(embeddings, repeats, axis=0)
    block_until_ready(scaled)
    benchmark(
        f"cosine similarity matrix scaled {tuple(int(v) for v in scaled.shape)}",
        cosine_similarity_matrix,
        (scaled, scaled),
        report,
    )

    key = jax.random.PRNGKey(0)
    report.add("")
    centroids, labels, inertia_trace = benchmark(
        f"k-means k={clusters} iters={kmeans_iters} n={embeddings.shape[0]}",
        kmeans,
        (embeddings, clusters, kmeans_iters, key),
        report,
    )
    benchmark(
        f"k-means scaled k={clusters} iters={kmeans_iters} n={scaled.shape[0]}",
        kmeans,
        (scaled, clusters, kmeans_iters, key),
        report,
    )
    final_inertia = kmeans_inertia(embeddings, centroids, labels)
    block_until_ready(final_inertia)
    report.add(
        f"K-Means inertia trace: start={float(inertia_trace[0]):.4f} "
        f"end={float(inertia_trace[-1]):.4f} final={float(final_inertia):.4f}"
    )

    same_mean, diff_mean = grouped_mean_similarity(sim, concept_ids, modality_ids)
    block_until_ready((same_mean, diff_mean))
    report.add("")
    report.add("Cross-modal neighborhood check (upper triangle, different modalities)")
    report.add(f"  mean similarity, same concept:      {float(same_mean):.4f}")
    report.add(f"  mean similarity, different concept: {float(diff_mean):.4f}")

    tau_value = jnp.float32(tau)
    duplicates = cross_modal_duplicates(sim, modality_ids, tau_value)
    block_until_ready(duplicates)
    pair_count = int(jnp.sum(duplicates)) // 2
    report.add(f"Cross-modal near-duplicates at tau={tau:.2f}: {pair_count} unique pairs")
    duplicate_list = jnp.argwhere(duplicates).tolist()
    printed: set[tuple[int, int]] = set()
    for left, right in duplicate_list:
        i, j = int(left), int(right)
        if i >= j or (i, j) in printed:
            continue
        printed.add((i, j))
        report.add(
            f"  {samples[i].modality}:{samples[i].concept}  ~  "
            f"{samples[j].modality}:{samples[j].concept}  "
            f"sim={float(sim[i, j]):.4f}"
        )
    if not printed:
        report.add("  no pair crossed the threshold")

    pair_k = min(8, embeddings.shape[0] * (embeddings.shape[0] - 1) // 2)
    pair_vals, pair_i, pair_j = top_cross_modal_pairs(sim, modality_ids, pair_k)
    block_until_ready((pair_vals, pair_i, pair_j))
    report.add("Strongest cross-modal pairs")
    for value, i, j in zip(pair_vals.tolist(), pair_i.tolist(), pair_j.tolist()):
        if value == float("-inf"):
            continue
        report.add(
            f"  {float(value):.4f}  {samples[int(i)].modality}:{samples[int(i)].concept}"
            f"  ~  {samples[int(j)].modality}:{samples[int(j)].concept}"
        )

    report.add("")
    report.add(f"Cross-modal top-{k_retrieve} retrieval from each text query")
    hits = 0
    trials = 0
    for text_index in text_rows:
        query = embeddings[text_index]
        (image_vals, image_idx), (audio_vals, audio_idx) = cross_modal_topk(
            query, image_bank, audio_bank, k_retrieve
        )
        block_until_ready(((image_vals, image_idx), (audio_vals, audio_idx)))
        query_sample = samples[text_index]
        report.add(f"  query {query_sample.domain}/{query_sample.concept}")
        for rank, (bank_index, score) in enumerate(zip(image_idx.tolist(), image_vals.tolist()), start=1):
            hit = samples[image_rows[int(bank_index)]]
            mark = "MATCH" if hit.concept == query_sample.concept else "other"
            report.add(f"    image #{rank} {hit.concept} sim={float(score):.4f} {mark}")
        for rank, (bank_index, score) in enumerate(zip(audio_idx.tolist(), audio_vals.tolist()), start=1):
            hit = samples[audio_rows[int(bank_index)]]
            mark = "MATCH" if hit.concept == query_sample.concept else "other"
            report.add(f"    audio #{rank} {hit.concept} sim={float(score):.4f} {mark}")
        top_image = samples[image_rows[int(image_idx[0])]]
        top_audio = samples[audio_rows[int(audio_idx[0])]]
        trials += 2
        hits += int(top_image.concept == query_sample.concept)
        hits += int(top_audio.concept == query_sample.concept)
    report.add(f"Top-1 cross-modal concept hits: {hits}/{trials}")

    label_list = [int(value) for value in labels.tolist()]
    report.add("")
    report.add("K-Means cluster membership")
    for cluster in range(clusters):
        members = [samples[i] for i, label in enumerate(label_list) if label == cluster]
        if not members:
            report.add(f"  cluster {cluster}: empty")
            continue
        domains = [member.domain for member in members]
        concepts = sorted({member.concept for member in members})
        modalities = [member.modality for member in members]
        report.add(
            f"  cluster {cluster}: n={len(members)} majority_domain={_majority(domains)} "
            f"concepts={concepts} modalities={modalities}"
        )

    coords, explained = pca_project(embeddings)
    block_until_ready((coords, explained))
    figure = build_figure(coords, explained, labels, samples)
    write_figure(figure, html_path)
    report.add("")
    report.add(f"Wrote interactive scatter: {html_path}")
    report.add(
        "Point color is the JAX K-Means cluster. "
        "Point shape is modality: circle=Text, square=Image, diamond=Audio."
    )
    if show_browser:
        try:
            figure.show()
        except Exception as exc:
            report.add(f"Plotly could not open a browser window ({exc}). Open the HTML file.")


def run_self_test(report: Report) -> None:
    """Exercise every JIT kernel on synthetic unit vectors with known neighborhoods."""
    report.add("Self-test: synthetic 768-d vectors (no Gemini API call)")
    n_concepts = 5
    modalities = ("Text", "Image", "Audio")
    key = jax.random.PRNGKey(0)
    center_key, noise_key = jax.random.split(key)
    centers = jax.random.normal(center_key, (n_concepts, OUTPUT_DIM), dtype=jnp.float32)
    centers = centers / jnp.linalg.norm(centers, axis=1, keepdims=True)
    # Per-coordinate scale keeps the noise norm small in 768-d, so same-concept
    # modalities stay above the 0.85 duplicate threshold.
    noise = jax.random.normal(noise_key, (n_concepts, 3, OUTPUT_DIM), dtype=jnp.float32) * jnp.float32(0.01)
    points = centers[:, None, :] + noise
    points = points.reshape(-1, OUTPUT_DIM)
    points = points / jnp.linalg.norm(points, axis=1, keepdims=True)
    embeddings = jnp.array(points, dtype=jnp.float32)
    domains = ("Sports", "Tech", "Nature")
    samples: list[Sample] = []
    for concept_id in range(n_concepts):
        for modality in modalities:
            samples.append(
                Sample(
                    domain=domains[concept_id % 3],
                    concept=f"concept_{concept_id}",
                    modality=modality,
                    text=f"synthetic {modality} for concept {concept_id}",
                    source_title="synthetic",
                    source_url="",
                    local_path=None,
                    mime=None,
                )
            )
    concept_ids, modality_ids = _ids(samples)
    # Benchmark before any other call so the first timing includes XLA compilation.
    benchmark(
        "self-test cosine similarity",
        cosine_similarity_matrix,
        (embeddings, embeddings),
        report,
    )
    scaled = jnp.repeat(embeddings, 32, axis=0)
    block_until_ready(scaled)
    benchmark(
        "self-test cosine similarity scaled",
        cosine_similarity_matrix,
        (scaled, scaled),
        report,
    )
    benchmark(
        "self-test k-means",
        kmeans,
        (embeddings, n_concepts, 15, jax.random.PRNGKey(1)),
        report,
    )
    benchmark(
        "self-test k-means scaled",
        kmeans,
        (scaled, n_concepts, 15, jax.random.PRNGKey(1)),
        report,
    )
    sim = cosine_similarity_matrix(embeddings, embeddings)
    block_until_ready(sim)
    if sim.shape != (len(samples), len(samples)):
        raise AssertionError(f"Unexpected similarity shape {sim.shape}")
    if not bool(jnp.allclose(sim, sim.T, atol=1e-4)):
        raise AssertionError("Cosine matrix is not symmetric.")
    diagonal = jnp.diag(sim)
    if float(jnp.min(diagonal)) < 0.99:
        raise AssertionError("Self-similarity is below 1.")

    same_mean, diff_mean = grouped_mean_similarity(sim, concept_ids, modality_ids)
    block_until_ready((same_mean, diff_mean))
    if float(same_mean) <= float(diff_mean):
        raise AssertionError("Same-concept pairs are not closer than different-concept pairs.")
    if float(same_mean) <= 0.85:
        raise AssertionError("Same-concept cross-modal similarity did not clear tau=0.85.")

    duplicates = cross_modal_duplicates(sim, modality_ids, jnp.float32(0.85))
    block_until_ready(duplicates)
    if int(jnp.sum(duplicates)) < 1:
        raise AssertionError("Duplicate detector found no pairs above tau.")
    for i, j in jnp.argwhere(duplicates).tolist():
        if samples[int(i)].concept != samples[int(j)].concept:
            raise AssertionError("Duplicate threshold flagged two different concepts.")

    image_rows = [i for i, sample in enumerate(samples) if sample.modality == "Image"]
    audio_rows = [i for i, sample in enumerate(samples) if sample.modality == "Audio"]
    (image_vals, image_idx), (audio_vals, audio_idx) = cross_modal_topk(
        embeddings[0], embeddings[jnp.array(image_rows)], embeddings[jnp.array(audio_rows)], 2
    )
    block_until_ready(((image_vals, image_idx), (audio_vals, audio_idx)))
    if samples[image_rows[int(image_idx[0])]].concept != samples[0].concept:
        raise AssertionError("Top image hit missed the query concept.")
    if samples[audio_rows[int(audio_idx[0])]].concept != samples[0].concept:
        raise AssertionError("Top audio hit missed the query concept.")
    if float(image_vals[0]) < float(image_vals[1]):
        raise AssertionError("top_k values are not descending.")

    _centroids, labels, inertia_trace = kmeans(embeddings, n_concepts, 15, jax.random.PRNGKey(1))
    block_until_ready((labels, inertia_trace))
    if labels.shape != (len(samples),):
        raise AssertionError("Unexpected label shape.")
    if int(jnp.min(labels)) < 0 or int(jnp.max(labels)) >= n_concepts:
        raise AssertionError("Labels fell outside 0..k-1.")
    if float(inertia_trace[-1]) > float(inertia_trace[0]) + 1e-3:
        raise AssertionError("K-Means inertia increased.")

    coords, explained = pca_project(embeddings)
    block_until_ready((coords, explained))
    if coords.shape != (len(samples), 2):
        raise AssertionError("PCA did not return 2 coordinates.")
    figure = build_figure(
        coords,
        explained,
        labels,
        samples,
        title="Synthetic self-test scatter (not a Gemini embedding run)",
    )
    symbols = {trace.marker.symbol for trace in figure.data}
    if not {"circle", "square", "diamond"}.issubset(symbols):
        raise AssertionError(f"Modality symbols missing from figure: {symbols}")
    html_path = ROOT / "artifacts" / "self_test_clusters.html"
    write_figure(figure, html_path)
    report.add(f"Self-test figure: {html_path}")
    report.add(
        f"Self-test neighborhood gap: same={float(same_mean):.4f} different={float(diff_mean):.4f}"
    )
    report.add("Self-test passed.")


def _sample_cache_key(sample: Sample, stamp: str | None) -> tuple:
    text = sample.text if sample.modality == "Text" else ""
    return (
        _stored_path(sample.local_path),
        sample.modality,
        text,
        sample.clip_start,
        sample.clip_end,
        stamp,
    )


def _reusable_vectors(path: Path) -> dict[tuple, list[float]]:
    """Rows from an older index. A missing stamp matches the file as it is now."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if payload.get("model") != EMBED_MODEL or payload.get("dim") != OUTPUT_DIM:
        return {}
    items = payload.get("items")
    vectors = payload.get("vectors")
    if not isinstance(items, list) or not isinstance(vectors, list) or len(items) != len(vectors):
        return {}
    reusable: dict[tuple, list[float]] = {}
    for item, vector in zip(items, vectors):
        if not isinstance(vector, list) or len(vector) != OUTPUT_DIM:
            continue
        text = item.get("text") or ""
        if item.get("modality") != "Text":
            text = ""
        reusable[(
            item.get("path") or "",
            item.get("modality"),
            text,
            item.get("start"),
            item.get("end"),
            item.get("stamp"),
        )] = vector
    return reusable


def build_index(
    data_dir: Path,
    cache_path: Path,
    concurrency: int,
    refresh: bool,
    report: Report,
) -> tuple[list[Sample], list[list[float]]]:
    """Embed new or changed files in the data folder. Reuse the cache when nothing changed."""
    samples = scan_media_dir(data_dir, report)
    if not samples:
        return [], []
    signature = _cache_signature(samples, EMBED_MODEL)
    cached = None if refresh else _read_cache(cache_path, signature)
    if cached is not None:
        report.add(f"Loaded cached {EMBED_MODEL} embeddings from {cache_path}")
        return samples, cached
    reused = {} if refresh else _reusable_vectors(cache_path)
    vectors: list[list[float] | None] = []
    pending: list[Sample] = []
    pending_at: list[int] = []
    for index, sample in enumerate(samples):
        stamp = _file_stamp(sample.local_path)
        vector = reused.get(_sample_cache_key(sample, stamp))
        if vector is None and not refresh:
            vector = reused.get(_sample_cache_key(sample, None))
        if vector is None:
            vectors.append(None)
            pending.append(sample)
            pending_at.append(index)
        else:
            vectors.append(vector)
    if pending:
        report.add(f"Embedding {len(pending)} new or changed files. Reusing {len(samples) - len(pending)}.")
        fresh, _model_used = asyncio.run(embed_samples(pending, concurrency, report))
        for index, vector in zip(pending_at, fresh):
            vectors[index] = vector
    else:
        report.add(f"Reused all {len(samples)} embeddings.")
    filled = [vector for vector in vectors if vector is not None]
    if len(filled) != len(samples):
        raise RuntimeError("Indexing left some files without a vector.")
    _write_cache(cache_path, signature, EMBED_MODEL, samples, filled)
    report.add(f"Cached embeddings: {cache_path}")
    return samples, filled


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run JAX kernels on synthetic vectors.")
    parser.add_argument("--refresh-embeddings", action="store_true", help="Ignore the embedding cache.")
    parser.add_argument("--clusters", type=int, default=3, help="K-Means K. Default matches the three domains.")
    parser.add_argument("--tau", type=float, default=0.85, help="Cross-modal duplicate cosine threshold.")
    parser.add_argument("--topk", type=int, default=3, help="Cross-modal neighbors to retrieve.")
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-browser", action="store_true", help="Write the Plotly HTML without opening it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv(ENV_PATH)
    try:
        jax.config.update("jax_enable_x64", False)
    except AttributeError:
        pass
    report = Report()
    if args.self_test:
        run_self_test(report)
        report.dump(ROOT / "artifacts" / "self_test.log")
        return

    if args.clusters < 1:
        raise SystemExit("--clusters must be at least 1.")
    samples, vectors = build_index(
        args.data_dir,
        args.cache,
        args.concurrency,
        args.refresh_embeddings,
        report,
    )
    if not samples:
        raise SystemExit(
            f"No text, video, or audio files in {args.data_dir}. "
            "Drop matching names in that folder, such as forest.txt, forest.mp4, and forest.wav."
        )
    if args.clusters > len(samples):
        args.clusters = len(samples)
    embeddings = jnp.array(vectors, dtype=jnp.float32)
    run_analytics(
        embeddings,
        samples,
        clusters=args.clusters,
        tau=args.tau,
        topk=args.topk,
        kmeans_iters=args.kmeans_iters,
        report=report,
        html_path=args.html,
        show_browser=not args.no_browser,
    )
    report.dump(args.log)


if __name__ == "__main__":
    main()
