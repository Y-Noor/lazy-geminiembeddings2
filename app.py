"""Search text, video, and audio dropped in the data folder.

Drop matching filenames directly into data/trimodal, such as forest.txt,
forest.mp4, and forest.wav. Index them, then ask for what you want.

Setup::

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import streamlit as st
from google import genai
from google.genai import types

from jax_gemini2_trimodal import (
    DEFAULT_CACHE,
    DEFAULT_DATA_DIR,
    EMBED_MODEL,
    MAX_MEDIA_BYTES,
    MAX_PDF_BYTES,
    MAX_VIDEO_BYTES,
    OUTPUT_DIM,
    ROOT,
    Report,
    Sample,
    _api_key,
    _file_stamp,
    _modality_for,
    _stored_path,
    block_until_ready,
    build_figure,
    build_index,
    kmeans,
    load_dotenv,
    pca_project,
    topk_against_index,
)

def _resolve_path(stored: str) -> Path | None:
    if not stored:
        return None
    path = Path(stored)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _indexable(path: Path) -> bool:
    """Same skip rules as indexing, without opening video or PDF contents."""
    modality = _modality_for(path)
    if modality is None:
        return False
    size = path.stat().st_size
    if modality == "Video":
        return size <= MAX_VIDEO_BYTES
    if modality == "PDF":
        return size <= MAX_PDF_BYTES
    if size > MAX_MEDIA_BYTES:
        return False
    if modality == "Text":
        return bool(path.read_text(encoding="utf-8", errors="replace").strip())
    return True


def cache_is_current() -> bool:
    """Compare file size and mtime with the cache. Does not scan or embed."""
    if not DEFAULT_CACHE.is_file() or not DEFAULT_DATA_DIR.is_dir():
        return False
    try:
        payload = json.loads(DEFAULT_CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    indexed: dict[str, set[str]] = {}
    for item in payload.get("items") or []:
        indexed.setdefault(item.get("path") or "", set()).add(item.get("stamp") or "")
    seen: set[str] = set()
    for path in DEFAULT_DATA_DIR.iterdir():
        if not path.is_file() or path.name.startswith(".") or path.suffix == ".part":
            continue
        if not _indexable(path):
            continue
        stored = _stored_path(path)
        seen.add(stored)
        if _file_stamp(path) not in indexed.get(stored, set()):
            return False
    return all(not stored or stored in seen for stored in indexed)


@st.cache_resource
def load_index() -> tuple[jax.Array, list[dict]]:
    """Load the cached matrix and the file behind each row."""
    if not DEFAULT_CACHE.is_file():
        raise FileNotFoundError(f"Nothing is indexed yet. Drop files into {DEFAULT_DATA_DIR}.")
    payload = json.loads(DEFAULT_CACHE.read_text(encoding="utf-8"))
    if payload.get("model") != EMBED_MODEL or payload.get("dim") != OUTPUT_DIM:
        raise RuntimeError(
            f"Cache was built for {payload.get('model')} at dim {payload.get('dim')}. "
            f"This app expects {EMBED_MODEL} at {OUTPUT_DIM}."
        )
    vectors = payload["vectors"]
    items = payload["items"]
    if len(vectors) != len(items):
        raise RuntimeError("Cache items and vectors are different lengths.")
    if any("path" not in item for item in items):
        raise RuntimeError("This index has no file paths. Index the data folder again.")
    records = [
        {
            "domain": item["domain"],
            "concept": item["concept"],
            "modality": item["modality"],
            "text": item.get("text") or "",
            "path": _resolve_path(item.get("path") or ""),
            "start": item.get("start"),
            "end": item.get("end"),
        }
        for item in items
    ]
    embeddings = jnp.array(vectors, dtype=jnp.float32)
    return embeddings, records


def embed_query(text: str) -> jax.Array:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    client = genai.Client(api_key=_api_key())
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=f"task: clustering | query: {text}",
        config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
    )
    values = response.embeddings[0].values if response.embeddings else None
    if not values or len(values) != OUTPUT_DIM:
        raise RuntimeError(f"Query embedding was not {OUTPUT_DIM} dimensions.")
    return jnp.array(values, dtype=jnp.float32)


def search(query: jax.Array, matrix: jax.Array, k: int) -> tuple[list[float], list[int], list[float]]:
    values, indices, sims = topk_against_index(query, matrix, k)
    block_until_ready((values, indices, sims))
    return values.tolist(), [int(i) for i in indices.tolist()], sims.tolist()


def top_modality(
    sims: list[float], records: list[dict], modality: str, n: int
) -> list[tuple[int, float]]:
    ranked = [
        (index, score)
        for index, (score, record) in enumerate(zip(sims, records))
        if record["modality"] == modality
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:n]


def embedding_figure(embeddings: jax.Array, records: list[dict]):
    """PCA scatter of the cached matrix. Color is cluster, shape is modality."""
    samples = [
        Sample(
            domain=record["domain"],
            concept=record["concept"],
            modality=record["modality"],
            text=record["text"] or record["concept"],
            source_title=record["path"].name if record["path"] else record["concept"],
            source_url="",
            local_path=record["path"],
            mime=None,
        )
        for record in records
    ]
    n = int(embeddings.shape[0])
    clusters = max(1, min(len({record["concept"] for record in records}), n))
    _centroids, labels, _inertia = kmeans(embeddings, clusters, 15, jax.random.PRNGKey(0))
    coords, explained = pca_project(embeddings)
    block_until_ready((coords, explained, labels))
    return build_figure(
        coords,
        explained,
        labels,
        samples,
        title="Cached embeddings (JAX PCA + JAX K-Means)",
    )


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def _render_pdf_page(path: Path, page_number: int):
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(path))
    try:
        page = document[page_number - 1]
        bitmap = page.render(scale=2)
        image = bitmap.to_pil()
        page.close()
        bitmap.close()
        return image
    finally:
        document.close()


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _show_card(hit: dict) -> None:
    label = f"{hit['concept']}  ·  {hit['score']:.3f}"
    if hit["modality"] == "Video" and hit.get("start") is not None and hit.get("end") is not None:
        label += f"  ·  {_clock(float(hit['start']))}–{_clock(float(hit['end']))}"
    elif hit["modality"] == "PDF" and hit.get("start") is not None:
        label += f"  ·  page {int(hit['start'])}"
    st.caption(label)
    path = hit["path"]
    if path is None or not path.is_file():
        st.write(hit["text"] or "File is missing.")
        return
    if hit["modality"] == "Image":
        st.image(str(path), width="stretch")
    elif hit["modality"] == "Video":
        kwargs = {}
        if hit.get("start") is not None:
            kwargs["start_time"] = float(hit["start"])
        if hit.get("end") is not None:
            kwargs["end_time"] = float(hit["end"])
        st.video(str(path), **kwargs)
    elif hit["modality"] == "Audio":
        st.audio(str(path))
    elif hit["modality"] == "Text":
        st.text(path.read_text(encoding="utf-8", errors="replace")[:4000])
        return
    elif hit["modality"] == "PDF":
        matched = int(hit["start"]) if hit.get("start") is not None else 1
        count = _pdf_page_count(path)
        page = matched
        if count > 1:
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=count,
                value=min(matched, count),
                key=f"pdf-page-{hit['concept']}-{matched}",
            )
        st.image(_render_pdf_page(path, int(page)), width="stretch")
        return
    if hit["text"] and hit["modality"] != "Video":
        st.write(hit["text"])


def _show_group(title: str, hits: list[dict], per_row: int) -> None:
    if not hits:
        return
    st.subheader(title)
    for offset in range(0, len(hits), per_row):
        batch = hits[offset : offset + per_row]
        columns = st.columns(per_row)
        for column, hit in zip(columns, batch):
            with column:
                _show_card(hit)


def main() -> None:
    st.set_page_config(page_title="Multimodal search", layout="wide")
    st.title("Multimodal search")
    st.caption(f"Query embedding: `{EMBED_MODEL}` at {OUTPUT_DIM} dimensions.")
    st.info(
        f"Drop matching files directly into `{DEFAULT_DATA_DIR}`, "
        "for example forest.txt, forest.pdf, forest.jpg, forest.mp4, and forest.wav, then index the folder. "
        "Videos larger than 20 MB are split into clips of about 2 minutes. "
        "Each PDF page is embedded on its own."
    )
    if st.button("Index data folder"):
        report = Report()
        with st.spinner("Embedding files in the data folder..."):
            try:
                samples, _vectors = build_index(
                    DEFAULT_DATA_DIR, DEFAULT_CACHE, concurrency=3, refresh=False, report=report
                )
            except SystemExit as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
                st.text("\n".join(report.lines))
                return
        st.text("\n".join(report.lines))
        if not samples:
            st.warning("No text, video, or audio files were found.")
            return
        load_index.clear()
        st.rerun()
    try:
        embeddings, records = load_index()
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        st.warning(str(exc))
        return
    if not cache_is_current():
        st.warning("The data folder has files that are not in the index yet.")

    if st.button("Visualize embeddings"):
        st.session_state["show_embeddings"] = True
    if st.session_state.get("show_embeddings"):
        if len(records) < 2:
            st.warning("Index at least two files before visualizing embeddings.")
        else:
            with st.spinner("Projecting embeddings with JAX PCA..."):
                figure = embedding_figure(embeddings, records)
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "Color is the JAX k-means cluster. "
                "Shape is modality: circle text, square image, diamond audio, triangle video, star PDF."
            )

    with st.form("search"):
        query = st.text_input("Query", placeholder="birds singing in a forest")
        k = st.slider(
            "Matches per type",
            min_value=1,
            max_value=max(1, min(12, len(records))),
            value=min(3, len(records)),
        )
        submitted = st.form_submit_button("Search", type="primary")
    if not submitted:
        return
    if not query.strip():
        st.warning("Enter a query.")
        return

    with st.spinner("Embedding the query and searching the indexed data matrix..."):
        try:
            query_vector = embed_query(query.strip())
        except SystemExit as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Could not embed the query: {exc}")
            return
        values, indices, sims = search(query_vector, embeddings, min(k, len(records)))

    def rows_for(modality: str) -> list[dict]:
        rows = []
        for index, score in top_modality(sims, records, modality, k):
            row = dict(records[index])
            row["score"] = float(score)
            rows.append(row)
        return rows

    shown = {
        "Images": rows_for("Image"),
        "Videos": rows_for("Video"),
        "Audio": rows_for("Audio"),
        "PDFs": rows_for("PDF"),
        "Text": rows_for("Text"),
    }
    _show_group("Images", shown["Images"], 4)
    _show_group("Videos", shown["Videos"], 2)
    _show_group("Audio", shown["Audio"], 2)
    _show_group("PDFs", shown["PDFs"], 2)
    _show_group("Text", shown["Text"], 2)

    hits = sorted(
        (hit for group in shown.values() for hit in group),
        key=lambda hit: hit["score"],
        reverse=True,
    )
    st.subheader("Top matches")
    for rank, hit in enumerate(hits, start=1):
        when = ""
        if hit["modality"] == "Video" and hit.get("start") is not None:
            when = f" · {_clock(float(hit['start']))}"
        elif hit["modality"] == "PDF" and hit.get("start") is not None:
            when = f" · page {int(hit['start'])}"
        st.write(
            f"{rank}. {hit['modality']} · {hit['concept']} · {hit['score']:.3f}{when}"
        )


if __name__ == "__main__":
    main()
