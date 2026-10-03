import io

from PIL import Image

from hazop_agent.pid import load_pid


def _png(data):
    return Image.open(io.BytesIO(data))


def test_example_jpeg_loads_as_png(pid_path):
    data, media = load_pid(pid_path)
    assert media == "image/png" and _png(data).size == (1480, 708)


def test_large_image_is_downscaled_and_metadata_dropped(tmp_path):
    p = tmp_path / "big.png"
    img = Image.new("RGB", (4000, 1000), "white")
    img.save(p, pnginfo=None)
    data, _ = load_pid(p, max_side=1000)
    out = _png(data)
    assert out.size == (1000, 250) and "exif" not in out.info


def test_pdf_first_page(tmp_path):
    p = tmp_path / "pid.pdf"
    Image.new("RGB", (600, 400), "white").save(p)       # Pillow writes a one-page PDF
    data, media = load_pid(p)
    assert media == "image/png" and max(_png(data).size) > 100


def test_unsupported_type(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("x")
    try:
        load_pid(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
