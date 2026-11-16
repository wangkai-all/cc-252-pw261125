"""WAV payload helpers: minimal 16-bit mono PCM RIFF writer/parser and a
sine generator used by the device simulator. No audio processing beyond
what indexing needs (parse header, count frames, hash PCM)."""
import hashlib
import math
import struct

ALLOWED_RATES = {8000, 16000, 22050, 44100, 48000}


def make_wav(pcm, sample_rate):
    """Wrap raw 16-bit mono PCM bytes in a RIFF/WAVE container."""
    byte_rate = sample_rate * 2
    block_align = 2
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                byte_rate, block_align, 16)
    return hdr + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm


def parse_wav(buf):
    """Parse a WAV buffer. Returns dict(rate, frames, pcm) or raises ValueError."""
    if len(buf) < 44 or buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4]
        size = struct.unpack("<I", buf[pos + 4:pos + 8])[0]
        body = buf[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            if len(body) < 16:
                raise ValueError("short fmt chunk")
            audio_fmt, channels, rate, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (audio_fmt, channels, rate, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None:
        raise ValueError("missing fmt chunk")
    if data is None:
        raise ValueError("missing data chunk")
    audio_fmt, channels, rate, bits = fmt
    if audio_fmt != 1:
        raise ValueError("not PCM")
    if channels != 1:
        raise ValueError("not mono")
    if bits != 16:
        raise ValueError("not 16-bit")
    if len(data) % 2:
        raise ValueError("odd PCM byte count")
    return {"rate": rate, "frames": len(data) // 2, "pcm": data}


def pcm_sha256(pcm):
    return hashlib.sha256(pcm).hexdigest()


def gen_tone(counter_start, nsamples, rate, freq, amplitude=0.3):
    """Deterministic sine continuing phase from the absolute sample counter,
    so consecutive segments of one device form a continuous tone."""
    out = bytearray()
    amp = int(amplitude * 32767)
    for i in range(nsamples):
        t = (counter_start + i) / rate
        v = int(amp * math.sin(2.0 * math.pi * freq * t))
        out += struct.pack("<h", max(-32768, min(32767, v)))
    return bytes(out)
