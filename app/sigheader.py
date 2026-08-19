"""Tolerant decoder for the base64 `signature` on Claude thinking blocks.

The signature is a protobuf whose envelope carries a PLAINTEXT header; the
header's f6 is the id of the model that actually served the request, which is
not always the model that was asked for (see
docs/superpowers/specs/2026-08-18-served-model-signature.md).

The format is undocumented and has already changed four times in six weeks
(the v4 header drops f6 entirely), so every function here is best-effort:
garbage in gets Nones out, never an exception. A signature we cannot read must
never cost us the event it rides on.

Pure stdlib on purpose: `split_signature` is a verbatim copy of the one in
client/claude-stats-push.py (which may not import anything outside the standard
library), and app/tests/test_sig_client_parity.py asserts the two copies stay
byte-identical, so edit BOTH or neither. That is why it keeps its own nested
walker instead of using _walk below: the client cannot carry module-level
helpers. Everything else in this file is server-side only.
"""

import base64

# Header field numbers we read. The rest are recorded by number in `fields`
# so a format change shows up in the data instead of silently disappearing.
_F_MODEL = 6      # served model id, utf-8. Absent from v4 headers.
_F_KIND = 8       # "thinking" | "narration"
_F_TAG = 10       # experiment tag, e.g. MYCRO_MODEL_MANATEE

MAX_MODEL_LEN = 64
MAX_FIELDS_LEN = 128


def split_signature(b64):
    """Split a thinking-block signature blob into (version, header_b64, cipher_len).

    The blob is protobuf: top level f1 = varint format version (absent means 0),
    f2 = envelope; the envelope's f1 is the plaintext header and f5 is the
    ciphertext. Returns the header re-encoded as standard base64, or None when
    the blob has no envelope/header.

    Deliberately tolerant: the header format changed four times in six weeks,
    so an unreadable signature must degrade to (0, None, 0) rather than raise.
    Losing an event because its signature is a shape we have not seen would be
    far worse than losing the signature.

    Kept self-contained (nested helpers, no module-level dependencies beyond
    base64) because an identical copy lives in the server's app/sigheader.py
    and a test asserts the two sources match byte for byte.
    """
    try:
        def read_varint(buf, i):
            """Return (value, index just past the varint)."""
            value = shift = 0
            while True:
                byte = buf[i]
                i += 1
                value |= (byte & 0x7F) << shift
                if not byte & 0x80:
                    return value, i
                shift += 7
                if shift > 63:
                    raise ValueError("varint too long")

        def walk(buf):
            """Yield (field_number, value) for one protobuf message.

            Varints come back as int and length-delimited fields as bytes;
            fixed-width fields are skipped since no field we want uses them.
            """
            i, end = 0, len(buf)
            while i < end:
                tag, i = read_varint(buf, i)
                field, wire = tag >> 3, tag & 7
                if field == 0:
                    raise ValueError("field number 0 is not valid protobuf")
                if wire == 0:
                    value, i = read_varint(buf, i)
                    yield field, value
                elif wire == 2:
                    size, i = read_varint(buf, i)
                    stop = i + size
                    if stop > end:
                        raise ValueError("length-delimited field overruns buffer")
                    yield field, buf[i:stop]
                    i = stop
                elif wire == 5:
                    i += 4
                elif wire == 1:
                    i += 8
                else:
                    raise ValueError("unsupported wire type")
                if i > end:
                    raise ValueError("field overruns buffer")

        # Transcripts store the blob unpadded often enough to matter.
        raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
        version, envelope = 0, None
        for field, value in walk(raw):
            if field == 1 and isinstance(value, int):
                version = value
            elif field == 2 and isinstance(value, bytes):
                envelope = value
        if envelope is None:
            return version, None, 0
        header, cipher_len = None, 0
        for field, value in walk(envelope):
            if field == 1 and isinstance(value, bytes):
                header = value
            elif field == 5 and isinstance(value, bytes):
                cipher_len = len(value)
        if header is None:
            return version, None, 0
        return version, base64.b64encode(header).decode("ascii"), cipher_len
    except Exception:
        return 0, None, 0


def _walk(buf):
    """Yield (field_number, wire_type, payload) from a protobuf message.

    Stops at the first byte that does not parse rather than raising, so a
    truncated or reshaped header still surrenders the fields it did carry.
    Server-side twin of the walker inlined in split_signature.
    """
    i, n = 0, len(buf)
    while i < n:
        tag, i = _varint(buf, i, n)
        if tag is None:
            return
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(buf, i, n)
            if val is None:
                return
            yield (field, wire, val)
        elif wire == 2:
            ln, i = _varint(buf, i, n)
            if ln is None or i + ln > n:
                return
            yield (field, wire, buf[i:i + ln])
            i += ln
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            if i + width > n:
                return
            yield (field, wire, buf[i:i + width])
            i += width
        else:
            return  # groups (3/4) and unknown wire types: stop here


def _varint(buf, i, n):
    """Read one varint at `buf[i]`. Returns (value, next_index), or
    (None, i) when the varint is truncated or absurdly long."""
    val = 0
    shift = 0
    while True:
        if i >= n or shift > 63:
            return (None, i)
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return (val, i)
        shift += 7


def _text(payload, limit):
    """Decode a length-delimited field as bounded utf-8 text, or None."""
    if not isinstance(payload, (bytes, bytearray)):
        return None
    try:
        return payload.decode("utf-8")[:limit] or None
    except UnicodeDecodeError:
        return None


def decode_header(header_b64):
    """Decode a base64 header into {served_model, fields, kind, tag}.

    `fields` is the comma-joined, numerically sorted set of field numbers the
    header carries ("1,3,5,6,7,8,11"), the cheapest fingerprint of the header
    shape, so a format change is visible in the data. Unreadable input yields
    all-None with an empty `fields`; never raises.
    """
    blank = {"served_model": None, "fields": "", "kind": None, "tag": None}
    if not isinstance(header_b64, str) or not header_b64:
        return blank
    try:
        raw = base64.b64decode(header_b64, validate=False)
    except Exception:
        return blank

    seen = set()
    served_model = None
    kind = None
    tag = None
    for field, wire, payload in _walk(raw):
        seen.add(field)
        if wire != 2:
            continue
        if field == _F_MODEL:
            served_model = _text(payload, MAX_MODEL_LEN)
        elif field == _F_KIND:
            kind = _text(payload, MAX_MODEL_LEN)
        elif field == _F_TAG:
            tag = _text(payload, MAX_MODEL_LEN)
    return {
        "served_model": served_model,
        "fields": ",".join(str(f) for f in sorted(seen))[:MAX_FIELDS_LEN],
        "kind": kind,
        "tag": tag,
    }
