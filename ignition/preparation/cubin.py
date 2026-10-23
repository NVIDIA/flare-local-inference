"""Native ELF name redaction preserving CUDA section and symbol indices."""


def redact_symbols(image):
    from flare_heliopause import cubin_redact_symbols

    return cubin_redact_symbols(image)
