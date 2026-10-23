"""Preparation-scoped native CUPTI module and allocation recording."""


def cupti_library(library=None):
    if library is None:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        library = load_nvidia_dynamic_lib("cupti").abs_path
    return library


def strip_debug(image):
    from flare_heliopause import cubin_strip_debug

    return cubin_strip_debug(image)


def strip_global_source(symbol, data):
    from flare_heliopause import cubin_strip_global_source

    return cubin_strip_global_source(symbol, data)


def global_symbols(image):
    from flare_heliopause import cubin_global_symbols

    return cubin_global_symbols(image)


class ModuleImages:
    """Keep native module images and driver allocation records through export."""

    active = 0

    def __init__(self, library=None):
        from flare_heliopause import NativeModuleRecorder

        self.native = NativeModuleRecorder(str(cupti_library(library)))
        self._active = False

    @property
    def allocations(self):
        return self.native.allocations()

    @property
    def allocation_contexts(self):
        return self.native.allocation_contexts()

    def __enter__(self):
        self.native.start()
        self._active = True
        ModuleImages.active += 1
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.native.stop()
        finally:
            if self._active:
                ModuleImages.active -= 1
                self._active = False
