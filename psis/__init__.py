from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("psis")
except PackageNotFoundError:
    # package is not installed
    pass
