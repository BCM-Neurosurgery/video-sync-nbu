class AudioGroupDiscoverError(Exception):
    """Raised when audio group discovery fails."""

    pass


class TargetBuildError(Exception):
    """Raised when target building fails."""

    pass


class AudioDecodingError(Exception):
    """Raised when audio decoding fails."""

    pass


class SyncError(Exception):
    """Raised when syncing fails."""

    pass


class SerialAnalysisError(Exception):
    """Raised when serial analysis fails."""

    pass


class GapFillError(Exception):
    """Raised when gap filling fails."""

    pass


class FilteredError(Exception):
    """Raised when filtering fails."""

    pass


class VideoDiscoverError(Exception):
    """Raised when video discovery fails."""

    pass


class VideoAnalysisError(Exception):
    """Raised when video analysis fails."""

    pass


class VideoFrameIDAnalysisError(Exception):
    """Raised when video frame ID analysis fails."""

    pass


class AnchorError(Exception):
    """Raised when anchor-related errors occur."""

    pass


class ClipError(Exception):
    """Raised when clipping fails."""

    pass


class AudioPaddingError(Exception):
    """Raised when audio padding fails."""

    pass


class AudioPlanError(Exception):
    """Raised when audio plan application fails."""

    pass


class VideoPaddingError(Exception):
    """Raised when video padding fails."""

    pass


class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg binary is not found."""


class SplitFailureError(RuntimeError):
    """Raised when ffmpeg completes without producing expected output."""
