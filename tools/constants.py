"""
Constants and configuration values for the tools package.

This module centralizes magic numbers, thresholds, and configuration values
used throughout the retrieval tools to improve maintainability and clarity.
"""

from typing import Final

# RRF (Reciprocal Rank Fusion) Constants
RRF_K_VALUE: Final[int] = 60
"""RRF smoothing constant (typically 60)"""

RRF_DEFAULT_RANK_WINDOW: Final[int] = 100
"""Default window size for RRF calculation"""

RRF_DEFAULT_RERANK_WINDOW_MIN: Final[int] = 10
"""Minimum number of candidates to rerank"""

# Scoring Ranges
CROSS_ENCODER_MAX_SCORE: Final[float] = 90.0
"""Maximum score from cross-encoder (0-90 range)"""

RECENCY_MAX_BOOST: Final[float] = 10.0
"""Maximum recency boost score (0-10 range)"""

TOTAL_MAX_SCORE: Final[float] = 100.0
"""Maximum total score (cross-encoder + recency)"""

RRF_NORMALIZATION_MAX: Final[float] = 100.0
"""Maximum value for RRF score normalization"""

# Recency Calculation
RECENCY_DECAY_CONSTANT: Final[int] = 3
"""Decay constant for exponential recency calculation (smaller = steeper)"""

# Score Normalization
MIN_SCORE_RANGE: Final[float] = 1e-6
"""Minimum score range to avoid division by zero"""

# Danish Medical Text Processing
DANISH_STOPWORDS: Final[set] = {
    'og', 'i', 'jeg', 'det', 'at', 'en', 'den', 'til', 'er', 'som',
    'på', 'de', 'med', 'han', 'af', 'for', 'ikke', 'der', 'var', 'mig',
    'sig', 'men', 'et', 'har', 'om', 'vi', 'min', 'havde', 'ham', 'hun',
    'nu', 'over', 'da', 'fra', 'du', 'ud', 'sin', 'dem', 'os', 'op',
    'man', 'hans', 'hvor', 'eller', 'hvad', 'skal', 'selv', 'her',
    'alle', 'vil', 'blev', 'kunne', 'ind', 'når', 'være', 'dog', 'noget',
    'havde', 'mod', 'disse', 'hvis', 'din', 'nogle', 'hos', 'blive',
    'mange', 'ad', 'bliver', 'hendes', 'været', 'thi', 'jer', 'sådan'
}
"""Danish stopwords for keyword search filtering"""

MEDICAL_TERM_MIN_LENGTH: Final[int] = 4
"""Minimum length to keep a token even if it's a stopword"""

# Search Result Formatting
DEFAULT_MAX_REFERENCES: Final[int] = 3
"""Default maximum number of source references to display"""

SNIPPET_MAX_LENGTH: Final[int] = 150
"""Maximum length for content snippets in references"""
