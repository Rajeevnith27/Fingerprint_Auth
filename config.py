import os

# --- Application and File Settings ---
UPLOAD_FOLDER = 'uploads' # Directory to temporarily store uploaded images


# --- Fuzzy Extractor and Cryptographic Settings ---
# These parameters directly affect the security and error tolerance of the fuzzy extractor.
# Adjusting RS_ECC_SYMBOLS is critical for managing noise/variations in fingerprint features.
KEY_LENGTH = 16          # Length of the cryptographic key in bytes (e.g., 16 for AES-128)
SEED_LENGTH = 32         # Length of the random seed in bytes
RS_ECC_SYMBOLS = 60      # Number of Reed-Solomon error correction symbols.
                         # Higher values allow more errors to be corrected, improving robustness
                         # but slightly increasing helper data size. Start with 40-60.


# --- Fingerprint Feature Extraction Parameters ---
# These parameters are used in skimage.feature.local_binary_pattern
RADIUS = 1               # Radius for LBP. Defines the circular neighborhood.
N_POINTS = 8 * RADIUS    # Number of sampling points on the circle. (Usually 8 * RADIUS)


# --- Authentication and Quality Thresholds ---
# These thresholds define what is considered a "valid" fingerprint for enrollment/authentication
# and what constitutes a "match". These are crucial for system performance (FAR/FRR).

MATCH_THRESHOLD = 0.88   # Cosine similarity threshold for a potential match (0.0 to 1.0)
                         # A value like 0.88-0.95 is typical for good matches after alignment.

REQUIRE_CRYPTO_FOR_100_PERCENT = True # If True, even 100% similarity requires cryptographic verification.
                                      # If False, 100% similarity bypasses the cryptographic check.
                                      # Set to True for higher security; False for potential speedup in perfect cases.


FINGERPRINT_THRESHOLDS = {
    # Overall score for fingerprint validity (mean of individual metric scores)
    'overall_score': 0.70, # Adjusted: Combined quality score (0.0 to 1.0).

    # Individual metric thresholds for fingerprint quality detection
    'ridge_density': 0.25, # Adjusted: Proportion of image covered by ridges (0.0 to 1.0).
                           # Too low might be blank, too high might be smudged.

    'orientation_consistency': 0.65, # Adjusted: How consistent the ridge flow direction is (0.0 to 1.0).
                                    # Higher means clearer, more parallel ridges.

    'minutiae_count': 25,          # Adjusted: Minimum number of detected minutiae points.
                                    # Good prints typically have 30-80 minutiae.

    'contrast_ratio': 0.90,        # Adjusted: Ratio of standard deviation to mean pixel intensity.
                                    # Indicates clear separation between ridges and valleys.

    'frequency_score': 0.20,       # Adjusted: Energy in the typical fingerprint ridge frequency band (0.0 to 1.0).
                                    # Higher means clearer, more regular ridge spacing.

    'ridge_regularity': 0.75,      # Adjusted: How orderly and predictable the ridge patterns are (0.0 to 1.0).
                                    # Higher means less chaotic or noisy patterns.

    # LBP entropy score (not directly thresholded, but contributes to overall_score)
    # 'lbp_entropy': (no direct threshold, as it's balanced around an ideal value)

    'ridge_flow_continuity': 0.45 # NEW/Adjusted: Measures how continuous and unbroken ridge lines are (0.0 to 1.0).
                                  # Higher means less fragmentation or noise.
}
