import numpy as np
import cv2
import hashlib
from skimage.feature import local_binary_pattern
from reedsolo import RSCodec, ReedSolomonError
import os
import uuid

# Import parameters from config
from config import RADIUS, N_POINTS, SEED_LENGTH, KEY_LENGTH, RS_ECC_SYMBOLS, FINGERPRINT_THRESHOLDS

# Initialize Reed-Solomon Codec
rsc = RSCodec(RS_ECC_SYMBOLS)


def detect_ridge_patterns(img):
    """Detects advanced ridge patterns (density, orientations, frequency)."""
    # Use adaptive thresholding for better binarization irrespective of lighting
    block_size = 35 # Must be odd
    C_param = 15 # Constant subtracted from the mean
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C_param)

    # Invert binary image if ridges are dark (common for fingerprint images after some preprocessing)
    # The adaptive threshold might make ridges white, we often want them black for morphological ops.
    if np.mean(binary_img) > 127: # If most of it is white (ridges are white)
        binary_img = cv2.bitwise_not(binary_img) # Invert to make ridges black

    # Apply morphological operations to enhance ridge structure and remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img_morphed = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_img_morphed = cv2.morphologyEx(binary_img_morphed, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Calculate gradients for orientation and magnitude
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude and orientation from the original (or enhanced) grayscale image
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientations = np.arctan2(grad_y, grad_x + 1e-8) # Add epsilon to avoid division by zero

    # Ridge density calculation based on processed binary image
    # Use sum of white pixels (ridges) and normalize
    ridge_density = np.sum(binary_img_morphed == 0) / binary_img_morphed.size # Assuming black ridges
    
    # Analyze frequency spectrum (e.g., for consistent ridge spacing)
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)

    return ridge_density, orientations, magnitude_spectrum, binary_img_morphed, magnitude


def calculate_orientation_consistency(orientations, magnitude):
    """Calculates consistency of ridge orientations, weighted by gradient magnitude."""
    h, w = orientations.shape
    block_size = 16
    consistencies = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_orient = orientations[y:y+block_size, x:x+block_size]
            block_magn = magnitude[y:y+block_size, x:x+block_size]

            # Filter out low-magnitude areas (noise)
            strong_indices = block_magn > np.mean(block_magn) * 0.5 # Tune threshold
            if np.sum(strong_indices) < block_size * block_size * 0.1: # Require at least 10% strong pixels
                continue

            # Compute circular mean for the block (angles doubled for 180-deg periodicity)
            valid_orient = block_orient[strong_indices]
            if len(valid_orient) > 0:
                mean_sin_2theta = np.mean(np.sin(2 * valid_orient))
                mean_cos_2theta = np.mean(np.cos(2 * valid_orient))
                # Consistency is the magnitude of the mean vector
                consistency = np.sqrt(mean_sin_2theta**2 + mean_cos_2theta**2)
                consistencies.append(consistency)
    
    return np.mean(consistencies) if consistencies else 0.0


def calculate_ridge_flow_continuity(binary_img):
    """Calculates continuity of ridge flow using connected components or skeletonization."""
    # Skeletonize the binary image to get thin ridges
    skeleton = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    
    # Find connected components in the skeleton
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, 8, cv2.CV_32S)

    # Analyze properties of components (e.g., length, straightness)
    # A simple metric: average length of significant components
    total_ridge_pixels = 0
    num_significant_ridges = 0
    
    for i in range(1, num_labels): # Skip background label 0
        component_pixels = np.sum(labels == i)
        if component_pixels > 20: # Only consider components above a certain pixel count
            total_ridge_pixels += component_pixels
            num_significant_ridges += 1
            
    if num_significant_ridges > 0:
        avg_ridge_length = total_ridge_pixels / num_significant_ridges
        # Normalize to a score (example normalization, needs tuning)
        # Assuming typical ridge lengths around 50-100 pixels in a 256x256 image
        return min(avg_ridge_length / 70.0, 1.0) # Tune 70.0 based on observed good quality avg length
    return 0.0


def detect_minutiae_points(img):
    """Detects minutiae points for fingerprint validation."""
    # Use more robust binarization for minutiae detection as well
    block_size = 35 # Must be odd
    C_param = 15 # Constant subtracted from the mean
    binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C_param)
    if np.mean(binary_img) > 127:
        binary_img = cv2.bitwise_not(binary_img)

    # Skeletonize for better minutiae representation
    skeleton = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_GUOHALL)

    # Minutiae can be found by analyzing local neighborhoods in the skeleton
    # This is a very simplified minutiae count. True minutiae extraction is complex.
    # For demo, we stick to goodFeaturesToTrack on the skeleton
    corners = cv2.goodFeaturesToTrack(
        skeleton, maxCorners=250, qualityLevel=0.01, minDistance=5, blockSize=3 # Adjusted qualityLevel
    )
    minutiae_count = len(corners) if corners is not None else 0
    return minutiae_count, corners


def analyze_frequency_domain(img):
    """Analyzes frequency characteristics typical of fingerprints (ridge frequency)."""
    f_transform = np.fft.fft2(img)
    magnitude_spectrum = np.abs(np.fft.fftshift(f_transform))
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Define a radial band for typical fingerprint ridge frequencies
    # These radii correspond to pixel distances from center in frequency domain
    # and relate to spatial frequencies (ridges per pixel)
    inner_radius = 5  # Corresponds to lower spatial frequencies (wider ridges)
    outer_radius = 30 # Corresponds to higher spatial frequencies (narrower ridges)

    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Sum energy within the target frequency band
    fingerprint_freq_energy = np.sum(magnitude_spectrum[(distances >= inner_radius) & (distances < outer_radius)])
    total_energy = np.sum(magnitude_spectrum)
    
    return fingerprint_freq_energy / total_energy if total_energy > 0 else 0.0


def comprehensive_fingerprint_detection(image_path):
    """Performs comprehensive fingerprint detection using multiple criteria.
    Returns: (is_fingerprint, confidence_score, detailed_analysis)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return False, 0.0, {"error": "Invalid image file or file not found"}

        img = cv2.resize(img, (256, 256))
        
        # --- Improved image inversion check ---
        # Determine if image needs inversion (light ridges on dark background vs dark ridges on light background)
        # Most fingerprint algorithms expect dark ridges on light background or vice-versa.
        # Let's try to standardize to dark ridges on light background (inverted if needed)
        # Find dominant color; if average is high, it's likely light background with dark ridges (good)
        # If average is low, it's likely dark background with light ridges, so invert to match expectation.
        if np.mean(img) < 100: # If average pixel value is low (dark overall background)
             img = cv2.bitwise_not(img) # Invert to make background lighter, ridges darker.


        analysis = {}
        scores = []

        # Ridge Pattern Analysis (now returns more values)
        ridge_density, orientations, magnitude_spectrum, binary_img_morphed, magnitude = detect_ridge_patterns(img)
        analysis['ridge_density'] = ridge_density
        scores.append(min(ridge_density / FINGERPRINT_THRESHOLDS['ridge_density'], 1.0) if FINGERPRINT_THRESHOLDS['ridge_density'] > 0 else 0)

        # Orientation Consistency (now uses magnitude for weighting)
        orientation_consistency = calculate_orientation_consistency(orientations, magnitude)
        analysis['orientation_consistency'] = orientation_consistency
        scores.append(min(orientation_consistency / FINGERPRINT_THRESHOLDS['orientation_consistency'], 1.0) if FINGERPRINT_THRESHOLDS['orientation_consistency'] > 0 else 0)

        # Minutiae Detection (now uses skeletonized image for better results)
        minutiae_count, _ = detect_minutiae_points(img) # Pass original image for adaptive thresholding
        analysis['minutiae_count'] = minutiae_count
        scores.append(min(minutiae_count / FINGERPRINT_THRESHOLDS['minutiae_count'], 1.0) if FINGERPRINT_THRESHOLDS['minutiae_count'] > 0 else 0)

        # Contrast Analysis
        contrast_ratio = np.std(img) / (np.mean(img) + 1e-8)
        analysis['contrast_ratio'] = contrast_ratio
        scores.append(min(contrast_ratio / FINGERPRINT_THRESHOLDS['contrast_ratio'], 1.0) if FINGERPRINT_THRESHOLDS['contrast_ratio'] > 0 else 0)

        # Frequency Domain Analysis (improved)
        frequency_score = analyze_frequency_domain(img)
        analysis['frequency_score'] = frequency_score
        scores.append(min(frequency_score / FINGERPRINT_THRESHOLDS['frequency_score'], 1.0) if FINGERPRINT_THRESHOLDS['frequency_score'] > 0 else 0)

        # Edge Regularity (still useful as is)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_angles = np.arctan2(grad_y, grad_x)
        hist, _ = np.histogram(edge_angles, bins=36, range=(-np.pi, np.pi))
        hist_normalized = hist / (np.sum(hist) + 1e-8)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
        regularity = 1.0 - (entropy / np.log2(36)) if np.log2(36) > 0 else 0.0
        analysis['ridge_regularity'] = regularity
        scores.append(min(regularity / FINGERPRINT_THRESHOLDS['ridge_regularity'], 1.0) if FINGERPRINT_THRESHOLDS['ridge_regularity'] > 0 else 0)

        # Texture Analysis - Local Binary Pattern entropy
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
        analysis['lbp_entropy'] = lbp_entropy
        ideal_lbp_entropy = 3.0 # This value might need tuning
        entropy_score = max(0.0, 1.0 - abs(lbp_entropy - ideal_lbp_entropy) / ideal_lbp_entropy)
        scores.append(entropy_score)

        # NEW: Ridge Flow Continuity Check
        # Ensure ximgproc is available. If not, this will cause an error or need a fallback.
        # You might need to install opencv-contrib-python if not already installed (`pip install opencv-contrib-python`)
        try:
            ridge_flow_continuity = calculate_ridge_flow_continuity(binary_img_morphed) # Use the processed binary image
            analysis['ridge_flow_continuity'] = ridge_flow_continuity
            scores.append(min(ridge_flow_continuity / FINGERPRINT_THRESHOLDS['ridge_flow_continuity'], 1.0) if FINGERPRINT_THRESHOLDS['ridge_flow_continuity'] > 0 else 0)
        except AttributeError:
            print("Warning: cv2.ximgproc not found. Ridge flow continuity will be 0.")
            analysis['ridge_flow_continuity'] = 0.0
            scores.append(0.0) # Add a score of 0 if ximgproc is not available


        overall_score = np.mean(scores) if scores else 0.0
        analysis['overall_score'] = overall_score
        analysis['individual_scores'] = {
            'ridge_density_score': scores[0] if len(scores) > 0 else 0,
            'orientation_score': scores[1] if len(scores) > 1 else 0,
            'minutiae_score': scores[2] if len(scores) > 2 else 0,
            'contrast_score': scores[3] if len(scores) > 3 else 0,
            'frequency_score': scores[4] if len(scores) > 4 else 0,
            'regularity_score': scores[5] if len(scores) > 5 else 0,
            'lbp_entropy_score': scores[6] if len(scores) > 6 else 0,
            'ridge_flow_continuity_score': scores[7] if len(scores) > 7 else 0 # New score index
        }

        is_fingerprint = (
            overall_score >= FINGERPRINT_THRESHOLDS['overall_score'] and
            ridge_density >= FINGERPRINT_THRESHOLDS['ridge_density'] and
            minutiae_count >= FINGERPRINT_THRESHOLDS['minutiae_count'] and
            orientation_consistency >= FINGERPRINT_THRESHOLDS['orientation_consistency'] and
            # Add new threshold check
            (ridge_flow_continuity >= FINGERPRINT_THRESHOLDS['ridge_flow_continuity'] if 'ridge_flow_continuity' in FINGERPRINT_THRESHOLDS else True)
        )
        return is_fingerprint, overall_score, analysis

    except Exception as e:
        print(f"Error during comprehensive fingerprint detection: {e}")
        return False, 0.0, {"error": f"Detection failed: {str(e)}"}


def extract_enhanced_features(image_path, target_len=None):
    """Extracts enhanced features from a validated fingerprint image, with improved rotation alignment."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file or file not found at path: " + image_path)

    img = cv2.resize(img, (256, 256))
    img = cv2.equalizeHist(img) # Still using equalizeHist first
    
    # Re-apply the robust inversion check from comprehensive_fingerprint_detection
    if np.mean(img) < 100: # If average pixel value is low (dark overall background)
         img = cv2.bitwise_not(img) # Invert to make background lighter, ridges darker.


    # --- IMPROVED ORIENTATION ESTIMATION ---
    h, w = img.shape
    block_size = 16 # Analyze orientation in 16x16 pixel blocks
    orientations_rad = [] # Store orientations in radians

    # Calculate gradients for the entire image
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2) # Calculate magnitude for filtering

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block_gx = grad_x[y:y+block_size, x:x+block_size]
            block_gy = grad_y[y:y+block_size, x:x+block_size]
            block_magn = magnitude[y:y+block_size, x:x+block_size]

            # Compute sum of 2*Gx*Gy and sum of (Gx^2 - Gy^2) over the block
            # These are components used for averaging circular quantities (angles)
            Vx = np.sum(2 * block_gx * block_gy)
            Vy = np.sum(block_gx**2 - block_gy**2)

            # Filter out weak gradient blocks that might have noisy orientations
            # Require a certain average magnitude in the block
            if np.mean(block_magn) > (np.mean(magnitude) * 0.2): # Tune threshold (e.g., 20% of global mean magnitude)
                angle = 0.5 * np.arctan2(Vx, Vy + 1e-8) # This maps to -pi/2 to pi/2 for ridge direction
                orientations_rad.append(angle)

    dominant_orientation_rad = 0.0
    if orientations_rad:
        # Use circular mean to get a robust average orientation
        mean_sin_2theta = np.mean(np.sin(2 * np.array(orientations_rad)))
        mean_cos_2theta = np.mean(np.cos(2 * np.array(orientations_rad)))

        # Convert back to angle in radians (0 to pi range for ridge orientation)
        dominant_orientation_rad = 0.5 * np.arctan2(mean_sin_2theta, mean_cos_2theta)

    # Convert to degrees (0 to 180 range for ridge orientation)
    dominant_orientation_deg = np.degrees(dominant_orientation_rad) % 180

    # Calculate rotation angle needed to align to a target orientation (e.g., 90 degrees for vertical ridges)
    target_orientation_deg = 90 # Common target for fingerprint ridges to be vertical
    rotation_angle = dominant_orientation_deg - target_orientation_deg
    
    # Perform the rotation
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    img_aligned = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # --- DEBUGGING CODE: SAVE THE ALIGNED IMAGE ---
    debug_output_dir = 'debug_aligned_images'
    if not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)
    
    unique_filename = f"aligned_fingerprint_{uuid.uuid4().hex}.png"
    full_debug_path = os.path.join(debug_output_dir, unique_filename)
    cv2.imwrite(full_debug_path, img_aligned)
    print(f"DEBUG: Saved aligned image to {full_debug_path}")
    # --- END DEBUGGING CODE ---

    features = []

    # All feature extraction now uses the ALIGNED image: img_aligned
    # Enhanced LBP features
    lbp = local_binary_pattern(img_aligned, N_POINTS, RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=N_POINTS + 2, range=(0, N_POINTS + 2))
    features.extend(lbp_hist.astype(np.float32))

    # Gabor filter responses
    for angle in np.arange(0, 180, 22.5):
        kernel = cv2.getGaborKernel((21, 21), 4.0, np.radians(angle), 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_aligned, cv2.CV_8UC3, kernel)
        features.extend([np.mean(filtered), np.std(filtered)])

    # Gradient features
    grad_x_feat = cv2.Sobel(img_aligned, cv2.CV_64F, 1, 0, ksize=3)
    grad_y_feat = cv2.Sobel(img_aligned, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x_feat**2 + grad_y_feat**2)
    grad_orientation = np.arctan2(grad_y_feat, grad_x_feat)
    features.extend([np.mean(grad_magnitude), np.std(grad_magnitude)])

    # Orientation Histogram for overall orientation distribution
    orientation_hist, _ = np.histogram(grad_orientation, bins=36, range=(-np.pi, np.pi))
    features.extend(orientation_hist.astype(np.float32))

    # Zernike moments
    moments = cv2.moments(img_aligned.astype(np.uint8))
    if moments['m00'] != 0:
        hu_moments = cv2.HuMoments(moments).flatten()
        features.extend([-np.sign(h) * np.log10(abs(h) + 1e-8) for h in hu_moments])
    else:
        features.extend([0.0] * 7)

    feature_bytes = np.array(features, dtype=np.float32).tobytes()

    if target_len:
        if len(feature_bytes) < target_len:
            feature_bytes += bytes(target_len - len(feature_bytes))
        elif len(feature_bytes) > target_len:
            feature_bytes = feature_bytes[:target_len]
    return feature_bytes


def calculate_feature_similarity(features1_bytes, features2_bytes):
    """Calculates cosine similarity between two feature vectors."""
    try:
        features1 = np.frombuffer(features1_bytes, dtype=np.float32)
        features2 = np.frombuffer(features2_bytes, dtype=np.float32)

        min_len = min(len(features1), len(features2))
        if min_len == 0: return 0.0

        features1 = features1[:min_len]
        features2 = features2[:min_len]

        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0: return 0.0

        similarity = np.dot(features1 / norm1, features2 / norm2)
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0


def strong_extractor(K_bytes, seed_bytes):
    """Strong extractor using SHA-256 (HMAC-style)."""
    return hashlib.sha256(K_bytes + seed_bytes).digest()

