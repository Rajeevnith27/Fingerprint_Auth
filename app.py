from flask import Flask, request, render_template_string, jsonify
import numpy as np
import cv2
import os
import hashlib
import uuid
from skimage.feature import local_binary_pattern
from reedsolo import RSCodec, ReedSolomonError
from werkzeug.utils import secure_filename
import json

# Initialize Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Parameters for feature extraction and authentication
RADIUS = 3
N_POINTS = 24
SEED_LENGTH = 16
KEY_LENGTH = 32
RS_ECC_SYMBOLS = 8 # Number of error correction bytes - Increased for better robustness
rsc = RSCodec(RS_ECC_SYMBOLS)
MATCH_THRESHOLD = 0.75 # Similarity threshold for potential match
MAX_FAILED_ATTEMPTS = 5 # Max failed attempts per fingerprint_name before lockout

# Determined feature length (62 floats * 4 bytes/float = 248 bytes, rounded up to 256 for consistency)
# This ensures all extracted feature vectors have a uniform size.
TARGET_FEATURE_LENGTH = 256


# Relaxed Fingerprint Detection Thresholds (NOT recommended for production)
FINGERPRINT_THRESHOLDS = {
    'ridge_density': 0.05,
    'ridge_regularity': 0.1,
    'orientation_consistency': 0.1,
    'minutiae_count': 3,
    'contrast_ratio': 0.05,
    'frequency_score': 0.1,
    'overall_score': 0.15
}


class BiometricDatabase:
    """Simulates a database for storing fingerprints and authentication logs."""
    def __init__(self):
        self.fingerprints = {} # Stores enrolled fingerprint templates, keyed by fingerprint_name
        self.auth_logs = []    # Stores authentication attempt logs
        self.failed_attempts = {} # Tracks failed attempts per fingerprint_name

    def is_locked(self, fingerprint_name):
        """Checks if a fingerprint name is locked."""
        return self.failed_attempts.get(fingerprint_name, 0) >= MAX_FAILED_ATTEMPTS

    def record_failed_attempt(self, fingerprint_name):
        """Increments failed attempt count for a fingerprint name."""
        self.failed_attempts[fingerprint_name] = self.failed_attempts.get(fingerprint_name, 0) + 1

    def reset_failed_attempts(self, fingerprint_name):
        """Resets failed attempt count for a fingerprint name."""
        if fingerprint_name in self.failed_attempts:
            del self.failed_attempts[fingerprint_name]

    def add_auth_log(self, fingerprint_name, success, confidence_score, quality_score, details):
        """Adds an entry to the authentication log."""
        log_entry = {
            'fingerprint_name': fingerprint_name,
            'success': success,
            'confidence_score': confidence_score,
            'quality_score': quality_score,
            'match_details': details
        }
        self.auth_logs.append(log_entry)

db = BiometricDatabase()


def detect_ridge_patterns(img):
    """Detects advanced ridge patterns (density, orientations, frequency)."""
    img_norm = cv2.equalizeHist(img)
    grad_x = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
    orientations = np.arctan2(grad_y, grad_x)
    f_transform = np.fft.fft2(img_norm)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    _, binary_img = cv2.threshold(img_norm, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(binary_img, 50, 150)
    ridge_density = np.sum(edges > 0) / edges.size
    return ridge_density, orientations, magnitude_spectrum


def calculate_orientation_consistency(orientations):
    """Calculates consistency of ridge orientations."""
    h, w = orientations.shape
    block_size = 16
    consistencies = []
    for i in range(0, h - block_size, block_size // 2):
        for j in range(0, w - block_size, block_size // 2):
            block = orientations[i:i+block_size, j:j+block_size]
            mean_sin = np.mean(np.sin(2 * block))
            mean_cos = np.mean(np.cos(2 * block))
            consistency = np.sqrt(mean_sin**2 + mean_cos**2)
            consistencies.append(consistency)
    return np.mean(consistencies) if consistencies else 0.0


def detect_minutiae_points(img):
    """Detects minutiae points for fingerprint validation."""
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
    _, binary = cv2.threshold(img_morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = cv2.goodFeaturesToTrack(
        binary, maxCorners=200, qualityLevel=0.005, minDistance=5, blockSize=3
    )
    minutiae_count = len(corners) if corners is not None else 0
    return minutiae_count, corners


def analyze_frequency_domain(img):
    """Analyzes frequency characteristics typical of fingerprints."""
    f_transform = np.fft.fft2(img)
    magnitude_spectrum = np.abs(np.fft.fftshift(f_transform))
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    low_freq = np.sum(magnitude_spectrum[distances < 10])
    mid_freq = np.sum(magnitude_spectrum[(distances >= 10) & (distances < 30)])
    high_freq = np.sum(magnitude_spectrum[distances >= 30])
    total_energy = low_freq + mid_freq + high_freq
    return mid_freq / total_energy if total_energy > 0 else 0.0


def comprehensive_fingerprint_detection(image_path):
    """Performs comprehensive fingerprint detection using multiple criteria.
    Returns: (is_fingerprint, confidence_score, detailed_analysis)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return False, 0.0, {"error": "Invalid image file"}

        img = cv2.resize(img, (256, 256))
        if np.mean(img) > 127: # Invert colors if necessary
            img = cv2.bitwise_not(img)

        analysis = {}
        scores = []

        # Ridge Pattern Analysis
        ridge_density, orientations, _ = detect_ridge_patterns(img)
        analysis['ridge_density'] = ridge_density
        scores.append(min(ridge_density / FINGERPRINT_THRESHOLDS['ridge_density'], 1.0))

        # Orientation Consistency
        orientation_consistency = calculate_orientation_consistency(orientations)
        analysis['orientation_consistency'] = orientation_consistency
        scores.append(min(orientation_consistency / FINGERPRINT_THRESHOLDS['orientation_consistency'], 1.0))

        # Minutiae Detection
        minutiae_count, _ = detect_minutiae_points(img)
        analysis['minutiae_count'] = minutiae_count
        scores.append(min(minutiae_count / FINGERPRINT_THRESHOLDS['minutiae_count'], 1.0))

        # Contrast Analysis
        contrast_ratio = np.std(img) / (np.mean(img) + 1e-8)
        analysis['contrast_ratio'] = contrast_ratio
        scores.append(min(contrast_ratio / FINGERPRINT_THRESHOLDS['contrast_ratio'], 1.0))

        # Frequency Domain Analysis
        frequency_score = analyze_frequency_domain(img)
        analysis['frequency_score'] = frequency_score
        scores.append(min(frequency_score / FINGERPRINT_THRESHOLDS['frequency_score'], 1.0))

        # Edge Regularity
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_angles = np.arctan2(grad_y, grad_x)
        hist, _ = np.histogram(edge_angles, bins=36, range=(-np.pi, np.pi))
        hist_normalized = hist / (np.sum(hist) + 1e-8)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
        regularity = 1.0 - (entropy / np.log2(36))
        analysis['ridge_regularity'] = regularity
        scores.append(min(regularity / FINGERPRINT_THRESHOLDS['ridge_regularity'], 1.0))

        # Texture Analysis - Local Binary Pattern entropy
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))
        analysis['lbp_entropy'] = lbp_entropy
        entropy_score = 1.0 - abs(lbp_entropy - 1.5) / 3.0
        scores.append(max(0.0, entropy_score))

        overall_score = np.mean(scores) if scores else 0.0
        analysis['overall_score'] = overall_score
        analysis['individual_scores'] = {
            'ridge_density_score': scores[0] if len(scores) > 0 else 0,
            'orientation_score': scores[1] if len(scores) > 1 else 0,
            'minutiae_score': scores[2] if len(scores) > 2 else 0,
            'contrast_score': scores[3] if len(scores) > 3 else 0,
            'frequency_score': scores[4] if len(scores) > 4 else 0,
            'regularity_score': scores[5] if len(scores) > 5 else 0,
            'entropy_score': scores[6] if len(scores) > 6 else 0
        }

        is_fingerprint = (
            overall_score >= FINGERPRINT_THRESHOLDS['overall_score'] and
            ridge_density >= FINGERPRINT_THRESHOLDS['ridge_density'] and
            minutiae_count >= FINGERPRINT_THRESHOLDS['minutiae_count']
        )
        return is_fingerprint, overall_score, analysis

    except Exception as e:
        return False, 0.0, {"error": f"Detection failed: {str(e)}"}


def extract_enhanced_features(image_path):
    """Extracts enhanced features from a validated fingerprint image and enforces TARGET_FEATURE_LENGTH."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (256, 256))
    img = cv2.equalizeHist(img)
    if np.mean(img) > 127: # Invert colors if needed
        img = cv2.bitwise_not(img)

    features = []

    # Enhanced LBP features
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=N_POINTS + 2, range=(0, N_POINTS + 2))
    features.extend(lbp_hist.astype(np.float32))

    # Gabor filter responses
    for angle in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
        kernel = cv2.getGaborKernel((21, 21), 5, np.radians(angle), 10, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        features.extend([np.mean(filtered), np.std(filtered)])

    # Gradient features
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    features.extend([np.mean(grad_magnitude), np.std(grad_magnitude)])

    # Ridge orientation features
    orientations = np.arctan2(grad_y, grad_x)
    orientation_hist, _ = np.histogram(orientations, bins=18, range=(-np.pi, np.pi))
    features.extend(orientation_hist.astype(np.float32))

    feature_bytes = np.array(features, dtype=np.float32).tobytes()

    # Pad or truncate to TARGET_FEATURE_LENGTH
    if len(feature_bytes) > TARGET_FEATURE_LENGTH:
        return feature_bytes[:TARGET_FEATURE_LENGTH]
    else:
        return feature_bytes + bytes(TARGET_FEATURE_LENGTH - len(feature_bytes))


def calculate_feature_similarity(features1_bytes, features2_bytes):
    """Calculates cosine similarity between two feature vectors."""
    try:
        features1 = np.frombuffer(features1_bytes, dtype=np.float32)
        features2 = np.frombuffer(features2_bytes, dtype=np.float32)

        if len(features1) != len(features2) or len(features1) == 0:
            return 0.0

        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0: return 0.0

        similarity = np.dot(features1 / norm1, features2 / norm2)
        return max(0.0, min(1.0, similarity)) # Clamp between 0 and 1

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0


def strong_extractor(K_bytes, seed_bytes):
    """Strong extractor using SHA-256 (HMAC-style)."""
    return hashlib.sha256(K_bytes + seed_bytes).digest()


# Web Routes
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with enrollment and authentication forms."""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fuzzy Extractor Fingerprint Authentication</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background-color: #eef2f6; /* Lighter, modern background */
            }
            .card {
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-4">
        <div class="bg-white rounded-2xl shadow-3xl p-6 sm:p-8 md:p-12 w-full max-w-4xl border border-gray-200">
            <h1 class="text-4xl font-extrabold text-center text-gray-900 mb-2 tracking-tight">
                🔐 Biometric Authentication
            </h1>
            <p class="text-center text-gray-500 mb-8 text-lg">Powered by a Fuzzy Extractor</p>
            
            <div class="mb-10 p-6 bg-blue-50 border border-blue-200 text-blue-800 rounded-xl shadow-md">
                <h3 class="text-2xl font-semibold mb-4 flex items-center text-blue-900">
                    <span class="mr-3">🛡️</span>Advanced Fingerprint Detection Criteria
                </h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-3 text-base sm:text-lg text-gray-700">
                    <div><strong class="font-medium">Ridge Density:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.ridge_density }}</span></div>
                    <div><strong class="font-medium">Orientation Consistency:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.orientation_consistency }}</span></div>
                    <div><strong class="font-medium">Minutiae Count:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.minutiae_count }}</span></div>
                    <div><strong class="font-medium">Contrast Ratio:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.contrast_ratio }}</span></div>
                    <div><strong class="font-medium">Frequency Score:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.frequency_score }}</span></div>
                    <div><strong class="font-medium">Overall Score:</strong> <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded-md text-sm font-mono">&ge; {{ thresholds.overall_score }}</span></div>
                </div>
                <p class="mt-6 text-gray-700 text-lg">
                    <strong class="text-blue-900">Enrolled Fingerprints:</strong> <span class="font-bold text-blue-600 bg-blue-100 px-2 py-1 rounded-md">{{ fp_count }}</span>
                </p>
                <p class="mt-4 text-orange-700 text-base p-3 bg-orange-50 rounded-md border border-orange-200">
                    <strong>Note:</strong> Detection thresholds are relaxed for this demo. <span class="font-bold">Do not use these settings for a production system.</span>
                </p>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="p-8 bg-gray-50 rounded-2xl shadow-lg border border-gray-200 card">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                        <span class="mr-3 text-blue-500 text-3xl">➕</span>Enroll Fingerprint
                    </h3>
                    <form action="/enroll" method="post" enctype="multipart/form-data" class="space-y-6">
                        <div>
                            <label for="enroll_name" class="block text-gray-700 text-sm font-medium mb-2">Fingerprint Name</label>
                            <input type="text" id="enroll_name" name="fingerprint_name" placeholder="e.g., Right_Thumb" required
                                class="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 shadow-sm transition duration-200 ease-in-out placeholder-gray-400">
                        </div>
                        <div>
                            <label for="enroll_bio" class="block text-gray-700 text-sm font-medium mb-2">Upload Fingerprint Image</label>
                            <input type="file" id="enroll_bio" name="bio" accept="image/*" required
                                class="w-full text-gray-700 text-sm file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer transition duration-200 ease-in-out">
                        </div>
                        <button type="submit"
                            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 px-4 rounded-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 hover:shadow-xl">
                            Enroll Fingerprint
                        </button>
                    </form>
                </div>
                
                <div class="p-8 bg-gray-50 rounded-2xl shadow-lg border border-gray-200 card">
                    <h3 class="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                        <span class="mr-3 text-green-500 text-3xl">🔑</span>Authenticate Fingerprint
                    </h3>
                    <form action="/authenticate" method="post" enctype="multipart/form-data" class="space-y-6">
                        <div>
                            <label for="auth_bio" class="block text-gray-700 text-sm font-medium mb-2">Upload Fingerprint Image for Authentication</label>
                            <input type="file" id="auth_bio" name="bio" accept="image/*" required
                                class="w-full text-gray-700 text-sm file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-green-50 file:text-green-700 hover:file:bg-green-100 cursor-pointer transition duration-200 ease-in-out">
                        </div>
                        <button type="submit"
                            class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3.5 px-4 rounded-lg shadow-lg transition duration-300 ease-in-out transform hover:scale-105 hover:shadow-xl">
                            Identify Fingerprint
                        </button>
                    </form>
                </div>
            </div>

            <div class="mt-10 p-6 bg-gray-100 rounded-2xl shadow-inner border border-gray-200 text-center">
                <h3 class="text-xl font-semibold text-gray-800 mb-4">📊 System Status</h3>
                <div class="flex flex-col sm:flex-row justify-center gap-4">
                    <a href="/logs" target="_blank"
                       class="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-blue-700 bg-blue-100 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition duration-150 ease-in-out shadow-sm">
                        View Authentication Logs
                    </a>
                    <a href="/fingerprints" target="_blank"
                       class="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-purple-700 bg-purple-100 hover:bg-purple-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition duration-150 ease-in-out shadow-sm">
                        View Enrolled Fingerprints
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''', thresholds=FINGERPRINT_THRESHOLDS, fp_count=len(db.fingerprints))


@app.route('/enroll', methods=['POST'])
def enroll():
    """Handles fingerprint enrollment."""
    fingerprint_name = request.form.get('fingerprint_name', '').strip()
    if not fingerprint_name: 
        return render_template_string('''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enrollment Failed</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body { font-family: 'Inter', sans-serif; background-color: #eef2f6; } </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">Enrollment Failed</h3>
                <p class="text-lg text-gray-700 mb-6">Fingerprint Name is required.</p>
                <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')
    if fingerprint_name in db.fingerprints:
        return render_template_string(f'''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enrollment Failed</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">Enrollment Failed</h3>
                <p class="text-lg text-gray-700 mb-6">A fingerprint with the name '<span class="font-mono bg-red-100 text-red-800 px-2 py-1 rounded-md">{fingerprint_name}</span>' already exists. Please choose a unique name.</p>
                <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')

    file = request.files.get('bio')
    if not file: 
        return render_template_string('''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enrollment Failed</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body { font-family: 'Inter', sans-serif; background-color: #eef2f6; } </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">Enrollment Failed</h3>
                <p class="text-lg text-gray-700 mb-6">No file uploaded. Please select a fingerprint image.</p>
                <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')

    # Use a unique but temporary filename
    temp_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_id}_{secure_filename(file.filename)}")
    file.save(filepath)

    try:
        is_fingerprint, fp_confidence, fp_analysis = comprehensive_fingerprint_detection(filepath)
        if not is_fingerprint:
            os.remove(filepath)
            return render_template_string(f'''
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Enrollment Failed</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
                <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
            </head>
            <body class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                    <div class="text-6xl text-red-500 mb-4">❌</div>
                    <h3 class="text-3xl font-bold mb-3 text-red-800">ENROLLMENT FAILED</h3>
                    <p class="text-lg text-gray-700 mb-4">The uploaded image is not a valid fingerprint.</p>
                    <ul class="list-none space-y-2 text-base text-gray-700 mb-6 mx-auto w-fit text-left">
                        <li><strong>Detection Score:</strong> <span class="font-mono bg-red-100 text-red-800 px-2 rounded-md">{fp_confidence:.3f}</span> (min: {FINGERPRINT_THRESHOLDS['overall_score']})</li>
                        <li><strong>Ridge Density:</strong> <span class="font-mono bg-red-100 text-red-800 px-2 rounded-md">{fp_analysis.get('ridge_density', 0):.3f}</span> (min: {FINGERPRINT_THRESHOLDS['ridge_density']})</li>
                        <li><strong>Minutiae Count:</strong> <span class="font-mono bg-red-100 text-red-800 px-2 rounded-md">{fp_analysis.get('minutiae_count', 0)}</span> (min: {FINGERPRINT_THRESHOLDS['minutiae_count']})</li>
                    </ul>
                    <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                        ← Back to Main Page
                    </a>
                </div>
            </body>
            </html>
            ''')

        x = extract_enhanced_features(filepath) # x is now guaranteed to be TARGET_FEATURE_LENGTH
        
        K = os.urandom(KEY_LENGTH) # Generate random secret key
        C = rsc.encode(K) # Encode key with Reed-Solomon for error correction
        
        # P = C XOR x (helper data)
        # Ensure x is at least as long as C for XORing - this is guaranteed since TARGET_FEATURE_LENGTH > KEY_LENGTH + RS_ECC_SYMBOLS
        P = bytes([a ^ b for a, b in zip(C, x[:len(C)])]) + x[len(C):]

        seed = os.urandom(SEED_LENGTH)
        R = strong_extractor(K, seed) # Generate verification hash

        db.fingerprints[fingerprint_name] = {
            'fingerprint_name': fingerprint_name,
            'helper_data': P,
            'verification_hash': R,
            'seed': seed,
            'feature_length': len(x), # Will be TARGET_FEATURE_LENGTH
            'codeword_length': len(C),
            'raw_features': x, # For demo, storing raw features for similarity. NOT secure for production.
            'fingerprint_confidence': fp_confidence, #how fingerprint is clear and good
            'fingerprint_analysis': fp_analysis, #reason behind the confidence
        }
        
        # Convert K to hex for display
        k_hex = K.hex()

        return render_template_string(f'''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enrollment Successful</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-green-200 text-center">
                <div class="text-6xl text-green-500 mb-4">✅</div>
                <h3 class="text-3xl font-bold mb-3 text-green-800">ENROLLMENT SUCCESSFUL</h3>
                <p class="text-lg text-gray-700 mb-2">Fingerprint Name: <span class="font-mono bg-green-100 text-green-800 px-2 py-1 rounded-md">{fingerprint_name}</span></p>
                <p class="text-lg text-gray-700 mb-4">Enrollment Confidence: <span class="font-mono bg-green-100 text-green-800 px-2 py-1 rounded-md">{fp_confidence:.3f}</span></p>
                <div class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 break-all">
                    <strong class="font-semibold text-red-900">⚠️ For Demo Only: Secret Key (K):</strong><br>
                    <span class="font-mono">{k_hex}</span>
                </div>
                <a href='/' class='mt-6 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')
    except Exception as e:
        return render_template_string(f'''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enrollment Failed</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">Enrollment Failed</h3>
                <p class="text-lg text-gray-700 mb-6">An unexpected error occurred: <span class="font-mono bg-red-100 text-red-800 px-2 py-1 rounded-md">{str(e)}</span></p>
                <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handles fingerprint authentication (identification)."""
    file = request.files.get('bio')  # Receive uploaded image
    if not file:
        return render_template_string('''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Authentication Failed</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body { font-family: 'Inter', sans-serif; background-color: #eef2f6; } </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">Authentication Failed</h3>
                <p class="text-lg text-gray-700 mb-6">No file uploaded. Please select a fingerprint image.</p>
                <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')

    temp_filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"auth_temp_{uuid.uuid4()}_{secure_filename(file.filename)}"
    )
    file.save(temp_filepath)

    # Initialize authentication results
    overall_match_details = "Authentication failed: No matching fingerprint found."
    overall_success = False # Default to False, will be set to True on successful match
    max_similarity_score_overall = 0.0
    matched_fingerprint_name = "N/A"
    k_recovered_hex = "N/A" # Initialize k_recovered_hex for all cases

    try:
        # Step 1: Validate fingerprint
        is_fingerprint_presented, fp_confidence_presented, _ = comprehensive_fingerprint_detection(temp_filepath)
        if not is_fingerprint_presented:
            os.remove(temp_filepath)
            db.add_auth_log("N/A", False, 0.0, fp_confidence_presented,
                            "Authentication failed: Presented image is not a valid fingerprint.")
            return render_template_string(f'''
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authentication Failed</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
                <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
            </head>
            <body class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                    <div class="text-6xl text-red-500 mb-4">❌</div>
                    <h3 class="text-3xl font-bold mb-3 text-red-800">AUTHENTICATION FAILED</h3>
                    <p class="text-lg text-gray-700 mb-4">The uploaded image did not pass fingerprint validation checks.</p>
                    <p class="text-lg text-gray-700 mb-6"><strong>Detection Score:</strong> <span class="font-mono bg-red-100 text-red-800 px-2 py-1 rounded-md">{fp_confidence_presented:.3f}</span></p>
                    <a href='/' class='mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                        ← Back to Main Page
                    </a>
                </div>
            </body>
            </html>
            ''')

        # Step 2: Extract feature vector - now guaranteed to be TARGET_FEATURE_LENGTH
        x_prime = extract_enhanced_features(temp_filepath)

        # Step 3: Compare against all enrolled fingerprints
        for fingerprint_name, enrolled_template in db.fingerprints.items():
            if db.is_locked(fingerprint_name):
                db.add_auth_log(fingerprint_name, False, 0.0, fp_confidence_presented,
                                f"Authentication failed: Fingerprint '{fingerprint_name}' is locked.")
                continue

            # x_prime is already TARGET_FEATURE_LENGTH, so no need for padding/trimming here
            similarity_score = calculate_feature_similarity(enrolled_template['raw_features'], x_prime)
            max_similarity_score_overall = max(max_similarity_score_overall, similarity_score)

            if similarity_score >= MATCH_THRESHOLD:
                try:
                    # Fuzzy commitment decoding
                    C_prime = bytes([a ^ b for a, b in zip(enrolled_template['helper_data'][:enrolled_template['codeword_length']],
                                                           x_prime[:enrolled_template['codeword_length']])])
                    K_recovered_with_parity, _, _ = rsc.decode(C_prime)
                    K_recovered = K_recovered_with_parity[:KEY_LENGTH]
                    R_prime = strong_extractor(K_recovered, enrolled_template['seed'])

                    if R_prime == enrolled_template['verification_hash']:
                        db.reset_failed_attempts(fingerprint_name)
                        db.add_auth_log(fingerprint_name, True, similarity_score, fp_confidence_presented,
                                        "Authentication successful.")
                        matched_fingerprint_name = fingerprint_name
                        overall_success = True
                        k_recovered_hex = K_recovered.hex() # Store recovered K for display
                        break # Break loop on successful match
                    else:
                        db.record_failed_attempt(fingerprint_name)
                        db.add_auth_log(fingerprint_name, False, similarity_score, fp_confidence_presented,
                                        "Authentication failed: Cryptographic verification mismatch.")
                        k_recovered_hex = K_recovered.hex() + " (Crypto Mismatch)" # Indicate mismatch but show recovered value
                except (ReedSolomonError, ValueError) as e:
                    db.record_failed_attempt(fingerprint_name)
                    db.add_auth_log(fingerprint_name, False, similarity_score, fp_confidence_presented,
                                    f"Authentication failed: Key recovery error ({str(e)}).")
                    k_recovered_hex = "Recovery Failed" # Indicate recovery failure

        # 🔒 Re-added the bypass logic for very high similarity, for research/testing flexibility
        if not overall_success and max_similarity_score_overall >= MATCH_THRESHOLD and max_similarity_score_overall >= 0.99:
            overall_success = True
            matched_fingerprint_name = "Match (Similarity 100% - Bypassed Crypto)" # Indicate bypass
            db.add_auth_log(matched_fingerprint_name, True, max_similarity_score_overall,
                            fp_confidence_presented, "Bypassed crypto: Very high similarity.")
            k_recovered_hex = "Bypassed (Not recovered)" # In this case, K is not truly recovered via crypto


        # Step 4: Final result display
        if overall_success:
            return render_template_string(f'''
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authentication Success</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
                <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
            </head>
            <body class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-green-200 text-center">
                    <div class="text-6xl text-green-500 mb-4">✅</div>
                    <h3 class="text-3xl font-bold mb-3 text-green-800">AUTHENTICATION SUCCESSFUL</h3>
                    <p class="text-lg text-gray-700 mb-4">Identified Fingerprint: <strong class="font-mono bg-green-100 text-green-800 px-2 py-1 rounded-md">{matched_fingerprint_name}</strong></p>
                    
                    <p class="text-lg text-gray-700 mb-2"><strong>Match Confidence:</strong></p>
                    <div class="w-full bg-gray-200 rounded-full h-8 mb-4 overflow-hidden">
                        <div class="bg-green-500 h-full flex items-center justify-center text-white text-lg font-bold transition-all duration-500 ease-in-out rounded-full"
                            style="width: {max_similarity_score_overall*100:.0f}%;">
                            {max_similarity_score_overall*100:.0f}% Match
                        </div>
                    </div>
                    
                    <p class="text-lg text-gray-700 mb-4">Presented Fingerprint Quality: <span class="font-mono bg-green-100 text-green-800 px-2 py-1 rounded-md">{fp_confidence_presented:.3f}</span></p>
                    <div class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 break-all">
                        <strong class="font-semibold text-red-900">⚠️ For Demo Only: Recovered Key (K'):</strong><br>
                        <span class="font-mono">{k_recovered_hex}</span>
                    </div>
                    <a href='/' class='mt-6 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                        ← Back to Main Page
                    </a>
                </div>
            </body>
            </html>
            ''')
        else:
            # Choose progress bar color based on similarity
            progress_bar_color = "bg-red-500"
            if max_similarity_score_overall >= 0.5:
                progress_bar_color = "bg-orange-500"
            if max_similarity_score_overall >= MATCH_THRESHOLD:
                progress_bar_color = "bg-yellow-500" # High similarity, but failed crypto/recovery

            return render_template_string(f'''
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Authentication Failed</title>
                <script src="https://cdn.tailwindcss.com"></script>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
                <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
            </head>
            <body class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                    <div class="text-6xl text-red-500 mb-4">❌</div>
                    <h3 class="text-3xl font-bold mb-3 text-red-800">AUTHENTICATION FAILED</h3>
                    <p class="text-lg text-gray-700 mb-4">No matching fingerprint found. Please try again with a clearer scan.</p>
                    
                    <p class="text-lg text-gray-700 mb-2"><strong>Highest Match Confidence:</strong></p>
                    <div class="w-full bg-gray-200 rounded-full h-8 mb-4 overflow-hidden">
                        <div class="{progress_bar_color} h-full flex items-center justify-center text-white text-lg font-bold transition-all duration-500 ease-in-out rounded-full"
                            style="width: {max_similarity_score_overall*100:.0f}%;">
                            {max_similarity_score_overall*100:.0f}% Match
                        </div>
                    </div>
                    
                    <p class="text-lg text-gray-700 mb-4">Presented Fingerprint Quality: <span class="font-mono bg-red-100 text-red-800 px-2 py-1 rounded-md">{fp_confidence_presented:.3f}</span></p>
                    <div class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 break-all">
                        <strong class="font-semibold text-red-900">⚠️ For Demo Only: Recovered Key (K'):</strong><br>
                        <span class="font-mono">{k_recovered_hex}</span>
                    </div>
                    <a href='/' class='mt-6 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                        ← Back to Main Page
                    </a>
                </div>
            </body>
            </html>
            ''')

    except Exception as e:
        db.add_auth_log("N/A (Error)", False, 0.0, 0.0, f"Authentication error: {str(e)}")
        # Ensure k_recovered_hex is set even in a general exception
        k_recovered_hex_on_error = f"Error during processing: {str(e)}"
        return render_template_string(f'''
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Authentication Error</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
            <style> body {{ font-family: 'Inter', sans-serif; background-color: #eef2f6; }} </style>
        </head>
        <body class="flex items-center justify-center min-h-screen p-4">
            <div class="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-xl border border-red-200 text-center">
                <div class="text-6xl text-red-500 mb-4">❌</div>
                <h3 class="text-3xl font-bold mb-3 text-red-800">AUTHENTICATION FAILED</h3>
                <p class="text-lg text-gray-700 mb-4">An unexpected error occurred during the authentication process.</p>
                <div class="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 break-all">
                    <strong class="font-semibold text-red-900">Error Details:</strong><br>
                    <span class="font-mono">{k_recovered_hex_on_error}</span>
                </div>
                <a href='/' class='mt-6 inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2.5 px-6 rounded-lg transition duration-200'>
                    ← Back to Main Page
                </a>
            </div>
        </body>
        </html>
        ''')

    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


@app.route('/logs', methods=['GET'])
def get_logs():
    """Returns all authentication logs as JSON."""
    return jsonify(db.auth_logs)

@app.route('/fingerprints', methods=['GET'])
def get_fingerprints():
    """Returns a list of enrolled fingerprints, omitting sensitive data."""
    safe_fp_list = []
    for fingerprint_name, template in db.fingerprints.items():
        safe_fp_list.append({
            "fingerprint_name": fingerprint_name,
            "feature_length": template.get("feature_length"),
            "fingerprint_confidence_at_enrollment": template.get("fingerprint_confidence"),
        })
    return jsonify(safe_fp_list)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port)