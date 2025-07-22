from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import uuid
from werkzeug.utils import secure_filename
import time
import json # For BiometricDatabase if using JSON files
import sqlite3 # Just in case the user's previous app.py used it for get_db_connection
from dotenv import load_dotenv

load_dotenv()

port=PORT or 5000

# Import modules from your project
# Assuming config.py and features.py are in the same directory
from config import UPLOAD_FOLDER, FINGERPRINT_THRESHOLDS, MATCH_THRESHOLD, REQUIRE_CRYPTO_FOR_100_PERCENT, KEY_LENGTH, SEED_LENGTH, RS_ECC_SYMBOLS
from features import comprehensive_fingerprint_detection, extract_enhanced_features, calculate_feature_similarity, strong_extractor
from reedsolo import RSCodec, ReedSolomonError # Make sure reedsolo is imported here too
import cv2 # For image processing (though less direct use in this simplified version)
import numpy as np # For image processing (though less direct use in this simplified version)

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Reed-Solomon Codec
rsc = RSCodec(RS_ECC_SYMBOLS)

# --- BiometricDatabase Class (Copied directly into app.py for simplicity) ---
# In a larger project, this would typically be in a separate file like database.py
class BiometricDatabase:
    """
    Simulates a database for storing fingerprints and authentication logs with persistence
    using JSON files.
    """
    def __init__(self, db_dir='database'):
        self.db_dir = db_dir
        self.fingerprints_file = os.path.join(db_dir, 'fingerprints.json')
        self.auth_logs_file = os.path.join(db_dir, 'auth_logs.json')

        os.makedirs(db_dir, exist_ok=True)

        self.fingerprints = self._load_data(self.fingerprints_file)
        self.auth_logs = self._load_data(self.auth_logs_file)

        # Ensure that 'raw_features', 'helper_data', etc., are loaded as bytes from lists
        fingerprints_to_remove = []
        for name, data in list(self.fingerprints.items()): # Use list() for safe modification during iteration
            try:
                # Convert list back to bytes for features and cryptographic data
                if 'raw_features' in data and isinstance(data['raw_features'], list):
                    self.fingerprints[name]['raw_features'] = bytes(data['raw_features'])
                else: raise ValueError(f"Missing or invalid 'raw_features' for {name}")

                if 'helper_data' in data and isinstance(data['helper_data'], list):
                    self.fingerprints[name]['helper_data'] = bytes(data['helper_data'])
                else: raise ValueError(f"Missing or invalid 'helper_data' for {name}")

                if 'verification_hash' in data and isinstance(data['verification_hash'], list):
                    self.fingerprints[name]['verification_hash'] = bytes(data['verification_hash'])
                else: raise ValueError(f"Missing or invalid 'verification_hash' for {name}")

                if 'seed' in data and isinstance(data['seed'], list):
                    self.fingerprints[name]['seed'] = bytes(data['seed'])
                else: raise ValueError(f"Missing or invalid 'seed' for {name}")

                # Ensure other critical keys exist
                if not all(k in data for k in ['feature_length', 'codeword_length', 'fingerprint_confidence', 'fingerprint_analysis']):
                    raise ValueError(f"Missing essential metadata keys for {name}")

            except (ValueError, TypeError) as e:
                print(f"Data corruption detected for fingerprint '{name}': {e}. This entry will be ignored.")
                fingerprints_to_remove.append(name)

        for name in fingerprints_to_remove:
            del self.fingerprints[name]
        self._save_data(self.fingerprints, self.fingerprints_file) # Save cleaned state

    def _load_data(self, filepath):
        """Loads data from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {filepath}: {e}. Returning empty dict/list.")
                return {} if "fingerprints" in filepath else []
            except Exception as e:
                print(f"An unexpected error occurred while loading {filepath}: {e}. Returning empty dict/list.")
                return {} if "fingerprints" in filepath else []
        return {} if "fingerprints" in filepath else []

    def _save_data(self, data, filepath):
        """Saves data to a JSON file."""
        def convert_bytes_to_list(obj):
            if isinstance(obj, bytes):
                return list(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4, default=convert_bytes_to_list)
        except TypeError as e:
            print(f"Serialization error for {filepath}: {e}. Data might not be saved correctly.")
        except Exception as e:
            print(f"Failed to save data to {filepath}: {e}.")

    def add_auth_log(self, fingerprint_name, success, confidence_score, quality_score, details):
        """Adds an entry to the authentication log and saves."""
        log_entry = {
            'timestamp': time.time(),
            'fingerprint_name': fingerprint_name,
            'success': success,
            'confidence_score': confidence_score,
            'quality_score': quality_score,
            'match_details': details
        }
        self.auth_logs.append(log_entry)
        self._save_data(self.auth_logs, self.auth_logs_file)

    def enroll_fingerprint(self, fingerprint_name, helper_data, verification_hash, seed, feature_length, raw_features, fingerprint_confidence, fingerprint_analysis, codeword_length):
        """
        Adds a new fingerprint template to the database and saves.
        Includes validation to prevent overwriting existing fingerprints.
        """
        if fingerprint_name in self.fingerprints:
            raise ValueError(f"Fingerprint '{fingerprint_name}' already exists in the database.")

        if not isinstance(fingerprint_name, str) or not fingerprint_name:
            raise TypeError("fingerprint_name must be a non-empty string.")
        if not isinstance(helper_data, bytes) or not isinstance(verification_hash, bytes) or \
           not isinstance(seed, bytes) or not isinstance(raw_features, bytes):
            raise TypeError("helper_data, verification_hash, seed, and raw_features must be bytes.")
        if not isinstance(feature_length, int) or not isinstance(codeword_length, int):
            raise TypeError("feature_length and codeword_length must be integers.")
        if not isinstance(fingerprint_confidence, (float, int)) or not isinstance(fingerprint_analysis, dict):
            raise TypeError("fingerprint_confidence must be a number and fingerprint_analysis a dict.")

        template = {
            'fingerprint_name': fingerprint_name,
            'helper_data': helper_data,
            'verification_hash': verification_hash,
            'seed': seed,
            'feature_length': feature_length,
            'codeword_length': codeword_length,
            'raw_features': raw_features, # Still storing raw features for simplified similarity calculation in demo
            'fingerprint_confidence': fingerprint_confidence,
            'fingerprint_analysis': fingerprint_analysis,
            'enrollment_timestamp': time.time()
        }
        self.fingerprints[fingerprint_name] = template
        self._save_data(self.fingerprints, self.fingerprints_file)

    def get_fingerprint_template(self, fingerprint_name):
        """Retrieves a fingerprint template by name."""
        return self.fingerprints.get(fingerprint_name)

    def get_all_fingerprint_names(self):
        """Returns a list of all enrolled fingerprint names."""
        return list(self.fingerprints.keys())

    def get_all_auth_logs(self):
        """Returns all stored authentication logs."""
        return self.auth_logs

    def get_safe_fingerprint_list(self):
        """Returns a list of enrolled fingerprints, omitting sensitive data for display."""
        safe_fp_list = []
        for fingerprint_name, template in self.fingerprints.items():
            safe_fp_list.append({
                "fingerprint_name": fingerprint_name,
                "feature_length": template.get("feature_length"),
                "fingerprint_confidence_at_enrollment": template.get("fingerprint_confidence"),
                "enrollment_timestamp": template.get("enrollment_timestamp")
            })
        return safe_fp_list

    def delete_fingerprint(self, fingerprint_name):
        """Deletes a fingerprint template and its associated failed attempts."""
        if fingerprint_name in self.fingerprints:
            del self.fingerprints[fingerprint_name]
            self._save_data(self.fingerprints, self.fingerprints_file)
            return True
        return False

# Instantiate the database
db = BiometricDatabase()

# Routes
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with enrollment and authentication forms."""
    fp_count = len(db.get_all_fingerprint_names())
    return render_template('index.html',
                           thresholds=FINGERPRINT_THRESHOLDS,
                           fp_count=fp_count,
                           require_crypto_for_100=REQUIRE_CRYPTO_FOR_100_PERCENT)

@app.route('/enroll', methods=['POST'])
def enroll():
    """Handles fingerprint enrollment."""
    fingerprint_name = request.form.get('fingerprint_name', '').strip()
    if not fingerprint_name:
        return render_template('error.html',
                               title="Enrollment Failed",
                               message="The fingerprint name is required for enrollment. Please provide a unique name.",
                               back_url='/enroll_form',
                               back_text='Try Enrollment Again')

    file = request.files.get('bio')
    if not file:
        return render_template('error.html',
                               title="Enrollment Failed",
                               message="Please select an image file to upload for enrollment.",
                               back_url='/enroll_form',
                               back_text='Try Enrollment Again')

    temp_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_id}_{secure_filename(file.filename)}")
    file.save(filepath)

    try:
        is_fingerprint, fp_confidence, fp_analysis = comprehensive_fingerprint_detection(filepath)

        quality_class = "low"
        if fp_confidence >= 0.3: quality_class = "medium"
        if fp_confidence >= 0.6: quality_class = "high"

        if not is_fingerprint:
            db.add_auth_log(fingerprint_name, False, 0.0, fp_confidence, "Enrollment failed: Not a valid fingerprint (detection).")
            return render_template('error.html',
                                   title="Enrollment Failed - Not a Valid Fingerprint",
                                   message="The system could not validate the uploaded image as a real fingerprint based on its characteristics.",
                                   details={
                                       "Detection Score": {"score": fp_confidence, "class": quality_class},
                                       "Detailed Analysis": fp_analysis
                                   },
                                   back_url='/enroll_form',
                                   back_text='Try Enrollment Again')

        # extract_enhanced_features will now handle its internal rotation
        x = extract_enhanced_features(filepath)

        K = os.urandom(KEY_LENGTH)
        # Pad K to be at least RS_ECC_SYMBOLS long if necessary for Reed-Solomon encoding
        if len(K) < RS_ECC_SYMBOLS:
            K_padded = K + os.urandom(RS_ECC_SYMBOLS - len(K))
        else:
            K_padded = K

        C = rsc.encode(K_padded)

        # Ensure x_truncated_for_P has enough bytes to match C
        x_truncated_for_P = x[:len(C)]
        # If x is shorter than C, pad it with zeros to prevent IndexError
        if len(x_truncated_for_P) < len(C):
            x_truncated_for_P += bytes(len(C) - len(x_truncated_for_P))
        
        # Perform XOR for helper data. Note: Python 3.x bytes objects can be XORed directly if lengths match
        P_core = bytes([a ^ b for a, b in zip(C, x_truncated_for_P)])
        P = P_core + x[len(C):] # P combines XORed part and remaining features

        seed = os.urandom(SEED_LENGTH)
        R = strong_extractor(K, seed)

        db.enroll_fingerprint(
            fingerprint_name=fingerprint_name,
            helper_data=P,
            verification_hash=R,
            seed=seed,
            feature_length=len(x),
            codeword_length=len(C),
            raw_features=x, # Still storing raw features for simplified similarity calculation in demo
            fingerprint_confidence=fp_confidence,
            fingerprint_analysis=fp_analysis
        )
        db.add_auth_log(fingerprint_name, True, fp_confidence, fp_confidence, "Enrollment successful.")

        return render_template('success.html',
                               title="Enrollment Successful",
                               message="This fingerprint has been successfully stored.",
                               details={
                                   "Fingerprint Name": fingerprint_name,
                                   "Enrollment Confidence (Quality)": {
                                       "score": fp_confidence,
                                       "class": quality_class
                                   },
                                   "Detailed Analysis at Enrollment": fp_analysis
                               },
                               back_url='/enroll_form',
                               back_text='Enroll Another Fingerprint')

    except ValueError as ve:
        db.add_auth_log(fingerprint_name, False, 0.0, 0.0, f"Enrollment failed: {str(ve)}")
        return render_template('error.html',
                               title="Enrollment Failed",
                               message=f"Error: {str(ve)}. This usually means the fingerprint name is already taken or there was an issue with the data provided.",
                               back_url='/enroll_form',
                               back_text='Try Enrollment Again')
    except Exception as e:
        db.add_auth_log(fingerprint_name, False, 0.0, 0.0, f"Enrollment failed: Unexpected error: {str(e)}")
        return render_template('error.html',
                               title="Enrollment Failed - Unexpected Error",
                               message=f"An internal error occurred: {str(e)}. Please try again later or contact support.",
                               back_url='/enroll_form',
                               back_text='Try Enrollment Again')
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Handles fingerprint authentication (identification - 1-to-many), relying only on
    the single, robust alignment from features.py. No explicit 180-degree comparison."""
    file = request.files.get('bio')
    if not file:
        return render_template('error.html',
                               title="Authentication Failed",
                               message="Please select an image file to upload for authentication.",
                               back_url='/authenticate_form',
                               back_text='Try Authentication Again')

    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"auth_temp_{uuid.uuid4()}_{secure_filename(file.filename)}")
    file.save(temp_filepath)

    overall_match_details = "Authentication failed: No matching fingerprint found."
    max_similarity_score_overall = 0.0
    identified_fingerprint_name = "N/A"
    fp_confidence_presented = 0.0
    key_recovered_for_best_match = False
    
    try:
        is_fingerprint_presented, fp_confidence_presented, _ = comprehensive_fingerprint_detection(temp_filepath)
        quality_class_presented = "low"
        if fp_confidence_presented >= 0.3: quality_class = "medium"
        if fp_confidence_presented >= 0.6: quality_class = "high"

        if not is_fingerprint_presented:
            db.add_auth_log("N/A", False, 0.0, fp_confidence_presented, "Authentication failed: Presented image is not a valid fingerprint (detection).")
            return render_template('error.html',
                                   title="Authentication Failed - Not a Fingerprint",
                                   message="The uploaded image did not pass fingerprint validation checks.",
                                   details={
                                       "Detection Score": {"score": fp_confidence_presented, "class": quality_class_presented},
                                       "Please ensure the image is a clear fingerprint.": ""
                                   },
                                   back_url='/authenticate_form',
                                   back_text='Try Authentication Again')

        # Extract features for the presented fingerprint ONCE
        # This relies entirely on the improved rotation alignment in features.py
        presented_features_bytes = extract_enhanced_features(temp_filepath)
        
        # Iterate through all enrolled fingerprints in the database
        enrolled_fingerprint_names = db.get_all_fingerprint_names()

        for fingerprint_name in enrolled_fingerprint_names:
            enrolled_template = db.get_fingerprint_template(fingerprint_name)

            if not enrolled_template:
                continue # Skip if template somehow not found

            current_enrolled_features = enrolled_template['raw_features']
            current_helper_data = enrolled_template['helper_data']
            current_seed = enrolled_template['seed']
            current_codeword_length = enrolled_template['codeword_length']

            # Ensure lengths match for XOR operations and similarity
            target_feat_len = len(current_enrolled_features)
            
            # Pad presented features to match enrolled length if necessary
            p_feat = presented_features_bytes[:target_feat_len]
            if len(p_feat) < target_feat_len: p_feat += bytes(target_feat_len - len(p_feat))


            # --- Compare with the aligned presented features ---
            current_similarity = calculate_feature_similarity(current_enrolled_features, p_feat)
            print(f"Debug: Comparing '{fingerprint_name}': {current_similarity:.2f}")

            attempt_successful = False # Track success for current user
            if current_similarity >= MATCH_THRESHOLD:
                try:
                    # Extract core data for XOR
                    core_enrolled_features = current_enrolled_features[:current_codeword_length]
                    core_presented_features = p_feat[:current_codeword_length]
                    
                    if len(core_presented_features) < len(core_enrolled_features):
                        core_presented_features += bytes(len(core_enrolled_features) - len(core_presented_features))

                    # Perform XOR for key recovery
                    C_prime = bytes([a ^ b for a, b in zip(current_helper_data[:len(core_enrolled_features)], core_presented_features)])
                    
                    K_recovered_with_parity, _, _ = rsc.decode(C_prime)
                    K_recovered = K_recovered_with_parity[:KEY_LENGTH]
                    R_prime = strong_extractor(K_recovered, current_seed)

                    if R_prime == enrolled_template['verification_hash']:
                        attempt_successful = True
                        # If this is the best match so far, update overall best
                        if current_similarity > max_similarity_score_overall:
                            max_similarity_score_overall = current_similarity
                            identified_fingerprint_name = fingerprint_name
                            key_recovered_for_best_match = True
                            overall_match_details = f"Authentication successful for '{fingerprint_name}'."
                        print(f"Debug: Key recovered for '{fingerprint_name}'.")

                except (ReedSolomonError, ValueError, IndexError) as e:
                    print(f"Debug: Key recovery failed for '{fingerprint_name}': {e}")
                    pass # Continue to next fingerprint if key recovery failed for this one
            
            # Track overall best match even if key recovery failed for it
            # This ensures max_similarity_score_overall reflects the highest similarity found
            if current_similarity > max_similarity_score_overall:
                max_similarity_score_overall = current_similarity
                # Identified name here might not have successful key recovery yet,
                # but it's the best similarity so far.
                identified_fingerprint_name = fingerprint_name


        # --- Final Authentication Decision ---
        
        quality_class_match = "low"
        if max_similarity_score_overall >= 0.5: quality_class_match = "medium"
        if max_similarity_score_overall >= MATCH_THRESHOLD: quality_class_match = "high"


        if identified_fingerprint_name != "N/A" and key_recovered_for_best_match:
            db.add_auth_log(identified_fingerprint_name, True, max_similarity_score_overall, fp_confidence_presented, "Identification successful.")
            return render_template('success.html',
                                   title="Identification Successful",
                                   message=f"Access Granted for: {identified_fingerprint_name}!",
                                   details={
                                       "Identified Fingerprint": identified_fingerprint_name,
                                       "Match Confidence": {
                                           "score": max_similarity_score_overall,
                                           "class": quality_class_match
                                       },
                                       "Presented Fingerprint Quality": {
                                           "score": fp_confidence_presented,
                                           "class": quality_class_presented
                                       }
                                   },
                                   back_url='/authenticate_form',
                                   back_text='Authenticate Another')
        else:
            final_message = f"Identification failed: No matching fingerprint found or key recovery failed for best match. Highest score found: {max_similarity_score_overall:.2f}."
            if identified_fingerprint_name != "N/A": # If a fingerprint was identified as best match by score, but key failed
                final_message = f"Best match found for '{identified_fingerprint_name}' with {max_similarity_score_overall:.2f}% similarity, but key recovery failed."

            db.add_auth_log("N/A", False, max_similarity_score_overall, fp_confidence_presented, final_message)
            return render_template('error.html',
                                   title="Identification Failed",
                                   message=final_message,
                                   details={
                                       "Highest Match Confidence Found": {
                                           "score": max_similarity_score_overall,
                                           "class": quality_class_match
                                       },
                                       "Presented Fingerprint Quality": {
                                           "score": fp_confidence_presented,
                                           "class": quality_class_presented
                                       }
                                   },
                                   back_url='/authenticate_form',
                                   back_text='Try Authentication Again')

    except Exception as e:
        db.add_auth_log("N/A (Error)", False, 0.0, fp_confidence_presented, f"Authentication error: {str(e)}")
        return render_template('error.html',
                               title="Authentication Failed - Unexpected Error",
                               message=f"An unexpected error occurred: {str(e)}. Please try again later or contact support.",
                               back_url='/authenticate_form',
                               back_text='Try Authentication Again')
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

@app.route('/logs', methods=['GET'])
def get_logs():
    """Returns all authentication logs as JSON."""
    logs = db.get_all_auth_logs()
    for log in logs:
        if 'timestamp' in log:
            log['timestamp_human'] = time.ctime(log['timestamp'])
    return jsonify(logs)

@app.route('/fingerprints', methods=['GET'])
def get_fingerprints():
    """Returns a list of enrolled fingerprints, omitting sensitive data."""
    fps = db.get_safe_fingerprint_list()
    for fp in fps:
        if 'enrollment_timestamp' in fp:
            fp['enrollment_timestamp_human'] = time.ctime(fp['enrollment_timestamp'])
    return jsonify(fps)

@app.route('/delete_fingerprint/<string:fingerprint_name>', methods=['GET'])
def delete_fingerprint(fingerprint_name):
    """Deletes a fingerprint template by name."""
    if db.delete_fingerprint(fingerprint_name):
        return render_template('delete_success.html', fingerprint_name=fingerprint_name)
    else:
        return render_template('error.html',
                               title="Deletion Failed",
                               message=f"Fingerprint '{fingerprint_name}' not found.",
                               back_url='/fingerprints',
                               back_text='View Enrolled Fingerprints')

@app.route('/enroll_form', methods=['GET'])
def enroll_form():
    """Renders a clean enrollment form page."""
    return render_template('enroll_form.html')

@app.route('/authenticate_form', methods=['GET'])
def authenticate_form():
    """Renders a clean authentication form page (no user ID input needed)."""
    return render_template('authenticate_form.html')


if __name__ == '__main__':
    app.run(debug=True,port=port)

