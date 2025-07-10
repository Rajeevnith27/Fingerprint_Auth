import os
import json
import time

class BiometricDatabase:
    """
    Simulates a database for storing fingerprints and authentication logs with persistence.

    Features:
    - Data persistence using JSON files.
    - Structured storage for enrolled fingerprint templates.
    - Comprehensive authentication logging.
    - Data validation during loading.
    """
    def __init__(self, db_dir='database'):
        self.db_dir = db_dir
        self.fingerprints_file = os.path.join(db_dir, 'fingerprints.json')
        self.auth_logs_file = os.path.join(db_dir, 'auth_logs.json')

        os.makedirs(db_dir, exist_ok=True)

        self.fingerprints = self._load_data(self.fingerprints_file)
        self.auth_logs = self._load_data(self.auth_logs_file)

        # Ensure that 'raw_features', 'helper_data', etc., are loaded as bytes from lists
        # This loop also cleans up any entries where these might be missing or malformed
        fingerprints_to_remove = []
        for name, data in self.fingerprints.items():
            try:
                if 'raw_features' in data and isinstance(data['raw_features'], list):
                    self.fingerprints[name]['raw_features'] = bytes(data['raw_features'])
                elif 'raw_features' not in data:
                    raise ValueError(f"Missing 'raw_features' for {name}")

                if 'helper_data' in data and isinstance(data['helper_data'], list):
                    self.fingerprints[name]['helper_data'] = bytes(data['helper_data'])
                elif 'helper_data' not in data:
                    raise ValueError(f"Missing 'helper_data' for {name}")

                if 'verification_hash' in data and isinstance(data['verification_hash'], list):
                    self.fingerprints[name]['verification_hash'] = bytes(data['verification_hash'])
                elif 'verification_hash' not in data:
                    raise ValueError(f"Missing 'verification_hash' for {name}")

                if 'seed' in data and isinstance(data['seed'], list):
                    self.fingerprints[name]['seed'] = bytes(data['seed'])
                elif 'seed' not in data:
                    raise ValueError(f"Missing 'seed' for {name}")

                # Ensure other critical keys exist
                if not all(k in data for k in ['feature_length', 'codeword_length', 'fingerprint_confidence', 'fingerprint_analysis']):
                    raise ValueError(f"Missing essential keys for {name}")

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
            'raw_features': raw_features,
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