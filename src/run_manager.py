#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine - Run Manager
"""

import os
import json
import glob
import pickle
from datetime import datetime

class RunManager:
    """
    Manages simulation runs, numbering, and snapshot lifecycle.
    """
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"📁 Created datasets directory: {base_dir}")
    
    def get_next_run_number(self):
        """Get next available run number"""
        existing = glob.glob(f"{self.base_dir}/run_*")
        if not existing: return 1
        numbers = []
        for path in existing:
            try: numbers.append(int(path.split('_')[-1]))
            except: pass
        return max(numbers) + 1 if numbers else 1
    
    def create_new_run(self, config=None, resumed_from=None):
        """Create new run directory."""
        run_num = self.get_next_run_number()
        run_name = f"run_{run_num:03d}"
        run_dir = os.path.join(self.base_dir, run_name)
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
        
        print(f"\n{'═'*80}")
        print(f"📁 NEW RUN: {run_name}")
        if resumed_from:
            print(f"♻️  RESUMED FROM: {resumed_from}")
        print(f"{'═'*80}")
        
        if config is None: config = {}
        config['run_number'] = run_num
        config['run_name'] = run_name
        config['timestamp'] = datetime.now().isoformat()
        if resumed_from: config['resumed_from'] = resumed_from
        
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        return run_dir
    
    def get_run_dir(self, run_number):
        """
        Get directory for specific run number.
        Args:
            run_number: Run number (int) or "latest" (str)
        """
        if isinstance(run_number, str) and run_number.lower() == "latest":
            return self.find_latest_run_dir()
        
        # Falls run_number als String (z.B. "005") kommt
        try:
            run_number = int(run_number)
        except:
            return None
            
        run_name = f"run_{run_number:03d}"
        run_dir = os.path.join(self.base_dir, run_name)
        
        if os.path.exists(run_dir):
            return run_dir
        return None

    # ───────────────────────────────────────────────────────────────────────
    # SNAPSHOT HANDLING
    # ───────────────────────────────────────────────────────────────────────

    def find_latest_run_dir(self):
        """Finds the directory of the most recent run."""
        runs = sorted(glob.glob(f"{self.base_dir}/run_*"))
        if not runs: return None
        # Sort key logic to handle run_002 vs run_010 correctly
        runs.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        return runs[-1]

    def find_snapshot(self, step_query, run_dir=None):
        """
        Find a specific snapshot pickle file.
        
        Args:
            step_query: 'latest' or integer (e.g. 800)
            run_dir: Specific run directory, or None (searches latest run)
        """
        if run_dir is None:
            run_dir = self.find_latest_run_dir()
            if not run_dir: return None, None
        
        snap_dir = os.path.join(run_dir, "snapshots")
        all_snaps = glob.glob(f"{snap_dir}/snapshot_step_*.pkl")
        
        if not all_snaps: return None, None

        # Helper to extract step number from filename
        def get_step(fname):
            try: return int(fname.split('step_')[-1].split('.')[0])
            except: return -1
            
        all_snaps.sort(key=get_step)
        
        target_file = None
        if step_query == 'latest':
            target_file = all_snaps[-1]
        else:
            try:
                target_step = int(step_query)
                # Find exact match
                for s in all_snaps:
                    if get_step(s) == target_step:
                        target_file = s
                        break
            except: pass
            
        return target_file, run_dir

    def prune_old_snapshots(self, run_dir, keep_count):
        """
        Deletes old snapshots, keeping only the 'keep_count' most recent ones.
        """
        snap_dir = os.path.join(run_dir, "snapshots")
        all_snaps = glob.glob(f"{snap_dir}/snapshot_step_*.pkl")
        
        if len(all_snaps) <= keep_count:
            return

        def get_step(fname):
            try: return int(fname.split('step_')[-1].split('.')[0])
            except: return -1
            
        all_snaps.sort(key=get_step)
        to_remove = all_snaps[:-keep_count]
        
        count = 0
        for f in to_remove:
            try:
                os.remove(f)
                count += 1
            except Exception as e:
                print(f"⚠️ Failed to prune {f}: {e}")
                
        if count > 0:
            print(f"🧹 Pruned {count} old snapshots.")

    def load_snapshot_data(self, filepath):
        """Loads and returns the dictionary from a pickle file."""
        if not filepath or not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"❌ Error loading snapshot {filepath}: {e}")
            return None

# ───────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (Das fehlte!)
# ───────────────────────────────────────────────────────────────────────

def get_default_config():
    """Get default configuration for a run"""
    return {
        'version': 'v3.3',
        'max_steps': 10000,
        'N_init': 5,
        'snapshot_interval': 200,
        'validation_interval': 200,
        'console_log_seconds': 10,
        'history_log_steps': 100,
    }

# For testing
if __name__ == "__main__":
    rm = RunManager()
    print(f"Latest run: {rm.find_latest_run_dir()}")
