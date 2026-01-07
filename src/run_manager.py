#!/usr/bin/env python3
"""
PHOENIX v3.3: Emergent Universe Engine
Copyright (C) 2026 Marcel Langjahr. All Rights Reserved.

Licensed under GPL-3.0
Author: Marcel Langjahr
Contact: marcel@langjahr.org
═══════════════════════════════════════════════════════════════════════════════
PHOENIX v3.3 - RUN MANAGER
═══════════════════════════════════════════════════════════════════════════════

Manages simulation runs with automatic numbering and configuration tracking.

FEATURES:
─────────
✅ Auto-numbered runs (run_001, run_002, ...)
✅ Configuration saved per run
✅ No overwriting of old data
✅ Easy selection for analysis tools

USAGE:
─────
# From simulation:
run_manager = RunManager()
run_dir = run_manager.create_new_run(config)

# From tools:
run_manager = RunManager()
run_dir = run_manager.get_run_dir(3)  # Get run_003
"""

import os
import json
import glob
from datetime import datetime

class RunManager:
    """
    Manages simulation runs with automatic numbering.
    """
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        
        # Create base directory if needed
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"📁 Created datasets directory: {base_dir}")
    
    def get_next_run_number(self):
        """Get next available run number"""
        existing = glob.glob(f"{self.base_dir}/run_*")
        
        if not existing:
            return 1
        
        # Extract numbers
        numbers = []
        for path in existing:
            try:
                num = int(path.split('_')[-1])
                numbers.append(num)
            except:
                pass
        
        if not numbers:
            return 1
        
        return max(numbers) + 1
    
    def create_new_run(self, config=None):
        """
        Create new run directory with auto-numbering.
        
        Args:
            config: Dict with run configuration
        
        Returns:
            str: Path to run directory
        """
        run_num = self.get_next_run_number()
        run_name = f"run_{run_num:03d}"
        run_dir = os.path.join(self.base_dir, run_name)
        
        # Create directories
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
        
        print(f"\n{'═'*80}")
        print(f"📁 NEW RUN: {run_name}")
        print(f"{'═'*80}")
        print(f"Directory: {run_dir}")
        
        # Save config
        if config is None:
            config = {}
        
        config['run_number'] = run_num
        config['run_name'] = run_name
        config['timestamp'] = datetime.now().isoformat()
        
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved: {config_path}")
        print(f"{'═'*80}\n")
        
        return run_dir
    
    def get_run_dir(self, run_number):
        """
        Get directory for specific run number.
        
        Args:
            run_number: Run number (1, 2, 3, ...) or "latest"
        
        Returns:
            str: Path to run directory or None if not found
        """
        if isinstance(run_number, str) and run_number.lower() == "latest":
            existing = sorted(glob.glob(f"{self.base_dir}/run_*"))
            if not existing:
                print(f"❌ No runs found in {self.base_dir}")
                return None
            return existing[-1]
        
        # Konvertiere run_number zu int, falls es ein String ist
        try:
            if isinstance(run_number, str):
                run_number = int(run_number)
        except ValueError:
            print(f"❌ Invalid run number: {run_number}")
            print(f"   Must be integer or 'latest'")
            return None
        
        run_name = f"run_{run_number:03d}"
        run_dir = os.path.join(self.base_dir, run_name)
        
        if not os.path.exists(run_dir):
            print(f"❌ Run not found: {run_name}")
            print(f"   Available runs: {self.list_runs()}")
            return None
        
        return run_dir
    
    def list_runs(self):
        """List all available runs"""
        runs = sorted(glob.glob(f"{self.base_dir}/run_*"))
        
        if not runs:
            return []
        
        run_info = []
        
        for run_dir in runs:
            run_name = os.path.basename(run_dir)
            
            # Load config if available
            config_path = os.path.join(run_dir, "config.json")
            config = {}
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Count files
            n_snapshots = len(glob.glob(f"{run_dir}/snapshots/*.pkl"))
            has_history = os.path.exists(os.path.join(run_dir, "history.pkl"))
            
            run_info.append({
                'name': run_name,
                'number': int(run_name.split('_')[-1]),
                'timestamp': config.get('timestamp', 'unknown'),
                'max_steps': config.get('max_steps', 'unknown'),
                'n_snapshots': n_snapshots,
                'has_history': has_history,
            })
        
        return run_info
    
    def print_run_summary(self):
        """Print summary of all runs"""
        runs = self.list_runs()
        
        if not runs:
            print(f"📁 No runs found in {self.base_dir}")
            return
        
        print(f"\n{'═'*80}")
        print(f"AVAILABLE RUNS")
        print(f"{'═'*80}\n")
        
        print(f"{'Run':<8} {'Date':<20} {'Steps':<8} {'Snapshots':<12} {'History':<8}")
        print(f"{'─'*80}")
        
        for r in runs:
            date_str = r['timestamp'][:19] if r['timestamp'] != 'unknown' else 'unknown'
            history_str = '✓' if r['has_history'] else '✗'
            
            print(f"{r['name']:<8} {date_str:<20} {str(r['max_steps']):<8} "
                  f"{r['n_snapshots']:<12} {history_str:<8}")
        
        print(f"\n{'═'*80}\n")

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_default_config():
    """Get default configuration for a run"""
    return {
        'version': 'v3.2',
        'max_steps': 10000,
        'N_init': 5,
        'snapshot_interval': 200,
        'validation_interval': 200,
        'console_log_seconds': 10,
        'history_log_steps': 100,
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test
    manager = RunManager()
    
    # List existing runs
    manager.print_run_summary()
    
    # Create new run
    config = get_default_config()
    config['test'] = True
    
    run_dir = manager.create_new_run(config)
    
    print(f"Created: {run_dir}")
    
    # List again
    manager.print_run_summary()
