#!/usr/bin/env python

import os
import sys
import argparse
import logging
import json
import gc
from datetime import datetime
from multiprocessing import Process, Queue

# Local imports
sys.path.append('./Source')
sys.path.append('./_dependency')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import load_data


def setup_logging(log_dir: str = 'logs') -> str:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'shards_only_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_file


def get_total_patients() -> int:
    try:
        _, _, _ = load_data(batch_size=1)
        A_full, _, _ = load_data(batch_size=None)
        total_patients = A_full.shape[0]
        logging.info(f"Total patients in dataset: {total_patients}")
        return total_patients
    except Exception as exc:
        logging.error(f"Error getting total patients: {exc}")
        raise


def _run_shard_process(shard_start: int, shard_size: int, pnum: int, shard_id: int, queue: Queue) -> None:
    try:
        logging.info(f"--- Starting Shard {shard_id} (PID: {os.getpid()}) ---")

        patient_indices = list(range(shard_start, shard_start + shard_size))
        patient_indices_str = ",".join(map(str, patient_indices))

        shard_args = [
            sys.executable, 'run_model.py',
            '--pnum', str(pnum),
            '--patient_indices', patient_indices_str
        ]

        env = os.environ.copy()
        env['SHARD_START'] = str(shard_start)
        env['SHARD_ID'] = str(shard_id)

        logging.info(f"Command: {' '.join(shard_args)}")

        # Stream output directly; avoid buffering to memory
        import subprocess
        subprocess.run(
            shard_args,
            env=env,
            check=True,
        )

        # Locate latest results directory produced by run_model.py
        res_path = './Res'
        results_dir = None
        if os.path.exists(res_path):
            dirs = [
                os.path.join(res_path, d)
                for d in os.listdir(res_path)
                if d.startswith('hierarchical_gp_') and os.path.isdir(os.path.join(res_path, d))
            ]
            if dirs:
                results_dir = max(dirs, key=os.path.getmtime)

        if results_dir is None:
            queue.put(f"ERROR: Could not find results directory for shard {shard_id}")
        else:
            queue.put(results_dir)

    except Exception as exc:
        queue.put(f"ERROR: Shard {shard_id} failed with error: {exc}")


def run_shard(shard_start: int, shard_size: int, pnum: int, shard_id: int) -> str:
    queue: Queue = Queue()
    process: Process = Process(target=_run_shard_process, args=(shard_start, shard_size, pnum, shard_id, queue))
    process.start()
    process.join()
    try:
        result = queue.get(timeout=600)
    except Exception:
        result = "ERROR: No result received from shard process"
    finally:
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass
        if process.is_alive():
            process.terminate()
        gc.collect()
    return result


def save_manifest(shard_results, failed_shards, args, total_patients) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'./Res/sharded_run_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.json')

    manifest = {
        'timestamp': timestamp,
        'total_patients': total_patients,
        'shard_size': args.shard_size,
        'pnum': args.pnum,
        'num_shards_planned': (total_patients + args.shard_size - 1) // args.shard_size,
        'shard_results_dirs': shard_results,
        'failed_shards': failed_shards,
        'args': vars(args),
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Manifest saved to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run shards only and write a manifest for later combination')
    parser.add_argument('--pnum', default=500, type=int, help='Number of posterior samples per shard')
    parser.add_argument('--shard_size', default=1000, type=int, help='Number of patients per shard')
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting shards-only run. Arguments: {args}")

    try:
        total_patients = get_total_patients()
        num_shards = (total_patients + args.shard_size - 1) // args.shard_size
        logging.info(f"Total patients: {total_patients}")
        logging.info(f"Shard size: {args.shard_size}")
        logging.info(f"Number of shards: {num_shards}")

        shard_results = []
        failed_shards = []

        for shard_id in range(num_shards):
            shard_start = shard_id * args.shard_size
            shard_size = min(args.shard_size, total_patients - shard_start)
            logging.info(f"Processing shard {shard_id + 1}/{num_shards}")

            result = run_shard(shard_start, shard_size, args.pnum, shard_id + 1)
            if isinstance(result, str) and result.startswith('ERROR:'):
                logging.error(result)
                failed_shards.append(shard_id + 1)
            else:
                shard_results.append(result)
                logging.info(f"Shard {shard_id + 1} completed: {result}")

        if failed_shards:
            logging.warning(f"Failed shards: {failed_shards}")

        manifest_path = save_manifest(shard_results, failed_shards, args, total_patients)
        logging.info("Shards-only execution completed.")
        logging.info(f"Manifest path: {manifest_path}")

    except Exception as exc:
        logging.error(f"An error occurred during shards-only execution: {exc}", exc_info=True)
        raise
    finally:
        gc.collect()


if __name__ == '__main__':
    main()




