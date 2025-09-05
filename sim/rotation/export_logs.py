import os
import argparse
import json
from tensorboard.backend.event_processing import event_accumulator
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorboard')

def export_tensorboard_logs(log_dir):
    """
    Extracts all scalar data from a TensorBoard event file into a JSON format
    and prints it to standard output.
    """
    event_file = None
    try:
        for root, _, files in os.walk(log_dir):
            for file in files:
                if "events.out.tfevents" in file:
                    event_file = os.path.join(root, file)
                    break
            if event_file:
                break
    except Exception as e:
        print(f"Error walking directory {log_dir}: {e}")
        return

    if not event_file:
        print(f"Error: No TensorBoard event file found in {log_dir}")
        return

    try:
        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        all_data = {}
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            all_data[tag] = {
                "steps": [e.step for e in events],
                "values": [e.value for e in events]
            }
            
        print(json.dumps(all_data, indent=2))

    except Exception as e:
        print(f"An error occurred while processing {event_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar data to a JSON string.")
    parser.add_argument("log_dir", type=str, help="Path to the directory containing the tfevents file (e.g., 'runs/my_run/logs/').")
    args = parser.parse_args()
    export_tensorboard_logs(args.log_dir)
