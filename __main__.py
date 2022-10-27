import argparse
import os
import sys

import lwm


def main() -> int:
    app_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--version", action="store_true", help="Returns the application version.")
    parser.add_argument(
        "-c", "--config", type=str, help="Path to configuration file.",
        default=os.path.join(app_path, "settings.yaml"))
    parser.add_argument("-l", "--log", type=str, help="Path of log file.", default=None)

    args = parser.parse_args()

    if args.version:
        import importlib.metadata
        print(importlib.metadata.version("LogistiekeWegvervoerModule"))
        return 0

    return lwm.execute(args.config, log_path=args.log)


if __name__ == "__main__":
    sys.exit(main())
