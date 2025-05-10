import argparse
from .core import main

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run main logic')
    args = parser.parse_args()

    if args.run:
        main.run()

if __name__ == '__main__':
    main_cli()
