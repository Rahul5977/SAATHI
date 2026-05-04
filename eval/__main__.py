"""Allow `python -m eval` to drive the runner."""

from eval.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
