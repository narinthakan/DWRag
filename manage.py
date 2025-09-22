#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import io


def main():
    """Run administrative tasks."""

    # ✅ บังคับ stdout/stderr ให้ใช้ UTF-8 (แก้ ascii codec error)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DWRag.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
