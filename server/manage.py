# DIA/manage.py
#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from decouple import config

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Check if the command is 'runserver' and modify the port if provided
    if len(sys.argv) > 1 and sys.argv[1] == 'runserver':
        if len(sys.argv) == 2:  # Only 'runserver' without port
            port = config('DJANGO_PORT', default='8000')
            sys.argv.append(f'0.0.0.0:{port}')  # Ensure binding to 0.0.0.0
        elif len(sys.argv) == 3 and ':' not in sys.argv[2]:  # Only port specified (e.g., runserver 3333)
            port = sys.argv[2]
            sys.argv[2] = f'0.0.0.0:{port}'  # Modify to include 0.0.0.0

    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
