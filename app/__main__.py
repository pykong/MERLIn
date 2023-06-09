import sys

sys.dont_write_bytecode = True


from .cli import cli

if __name__ == "__main__":
    cli()
