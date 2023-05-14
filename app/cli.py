import typer

# from loguru import logger

__all__ = ["cli"]

cli = typer.Typer()


# @logger.catch
@cli.command()
def hello(name: str) -> None:
    """Greet user by name.

    Args:
        name (str): The user's name.
    """
    # logger.debug(f"hello called with name:{name}")
    typer.echo(f"Hello, {name}!")
