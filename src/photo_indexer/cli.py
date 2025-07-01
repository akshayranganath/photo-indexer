"""
photo_indexer.cli
~~~~~~~~~~~~~~~~~

Command-line interface for *Photo-Indexer*.

The CLI is a thin wrapper that

1. Boots the global logging system.
2. Parses user options (via **Click**).
3. Delegates heavy lifting to :pymod:`photo_indexer.workers`.

You get a single sub-command:

    $ pi index /path/to/RAWs  [OPTIONS]

which walks every *.NEF*, runs the full vision pipeline, and stores the
results in the chosen database backend.

Captioning supports both local (Ollama) and remote (OpenAI) providers:

    $ pi index /photos --caption-provider ollama  # Uses llama3.2-vision:latest
    $ pi index /photos --caption-provider openai  # Uses gpt-4o (requires API key)

The entry-point name **`pi`** is registered in *pyproject.toml* so it
becomes available after

    $ pip install -e .

or

    $ poetry install
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click

from photo_indexer.utils.logging import get_logger #,setup
from photo_indexer.config import IndexerSettings

# -- lazy import to avoid torch startup when showing --help -------------------
def _lazy_worker_import():
    #from photo_indexer.workers import run_index  # local import to defer heavy deps    
    from photo_indexer.workers import index_folder

    return index_folder


_log = get_logger(__name__)


# ---------------------------------------------------------------------------#
# Click helpers                                                               #
# ---------------------------------------------------------------------------#
class _PathExists(click.Path):
    """Click Path subtype that enforces *exists=True* by default."""

    def __init__(self, **kwargs):
        super().__init__(exists=True, file_okay=True, dir_okay=True, readable=True, **kwargs)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli() -> None:  # pragma: no cover
    """Photo-Indexer command-line tool."""
    # The group does nothing by itself; sub-commands below do the work.


@cli.command("index", help="Index all .NEF photos under PHOTO_ROOT.")
@click.argument("photo_root", type=_PathExists(path_type=Path))
@click.option(
    "--workers",
    "-w",
    type=int,
    metavar="N",
    default=os.cpu_count(),
    show_default="CPU core count",
    help="Concurrent worker threads.",
)
@click.option(
    "--db",
    type=click.Choice(["sqlite", "duckdb"], case_sensitive=False),
    default="sqlite",
    show_default=True,
    help="Storage backend.",
)
@click.option(
    "--thumb-size",
    type=int,
    default=512,
    show_default=True,
    help="Longest edge of cached JPEG thumbnails (pixels).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable DEBUG-level logging.",
)
@click.option(
    "--caption-provider",
    type=click.Choice(["ollama", "openai"], case_sensitive=False),
    default=None,
    help="Captioning provider: 'ollama' (local) or 'openai' (remote).",
)
@click.option(
    "--caption-model",
    type=str,
    default=None,
    help="Caption model name. Defaults: 'llama3.2-vision:latest' (Ollama) or 'gpt-4o' (OpenAI).",
)
@click.option(
    "--ollama-host",
    type=str,
    default=None,
    help="Ollama host URL (default: http://localhost:11434).",
)
@click.option(
    "--openai-api-key",
    type=str,
    default=None,
    help="OpenAI API key (overrides OPENAI_API_KEY env var).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run pipeline but skip final DB insert (for timing tests).",
)
def cmd_index(
    photo_root: Path,
    workers: int,
    db: str,
    thumb_size: int,
    caption_provider: str | None,
    caption_model: str | None,
    ollama_host: str | None,
    openai_api_key: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """
    Walk PHOTO_ROOT recursively, process every *.NEF* and write results to
    a local database (default *data/db/photo_index.sqlite*).
    
    Captioning can use either local Ollama models or remote OpenAI models:
    
    \b
    # Use local Ollama with default model (llama3.2-vision:latest)
    pi index /photos --caption-provider ollama
    
    \b  
    # Use OpenAI with default model (gpt-4o) - requires API key
    pi index /photos --caption-provider openai --openai-api-key sk-...
    
    \b
    # Override specific models
    pi index /photos --caption-provider openai --caption-model gpt-4-vision-preview
    
    Settings can also be configured via ~/.config/photo_indexer/config.yaml
    """
    #setup(verbose=verbose)

    _log.info("Photo-Indexer starting (root=%s, workers=%d)", photo_root, workers)

    index_folder = _lazy_worker_import()

    try:
        # Map database backend to file path
        db_file = f"data/db/photo_index.{db.lower()}"
        
        # Load base settings from config file (if exists)
        from photo_indexer.config import load_config
        base_settings = load_config()
        
        # Override with CLI parameters
        settings_kwargs = {
            "workers": workers,
            "db_path": Path(db_file),
            "thumbnail_size": thumb_size,
        }
        
        # Override caption provider settings if provided
        if caption_provider is not None:
            provider_choice = caption_provider.lower()
            settings_kwargs["caption_provider"] = provider_choice
            
            # Set appropriate default model for the chosen provider if not explicitly specified
            if caption_model is not None:
                settings_kwargs["caption_model"] = caption_model
            else:
                # Use provider-specific defaults when provider is explicitly chosen
                if provider_choice == "openai":
                    settings_kwargs["caption_model"] = "gpt-4o"
                elif provider_choice == "ollama":
                    settings_kwargs["caption_model"] = "llama3.2-vision:latest"
                else:
                    settings_kwargs["caption_model"] = base_settings.caption_model
        else:
            # Use config file settings when no provider specified
            settings_kwargs["caption_provider"] = base_settings.caption_provider
            if caption_model is not None:
                settings_kwargs["caption_model"] = caption_model
            else:
                settings_kwargs["caption_model"] = base_settings.caption_model
            
        if ollama_host is not None:
            settings_kwargs["ollama_host"] = ollama_host
        else:
            settings_kwargs["ollama_host"] = base_settings.ollama_host
            
        if openai_api_key is not None:
            settings_kwargs["openai_api_key"] = openai_api_key
        else:
            settings_kwargs["openai_api_key"] = base_settings.openai_api_key
        
        # Copy other caption settings from base config
        settings_kwargs.update({
            "caption_prompt": base_settings.caption_prompt,
            "caption_temperature": base_settings.caption_temperature,
            "caption_max_tokens": base_settings.caption_max_tokens,
            "openai_base_url": base_settings.openai_base_url,
            "scene_model": base_settings.scene_model,
            "people_model": base_settings.people_model,
        })
        
        settings = IndexerSettings(**settings_kwargs)
        
        # Validate OpenAI configuration
        if settings.caption_provider == "openai":
            import os
            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                _log.error("OpenAI provider requires an API key. Provide via --openai-api-key or OPENAI_API_KEY environment variable.")
                sys.exit(1)
        
        # Log the caption provider being used
        model_source = "user-specified" if caption_model else "default"
        _log.info("Using caption provider: %s (model: %s, %s)", 
                 settings.caption_provider, settings.caption_model, model_source)
        
        index_folder(photo_root, settings=settings, dry_run=dry_run)
    except KeyboardInterrupt:
        _log.warning("Interrupted by user – exiting.")
        sys.exit(130)
    except Exception as exc:  # pylint: disable=broad-except
        _log.exception("Fatal error: %s", exc)
        sys.exit(1)

    _log.info("Done – bye.")


@cli.command("ui", help="Launch the web-based photo search interface.")
@click.option(
    "--port",
    "-p",
    type=int,
    default=8501,
    show_default=True,
    help="Port for the Streamlit web server.",
)
@click.option(
    "--host",
    type=str,
    default="localhost",
    show_default=True,
    help="Host address for the Streamlit web server.",
)
def cmd_ui(port: int, host: str) -> None:
    """
    Launch the Streamlit web interface for searching and browsing indexed photos.
    
    The UI provides:
    - Full-text search across photo descriptions, locations, and scenes
    - Grid-based photo browsing with thumbnails
    - Photo metadata display (date, people count, etc.)
    - Database statistics and management
    
    \b
    Example usage:
    pi ui                    # Launch on default port 8501
    pi ui --port 8080        # Launch on custom port
    pi ui --host 0.0.0.0     # Allow external connections
    
    Once running, open your browser to http://localhost:8501 (or your chosen host/port).
    """
    import subprocess
    import sys
    from pathlib import Path
    
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "ui" / "app.py"
    
    if not app_path.exists():
        _log.error("UI app not found at %s", app_path)
        sys.exit(1)
    
    _log.info("Starting Photo-Indexer web UI on %s:%d", host, port)
    _log.info("Open your browser to: http://%s:%d", host, port)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        _log.error("Failed to start Streamlit: %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        _log.info("Web UI stopped by user.")
    except Exception as exc:
        _log.exception("Error running web UI: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------#
# Stand-alone invocation (python -m photo_indexer.cli)                        #
# ---------------------------------------------------------------------------#
def main() -> None:  # pragma: no cover
    """Module-level entry-point so `python -m photo_indexer.cli …` works."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
