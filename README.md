# everos-cc-plugin

Self-evolving long-term memory for Claude Code: a Claude Code plugin (`.claude-plugin/`,
`hooks/`) paired with a vendored, pruned copy of the [EverOS](https://github.com/EverMind-AI/EverOS)
memory engine (`src/everos/`) — no demos/tests/docs from upstream, just the code needed to
run the engine and keep developing it in place.

Two independent pieces live in this one folder:

- **Engine** (`src/everos/` + `pyproject.toml`) — a `pip`-installable Python package. Runs as
  its own persistent HTTP server process (`everos server start`); it is not a library you
  import and call in-process.
- **Plugin** (`.claude-plugin/`, `hooks/`) — thin HTTP client hooks (stdlib only, no
  third-party dependencies) that call the running engine on `UserPromptSubmit` (recall),
  `Stop` (deferred capture), and `SessionEnd` (flush).

## Starting the engine

```bash
pip install -e .                          # from this directory
everos init --root /path/to/everos_root   # generates everos.toml
# edit /path/to/everos_root/everos.toml — fill in [llm] and [embedding]
# (model / api_key / base_url), both required
everos server start --root /path/to/everos_root --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health         # expect {"status": "ok", ...}
```

## Installing the plugin (point it at the running engine)

```bash
/plugin marketplace add /nas/zpf/memory/code/Muse_cc/everos_cc_plugin
/plugin install everos-memory@everos-cc-plugin-marketplace --config baseUrl=http://127.0.0.1:8000
```

Remaining `userConfig` fields (recall method, flush thresholds, etc.) have working defaults —
see `.claude-plugin/plugin.json`.

## License

`src/everos/` is EverMind AI's EverOS, Apache-2.0 (see `LICENSE`, `NOTICE`).
