/**
 * J7Scope J-Space plugin for opencode.
 *
 * The actual read-out is proxied by `j7scope-serve` — opencode only needs to
 * route generation through it (see ../opencode.json). This plugin is a small
 * convenience layer: on startup it checks the sidecar is up, prints where the
 * live J-Space viewer lives, and (optionally) opens it in the browser so the
 * workspace read-out is visible next to the session.
 *
 * Install: copy this file to `.opencode/plugin/j7scope.ts` in your project
 * (or `~/.config/opencode/plugin/`). opencode loads it automatically.
 *
 * Env:
 *   J7SCOPE_SIDECAR   base URL of the sidecar (default http://127.0.0.1:8799)
 *   J7SCOPE_OPEN      "1" to auto-open the viewer in the default browser
 */

import type { Plugin } from "@opencode-ai/plugin";

const SIDECAR = process.env.J7SCOPE_SIDECAR ?? "http://127.0.0.1:8799";

export const J7Scope: Plugin = async ({ $, client }) => {
  let announced = false;

  async function announce() {
    if (announced) return;
    announced = true;

    let health: any = null;
    try {
      const res = await fetch(`${SIDECAR}/health`);
      health = await res.json();
    } catch {
      console.warn(
        `[j7scope] sidecar not reachable at ${SIDECAR}. ` +
          `Start it with:  python -m j7scope_serve --backend mock`
      );
      return;
    }

    const tag = health?.is_demo ? " (DEMO — synthetic read-out)" : "";
    console.log(
      `[j7scope] live J-Space viewer${tag}: ${SIDECAR}/  ` +
        `· model=${health?.model} layer=${health?.layer}`
    );

    // A toast is nicer than a log line, but the TUI client surface varies by
    // opencode version — feature-detect and fail quiet.
    try {
      await (client as any)?.tui?.showToast?.({
        message: `J-Space live at ${SIDECAR}/`,
        variant: "info",
      });
    } catch {
      /* older opencode: log line above is the fallback */
    }

    if (process.env.J7SCOPE_OPEN === "1") {
      const opener =
        process.platform === "darwin"
          ? "open"
          : process.platform === "win32"
          ? "start"
          : "xdg-open";
      try {
        await $`${opener} ${SIDECAR}/`;
      } catch {
        /* headless / no browser: viewer URL is logged above */
      }
    }
  }

  return {
    // Fires on every opencode event; we only need the first one to announce.
    event: async () => {
      await announce();
    },
  };
};
