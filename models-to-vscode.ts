#!/usr/bin/env node
/**
 * Convert an OpenAI-compatible `GET /models` response into a VS Code
 * `chatLanguageModels.json` provider group (vendor "customendpoint").
 *
 * Runs on Node >= 22.18 (native TypeScript type stripping), Bun or Deno.
 *
 * Examples:
 *   curl -s -H "Authorization: Bearer $KEY" https://api.example.com/v1/models \
 *     | node models-to-vscode.ts --base-url https://api.example.com/v1 --name Example
 *
 *   node models-to-vscode.ts --fetch --base-url http://localhost:8000/v1 --name vLLM \
 *     --merge ~/.config/Code/User/chatLanguageModels.json
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { parseArgs } from "node:util";

// ---- Edit these for your provider. Command-line flags override them. ----
const BASE_URL = "https://api.example.com/v1";
const API_KEY = ""; // leave empty to use $OPENAI_API_KEY
// --------------------------------------------------------------------------

type ApiType = "chat-completions" | "responses" | "messages";

interface VSCodeModel {
  id: string;
  name: string;
  url: string;
  toolCalling: boolean;
  vision: boolean;
  maxOutputTokens: number;
  maxInputTokens?: number;
  contextWindow?: number;
  thinking?: boolean;
  apiType?: ApiType;
}

interface ProviderGroup {
  name: string;
  vendor: string;
  apiKey?: string;
  apiType?: ApiType;
  models: VSCodeModel[];
  [key: string]: unknown;
}

/** Loose shape covering fields seen across OpenAI, OpenRouter, vLLM, Groq, LiteLLM, etc. */
interface RawModel {
  id: string;
  name?: string;
  display_name?: string;
  context_length?: number;
  context_window?: number;
  max_model_len?: number;
  max_context_length?: number;
  max_tokens?: number;
  max_output_tokens?: number;
  max_completion_tokens?: number;
  top_provider?: { context_length?: number; max_completion_tokens?: number | null };
  architecture?: { input_modalities?: string[]; modality?: string };
  supported_parameters?: string[];
  capabilities?: Record<string, unknown> | string[];
  [key: string]: unknown;
}

const USAGE = `Usage: node models-to-vscode.ts [input.json] [options]

Input is read from the file argument, stdin, or fetched with --fetch.

Options:
  --base-url <url>        API base URL (default: BASE_URL in the script, ${BASE_URL})
  --name <name>           Provider group name (default: hostname of base URL)
  --fetch                 Fetch <base-url>/models instead of reading input
  --api-key <key>         Bearer token for --fetch (default: API_KEY in the script, then $OPENAI_API_KEY)
  --embed-api-key         Also write the API key into the output (stored in plain text!)
  --api-type <type>       chat-completions | responses | messages (default: chat-completions)
  --filter <regex>        Only include model ids matching this regex
  --exclude <regex>       Drop model ids matching this regex
  --max-input <n>         Fallback max input tokens  (default: 128000)
  --max-output <n>        Fallback max output tokens (default: 16000)
  --tool-calling <bool>   Fallback tool calling flag (default: true)
  --vision <bool>         Fallback vision flag       (default: false)
  --merge <file>          Merge into an existing chatLanguageModels.json (replaces group with same name)
  -o, --out <file>        Write to file instead of stdout (with --merge, defaults to the merge file)
  -h, --help              Show this help`;

function die(msg: string): never {
  console.error(`error: ${msg}`);
  process.exit(1);
}

function parseBool(v: string, flag: string): boolean {
  if (/^(true|1|yes)$/i.test(v)) return true;
  if (/^(false|0|no)$/i.test(v)) return false;
  die(`${flag} expects true/false, got "${v}"`);
}

function parseIntArg(v: string, flag: string): number {
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) die(`${flag} expects a positive integer, got "${v}"`);
  return n;
}

function firstNumber(...vals: unknown[]): number | undefined {
  for (const v of vals) if (typeof v === "number" && Number.isFinite(v) && v > 0) return v;
  return undefined;
}

function hasCapability(m: RawModel, key: string): boolean | undefined {
  const c = m.capabilities;
  if (Array.isArray(c)) return c.includes(key) ? true : undefined;
  if (c && typeof c === "object" && typeof c[key] === "boolean") return c[key] as boolean;
  return undefined;
}

async function readStdin(): Promise<string> {
  if (process.stdin.isTTY) die("no input: pass a file, pipe JSON to stdin, or use --fetch\n\n" + USAGE);
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

function extractModels(json: unknown): RawModel[] {
  // { object: "list", data: [...] }  |  { models: [...] }  |  [...]
  const list = Array.isArray(json)
    ? json
    : (json as { data?: unknown }).data ?? (json as { models?: unknown }).models;
  if (!Array.isArray(list)) die("input does not look like a /models response (expected a `data` array)");
  return list.filter((m): m is RawModel => !!m && typeof (m as RawModel).id === "string");
}

interface Defaults {
  maxInput: number;
  maxOutput: number;
  toolCalling: boolean;
  vision: boolean;
}

function convertModel(m: RawModel, chatUrl: string, d: Defaults): VSCodeModel {
  const params = m.supported_parameters;
  const modalities = m.architecture?.input_modalities ?? [];
  const modalityStr = m.architecture?.modality ?? "";

  const contextWindow = firstNumber(
    m.context_length,
    m.context_window,
    m.max_model_len,
    m.max_context_length,
    m.top_provider?.context_length,
  );
  let maxOutput =
    firstNumber(
      m.max_output_tokens,
      m.max_completion_tokens,
      m.top_provider?.max_completion_tokens,
      // Some servers use max_tokens for output limit; only trust it if smaller than the context window.
      contextWindow && m.max_tokens && m.max_tokens < contextWindow ? m.max_tokens : undefined,
    ) ?? d.maxOutput;
  if (contextWindow && maxOutput >= contextWindow) maxOutput = Math.floor(contextWindow / 4);

  const toolCalling =
    (params ? params.includes("tools") : undefined) ??
    hasCapability(m, "tool_calling") ??
    hasCapability(m, "function_calling") ??
    hasCapability(m, "tools") ??
    d.toolCalling;

  const vision =
    (modalities.length ? modalities.includes("image") : undefined) ??
    (modalityStr ? /image/.test(modalityStr.split("->")[0]) : undefined) ??
    hasCapability(m, "vision") ??
    d.vision;

  const thinking =
    (params ? params.includes("reasoning") || params.includes("include_reasoning") : undefined) ??
    hasCapability(m, "reasoning") ??
    hasCapability(m, "thinking");

  const out: VSCodeModel = {
    id: m.id,
    name: (typeof m.name === "string" && m.name) || (typeof m.display_name === "string" && m.display_name) || m.id,
    url: chatUrl,
    toolCalling,
    vision,
    maxOutputTokens: maxOutput,
  };
  if (contextWindow) out.contextWindow = contextWindow;
  else out.maxInputTokens = d.maxInput;
  if (thinking) out.thinking = true;
  return out;
}

async function main(): Promise<void> {
  const { values: o, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      "base-url": { type: "string" },
      name: { type: "string" },
      fetch: { type: "boolean", default: false },
      "api-key": { type: "string" },
      "embed-api-key": { type: "boolean", default: false },
      "api-type": { type: "string", default: "chat-completions" },
      filter: { type: "string" },
      exclude: { type: "string" },
      "max-input": { type: "string", default: "128000" },
      "max-output": { type: "string", default: "16000" },
      "tool-calling": { type: "string", default: "true" },
      vision: { type: "string", default: "false" },
      merge: { type: "string" },
      out: { type: "string", short: "o" },
      help: { type: "boolean", short: "h", default: false },
    },
  });

  if (o.help) {
    console.log(USAGE);
    return;
  }
  const baseUrl = (o["base-url"] ?? BASE_URL).replace(/\/+$/, "");
  let host: string;
  try {
    host = new URL(baseUrl).host;
  } catch {
    die(`invalid --base-url "${baseUrl}"`);
  }

  const apiType = o["api-type"] as ApiType;
  const endpointPath: Record<ApiType, string> = {
    "chat-completions": "/chat/completions",
    responses: "/responses",
    messages: "/messages",
  };
  if (!(apiType in endpointPath)) die(`--api-type must be one of ${Object.keys(endpointPath).join(", ")}`);
  const chatUrl = baseUrl + endpointPath[apiType];

  const apiKey = o["api-key"] || API_KEY || process.env.OPENAI_API_KEY;

  // --- read input ---
  let raw: string;
  if (o.fetch) {
    const res = await fetch(`${baseUrl}/models`, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    });
    if (!res.ok) die(`GET ${baseUrl}/models -> ${res.status} ${res.statusText}`);
    raw = await res.text();
  } else if (positionals[0]) {
    raw = readFileSync(positionals[0], "utf8");
  } else {
    raw = await readStdin();
  }

  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (e) {
    die(`input is not valid JSON: ${(e as Error).message}`);
  }

  // --- convert ---
  const include = o.filter ? new RegExp(o.filter) : undefined;
  const exclude = o.exclude ? new RegExp(o.exclude) : undefined;
  const defaults: Defaults = {
    maxInput: parseIntArg(o["max-input"]!, "--max-input"),
    maxOutput: parseIntArg(o["max-output"]!, "--max-output"),
    toolCalling: parseBool(o["tool-calling"]!, "--tool-calling"),
    vision: parseBool(o.vision!, "--vision"),
  };

  const models = extractModels(json)
    .filter((m) => (!include || include.test(m.id)) && (!exclude || !exclude.test(m.id)))
    .map((m) => convertModel(m, chatUrl, defaults))
    .sort((a, b) => a.id.localeCompare(b.id));

  if (models.length === 0) console.error("warning: no models matched");

  const group: ProviderGroup = { name: o.name ?? host, vendor: "customendpoint", models };
  if (apiType !== "chat-completions") group.apiType = apiType;
  if (o["embed-api-key"]) {
    if (!apiKey) die("--embed-api-key given but no key (use --api-key or $OPENAI_API_KEY)");
    group.apiKey = apiKey;
  }

  // --- merge / output ---
  let result: ProviderGroup[] = [group];
  if (o.merge) {
    const existing: ProviderGroup[] = existsSync(o.merge)
      ? JSON.parse(readFileSync(o.merge, "utf8").trim() || "[]")
      : [];
    if (!Array.isArray(existing)) die(`${o.merge} is not a JSON array`);
    const idx = existing.findIndex((g) => g.name === group.name && g.vendor === group.vendor);
    if (idx >= 0) {
      // Keep any settings we don't manage (apiKey reference, custom flags) from the old group.
      const { models: _old, ...rest } = existing[idx];
      existing[idx] = { ...rest, ...group };
    } else {
      existing.push(group);
    }
    result = existing;
  }

  const text = JSON.stringify(result, null, "\t") + "\n";
  const outFile = o.out ?? o.merge;
  if (outFile) {
    writeFileSync(outFile, text);
    console.error(`wrote ${models.length} model(s) to ${outFile} (group "${group.name}")`);
  } else {
    process.stdout.write(text);
  }
}

main().catch((e) => die((e as Error).message));
