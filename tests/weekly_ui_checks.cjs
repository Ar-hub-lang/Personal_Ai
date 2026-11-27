// Exercise the app's actual event handlers with a small DOM stub and the isolated FastAPI server.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const [baseURL, scenario, model] = process.argv.slice(2);
const app = fs.readFileSync("frontend/app.js", "utf8");
const page = fs.readFileSync("frontend/index.html", "utf8");
const options = [...page.matchAll(/<option value="([^"]*)">([^<]*)<\/option>/g)]
  .map(([, value, text]) => ({ value, text }));
class Element {
  constructor() {
    this.value = "";
    this.dataset = {};
    this.handlers = {};
    this.children = [];
  }
  addEventListener(type, callback) { this.handlers[type] = callback; }
  setAttribute() {}
  append(...children) {
    for (const child of children) if (child instanceof Element) child.parent = this;
    this.children.push(...children);
  }
  querySelector() { return this.children[1]; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
  }
  replaceChildren() { this.children = []; }
}

const elements = new Map();
const document = {
  querySelector(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement: () => new Element(),
  createTextNode: (value) => value,
};
const select = (id) => document.querySelector(id);
select("#model").options = options;
let approve = false;
let confirmations = 0;
const calls = [];
const pending = [];
let sessionCookie = "";
function request(url, options) {
  const body = typeof options?.body === "string" ? JSON.parse(options.body) : null;
  calls.push({ url, method: options?.method, body });
  const headers = new Headers(options?.headers);
  if (sessionCookie) headers.set("Cookie", sessionCookie);
  const promise = fetch(baseURL + url, { ...options, headers }).then((response) => {
    const cookie = response.headers.get("set-cookie");
    if (cookie) sessionCookie = cookie.split(";")[0];
    return response;
  });
  pending.push(promise);
  return promise;
}
const context = vm.createContext({
  document,
  window: { document, confirm: () => { confirmations += 1; return approve; } },
  fetch: request,
  TextDecoderStream,
  AbortController,
  URLSearchParams,
});
vm.runInContext(app, context);

async function main() {
  await Promise.all(pending);
  for (let attempt = 0; attempt < 30 && select("#send").disabled !== false; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal(select("#new-conversation").hidden, false, "IT Support mode did not load");

  if (scenario === "switch") {
    const prompts = [
      "What is the FableDesk VPN gateway hostname?",
      "Which operating systems does it support?",
      "What is the FableDesk VPN gateway hostname?",
    ];
    const sequence = ["qwen3:1.7b", "qwen3:4b", "qwen3:1.7b"];
    for (const [index, prompt] of prompts.entries()) {
      select("#model").value = sequence[index];
      select("#prompt").value = prompt;
      await select("#chat-form").handlers.submit({ preventDefault() {} });
      assert.notEqual(select("#chat-status").dataset.state, "error", select("#chat-status").textContent);
    }
    const chat = calls.filter((call) => call.url === "/documents/ask/stream");
    assert.deepEqual(chat.map((call) => call.body.model), sequence);
    assert.match(chat[1].body.previous_question, /FableDesk VPN gateway/);
    const replies = select("#messages").children.filter((message) => message.dataset.speaker === "Assistant");
    assert.equal(replies.length, 3, "Earlier messages did not remain in the current tab");
    assert.match(replies[0].children[1].textContent, /vpn\.fabledesk\.invalid/i);
    assert.match(replies[1].children[1].textContent, /macOS.*Windows|Windows.*macOS/i);
    assert.match(replies[2].children[1].textContent, /vpn\.fabledesk\.invalid/i);
  } else if (scenario === "permission") {
    const file = new Blob(["The test workstation inventory code is BIRCH-402."], { type: "text/plain" });
    file.name = "weekly-approval.txt";
    select("#pdf-file").files = [file];
    await select("#pdf-file").handlers.change();
    assert.equal(confirmations, 1, "Add guide did not request confirmation");
    assert.equal(calls.filter((call) => call.method === "POST").length, 0, "Declining sent an upload");
    approve = true;
    await select("#pdf-file").handlers.change();
    assert.equal(confirmations, 2);
    assert.equal(calls.filter((call) => call.method === "POST").length, 1);
    const catalog = await fetch(baseURL + "/documents").then((response) => response.json());
    assert.ok(catalog.documents.some((guide) => guide.filename === file.name), select("#pdf-status").textContent);
  } else {
    throw new Error("Unknown scenario " + scenario);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
