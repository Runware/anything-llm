const { RunwareLLM } = require("../../../../utils/AiProviders/runware");
const RunwareProvider = require("../../../../utils/agents/aibitat/providers/runware.js");

const ORIGINAL_ENV = process.env;

beforeEach(() => {
  process.env = {
    ...ORIGINAL_ENV,
    RUNWARE_API_KEY: "test-key",
    RUNWARE_MODEL_PREF: "openai-gpt-5-mini",
  };
});

afterEach(() => {
  process.env = ORIGINAL_ENV;
  jest.restoreAllMocks();
});

describe("RunwareLLM", () => {
  it("throws when no API key is set", () => {
    delete process.env.RUNWARE_API_KEY;
    expect(() => new RunwareLLM()).toThrow("No Runware API key was set.");
  });

  it("defaults the base URL to the Runware API", () => {
    const llm = new RunwareLLM();
    expect(llm.openai.baseURL).toBe("https://api.runware.ai/v1");
  });

  it("honors RUNWARE_BASE_URL for proxies", () => {
    process.env.RUNWARE_BASE_URL = "http://localhost:9999/v1";
    const llm = new RunwareLLM();
    expect(llm.openai.baseURL).toBe("http://localhost:9999/v1");
  });

  it("falls back to a sane default context window", () => {
    const llm = new RunwareLLM(null, "openai-gpt-5-mini");
    expect(llm.promptWindowLimit()).toBe(128000);
  });

  it("honors RUNWARE_MODEL_TOKEN_LIMIT when set", () => {
    process.env.RUNWARE_MODEL_TOKEN_LIMIT = "65536";
    const llm = new RunwareLLM(null, "openai-gpt-5-mini");
    expect(llm.promptWindowLimit()).toBe(65536);
  });

  it("constructs an OpenAI-style prompt with context", () => {
    const llm = new RunwareLLM();
    const messages = llm.constructPrompt({
      systemPrompt: "You are helpful.",
      contextTexts: ["some context"],
      userPrompt: "hello",
    });
    expect(messages[0].role).toBe("system");
    expect(messages[0].content).toContain("You are helpful.");
    expect(messages[0].content).toContain("some context");
    expect(messages[messages.length - 1]).toEqual({
      role: "user",
      content: "hello",
    });
  });

  it("prepends reasoning_content to the text response", async () => {
    const llm = new RunwareLLM();
    const output = {
      choices: [
        { message: { content: "answer", reasoning_content: "thinking" } },
      ],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    };
    const create = jest.fn(() =>
      Object.assign(Promise.resolve(output), {
        catch: () => Promise.resolve(output),
      })
    );
    llm.openai = { chat: { completions: { create } } };

    const result = await llm.getChatCompletion([
      { role: "user", content: "hi" },
    ]);
    expect(result.textResponse).toBe(" thinkingthinkinganswer");
  });
});

describe("RunwareProvider", () => {
  it("uses the Runware base URL and API key", () => {
    const provider = new RunwareProvider({ model: "openai-gpt-5-mini" });
    expect(provider.client.baseURL).toBe("https://api.runware.ai/v1");
    expect(provider.model).toBe("openai-gpt-5-mini");
  });

  it("honors RUNWARE_BASE_URL for proxies", () => {
    process.env.RUNWARE_BASE_URL = "http://localhost:9999/v1";
    const provider = new RunwareProvider({ model: "openai-gpt-5-mini" });
    expect(provider.client.baseURL).toBe("http://localhost:9999/v1");
  });
});
