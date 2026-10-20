const { NativeEmbedder } = require("../../EmbeddingEngines/native");
const {
  LLMPerformanceMonitor,
} = require("../../helpers/chat/LLMPerformanceMonitor");
const {
  handleDefaultStreamResponseV2,
  formatChatHistory,
} = require("../../helpers/chat/responses");
const {
  temperatureParam,
} = require("../../agents/aibitat/providers/helpers/tooled");
const { MODEL_MAP } = require("../modelMap");

const DEFAULT_BASE_URL = "https://api.runware.ai/v1";

class RunwareLLM {
  constructor(embedder = null, modelPreference = null) {
    const { OpenAI: OpenAIApi } = require("openai");
    if (!process.env.RUNWARE_API_KEY)
      throw new Error("No Runware API key was set.");

    this.className = "RunwareLLM";
    this.openai = new OpenAIApi({
      baseURL: process.env.RUNWARE_BASE_URL || DEFAULT_BASE_URL,
      apiKey: process.env.RUNWARE_API_KEY,
    });
    this.model =
      modelPreference || process.env.RUNWARE_MODEL_PREF || "openai-gpt-5-mini";
    this.limits = {
      history: this.promptWindowLimit() * 0.15,
      system: this.promptWindowLimit() * 0.15,
      user: this.promptWindowLimit() * 0.7,
    };

    this.embedder = embedder ?? new NativeEmbedder();
    this.#log(`Initialized with model: ${this.model}`);
  }

  #log(text, ...args) {
    console.log(`\x1b[36m[${this.className}]\x1b[0m ${text}`, ...args);
  }

  #appendContext(contextTexts = []) {
    if (!contextTexts || !contextTexts.length) return "";
    return (
      "\nContext:\n" +
      contextTexts
        .map((text, i) => {
          return `[CONTEXT ${i}]:\n${text}\n[END CONTEXT ${i}]\n\n`;
        })
        .join("")
    );
  }

  streamingEnabled() {
    return "streamGetChatCompletion" in this;
  }

  static promptWindowLimit(modelName) {
    return (
      Number(process.env.RUNWARE_MODEL_TOKEN_LIMIT) ||
      MODEL_MAP.get("runware", modelName) ||
      128000
    );
  }

  promptWindowLimit() {
    return this.constructor.promptWindowLimit(this.model);
  }

  // The Runware inference API is OpenAI-compatible, so any model the account
  // has access to can be used. The available models are surfaced to the UI via
  // the customModels helper, so we only need to ensure a model is present here.
  async isValidChatCompletionModel(_modelName = "") {
    return true;
  }

  /**
   * Generates appropriate content array for a message + attachments.
   * @param {{userPrompt:string, attachments: import("../../helpers").Attachment[]}}
   * @returns {string|object[]}
   */
  #generateContent({ userPrompt, attachments: _attachments = [] }) {
    return userPrompt;
  }

  constructPrompt({
    systemPrompt = "",
    contextTexts = [],
    chatHistory = [],
    userPrompt = "",
    attachments = [],
  }) {
    const prompt = {
      role: "system",
      content: `${systemPrompt}${this.#appendContext(contextTexts)}`,
    };
    return [
      prompt,
      ...formatChatHistory(chatHistory, this.#generateContent),
      {
        role: "user",
        content: this.#generateContent({ userPrompt, attachments }),
      },
    ];
  }

  /**
   * Parses and prepends reasoning from the response and returns the full text response.
   * Used for getChatCompletions to render thinking text if present in full response.
   * @param {Object} message - The message object from the Runware response.
   * @returns {string}
   */
  #parseReasoningFromResponse({ message }) {
    let textResponse = message?.content ?? "";
    if (
      !!message?.reasoning_content &&
      message.reasoning_content.trim().length > 0
    )
      textResponse = ` thinking${message.reasoning_content}${textResponse}`;
    return textResponse;
  }

  async getChatCompletion(
    messages = null,
    { temperature = this.temperature } = {}
  ) {
    const result = await LLMPerformanceMonitor.measureAsyncFunction(
      this.openai.chat.completions
        .create({
          model: this.model,
          messages,
          ...temperatureParam(temperature),
        })
        .catch((e) => {
          console.error(e);
          throw new Error(e.message);
        })
    );

    if (
      !result.output.hasOwnProperty("choices") ||
      result.output.choices.length === 0
    )
      return null;

    return {
      textResponse: this.#parseReasoningFromResponse(result.output.choices[0]),
      metrics: {
        prompt_tokens: result.output.usage.prompt_tokens || 0,
        completion_tokens: result.output.usage.completion_tokens || 0,
        total_tokens: result.output.usage.total_tokens || 0,
        outputTps: result.output.usage.completion_tokens / result.duration,
        duration: result.duration,
        model: this.model,
        provider: this.className,
        timestamp: new Date(),
      },
    };
  }

  async streamGetChatCompletion(
    messages = null,
    { temperature = this.temperature } = {}
  ) {
    const measuredStreamRequest = await LLMPerformanceMonitor.measureStream({
      func: this.openai.chat.completions.create({
        model: this.model,
        stream: true,
        messages,
        ...temperatureParam(temperature),
      }),
      messages,
      runPromptTokenCalculation: false,
      modelTag: this.model,
      provider: this.className,
    });
    return measuredStreamRequest;
  }

  handleStream(response, stream, responseProps) {
    return handleDefaultStreamResponseV2(response, stream, responseProps);
  }

  // Simple wrapper for dynamic embedder & normalize interface for all LLM implementations
  async embedTextInput(textInput) {
    return await this.embedder.embedTextInput(textInput);
  }
  async embedChunks(textChunks = []) {
    return await this.embedder.embedChunks(textChunks);
  }

  async compressMessages(promptArgs = {}, rawHistory = []) {
    const { messageArrayCompressor } = require("../../helpers/chat");
    const messageArray = this.constructPrompt(promptArgs);
    return await messageArrayCompressor(this, messageArray, rawHistory);
  }
}

module.exports = {
  RunwareLLM,
};
