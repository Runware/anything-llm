const OpenAI = require("openai");
const Provider = require("./ai-provider.js");
const InheritMultiple = require("./helpers/classes.js");
const UnTooled = require("./helpers/untooled.js");
const {
  tooledStream,
  tooledComplete,
  temperatureParam,
} = require("./helpers/tooled.js");
const { RetryError } = require("../error.js");

const DEFAULT_BASE_URL = "https://api.runware.ai/v1";

/**
 * The agent provider for the Runware provider.
 * Supports true OpenAI-compatible tool calling when enabled via ENV,
 * falling back to the UnTooled prompt-based approach otherwise.
 */
class RunwareProvider extends InheritMultiple([Provider, UnTooled]) {
  model;

  constructor(config = {}) {
    const { model = "openai-gpt-5-mini" } = config;
    super();
    this.providerTag = "runware";
    const client = new OpenAI({
      baseURL: process.env.RUNWARE_BASE_URL || DEFAULT_BASE_URL,
      apiKey: process.env.RUNWARE_API_KEY ?? null,
    });

    this._client = client;
    this.model = model;
    this.verbose = true;
  }

  get client() {
    return this._client;
  }

  get supportsAgentStreaming() {
    return true;
  }

  async #handleFunctionCallChat({ messages = [] }) {
    return await this.client.chat.completions
      .create({
        model: this.model,
        ...temperatureParam(this.temperature),
        messages,
      })
      .then((result) => {
        if (!result.hasOwnProperty("choices"))
          throw new Error("Runware chat: No results!");
        if (result.choices.length === 0)
          throw new Error("Runware chat: No results length!");
        return result.choices[0].message.content;
      })
      .catch((_) => {
        return null;
      });
  }

  async #handleFunctionCallStream({ messages = [] }) {
    return await this.client.chat.completions.create({
      model: this.model,
      ...temperatureParam(this.temperature),
      stream: true,
      messages,
    });
  }

  /**
   * Stream a chat completion with tool calling support.
   * Uses native tool calling when enabled via ENV, otherwise falls back to UnTooled.
   */
  async stream(messages, functions = [], eventHandler = null) {
    const useNative = this.supportsNativeToolCalling();

    if (!useNative) {
      return await UnTooled.prototype.stream.call(
        this,
        messages,
        functions,
        this.#handleFunctionCallStream.bind(this),
        eventHandler
      );
    }

    this.providerLog(
      "Provider.stream (tooled) - will process this chat completion."
    );

    try {
      return await tooledStream(
        this.client,
        this.model,
        messages,
        functions,
        eventHandler,
        { provider: this }
      );
    } catch (error) {
      console.error(error.message, error);
      if (error instanceof OpenAI.AuthenticationError) throw error;
      if (
        error instanceof OpenAI.RateLimitError ||
        error instanceof OpenAI.InternalServerError ||
        error instanceof OpenAI.APIError
      ) {
        throw new RetryError(error.message);
      }
      throw error;
    }
  }

  /**
   * Create a non-streaming completion with tool calling support.
   * Uses native tool calling when enabled via ENV, otherwise falls back to UnTooled.
   */
  async complete(messages, functions = []) {
    const useNative = this.supportsNativeToolCalling();

    if (!useNative) {
      return await UnTooled.prototype.complete.call(
        this,
        messages,
        functions,
        this.#handleFunctionCallChat.bind(this)
      );
    }

    try {
      const result = await tooledComplete(
        this.client,
        this.model,
        messages,
        functions,
        this.getCost.bind(this),
        { provider: this }
      );

      if (result.retryWithError) {
        return this.complete([...messages, result.retryWithError], functions);
      }

      return result;
    } catch (error) {
      if (error instanceof OpenAI.AuthenticationError) throw error;
      if (
        error instanceof OpenAI.RateLimitError ||
        error instanceof OpenAI.InternalServerError ||
        error instanceof OpenAI.APIError
      ) {
        throw new RetryError(error.message);
      }
      throw error;
    }
  }

  /**
   * Get the cost of the completion.
   * @param _usage The completion to get the cost for.
   * @returns The cost of the completion.
   */
  getCost(_usage) {
    return 0;
  }
}

module.exports = RunwareProvider;
