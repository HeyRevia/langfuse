import { type ZodSchema } from "zod";

import { ChatAnthropic } from "@langchain/anthropic";
import { ChatVertexAI } from "@langchain/google-vertexai";
import { ChatBedrockConverse } from "@langchain/aws";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  BytesOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { IterableReadableStream } from "@langchain/core/utils/stream";
import { ChatOpenAI } from "@langchain/openai";
import GCPServiceAccountKeySchema, {
  BedrockConfigSchema,
  BedrockCredentialSchema,
} from "../../interfaces/customLLMProviderConfigSchemas";
import { processEventBatch } from "../ingestion/processEventBatch";
import { ingestionEvent, eventTypes, TraceBody } from "../ingestion/types";
import { logger } from "../logger";
import {
  ChatMessage,
  ChatMessageRole,
  ChatMessageType,
  LLMAdapter,
  LLMJSONSchema,
  LLMToolCall,
  LLMToolChunkConfig,
  LLMToolChunkByPropertyCountConfig,
  LLMToolChunkByPropertyGroupConfig,
  LLMToolChunkByPropertyCountConfigSchema,
  LLMToolChunkByPropertyGroupConfigSchema,
  LLMToolDefinition,
  ModelParams,
  ToolCallResponse,
  ToolCallResponseSchema,
  TraceParams,
  LLMToolCallChunkPrefix,
} from "./types";
import { CallbackHandler } from "langfuse-langchain";
import type { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { z } from "zod";

type ProcessTracedEvents = () => Promise<void>;

type LLMCompletionParams = {
  messages: ChatMessage[];
  modelParams: ModelParams;
  structuredOutputSchema?: ZodSchema | LLMJSONSchema;
  callbacks?: BaseCallbackHandler[];
  baseURL?: string;
  apiKey: string;
  extraHeaders?: Record<string, string>;
  maxRetries?: number;
  config?: Record<string, string> | null;
  traceParams?: TraceParams;
  throwOnError?: boolean; // default is true
};

type FetchLLMCompletionParams = LLMCompletionParams & {
  streaming: boolean;
  tools?: LLMToolDefinition[];
};

export async function fetchLLMCompletion(
  params: LLMCompletionParams & {
    streaming: true;
  },
): Promise<{
  completion: IterableReadableStream<Uint8Array>;
  processTracedEvents: ProcessTracedEvents;
}>;

export async function fetchLLMCompletion(
  params: LLMCompletionParams & {
    streaming: false;
  },
): Promise<{
  completion: string;
  processTracedEvents: ProcessTracedEvents;
}>;

export async function fetchLLMCompletion(
  params: LLMCompletionParams & {
    streaming: false;
    structuredOutputSchema: ZodSchema;
  },
): Promise<{
  completion: Record<string, unknown>;
  processTracedEvents: ProcessTracedEvents;
}>;

export async function fetchLLMCompletion(
  params: LLMCompletionParams & {
    tools: LLMToolDefinition[];
    streaming: false;
  },
): Promise<{
  completion: ToolCallResponse;
  processTracedEvents: ProcessTracedEvents;
}>;

export async function fetchLLMCompletion(
  params: FetchLLMCompletionParams,
): Promise<{
  completion:
    | string
    | IterableReadableStream<Uint8Array>
    | Record<string, unknown>
    | ToolCallResponse;
  processTracedEvents: ProcessTracedEvents;
}> {
  // the apiKey must never be printed to the console
  const {
    messages,
    tools,
    modelParams,
    streaming,
    callbacks,
    apiKey,
    baseURL,
    maxRetries,
    config,
    traceParams,
    extraHeaders,
    throwOnError = true,
  } = params;

  let finalCallbacks: BaseCallbackHandler[] | undefined = callbacks ?? [];
  let processTracedEvents: ProcessTracedEvents = () => Promise.resolve();

  if (traceParams) {
    const handler = new CallbackHandler({
      _projectId: traceParams.projectId,
      _isLocalEventExportEnabled: true,
      tags: traceParams.tags,
    });
    // handler.debug(true);
    finalCallbacks.push(handler);

    processTracedEvents = async () => {
      try {
        let events = await handler.langfuse._exportLocalEvents(
          traceParams.projectId,
        );
        await processEventBatch(
          JSON.parse(JSON.stringify(events)), // stringify to emulate network event batch from network call
          traceParams.authCheck,
        );
      } catch (e) {
        logger.error("Failed to process traced events", { error: e });
      }
    };
  }

  finalCallbacks = finalCallbacks.length > 0 ? finalCallbacks : undefined;

  let finalMessages: BaseMessage[];
  // VertexAI requires at least 1 user message
  if (modelParams.adapter === LLMAdapter.VertexAI && messages.length === 1) {
    finalMessages = [new HumanMessage(messages[0].content)];
  } else {
    finalMessages = messages.map((message) => {
      if (message.role === ChatMessageRole.User)
        return new HumanMessage(message.content);
      if (
        message.role === ChatMessageRole.System ||
        message.role === ChatMessageRole.Developer
      )
        return new SystemMessage(message.content);

      if (message.type === ChatMessageType.ToolResult)
        return new ToolMessage({
          content: message.content,
          tool_call_id: message.toolCallId,
        });

      return new AIMessage({
        content: message.content,
        tool_calls:
          message.type === ChatMessageType.AssistantToolCall
            ? (message.toolCalls as any)
            : undefined,
      });
    });
  }

  finalMessages = finalMessages.filter(
    (m) => m.content.length > 0 || "tool_calls" in m,
  );

  let chatModel:
    | ChatOpenAI
    | ChatAnthropic
    | ChatBedrockConverse
    | ChatVertexAI
    | ChatGoogleGenerativeAI;
  if (modelParams.adapter === LLMAdapter.Anthropic) {
    chatModel = new ChatAnthropic({
      anthropicApiKey: apiKey,
      anthropicApiUrl: baseURL,
      modelName: modelParams.model,
      temperature: modelParams.temperature,
      maxTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      callbacks: finalCallbacks,
      clientOptions: { maxRetries, timeout: 1000 * 60 * 2 }, // 2 minutes timeout
    });
  } else if (modelParams.adapter === LLMAdapter.OpenAI) {
    chatModel = new ChatOpenAI({
      openAIApiKey: apiKey,
      modelName: modelParams.model,
      temperature: modelParams.temperature,
      maxTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      streamUsage: false, // https://github.com/langchain-ai/langchainjs/issues/6533
      callbacks: finalCallbacks,
      maxRetries,
      configuration: {
        baseURL,
        defaultHeaders: extraHeaders,
      },
      timeout: 1000 * 60 * 2, // 2 minutes timeout
    });
  } else if (modelParams.adapter === LLMAdapter.Azure) {
    chatModel = new ChatOpenAI({
      azureOpenAIApiKey: apiKey,
      azureOpenAIBasePath: baseURL,
      azureOpenAIApiDeploymentName: modelParams.model,
      azureOpenAIApiVersion: "2025-02-01-preview",
      temperature: modelParams.temperature,
      maxTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      callbacks: finalCallbacks,
      maxRetries,
      timeout: 1000 * 60 * 2, // 2 minutes timeout
      configuration: {
        defaultHeaders: extraHeaders,
      },
    });
  } else if (modelParams.adapter === LLMAdapter.Bedrock) {
    const { region } = BedrockConfigSchema.parse(config);
    const credentials = BedrockCredentialSchema.parse(JSON.parse(apiKey));

    chatModel = new ChatBedrockConverse({
      model: modelParams.model,
      region,
      credentials,
      temperature: modelParams.temperature,
      maxTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      callbacks: finalCallbacks,
      maxRetries,
      timeout: 1000 * 60 * 2, // 2 minutes timeout
    });
  } else if (modelParams.adapter === LLMAdapter.VertexAI) {
    const credentials = GCPServiceAccountKeySchema.parse(JSON.parse(apiKey));

    // Requests time out after 60 seconds for both public and private endpoints by default
    // Reference: https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions#send-request
    chatModel = new ChatVertexAI({
      modelName: modelParams.model,
      temperature: modelParams.temperature,
      maxOutputTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      callbacks: finalCallbacks,
      maxRetries,
      authOptions: {
        projectId: credentials.project_id,
        credentials,
      },
    });
  } else if (modelParams.adapter === LLMAdapter.GoogleAIStudio) {
    chatModel = new ChatGoogleGenerativeAI({
      model: modelParams.model,
      temperature: modelParams.temperature,
      maxOutputTokens: modelParams.max_tokens,
      topP: modelParams.top_p,
      callbacks: finalCallbacks,
      maxRetries,
      apiKey,
    });
  } else if (modelParams.adapter === LLMAdapter.Atla) {
    // Atla models do not support:
    // - temperature
    // - max_tokens
    // - top_p
    chatModel = new ChatOpenAI({
      openAIApiKey: apiKey,
      modelName: modelParams.model,
      callbacks: finalCallbacks,
      maxRetries,
      configuration: {
        baseURL: baseURL,
        defaultHeaders: extraHeaders,
      },
      timeout: 1000 * 60, // 1 minute timeout
    });
  } else {
    // eslint-disable-next-line no-unused-vars
    const _exhaustiveCheck: never = modelParams.adapter;
    throw new Error(
      `This model provider is not supported: ${_exhaustiveCheck}`,
    );
  }

  const runConfig = {
    callbacks: finalCallbacks,
    runId: traceParams?.traceId,
    runName: traceParams?.traceName,
  };

  try {
    if (params.structuredOutputSchema) {
      return {
        completion: await (chatModel as ChatOpenAI) // Typecast necessary due to https://github.com/langchain-ai/langchainjs/issues/6795
          .withStructuredOutput(params.structuredOutputSchema)
          .invoke(finalMessages, runConfig),
        processTracedEvents,
      };
    }

    /*
  Workaround OpenAI reasoning models:
  
  This is a temporary workaround to avoid sending unsupported parameters to OpenAI's O1 models.
  O1 models do not support:
  - system messages
  - top_p
  - max_tokens at all, one has to use max_completion_tokens instead
  - temperature different than 1

  Reference: https://platform.openai.com/docs/guides/reasoning/beta-limitations
  */
    if (
      modelParams.model.startsWith("o1-") ||
      modelParams.model.startsWith("o3-")
    ) {
      const filteredMessages = finalMessages.filter((message) => {
        return (
          modelParams.model.startsWith("o3-") || message._getType() !== "system"
        );
      });

      return {
        completion: await new ChatOpenAI({
          openAIApiKey: apiKey,
          modelName: modelParams.model,
          temperature: 1,
          maxTokens: undefined,
          topP: undefined,
          callbacks,
          maxRetries,
          modelKwargs: {
            max_completion_tokens: modelParams.max_tokens,
          },
          configuration: {
            baseURL,
          },
          timeout: 1000 * 60 * 2, // 2 minutes timeout
        })
          .pipe(new StringOutputParser())
          .invoke(filteredMessages, runConfig),
        processTracedEvents,
      };
    }

    if (tools && tools.length > 0) {
      const batchSize = 2;
      const toolBatches = chunkArray(splitTools(tools), batchSize);

      let allToolCalls: LLMToolCall[] = [];
      let allContents: any[] = [];

      for (const batch of toolBatches) {
        const langchainTools = batch.map((tool) => ({
          type: "function",
          function: tool,
        }));

        const result = await chatModel
          .bindTools(langchainTools)
          .invoke(finalMessages, runConfig);

        const parsed = ToolCallResponseSchema.safeParse(result);
        if (!parsed.success)
          throw Error("Failed to parse LLM tool call result");

        allToolCalls.push(...(parsed.data.tool_calls || []));
        allContents.push(parsed.data.content);
      }

      return {
        completion: {
          content: allContents,
          tool_calls: mergeToolCalls(allToolCalls),
        },
        processTracedEvents,
      };
    }

    if (streaming) {
      return {
        completion: await chatModel
          .pipe(new BytesOutputParser())
          .stream(finalMessages, runConfig),
        processTracedEvents,
      };
    }

    return {
      completion: await chatModel
        .pipe(new StringOutputParser())
        .invoke(finalMessages, runConfig),
      processTracedEvents,
    };
  } catch (error) {
    if (throwOnError) {
      throw error;
    }

    return { completion: "", processTracedEvents };
  }
}

function splitTools(tools: LLMToolDefinition[]): LLMToolDefinition[] {
  const result: LLMToolDefinition[] = [];
  for (const tool of tools) {
    if (
      tool.chunks_config &&
      "properties" in tool.parameters &&
      typeof tool.parameters.properties === "object"
    ) {
      const propertyNames = Object.keys(
        tool.parameters.properties as Record<string, unknown>,
      );
      const toolName = tool.name;

      let propertyNameGroups: string[][] = [];

      if (
        LLMToolChunkByPropertyCountConfigSchema.safeParse(tool.chunks_config)
          .success
      ) {
        const propertyCountChunkConfig =
          tool.chunks_config as LLMToolChunkByPropertyCountConfig;
        propertyNameGroups = chunkArray(
          propertyNames,
          propertyCountChunkConfig.propertyCount,
        );
      } else if (
        LLMToolChunkByPropertyGroupConfigSchema.safeParse(tool.chunks_config)
          .success
      ) {
        const propertyGroupChunkConfig =
          tool.chunks_config as LLMToolChunkByPropertyGroupConfig;
        let remainingPropertyNames = propertyNames;
        let chunksGroups = [];

        for (const group of propertyGroupChunkConfig.propertyGroup) {
          const validProperties = group.filter((property) =>
            remainingPropertyNames.includes(property),
          );

          if (validProperties.length > 0) {
            chunksGroups.push(validProperties);
            remainingPropertyNames = remainingPropertyNames.filter(
              (property) => !validProperties.includes(property),
            );
          }
        }

        if (remainingPropertyNames.length > 0) {
          chunksGroups.push(remainingPropertyNames);
        }

        propertyNameGroups = chunksGroups;
      }

      if (propertyNameGroups.length > 0) {
        let i = 0;
        for (const group of propertyNameGroups) {
          result.push({
            ...tool,
            name: `${LLMToolCallChunkPrefix}${i}_${toolName}`,
            parameters: {
              ...tool.parameters,
              properties: Object.fromEntries(
                group.map((property) => [
                  property,
                  (tool.parameters.properties as Record<string, unknown>)[
                    property
                  ],
                ]),
              ),
            },
          });
          i++;
        }
      }
    } else {
      result.push(tool);
    }
  }

  return result;
}

function mergeToolCalls(toolCalls: LLMToolCall[]): LLMToolCall[] {
  const result: LLMToolCall[] = toolCalls.filter(
    (toolCall) => !toolCall.name.startsWith(LLMToolCallChunkPrefix),
  );

  const splitToolCalls = toolCalls.filter((toolCall) =>
    toolCall.name.startsWith(LLMToolCallChunkPrefix),
  );

  const mergedToolCalls: Record<string, LLMToolCall> = {};
  for (const toolCall of splitToolCalls) {
    const originalToolName = toolCall.name.split("_").slice(1).join("_");
    if (!mergedToolCalls[originalToolName]) {
      mergedToolCalls[originalToolName] = {
        name: originalToolName,
        id: toolCall.id,
        args: {
          ...toolCall.args,
        },
      };
    } else {
      mergedToolCalls[originalToolName].args = {
        ...mergedToolCalls[originalToolName].args,
        ...toolCall.args,
      };
    }
  }

  return result.concat(Object.values(mergedToolCalls));
}

function chunkArray<T>(arr: T[], size: number): T[][] {
  const res: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    res.push(arr.slice(i, i + size));
  }
  return res;
}
