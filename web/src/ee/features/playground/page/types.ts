import {
  type PromptVariable,
  type ChatMessage,
  type UIModelParams,
  type LLMFunctionCall,
} from "@langfuse/shared";

export type PlaygroundCache = {
  messages: ChatMessage[];
  modelParams?: Partial<UIModelParams> &
    Pick<UIModelParams, "provider" | "model">;
  output?: string | null;
  promptVariables?: PromptVariable[];
  functionCall?: LLMFunctionCall[];
} | null;
